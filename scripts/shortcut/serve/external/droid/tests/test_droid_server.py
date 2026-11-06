"""CPU checks of the π0.5-DROID sc-policy/1 server (T017): openpi normalization / state tokens / image resize /
action conversion, the joint_velocity action space on the wire (hello, client checks, joint_pos), seed
determinism, registry row, and the sc-bench mock flow (ours harness → serve.sh → server.py, joint env).

    uv run --extra shortcut-bench pytest scripts/shortcut/serve/external/droid/tests -q

The real checkpoint on CPU: ../cpu_check.py (needs LIBERO + ~17 GB RAM).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack")
torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
SHORTCUT = PKG.parents[2]
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "serve"), str(SHORTCUT / "parallel")]


def _load(name: str, path: Path):
    """Own module names: serve/external/starvla has a server.py too (same pytest process)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ct = _load("droid_contract", PKG / "contract.py")
sv = _load("droid_server", PKG / "server.py")

import joint_control  # noqa: E402
from bench.core import registry  # noqa: E402
from bench.core.client import PolicyClient, upright  # noqa: E402
from bench.core.protocol import (  # noqa: E402
    ACTION_SPACE_JOINT_VELOCITY,
    ACTION_SPACE_OSC,
    ProtocolError,
    validate_obs,
)

TASK = "put the bowl on the stove"


def frame(seed: int, hw: tuple[int, int] = (256, 256)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0 : hw[0], 0 : hw[1]]
    img = np.stack([(x * 255) // hw[1], (y * 255) // hw[0], (x + y) % 256], -1).astype(np.int64)
    return np.clip(img + rng.integers(0, 40, img.shape), 0, 255).astype(np.uint8)


def wire(seed: int, hw: int = 256, joints: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    o = {
        "image": upright(frame(seed, (hw, hw))),
        "image2": upright(frame(seed + 1000, (hw, hw))),
        "state": np.r_[rng.normal(size=6), 0.03, -0.03].astype(np.float32),
    }
    if joints:
        o["joint_pos"] = np.array([0.0, 0.2, 0.0, -2.0, 0.0, 2.3, 0.8], np.float32) + 0.1 * rng.normal(
            size=7
        ).astype(np.float32)
    return o


# ---------------------------------------------------------------- stats / normalization


def test_stats_file_is_the_pinned_openpi_asset():
    assert ct.stats_sha256() == ct.NORM_STATS_SHA256
    raw = ct.load_openpi_stats()
    assert raw["state"]["q01"].shape == (32,) and raw["actions"]["q99"].shape == (32,)
    assert np.all(raw["state"]["q01"][8:] == 0) and np.all(raw["actions"]["q99"][8:] == 0)  # padding
    st = ct.lerobot_stats(raw)
    assert set(st) == {"observation.state", "action"}
    assert all(v.shape == (8,) for s in st.values() for v in s.values())
    assert raw["actions"]["q99"][7] == pytest.approx(0.9998, abs=1e-3)  # gripper position in [0, 1]


def test_stats_sha_is_checked(tmp_path):
    p = tmp_path / "norm_stats.json"
    p.write_text(ct.NORM_STATS.read_text().replace("0.", "0.0", 1))
    with pytest.raises(ValueError, match="sha256"):
        ct.load_openpi_stats(p)


def _lerobot_norm_steps():
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.lerobot_types import TransitionKey
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from lerobot.processor.converters import create_transition

    stats = ct.lerobot_stats(ct.load_openpi_stats())
    feats = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }
    nm = {FeatureType.STATE: NormalizationMode.QUANTILES, FeatureType.ACTION: NormalizationMode.QUANTILES}
    norm = NormalizerProcessorStep(features=feats, norm_map=nm, stats=stats)
    unnorm = UnnormalizerProcessorStep(features={"action": feats["action"]}, norm_map=nm, stats=stats)

    def normalize_state(x: np.ndarray) -> np.ndarray:
        tr = create_transition(observation={"observation.state": torch.from_numpy(x)})
        return norm(tr)[TransitionKey.OBSERVATION]["observation.state"].numpy()

    def unnormalize_action(y: np.ndarray) -> np.ndarray:
        return unnorm(create_transition(action=torch.from_numpy(y)))[TransitionKey.ACTION].numpy()

    return normalize_state, unnormalize_action


def test_lerobot_quantile_normalization_equals_openpi():
    pytest.importorskip("lerobot")
    normalize_state, unnormalize_action = _lerobot_norm_steps()
    raw = ct.load_openpi_stats()
    rng = np.random.default_rng(0)
    q01, q99 = raw["state"]["q01"][:8], raw["state"]["q99"][:8]
    states = (q01 + (q99 - q01) * rng.uniform(-0.1, 1.1, (4000, 8))).astype(np.float32)
    ours = normalize_state(states)
    ref = ct.openpi_normalize(states.astype(np.float64), raw["state"])  # 8-D x → stats sliced to 8, as openpi
    np.testing.assert_allclose(ours, ref, atol=2e-6)
    # the prompt tokens: identical bins (float32 vs float64 may only differ exactly at a bin edge)
    b_ours, b_ref = ct.discretize_state(ours), ct.discretize_state(ref)
    diff = b_ours != b_ref
    assert diff.mean() < 1e-3
    edges = np.linspace(-1, 1, 257)[:-1]
    assert np.all(np.min(np.abs(ref[diff][:, None] - edges[None]), axis=1) < 1e-5)
    # actions: unnormalize the policy's 8 dims; openpi unnormalizes 32 (padding dims pass through) then [:8]
    y = rng.uniform(-1.2, 1.2, (500, 8)).astype(np.float32)
    ref_a = ct.openpi_unnormalize(np.concatenate([y, np.zeros((500, 24), np.float32)], -1), raw["actions"])[
        :, :8
    ]
    np.testing.assert_allclose(unnormalize_action(y), ref_a, atol=2e-6)


def test_discretize_matches_openpi_tokenizer_formula():
    x = np.array([-1.5, -1.0, -0.99, 0.0, 0.999, 1.0, 2.0])
    np.testing.assert_array_equal(ct.discretize_state(x), [-1, 0, 1, 128, 255, 255, 255])


# ---------------------------------------------------------------- observation


def test_resize_with_pad_like_openpi_client():
    sq = frame(0)
    np.testing.assert_array_equal(
        ct.resize_with_pad(sq, 224, 224), np.asarray(Image.fromarray(sq).resize((224, 224), Image.BILINEAR))
    )
    assert ct.resize_with_pad(frame(0, (224, 224)), 224, 224) is not None
    wide = frame(1, (180, 320))  # a DROID-like 16:9 frame: scaled to 224 × 126, black bands top / bottom
    out = ct.resize_with_pad(wide, 224, 224)
    assert out.shape == (224, 224, 3)
    new_h = int(180 / (320 / 224))
    pad = int((224 - new_h) / 2)
    assert np.all(out[:pad] == 0) and np.all(out[pad + new_h :] == 0)
    np.testing.assert_array_equal(
        out[pad : pad + new_h], np.asarray(Image.fromarray(wide).resize((224, new_h), Image.BILINEAR))
    )


def test_gripper_position_and_state():
    assert ct.gripper_position(np.array([0.04, -0.04])) == pytest.approx(0.0)
    assert ct.gripper_position(np.array([0.0, 0.0])) == pytest.approx(1.0)
    assert ct.gripper_position(np.array([0.039, -0.039])) == pytest.approx(0.025, abs=1e-6)  # LIBERO open
    assert ct.gripper_position(np.array([0.05, -0.05])) == 0.0  # clipped
    assert ct.PANDA_MAX_GRIPPER_WIDTH == joint_control.PANDA_MAX_GRIPPER_WIDTH
    o = wire(3)
    s = ct.droid_state(o)
    assert s.dtype == np.float32 and s.shape == (8,)
    np.testing.assert_array_equal(s[:7], o["joint_pos"])
    assert s[7] == pytest.approx(1 - 0.06 / 0.08)


def test_droid_images_map_cameras():
    o = wire(5)
    imgs = ct.droid_images(o)
    assert list(imgs) == ["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"]
    np.testing.assert_array_equal(
        imgs["observation.images.base_0_rgb"], ct.resize_with_pad(o["image"], 224, 224)
    )
    np.testing.assert_array_equal(
        imgs["observation.images.left_wrist_0_rgb"], ct.resize_with_pad(o["image2"], 224, 224)
    )


def test_to_env_actions_like_openpi_droid_client():
    rng = np.random.default_rng(1)
    raw = rng.normal(scale=0.8, size=(15, 8)).astype(np.float32)
    raw[:, 7] = rng.uniform(0, 1, 15)
    out = ct.to_env_actions(raw)
    # examples/droid/main.py: gripper → 1 if > 0.5 else 0, then np.clip(action, -1, 1)
    ref = np.concatenate([raw[:, :7], (raw[:, 7:] > 0.5).astype(np.float32)], -1).clip(-1, 1)
    np.testing.assert_array_equal(out[:, :7], ref[:, :7])
    np.testing.assert_array_equal(out[:, 7], np.where(ref[:, 7] == 1, 1.0, -1.0))  # LIBERO: +1 close, -1 open
    with pytest.raises(ValueError):
        ct.to_env_actions(raw[:, :7])


# ---------------------------------------------------------------- protocol


def test_validate_obs_joint_pos():
    validate_obs(wire(0, joints=False))  # optional for everyone else
    with pytest.raises(ProtocolError, match="joint_pos"):
        validate_obs(wire(0, joints=False), require_joint_pos=True)
    bad = wire(0)
    bad["joint_pos"] = bad["joint_pos"].astype(np.float64)
    with pytest.raises(ProtocolError, match="joint_pos"):
        validate_obs(bad)


def mock_engine(n: int | None = None) -> object:
    args = sv.build_parser().parse_args(
        ["--backend", "mock", "--ckpt-id", "pi05droid"] + (["--n-action-steps", str(n)] if n else [])
    )
    return sv.build_engine(args)


def test_engine_hello_and_chunks():
    e = mock_engine()
    h = e.hello
    assert (
        h["action_space"] == ACTION_SPACE_JOINT_VELOCITY and h["action_dim"] == 8 and h["n_action_steps"] == 8
    )
    assert h["chunk_size"] == 15 and h["exec"]["exec_mode"] == "single"
    ident = h["identity"]
    assert ident["policy_type"] == "pi05" and ident["stats_sha256"] == ct.NORM_STATS_SHA256
    assert {k: ident[k] for k in joint_control.policy_contract()} == joint_control.policy_contract()
    e.reset("a", 7, TASK)
    acts = e.infer_one("a", wire(0, 32), TASK, False)
    assert acts.shape == (8, 8) and acts.dtype == np.float32
    assert np.all(np.abs(acts) <= 1) and np.all(np.isin(acts[:, 7], (-1, 1)))
    e.reset("a", 7, TASK)
    np.testing.assert_array_equal(e.infer_one("a", wire(0, 32), TASK, False), acts)
    e.reset("b", 8, TASK)
    assert not np.array_equal(e.infer_one("b", wire(0, 32), TASK, False), acts)
    seen = e.model.seen[-1]
    assert seen["images"]["observation.images.base_0_rgb"].shape == (224, 224, 3)
    with pytest.raises(ProtocolError, match="joint_pos"):
        e.infer_one("a", wire(0, 32, joints=False), TASK, False)
    with pytest.raises(SystemExit):
        mock_engine(16)


def test_engine_noise_per_episode_is_independent_of_interleaving():
    e = mock_engine()
    e.reset("a", 1, TASK)
    a1 = e.infer_one("a", wire(0, 32), TASK, False)
    a2 = e.infer_one("a", wire(1, 32), TASK, False)
    e.reset("a", 1, TASK)
    e.reset("b", 2, TASK)
    assert np.array_equal(e.infer_one("a", wire(0, 32), TASK, False), a1)
    e.infer_one("b", wire(0, 32), TASK, False)
    assert np.array_equal(e.infer_one("a", wire(1, 32), TASK, False), a2)


@pytest.fixture
def server_url(tmp_path):
    procs = []

    def start(*extra: str) -> str:
        ready = tmp_path / f"ready{len(procs)}.json"
        cmd = [sys.executable, str(PKG / "server.py"), "--backend", "mock", "--port", "0"]
        cmd += ["--ready-file", str(ready), *extra]
        procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}))
        for _ in range(600):
            if ready.exists():
                return json.loads(ready.read_text())["url"]
            assert procs[-1].poll() is None, "server died"
            time.sleep(0.1)
        raise TimeoutError

    yield start
    for p in procs:
        p.terminate()
        p.wait(10)


def test_server_process_action_space_handshake(server_url):
    url = server_url()
    with pytest.raises(ProtocolError, match="action_space 'joint_velocity'"):
        PolicyClient(url)  # default client: LIBERO OSC only → refuses
    pc = PolicyClient(url, action_spaces=(ACTION_SPACE_OSC, ACTION_SPACE_JOINT_VELOCITY))
    assert pc.action_space == ACTION_SPACE_JOINT_VELOCITY and pc.action_dim == 8
    pc.reset("e", 3, TASK)
    with pytest.raises(ProtocolError, match="joint_pos"):
        pc.infer("e", wire(0, 32, joints=False), TASK)  # checked client-side before sending
    chunk = pc.infer("e", wire(0, 32), TASK)
    assert chunk.shape == (8, 8)
    pc.reset("e", 3, TASK)
    np.testing.assert_array_equal(pc.infer("e", wire(0, 32), TASK), chunk)
    one = pc.infer("e", wire(0, 32), TASK, step=True)
    assert one.shape == (1, 8)
    pc.close()


def test_server_refuses_batch_and_unpinned_ckpt(tmp_path):
    r = subprocess.run(
        [sys.executable, str(PKG / "server.py"), "--backend", "mock", "--batch", "2"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "exec_mode single" in r.stderr
    with pytest.raises(SystemExit, match="pinned"):
        sv.resolve_ckpt(sv.build_parser().parse_args(["--ckpt", "someone/other"]))
    with pytest.raises(SystemExit, match="pinned"):
        sv.resolve_ckpt(sv.build_parser().parse_args(["--ckpt", str(tmp_path)]))
    d, rev = sv.resolve_ckpt(sv.build_parser().parse_args(["--ckpt", str(tmp_path), "--allow-unpinned"]))
    assert d == str(tmp_path) and rev.startswith("local:")


def test_revision_must_be_the_pinned_sha_and_records_the_resolved_commit(tmp_path, monkeypatch):
    """--revision main (movable) or another sha only with --allow-unpinned; the identity records the commit sha
    of the snapshot dir the hub resolved, never the requested name."""
    import huggingface_hub

    head = "f" * 40  # what `main` resolves to in this fake hub
    asked: list[str] = []

    def fake_snapshot_download(repo: str, revision: str | None = None, **kw: object) -> str:
        asked.append(str(revision))
        sha = ct.CKPT_SHA if revision == ct.CKPT_SHA else head if revision == "main" else str(revision)
        d = tmp_path / "models--lerobot--pi05_droid" / "snapshots" / sha
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    def resolve(*argv: str) -> tuple[str, str]:
        return sv.resolve_ckpt(sv.build_parser().parse_args(list(argv)))

    _d, rev = resolve()  # default: the pinned sha
    assert rev == ct.CKPT_SHA and asked == [ct.CKPT_SHA]
    for bad in ("main", "v1.0", "a" * 40):
        with pytest.raises(SystemExit, match="allow-unpinned"):
            resolve("--revision", bad)
    assert asked == [ct.CKPT_SHA]  # refused before downloading
    d, rev = resolve("--revision", "main", "--allow-unpinned")
    assert rev == head and rev != "main" and Path(d).name == head
    _d, rev = resolve("--revision", ct.CKPT_SHA)
    assert rev == ct.CKPT_SHA
    with pytest.raises(SystemExit, match="commit sha"):
        sv.snapshot_commit(str(tmp_path / "not-a-snapshot"))


# ---------------------------------------------------------------- registry / sc-bench wiring


def test_registry_builtin_pi05droid_row(tmp_path):
    path = tmp_path / "ckpt_registry.tsv"
    row = registry.resolve("pi05droid", path)
    assert row["server"] == "droid" and row["adapter"] is None and row["path_or_hub"] == ct.CKPT_REPO
    notes = registry.parse_notes(row["notes"])
    assert notes["ckpt_sha"] == ct.CKPT_SHA and notes["stats_sha256"] == ct.NORM_STATS_SHA256
    assert (
        notes["n_action_steps"] == "8"
        and notes["zero_shot"] == "1"
        and notes["action_space"] == "joint_velocity"
    )
    assert registry.external_server_launcher("droid") == PKG / "serve.sh"


def test_sc_bench_mock_flow_joint_velocity(tmp_path):
    """sc-bench --server droid (mock): the ours harness builds the joint_velocity env (MockJointEnv), sends
    joint_pos, executes 8-D actions; identities and bench_version carry the control contract, so the run does not
    pair / mix silently with OSC runs. A battery pinning another action space fails before any episode."""
    pytest.importorskip("lerobot")
    from bench.core import report, runner

    fixture = SHORTCUT / "tests" / "fixtures" / "libero_goal_2tasks.json"
    bat = tmp_path / "b.yaml"
    bat.write_text(
        "name: dr\njobs:\n  - {benchmark: ours, suite: libero_goal, conditions: [C0, C4], tasks: '1', "
        f"episodes: 2, video: 'first:1', instr_set: '{fixture}', action_space: joint_velocity}}\n"
    )
    root = tmp_path / "out"
    argv = ["--ckpt", "mock/droid", "--backend", "mock", "--server", "droid", "--battery", str(bat)]
    argv += ["--out-root", str(root)]
    assert runner.main(argv) == 0
    run = root / "runs" / "bench-droid-dr"
    meta = json.loads((run / "jobs" / "ours-goal-C0" / "run_meta.json").read_text())
    ident = meta["identity"]
    assert (
        ident["policy"] == "pi05"
        and ident["action_space"] == "joint_velocity"
        and ident["control_freq"] == 15
    )
    assert ident["max_joint_delta"] == 0.2 and ident["joint_kp"] == joint_control.JOINT_KP
    assert ident["stats_sha256"] == ct.NORM_STATS_SHA256 and ident["n_action_steps"] == 8
    assert "camera" in ident and "ckpt_sha" in ident
    eps = [json.loads(x) for x in (run / "jobs" / "ours-goal-C0" / "episodes.jsonl").read_text().splitlines()]
    assert len(eps) == 2 and all(e["bench_version"].endswith("/joint_velocity") for e in eps)
    steps = np.load(run / "jobs" / "ours-goal-C0" / eps[0]["steps_file"])
    assert steps["action"].shape[1] == 8
    np.testing.assert_array_equal(steps["gripper_cmd"], steps["action"][:, 7])

    # an OSC run of the same cells → sc-report marks the rows mixed (bench_version)
    bat2 = tmp_path / "b2.yaml"
    bat2.write_text(
        bat.read_text().replace(", action_space: joint_velocity", "").replace("name: dr", "name: os")
    )
    assert (
        runner.main(
            ["--ckpt", "mock/osc", "--backend", "mock", "--battery", str(bat2), "--out-root", str(root)]
        )
        == 0
    )
    eps2, _dups, _w = report.load([run, root / "runs" / "bench-osc-os"])
    rows = report.table(eps2)
    c0 = [r for r in rows if r["detail"] == "C0" and r["suite"] == "libero_goal"]
    assert len(c0) == 2 and all("bench_version" in r["mixed"] for r in c0)
    assert any("bench_version" in w for w in report.warnings(rows))

    # a battery pinning joint_velocity against an OSC server: refused
    bad = tmp_path / "b3.yaml"
    bad.write_text(bat.read_text().replace("name: dr", "name: bad"))
    with pytest.raises(SystemExit, match="pins action_space"):
        runner.main(
            ["--ckpt", "mock/osc2", "--backend", "mock", "--battery", str(bad), "--out-root", str(root)]
        )
