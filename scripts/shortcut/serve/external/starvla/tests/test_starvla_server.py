"""CPU checks of the StarVLA sc-policy/1 server (T016): input / output contract vs StarVLA's LIBERO eval, seed
determinism, protocol, registry + sc-bench wiring. Mock model (no checkpoint).

    uv run --extra shortcut-bench pytest scripts/shortcut/serve/external/starvla/tests -q          # lerobot venv
    cd $STARVLA_SRC && PYTHONPATH=$STARVLA_SRC $STARVLA_PYTHON -m pytest --noconftest -c /dev/null --rootdir $T $T  # StarVLA
    venv (T = this tests dir; the repo conftest imports lerobot)

In the StarVLA venv the reference test also runs StarVLA's own WebsocketPolicyServer + LIBERO ModelClient against
the same mock model (compare.py, the path the real-checkpoint check uses). In the lerobot venv the sc-bench mock
flow runs (ours harness → external server launcher).
"""

from __future__ import annotations

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
sys.path[:0] = [str(PKG), str(SHORTCUT), str(SHORTCUT / "serve"), str(SHORTCUT / "parallel")]

import server as sv  # noqa: E402
from bench.core import registry  # noqa: E402
from bench.core.client import PolicyClient, upright  # noqa: E402
from bench.core.protocol import ProtocolError  # noqa: E402

TASK = "put the bowl on the plate"


def raw_frame(seed: int, hw: int = 256) -> np.ndarray:
    """Stand-in for a raw (upside-down) LIBERO render: structured, so rotations / resizes are visible."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:hw, 0:hw]
    img = np.stack([(x * 255) // hw, (y * 255) // hw, (x + y) % 256], -1).astype(np.int64)
    return np.clip(img + rng.integers(0, 40, img.shape), 0, 255).astype(np.uint8)


def wire(seed: int, hw: int = 256) -> dict:
    return {
        "image": upright(raw_frame(seed, hw)),
        "image2": upright(raw_frame(seed + 1000, hw)),
        "state": np.zeros(8, np.float32),
    }


def engine(cameras: str = sv.DEFAULT_CAMERAS, n: int | None = None) -> sv.StarVLAEngine:
    return sv.StarVLAEngine(sv.MockWrapper(), "cpu", n, cameras, {"ckpt": "mock", "ckpt_id": "mock"})


# ---------------------------------------------------------------- input contract


def test_resize_is_starvla_client_resize():
    """model2libero_interface.ModelClient.step: PIL bilinear (W, H) resize when the shape is not 224×224."""
    img = raw_frame(0)
    want = np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
    np.testing.assert_array_equal(sv.resize_like_starvla(img), want)
    small = raw_frame(1, 224)
    np.testing.assert_array_equal(sv.resize_like_starvla(small), small)  # already 224: untouched
    assert not np.array_equal(want, np.asarray(Image.fromarray(img).resize((224, 224), Image.BICUBIC)))


def test_wire_to_example_equals_starvla_eval_input():
    """Raw render → our wire (upright) → example == eval_libero.py: raw[::-1, ::-1] for both views, order
    [agentview, wrist], then the client's resize; lang verbatim; no state key."""
    ra, rw = raw_frame(3), raw_frame(4)
    obs = {"image": upright(ra), "image2": upright(rw), "state": np.ones(8, np.float32)}
    ex = sv.wire_to_example(obs, TASK)
    starvla = [np.ascontiguousarray(ra[::-1, ::-1]), np.ascontiguousarray(rw[::-1, ::-1])]
    want = [np.asarray(Image.fromarray(i).resize((224, 224), Image.BILINEAR)) for i in starvla]
    assert set(ex) == {"image", "lang"} and ex["lang"] == TASK
    assert len(ex["image"]) == 2
    for got, w in zip(ex["image"], want, strict=True):
        assert got.dtype == np.uint8 and got.shape == (224, 224, 3)
        np.testing.assert_array_equal(got, w)
    only = sv.wire_to_example(obs, TASK, "agentview")
    assert len(only["image"]) == 1
    np.testing.assert_array_equal(only["image"][0], want[0])
    with pytest.raises(ProtocolError):
        sv.wire_to_example({"image": obs["image"], "state": obs["state"]}, TASK)


# ---------------------------------------------------------------- output contract


def starvla_binarize(open_val):  # eval_libero._binarize_gripper_open, verbatim
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def test_to_env_actions_equals_eval_libero():
    rng = np.random.default_rng(0)
    chunk = rng.normal(size=(8, 7)).astype(np.float32)
    chunk[:, 6] = [0.0, 1.0, 0.5, 0.5000001, 0.49, 0.9, -0.2, 1.3]
    got = sv.to_env_actions(chunk)
    for t in range(8):
        want = np.concatenate([chunk[t, :3], chunk[t, 3:6], starvla_binarize(chunk[t, 6:7])])
        np.testing.assert_array_equal(got[t], want)
    assert got.dtype == np.float32
    assert list(got[:4, 6]) == [1.0, -1.0, 1.0, -1.0]  # 1 = open → -1 (LIBERO open), 0.5 → close
    with pytest.raises(ValueError):
        sv.to_env_actions(np.zeros((8, 8)))


# ---------------------------------------------------------------- engine: chunks, seeds, hello


def test_engine_chunks_and_hello():
    e = engine()
    h = e.hello
    assert h["policy_type"] == "starvla" and h["exec_mode"] == "single" and h["requires_step"] is False
    assert h["chunk_size"] == 8 and h["n_action_steps"] == 8 and h["image_hw"] == [256, 256]
    assert h["image_orientation"] == "hf_libero_upright" and h["action_dim"] == 7 and h["state_dim"] == 8
    e.reset("a", 7, TASK)
    out = e.infer_one("a", wire(0), TASK, False)
    assert out.shape == (8, 7) and out.dtype == np.float32 and set(np.unique(out[:, 6])) <= {-1.0, 1.0}
    call = e.w.seen[-1]
    assert call["unnorm_key"] is None and {k: call[k] for k in sv.PREDICT_KWARGS} == sv.PREDICT_KWARGS
    assert engine(n=5).hello["n_action_steps"] == 5
    with pytest.raises(SystemExit):
        engine(n=9)
    with pytest.raises(ProtocolError, match="unknown episode"):
        e.infer_one("nope", wire(0), TASK, False)


def test_engine_noise_per_episode_seed():
    """Same seed → same chunks; another seed → other chunks; interleaving episodes changes nothing;
    step mode serves the same chunk one action at a time."""

    def run(e, eid, seed, n=3):
        e.reset(eid, seed, TASK)
        return [e.infer_one(eid, wire(t), TASK, False) for t in range(n)]

    a = run(engine(), "x", 11)
    b = run(engine(), "x", 11)
    for u, v in zip(a, b, strict=True):
        np.testing.assert_array_equal(u, v)
    c = run(engine(), "x", 12)
    assert not np.array_equal(a[0], c[0])
    e = engine()
    e.reset("x", 11, TASK)
    e.reset("y", 22, TASK)
    for t in range(3):
        e.infer_one("y", wire(t + 5), TASK, False)
        torch.randn(5)  # foreign RNG use between calls must not matter either
        np.testing.assert_array_equal(e.infer_one("x", wire(t), TASK, False), a[t])
    e.reset("s", 11, TASK)
    steps = [e.infer_one("s", wire(0), TASK, True) if not e.episode("s").queue else None for _ in range(1)]
    seq = [steps[0]] + [e.serve_from_queue("s") for _ in range(7)]
    np.testing.assert_array_equal(np.concatenate(seq), a[0])


# ---------------------------------------------------------------- server process (protocol)


@pytest.fixture
def server_url(tmp_path):
    procs = []

    def start(*extra: str) -> str:
        ready = tmp_path / f"ready{len(procs)}.json"
        cmd = [sys.executable, str(PKG / "server.py"), "--backend", "mock", "--port", "0"]
        cmd += ["--ready-file", str(ready), "--ckpt", "mock/starvla", *extra]
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


def test_server_process_protocol(server_url):
    pc = PolicyClient(server_url())
    h = pc.info
    assert h["policy_type"] == "starvla" and h["n_action_steps"] == 8 and h["exec"]["exec_mode"] == "single"
    ident = h["identity"]
    assert ident["policy_type"] == "starvla" and ident["camera"] == "agentview+wrist"
    assert {"starvla_sha", "ckpt_sha", "base_vlm_sha"} <= set(ident)
    pc.reset("e", 3, TASK)
    chunk = pc.infer("e", wire(0, 32), TASK)
    assert chunk.shape == (8, 7)
    pc.reset("e", 3, TASK)
    np.testing.assert_array_equal(pc.infer("e", wire(0, 32), TASK), chunk)
    pc.close()


def test_server_refuses_batch():
    r = subprocess.run(
        [sys.executable, str(PKG / "server.py"), "--backend", "mock", "--batch", "2"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "exec_mode single" in r.stderr


def test_resolve_paths_checks_pins(tmp_path, monkeypatch):
    pt = tmp_path / "ck" / "checkpoints" / "steps_100000_pytorch_model.pt"
    pt.parent.mkdir(parents=True)
    pt.write_bytes(b"x")
    vlm = tmp_path / "vlm"
    vlm.mkdir()
    (vlm / "config.json").write_text("{}")
    args = sv.build_parser().parse_args(
        ["--ckpt", str(pt), "--base-vlm", str(vlm), "--starvla-src", str(tmp_path)]
    )
    with pytest.raises(SystemExit, match="pinned"):
        sv.resolve_paths(args)
    (tmp_path / "ck" / ".sc_revision").write_text(sv.CKPT_SHA)
    (vlm / ".sc_revision").write_text(sv.BASE_VLM_SHA)
    monkeypatch.setattr(sv, "git_head", lambda p: sv.STARVLA_SHA)
    got = sv.resolve_paths(args)
    assert got["pt"] == pt and got["ckpt_sha"] == sv.CKPT_SHA
    monkeypatch.setenv("STARVLA_CKPT", str(pt))
    args2 = sv.build_parser().parse_args(["--base-vlm", str(vlm)])  # default --ckpt = pinned hub id
    assert sv.resolve_paths(args2)["pt"] == pt


# ---------------------------------------------------------------- registry / sc-bench wiring


def test_registry_builtin_starvla_row(tmp_path):
    path = tmp_path / "ckpt_registry.tsv"
    row = registry.resolve("starvla-qwen3pi-libero", path)
    assert row["server"] == "starvla" and row["adapter"] is None and row["policy_type"] == "starvla"
    notes = registry.parse_notes(row["notes"])
    assert notes["ckpt_sha"] == sv.CKPT_SHA and notes["starvla_sha"] == sv.STARVLA_SHA
    assert notes["n_action_steps"] == "8"
    reg = registry.read_registry(path)
    assert {"pi05lib", "pi05base", "starvla-qwen3pi-libero"} <= set(reg)
    registry.resolve("starvla-qwen3pi-libero", path)  # second resolve: no duplicate row
    assert path.read_text().count("starvla-qwen3pi-libero") == 1
    assert registry.external_server_launcher("starvla") == PKG / "serve.sh"
    with pytest.raises(SystemExit):
        registry.external_server_launcher("nope")


def test_starvla_reference_path_equals_ours():
    """StarVLA's own WebsocketPolicyServer + LIBERO ModelClient vs our server, same (mock) model and seed."""
    pytest.importorskip("deployment.model_server.policy_wrapper")
    pytest.importorskip("examples.simBenchmarks.LIBERO.eval_files.model2libero_interface")
    import compare

    src = Path(sys.modules["deployment"].__path__[0]).parent
    frames = {
        "agentview": np.stack([raw_frame(0), raw_frame(1)]),
        "wrist": np.stack([raw_frame(10), raw_frame(11)]),
        "instruction": np.asarray(TASK),
        "seed": np.asarray(7_000_100),
    }
    r = compare.compare(sv.MockWrapper(), src, frames, 2)
    assert r["pass"], r["max_abs_diff"]
    assert r["n_actions"] == 16


def test_sc_bench_mock_flow_external_server(tmp_path):
    """sc-bench --server starvla (mock): the ours harness runs against server.py through serve.sh; the job's
    run_meta identity carries the StarVLA fields."""
    pytest.importorskip("lerobot")
    from bench.core import runner

    fixture = SHORTCUT / "tests" / "fixtures" / "libero_goal_2tasks.json"
    bat = tmp_path / "b.yaml"
    bat.write_text(
        "name: sv\njobs:\n  - {benchmark: ours, suite: libero_goal, conditions: [C0, C4], tasks: '1', "
        f"episodes: 2, video: 'first:1', instr_set: '{fixture}'}}\n"
    )
    root = tmp_path / "out"
    argv = ["--ckpt", "mock/starvla", "--backend", "mock", "--server", "starvla", "--battery", str(bat)]
    argv += ["--out-root", str(root)]
    assert runner.main(argv) == 0
    run = root / "runs" / "bench-starvla-sv"
    bm = json.loads((run / "bench_meta.json").read_text())
    assert bm["identity"]["server"] == "starvla" and bm["identity"]["n_action_steps"] == 8
    meta = json.loads((run / "jobs" / "ours-goal-C0" / "run_meta.json").read_text())
    ident = meta["identity"]
    assert (
        ident["policy"] == "starvla" and ident["camera"] == "agentview+wrist" and ident["n_action_steps"] == 8
    )
    assert {"starvla_sha", "ckpt_sha", "base_vlm_sha", "exec_mode"} <= set(ident)
    eps = [json.loads(x) for x in (run / "jobs" / "ours-goal-C0" / "episodes.jsonl").read_text().splitlines()]
    assert len(eps) == 2
