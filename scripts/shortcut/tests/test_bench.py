"""CPU-only checks for the testbed core (T010): protocol, client checks, registry, battery, policy server
(single / batched), the whole sc-bench flow with resume, and sc-report. Mock env + mock policy.

    uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_bench.py -q
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack")

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parent
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "parallel")]

from bench.core import (  # noqa: E402
    battery,
    records,
    registry,
    report as sc_report,  # noqa: E402
    runner as sc_bench,  # noqa: E402
)
from bench.core.adapters import ADAPTERS  # noqa: E402
from bench.core.client import PolicyClient, libero_state, quat2axisangle  # noqa: E402
from bench.core.protocol import ORIENT_RAW, ProtocolError, packb, unpackb, validate_obs  # noqa: E402

FIXTURE = HERE / "fixtures" / "libero_goal_2tasks.json"
TIMING = {"time_policy_s", "time_env_s", "wall_time_s", "policy_server_ms"}


def obs(seed: int = 0, hw: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "image": rng.integers(0, 255, (hw, hw, 3), dtype=np.uint8),
        "image2": rng.integers(0, 255, (hw, hw, 3), dtype=np.uint8),
        "state": rng.normal(size=8).astype(np.float32),
    }


# ---------------------------------------------------------------- protocol / client helpers


def test_msgpack_numpy_roundtrip():
    msg = {
        "type": "infer",
        "obs": obs(),
        "f64": np.arange(6, dtype=np.float64).reshape(2, 3),
        "bool": np.array([True, False]),
        "scalar": np.float32(1.5),
        "nested": [{"x": np.int64(3)}],
    }
    back = unpackb(packb(msg))
    for k in ("image", "image2", "state"):
        assert back["obs"][k].dtype == msg["obs"][k].dtype
        np.testing.assert_array_equal(back["obs"][k], msg["obs"][k])
    np.testing.assert_array_equal(back["f64"], msg["f64"])
    np.testing.assert_array_equal(back["bool"], msg["bool"])
    assert back["scalar"] == np.float32(1.5) and back["nested"][0]["x"] == 3


def test_validate_obs():
    validate_obs(obs())
    bad = obs()
    bad["state"] = bad["state"].astype(np.float64)
    with pytest.raises(ProtocolError, match="float32"):
        validate_obs(bad)
    bad = obs()
    bad["image"] = bad["image"].astype(np.float32)
    with pytest.raises(ProtocolError, match="uint8"):
        validate_obs(bad)
    with pytest.raises(ProtocolError, match="expects"):
        validate_obs(obs(hw=32), image_hw=(64, 64))


def test_libero_state_close_to_lerobot():
    import torch

    from lerobot.processor.env_processor import LiberoProcessorStep

    rng = np.random.default_rng(1)
    step = LiberoProcessorStep()
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        pos, grip = rng.normal(size=3), rng.normal(size=2) * 0.04
        nested = {
            "eef": {"pos": torch.from_numpy(pos)[None], "quat": torch.from_numpy(q)[None]},
            "gripper": {"qpos": torch.from_numpy(grip)[None]},
        }
        ref = step._process_observation({"observation.robot_state": nested})["observation.state"][0].numpy()
        np.testing.assert_allclose(libero_state(pos, q, grip), ref, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3, np.float32))


# ---------------------------------------------------------------- registry / battery


def test_registry_create_and_resolve(tmp_path):
    path = tmp_path / "ckpt_registry.tsv"
    assert registry.ensure_registry(path)
    assert not registry.ensure_registry(path)
    header = path.read_text().splitlines()[0].split("\t")
    assert tuple(header) == registry.COLUMNS
    lib = registry.resolve("pi05lib", path)
    assert lib["path_or_hub"] == "lerobot/pi05_libero_finetuned" and lib["adapter"] is None
    base = registry.resolve("lerobot/pi05_base", path)  # hub id of a registered row → that row
    assert base["ckpt_id"] == "pi05base" and base["adapter"] == "lerobot/pi05_libero_finetuned"
    d = tmp_path / "train" / "tb-smolvla-a" / "checkpoints" / "last" / "pretrained_model"
    d.mkdir(parents=True)
    loc = registry.resolve(str(d), path)
    assert loc["ckpt_id"] == "tb-smolvla-a" and not loc["registered"]
    with pytest.raises(SystemExit):
        registry.resolve("no-such-ckpt", path)


def test_battery_expand_and_shipped_files():
    b = {"_path": "x", "defaults": {"episodes": 5}, "jobs": [
        {"benchmark": "ours", "suites": ["libero_goal", "libero_spatial"], "conditions": ["C0", "C4"]},
        {"benchmark": "ours", "suite": "libero_10", "conditions": ["C0"], "repeats": [0, 1], "episodes": 2},
    ]}  # fmt: skip
    jobs = battery.expand_jobs(b)
    assert len(jobs) == 6
    assert {(j["suite"], j["condition"]) for j in jobs[:4]} == {
        (s, c) for s in ("libero_goal", "libero_spatial") for c in ("C0", "C4")
    }
    assert jobs[0]["episodes"] == 5 and jobs[-1]["episodes"] == 2 and jobs[-1]["repeat"] == 1
    for name in ("quick", "full", "smoke"):
        jobs = battery.expand_jobs(battery.load_battery(name))
        assert all(j["benchmark"] in ADAPTERS for j in jobs)  # T012 / T013 add their benchmarks' jobs
    quick = battery.expand_jobs(battery.load_battery("quick"))
    assert {j["condition"] for j in quick if j["benchmark"] == "ours"} == {"C0", "C1", "C4", "C5"}
    tm = battery.TimeModel()
    assert tm.steps_per_s(4) > tm.steps_per_s(1)


def test_record_fields():
    rec = records.standard_fields(
        benchmark="ours", bench_version="v", model="m", ckpt="c", suite="libero_goal", task_id=1,
        variant_id="C4-a", init_idx=0, pert=records.perturbation("language", "object_swap", "C4"),
        instruction_original="a", instruction_given="b", success=False, success_orig=True, success_new=False,
        video=None, seed=7,
    )  # fmt: skip
    assert records.missing_fields(rec) == []
    with pytest.raises(ValueError):
        records.perturbation("weather", 1, "x")


def test_append_jsonl_after_torn_line(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"a": 1}\n{"b": ')  # writer killed mid-line
    records.append_jsonl(p, {"c": 3})
    assert records.read_jsonl(p) == [{"a": 1}, {"c": 3}]


# ---------------------------------------------------------------- policy server


@pytest.fixture
def server(tmp_path):
    procs = []

    def start(*extra: str) -> str:
        ready = tmp_path / f"ready{len(procs)}.json"
        cmd = [sys.executable, str(SHORTCUT / "serve" / "policy_server.py"), "--ckpt", "mock/pi05lib"]
        cmd += ["--backend", "mock", "--port", "0", "--ready-file", str(ready), *extra]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        procs.append(subprocess.Popen(cmd, env=env))
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


def test_server_hello_and_checks(server):
    url = server()
    pc = PolicyClient(url)
    h = pc.info
    assert h["protocol_version"] == "sc-policy/1" and h["action_dim"] == 7 and h["state_dim"] == 8
    assert h["exec_mode"] == "single" and h["exec"]["exec_mode"] == "single"
    with pytest.raises(ProtocolError, match="orientation|expects images"):
        PolicyClient(url, orientation=ORIENT_RAW)
    with pytest.raises(ProtocolError, match="unknown episode"):
        pc.infer("nope", obs(), "x")
    pc.reset("e", 1, "put the bowl on the plate")
    chunk = pc.infer("e", obs(), "put the bowl on the plate")
    assert chunk.shape == (10, 7) and chunk.dtype == np.float32
    pc.reset("e", 1, "put the bowl on the plate")
    steps = np.concatenate([pc.infer("e", obs(), "put the bowl on the plate", step=True) for _ in range(10)])
    np.testing.assert_array_equal(steps, chunk)  # server-side queue = the same chunk, one action at a time
    pc.close()


def test_server_single_noise_is_per_episode(server):
    """Interleaving episodes on one single-mode server gives each the chunks it gets alone."""
    url = server()
    a, b = PolicyClient(url), PolicyClient(url)
    alone = []
    a.reset("x", 11, "put the bowl on the stove")
    for t in range(3):
        alone.append(a.infer("x", obs(t), "put the bowl on the stove"))
    a.reset("x", 11, "put the bowl on the stove")
    b.reset("y", 22, "put the bowl on the plate")
    for t in range(3):
        b.infer("y", obs(t + 5), "put the bowl on the plate")
        np.testing.assert_array_equal(a.infer("x", obs(t), "put the bowl on the stove"), alone[t])


def test_server_batched_matches_single(server):
    """batched-N: per-episode generators + padding → each row equals the single-mode chunk (mock is row-wise)."""
    single, batched = server(), server("--batch", "3", "--batch-wait-ms", "50")
    s, b1, b2 = PolicyClient(single), PolicyClient(batched), PolicyClient(batched)
    assert b1.info["exec_mode"] == "batched-3"
    for c in (s, b1, b2):
        c.reset("e", 5, "put the bowl on the plate")
    import threading

    out = {}

    def run(c, key):
        out[key] = [c.infer("e", obs(t), "put the bowl on the plate") for t in range(3)]

    ts = [threading.Thread(target=run, args=(c, k)) for c, k in ((b1, "b1"), (b2, "b2"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ref = [s.infer("e", obs(t), "put the bowl on the plate") for t in range(3)]
    for k in ("b1", "b2"):
        for got, want in zip(out[k], ref, strict=True):
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


# ---------------------------------------------------------------- sc-bench flow


def write_battery(tmp_path: Path, episodes: int = 3) -> Path:
    p = tmp_path / "mock.yaml"
    p.write_text(
        "name: mocktest\njobs:\n"
        "  - {benchmark: ours, suite: libero_goal, conditions: [C0, C4, C5], tasks: '1,8', "
        f"episodes: {episodes}, video: 'first:1', instr_set: '{FIXTURE}'}}\n"
    )
    return p


def bench_argv(tmp_path: Path, bat: Path, root: Path, *extra: str, ckpt: str = "mock/pi05lib") -> list[str]:
    return ["--ckpt", ckpt, "--backend", "mock", "--battery", str(bat), "--out-root", str(root), *extra]


def episodes(root: Path, run_id: str) -> dict[tuple, dict]:
    out = {}
    for f in sorted((root / "runs" / run_id / "jobs").glob("*/episodes.jsonl")):
        for r in records.read_jsonl(f):
            k = (r["condition"], r["task_id"], r["variant"], r["init_idx"])
            assert k not in out, f"duplicate {k}"
            out[k] = {x: v for x, v in r.items() if x not in TIMING}
    return out


def test_sc_bench_flow_resume_and_report(tmp_path):
    bat = write_battery(tmp_path, episodes=6)
    ref_root = tmp_path / "ref"
    assert sc_bench.main(bench_argv(tmp_path, bat, ref_root, "--clients-per-gpu", "2")) == 0
    ref = episodes(ref_root, "bench-pi05lib-mocktest")
    assert {k[0] for k in ref} == {"C0", "C4", "C5"} and len(ref) % 6 == 0
    for r in ref.values():
        assert records.missing_fields(r) == []
    run = ref_root / "runs" / "bench-pi05lib-mocktest"
    assert (run / "STATUS").read_text().strip() == "done"
    meta = json.loads((run / "jobs" / "ours-goal-C4" / "run_meta.json").read_text())
    assert meta["identity"]["exec_mode"] == "batched-2" and meta["identity"]["via"] == "policy_server"
    tsv = (ref_root / "results.tsv").read_text().splitlines()
    assert tsv[0].startswith("run_id\tstatus") and tsv[1].startswith("bench-pi05lib-mocktest\tdone")

    # kill the whole process tree mid-run, rerun the same command → same episodes, no duplicates
    root = tmp_path / "kill"
    argv = [
        sys.executable,
        str(SHORTCUT / "bench" / "core" / "runner.py"),
        *bench_argv(tmp_path, bat, root, "--clients-per-gpu", "2"),
    ]
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True, start_new_session=True)
    assert p.stdout is not None
    for line in p.stdout:
        if "(2/" in line and "units)" in line:
            break
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    partial = episodes(root, "bench-pi05lib-mocktest")
    assert 0 < len(partial) < len(ref)
    assert sc_bench.main(bench_argv(tmp_path, bat, root, "--clients-per-gpu", "2")) == 0
    assert episodes(root, "bench-pi05lib-mocktest") == ref

    # same run dir, another exec mode → refused
    with pytest.raises(SystemExit, match="another configuration"):
        sc_bench.main(bench_argv(tmp_path, bat, root, "--clients-per-gpu", "1"))

    # single-mode run of a second "model" + report
    assert sc_bench.main(bench_argv(tmp_path, bat, ref_root, ckpt="mock/tb-smol")) == 0
    out = tmp_path / "report"
    assert (
        sc_report.main(
            [
                "--runs",
                "bench-pi05lib-mocktest,tb-smol",
                "--out-root",
                str(ref_root),
                "--out",
                str(out),
                "--videos",
                "2",
            ]
        )
        == 0
    )
    rows = (out / "table.csv").read_text().splitlines()
    assert rows[0].startswith("model,benchmark,suite,dimension")
    assert any(",C4," in r and "exec" in r.split(",")[-1] for r in rows[1:])  # batched vs single → mixed
    man = json.loads((out / "manifest.json").read_text())
    assert len(man) == 2
    for m in man:
        assert {
            "id",
            "model",
            "suite",
            "task_id",
            "init",
            "conditions",
            "instructions",
            "outcomes",
            "type",
            "why",
            "gif",
            "gif_bytes",
            "mp4",
            "thumb",
        } <= set(m)
        assert Path(m["gif"]).exists() and m["gif_bytes"] <= 3 * 1024 * 1024
    assert "Warnings" in (out / "report.md").read_text()


def test_dry_run_prints_estimate(tmp_path, capsys):
    bat = write_battery(tmp_path)
    assert sc_bench.main(bench_argv(tmp_path, bat, tmp_path / "o", "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "ours-goal-C4" in out and "4×A100" in out and "GPU 0 (4090), single" in out
    assert not (tmp_path / "o" / "runs").exists()


# ---------------------------------------------------------------- sc-report: pairing, ids, de-duplication


def fake_run(root: Path, name: str, model: str, eps: list[dict], exec_mode: str = "single") -> Path:
    """A bench run dir with one ours job; each episode gets a tiny 2-frame video."""
    import imageio.v2 as imageio

    jd = root / "runs" / name / "jobs" / "ours-goal-C0"
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "run_meta.json").write_text(json.dumps({"identity": {"exec_mode": exec_mode, "gpu": "cpu"}}))
    with open(jd / "episodes.jsonl", "w") as f:
        for e in eps:
            rel = f"videos/{e['task_id']}_{e['init_idx']}_{e.get('repeat_idx', 0)}_{e['seed']}.mp4"
            (jd / rel).parent.mkdir(parents=True, exist_ok=True)
            frames: list[np.ndarray] = [np.full((64, 128, 3), 40 * i, np.uint8) for i in range(2)]
            imageio.mimwrite(jd / rel, frames, fps=10, codec="libx264", macro_block_size=8)
            rec = records.standard_fields(
                benchmark="ours", bench_version="v1", model=model, ckpt=f"ck/{model}", suite="libero_goal",
                task_id=e["task_id"], variant_id="orig", init_idx=e["init_idx"],
                pert=records.perturbation("none", "original", "C0"), instruction_original="a",
                instruction_given="a", success=e.get("success", True), success_orig=e.get("success", True),
                success_new=None, video=rel, seed=e["seed"],
            )  # fmt: skip
            extra = {"condition": "C0", "variant": "orig", "repeat_idx": e.get("repeat_idx", 0)}
            f.write(json.dumps({**rec, **extra}) + "\n")
    return root / "runs" / name


def test_report_video_pairs_need_same_seed(tmp_path):
    fake_run(
        tmp_path,
        "ra",
        "A",
        [{"task_id": 1, "init_idx": 0, "seed": 7}, {"task_id": 1, "init_idx": 1, "seed": 7}],
    )
    fake_run(
        tmp_path,
        "rb",
        "B",
        [{"task_id": 1, "init_idx": 0, "seed": 8}, {"task_id": 1, "init_idx": 1, "seed": 7}],
    )
    eps, _, _ = sc_report.load(sc_report.resolve_runs(["ra", "rb"], tmp_path))
    cases, warns = sc_report.video_cases(eps, ["A", "B"], 10)
    assert [[r["init_idx"] for r in c] for c in cases] == [[1, 1]]  # init 0: seeds 7 vs 8 → not paired
    assert len(warns) == 1 and "seeds differ" in warns[0] and "init 0" in warns[0]


def test_report_video_ids_keep_repeats_apart(tmp_path):
    eps_a = [{"task_id": 1, "init_idx": 0, "seed": 7 + k, "repeat_idx": k} for k in range(3)]
    fake_run(tmp_path, "ra", "A", eps_a)
    fake_run(tmp_path, "rb", "B", eps_a)
    out = tmp_path / "rep"
    assert (
        sc_report.main(["--runs", "ra,rb", "--out-root", str(tmp_path), "--out", str(out), "--videos", "5"])
        == 0
    )
    man = json.loads((out / "manifest.json").read_text())
    assert len(man) == 3 and len({m["id"] for m in man}) == 3 and len({m["gif"] for m in man}) == 3
    assert sorted(m["repeat_idx"] for m in man) == [0, 1, 2]
    assert all(Path(m["mp4"]).exists() and Path(m["thumb"]).exists() for m in man)


def test_report_deduplicates_runs_and_episodes(tmp_path):
    base = [{"task_id": t, "init_idx": i, "seed": 7} for t in (1, 8) for i in range(2)]
    quick = fake_run(tmp_path, "bench-A-quick", "A", base[:2])
    fake_run(tmp_path, "bench-A-full", "A", base)  # full ⊃ quick
    # the same run as a path and again through the ckpt id (which also brings full)
    runs = sc_report.resolve_runs([str(quick), "A"], tmp_path)
    assert len(runs) == 2
    eps, dups, warns = sc_report.load(runs)
    assert len(eps) == 4 and len(dups) == 2 and not any(d["conflict"] for d in dups)
    assert {d["used"].split("/")[0] for d in dups} == {"bench-A-quick"}
    assert warns and "duplicate" in warns[0]
    # another exec mode is another sample, not a duplicate
    fake_run(tmp_path, "bench-A-K4", "A", base[:1], exec_mode="batched-4")
    eps, dups, _ = sc_report.load(sc_report.resolve_runs(["bench-A-quick", "bench-A-K4"], tmp_path))
    assert len(eps) == 3 and not dups
    # same key, different outcome → warning naming the run that was used
    fake_run(tmp_path, "bench-A-redo", "A", [{**base[0], "success": False}])
    eps, dups, warns = sc_report.load(sc_report.resolve_runs(["bench-A-quick", "bench-A-redo"], tmp_path))
    assert len(eps) == 2 and [d["conflict"] for d in dups] == [True]
    assert any("conflicting" in w and "used bench-A-quick" in w for w in warns)
    out = tmp_path / "rep"
    argv = [
        "--runs",
        "bench-A-quick,bench-A-redo",
        "--out-root",
        str(tmp_path),
        "--out",
        str(out),
        "--videos",
        "0",
    ]
    assert sc_report.main(argv) == 0
    assert "True" in (out / "duplicates.csv").read_text()
    assert "conflicting" in (out / "report.md").read_text()


def test_setup_server_syncs_shortcut_bench_extra():
    lines = (SHORTCUT / "setup_server.sh").read_text().splitlines()
    assert any(x.startswith("uv sync") and "--extra shortcut-bench" in x for x in lines)
