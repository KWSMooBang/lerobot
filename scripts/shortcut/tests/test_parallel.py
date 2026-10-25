"""CPU-only checks for the multi-GPU path (mock env + mock policy, fake GPU ids):
uv run pytest scripts/shortcut/tests/test_parallel.py -q

The reference is run_eval.py's single-env loop on the same mock backend; the parallel path (queue, work
stealing, batched env subprocesses, per-episode noise generators, merge, resume) must reproduce it exactly
except for timing fields.
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
import torch

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parent
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "parallel")]

import metrics  # noqa: E402
import run_eval  # noqa: E402
import run_multi  # noqa: E402
from bundle import PolicyBundle  # noqa: E402
from common import check_resume, episode_seed  # noqa: E402
from mock_backend import load_mock_policy  # noqa: E402
from resources import ThroughputModel, choose  # noqa: E402
from workqueue import WorkQueue, write_queue  # noqa: E402

from lerobot.policies.common.flow_matching import sample_noise  # noqa: E402

FIXTURE = HERE / "fixtures" / "libero_goal_2tasks.json"
CONDS = ["C0", "C1", "C3", "C4", "C5"]
COMMON = ["--instr-set", str(FIXTURE), "--tasks", "1,8", "--episodes", "4", "--video", "first:1"]
SKIP_FIELDS = set(run_eval.TIMING_FIELDS)


def run_single_reference(root: Path, conds: list[str] = CONDS, tag: str = "pi05lib") -> Path:
    for c in conds:
        run_eval.main(
            ["--policy", "pi05", "--checkpoint", f"mock/{tag}", "--model-tag", tag, "--suite", "libero_goal"]
            + ["--condition", c, "--backend", "mock", "--out", str(root / "runs" / f"sc-{tag}-goal-{c}")]
            + COMMON
        )
    return root


def multi_argv(root: Path, *extra: str, conds: list[str] = CONDS, ckpt: str = "pi05lib") -> list[str]:
    return [
        "--backend", "mock", "--ckpt", ckpt, "--suites", "goal", "--conds", ",".join(conds),
        "--out-root", str(root), *COMMON, *extra,
    ]  # fmt: skip


def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in SKIP_FIELDS}


def records(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "episodes.jsonl").read_text().splitlines()]


def assert_same_runs(ref_root: Path, root: Path, conds: list[str] = CONDS, tag: str = "pi05lib") -> None:
    for c in conds:
        a_dir, b_dir = ref_root / "runs" / f"sc-{tag}-goal-{c}", root / "runs" / f"sc-{tag}-goal-{c}"
        a, b = records(a_dir), records(b_dir)
        key = lambda r: (r["task_id"], r["init_idx"], r["variant"])  # noqa: E731
        assert len({key(r) for r in b}) == len(b), "duplicate records"
        assert [strip(r) for r in sorted(a, key=key)] == [strip(r) for r in sorted(b, key=key)], c
        # merged file is in (task, init, variant-position) order
        assert [(r["task_id"], r["init_idx"]) for r in b] == sorted((r["task_id"], r["init_idx"]) for r in b)
        for r in b:
            x, y = np.load(a_dir / r["steps_file"]), np.load(b_dir / r["steps_file"])
            assert set(x.files) == set(y.files)
            for f in x.files:
                assert np.array_equal(x[f], y[f]), (c, r["steps_file"], f)
            assert (r["video"] is None) or (b_dir / r["video"]).exists()
        assert not (b_dir / "parts").exists()
        for d in ("steps", "videos"):
            assert sorted(p.relative_to(a_dir) for p in (a_dir / d).rglob("*")) == sorted(
                p.relative_to(b_dir) for p in (b_dir / d).rglob("*")
            )


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return run_single_reference(tmp_path_factory.mktemp("single"))


# ---------------------------------------------------------------- noise


def test_episode_generator_reproduces_global_seed_stream():
    shape = (1, 16, 8)
    torch.manual_seed(1234)
    want = [sample_noise(shape, "cpu") for _ in range(3)]
    policy, *procs, _, meta = load_mock_policy("mock/x", 10)
    bundle = PolicyBundle(policy, *procs, meta, "cpu", 10)
    gen = bundle.generator(1234)
    torch.manual_seed(999)  # the global RNG must not matter
    got = [bundle.draw_noise(gen) for _ in range(3)]
    assert all(torch.equal(a, b) for a, b in zip(want, got, strict=True))


def test_batched_rows_do_not_depend_on_batch_mates():
    policy, *procs, _, meta = load_mock_policy("mock/x", 10)
    bundle = PolicyBundle(policy, *procs, meta, "cpu", 10)
    obs = [
        {
            "pixels": {"image": np.full((8, 8, 3), i, np.uint8), "image2": np.zeros((8, 8, 3), np.uint8)},
            "robot_state": {"eef": {"pos": np.full(3, 0.01 * i), "quat": np.zeros(4)}, "gripper": {"qpos": np.zeros(2)}},
        }
        for i in range(3)
    ]  # fmt: skip
    noise = torch.cat([bundle.draw_noise(bundle.generator(s)) for s in (1, 2, 3)])
    full = bundle.infer(obs, ["a", "b", "c"], noise)
    alone = bundle.infer(obs[1:2], ["b"], noise[1:2])
    assert np.array_equal(full[1:2], alone)


# ---------------------------------------------------------------- queue


def _units(n_per_group=6):
    out = []
    for g in ("g0", "g1"):
        for i in range(n_per_group):
            out.append({"uid": f"{g}/u{i}", "group": g, "suite": "s", "task_id": i % 2})
    return out


def test_queue_claims_are_unique_and_prefer_location(tmp_path):
    write_queue(tmp_path, _units())
    a, b = WorkQueue(tmp_path, "wa"), WorkQueue(tmp_path, "wb")
    first = a.claim("g0", prefer=("s", 1))
    assert first["task_id"] == 1
    got = [first["uid"]]
    while True:
        u = (b if len(got) % 2 else a).claim("g0")
        if u is None:
            break
        got.append(u["uid"])
    assert sorted(got) == sorted(f"g0/u{i}" for i in range(6))
    assert a.pending_groups() == ["g1"]
    # work stealing: the drained worker takes the other group's units
    assert b.claim("g1")["group"] == "g1"


def test_queue_release_and_fail_after_attempts(tmp_path):
    write_queue(tmp_path, _units(2))
    w = WorkQueue(tmp_path, "w")
    coord = WorkQueue(tmp_path, "coord")
    u = w.claim("g0")
    released, failed = coord.release_worker("w", max_attempts=2)
    assert released == [u["uid"]] and not failed
    assert w.claim("g0")["uid"] == u["uid"]  # released unit comes back first
    released, failed = coord.release_worker("w", max_attempts=2)
    assert failed == [u["uid"]]
    snap = coord.snapshot()
    assert snap["fail"] == [u["uid"]] and len(snap["pending"]) == 3


def test_torn_part_line_is_skipped(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n" + '{"a": 2, "b"')
    assert run_multi.read_records(p) == [{"a": 1}]


# ---------------------------------------------------------------- end to end (mock)


def test_multi_matches_single(reference, tmp_path):
    rc = run_multi.main(multi_argv(tmp_path, "--gpus", "0,1,2", "--procs-per-gpu", "2", "--num-envs", "3"))
    assert rc == 0
    assert_same_runs(reference, tmp_path)
    meta = json.loads((tmp_path / "runs" / "sc-pi05lib-goal-C4" / "run_meta.json").read_text())
    ref_meta = json.loads((reference / "runs" / "sc-pi05lib-goal-C4" / "run_meta.json").read_text())
    assert meta["identity"]["exec_mode"] == "batched-3" and ref_meta["identity"]["exec_mode"] == "single"
    same = {k for k in ref_meta["identity"] if k != "exec_mode"}
    assert all(meta["identity"][k] == ref_meta["identity"][k] for k in same)
    assert meta["inference"]["policy_type"] == "mock" and meta["exec"]["num_envs"] == 3
    tsv = (tmp_path / "results.tsv").read_text().splitlines()
    assert tsv[0].startswith("run_id\tstatus") and len(tsv) == 1 + len(CONDS)
    assert all("\tdone\t" in line for line in tsv[1:])


def test_run_eval_num_envs_and_n1_is_single(reference, tmp_path):
    out = tmp_path / "runs" / "sc-pi05lib-goal-C4"
    argv = [
        "--policy",
        "pi05",
        "--checkpoint",
        "mock/pi05lib",
        "--model-tag",
        "pi05lib",
        "--suite",
        "libero_goal",
    ]
    run_eval.main(
        [*argv, "--condition", "C4", "--backend", "mock", "--out", str(out), "--num-envs", "4", *COMMON]
    )
    assert_same_runs(reference, tmp_path, ["C4"])
    assert {p.name for p in out.iterdir()} == {
        "episodes.jsonl",
        "run_meta.json",
        "steps",
        "stdout.log",
        "videos",
    }
    rc = run_multi.main(multi_argv(tmp_path / "n1", "--gpus", "0", "--num-envs", "1", conds=["C0"]))
    assert rc == 0
    meta = json.loads((tmp_path / "n1" / "runs" / "sc-pi05lib-goal-C0" / "run_meta.json").read_text())
    assert meta["identity"]["exec_mode"] == "single"


def test_worker_crash_is_released_and_rerun(reference, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SC_TEST_HOOKS", "claim_exit:sc-pi05lib-goal-C3/t08/p1/i02:1")
    rc = run_multi.main(multi_argv(tmp_path, "--gpus", "7,9", "--procs-per-gpu", "1", "--num-envs", "2"))
    assert rc == 0
    assert "exited 9: released" in capsys.readouterr().out
    assert_same_runs(reference, tmp_path)


def test_resume_after_kill_skips_done_units(reference, tmp_path):
    argv = multi_argv(tmp_path, "--gpus", "0", "--procs-per-gpu", "1", "--num-envs", "2")
    cmd = [sys.executable, str(SHORTCUT / "parallel" / "run_multi.py"), *argv]
    proc = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    n = 0
    while time.time() < deadline and proc.poll() is None:
        parts = list((tmp_path / "runs").glob("*/parts/*.jsonl"))
        n = sum(len(p.read_text().splitlines()) for p in parts)
        if n >= 8:
            break
        time.sleep(0.02)
    assert proc.poll() is None, "run finished before it could be interrupted"
    os.killpg(proc.pid, signal.SIGKILL)  # coordinator, workers and env subprocesses
    proc.wait()
    total = 4 * 2 * (1 + 1 + 3 + 1 + 1)  # episodes x tasks x variants per condition
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    pending = int(out.stdout.split("[multi] ")[1].split(" pending")[0])
    assert pending <= total - n
    assert_same_runs(reference, tmp_path)


def test_two_checkpoints_steal_and_pair(tmp_path):
    ref = run_single_reference(tmp_path / "ref", ["C0", "C4"], "pi05base")
    rc = run_multi.main(
        multi_argv(
            tmp_path / "m",
            "--gpus",
            "0,1",
            "--procs-per-gpu",
            "1",
            "--num-envs",
            "2",
            conds=["C0", "C4"],
            ckpt="pi05lib,pi05base",
        )
    )
    assert rc == 0
    assert_same_runs(ref, tmp_path / "m", ["C0", "C4"], "pi05base")


def test_dry_run_counts_units(tmp_path, capsys):
    assert run_multi.main(multi_argv(tmp_path, "--gpus", "0,1,2,3", "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "total: 56 units to run" in out and "GPU-h" in out
    assert not (tmp_path / "runs").exists()


# ---------------------------------------------------------------- helpers


def test_auto_resources():
    p, n, why = choose(["0", "1", "2", "3"], "auto", "auto", 80.0, cores=64)
    assert p == 2 and n == 7 and any("num-envs=7" in w for w in why)
    p, n, _ = choose(["0"], "auto", "auto", 24.0, cores=224)
    assert p == 1 and n == 15
    assert choose(["0", "1"], "auto", "auto", None, cores=4)[:2] == (1, 1)
    m = ThroughputModel()
    assert m.gpu_steps_per_s(2, 8, 10) > 1000 / m.single_ms_per_step


def test_legacy_identity_resumes_as_single(tmp_path):
    ident = {"checkpoint": "c", "suite": "libero_goal", "condition": "C0", "repeat_idx": 0, "seed": 7}
    old_meta = {"identity": ident, "gpu": "NVIDIA GeForce RTX 4090", "torch": torch.__version__}
    ep = tmp_path / "episodes.jsonl"
    new = {
        **ident,
        "exec_mode": "single",
        "gpu": "NVIDIA GeForce RTX 4090",
        "torch": torch.__version__,
        "cuda": "12.8",
    }
    check_resume(new, old_meta, ep)
    with pytest.raises(SystemExit, match="another configuration"):
        check_resume({**new, "exec_mode": "batched-8"}, old_meta, ep)
    with pytest.raises(SystemExit, match="another configuration"):
        check_resume({**new, "gpu": "NVIDIA A100-SXM4-80GB"}, old_meta, ep)


def _ep(cond, ok, exec_mode, init=0, variant="orig"):
    return {
        "model": "m", "ckpt": "c", "suite": "libero_goal", "condition": cond, "variant": variant, "task_id": 0,
        "init_idx": init, "repeat_idx": 0, "seed": episode_seed(7, 0, 0, init), "success_orig": ok,
        "_root": exec_mode, "_exec": (exec_mode, "gpu"),
    }  # fmt: skip


def test_metrics_pairs_only_within_exec_mode(monkeypatch):
    c0 = [_ep("C0", True, "single", i) for i in range(3)]
    c4 = [_ep("C4", False, "batched-8", i, "a") for i in range(3)]
    ref, _ = metrics.index_c0(c0)
    assert metrics.aggregate_variants(c4, ref) == []
    metrics.check_unique(c0 + [_ep("C0", True, "batched-8", i) for i in range(3)])  # not duplicates
    monkeypatch.setattr(metrics, "ALLOW_MIXED_EXEC", True)
    ref, _ = metrics.index_c0(c0)
    assert len(metrics.aggregate_variants(c4, ref)) == 3


# ---------------------------------------------------------------- record commit vs. done mark


def _res(v: float) -> dict:
    return {"arrays": {"x": np.full(50, v)}, "frames": [np.full((64, 128, 3), int(v), np.uint8)] * 5}


def test_save_episode_is_atomic(tmp_path, monkeypatch):
    steps, video = run_eval.save_episode(tmp_path, "s/t00/C0_orig_init00", _res(1.0), True)
    rec = {"steps_file": steps, "video": video}
    before = {f: (tmp_path / f).read_bytes() for f in (steps, video)}
    assert run_multi.record_intact(tmp_path, rec)

    def dying_write(path, frames):  # killed after part of the new mp4 is on disk
        Path(path).write_bytes(b"\x00" * 100)
        raise KeyboardInterrupt

    monkeypatch.setattr(run_eval, "write_video", dying_write)
    with pytest.raises(KeyboardInterrupt):
        run_eval.save_episode(tmp_path, "s/t00/C0_orig_init00", _res(2.0), True)
    # the npz was replaced by a complete new copy, the mp4 still is the complete old one
    assert (tmp_path / video).read_bytes() == before[video]
    assert np.array_equal(np.load(tmp_path / steps)["x"], np.full(50, 2.0))
    assert run_multi.record_intact(tmp_path, rec)
    (tmp_path / steps).write_bytes(before[steps][:40])  # a truncated file (pre-fix writer)
    assert not run_multi.record_intact(tmp_path, rec)


X = "sc-pi05lib-goal-C3/t08/p1/i02"


def _x_records(root: Path) -> list[dict]:
    run = root / "runs" / "sc-pi05lib-goal-C3"
    srcs = [run / "episodes.jsonl", *sorted((run / "parts").glob("*.jsonl"))]
    out = []
    for src in srcs:
        for r in run_multi.read_records(src):
            if (r["task_id"], r["variant"], r["init_idx"]) == (8, "p1", 2):
                out.append(r)
    return out


def test_committed_record_is_kept_on_retry(reference, tmp_path, monkeypatch, capsys):
    """Worker dies after committing X's record but before marking X done; the retry keeps the record."""
    monkeypatch.setenv("SC_TEST_HOOKS", f"commit_exit:{X}:1")
    rc = run_multi.main(multi_argv(tmp_path, "--gpus", "0", "--procs-per-gpu", "1", "--num-envs", "2"))
    assert rc == 0
    assert "exited 9: released" in capsys.readouterr().out
    logs = "".join(p.read_text() for p in (tmp_path / "multi").glob("*/w*.log"))
    assert f"{X}: record from an earlier attempt is intact; done" in logs
    assert len(_x_records(tmp_path)) == 1
    assert_same_runs(reference, tmp_path)


def test_commit_kill_then_retry_kill_then_rerun(reference, tmp_path):
    """Record of X committed → worker killed → the retry is killed with the whole session → rerun."""
    argv = multi_argv(tmp_path, "--gpus", "0", "--procs-per-gpu", "1", "--num-envs", "2")
    cmd = [sys.executable, str(SHORTCUT / "parallel" / "run_multi.py"), *argv]
    env = {**os.environ, "SC_TEST_HOOKS": f"commit_exit:{X}:1;claim_killgroup:{X}:2"}
    proc = subprocess.run(cmd, env=env, start_new_session=True, capture_output=True, text=True, timeout=300)
    assert proc.returncode == -signal.SIGKILL, proc.stdout + proc.stderr
    first = _x_records(tmp_path)
    assert len(first) == 1
    run = tmp_path / "runs" / "sc-pi05lib-goal-C3"
    assert run_multi.record_intact(run, first[0])
    n_done = 0  # finished runs were already merged into episodes.jsonl; the rest sit in parts/
    for run_dir in (tmp_path / "runs").iterdir():
        keys = {
            run_multi.record_key(r)
            for src in [run_dir / "episodes.jsonl", *(run_dir / "parts").glob("*.jsonl")]
            for r in run_multi.read_records(src)
            if run_multi.record_files_exist(run_dir, r)
        }
        n_done += len(keys)
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    pending = int(out.stdout.split("[multi] ")[1].split(" pending")[0])
    assert pending == 56 - n_done  # X is not run again
    assert len(_x_records(tmp_path)) == 1
    assert not list((tmp_path / "runs").rglob("*.tmp-*"))
    assert_same_runs(reference, tmp_path)
