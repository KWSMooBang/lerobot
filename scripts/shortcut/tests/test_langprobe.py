"""CPU checks for the T018 language probe (analysis/langprobe): metric math, the query plan (same noise across
instructions), the whole make_obs → probe → measure flow against mock servers (lerobot, droid, starvla), and the
open-loop execution in the real LIBERO OSC and joint-velocity envs (subprocess, osmesa; skipped without LIBERO).

    uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_langprobe.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack")

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parent
LP = SHORTCUT / "analysis" / "langprobe"
sys.path[:0] = [str(LP), str(SHORTCUT), str(SHORTCUT / "parallel")]

import lp_common as lp  # noqa: E402
import measure  # noqa: E402
import probe  # noqa: E402

GL = os.environ.get("SC_TEST_MUJOCO_GL", "osmesa")


def run(script: str, *args: str, timeout: int = 900) -> str:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": GL}
    if GL == "osmesa":
        env["PYOPENGL_PLATFORM"] = "osmesa"
    p = subprocess.run(
        [sys.executable, str(LP / script), *args], env=env, capture_output=True, text=True, timeout=timeout
    )
    assert p.returncode == 0, f"{script} failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    return p.stdout


# ---------------------------------------------------------------- metric math


def test_interp_and_geometry() -> None:
    traj = np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0]])
    assert np.allclose(measure.interp_at(traj, 20, 0.05), [1, 0, 0])
    assert np.allclose(measure.interp_at(traj, 20, 0.075), [2, 0, 0])
    assert np.isnan(measure.interp_at(traj, 20, 0.2)).all()  # beyond the chunk
    assert measure.cos(np.array([1.0, 0, 0]), np.array([2.0, 0, 0])) == pytest.approx(1)
    assert np.isnan(measure.cos(np.zeros(3), np.ones(3)))
    q = np.array([0.0, 0, 0, 1])
    q90 = np.array([0.0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    assert measure.rot_angle(q, q90) == pytest.approx(np.pi / 2)
    assert measure.rot_angle(q90, -q90) == pytest.approx(0, abs=1e-6)  # double cover
    assert np.allclose(measure.quat_xyzw_to_mat(q90) @ [1, 0, 0], [0, 1, 0])


def test_tilt_sign() -> None:
    u_o, u_n = np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    d_orig = np.array([0.1, 0, 0])
    assert measure.tilt(np.array([0, 0.1, 0]), d_orig, u_o, u_n) > 0  # towards the new target
    assert measure.tilt(d_orig, d_orig, u_o, u_n) == pytest.approx(0)
    assert measure.tilt(np.array([0.1, -0.05, 0]), d_orig, u_o, u_n) < 0


def test_action_diff() -> None:
    a = np.zeros((10, 7))
    b = np.zeros((10, 7))
    b[:, :3] = 0.3  # arm L2 per step = sqrt(3·0.09)
    a[:, 6], b[:, 6] = -1, -1
    b[:4, 6] = 1
    l2, flip = measure.action_diff(a, b, 8)
    assert l2 == pytest.approx(np.sqrt(0.27))
    assert flip == pytest.approx(4 / 8)


def _disp_rows() -> pd.DataFrame:
    """One model, one observation: seeds differ by 1 mm, blank by 10 mm, a swap by 20 mm towards +y."""
    base = {"model": "m", "obs_idx": 0, "obs_id": "A/x/t0/i0", "suite": "libero_goal", "task_id": 0,
            "action_space": "osc_pose_delta", "quat_t": [0.0, 0, 0, 1], "grip_cmd": -1.0, "p0": [0.0, 0, 0]}  # fmt: skip
    rows = [
        {**base, "instr_id": "orig_s0", "kind": "orig", "swap_type": None, "dx": 0.05, "dy": 0.0, "dz": 0.0},
        {**base, "instr_id": "orig_s1", "kind": "orig", "swap_type": None, "dx": 0.051, "dy": 0.0, "dz": 0.0},
        {
            **base,
            "instr_id": "orig_s2",
            "kind": "orig",
            "swap_type": None,
            "dx": 0.05,
            "dy": 0.001,
            "dz": 0.0,
        },
        {**base, "instr_id": "blank", "kind": "blank", "swap_type": None, "dx": 0.04, "dy": 0.0, "dz": 0.0},
        {
            **base,
            "instr_id": "C4-a",
            "kind": "swap",
            "swap_type": "object",
            "dx": 0.05,
            "dy": 0.02,
            "dz": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def test_pairs_and_summary() -> None:
    pairs = measure.pair_table(_disp_rows())
    assert sorted(pairs[pairs.role == "floor"].pair) == [
        "orig_s1~orig_s0",
        "orig_s2~orig_s0",
        "orig_s2~orig_s1",
    ]
    eff = pairs[pairs.role == "effect"].set_index("alt_kind").eff_m
    assert eff["blank"] == pytest.approx(0.01) and eff["swap_object"] == pytest.approx(0.02)
    s = measure.summarize_pairs(pairs, ["model"]).set_index("alt_kind")
    floor_mm = 1000 * np.mean([0.001, 0.001, np.sqrt(2) * 0.001])
    assert s.loc["blank", "floor_mm"] == pytest.approx(floor_mm)
    assert s.loc["swap_object", "ratio"] == pytest.approx(20 / floor_mm)
    assert s.loc["swap", "n_pairs"] == 1 and s.loc["swap", "frac_above"] == 1.0


def test_align_table() -> None:
    disp = _disp_rows()
    probe_df = disp.assign(orig_target=None, new_target=None)
    probe_df.loc[probe_df.instr_id == "C4-a", ["orig_target", "new_target"]] = ["bowl_1", "plate_1"]
    meta = [{"obs_idx": 0, "objects": {"bowl_1": [0.3, 0.0, 0.0], "plate_1": [0.0, 0.3, 0.0]}}]
    al = measure.align_table(disp, probe_df, meta)
    assert len(al) == 1
    r = al.iloc[0]
    assert r.tilt > 0 and r.cos_shift == pytest.approx(np.cos(np.pi / 4)) and r.cos_shift_xy > 0.7
    assert abs(r.tilt_null) < r.tilt


# ---------------------------------------------------------------- query plan


def test_query_plan_shares_noise() -> None:
    sets = lp.load_instruction_sets()
    rows = [
        {"obs_idx": 0, "obs_id": "A/libero_goal/t1/i3", "suite": "libero_goal", "task_id": 1, "init_idx": 3}
    ]
    q = probe.plan_queries("A", rows, sets, repeat_k=1)
    seeds = {x["instr_id"]: x["seed"] for x in q}
    s0 = lp.episode_seed(7, 0, 1, 3)  # = sc-bench's episode seed (seed 7, repeat 0)
    assert seeds["orig_s0"] == seeds["blank"] == seeds["C4-a"] == seeds["orig_s0_repeat"] == s0
    assert len({seeds["orig_s0"], seeds["orig_s1"], seeds["orig_s2"]}) == 3
    entry = lp.task_entry(sets, "libero_goal", 1)
    assert len(q) == 3 + 1 + len(entry["swaps"]) + 1
    assert next(x for x in q if x["instr_id"] == "blank")["instruction"] == ""
    # full set A: 40 tasks × 5 inits × (3 seeds + blank + all swaps)
    n = lp.count_queries(sets, [(s, t) for s in lp.SUITES for t in range(10)])
    n_swaps = sum(len(t["swaps"]) for s in lp.SUITES for t in sets[s]["tasks"])
    assert n == {"orig": 120, "blank": 40, "swap": n_swaps}


# ---------------------------------------------------------------- mock flow


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    t = tmp_path_factory.mktemp("lp")
    run("make_obs.py", "--out", str(t / "obs"), "--backend", "mock", "--suites", "libero_goal,libero_10",
        "--tasks", "1,4", "--inits", "2", "--workers", "1")  # fmt: skip
    args = ["--backend", "mock", "--obs-dir", str(t / "obs"), "--out", str(t / "run"), "--registry", str(t / "r.tsv"),
            "--models", "pi05lib,pi05droid,starvla-qwen3pi-libero", "--repeat-check", "2"]  # fmt: skip
    run("probe.py", *args)
    out = run("probe.py", *args)  # resume: nothing new
    assert "'n_new': 0" in out and "'n_new': 1" not in out
    run(
        "measure.py",
        "--backend",
        "mock",
        "--run",
        str(t / "run"),
        "--obs-dir",
        str(t / "obs"),
        "--workers",
        "1",
    )
    return t


def test_mock_obs_manifest(mock_run: Path) -> None:
    man = json.loads((mock_run / "obs" / "manifest.json").read_text())
    for s in ("A", "B"):
        assert man["sets"][s]["npz_sha256"] == lp.sha256_file(mock_run / "obs" / f"set_{s}.npz")
    arrays, rows = lp.load_set(mock_run / "obs", "A")
    assert len(rows) == 8 and arrays["image_360"].shape == (8, 360, 360, 3) and "joint_pos" in arrays
    arrays_b, rows_b = lp.load_set(mock_run / "obs", "B")
    assert len(rows_b) == 4 * 2 * 3 and "joint_pos" not in arrays_b


def test_mock_noise_fixed(mock_run: Path) -> None:
    for m in ("pi05lib", "pi05droid", "starvla-qwen3pi-libero"):
        meta = json.loads((mock_run / "run" / m / "probe_meta.json").read_text())
        assert meta["sets"]["A"]["repeat_max_abs_diff"] == 0.0, m  # same seed + same instruction → same chunk
        df = pd.read_parquet(mock_run / "run" / m / "probe_A.parquet")
        a = {(r.obs_idx, r.instr_id): np.asarray(r.actions) for r in df.itertuples()}
        assert any(
            not np.array_equal(a[(o, "orig_s0")], a[(o, "orig_s1")]) for o, _ in a
        )  # other seed differs
        assert any(
            not np.array_equal(a[(o, "orig_s0")], a[(o, "blank")]) for o, _ in a
        )  # text matters (mock)
    droid = json.loads((mock_run / "run" / "pi05droid" / "probe_meta.json").read_text())
    assert droid["sets"]["B"] == {"skipped": "no joint_pos"}
    assert (
        json.loads((mock_run / "run" / "pi05droid" / "hello.json").read_text())["action_space"]
        == "joint_velocity"
    )


def test_mock_measure_outputs(mock_run: Path) -> None:
    out = mock_run / "run" / "measure"
    disp = pd.read_csv(out / "metrics_displacement.csv")
    assert set(disp.model) == {"pi05lib", "pi05droid", "starvla-qwen3pi-libero"}
    assert {"blank", "swap", "swap_object", "swap_target"} <= set(disp.alt_kind)
    assert (disp.floor_mm > 0).all() and disp.ratio.notna().all()
    act = pd.read_csv(out / "metrics_action.csv")
    assert set(act[act.set == "B"].model) == {"pi05lib", "starvla-qwen3pi-libero"}
    assert act[act.set == "B"].gt_l2.notna().all()
    assert (out / "summary.md").read_text().startswith("# Language probe")
    meta = json.loads((out / "measure_meta.json").read_text())
    assert meta["n_obs_A"] == 8 and meta["nan_displacements"] == 0


def test_dry_run(mock_run: Path) -> None:
    out = run("probe.py", "--backend", "mock", "--obs-dir", str(mock_run / "obs"), "--models",
              "pi05droid,pi05lib", "--dry-run", "--est-ms", "pi05lib=100")  # fmt: skip
    assert "skipped: no joint_pos" in out and "GPU-h" in out


# ---------------------------------------------------------------- real LIBERO sim (OSC + joint velocity)


@pytest.mark.skipif(
    importlib.util.find_spec("libero") is None or importlib.util.find_spec("robosuite") is None,
    reason="LIBERO / robosuite not installed",
)
def test_libero_execution(tmp_path: Path) -> None:
    """Real set A for one task (2 inits), mock lerobot + droid servers, measured in the real OSC and joint envs:
    restore path == fresh reset + replay exactly; every chunk gives a finite displacement."""
    run("make_obs.py", "--out", str(tmp_path / "obs"), "--sets", "A", "--suites", "libero_goal", "--tasks", "1",
        "--inits", "2", "--workers", "1")  # fmt: skip
    arrays, rows = lp.load_set(tmp_path / "obs", "A")
    assert arrays["image_256"].shape == (2, 256, 256, 3) and arrays["image_360"].shape == (2, 360, 360, 3)
    assert "akita_black_bowl_1" in rows[0]["objects"] and rows[0]["seed"] == lp.episode_seed(7, 0, 1, 0)
    run("probe.py", "--backend", "mock", "--obs-dir", str(tmp_path / "obs"), "--out", str(tmp_path / "run"),
        "--registry", str(tmp_path / "r.tsv"), "--models", "pi05lib,pi05droid", "--sets", "A")  # fmt: skip
    run("measure.py", "--run", str(tmp_path / "run"), "--obs-dir", str(tmp_path / "obs"), "--workers", "2",
        "--self-check", "2")  # fmt: skip
    meta = json.loads((tmp_path / "run" / "measure" / "measure_meta.json").read_text())
    assert meta["self_check_n"] == 2 and meta["self_check_max_dev_m"] == 0.0
    disp = pd.read_parquet(tmp_path / "run" / "measure" / "displacement_A.parquet")
    assert set(disp.action_space) == {"osc_pose_delta", "joint_velocity"}
    assert disp[["dx", "dy", "dz"]].notna().all().all()
    assert (disp.groupby("action_space").disp_m.max() > 1e-3).all()  # both envs move the arm


# ---------------------------------------------------------------- review fixes (resume / cache identity, set B, NaN)


def run_fail(script: str, *args: str) -> str:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run(
        [sys.executable, str(LP / script), *args], env=env, capture_output=True, text=True, timeout=600
    )
    assert p.returncode != 0, f"{script} should have failed:\n{p.stdout[-2000:]}"
    return p.stdout + p.stderr


def _copy_run(mock_run: Path, dst: Path) -> Path:
    import shutil

    shutil.copytree(mock_run / "obs", dst / "obs")
    shutil.copytree(mock_run / "run", dst / "run")
    return dst


def _rewrite_first(path: Path, **change: object) -> None:
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec.update(change)
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    "change", [{"instruction": "put the plate on the stove"}, {"seed": 1}, {"obs_npz_sha256": "0" * 64}]
)
def test_probe_resume_identity(mock_run: Path, tmp_path: Path, change: dict) -> None:
    t = _copy_run(mock_run, tmp_path)
    jl = t / "run" / "pi05lib" / "probe_A.jsonl"
    assert json.loads(jl.read_text().splitlines()[0])["obs_npz_sha256"] == lp.set_sha(t / "obs", "A")
    _rewrite_first(jl, **change)
    out = run_fail("probe.py", "--backend", "mock", "--obs-dir", str(t / "obs"), "--out", str(t / "run"),
                   "--registry", str(t / "r.tsv"), "--models", "pi05lib", "--sets", "A", "--repeat-check", "2")  # fmt: skip
    assert "asked differently" in out


def test_probe_resume_legacy_rows() -> None:
    """Rows without obs_npz_sha256 (pre-fix runs): text and seed are still checked, the run continues."""
    rows = [{"obs_idx": 0, "instr_id": "orig_s0", "instruction": "a", "seed": 1}]
    q = [{"obs_idx": 0, "instr_id": "orig_s0", "instruction": "a", "seed": 1}]
    assert probe.check_resume(rows, q, "x", Path("p")) == {(0, "orig_s0")}
    with pytest.raises(SystemExit):
        probe.check_resume([{**rows[0], "seed": 2}], q, "x", Path("p"))


def test_measure_cache_identity(mock_run: Path, tmp_path: Path) -> None:
    t = _copy_run(mock_run, tmp_path)
    args = ["--backend", "mock", "--run", str(t / "run"), "--obs-dir", str(t / "obs"), "--workers", "1"]
    ident_path = t / "run" / "pi05lib" / "exec_A.identity.json"
    ident = json.loads(ident_path.read_text())
    assert ident["obs_A_npz_sha256"] == lp.set_sha(t / "obs", "A") and ident["backend"] == "mock"
    assert "executing" not in run("measure.py", *args)  # unchanged → cache
    ident["obs_A_npz_sha256"] = "0" * 64  # as if the cache came from another observation file
    ident_path.write_text(json.dumps(ident))
    out = run("measure.py", *args)
    n_pi05lib = len(pd.read_parquet(t / "run" / "pi05lib" / "exec_A.parquet"))
    assert f"executing {n_pi05lib} chunks" in out  # only the stale model is re-executed
    df = pd.read_parquet(t / "run" / "pi05lib" / "probe_A.parquet")
    df["obs_npz_sha256"] = "1" * 64  # probe rows from another observation file → refused
    df.to_parquet(t / "run" / "pi05lib" / "probe_A.parquet", index=False)
    assert "probed on observation file" in run_fail("measure.py", *args)


def test_episode_files_split(tmp_path: Path) -> None:
    import make_obs
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = tmp_path / "data" / "chunk-000"
    d.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0, 0, 1]}), d / "file-000.parquet")
    pq.write_table(pa.table({"episode_index": [2]}), d / "file-001.parquet")
    files = make_obs.episode_files(tmp_path)
    assert files == {0: d / "file-000.parquet", 1: d / "file-000.parquet", 2: d / "file-001.parquet"}
    pq.write_table(pa.table({"episode_index": [2, 3]}), d / "file-002.parquet")
    with pytest.raises(SystemExit, match="span several parquet files"):
        make_obs.episode_files(tmp_path)


def test_frac_above_skips_nan() -> None:
    eff = pd.Series([0.02, 0.01, np.nan, 0.03])
    floor = pd.Series([0.01, 0.02, 0.01, np.nan])
    assert measure.frac_above(eff, floor) == pytest.approx(0.5)  # 2 valid pairs, 1 above
    assert np.isnan(measure.frac_above(pd.Series([np.nan]), pd.Series([0.1])))
