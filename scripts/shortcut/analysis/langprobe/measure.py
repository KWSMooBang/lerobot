"""T018 step 3: execute every probed chunk open-loop from the stored initial sim state, then the language metrics.
CPU only (MuJoCo physics, cameras off); runs anywhere the lerobot + LIBERO venv runs.

    uv run --extra shortcut-bench python scripts/shortcut/analysis/langprobe/measure.py \
        --run <probe out dir> [--obs-dir <obs>] [--models a,b] [--workers 40] [--t-eval 0.4]

Execution (set A): per (model, observation, instruction) the env of the model's action space (OSC_POSE 20 Hz =
lerobot LiberoEnv; joint_velocity 15 Hz = joint_env.JointVelocityLiberoEnv) is put into the observation's sim
state (MuJoCo qpos/qvel/time + qacc_warmstart + the Panda gripper controller's accumulated command, all from
make_obs), then the chunk's actions are stepped one by one. The EEF pose is read after each step. Both action
spaces end in the same unit: EEF displacement in world metres after a fixed sim time (--t-eval, default 0.4 s
= 8 OSC steps = 6 joint steps; the shortest chunk, StarVLA's 8 OSC steps, covers it).
--self-check N: for the first N set-A observations of each task the restore path is compared with a fresh env
reset (EpisodeStepper-style, as make_obs) + LiberoEnv.step replay (OSC only; must agree to float precision):
measure_meta.json → self_check_max_dev_m. Poses are read after sim.forward() with observables forced (read_pose).

Metrics (per model; set A also per suite and task):
  1. displacement  Δ(instr) = eef_pos(t_eval) − eef_pos(0).
       effect  = ‖Δ(alt) − Δ(orig_s0)‖ for alt ∈ {blank, every swap}   (same noise: only the text differs)
       floor   = ‖Δ(orig_si) − Δ(orig_sj)‖ over the 3 seed pairs        (same text, other noise)
       reported: means in mm, ratio = mean effect / mean floor, frac_above = share of (obs, alt) whose effect
       exceeds that observation's mean floor, |Δ(orig)| for scale; the same for the gripper command (mean of the
       last action dim over the executed steps, −1 open … +1 close) and the EEF rotation (angle between the two
       end orientations).
  2. goal alignment (swaps whose orig_target / new_target are single named objects or regions): with p0 the
     initial EEF and u_o / u_n the directions to the original / new target (object positions at the observation):
       tilt      = [cos(Δalt, u_n) − cos(Δalt, u_o)] − [cos(Δorig, u_n) − cos(Δorig, u_o)]     (> 0: towards new)
       cos_shift = cos(Δalt − Δorig, u_n − u_o)  (3-D and xy)
     null = the same with Δalt := Δ(orig_s1), Δ(orig_s2) (noise only). Reported per swap type (object / target).
  3. action space (sets A and B, compared only within an action space): over the first --n-act steps
       act_l2    = mean_k ‖a_alt[k, :-1] − a_orig[k, :-1]‖ (env-scale actions as the server returns them)
       grip_flip = mean_k [sign a_alt[k, −1] ≠ sign a_orig[k, −1]]
     with the same seed-pair floor; set B also gt_l2 = mean_k ‖a_orig_s0[k] − a_dataset[k]‖ (arm dims, context).

Output <run>/measure/: displacement_A.parquet (one row per probe query: Δ, rotation, gripper), pairs_A.parquet
(effect / floor pairs), align_A.parquet, actions_{A,B}.parquet, metrics_displacement.csv, metrics_align.csv,
metrics_action.csv, summary.md, measure_meta.json, arrows.png (a few observations, --figure).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parents[1]
sys.path[:0] = [p for p in (str(HERE), str(SHORTCUT), str(SHORTCUT / "parallel")) if p not in sys.path]

import lp_common as lp  # noqa: E402

ACTION_SPACE_OSC = "osc_pose_delta"
ACTION_SPACE_JOINT = "joint_velocity"
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
N_ACT_DEFAULT = 8
ALT_KINDS = ("blank", "swap", "swap_object", "swap_target")


# ---------------------------------------------------------------- geometry (pure numpy, unit-tested)


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle (rad) of the rotation between two xyzw quaternions."""
    a = abs(float(np.dot(np.asarray(q1) / np.linalg.norm(q1), np.asarray(q2) / np.linalg.norm(q2))))
    return float(2 * np.arccos(min(1.0, a)))


def cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def interp_at(traj: np.ndarray, hz: float, t: float) -> np.ndarray:
    """traj[k] = value after k steps (k = 0 … n); linear interpolation at time t (NaN beyond the chunk)."""
    x = t * hz
    n = len(traj) - 1
    if x > n + 1e-9:
        return np.full(traj.shape[1:], np.nan)
    k = min(int(np.floor(x + 1e-9)), n)
    if k == n or abs(x - k) < 1e-9:
        return np.asarray(traj[k], dtype=np.float64)
    f = x - k
    return (1 - f) * traj[k] + f * traj[k + 1]


def tilt(d_alt: np.ndarray, d_orig: np.ndarray, u_o: np.ndarray, u_n: np.ndarray) -> float:
    return (cos(d_alt, u_n) - cos(d_alt, u_o)) - (cos(d_orig, u_n) - cos(d_orig, u_o))


def action_diff(a: np.ndarray, b: np.ndarray, n: int) -> tuple[float, float]:
    """(mean arm L2, gripper sign-flip rate) over the first n steps."""
    k = min(n, len(a), len(b))
    a, b = np.asarray(a[:k], dtype=np.float64), np.asarray(b[:k], dtype=np.float64)
    l2 = float(np.mean(np.linalg.norm(a[:, :-1] - b[:, :-1], axis=1)))
    flip = float(np.mean(np.sign(a[:, -1]) != np.sign(b[:, -1])))
    return l2, flip


# ---------------------------------------------------------------- sims


def read_pose(e: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EEF pos, quat (xyzw), gripper qpos of the current sim state. sim.forward() first (MuJoCo's step leaves the
    kinematics one substep behind) and observables forced (they sample on their own timer and can lag a substep,
    ~1 mm at 0.5 m/s; the harness's logged poses have that lag, these do not)."""
    e.sim.forward()
    o = e._get_observations(force_update=True)
    return np.array(o["robot0_eef_pos"]), np.array(o["robot0_eef_quat"]), np.array(o["robot0_gripper_qpos"])


class LiberoSim:
    """One task's env in one action space; restore(i) puts it into set-A observation i's state."""

    def __init__(self, suite: str, task_id: int, action_space: str, arrays: dict[str, np.ndarray]):
        from envproc import _make_env
        from run_eval import inner

        spec: dict[str, Any] = {"suite": suite, "task_id": task_id, "max_steps": 10_000, "env_hw": [64, 64],
                                "control_mode": "relative"}  # fmt: skip
        if action_space == ACTION_SPACE_JOINT:
            spec["action_space"] = ACTION_SPACE_JOINT
        self.space = action_space
        self.hz = lp.CONTROL_HZ[action_space]
        self.env = _make_env("libero", spec, {})
        self.env.init_state_id = 0
        self.env.reset(seed=0)  # builds the scene (+ JOINT_POSITION controller / feed-forward patch)
        self.e = inner(self.env)
        self.raw = self.env._env
        self.e.ignore_done = True
        for cam in CAMERAS:
            self.e.modify_observable(cam, "enabled", False)
        self.arrays = arrays

    def restore(self, i: int) -> None:
        a = self.arrays
        n = int(a["sim_state_len"][i])
        state = a["sim_state"][i, :n]
        sim = self.e.sim
        if n != 1 + sim.model.nq + sim.model.nv:
            raise RuntimeError(
                f"obs {i}: sim state of length {n}, env has nq {sim.model.nq} nv {sim.model.nv}"
            )
        sim.set_state_from_flattened(state)
        nv = int(a["qacc_warmstart_len"][i])
        sim.data.qacc_warmstart[:] = a["qacc_warmstart"][i, :nv]
        sim.forward()
        self.e.robots[0].gripper.current_action = np.array(a["gripper_action"][i], dtype=np.float64)
        self.e.timestep = 0
        self.e.done = False

    def pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return read_pose(self.e)

    def step(self, action: np.ndarray) -> None:
        self.raw.step(np.asarray(action, dtype=np.float64))

    def close(self) -> None:
        self.env.close()


class MockSim:
    """mock_backend.MockEnv / MockJointEnv with the state set from the mock set A (tests)."""

    def __init__(self, suite: str, task_id: int, action_space: str, arrays: dict[str, np.ndarray]):
        from mock_backend import MockEnv, MockJointEnv, mock_suite

        cls = MockJointEnv if action_space == ACTION_SPACE_JOINT else MockEnv
        self.env = cls(mock_suite(suite).task_info[task_id], 10_000, (8, 8))
        self.env.reset()
        self.space = action_space
        self.hz = lp.CONTROL_HZ[action_space]
        self.arrays = arrays

    def restore(self, i: int) -> None:
        s = self.arrays["sim_state"][i]
        self.env.pos = np.array(s[:3], dtype=np.float64)
        self.env.grip = float(s[3])
        if hasattr(self.env, "q"):
            self.env.q = np.zeros(7)

    def pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = 0.04 if self.env.grip <= 0 else 0.01
        return self.env.pos.copy(), np.array([0.0, 0.0, 0.0, 1.0]), np.array([w, -w])

    def step(self, action: np.ndarray) -> None:
        self.env.step(np.asarray(action, dtype=np.float64))

    def close(self) -> None:
        pass


def rollout(sim: Any, i: int, actions: np.ndarray) -> dict[str, np.ndarray]:
    sim.restore(i)
    poses = [sim.pose()]
    for a in actions:
        sim.step(a)
        poses.append(sim.pose())
    pos, quat, grip = (np.array(x) for x in zip(*poses, strict=True))
    return {"pos": pos, "quat": quat, "grip": grip}


def _exec_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    """All probe rows of one (task, action space): run in a worker process."""
    import torch

    torch.set_num_threads(1)
    arrays, _ = lp.load_set(job["obs_dir"], "A", verify=False)
    cls = MockSim if job["backend"] == "mock" else LiberoSim
    sim = cls(job["suite"], job["task_id"], job["action_space"], arrays)
    out = []
    for r in job["rows"]:
        acts = np.asarray(r["actions"], dtype=np.float64).reshape(r["n_actions"], r["action_dim"])
        tr = rollout(sim, r["obs_idx"], acts)
        out.append({"model": r["model"], "obs_idx": r["obs_idx"], "instr_id": r["instr_id"], "hz": sim.hz,
                    "pos": tr["pos"].ravel().tolist(), "quat": tr["quat"].ravel().tolist(),
                    "grip": tr["grip"].ravel().tolist()})  # fmt: skip
    checks = []
    if job.get("self_check") and job["backend"] != "mock" and job["action_space"] == ACTION_SPACE_OSC:
        checks = self_check(job, arrays, sim)
    sim.close()
    return [*out, *({"self_check": c} for c in checks)]


def self_check(job: dict[str, Any], arrays: dict[str, np.ndarray], sim: Any) -> list[float]:
    """Restore path vs a fresh EpisodeStepper-style reset + the same actions (OSC env): max |Δpos| in m."""
    from envproc import _make_env

    done: set[int] = set()
    devs = []
    env = _make_env("libero", {"suite": job["suite"], "task_id": job["task_id"], "max_steps": 10_000,
                               "env_hw": [64, 64], "control_mode": "relative"}, {})  # fmt: skip
    for r in job["rows"]:
        i = r["obs_idx"]
        if i in done or len(done) >= job["self_check"] or r["instr_id"] != "orig_s0":
            continue
        done.add(i)
        meta = job["meta"][str(i)]
        np.random.seed(meta["seed"] % 2**32)
        env.init_state_id = meta["init_idx"]
        env.reset(seed=meta["seed"])
        acts = np.asarray(r["actions"], dtype=np.float64).reshape(r["n_actions"], r["action_dim"])
        ref = [read_pose(env._env.env)[0]]
        for a in acts:
            env.step(a)
            ref.append(read_pose(env._env.env)[0])
        tr = rollout(sim, i, acts)
        devs.append(float(np.max(np.abs(np.array(ref) - tr["pos"]))))
    env.close()
    return devs


# ---------------------------------------------------------------- loading


def load_probe(run: Path, model: str, set_name: str) -> pd.DataFrame | None:
    p = run / model / f"probe_{set_name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["model"] = model
    return df


def execute(
    run: Path, obs_dir: Path, models: list[str], args: argparse.Namespace
) -> tuple[pd.DataFrame, list]:
    _, meta_rows = lp.load_set(obs_dir, "A", verify=not args.no_verify)
    meta = {str(r["obs_idx"]): {"seed": r["seed"], "init_idx": r["init_idx"]} for r in meta_rows}
    jobs, cached = [], []
    for m in models:
        df = load_probe(run, m, "A")
        if df is None:
            continue
        cache = run / m / "exec_A.parquet"
        if cache.exists() and cache.stat().st_mtime >= (run / m / "probe_A.parquet").stat().st_mtime:
            cached.append(pd.read_parquet(cache))
            continue
        df = df[df["kind"] != "repeat"]
        for (suite, tid, space), g in df.groupby(["suite", "task_id", "action_space"], sort=False):
            cols = ["model", "obs_idx", "instr_id", "actions", "n_actions", "action_dim"]
            jobs.append({"obs_dir": str(obs_dir), "backend": args.backend, "suite": suite, "task_id": int(tid),
                         "action_space": space, "rows": g[cols].to_dict("records"), "meta": meta,
                         "self_check": args.self_check})  # fmt: skip
    checks: list[float] = []
    if jobs:
        t0 = time.time()
        print(
            f"[measure] executing {sum(len(j['rows']) for j in jobs)} chunks in {len(jobs)} jobs", flush=True
        )
        if args.workers > 1:
            with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
                parts = pool.map(_exec_job, jobs, chunksize=1)
        else:
            parts = [_exec_job(j) for j in jobs]
        recs = [r for p in parts for r in p]
        checks = [r["self_check"] for r in recs if "self_check" in r]
        new = pd.DataFrame([r for r in recs if "self_check" not in r])
        print(f"[measure] executed in {time.time() - t0:.0f} s", flush=True)
        for m, g in new.groupby("model"):
            g.to_parquet(run / str(m) / "exec_A.parquet", index=False)
            cached.append(g)
    return (pd.concat(cached, ignore_index=True) if cached else pd.DataFrame()), checks


# ---------------------------------------------------------------- metrics


def displacement_table(probe: pd.DataFrame, ex: pd.DataFrame, t_eval: float) -> pd.DataFrame:
    """One row per executed query: Δ at t_eval, end orientation, mean gripper command."""
    df = probe[probe["kind"] != "repeat"].merge(ex, on=["model", "obs_idx", "instr_id"], how="inner")
    rows = []
    for r in df.itertuples(index=False):
        pos = np.asarray(r.pos).reshape(-1, 3)
        quat = np.asarray(r.quat).reshape(-1, 4)
        acts = np.asarray(r.actions).reshape(r.n_actions, r.action_dim)
        k = int(round(t_eval * r.hz))
        d = interp_at(pos, r.hz, t_eval) - pos[0]
        qk = quat[min(k, len(quat) - 1)]
        rows.append({"model": r.model, "obs_idx": r.obs_idx, "obs_id": r.obs_id, "suite": r.suite,
                     "task_id": r.task_id, "instr_id": r.instr_id, "kind": r.kind,
                     "swap_type": getattr(r, "swap_type", None), "action_space": r.action_space,
                     "dx": d[0], "dy": d[1], "dz": d[2], "disp_m": float(np.linalg.norm(d)),
                     "quat_t": qk.tolist(), "grip_cmd": float(np.mean(acts[: max(k, 1), -1])),
                     "p0": pos[0].tolist()})  # fmt: skip
    return pd.DataFrame(rows)


def alt_kind(kind: str, swap_type: Any) -> str:
    if kind == "blank":
        return "blank"
    return f"swap_{swap_type}" if isinstance(swap_type, str) and swap_type else "swap"


def pair_table(disp: pd.DataFrame) -> pd.DataFrame:
    """effect pairs (alt vs orig_s0) and floor pairs (orig_si vs orig_sj) per (model, observation)."""
    out = []
    for (m, i), g in disp.groupby(["model", "obs_idx"], sort=False):
        by = {r.instr_id: r for r in g.itertuples(index=False)}
        if "orig_s0" not in by:
            continue
        base = by["orig_s0"]

        def rec(a: Any, b: Any, pair: str, kind: str, a_kind: str, m: Any = m, i: Any = i) -> dict[str, Any]:
            da, db = np.array([a.dx, a.dy, a.dz]), np.array([b.dx, b.dy, b.dz])
            return {"model": m, "obs_idx": i, "obs_id": a.obs_id, "suite": a.suite, "task_id": a.task_id,
                    "pair": pair, "role": kind, "alt_kind": a_kind, "action_space": a.action_space,
                    "eff_m": float(np.linalg.norm(da - db)), "eff_grip": abs(a.grip_cmd - b.grip_cmd),
                    "eff_rot": rot_angle(a.quat_t, b.quat_t), "orig_disp_m": float(np.linalg.norm(db))}  # fmt: skip

        seeds = [by[f"orig_s{s}"] for s in range(lp.N_SEEDS) if f"orig_s{s}" in by]
        for x in range(len(seeds)):
            for y in range(x + 1, len(seeds)):
                out.append(
                    rec(seeds[y], seeds[x], f"{seeds[y].instr_id}~{seeds[x].instr_id}", "floor", "floor")
                )
        for r in g.itertuples(index=False):
            if r.kind in ("blank", "swap"):
                out.append(rec(r, base, f"{r.instr_id}~orig_s0", "effect", alt_kind(r.kind, r.swap_type)))
    return pd.DataFrame(out)


def summarize_pairs(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """effect vs floor per group × alt kind (blank / swap / swap_object / swap_target)."""
    floor_obs = (
        pairs[pairs.role == "floor"].groupby(["model", "obs_idx"])[["eff_m", "eff_grip", "eff_rot"]].mean()
    )
    eff = pairs[pairs.role == "effect"].join(floor_obs, on=["model", "obs_idx"], rsuffix="_floor")
    eff = pd.concat([eff, eff[eff.alt_kind.str.startswith("swap_")].assign(alt_kind="swap")])
    rows = []
    for key, g in eff.groupby([*by, "alt_kind"], sort=True):
        key = key if isinstance(key, tuple) else (key,)
        fl = pairs[
            (pairs.role == "floor")
            & pairs.obs_idx.isin(g.obs_idx.unique())
            & (pairs.model == g.model.iloc[0])
        ]
        row = dict(zip([*by, "alt_kind"], key, strict=True))
        row.update(
            n_pairs=len(g),
            n_obs=g.obs_idx.nunique(),
            effect_mm=1000 * g.eff_m.mean(),
            floor_mm=1000 * fl.eff_m.mean(),
            ratio=g.eff_m.mean() / fl.eff_m.mean() if fl.eff_m.mean() > 0 else np.nan,
            effect_median_mm=1000 * g.eff_m.median(),
            frac_above=float((g.eff_m > g.eff_m_floor).mean()),
            orig_disp_mm=1000 * g.orig_disp_m.mean(),
            grip_effect=g.eff_grip.mean(),
            grip_floor=fl.eff_grip.mean(),
            rot_effect_deg=np.degrees(g.eff_rot.mean()),
            rot_floor_deg=np.degrees(fl.eff_rot.mean()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def target_pos(objects: dict[str, list[float]], name: Any) -> np.ndarray | None:
    if not isinstance(name, str) or "," in name or name not in objects:
        return None
    return np.asarray(objects[name], dtype=np.float64)


def align_table(disp: pd.DataFrame, probe: pd.DataFrame, meta_rows: list[dict]) -> pd.DataFrame:
    objs = {int(r["obs_idx"]): r["objects"] for r in meta_rows}
    sw = probe[probe["kind"] == "swap"][["model", "obs_idx", "instr_id", "orig_target", "new_target"]]
    d = disp.merge(sw, on=["model", "obs_idx", "instr_id"], how="left")
    out = []
    for (m, i), g in d.groupby(["model", "obs_idx"], sort=False):
        by = {r.instr_id: r for r in g.itertuples(index=False)}
        if "orig_s0" not in by:
            continue

        def vec(r: Any) -> np.ndarray:
            return np.array([r.dx, r.dy, r.dz])

        d0 = vec(by["orig_s0"])
        nulls = [vec(by[f"orig_s{s}"]) for s in range(1, lp.N_SEEDS) if f"orig_s{s}" in by]
        p0 = np.asarray(by["orig_s0"].p0)
        for r in g.itertuples(index=False):
            if r.kind != "swap":
                continue
            po, pn = target_pos(objs[int(i)], r.orig_target), target_pos(objs[int(i)], r.new_target)
            if po is None or pn is None:
                continue
            u_o, u_n = po - p0, pn - p0
            da = vec(r)
            rec = {"model": m, "obs_idx": i, "suite": r.suite, "task_id": r.task_id, "instr_id": r.instr_id,
                   "swap_type": r.swap_type, "action_space": r.action_space,
                   "target_sep_m": float(np.linalg.norm(pn - po)),
                   "tilt": tilt(da, d0, u_o, u_n), "cos_shift": cos(da - d0, u_n - u_o),
                   "cos_shift_xy": cos((da - d0)[:2], (u_n - u_o)[:2]),
                   "cos_alt_new": cos(da, u_n), "cos_alt_orig": cos(da, u_o),
                   "cos_orig_new": cos(d0, u_n), "cos_orig_orig": cos(d0, u_o)}  # fmt: skip
            nt = [tilt(dn, d0, u_o, u_n) for dn in nulls]
            ns = [cos(dn - d0, u_n - u_o) for dn in nulls]
            nxy = [cos((dn - d0)[:2], (u_n - u_o)[:2]) for dn in nulls]
            rec.update(tilt_null=np.nanmean(nt) if nt else np.nan, cos_shift_null=np.nanmean(ns) if ns else np.nan,
                       cos_shift_xy_null=np.nanmean(nxy) if nxy else np.nan)  # fmt: skip
            out.append(rec)
    return pd.DataFrame(out)


def summarize_align(al: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if al.empty:
        return al
    al = pd.concat([al, al.assign(swap_type="all")])
    agg = al.groupby([*by, "swap_type"]).agg(
        n=("tilt", "size"),
        tilt=("tilt", "mean"),
        tilt_null=("tilt_null", "mean"),
        frac_tilt_pos=("tilt", lambda x: float((x > 0).mean())),
        cos_shift=("cos_shift", "mean"),
        cos_shift_null=("cos_shift_null", "mean"),
        cos_shift_xy=("cos_shift_xy", "mean"),
        cos_shift_xy_null=("cos_shift_xy_null", "mean"),
        target_sep_mm=("target_sep_m", lambda x: 1000 * x.mean()),
    )
    return agg.reset_index()


def action_pairs(probe: pd.DataFrame, n_act: int, gt: np.ndarray | None) -> pd.DataFrame:
    out = []
    p = probe[probe["kind"] != "repeat"]
    for (m, i), g in p.groupby(["model", "obs_idx"], sort=False):
        acts = {r.instr_id: np.asarray(r.actions).reshape(r.n_actions, r.action_dim) for r in g.itertuples()}
        info = {r.instr_id: r for r in g.itertuples()}
        if "orig_s0" not in acts:
            continue
        r0 = info["orig_s0"]
        base = {"model": m, "obs_idx": i, "set": r0.set, "suite": r0.suite, "task_id": r0.task_id,
                "action_space": r0.action_space}  # fmt: skip
        seeds = [f"orig_s{s}" for s in range(lp.N_SEEDS) if f"orig_s{s}" in acts]
        for x in range(len(seeds)):
            for y in range(x + 1, len(seeds)):
                l2, fl = action_diff(acts[seeds[y]], acts[seeds[x]], n_act)
                out.append({**base, "role": "floor", "alt_kind": "floor", "act_l2": l2, "grip_flip": fl})
        for iid, r in info.items():
            if r.kind in ("blank", "swap"):
                l2, fl = action_diff(acts[iid], acts["orig_s0"], n_act)
                out.append({**base, "role": "effect", "alt_kind": alt_kind(r.kind, getattr(r, "swap_type", None)),
                            "act_l2": l2, "grip_flip": fl})  # fmt: skip
        if gt is not None and r0.action_space == ACTION_SPACE_OSC:
            k = min(n_act, len(acts["orig_s0"]), gt.shape[1])
            g_l2 = float(np.mean(np.linalg.norm(acts["orig_s0"][:k, :-1] - gt[int(i), :k, :-1], axis=1)))
            out.append({**base, "role": "gt", "alt_kind": "gt", "act_l2": g_l2, "grip_flip": np.nan})
    return pd.DataFrame(out)


def summarize_actions(ap: pd.DataFrame) -> pd.DataFrame:
    if ap.empty:
        return ap
    ap = pd.concat([ap, ap[ap.alt_kind.str.startswith("swap_")].assign(alt_kind="swap")])
    floor = ap[ap.role == "floor"].groupby(["model", "set"])[["act_l2", "grip_flip"]].mean()
    gt = ap[ap.role == "gt"].groupby(["model", "set"])["act_l2"].mean()
    rows = []
    for (m, s, k), g in ap[ap.role == "effect"].groupby(["model", "set", "alt_kind"]):
        f = floor.loc[(m, s)] if (m, s) in floor.index else None
        rows.append({"model": m, "set": s, "alt_kind": k, "action_space": g.action_space.iloc[0], "n": len(g),
                     "act_l2": g.act_l2.mean(), "act_l2_floor": f.act_l2 if f is not None else np.nan,
                     "ratio": g.act_l2.mean() / f.act_l2 if f is not None and f.act_l2 > 0 else np.nan,
                     "grip_flip": g.grip_flip.mean(), "grip_flip_floor": f.grip_flip if f is not None else np.nan,
                     "gt_l2": gt.get((m, s), np.nan)})  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report


def fmt_table(df: pd.DataFrame, cols: list[str], floatfmt: str = "{:.3g}") -> str:
    if df.empty:
        return "(none)\n"
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    lines = []
    for r in df[cols].itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def write_summary(
    out: Path, disp_m: pd.DataFrame, align_m: pd.DataFrame, act_m: pd.DataFrame, meta: dict
) -> None:
    t = meta["t_eval"]
    s = [f"# Language probe (T018) — {meta['run']}\n",
         f"Open-loop, no success rate. Same observation + same noise, only the instruction differs. EEF displacement "
         f"after {t} s of the chunk executed from the observation's sim state (set A, {meta['n_obs_A']} obs). "
         "effect = ‖Δ(alt) − Δ(orig_s0)‖, floor = ‖Δ(orig_si) − Δ(orig_sj)‖ (noise only); ratio = effect / floor; "
         "frac_above = share of (obs, alt) above that observation's floor.\n",
         f"Models: {', '.join(meta['models'])}. Self-check (restore vs fresh replay, OSC): max dev "
         f"{meta.get('self_check_max_dev_m')} m.\n",
         "## 1. Displacement effect (set A, all suites)\n"]  # fmt: skip
    all_rows = disp_m[disp_m.scope == "all"] if not disp_m.empty else disp_m
    s.append(fmt_table(all_rows, ["model", "alt_kind", "n_pairs", "effect_mm", "floor_mm", "ratio", "frac_above",
                                  "orig_disp_mm", "grip_effect", "grip_floor", "rot_effect_deg", "rot_floor_deg"]))  # fmt: skip
    s.append("\n### per suite (alt_kind = swap and blank)\n")
    suite_rows = (
        disp_m[(disp_m.scope == "suite") & disp_m.alt_kind.isin(["swap", "blank"])]
        if not disp_m.empty
        else disp_m
    )
    s.append(
        fmt_table(
            suite_rows,
            ["model", "suite", "alt_kind", "n_pairs", "effect_mm", "floor_mm", "ratio", "frac_above"],
        )
    )
    s.append("\n## 2. Goal alignment (set A swaps with named targets; > null = towards the new target)\n")
    s.append(fmt_table(align_m[align_m.scope == "all"] if not align_m.empty else align_m,
                       ["model", "swap_type", "n", "tilt", "tilt_null", "frac_tilt_pos", "cos_shift", "cos_shift_null",
                        "cos_shift_xy", "cos_shift_xy_null", "target_sep_mm"]))  # fmt: skip
    s.append(f"\n## 3. Action-space difference (first {meta['n_act']} steps; within an action space only)\n")
    s.append(fmt_table(act_m, ["model", "set", "alt_kind", "action_space", "n", "act_l2", "act_l2_floor", "ratio",
                               "grip_flip", "grip_flip_floor", "gt_l2"]))  # fmt: skip
    s.append("\nact_l2 in each model's own env-scale action units (OSC: ±1-scaled delta pose; joint_velocity: "
             "normalized joint velocity): compare ratios, not raw values, across action spaces. gt_l2 (set B) = "
             "orig_s0 vs the dataset's own next actions.\n")  # fmt: skip
    (out / "summary.md").write_text("\n".join(s))


def figure(out: Path, disp: pd.DataFrame, meta_rows: list[dict], n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = sorted(disp.obs_idx.unique())[:: max(1, len(disp.obs_idx.unique()) // n)][:n]
    models = sorted(disp.model.unique())
    fig, axes = plt.subplots(
        len(models), len(obs), figsize=(3.2 * len(obs), 3.2 * len(models)), squeeze=False
    )
    colors = {"orig": "#1f77b4", "blank": "#7f7f7f", "swap": "#d62728"}
    objs = {int(r["obs_idx"]): r for r in meta_rows}
    for a, m in enumerate(models):
        for b, i in enumerate(obs):
            ax = axes[a][b]
            g = disp[(disp.model == m) & (disp.obs_idx == i)]
            if g.empty:
                ax.axis("off")
                continue
            p0 = np.asarray(g.p0.iloc[0])
            pts = [p0[:2]]
            for name, pos in objs[int(i)]["objects"].items():
                if not name.endswith(("region", "side")) and "floor" not in name and "table" not in name:
                    ax.plot(pos[0], pos[1], "k.", ms=3)
                    ax.annotate(name.rsplit("_", 1)[0][:12], (pos[0], pos[1]), fontsize=5)
                    pts.append(np.asarray(pos[:2]))
            for r in g.itertuples():
                end = (p0[0] + 4 * r.dx, p0[1] + 4 * r.dy)
                pts.append(np.asarray(end))
                ax.annotate("", end, (p0[0], p0[1]),
                            arrowprops={"arrowstyle": "->", "color": colors[r.kind], "lw": 1})  # fmt: skip
            lo, hi = np.min(pts, axis=0) - 0.05, np.max(pts, axis=0) + 0.05
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_title(f"{m}\n{objs[int(i)]['obs_id']}", fontsize=6)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=5)
    fig.suptitle("EEF xy displacement ×4 (blue orig seeds, grey blank, red swaps), world frame", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "arrows.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="probe.py --out dir (one subdir per model)")
    ap.add_argument("--obs-dir", default=f"{lp.DEFAULT_ROOT}/analysis/langprobe/obs")
    ap.add_argument("--models", default=None, help="default: every subdir with probe_*.parquet")
    ap.add_argument("--out", default=None, help="default <run>/measure")
    ap.add_argument("--t-eval", type=float, default=lp.T_EVAL_S)
    ap.add_argument("--n-act", type=int, default=N_ACT_DEFAULT)
    ap.add_argument("--workers", type=int, default=min(40, os.cpu_count() or 1))
    ap.add_argument(
        "--self-check", type=int, default=1, help="obs per task compared with a fresh replay (0: off)"
    )
    ap.add_argument("--figure", type=int, default=6, help="observations in arrows.png (0: none)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--backend", choices=("libero", "mock"), default="libero")
    args = ap.parse_args(argv)
    run, obs_dir = Path(args.run), Path(args.obs_dir)
    out = Path(args.out) if args.out else run / "measure"
    out.mkdir(parents=True, exist_ok=True)
    models = args.models.split(",") if args.models else sorted(
        p.parent.name for p in run.glob("*/probe_*.parquet"))  # fmt: skip
    models = list(dict.fromkeys(models))
    _, meta_rows = lp.load_set(obs_dir, "A", verify=not args.no_verify)
    meta: dict[str, Any] = {"run": str(run), "models": models, "t_eval": args.t_eval, "n_act": args.n_act,
                            "started_at": datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip

    # set A: execution + displacement metrics
    probe_a = pd.concat([d for m in models if (d := load_probe(run, m, "A")) is not None], ignore_index=True) \
        if any((run / m / "probe_A.parquet").exists() for m in models) else pd.DataFrame()  # fmt: skip
    disp_m = align_m = pd.DataFrame()
    if not probe_a.empty:
        ex, checks = execute(run, obs_dir, models, args)
        meta["self_check_max_dev_m"] = max(checks) if checks else None
        meta["self_check_n"] = len(checks)
        disp = displacement_table(probe_a, ex, args.t_eval)
        meta["n_obs_A"] = int(disp.obs_idx.nunique())
        meta["nan_displacements"] = int(disp.dx.isna().sum())
        disp.to_parquet(out / "displacement_A.parquet", index=False)
        pairs = pair_table(disp)
        pairs.to_parquet(out / "pairs_A.parquet", index=False)
        disp_m = pd.concat([summarize_pairs(pairs, ["model"]).assign(scope="all"),
                            summarize_pairs(pairs, ["model", "suite"]).assign(scope="suite"),
                            summarize_pairs(pairs, ["model", "suite", "task_id"]).assign(scope="task")],
                           ignore_index=True)  # fmt: skip
        al = align_table(disp, probe_a, meta_rows)
        al.to_parquet(out / "align_A.parquet", index=False)
        align_m = pd.concat([summarize_align(al, ["model"]).assign(scope="all"),
                             summarize_align(al, ["model", "suite"]).assign(scope="suite")], ignore_index=True) \
            if not al.empty else al  # fmt: skip
        if args.figure:
            figure(out, disp, meta_rows, args.figure)
    meta.setdefault("n_obs_A", 0)
    disp_m.to_csv(out / "metrics_displacement.csv", index=False)
    align_m.to_csv(out / "metrics_align.csv", index=False)

    # action-space metrics: sets A and B
    act_parts = []
    for s in ("A", "B"):
        dfs = [d for m in models if (d := load_probe(run, m, s)) is not None]
        if not dfs:
            continue
        gt = lp.load_set(obs_dir, s, verify=False)[0].get("gt_action") if s == "B" else None
        ap_s = action_pairs(pd.concat(dfs, ignore_index=True), args.n_act, gt)
        ap_s.to_parquet(out / f"actions_{s}.parquet", index=False)
        act_parts.append(ap_s)
    act_m = summarize_actions(pd.concat(act_parts, ignore_index=True)) if act_parts else pd.DataFrame()
    act_m.to_csv(out / "metrics_action.csv", index=False)
    meta["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    (out / "measure_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    write_summary(out, disp_m, align_m, act_m, meta)
    print(f"[measure] wrote {out}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if os.environ["MUJOCO_GL"] == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    sys.exit(main())
