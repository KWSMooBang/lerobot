"""LIBERO-PRO worker (PRO venv): runs units = one (suite, axis, level, task) × its inits through the policy server.

Started by sc-bench (bench/core/runner.py) with SC_POLICY_URL, CUDA_VISIBLE_DEVICES, MUJOCO_EGL_DEVICE_ID.
Observations go over the wire in the training-data convention (`robosuite_obs_to_wire`: upright images,
eef pos + axis-angle + gripper qpos). One episodes.jsonl line per episode (standard fields + T001-style
outcome fields); episodes already in the file are skipped (resume).
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from common import episode_seed  # noqa: E402

from bench.core.client import PolicyClient, robosuite_obs_to_wire  # noqa: E402
from bench.core.records import append_jsonl, perturbation, read_jsonl, standard_fields  # noqa: E402
from bench.core.worker import run_worker  # noqa: E402
from bench.libero_pro import catalog, sim  # noqa: E402

AGENT, WRIST = "agentview_image", "robot0_eye_in_hand_image"


def episode_key(r: dict[str, Any]) -> tuple:
    return (r["task_id"], r["axis"], r["level"], r["init_idx"], r.get("repeat_idx", 0))


def video_rel(suite: str, task_id: int, axis: str, level: str, init_idx: int, rep: int) -> str:
    return f"{suite}/task{task_id:02d}/{axis}_{level}" + (f"_r{rep}" if rep else "") + f"_init{init_idx:02d}"


def run_episode(
    client: PolicyClient,
    env: Any,
    entry: dict[str, Any],
    unit: dict[str, Any],
    init_idx: int,
    seed: int,
    video: bool,
) -> dict[str, Any]:
    dual, max_steps = entry["dual"], int(unit["max_steps"])
    instr = entry["instruction_given"]
    eid = f"{unit['unit_id']}/i{init_idx}"
    label = f"{unit['model']}  PRO {entry['axis']}/{entry['level']}  init {init_idx:02d}"
    client.reset(eid, seed, instr)
    obs = env.reset(init_idx, seed)
    t_orig = t_new = None
    frames: list[np.ndarray] = []
    log: dict[str, list] = {k: [] for k in ("eef_pos", "gripper_qpos", "action", "goal_pro", "goal_orig")}
    grasped: list[str] = []
    contacts: list[str] = []

    def frame(t: int) -> None:
        if video:
            hdr = sim.header_lines(instr, label, t, t_orig, t_new, dual)
            frames.append(sim.compose_frame(obs[AGENT], obs[WRIST], hdr))

    frame(0)
    queue: deque = deque()
    t_policy = t_env = 0.0
    t = 0
    step_mode = bool(unit.get("requires_step"))
    while t < max_steps:
        if not queue:
            tic = time.perf_counter()
            queue.extend(client.infer(eid, robosuite_obs_to_wire(obs, AGENT, WRIST), instr, step=step_mode))
            t_policy += time.perf_counter() - tic
        a = queue.popleft()
        tic = time.perf_counter()
        obs, pro, orig = env.step(a)
        t_env += time.perf_counter() - tic
        t += 1
        if pro and t_new is None:
            t_new = t
        if orig and t_orig is None:
            t_orig = t
        touched, grasp = env.contacts()
        contacts.append("|".join(touched))
        grasped.append("|".join(grasp))
        log["eef_pos"].append(obs["robot0_eef_pos"])
        log["gripper_qpos"].append(obs["robot0_gripper_qpos"])
        log["action"].append(a)
        log["goal_pro"].append(pro)
        log["goal_orig"].append(bool(orig))
        if t % (sim.CONTROL_HZ // sim.VIDEO_FPS) == 0:
            frame(t)
        if pro and not dual:
            break
    client.end(eid)
    if video:  # hold the last frame ~1 s with the verdict
        frame(t)
        frames.extend([frames[-1]] * sim.VIDEO_FPS)
    first_grasp = first_grasp_t = None
    for i, (g, a) in enumerate(zip(grasped, log["action"], strict=True)):
        if g and a[6] > 0:
            first_grasp, first_grasp_t = g.split("|")[0], i + 1
            break
    arrays = {k: np.asarray(v, dtype=bool if k.startswith("goal") else np.float32) for k, v in log.items()}
    arrays["contacts"] = np.asarray(contacts, dtype=str)
    arrays["grasped"] = np.asarray(grasped, dtype=str)
    missing = list(getattr(env, "orig_missing", []))
    return {
        "frames": frames,
        "arrays": arrays,
        "success_pro": t_new is not None,
        "success_orig": (None if missing else t_orig is not None) if dual else t_new is not None,
        "success_new": (t_new is not None) if dual else None,
        "t_orig": (t_orig if dual else t_new),
        "t_new": t_new if dual else None,
        "orig_goal_missing_objects": missing,
        "first_grasp_object": first_grasp,
        "first_grasp_t": first_grasp_t,
        "first_contact_object": next((c.split("|")[0] for c in contacts if c), None),
        "n_steps": t,
        "time_policy_s": round(t_policy, 2),
        "time_env_s": round(t_env, 2),
    }


_ENV: dict[str, Any] = {}


def env_for(entry: dict[str, Any], unit: dict[str, Any]) -> Any:
    key = f"{entry['bddl']}|{unit['backend']}|{unit['env_hw']}"
    if key not in _ENV:
        for e in _ENV.values():
            e.close()
        _ENV.clear()
        _ENV[key] = sim.make_env(entry, unit["backend"], int(unit["env_hw"]))
    return _ENV[key]


def run_unit(unit: dict[str, Any], client: PolicyClient) -> int:
    out = Path(unit["job_dir"])
    ep_path = out / "episodes.jsonl"
    rep = int(unit.get("repeat_idx", 0))
    done = {episode_key(r) for r in read_jsonl(ep_path)}
    root = Path(unit["pro_root"])
    entry = catalog.task_entry(
        root,
        unit["suite"],
        unit["axis"],
        unit["level"],
        unit["task_id"],
        unit["instruction_source"],
        unit["init_source"],
    )
    env = None
    cap = unit.get("video_cap")
    written = 0
    for init_idx in range(int(unit["episodes"])):
        key = (unit["task_id"], unit["axis"], unit["level"], init_idx, rep)
        if key in done:
            continue
        if env is None:
            env = env_for(entry, unit)
            if unit["episodes"] > len(env.init_states):
                raise ValueError(f"{entry['init']}: only {len(env.init_states)} init states")
        seed = episode_seed(int(unit["seed"]), rep, unit["task_id"], init_idx)
        video = cap is None or init_idx < cap
        t0 = time.time()
        res = run_episode(client, env, entry, unit, init_idx, seed, video)
        rel = video_rel(unit["suite"], unit["task_id"], unit["axis"], unit["level"], init_idx, rep)
        steps_file = out / "steps" / f"{rel}.npz"
        steps_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(steps_file, **res.pop("arrays"))
        frames = res.pop("frames")
        video_file = None
        if video:
            video_file = f"videos/{rel}.mp4"
            sim.write_video(out / video_file, frames)
        rec = {
            "model": unit["model"],
            "ckpt": unit["ckpt"],
            "suite": unit["suite"],
            "task_id": unit["task_id"],
            "init_idx": init_idx,
            "repeat_idx": rep,
            "axis": unit["axis"],
            "level": unit["level"],
            "condition": f"pro:{unit['axis']}",
            "variant": unit["level"],
            "instruction": entry["instruction_given"],
            "original_instruction": entry["instruction_original"],
            "pro_instruction_original": entry["pro_instruction_original"],
            "instruction_bddl": entry["instruction_bddl"],
            "instruction_source": unit["instruction_source"],
            "init_source": entry["init_source"],
            "init_file": entry["init"],
            "pro_task": entry["task_name"],
            "pro_dir": entry["dir"],
            **res,
            "dual": entry["dual"],
            "early_stop": not entry["dual"],
            "max_steps": int(unit["max_steps"]),
            "steps_file": str(steps_file.relative_to(out)),
            "wall_time_s": round(time.time() - t0, 2),
        }
        rec.update(
            standard_fields(
                benchmark="libero_pro",
                bench_version=unit["bench_version"],
                model=unit["model"],
                ckpt=unit["ckpt"],
                suite=unit["suite"],
                task_id=unit["task_id"],
                variant_id=unit["level"],
                init_idx=init_idx,
                pert=perturbation(catalog.DIMENSION[unit["axis"]], unit["level"], f"pro:{unit['axis']}"),
                instruction_original=entry["instruction_original"],
                instruction_given=entry["instruction_given"],
                success=res["success_pro"],
                success_orig=res["success_orig"],
                success_new=res["success_new"],
                video=video_file,
                seed=seed,
            )
        )
        rec["policy_server_ms"] = round(client.server_ms, 1)
        client.server_ms = 0.0
        append_jsonl(ep_path, rec)
        print(
            f"[ep] {unit['suite']} t{unit['task_id']} {unit['axis']}/{unit['level']} init {init_idx:02d} "
            f"pro={rec['success']}@{res['t_new'] if entry['dual'] else res['t_orig']} orig={rec['success_orig']} "
            f"grasp={res['first_grasp_object']} steps={res['n_steps']} {rec['wall_time_s']}s",
            flush=True,
        )
        written += 1
    return written


if __name__ == "__main__":
    run_worker(run_unit)
