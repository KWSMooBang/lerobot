"""`libero_plus` worker: runs units (one original task × its selected LIBERO-Plus variants × inits) against the
policy server. Runs in the LIBERO-Plus venv (bench/libero_plus/adapter.py → worker_cmd); with the mock backend
in any venv (MockPlusEnv, no LIBERO).

Per episode: PlusEnv.reset (init state + 10 settle steps) → policy chunks over the wire until success or
max_steps (early stop, like ours C0–C3). Wire obs = robosuite_obs_to_wire (180° flip, eef pos + axis-angle +
gripper qpos). Writes episodes.jsonl (standard fields + T001-style outcome fields + plus_*), steps/*.npz,
videos/*.mp4 (10 fps, agentview | wrist 512×256 with a header, like ours).
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from bench.core.client import PolicyClient, robosuite_obs_to_wire  # noqa: E402
from bench.core.records import append_jsonl, perturbation, read_jsonl, standard_fields  # noqa: E402
from bench.core.worker import run_worker  # noqa: E402
from bench.libero_plus.registry import ORIGINAL_TASKS, PlusVariant, language_from_name  # noqa: E402

VIDEO_FPS, CONTROL_HZ, FRAME_SIDE = 10, 20, 256
WHITE, GREEN, RED, GREY = (255, 255, 255), (80, 255, 80), (255, 90, 90), (200, 200, 200)


def episode_seed(base: int, task_id: int, init_idx: int, repeat_idx: int = 0) -> int:
    """= common.episode_seed: the same (task, init) seed as ours, so the clean baseline pairs with C0."""
    return base * 1_000_000 + repeat_idx * 10_000 + task_id * 100 + init_idx


# ---------------------------------------------------------------- envs


class MockPlusEnv:
    """CPU stand-in with PlusEnv's surface (raw robosuite-style obs). Success when the eef reaches a target
    that depends on the task; perturbation index shifts it a little so variants differ."""

    def __init__(self, v: PlusVariant, hw: int = 32, control_mode: str = "relative"):
        self.v, self.hw = v, hw
        self.n_init = 1 if v.detail in ("add", "moved") else 50
        self.goal = np.array([0.1 * (v.task_id + 1), 0.0, 0.0]) + 0.001 * max(v.index, 0)
        self.pos = np.zeros(3)

    def _obs(self) -> dict[str, Any]:
        img: np.ndarray = np.empty((self.hw, self.hw, 3), np.uint8)
        img[:] = np.clip((self.pos + 0.5) * 255, 0, 255).astype(np.uint8)
        return {
            "agentview_image": img,
            "robot0_eye_in_hand_image": img[::-1].copy(),
            "robot0_eef_pos": self.pos.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
        }

    def reset(self, init_idx: int, seed: int) -> dict[str, Any]:
        np.random.seed(seed % 2**32)
        self.pos = self.goal + np.random.uniform(-0.2, 0.2, 3) + 0.001 * (init_idx % self.n_init)
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], bool]:
        self.pos = self.pos + 0.02 * np.clip(np.asarray(action, dtype=np.float64)[:3], -1, 1)
        return self._obs(), bool(np.linalg.norm(self.pos - self.goal) < 0.03)

    def close(self) -> None:
        pass


def instruction_for(v: PlusVariant, backend: str) -> str:
    """The fork's registry instruction (task.language): `_language_` BDDL sentence, or filename-derived."""
    if backend == "mock":
        return language_from_name(v.name)
    from bench.libero_plus import lp_env

    return lp_env.instruction(v)


def make_env(v: PlusVariant, unit: dict[str, Any]) -> Any:
    if unit["backend"] == "mock":
        return MockPlusEnv(v, unit["env_hw"], unit["control_mode"])
    from bench.libero_plus import lp_env

    return lp_env.PlusEnv(v, unit["env_hw"], unit["control_mode"])


# ---------------------------------------------------------------- video


def overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    import cv2

    h = 14
    roi = frame[: 6 + h * len(lines)]
    roi[:] = (roi * 0.45).astype(np.uint8)
    for i, (txt, color) in enumerate(lines):
        cv2.putText(frame, txt, (5, 13 + i * h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame


def compose_frame(wire: dict[str, np.ndarray], lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    import cv2

    views = [
        cv2.resize(wire[k], (FRAME_SIDE, FRAME_SIDE), interpolation=cv2.INTER_AREA)
        for k in ("image", "image2")
    ]
    return overlay(np.ascontiguousarray(np.concatenate(views, axis=1)), lines)


def header(v: PlusVariant, instr: str, init_idx: int, t: int, t_ok: int | None, tag: str) -> list:
    p = f"{v.dimension}/{v.detail}" + (f" L{v.level}" if v.level is not None else "")
    lines = [(f"{tag}  plus {p} {v.variant_id}  init {init_idx:02d}  t={t}".strip(), WHITE)]
    lines += [(f"> {w}" if i == 0 else f"  {w}", GREY) for i, w in enumerate(textwrap.wrap(instr, 62)[:2])]
    lines.append((f"success: yes@{t_ok}", GREEN) if t_ok is not None else ("success: no", RED))
    return lines


def write_mp4(path: Path, frames: list[np.ndarray]) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}{path.suffix}")
    imageio.mimwrite(tmp, frames, fps=VIDEO_FPS, codec="libx264", quality=7, macro_block_size=8)
    os.replace(tmp, path)


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}{path.suffix}")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------- episodes


def run_episode(
    client: PolicyClient,
    env: Any,
    v: PlusVariant,
    instr: str,
    init_idx: int,
    seed: int,
    unit: dict[str, Any],
    record_video: bool,
) -> dict[str, Any]:
    eid = f"{unit['unit_id']}/{v.variant_id}/i{init_idx}"
    client.reset(eid, seed, instr)
    tic = time.perf_counter()
    obs = env.reset(init_idx, seed)
    t_env = time.perf_counter() - tic
    wire = robosuite_obs_to_wire(obs)
    frames = [compose_frame(wire, header(v, instr, init_idx, 0, None, unit["model"]))] if record_video else []
    log: dict[str, list] = {k: [] for k in ("eef_pos", "eef_quat", "gripper_qpos", "action", "success")}
    queue: deque = deque()
    t_policy, t, t_ok = 0.0, 0, None
    while t < unit["max_steps"]:
        if not queue:
            tic = time.perf_counter()
            queue.extend(client.infer(eid, wire, instr, step=unit["requires_step"]))
            t_policy += time.perf_counter() - tic
        a = queue.popleft()
        tic = time.perf_counter()
        obs, ok = env.step(a)
        t_env += time.perf_counter() - tic
        t += 1
        wire = robosuite_obs_to_wire(obs)
        for k, key in (("eef_pos", "robot0_eef_pos"), ("eef_quat", "robot0_eef_quat")):
            log[k].append(obs[key])
        log["gripper_qpos"].append(obs["robot0_gripper_qpos"])
        log["action"].append(a)
        log["success"].append(ok)
        if record_video and t % (CONTROL_HZ // VIDEO_FPS) == 0:
            frames.append(compose_frame(wire, header(v, instr, init_idx, t, t_ok, unit["model"])))
        if ok:
            t_ok = t
            break
    client.end(eid)
    if record_video:  # hold the last frame ~1 s with the verdict
        frames += [compose_frame(wire, header(v, instr, init_idx, t, t_ok, unit["model"]))] * VIDEO_FPS
    arrays = {k: np.asarray(x, dtype=bool if k == "success" else np.float32) for k, x in log.items()}
    return {
        "success": t_ok is not None,
        "t_success": t_ok,
        "n_steps": t,
        "time_policy_s": round(t_policy, 2),
        "time_env_s": round(t_env, 2),
        "_arrays": arrays,
        "_frames": frames,
    }


def run_unit(unit: dict[str, Any], client: PolicyClient) -> int:
    out = Path(unit["job_dir"])
    ep_path = out / "episodes.jsonl"
    done = {(r["task_id"], r["variant_id"], r["init_idx"]) for r in read_jsonl(ep_path)}
    task_id = unit["task_id"]
    if unit["backend"] != "mock":
        from bench.libero_plus import lp_env

        lp_env.bootstrap()
    original = language_from_name(ORIGINAL_TASKS[unit["suite"]][task_id])
    cap = unit["video_cap"]
    written = 0
    for vi, vd in enumerate(unit["variants"]):
        v = PlusVariant(**vd)
        todo = [i for i in range(unit["episodes"]) if (task_id, v.variant_id, i) not in done]
        if not todo:
            continue
        registry_instr = instruction_for(v, unit["backend"])
        given = (
            registry_instr
            if v.dimension == "language" or unit["instruction_mode"] == "registry"
            else original
        )
        env = make_env(v, unit)
        try:
            for init_idx in todo:
                seed = episode_seed(unit["seed"], task_id, init_idx)
                rec_video = cap is None or vi * unit["episodes"] + init_idx < cap  # first N of the unit
                t0 = time.time()
                res = run_episode(client, env, v, given, init_idx, seed, unit, rec_video)
                rel = f"{v.suite}/task{task_id:02d}/{v.dimension}_{v.variant_id}_init{init_idx:02d}"
                steps_file = f"steps/{rel}.npz"
                save_npz(out / steps_file, res.pop("_arrays"))
                frames = res.pop("_frames")
                video = None
                if rec_video:
                    video = f"videos/{rel}.mp4"
                    write_mp4(out / video, frames)
                ok = bool(res["success"])
                rec = {
                    "model": unit["model"],
                    "policy": unit["policy"],
                    "ckpt": unit["ckpt"],
                    "suite": v.suite,
                    "task_id": task_id,
                    "init_idx": init_idx,
                    "init_state_idx": init_idx % env.n_init,
                    "repeat_idx": 0,
                    "condition": f"plus-{v.dimension}",
                    "variant": v.variant_id,
                    "instruction": given,
                    "original_instruction": original,
                    "success_orig": ok if unit["dual_goal"] else None,
                    "success_new": None,
                    "t_orig": res["t_success"] if unit["dual_goal"] else None,
                    "t_new": None,
                    **res,
                    "max_steps": unit["max_steps"],
                    "early_stop": True,
                    "steps_file": steps_file,
                    "wall_time_s": round(time.time() - t0, 2),
                    "plus_name": v.name,
                    "plus_index": v.index,
                    "plus_params": v.params,
                    "plus_instruction_registry": registry_instr,
                    "instruction_mode": unit["instruction_mode"],
                    "policy_server_ms": round(client.server_ms, 1),
                }
                client.server_ms = 0.0
                rec.update(
                    standard_fields(
                        benchmark="libero_plus",
                        bench_version=unit["bench_version"],
                        model=unit["model"],
                        ckpt=unit["ckpt"],
                        suite=v.suite,
                        task_id=task_id,
                        variant_id=v.variant_id,
                        init_idx=init_idx,
                        pert=perturbation(v.dimension, v.level, v.detail),
                        instruction_original=original,
                        instruction_given=given,
                        success=ok,
                        success_orig=ok if unit["dual_goal"] else None,
                        success_new=None,
                        video=video,
                        seed=seed,
                    )
                )
                append_jsonl(ep_path, rec)
                print(
                    f"[ep] plus {v.suite} t{task_id} {v.dimension}/{v.detail} L{v.level} {v.variant_id} "
                    f"init {init_idx} success={ok}@{res['t_success']} steps={res['n_steps']} "
                    f"{rec['wall_time_s']:.0f}s",
                    flush=True,
                )
                written += 1
        finally:
            env.close()
    return written


if __name__ == "__main__":
    run_worker(run_unit)
