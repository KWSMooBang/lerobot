"""`ours` worker: runs units (one task of a C0–C6 job) against the policy server.

Started by the coordinator (bench/core/runner.py) with SC_POLICY_URL, CUDA_VISIBLE_DEVICES and
MUJOCO_EGL_DEVICE_ID set. Env side is the T001 code unchanged (EpisodeStepper: early stop for C0–C3,
original + new goal every step for C4–C6, contact probe, step log, video); only the policy call goes over
the wire. Observations are converted with lerobot's own LiberoProcessorStep (180° flip + quat → axis-angle
in torch), so the server sees exactly what run_eval.py's in-process policy sees.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(SHORTCUT), str(SHORTCUT / "parallel")) if p not in sys.path]

import torch  # noqa: E402
from common import Variant, episode_seed, video_relpath  # noqa: E402
from envproc import _make_env  # noqa: E402
from run_eval import (  # noqa: E402
    ContactProbe,
    EpisodeStepper,
    GoalChecker,
    episode_line,
    episode_record,
    save_episode,
)

from bench.core.client import PolicyClient, upright  # noqa: E402
from bench.core.records import (  # noqa: E402
    OURS_PERTURBATION,
    append_jsonl,
    perturbation,
    read_jsonl,
    standard_fields,
)
from bench.core.worker import run_worker  # noqa: E402
from lerobot.processor.env_processor import LiberoProcessorStep  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

_LPS = LiberoProcessorStep()
_ENVS: dict[tuple, Any] = {}
_SUITES: dict[str, Any] = {}


def wire_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """LiberoEnv observation → wire obs, bit-identical to preprocess_observation + LiberoProcessorStep."""
    rs = obs["robot_state"]

    def t(x: Any) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x))[None]

    nested = {
        "eef": {"pos": t(rs["eef"]["pos"]), "quat": t(rs["eef"]["quat"])},
        "gripper": {"qpos": t(rs["gripper"]["qpos"])},
    }
    state = _LPS._process_observation({"observation.robot_state": nested})[OBS_STATE][0].numpy()
    return {
        "image": upright(obs["pixels"]["image"]),
        "image2": upright(obs["pixels"]["image2"]),
        "state": state,
    }


def env_for(unit: dict[str, Any]) -> Any:
    key = (unit["suite"], unit["task_id"], unit["max_steps"])
    if key not in _ENVS:
        for env in _ENVS.values():
            env.close()
        _ENVS.clear()
        spec = {k: unit[k] for k in ("suite", "task_id", "max_steps", "env_hw", "control_mode")}
        _ENVS[key] = _make_env(unit["backend"], spec, _SUITES)
    return _ENVS[key]


def run_episode_remote(
    client: PolicyClient,
    env: Any,
    v: Variant,
    init_idx: int,
    ep_seed: int,
    unit: dict[str, Any],
    new_goal: Any,
    record_video: bool,
    probe_cls: Any,
) -> dict[str, Any]:
    """run_eval.run_episode with the policy behind the server (reset → chunks of n_action_steps)."""
    eid = f"{unit['unit_id']}/{v.variant}/i{init_idx}"
    client.reset(eid, ep_seed, v.instruction)
    ep = EpisodeStepper(
        env, v, init_idx, ep_seed, unit["max_steps"], new_goal, record_video, unit["run"]["tag"], probe_cls
    )
    obs = ep.reset()
    if new_goal is not None:
        missing = new_goal.missing_objects(env)
        if missing:
            raise ValueError(f"{v.goal_bddl}: goal names {missing} not in scene of task {unit['task_id']}")
    queue: deque = deque()
    t_policy = 0.0
    done = ep.done
    step = unit["requires_step"]
    while not done:
        if not queue:
            tic = time.perf_counter()
            queue.extend(client.infer(eid, wire_obs(obs), v.instruction, step=step))
            t_policy += time.perf_counter() - tic
        done = ep.step(queue.popleft())
        obs = ep.obs
    client.end(eid)
    return ep.result(t_policy)


def run_unit(unit: dict[str, Any], client: PolicyClient) -> int:
    probe_cls: Any = ContactProbe
    goal_cls: Any = GoalChecker
    if unit["backend"] == "mock":
        from mock_backend import MockGoal, MockProbe

        probe_cls, goal_cls = MockProbe, MockGoal
    out = Path(unit["job_dir"])
    ep_path = out / "episodes.jsonl"
    run, rep, task_id = unit["run"], unit["repeat_idx"], unit["task_id"]
    done = {
        (r["task_id"], r["condition"], r["variant"], r["init_idx"], r["repeat_idx"])
        for r in read_jsonl(ep_path)
    }
    env = env_for(unit)
    n_init = len(env._init_states) if env._init_states is not None else 0
    if unit["episodes"] > n_init:
        raise ValueError(f"task {task_id} has only {n_init} init states")
    cap = unit["video_cap"]
    written = 0
    for vd in unit["variants"]:
        v = Variant(**vd)
        new_goal = goal_cls(v.goal_bddl) if v.goal_bddl else None
        for init_idx in range(unit["episodes"]):
            if (task_id, v.condition, v.variant, init_idx, rep) in done:
                continue
            ep_seed = episode_seed(unit["seed"], rep, task_id, init_idx)
            rec_video = cap is None or init_idx < cap
            t0 = time.time()
            res = run_episode_remote(client, env, v, init_idx, ep_seed, unit, new_goal, rec_video, probe_cls)
            rel = video_relpath(unit["suite"], task_id, v.condition, v.variant, init_idx, rep)
            steps_file, video_file = save_episode(out, rel, res, rec_video)
            rec = episode_record(
                run, task_id, v, init_idx, ep_seed, res, steps_file, video_file, time.time() - t0
            )
            dim, level = OURS_PERTURBATION[v.condition]
            dual = res["success_new"] is not None
            rec.update(
                standard_fields(
                    benchmark="ours",
                    bench_version=unit["bench_version"],
                    model=run["tag"],
                    ckpt=run["ckpt"],
                    suite=unit["suite"],
                    task_id=task_id,
                    variant_id=v.variant,
                    init_idx=init_idx,
                    pert=perturbation(dim, level, v.condition),
                    instruction_original=run["originals"][str(task_id)],
                    instruction_given=v.instruction,
                    success=bool(res["success_new"] if dual else res["success_orig"]),
                    success_orig=res["success_orig"],
                    success_new=res["success_new"],
                    video=video_file,
                    seed=ep_seed,
                )
            )
            rec["policy_server_ms"] = round(client.server_ms, 1)
            client.server_ms = 0.0
            append_jsonl(ep_path, rec)
            print(episode_line(rec), flush=True)
            written += 1
    return written


if __name__ == "__main__":
    run_worker(run_unit)
