"""One LIBERO (or mock) env per subprocess, driven over a Pipe by the batched runner.

Commands (parent → child), each answered with ("ok", payload) or ("err", traceback):
  ("start", spec)          build/reuse the task env, EpisodeStepper.reset() → (obs, done)
  ("step", (action, want_obs))  one env step → (done, obs if want_obs and not done else None)
  ("result", time_policy_s)     EpisodeStepper.result() (arrays + video frames) → dict
  ("close", None)
Observations travel only when the policy needs them (chunk boundaries), video frames once per episode.
The child inherits CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID from the worker, so it renders on the
worker's GPU.
"""

from __future__ import annotations

import contextlib
import os
import sys
import traceback
from collections import OrderedDict
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path[:0] = [p for p in (str(HERE), str(HERE.parent)) if p not in sys.path]

ENV_CACHE = 2  # task envs kept alive per subprocess (MuJoCo sim + renderer each)


class EnvError(RuntimeError):
    pass


def _make_env(backend: str, spec: dict[str, Any], suites: dict[str, Any]) -> Any:
    if backend == "mock":
        from mock_backend import MockEnv, mock_suite

        task = mock_suite(spec["suite"]).task_info[spec["task_id"]]
        return MockEnv(task, spec["max_steps"], spec["env_hw"])
    from run_eval import libero

    lib = libero()
    if spec["suite"] not in suites:
        suites[spec["suite"]] = lib.get_suite(spec["suite"])
    return lib.LiberoEnv(
        suites[spec["suite"]],
        spec["task_id"],
        spec["suite"],
        obs_type="pixels_agent_pos",
        observation_height=spec["env_hw"][0],
        observation_width=spec["env_hw"][1],
        control_mode=spec["control_mode"],
        episode_length=spec["max_steps"],
    )


def serve(conn: Connection, backend: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    from common import Variant
    from run_eval import ContactProbe, EpisodeStepper, GoalChecker

    probe_cls: Any = ContactProbe
    goal_cls: Any = GoalChecker
    if backend == "mock":
        from mock_backend import MockGoal, MockProbe

        probe_cls, goal_cls = MockProbe, MockGoal

    envs: OrderedDict[tuple, Any] = OrderedDict()
    goals: dict[str, Any] = {}
    suites: dict[str, Any] = {}
    stepper: EpisodeStepper | None = None
    while True:
        try:
            cmd, payload = conn.recv()
        except (EOFError, OSError):
            break
        if cmd == "close":
            break
        try:
            if cmd == "start":
                spec = payload
                key = (spec["suite"], spec["task_id"])
                env = envs.pop(key, None)
                if env is None:
                    while len(envs) >= ENV_CACHE:
                        envs.popitem(last=False)[1].close()
                    env = _make_env(backend, spec, suites)
                envs[key] = env
                n_init = len(env._init_states) if env._init_states is not None else 0
                if spec["init_idx"] >= n_init:
                    raise ValueError(f"task {spec['task_id']} has only {n_init} init states")
                v = Variant(**spec["variant"])
                new_goal = None
                if v.goal_bddl:
                    if v.goal_bddl not in goals:
                        goals[v.goal_bddl] = goal_cls(v.goal_bddl)
                    new_goal = goals[v.goal_bddl]
                stepper = EpisodeStepper(
                    env,
                    v,
                    spec["init_idx"],
                    spec["ep_seed"],
                    spec["max_steps"],
                    new_goal,
                    spec["record_video"],
                    spec["tag"],
                    probe_cls,
                )
                obs = stepper.reset()
                if new_goal is not None:
                    missing = new_goal.missing_objects(env)
                    if missing:
                        raise ValueError(
                            f"{v.goal_bddl}: goal names {missing} not in scene of task {spec['task_id']}"
                        )
                conn.send(("ok", (obs, stepper.done)))
            elif cmd == "step":
                assert stepper is not None
                action, want_obs = payload
                done = stepper.step(action)
                conn.send(("ok", (done, stepper.obs if want_obs and not done else None)))
            elif cmd == "result":
                assert stepper is not None
                res = stepper.result(payload)
                stepper = None
                conn.send(("ok", res))
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except Exception:
            stepper = None
            conn.send(("err", traceback.format_exc()))
    for env in envs.values():
        env.close()


class EnvClient:
    def __init__(self, ctx: Any, backend: str, name: str):
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=serve, args=(child, backend), name=name, daemon=True)
        self.proc.start()
        child.close()

    def send(self, cmd: str, payload: Any = None) -> None:
        try:
            self.conn.send((cmd, payload))
        except (BrokenPipeError, OSError) as e:
            raise EnvError(f"env process {self.proc.name} is gone: {e!r}") from e

    def recv(self) -> Any:
        try:
            status, payload = self.conn.recv()
        except (EOFError, OSError) as e:
            raise EnvError(f"env process {self.proc.name} died (exit {self.proc.exitcode})") from e
        if status == "err":
            raise EnvError(payload)
        return payload

    def alive(self) -> bool:
        return self.proc.is_alive()

    def close(self) -> None:
        with contextlib.suppress(Exception):  # already gone
            self.conn.send(("close", None))
        self.proc.join(10)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(5)
