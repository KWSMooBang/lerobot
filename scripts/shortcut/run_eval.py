"""LIBERO shortcut-observation eval: same env and init states, only the instruction string changes.

    uv run python scripts/shortcut/run_eval.py --policy pi05 --suite libero_goal --condition C4 \
        --instr-set scripts/shortcut/instructions/libero_goal.json --tasks 0-9 --episodes 50 \
        --repeat-idx 0 --seed 7 --out $RUN_DIR --video first:5
    # base π0.5 (not LIBERO-trained) with LIBERO config/processors from the finetune:
    ... --checkpoint lerobot/pi05_base   (→ --libero-adapter auto, --video all)
    # batched: N LIBERO envs in subprocesses, one policy call per chunk for all of them (exec_mode batched-N)
    ... --num-envs 8

Conditions (design doc §4): C0 original · C1 "" · C2 random tokens of the same length · C3 paraphrases ·
C4 object swap · C5 target swap · C6 another task's original sentence (same suite).
C0–C3 stop when the original goal holds (as lerobot-eval). C4–C6 run to the suite's max_steps and check
the original goal and the new goal (`goal_bddl`, predicates only; the scene is still the original BDDL)
every step.

Outputs under --out:
  run_meta.json                     args, inference settings, checkpoint training mix, versions
  episodes.jsonl                    one line per episode (appended; reruns skip finished episodes)
  steps/<suite>/taskNN/<cond>_<variant>[_rK]_initNN.npz    per-step log
  videos/<same path>.mp4            agentview | wrist, 10 fps, overlay (condition, instruction, success)

--num-envs 1 (default) is the single-env path of T001. --num-envs N > 1 runs the same episodes through
parallel/run_multi.py on this GPU (one worker); see parallel/README.md. --backend mock is a CPU-only fake
env + policy for the tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess  # nosec B404
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
from common import (  # noqa: E402
    CONDITIONS,
    DUAL_GOAL_CONDITIONS,
    POLICY_CKPT,
    SUITE_SHORT,
    Variant,
    check_resume,
    checkpoint_training_info,
    ensure_libero_config,
    episode_seed,
    exec_identity,
    load_instruction_set,
    load_policy,
    model_tag,
    needs_libero_adapter,
    parse_range,
    variants_for,
    video_relpath,
)

from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.processor.relative_action_processor import bind_relative_anchor  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

VIDEO_FPS = 10
CONTROL_HZ = 20
FRAME_SIDE = 256
DESCENT_DROP_M = 0.05  # first descent = eef z dropped this far below its start, then bottomed out
BACKENDS = ("libero", "mock")
TIMING_FIELDS = ("time_policy_s", "time_env_s", "wall_time_s")  # differ between reruns of one episode


@cache
def libero() -> SimpleNamespace:
    """LIBERO imports, deferred so the mock backend (CPU tests) runs without LIBERO assets."""
    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite

    return SimpleNamespace(
        get_libero_path=get_libero_path,
        robosuite_parse_problem=robosuite_parse_problem,
        TASK_SUITE_MAX_STEPS=TASK_SUITE_MAX_STEPS,
        LiberoEnv=LiberoEnv,
        get_suite=_get_suite,
    )


# ---------------------------------------------------------------- sim probes


def inner(env: Any) -> Any:
    assert env._env is not None, "reset() the env first"
    return env._env.env  # robosuite/LIBERO problem env


class GoalChecker:
    """Evaluate the :goal of a BDDL file against the running scene (conjunction, like _check_success)."""

    def __init__(self, bddl_path: str):
        self.path = bddl_path
        self.goal_state = libero().robosuite_parse_problem(bddl_path)["goal_state"]

    def missing_objects(self, env: Any) -> list[str]:
        states = inner(env).object_states_dict
        return sorted({name for s in self.goal_state for name in s[1:] if name not in states})

    def __call__(self, env: Any) -> bool:
        e = inner(env)
        return all(e._eval_predicate(s) for s in self.goal_state)


class ContactProbe:
    """Which scene objects/fixtures the gripper touches; 'grasped' = touched by both fingers."""

    def __init__(self, env: Any):
        e = inner(env)
        m = e.sim.model
        roots = {}
        for name, obj in {**e.objects_dict, **e.fixtures_dict}.items():
            roots[m.body_name2id(obj.root_body)] = name
        self.geom_obj: dict[int, str] = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            while b > 0 and b not in roots:
                b = int(m.body_parentid[b])
            if b in roots:
                self.geom_obj[g] = roots[b]
        grip = e.robots[0].gripper
        ids = lambda names: {m.geom_name2id(n) for n in names}  # noqa: E731
        self.grip_geoms = ids(grip.contact_geoms)
        self.left = ids(grip.important_geoms["left_finger"])
        self.right = ids(grip.important_geoms["right_finger"])
        self.sim = e.sim

    def __call__(self) -> tuple[list[str], list[str]]:
        d = self.sim.data
        touched: set[str] = set()
        left: set[str] = set()
        right: set[str] = set()
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.grip_geoms and b in self.geom_obj:
                    obj = self.geom_obj[b]
                    touched.add(obj)
                    if a in self.left:
                        left.add(obj)
                    if a in self.right:
                        right.add(obj)
        return sorted(touched), sorted(left & right)


def scene_groups(suite: Any) -> dict[int, int]:
    """k = number of tasks in the suite that share this task's scene (same objects, fixtures and
    initial regions). Chance level for SR_new under a language-blind policy is 1/k."""
    lib = libero()
    keys = {}
    for i, t in enumerate(suite.tasks):
        p = lib.robosuite_parse_problem(
            str(Path(lib.get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file)
        )
        regions = {
            name: (r.get("target"), json.dumps(r.get("ranges"), sort_keys=True))
            for name, r in p["regions"].items()
        }
        keys[i] = json.dumps(
            [
                sorted(p["fixtures"].items()),
                sorted(p["objects"].items()),
                sorted(regions.items()),
                p["initial_state"],
            ],
            sort_keys=True,
            default=str,
        )
    return {i: sum(k == keys[i] for k in keys.values()) for i in keys}


def first_descent(eef: np.ndarray, grip_cmd: np.ndarray) -> tuple[int | None, list[float] | None]:
    """xy where the gripper first comes down: z drops DESCENT_DROP_M below start, then the step where it
    stops descending (local min) or a close command arrives, whichever is first."""
    if len(eef) == 0:
        return None, None
    z = eef[:, 2]
    below = np.nonzero(z <= z[0] - DESCENT_DROP_M)[0]
    if len(below) == 0:
        return None, None
    t = int(below[0])
    while t + 1 < len(z) and z[t + 1] < z[t] - 1e-4 and grip_cmd[t] <= 0:
        t += 1
    return t, [float(eef[t, 0]), float(eef[t, 1])]


# ---------------------------------------------------------------- video


def overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    h = 14
    box_h = 6 + h * len(lines)
    roi = frame[:box_h]
    roi[:] = (roi * 0.45).astype(np.uint8)
    for i, (txt, color) in enumerate(lines):
        cv2.putText(frame, txt, (5, 13 + i * h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame


def compose_frame(obs: dict, header: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    views = []
    for key in ("image", "image2"):
        img = np.ascontiguousarray(obs["pixels"][key][::-1, ::-1])  # policy view (180° rotation)
        views.append(cv2.resize(img, (FRAME_SIDE, FRAME_SIDE), interpolation=cv2.INTER_AREA))
    return overlay(np.concatenate(views, axis=1), header)


def header_lines(
    v: Variant, init_idx: int, t: int, t_orig: int | None, t_new: int | None, dual: bool, tag: str = ""
) -> list[tuple[str, tuple[int, int, int]]]:
    white, green, red, grey = (255, 255, 255), (80, 255, 80), (255, 90, 90), (200, 200, 200)
    instr = v.instruction if v.instruction else '""'
    wrapped = textwrap.wrap(instr, 62)[:2] or ['""']
    lines = [(f"{tag}  {v.condition} {v.variant}  init {init_idx:02d}  t={t}".strip(), white)]
    lines += [(f"> {w}" if i == 0 else f"  {w}", grey) for i, w in enumerate(wrapped)]
    ok = lambda tt: (f"yes@{tt}", green) if tt is not None else ("no", red)  # noqa: E731
    s, c = ok(t_orig)
    if dual:
        s2, c2 = ok(t_new)
        lines += [(f"orig goal: {s}", c), (f"new goal:  {s2}", c2)]
    else:
        lines.append((f"success: {s}", c))
    return lines


# ---------------------------------------------------------------- rollout


def _batchify(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _batchify(v) for k, v in x.items()}
    return np.asarray(x)[None]


class EpisodeStepper:
    """Env side of one episode: reset() → obs, then step(action) until it returns True, then result().

    Shared by the single-env loop (run_episode) and the batched env subprocesses (parallel/envproc.py),
    so both log, stop and score episodes the same way. C0–C3 stop at the original goal; C4–C6 run to
    max_steps and track both goals."""

    def __init__(
        self,
        env: Any,
        v: Variant,
        init_idx: int,
        ep_seed: int,
        max_steps: int,
        new_goal: Callable[[Any], bool] | None,
        record_video: bool,
        tag: str = "",
        probe_cls: Callable[[Any], Callable[[], tuple[list[str], list[str]]]] = ContactProbe,
    ):
        self.env, self.v, self.init_idx, self.ep_seed = env, v, init_idx, ep_seed
        self.max_steps, self.new_goal, self.record_video, self.tag = max_steps, new_goal, record_video, tag
        self.probe_cls = probe_cls
        self.dual = v.condition in DUAL_GOAL_CONDITIONS

    def reset(self) -> dict:
        np.random.seed(self.ep_seed % 2**32)
        self.env.init_state_id = self.init_idx
        self.obs, _ = self.env.reset(seed=self.ep_seed)
        self.probe = self.probe_cls(self.env)
        self.t_orig: int | None = None
        self.t_new: int | None = None
        self.frames: list[np.ndarray] = []
        if self.record_video:
            self.frames.append(self._frame(0))
        self.log: dict[str, list] = {
            k: [] for k in ("eef_pos", "eef_quat", "gripper_qpos", "action", "goal_orig", "goal_new")
        }
        self.contacts: list[str] = []
        self.grasped: list[str] = []
        self.t_env = 0.0
        self.t = 0
        self.done = self.max_steps <= 0
        return self.obs

    def _frame(self, t: int) -> np.ndarray:
        return compose_frame(
            self.obs, header_lines(self.v, self.init_idx, t, self.t_orig, self.t_new, self.dual, self.tag)
        )

    def step(self, a: np.ndarray) -> bool:
        assert not self.done, "episode already finished"
        toc = time.perf_counter()
        self.obs, _, _, _, info = self.env.step(a)
        self.t_env += time.perf_counter() - toc
        t = self.t
        g_orig = bool(info["is_success"])
        g_new = bool(self.new_goal(self.env)) if self.new_goal is not None else False
        if g_orig and self.t_orig is None:
            self.t_orig = t + 1
        if g_new and self.t_new is None:
            self.t_new = t + 1
        touched, grasp = self.probe()
        rs = self.obs["robot_state"]
        self.log["eef_pos"].append(rs["eef"]["pos"])
        self.log["eef_quat"].append(rs["eef"]["quat"])
        self.log["gripper_qpos"].append(rs["gripper"]["qpos"])
        self.log["action"].append(a)
        self.log["goal_orig"].append(g_orig)
        self.log["goal_new"].append(g_new)
        self.contacts.append("|".join(touched))
        self.grasped.append("|".join(grasp))
        if self.record_video and (t + 1) % (CONTROL_HZ // VIDEO_FPS) == 0:
            self.frames.append(self._frame(t + 1))
        self.t = t + 1
        self.done = (not self.dual and g_orig) or self.t >= self.max_steps
        return self.done

    def result(self, time_policy_s: float) -> dict[str, Any]:
        dual, contacts, grasped, frames = self.dual, self.contacts, self.grasped, self.frames
        t_orig, t_new = self.t_orig, self.t_new
        if self.record_video:  # hold the last frame ~1 s with the final verdict
            last = self._frame(len(contacts))
            frames.extend([last] * VIDEO_FPS)

        arr = {
            k: np.asarray(x, dtype=np.float32 if k not in ("goal_orig", "goal_new") else bool)
            for k, x in self.log.items()
        }
        arr["gripper_cmd"] = arr["action"][:, 6] if len(arr["action"]) else np.zeros(0, np.float32)
        arr["gripper_closing"] = arr["gripper_cmd"] > 0
        arr["gripper_width"] = (
            arr["gripper_qpos"][:, 0] - arr["gripper_qpos"][:, 1] if len(arr["gripper_qpos"]) else np.zeros(0)
        )
        arr["contacts"] = np.asarray(contacts, dtype=str)
        arr["grasped"] = np.asarray(grasped, dtype=str)

        first_grasp = first_grasp_t = None
        for t, (g, closing) in enumerate(zip(grasped, arr["gripper_closing"], strict=True)):
            if g and closing:
                first_grasp, first_grasp_t = g.split("|")[0], t + 1
                break
        first_contact = next((c.split("|")[0] for c in contacts if c), None)
        descent_t, descent_xy = first_descent(arr["eef_pos"], arr["gripper_cmd"])
        return {
            "arrays": arr,
            "frames": frames,
            "success_orig": t_orig is not None,
            "success_new": (t_new is not None) if dual else None,
            "t_orig": t_orig,
            "t_new": t_new if dual else None,
            "first_grasp_object": first_grasp,
            "first_grasp_t": first_grasp_t,
            "first_contact_object": first_contact,
            "first_descent_xy": descent_xy,
            "first_descent_t": None if descent_t is None else descent_t + 1,
            "n_steps": len(contacts),
            "time_policy_s": round(time_policy_s, 2),
            "time_env_s": round(self.t_env, 2),
            "final_success_orig": bool(arr["goal_orig"][-1]) if len(contacts) else False,
            "final_success_new": (bool(arr["goal_new"][-1]) if len(contacts) else False) if dual else None,
        }


def run_episode(
    env: Any,
    policy: Any,
    pre: Any,
    post: Any,
    env_pre: Any,
    env_post: Any,
    v: Variant,
    init_idx: int,
    ep_seed: int,
    max_steps: int,
    new_goal: Callable[[Any], bool] | None,
    record_video: bool,
    tag: str = "",
    probe_cls: Callable[[Any], Callable[[], tuple[list[str], list[str]]]] = ContactProbe,
) -> dict[str, Any]:
    """Single-env episode (exec_mode single). Flow-matching noise comes from the global torch RNG
    seeded with ep_seed; the batched path draws the same stream from a per-episode generator."""
    policy.reset()
    pre.reset()
    post.reset()
    bind_relative_anchor(policy, pre)
    torch.manual_seed(ep_seed)
    ep = EpisodeStepper(env, v, init_idx, ep_seed, max_steps, new_goal, record_video, tag, probe_cls)
    obs = ep.reset()
    t_policy = 0.0
    done = ep.done
    while not done:
        tic = time.perf_counter()
        batch = preprocess_observation(_batchify(obs))
        batch["task"] = [v.instruction]
        batch = pre(env_pre(batch))
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = env_post({ACTION: post(action)})[ACTION]
        a = action.to("cpu").numpy()[0]
        t_policy += time.perf_counter() - tic
        done = ep.step(a)
        obs = ep.obs
    return ep.result(t_policy)


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=VIDEO_FPS, codec="libx264", quality=7, macro_block_size=8)


def tmp_path(path: Path) -> Path:
    """Sibling temp name with the same suffix (imageio picks the format from it)."""
    return path.with_name(f"{path.stem}.tmp-{os.getpid()}-{threading.get_ident()}{path.suffix}")


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_episode(out: Path, rel: str, res: dict[str, Any], record_video: bool) -> tuple[str, str | None]:
    """Write steps/<rel>.npz (+ videos/<rel>.mp4); pops arrays and frames from res. Returns both paths
    relative to out. Each file is written to a temp name and os.replace'd, so a writer killed mid-way
    never leaves a truncated file at the final path (an earlier complete copy stays intact)."""
    steps_file = out / "steps" / f"{rel}.npz"
    save_npz(steps_file, res.pop("arrays"))
    frames = res.pop("frames")
    video_file = None
    if record_video:
        video_file = f"videos/{rel}.mp4"
        final = out / video_file
        tmp = tmp_path(final)
        write_video(tmp, frames)
        os.replace(tmp, final)
    return str(steps_file.relative_to(out)), video_file


def episode_record(
    run: dict[str, Any],
    task_id: int,
    v: Variant,
    init_idx: int,
    ep_seed: int,
    res: dict[str, Any],
    steps_file: str,
    video_file: str | None,
    wall_time_s: float,
) -> dict[str, Any]:
    """One episodes.jsonl line. `run` = RunSpec.info() (model tag, checkpoint, suite, originals, scene_k)."""
    return {
        "model": run["tag"],
        "policy": run["policy"],
        "ckpt": run["ckpt"],
        "suite": run["suite"],
        "task_id": task_id,
        "init_idx": init_idx,
        "repeat_idx": run["repeat_idx"],
        "condition": v.condition,
        "variant": v.variant,
        "instruction": v.instruction,
        "original_instruction": run["originals"][str(task_id)],
        **res,
        "goal_bddl": v.goal_bddl,
        "variant_meta": v.meta,
        "scene_k": run["scene_k"][str(task_id)],
        "max_steps": run["max_steps"],
        "early_stop": v.condition not in DUAL_GOAL_CONDITIONS,
        "seed": ep_seed,
        "video": video_file,
        "steps_file": steps_file,
        "wall_time_s": round(wall_time_s, 2),
    }


def episode_line(rec: dict[str, Any]) -> str:
    return (
        f"[ep] task {rec['task_id']} {rec['condition']}/{rec['variant']} init {rec['init_idx']:02d} "
        f"orig={rec['success_orig']}@{rec['t_orig']} new={rec['success_new']}@{rec['t_new']} "
        f"grasp={rec['first_grasp_object']} steps={rec['n_steps']} {rec['wall_time_s']}s"
    )


# ---------------------------------------------------------------- run planning


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # nosec B607
    except Exception:
        return "unknown"


def parse_video(spec: str) -> int | None:
    """'all' -> None (no cap), 'none' -> 0, 'first:N' -> N first init indices per (task, variant)."""
    if spec == "all":
        return None
    if spec == "none":
        return 0
    if spec.startswith("first:"):
        return int(spec.split(":", 1)[1])
    raise ValueError(f"--video must be all|none|first:N, got {spec}")


def gpu_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


@dataclass
class RunSpec:
    """One (checkpoint, suite, condition, repeat) run, resolved from run_eval.py arguments."""

    args: argparse.Namespace
    ckpt: str
    adapter: str | None
    tag: str
    out: Path
    video_cap: int | None
    max_steps: int
    suite: Any
    instr_set: dict[str, Any]
    k_by_task: dict[int, int]
    identity: dict[str, Any]

    def info(self) -> dict[str, Any]:
        """JSON-safe fields the episode records need (keys are str for JSON round trips)."""
        return {
            "tag": self.tag,
            "policy": self.args.policy,
            "ckpt": self.ckpt,
            "adapter": self.adapter,
            "suite": self.args.suite,
            "condition": self.args.condition,
            "repeat_idx": self.args.repeat_idx,
            "max_steps": self.max_steps,
            "n_action_steps": self.args.n_action_steps,
            "originals": {str(i): t["original"] for i, t in self.instr_set["_suite_tasks"].items()},
            "scene_k": {str(i): k for i, k in self.k_by_task.items()},
            "out": str(self.out),
            "identity": self.identity,
            "backend": self.args.backend,
        }

    def done_keys(self) -> set[tuple]:
        """(task, condition, variant, init, repeat) already in episodes.jsonl."""
        done: set[tuple] = set()
        ep_path = self.out / "episodes.jsonl"
        if ep_path.exists():
            for line in ep_path.read_text().splitlines():
                r = json.loads(line)
                done.add((r["task_id"], r["condition"], r["variant"], r["init_idx"], r["repeat_idx"]))
        return done


def suite_tasks(args: argparse.Namespace) -> tuple[Any, dict[int, dict[str, Any]], dict[int, int]]:
    """(suite object, {task: {task_id, original, bddl}}, scene_k by task) for the chosen backend."""
    if args.backend == "mock":
        from mock_backend import mock_suite

        suite = mock_suite(args.suite)
        tasks = {i: dict(t) for i, t in enumerate(suite.task_info)}
        return suite, tasks, {i: len(tasks) for i in tasks}
    lib = libero()
    suite = lib.get_suite(args.suite)
    bddl_root = Path(lib.get_libero_path("bddl_files"))
    tasks = {
        i: {"task_id": i, "original": t.language, "bddl": str(bddl_root / t.problem_folder / t.bddl_file)}
        for i, t in enumerate(suite.tasks)
    }
    return suite, tasks, scene_groups(suite)


def default_max_steps(args: argparse.Namespace) -> int:
    if args.max_steps:
        return int(args.max_steps)
    if args.backend == "mock":
        from mock_backend import MOCK_MAX_STEPS

        return MOCK_MAX_STEPS
    return int(libero().TASK_SUITE_MAX_STEPS[args.suite])


def prepare_run(
    args: argparse.Namespace, exec_mode: str, gpu: str | None, identity_extra: dict[str, Any] | None = None
) -> RunSpec:
    """Resolve checkpoint/adapter/tag/video/max_steps, load the instruction set and suite, build the
    identity (incl. exec_mode, GPU model, torch, CUDA) and check it against an existing --out.
    `identity_extra` is merged last (bench/ours: the policy server's exec fields and `via`)."""
    ckpt = args.checkpoint or POLICY_CKPT[args.policy]
    adapter = args.libero_adapter
    if adapter == "auto":
        adapter = POLICY_CKPT[args.policy] if args.backend != "mock" and needs_libero_adapter(ckpt) else None
    elif adapter == "none":
        adapter = None
    tag = args.model_tag or model_tag(ckpt)
    if args.video is None:  # base checkpoints: qualitative comparison is the point → keep every video
        args.video = "all" if adapter else "first:5"
    out = Path(args.out)
    video_cap = parse_video(args.video)
    max_steps = default_max_steps(args)

    suite, tasks, k_by_task = suite_tasks(args)
    instr_set = load_instruction_set(args.instr_set)
    if instr_set.get("suite") not in (None, args.suite):
        raise ValueError(f"instruction set is for {instr_set['suite']}, not {args.suite}")
    instr_set["_suite_tasks"] = tasks

    # Resume guard: an out dir holds exactly one configuration. Checked before loading the policy.
    identity = {
        "policy": args.policy,
        "checkpoint": ckpt,
        "libero_adapter": adapter,
        "suite": args.suite,
        "condition": args.condition,
        "repeat_idx": args.repeat_idx,
        "seed": args.seed,
        "instr_set_sha256": hashlib.sha256(Path(args.instr_set).read_bytes()).hexdigest(),
        "n_action_steps": args.n_action_steps,
        "max_steps": max_steps,
        **exec_identity(exec_mode, gpu),
        **(identity_extra or {}),
    }
    if args.backend != "libero":
        identity["backend"] = args.backend
    prev_meta = json.loads((out / "run_meta.json").read_text()) if (out / "run_meta.json").exists() else None
    check_resume(identity, prev_meta, out / "episodes.jsonl")
    return RunSpec(args, ckpt, adapter, tag, out, video_cap, max_steps, suite, instr_set, k_by_task, identity)


def plan_variants(spec: RunSpec, tokenizer: Any) -> list[tuple[int, list[Variant]]]:
    """[(task_id, variants)] in run order; prints the same [skip]/[warn] notes as before."""
    args, instr_set = spec.args, spec.instr_set
    restrict = set(args.variants.split(",")) if args.variants else None
    out = []
    for task_id in parse_range(args.tasks):
        suite_task = instr_set["_suite_tasks"][task_id]
        entry = instr_set["_by_task"].get(task_id)
        if entry is None:
            if args.condition in ("C0", "C1", "C2"):
                entry = dict(suite_task)
            else:
                print(f"[skip] task {task_id}: not in instruction set", flush=True)
                continue
        variants = variants_for(args.condition, entry, instr_set, tokenizer, args.include_infeasible)
        if args.condition == "C0" and entry["original"] != suite_task["original"]:
            print(f"[warn] task {task_id}: JSON original != suite language; using suite language", flush=True)
            variants = [Variant("C0", "orig", suite_task["original"])]
        if restrict:
            variants = [v for v in variants if v.variant in restrict]
        if not variants:
            print(f"[skip] task {task_id}: no variants for {args.condition}", flush=True)
            continue
        out.append((task_id, variants))
    return out


def run_meta(
    spec: RunSpec, inf_meta: dict[str, Any] | None, prev_meta: dict[str, Any] | None, **extra: Any
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    meta = {
        "identity": spec.identity,
        "created": prev_meta["created"] if prev_meta else now,
        "resumed": [*(prev_meta or {}).get("resumed", []), now] if prev_meta else [],
        "args": vars(spec.args),
        "checkpoint": spec.ckpt,
        "model_tag": spec.tag,
        "checkpoint_training": (
            checkpoint_training_info(spec.ckpt) if spec.args.backend == "libero" else {"backend": "mock"}
        ),
        "inference": inf_meta,
        "max_steps": spec.max_steps,
        "video": {"fps": VIDEO_FPS, "frame": [FRAME_SIDE, 2 * FRAME_SIDE], "views": ["agentview", "wrist"]},
        "seed_rule": "base*1e6 + repeat*1e4 + task*100 + init (same across conditions)",
        "scene_k": spec.k_by_task,
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": spec.identity["gpu"] if spec.identity["gpu"] != "cpu" else None,
    }
    meta.update(extra)
    return meta


def make_env(args: argparse.Namespace, suite: Any, task_id: int, max_steps: int, inf_meta: dict) -> Any:
    if args.backend == "mock":
        from mock_backend import MockEnv

        return MockEnv(suite.task_info[task_id], max_steps, inf_meta["env_observation_hw"])
    return libero().LiberoEnv(
        suite,
        task_id,
        args.suite,
        obs_type="pixels_agent_pos",
        observation_height=inf_meta["env_observation_hw"][0],
        observation_width=inf_meta["env_observation_hw"][1],
        control_mode=inf_meta["control_mode"],
        episode_length=max_steps,
    )


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=sorted(POLICY_CKPT), required=True)
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="hub id: lerobot/pi05_libero_finetuned (default, tag pi05lib) or lerobot/pi05_base (tag pi05base)",
    )
    ap.add_argument(
        "--libero-adapter",
        default="auto",
        help="LIBERO-finetuned ckpt whose config+processors are used with --checkpoint weights. "
        "'auto': use the --policy default finetune when --checkpoint lacks LIBERO input keys; 'none': off",
    )
    ap.add_argument(
        "--model-tag", default=None, help="model label in episodes.jsonl (default from checkpoint)"
    )
    ap.add_argument("--suite", choices=list(SUITE_SHORT), required=True)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--instr-set", required=True)
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--episodes", type=int, default=50, help="first N init states per task")
    ap.add_argument("--repeat-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--video", default=None, help="all|none|first:N (default: all with adapter, else first:5)"
    )
    ap.add_argument("--n-action-steps", type=int, default=10, help="docs/source/libero.mdx (OpenPI setting)")
    ap.add_argument("--max-steps", type=int, default=None, help="default: lerobot TASK_SUITE_MAX_STEPS")
    ap.add_argument("--variants", default=None, help="comma list to restrict variant ids (debug)")
    ap.add_argument("--include-infeasible", action="store_true", help="also run swaps with feasible=false")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="1: single env (T001 path). N>1: N env subprocesses + batched policy calls (exec_mode batched-N)",
    )
    ap.add_argument(
        "--compile", action="store_true", help="torch.compile (batched path only; off by default)"
    )
    ap.add_argument(
        "--backend", choices=BACKENDS, default="libero", help="mock = CPU fake env/policy (tests)"
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.num_envs != 1 or args.compile:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "parallel"))
        from run_multi import run_single_dir

        run_single_dir(args)
        return
    if args.backend == "mock":
        sys.path.insert(0, str(Path(__file__).resolve().parent / "parallel"))
        args.device = "cpu"

    spec = prepare_run(args, "single", gpu_name(args.device))
    out = spec.out
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run_meta.json"
    ep_path = out / "episodes.jsonl"
    prev_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None

    probe_cls: Any = ContactProbe
    goal_cls: Any = GoalChecker
    if args.backend == "mock":
        from mock_backend import MockGoal, MockProbe, load_mock_policy

        policy, pre, post, env_pre, env_post, tokenizer, inf_meta = load_mock_policy(
            spec.ckpt, args.n_action_steps
        )
        probe_cls, goal_cls = MockProbe, MockGoal
    else:
        policy, pre, post, env_pre, env_post, tokenizer, inf_meta = load_policy(
            args.policy, spec.ckpt, args.suite, args.n_action_steps, args.device, libero_adapter=spec.adapter
        )
    with open(meta_path, "w") as f:
        json.dump(run_meta(spec, inf_meta, prev_meta), f, indent=2, default=str)

    done = spec.done_keys()
    run = spec.info()
    for task_id, variants in plan_variants(spec, tokenizer):
        env = make_env(args, spec.suite, task_id, spec.max_steps, inf_meta)
        n_init = len(env._init_states) if env._init_states is not None else 0
        if args.episodes > n_init:
            raise ValueError(f"task {task_id} has only {n_init} init states")
        for v in variants:
            new_goal = goal_cls(v.goal_bddl) if v.goal_bddl else None
            for init_idx in range(args.episodes):
                key = (task_id, args.condition, v.variant, init_idx, args.repeat_idx)
                if key in done:
                    continue
                ep_seed = episode_seed(args.seed, args.repeat_idx, task_id, init_idx)
                rec_video = spec.video_cap is None or init_idx < spec.video_cap
                t0 = time.time()
                if new_goal is not None and init_idx == 0:
                    env.reset(seed=ep_seed)
                    missing = new_goal.missing_objects(env)
                    if missing:
                        raise ValueError(
                            f"{v.goal_bddl}: goal names {missing} not in scene of task {task_id}"
                        )
                res = run_episode(
                    env,
                    policy,
                    pre,
                    post,
                    env_pre,
                    env_post,
                    v,
                    init_idx,
                    ep_seed,
                    spec.max_steps,
                    new_goal,
                    rec_video,
                    spec.tag,
                    probe_cls,
                )
                rel = video_relpath(args.suite, task_id, v.condition, v.variant, init_idx, args.repeat_idx)
                steps_file, video_file = save_episode(out, rel, res, rec_video)
                rec = episode_record(
                    run, task_id, v, init_idx, ep_seed, res, steps_file, video_file, time.time() - t0
                )
                with open(ep_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(episode_line(rec), flush=True)
        env.close()
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
