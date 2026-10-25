"""LIBERO shortcut-observation eval: same env and init states, only the instruction string changes.

    uv run python scripts/shortcut/run_eval.py --policy pi05 --suite libero_goal --condition C4 \
        --instr-set scripts/shortcut/instructions/libero_goal.json --tasks 0-9 --episodes 50 \
        --repeat-idx 0 --seed 7 --out $RUN_DIR --video first:5
    # base π0.5 (not LIBERO-trained) with LIBERO config/processors from the finetune:
    ... --checkpoint lerobot/pi05_base   (→ --libero-adapter auto, --video all)

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
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess  # nosec B404
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CONDITIONS,
    DUAL_GOAL_CONDITIONS,
    POLICY_CKPT,
    Variant,
    checkpoint_training_info,
    ensure_libero_config,
    episode_seed,
    load_instruction_set,
    load_policy,
    model_tag,
    needs_libero_adapter,
    parse_range,
    variants_for,
    video_relpath,
)

ensure_libero_config()

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs.bddl_utils import robosuite_parse_problem  # noqa: E402

from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.processor.relative_action_processor import bind_relative_anchor  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

VIDEO_FPS = 10
CONTROL_HZ = 20
FRAME_SIDE = 256
DESCENT_DROP_M = 0.05  # first descent = eef z dropped this far below its start, then bottomed out


# ---------------------------------------------------------------- sim probes


def inner(env: LiberoEnv) -> Any:
    assert env._env is not None, "reset() the env first"
    return env._env.env  # robosuite/LIBERO problem env


class GoalChecker:
    """Evaluate the :goal of a BDDL file against the running scene (conjunction, like _check_success)."""

    def __init__(self, bddl_path: str):
        self.path = bddl_path
        self.goal_state = robosuite_parse_problem(bddl_path)["goal_state"]

    def missing_objects(self, env: LiberoEnv) -> list[str]:
        states = inner(env).object_states_dict
        return sorted({name for s in self.goal_state for name in s[1:] if name not in states})

    def __call__(self, env: LiberoEnv) -> bool:
        e = inner(env)
        return all(e._eval_predicate(s) for s in self.goal_state)


class ContactProbe:
    """Which scene objects/fixtures the gripper touches; 'grasped' = touched by both fingers."""

    def __init__(self, env: LiberoEnv):
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
    keys = {}
    for i, t in enumerate(suite.tasks):
        p = robosuite_parse_problem(str(Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file))
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


def run_episode(
    env: LiberoEnv,
    policy: Any,
    pre: Any,
    post: Any,
    env_pre: Any,
    env_post: Any,
    v: Variant,
    init_idx: int,
    ep_seed: int,
    max_steps: int,
    new_goal: GoalChecker | None,
    record_video: bool,
    tag: str = "",
) -> dict[str, Any]:
    dual = v.condition in DUAL_GOAL_CONDITIONS
    policy.reset()
    pre.reset()
    post.reset()
    bind_relative_anchor(policy, pre)
    torch.manual_seed(ep_seed)
    np.random.seed(ep_seed % 2**32)
    env.init_state_id = init_idx
    obs, _ = env.reset(seed=ep_seed)
    probe = ContactProbe(env)
    frames: list[np.ndarray] = []
    if record_video:
        frames.append(compose_frame(obs, header_lines(v, init_idx, 0, None, None, dual, tag)))

    log: dict[str, list] = {
        k: [] for k in ("eef_pos", "eef_quat", "gripper_qpos", "action", "goal_orig", "goal_new")
    }
    contacts: list[str] = []
    grasped: list[str] = []
    t_orig = t_new = None
    t_policy = t_env = 0.0
    for t in range(max_steps):
        tic = time.perf_counter()
        batch = preprocess_observation(_batchify(obs))
        batch["task"] = [v.instruction]
        batch = pre(env_pre(batch))
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = env_post({ACTION: post(action)})[ACTION]
        a = action.to("cpu").numpy()[0]
        toc = time.perf_counter()
        obs, _, _, _, info = env.step(a)
        t_policy += toc - tic
        t_env += time.perf_counter() - toc

        g_orig = bool(info["is_success"])
        g_new = bool(new_goal(env)) if new_goal is not None else False
        if g_orig and t_orig is None:
            t_orig = t + 1
        if g_new and t_new is None:
            t_new = t + 1
        touched, grasp = probe()
        rs = obs["robot_state"]
        log["eef_pos"].append(rs["eef"]["pos"])
        log["eef_quat"].append(rs["eef"]["quat"])
        log["gripper_qpos"].append(rs["gripper"]["qpos"])
        log["action"].append(a)
        log["goal_orig"].append(g_orig)
        log["goal_new"].append(g_new)
        contacts.append("|".join(touched))
        grasped.append("|".join(grasp))
        if record_video and (t + 1) % (CONTROL_HZ // VIDEO_FPS) == 0:
            frames.append(compose_frame(obs, header_lines(v, init_idx, t + 1, t_orig, t_new, dual, tag)))
        if not dual and g_orig:
            break
    if record_video:  # hold the last frame ~1 s with the final verdict
        last = compose_frame(obs, header_lines(v, init_idx, len(contacts), t_orig, t_new, dual, tag))
        frames.extend([last] * VIDEO_FPS)

    arr = {
        k: np.asarray(x, dtype=np.float32 if k not in ("goal_orig", "goal_new") else bool)
        for k, x in log.items()
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
        "time_policy_s": round(t_policy, 2),
        "time_env_s": round(t_env, 2),
        "final_success_orig": bool(arr["goal_orig"][-1]) if len(contacts) else False,
        "final_success_new": (bool(arr["goal_new"][-1]) if len(contacts) else False) if dual else None,
    }


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=VIDEO_FPS, codec="libx264", quality=7, macro_block_size=8)


# ---------------------------------------------------------------- main


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


def main() -> None:
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
    ap.add_argument("--suite", choices=list(TASK_SUITE_MAX_STEPS)[:4], required=True)
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
    args = ap.parse_args()

    ckpt = args.checkpoint or POLICY_CKPT[args.policy]
    adapter = args.libero_adapter
    if adapter == "auto":
        adapter = POLICY_CKPT[args.policy] if needs_libero_adapter(ckpt) else None
    elif adapter == "none":
        adapter = None
    tag = args.model_tag or model_tag(ckpt)
    if args.video is None:  # base checkpoints: qualitative comparison is the point → keep every video
        args.video = "all" if adapter else "first:5"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_cap = parse_video(args.video)
    max_steps = args.max_steps or TASK_SUITE_MAX_STEPS[args.suite]

    suite = _get_suite(args.suite)
    bddl_root = Path(get_libero_path("bddl_files"))
    instr_set = load_instruction_set(args.instr_set)
    if instr_set.get("suite") not in (None, args.suite):
        raise ValueError(f"instruction set is for {instr_set['suite']}, not {args.suite}")
    instr_set["_suite_tasks"] = {
        i: {"task_id": i, "original": t.language, "bddl": str(bddl_root / t.problem_folder / t.bddl_file)}
        for i, t in enumerate(suite.tasks)
    }
    k_by_task = scene_groups(suite)

    policy, pre, post, env_pre, env_post, tokenizer, inf_meta = load_policy(
        args.policy, ckpt, args.suite, args.n_action_steps, args.device, libero_adapter=adapter
    )
    meta_path = out / "run_meta.json"
    meta = {
        "created": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "checkpoint": ckpt,
        "model_tag": tag,
        "checkpoint_training": checkpoint_training_info(ckpt),
        "inference": inf_meta,
        "max_steps": max_steps,
        "video": {"fps": VIDEO_FPS, "frame": [FRAME_SIDE, 2 * FRAME_SIDE], "views": ["agentview", "wrist"]},
        "seed_rule": "base*1e6 + repeat*1e4 + task*100 + init (same across conditions)",
        "scene_k": k_by_task,
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    ep_path = out / "episodes.jsonl"
    done: set[tuple] = set()
    if ep_path.exists():
        for line in ep_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task_id"], r["condition"], r["variant"], r["init_idx"], r["repeat_idx"]))
    restrict = set(args.variants.split(",")) if args.variants else None

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

        env = LiberoEnv(
            suite,
            task_id,
            args.suite,
            obs_type="pixels_agent_pos",
            observation_height=inf_meta["env_observation_hw"][0],
            observation_width=inf_meta["env_observation_hw"][1],
            control_mode=inf_meta["control_mode"],
            episode_length=max_steps,
        )
        n_init = len(env._init_states) if env._init_states is not None else 0
        if args.episodes > n_init:
            raise ValueError(f"task {task_id} has only {n_init} init states")
        for v in variants:
            new_goal = GoalChecker(v.goal_bddl) if v.goal_bddl else None
            for init_idx in range(args.episodes):
                key = (task_id, args.condition, v.variant, init_idx, args.repeat_idx)
                if key in done:
                    continue
                ep_seed = episode_seed(args.seed, args.repeat_idx, task_id, init_idx)
                rec_video = video_cap is None or init_idx < video_cap
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
                    max_steps,
                    new_goal,
                    rec_video,
                    tag,
                )
                rel = video_relpath(args.suite, task_id, v.condition, v.variant, init_idx, args.repeat_idx)
                steps_file = out / "steps" / f"{rel}.npz"
                steps_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(steps_file, **res.pop("arrays"))
                frames = res.pop("frames")
                video_file = None
                if rec_video:
                    video_file = f"videos/{rel}.mp4"
                    write_video(out / video_file, frames)
                rec = {
                    "model": tag,
                    "policy": args.policy,
                    "ckpt": ckpt,
                    "suite": args.suite,
                    "task_id": task_id,
                    "init_idx": init_idx,
                    "repeat_idx": args.repeat_idx,
                    "condition": v.condition,
                    "variant": v.variant,
                    "instruction": v.instruction,
                    "original_instruction": suite_task["original"],
                    **res,
                    "goal_bddl": v.goal_bddl,
                    "variant_meta": v.meta,
                    "scene_k": k_by_task[task_id],
                    "max_steps": max_steps,
                    "early_stop": v.condition not in DUAL_GOAL_CONDITIONS,
                    "seed": ep_seed,
                    "video": video_file,
                    "steps_file": str(steps_file.relative_to(out)),
                    "wall_time_s": round(time.time() - t0, 2),
                }
                with open(ep_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(
                    f"[ep] task {task_id} {v.condition}/{v.variant} init {init_idx:02d} "
                    f"orig={rec['success_orig']}@{rec['t_orig']} new={rec['success_new']}@{rec['t_new']} "
                    f"grasp={rec['first_grasp_object']} steps={rec['n_steps']} {rec['wall_time_s']}s",
                    flush=True,
                )
        env.close()
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
