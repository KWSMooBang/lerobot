"""Collect real pi0.5 outputs for the PID demo (GPU part).

1. One Goal rollout ("put the bowl on the plate", init_state 0) with decision-moment states + images.
2. Object t=0 grid (4 scenes x 4 instructions), 3. Goal t=0 grid (3 scenes x 4 instructions).
4. K=32 chunks per cell (one batch, different noise).

    LIBERO_CONFIG_PATH=... CUDA_VISIBLE_DEVICES=0 uv run python scripts/pid_demo/collect.py --out <dir>
Writes <out>/raw/{rollout.npz, rollout_meta.json, grid_object.npz, grid_goal.npz, grids_meta.json, run_meta.json}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import common as C
import numpy as np
import torch

K = 32
GOAL_ROLLOUT_TASK = 8  # "put the bowl on the plate"
GOAL_GRID_INSTR = [8, 9, 0, 7]  # first targets: bowl / wine bottle / middle drawer / stove
GOAL_TARGETS = {
    8: "akita_black_bowl",
    9: "wine_bottle",
    0: "cabinet_middle_drawer_handle",
    7: "stove_knob",
}
GOAL_SWAP_CANDIDATES = [
    ("akita_black_bowl_1", "plate_1"),
    ("akita_black_bowl_1", "wine_bottle_1"),
    ("akita_black_bowl_1", "cream_cheese_1"),
    ("wine_bottle_1", "cream_cheese_1"),
    ("plate_1", "cream_cheese_1"),
    ("plate_1", "wine_bottle_1"),
]


def log(*a):
    print("[pid-demo]", *a, flush=True)


# ---------------------------------------------------------------- 1. rollout
def run_rollout(policy, pre, post, env_pre, out: Path, seed: int = 0) -> dict:
    env = C.make_env("libero_goal", GOAL_ROLLOUT_TASK, 0)
    obs, _ = env.reset()
    task = env.task_description
    policy.reset()
    torch.manual_seed(seed)
    frames, states, eefs, grip_cmd = [], [], [], []
    success, step = False, 0
    max_steps = env._max_episode_steps
    while step < max_steps:
        frames.append((obs["pixels"]["image"].copy(), obs["pixels"]["image2"].copy()))
        states.append(C.sim_state(env))
        eefs.append(C.eef_world(env))
        batch = C.obs_to_batch(obs, task, env_pre, pre)
        with torch.inference_mode():
            action = post(policy.select_action(batch))
        a = action.float().cpu().numpy()[0]
        grip_cmd.append(float(a[6]))
        obs, _, terminated, _, info = env.step(a)
        step += 1
        if info["is_success"]:
            success = True
            break
        if terminated:
            break
    # decision moments
    g = np.sign(np.array(grip_cmd))
    flips = [t for t in range(1, len(g)) if g[t] != g[t - 1] and g[t] != 0]
    first_grasp = next((t for t in flips if g[t] > 0), None)
    moments: list[tuple[int, str]] = [(0, "t=0")]
    end_pre = first_grasp if first_grasp is not None else len(g)
    moments += [(t, f"pre-grasp t={t}") for t in range(10, end_pre, 10)]
    moments += [(t, f"gripper {'close' if g[t] > 0 else 'open'} t={t}") for t in flips]
    if first_grasp is not None:
        moments += [
            (first_grasp + dt, f"grasp+{dt} t={first_grasp + dt}")
            for dt in (20, 40)
            if first_grasp + dt < len(frames)
        ]
    moments = sorted(dict(moments).items())
    idx = [t for t, _ in moments]
    np.savez_compressed(
        out / "rollout.npz",
        steps=np.array(idx),
        agentview=np.stack([frames[t][0] for t in idx]),
        wrist=np.stack([frames[t][1] for t in idx]),
        mj_state=np.stack([states[t] for t in idx]),
        eef_world=np.stack([eefs[t] for t in idx]),
        grip_cmd_all=np.array(grip_cmd),
    )
    meta = {
        "suite": "libero_goal",
        "task_id": GOAL_ROLLOUT_TASK,
        "instruction": task,
        "init_state_id": 0,
        "seed": seed,
        "n_steps": step,
        "max_steps": max_steps,
        "success": success,
        "first_grasp_step": first_grasp,
        "gripper_flip_steps": flips,
        "moments": [{"step": t, "label": lab} for t, lab in moments],
    }
    (out / "rollout_meta.json").write_text(json.dumps(meta, indent=1))
    env.close()
    log("rollout", meta["instruction"], "success", success, "steps", step, "moments", idx)
    return meta


# ---------------------------------------------------------------- scenes
def scene_record(env, obs, label: str) -> dict:
    cands = C.candidate_positions_world(env)
    return {
        "label": label,
        "agentview": obs["pixels"]["image"].copy(),
        "wrist": obs["pixels"]["image2"].copy(),
        "eef_world": C.eef_world(env),
        "eef_base": C.to_base(env, C.eef_world(env)),
        "cand_names": list(cands),
        "cand_world": np.stack(list(cands.values())),
        "cand_base": C.to_base(env, np.stack(list(cands.values()))),
        "cam": C.camera_matrix(env),
        "base_rot": C.base_frame(env)[1],
        "mj_state": C.sim_state(env),
        "obs": obs,
    }


def object_scenes() -> tuple[list[int], dict]:
    """Pick 4 Object tasks (distinct targets) maximizing cells whose instruction target is in the scene."""
    objs, targets = {}, {}
    for tid in range(10):
        env = C.make_env("libero_object", tid, 0)
        env._ensure_env()
        e = env._env.env
        objs[tid] = sorted(n.removesuffix("_1") for n in e.objects_dict)
        targets[tid] = e.obj_of_interest[0].removesuffix("_1")
        env.close()
    best = max(
        itertools.combinations(range(10), 4),
        key=lambda s: (sum(targets[i] in objs[j] for i in s for j in s), [-x for x in s]),
    )
    return list(best), {"objects": objs, "targets": targets}


def build_object_grid() -> tuple[list[dict], list[dict], dict]:
    tids, info = object_scenes()
    scenes, instrs = [], []
    for j in tids:
        env = C.make_env("libero_object", j, 0)
        obs, _ = env.reset()  # init_state 0 + 10 no-op steps
        scenes.append(scene_record(env, obs, f"S{len(scenes)}: object task {j} layout"))
        scenes[-1]["task_id"] = j
        instrs.append({"task_id": j, "text": env.task_description, "target": info["targets"][j]})
        env.close()
    log(
        "object tasks",
        tids,
        "valid cells",
        sum(i["target"] in s["cand_names"] for i in instrs for s in scenes),
    )
    return scenes, instrs, info


def free_joint_addr(m, body: str) -> int:
    return m.jnt_qposadr[m.joint_name2id(f"{body}_joint0")]


def try_swap(a: str, b: str, settle: int = 10, tol_xy: float = 0.02, tol_z: float = 0.01):
    env = C.make_env("libero_goal", GOAL_ROLLOUT_TASK, 0)
    env.reset()
    e, m, d = C.sim_handles(env)
    movable = list(e.objects_dict)
    before = {n: d.qpos[free_joint_addr(m, n) : free_joint_addr(m, n) + 7].copy() for n in movable}
    if a is not None:
        ia, ib = free_joint_addr(m, a), free_joint_addr(m, b)
        xa, xb = d.qpos[ia : ia + 2].copy(), d.qpos[ib : ib + 2].copy()
        d.qpos[ia : ia + 2], d.qpos[ib : ib + 2] = xb, xa
        d.qvel[:] = 0
        e.sim.forward()
    intended = {n: d.qpos[free_joint_addr(m, n) : free_joint_addr(m, n) + 7].copy() for n in movable}
    obs = C.dummy_steps(env, settle)
    after = {n: d.qpos[free_joint_addr(m, n) : free_joint_addr(m, n) + 7].copy() for n in movable}
    report = {}
    ok = True
    for n in movable:
        dxy = float(np.linalg.norm(after[n][:2] - intended[n][:2]))
        dz = float(after[n][2] - before[n][2])
        report[n] = {"dxy": round(dxy, 4), "dz": round(dz, 4)}
        ok &= dxy < tol_xy and abs(dz) < tol_z
    # object-object contacts with penetration
    body_of = {m.body_rootid[e.obj_body_id[n]]: n for n in movable}
    pen = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.body_rootid[m.geom_bodyid[c.geom1]], m.body_rootid[m.geom_bodyid[c.geom2]]
        if b1 in body_of and b2 in body_of and b1 != b2 and c.dist < -0.002:
            pen.append((body_of[b1], body_of[b2], round(float(c.dist), 4)))
    ok &= not pen
    return env, obs, ok, {"settle_report": report, "penetrations": pen}


def build_goal_grid() -> tuple[list[dict], list[dict], dict]:
    scenes, checks = [], []
    env, obs, ok, rep = try_swap(None, None)
    scenes.append(scene_record(env, obs, "S0: original"))
    checks.append({"swap": None, "ok": ok, **rep})
    env.close()
    for a, b in GOAL_SWAP_CANDIDATES:
        if len(scenes) == 3:
            break
        env, obs, ok, rep = try_swap(a, b)
        checks.append({"swap": [a, b], "ok": ok, **rep})
        if ok:
            short = lambda n: n.removesuffix("_1").replace("akita_black_", "")  # noqa: E731
            scenes.append(scene_record(env, obs, f"S{len(scenes)}: swap {short(a)}<->{short(b)}"))
        env.close()
        log("swap", a, b, "ok" if ok else "REJECTED", rep)
    suite = C._get_suite("libero_goal")
    instrs = [
        {"task_id": t, "text": suite.get_task(t).language, "target": GOAL_TARGETS[t]} for t in GOAL_GRID_INSTR
    ]
    return scenes, instrs, {"swap_checks": checks}


# ---------------------------------------------------------------- sampling
def sample_grid(name, scenes, instrs, policy, pre, post, env_pre, out: Path, base_seed: int) -> dict:
    nV, nL = len(scenes), len(instrs)
    included = np.array([[ins["target"] in sc["cand_names"] for ins in instrs] for sc in scenes])
    actions = np.full((nV, nL, K, 50, 7), np.nan, dtype=np.float32)
    t_policy = 0.0
    for v, sc in enumerate(scenes):
        for li, ins in enumerate(instrs):
            if not included[v, li]:
                continue
            batch = C.obs_to_batch(sc["obs"], ins["text"], env_pre, pre, repeat=K)
            torch.cuda.synchronize()
            t0 = time.time()
            actions[v, li] = C.sample_chunks(policy, post, batch, seed=base_seed + 100 * v + li)
            torch.cuda.synchronize()
            t_policy += time.time() - t0
            log(name, "cell", v, li, "done")
    ncand = max(len(s["cand_names"]) for s in scenes)

    def pad(key):
        arr = np.full((nV, ncand, 3), np.nan)
        for v, s in enumerate(scenes):
            arr[v, : len(s[key])] = s[key]
        return arr

    np.savez_compressed(
        out / f"grid_{name}.npz",
        actions=actions,
        included=included,
        agentview=np.stack([s["agentview"] for s in scenes]),
        wrist=np.stack([s["wrist"] for s in scenes]),
        eef_world=np.stack([s["eef_world"] for s in scenes]),
        eef_base=np.stack([s["eef_base"] for s in scenes]),
        cand_world=pad("cand_world"),
        cand_base=pad("cand_base"),
        cam=np.stack([s["cam"] for s in scenes]),
        base_rot=np.stack([s["base_rot"] for s in scenes]),
        mj_state=np.stack([s["mj_state"] for s in scenes]),
    )
    return {
        "scenes": [
            {"label": s["label"], "cand_names": s["cand_names"], "task_id": s.get("task_id")} for s in scenes
        ],
        "instructions": instrs,
        "included": included.tolist(),
        "policy_seconds": t_policy,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    policy, pre, post, env_pre, ckpt_cfg = C.load_policy()
    t_loaded = time.time()
    log("policy loaded", ckpt_cfg)

    rollout = run_rollout(policy, pre, post, env_pre, out)
    t_rollout = time.time()

    meta = {}
    o_scenes, o_instrs, o_info = build_object_grid()
    meta["object"] = sample_grid("object", o_scenes, o_instrs, policy, pre, post, env_pre, out, 1000)
    meta["object"]["selection"] = o_info
    g_scenes, g_instrs, g_info = build_goal_grid()
    meta["goal"] = sample_grid("goal", g_scenes, g_instrs, policy, pre, post, env_pre, out, 2000)
    meta["goal"].update(g_info)
    (out / "grids_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    t_end = time.time()
    run = {
        "checkpoint": C.CKPT,
        "checkpoint_config": ckpt_cfg,
        "K": K,
        "obs_size": C.OBS_SIZE,
        "gpu": torch.cuda.get_device_name(0),
        "wall_seconds": {
            "load": t_loaded - t_start,
            "rollout": t_rollout - t_loaded,
            "grids": t_end - t_rollout,
            "total": t_end - t_start,
        },
        "rollout_success": rollout["success"],
    }
    (out / "run_meta.json").write_text(json.dumps(run, indent=1, default=str))
    log("done", run)


if __name__ == "__main__":
    main()
