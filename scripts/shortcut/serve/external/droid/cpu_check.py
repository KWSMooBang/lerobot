"""CPU check of the DROID server on a real LIBERO observation (T017 §5c; no GPU, ~17 GB RAM, minutes).

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES= uv run --extra shortcut-bench \
        python scripts/shortcut/serve/external/droid/cpu_check.py [--suite libero_goal --task 1] [--out F.json]

Renders the first observation of the joint_velocity env (joint_env.JointVelocityLiberoEnv, init 0) and builds the
wire obs as bench/ours/worker.py does, runs the DROID engine (lerobot/pi05_droid fp32 on CPU, pinned stats) for
one chunk, and checks: 15×8 raw actions finite, the 8 served env actions finite and in [-1, 1], gripper ±1. Then
executes the served chunk in the env and reports the joint motion (direction vs the commanded velocities).
Writes the prompt, the state bins, the raw / served actions and the joint motion to --out (JSON) and the two
model input images next to it (PNG).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parents[2]
sys.path[:0] = [
    p
    for p in (str(HERE), str(SHORTCUT), str(SHORTCUT / "serve"), str(SHORTCUT / "parallel"))
    if p not in sys.path
]

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7_000_101)
    ap.add_argument("--out", default="droid_cpu_check.json")
    args = ap.parse_args()

    import common

    common.ensure_libero_config()
    import contract
    import server
    import torch
    from bench.ours.worker import wire_obs
    from envproc import _make_env
    from PIL import Image

    torch.set_num_threads(min(32, torch.get_num_threads()))
    spec = {
        "suite": args.suite, "task_id": args.task, "max_steps": 50, "env_hw": [256, 256],
        "control_mode": "relative", "action_space": "joint_velocity",
    }  # fmt: skip
    env = _make_env("libero", spec, {})
    env.init_state_id = 0
    obs, _ = env.reset(seed=args.seed)
    wire = wire_obs(obs)
    instruction = env.task_description
    tic = time.time()
    eng = server.build_engine(server.build_parser().parse_args(["--device", "cpu", "--ckpt-id", "pi05droid"]))
    t_load = time.time() - tic
    model = eng.model
    pre_in = model.batch(wire, instruction)
    pre = model.pre(pre_in)
    prompt = model.pre.steps[-2].input_tokenizer.decode(
        pre["observation.language.tokens"][0][pre["observation.language.attention_mask"][0].bool()]
    )
    eng.reset("e0", args.seed, instruction)
    tic = time.time()
    raw = model.chunk(wire, instruction)
    t_infer = time.time() - tic
    eng.reset("e0", args.seed, instruction)
    served = eng.infer_one("e0", wire, instruction, step=False)
    q0 = obs["robot_state"]["joints"]["pos"].copy()
    for a in served:
        obs, *_ = env.step(a)
    dq = obs["robot_state"]["joints"]["pos"] - q0
    commanded = 0.2 * served[:, :7].sum(0)
    moving = np.abs(commanded) > 0.02
    sign_ok = bool(np.all(np.sign(dq[moving]) == np.sign(commanded[moving]))) if moving.any() else None
    checks = {
        "raw_shape": list(raw.shape),
        "raw_finite": bool(np.all(np.isfinite(raw))),
        "served_shape": list(served.shape),
        "served_finite": bool(np.all(np.isfinite(served))),
        "served_in_range": bool(np.all(np.abs(served) <= 1.0)),
        "gripper_binary": bool(np.all(np.isin(served[:, 7], (-1.0, 1.0)))),
        "joint_motion_sign_matches_command": sign_ok,
    }
    ok = (
        checks["raw_shape"] == [15, 8]
        and checks["served_shape"] == [8, 8]
        and all(checks[k] for k in ("raw_finite", "served_finite", "served_in_range", "gripper_binary"))
    )
    out = Path(args.out)
    for key, im in contract.droid_images(wire).items():
        Image.fromarray(im).save(out.with_name(f"{out.stem}_{key.split('.')[-1]}.png"))
    rep = {
        "ok": ok,
        "checks": checks,
        "suite": args.suite,
        "task": args.task,
        "instruction": instruction,
        "prompt": prompt,
        "droid_state": contract.droid_state(wire).tolist(),
        "raw_actions": np.round(raw, 4).tolist(),
        "served_actions": np.round(served, 4).tolist(),
        "joint_delta_after_chunk": np.round(dq, 4).tolist(),
        "commanded_joint_delta": np.round(commanded, 4).tolist(),
        "t_load_s": round(t_load, 1),
        "t_infer_cpu_s": round(t_infer, 1),
        "identity": eng.hello["identity"],
    }
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("ok", "checks", "prompt", "t_load_s", "t_infer_cpu_s")}, indent=2))
    print("served chunk:\n", np.round(served, 3))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
