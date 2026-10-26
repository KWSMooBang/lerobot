# ruff: noqa: N806  (P = camera matrix)
"""T008 step 2 (sim, GPU 0 only): agentview camera matrix per task + renders of the eval init states.

    CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src:scripts/shortcut \
        python scripts/shortcut/analysis/train_data_render.py --suite libero_goal \
        --cache /outputs/VLA-shortcut-exps/data/hf_libero_viz_cache

Same env path as the eval harness (`LiberoEnv`, hard reset, set_init_state, 10 no-op settle steps), 256×256 like
the training images. Renders are flipped 180° (`[::-1, ::-1]`) like `run_eval.compose_frame`, which gives the
orientation of the stored training images. Per task → render/<suite>_t<id>.npz:

  img (n_init, 256, 256, 3)   agentview after settle, upright
  eef (n_init, 3)             robot0_eef_pos (world frame, same frame as observation.state[:3])
  obj (n_init, K, 3), obj_names (K,)   movable objects' body positions after settle
  P   (4, 4)                  world → pixel matrix of the raw (unflipped) 256×256 render (robosuite camera_utils)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
from common import ensure_libero_config  # noqa: E402  (sets the NVIDIA EGL vendor before EGL loads)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--tasks", default="0-9")
    args = ap.parse_args()
    ensure_libero_config()
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    from lerobot.envs.libero import LiberoEnv, _get_suite

    out = args.cache / "render"
    out.mkdir(parents=True, exist_ok=True)
    suite = _get_suite(args.suite)
    a, b = (int(x) for x in args.tasks.split("-"))
    for tid in range(a, b + 1):
        env = LiberoEnv(
            suite, tid, args.suite, obs_type="pixels_agent_pos", observation_height=256, observation_width=256
        )
        n = len(env._init_states)
        imgs, eefs, objs = [], [], []
        P = None
        for i in range(n):
            env.init_state_id = i
            obs, _ = env.reset(seed=i)
            inner = env._env.env
            raw = inner._get_observations()
            imgs.append(np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1]))
            eefs.append(np.asarray(raw["robot0_eef_pos"], np.float32))
            names = sorted(inner.objects_dict)
            objs.append(
                np.stack([inner.sim.data.body_xpos[inner.obj_body_id[o]] for o in names]).astype(np.float32)
            )
            if P is None:
                P = get_camera_transform_matrix(inner.sim, "agentview", 256, 256)
        np.savez_compressed(
            out / f"{args.suite}_t{tid}.npz",
            img=np.stack(imgs),
            eef=np.stack(eefs),
            P=P,
            obj=np.stack(objs),
            obj_names=np.array(names),
        )
        print(args.suite, tid, n, flush=True)
        env.close()


if __name__ == "__main__":
    main()
