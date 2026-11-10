"""T019: render the clean (unperturbed) tasks of LIBERO-Plus / LIBERO-PRO in their own venv, like T008/T009 did
for our env, so their first frames can be compared with the training data (bench_render_compare.py).

    CUDA_VISIBLE_DEVICES= MUJOCO_EGL_DEVICE_ID=0 \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
    $LIBERO_PLUS_VENV/bin/python scripts/shortcut/analysis/bench_render.py --bench plus --suite libero_object \
        --inits 0-4 --cache /outputs/VLA-shortcut-exps/analysis/bench_render/plus          # CPU render (mesa)

Same episode start as each benchmark's worker: Plus `PlusEnv(original_variant)`, PRO `ProEnv(task_entry(axis none))`
(both: seed → reset → set_init_state → 10 no-op settle steps), 256×256, rotated 180° to the training orientation.
Per task → <cache>/render/<suite>_t<id>.npz with the T008 fields: img (n,256,256,3) upright, eef (n,3),
obj (n,K,3) movable-object positions after settle, obj_names (K,).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[1]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))


def parse_range(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def make_env(bench: str, suite: str, task: int, hw: int):  # noqa: ANN201
    if bench == "plus":
        from bench.libero_plus import lp_env
        from bench.libero_plus.registry import original_variant

        env = lp_env.PlusEnv(original_variant(suite, task), hw=hw)
        return env, env.env.env  # PlusEnv → OffScreenRenderEnv → problem env
    from bench.libero_pro import catalog, sim

    entry = catalog.task_entry(catalog.pro_root(), suite, "none", "original", task)
    pro_env = sim.ProEnv(entry, hw=hw)
    return pro_env, pro_env.env.env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", choices=("plus", "pro"), required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--inits", default="0-4")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--hw", type=int, default=256)
    args = ap.parse_args()
    out = args.cache / "render"
    out.mkdir(parents=True, exist_ok=True)
    for t in parse_range(args.tasks):
        env, inner = make_env(args.bench, args.suite, t, args.hw)
        imgs, eefs, objs = [], [], []
        names = sorted(inner.objects_dict)
        for i in parse_range(args.inits):
            obs = env.reset(i, i)
            imgs.append(np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1]))
            eefs.append(np.asarray(obs["robot0_eef_pos"], np.float32))
            objs.append(
                np.stack([inner.sim.data.body_xpos[inner.obj_body_id[o]] for o in names]).astype(np.float32)
            )
        np.savez_compressed(
            out / f"{args.suite}_t{t}.npz",
            img=np.stack(imgs),
            eef=np.stack(eefs),
            obj=np.stack(objs),
            obj_names=np.array(names),
            inits=np.array(parse_range(args.inits)),
        )
        print(args.bench, args.suite, t, len(imgs), flush=True)
        env.close()


if __name__ == "__main__":
    main()
