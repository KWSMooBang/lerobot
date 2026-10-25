"""Per-GPU self-test for a new server: EGL renderer, one LIBERO reset + render in env subprocesses, one
batched π0.5 chunk, one env step. Run with the worker's pinning (setup_server.sh does this per GPU):

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 \
        uv run python scripts/shortcut/parallel/selftest.py [--egl-only] [--num-envs 2]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [p for p in (str(HERE), str(HERE.parent)) if p not in sys.path]

import common  # noqa: E402,F401  (EGL vendor fallback runs at import)


def egl_renderer() -> dict[str, str | None]:
    import mujoco

    ctx = mujoco.GLContext(64, 64)
    ctx.make_current()
    from OpenGL import GL

    out = {
        "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
        "vendor": GL.glGetString(GL.GL_VENDOR).decode(),
        "egl_vendor_override": os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
        "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
    }
    ctx.free()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--egl-only", action="store_true")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--ckpt", default="lerobot/pi05_libero_finetuned")
    ap.add_argument("--suite", default="libero_goal")
    a = ap.parse_args()
    report: dict = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    ok = True
    try:
        report["egl"] = egl_renderer()
        if "llvmpipe" in report["egl"]["renderer"].lower() or "mesa" in report["egl"]["vendor"].lower():
            report["egl_problem"] = "software renderer (llvmpipe): rendering will be ~10x slower"
            ok = False
    except Exception as e:
        report["egl_error"] = repr(e)
        ok = False
    if a.egl_only:
        print(json.dumps(report, indent=1))
        return 0 if ok else 1

    import torch
    from batched import BatchedRunner, Slot, Stats
    from bundle import load_bundle
    from common import Variant, episode_seed
    from envproc import EnvClient

    report["torch_gpu"] = torch.cuda.get_device_name(0)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    slots = [Slot(i, EnvClient(ctx, "libero", f"selftest-env{i}")) for i in range(a.num_envs)]
    group = {"backend": "libero", "policy": "pi05", "ckpt": a.ckpt, "adapter": None, "n_action_steps": 10}
    bundle = load_bundle({**group, "suite": a.suite}, "cuda", False)
    report["load_s"] = round(time.time() - t0, 1)
    v = Variant("C0", "orig", "selftest")
    units = [
        {
            "uid": f"selftest/{i}",
            "suite": a.suite,
            "task_id": 0,
            "init_idx": i,
            "variant": {**asdict(v), "instruction": "open the middle drawer of the cabinet"},
            "ep_seed": episode_seed(7, 0, 0, i),
            "max_steps": 20,  # two chunks
            "record_video": False,
            "tag": "selftest",
        }
        for i in range(a.num_envs)
    ]
    results = []
    stats = Stats()
    runner = BatchedRunner(
        bundle,
        slots,
        claim=lambda loc: units.pop() if units else None,
        on_done=lambda u, res, wall: results.append(res["n_steps"]),
        on_fail=lambda u, err: report.setdefault("errors", []).append(err),
        restart=lambda i: slots[i].client,
        stats=stats,
    )
    t0 = time.time()
    runner.run()
    report["episodes"] = results
    report["run_s"] = round(time.time() - t0, 1)
    report["stats"] = stats.line()
    report["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    for s in slots:
        s.client.close()
    ok = ok and len(results) == a.num_envs and "errors" not in report
    print(json.dumps(report, indent=1))
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
