"""T022 basket distance: copy the 4090 run's videos into the results tree and render attention overlay videos, same
format as basket/4_tasks (libero3 / cand31 overlay). Run with the lerobot venv (cv2, imageio):

  OPENBLAS_CORETYPE=ZEN $W/lerobot/.venv/bin/python basket_far_package.py [--out /outputs/.../basket/far] [--workers 8]

  4_tasks/O<k>/{far20,far25}_iNN_<success|fail>_{eval,attention}.mp4   moved-basket episodes
  4_tasks/O<k>/ctl4090_iNN_<success|fail>_eval.mp4                       the paired original-scene rerun on the 4090
"""

import argparse
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np

W = Path("/workspace/VLA-shortcut-exps-kubectl")
RUN = W / "out/runs/bench-pi05lib-basket_far/jobs"
A = W / "basket/far/attn"
TAG = {"ctl_t": "ctl4090", "bsk_c20y21": "far20", "bsk_c25y20": "far25"}
LABEL = {"ctl4090": "original scene (4090 rerun)", "far20": "basket (0.20, 0.21)", "far25": "basket (0.25, 0.20)"}


def jobs():
    for job in ("ours-obj-custom-train10", "ours-obj-custom-basket"):
        for line in (RUN / job / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            tag = next(v for k, v in TAG.items() if e["variant"].startswith(k))
            yield job, tag, e


def one(item):
    job, tag, e, out = item
    sys.path[:0] = [str(W / "cand31/attn")]
    import attn_overlay as OV

    t, i = e["task_id"], e["init_idx"]
    res = "success" if e["success_new"] else "fail"
    d = out / "4_tasks" / f"O{t + 1}"
    d.mkdir(parents=True, exist_ok=True)
    src = RUN / job / "videos/libero_object" / f"task{t:02d}" / f"custom_{e['variant']}_init{i:02d}.mp4"
    shutil.copy2(src, d / f"{tag}_i{i:02d}_{res}_eval.mp4")
    if tag == "ctl4090":
        return 1
    name = f"{e['variant']}_i{i:02d}"
    a = dict(np.load(A / "out" / job / f"{name}.npz"))
    o = np.load(A / "obs" / job / f"{name}.npz")
    frames = {cam: o[OV.OBS_KEY[cam]] for cam in OV.CAMS}
    OV.video(a, frames, f"O{t + 1} {LABEL[tag]}: {e['instruction'][:60]} | {name} | {res}",
             d / f"{tag}_i{i:02d}_{res}_attention.mp4")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/VLA-shortcut-exps/kubectl/basket/far")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    items = [(j, tag, e, out) for j, tag, e in jobs()]
    with mp.get_context("spawn").Pool(min(a.workers, 8)) as pool:
        n = sum(pool.imap_unordered(one, items, chunksize=4))
    print(f"[package] {n} episodes → {out / '4_tasks'}")


if __name__ == "__main__":
    main()
