# ruff: noqa: N806  (E = episodes, L = length)
"""T008 step 1: one pass over the HuggingFaceVLA/libero parquet files → small analysis cache.

    PYTHONPATH=src python scripts/shortcut/analysis/train_data_extract.py \
        --root /outputs/VLA-shortcut-exps/data/hf_libero --cache /outputs/VLA-shortcut-exps/data/hf_libero_viz_cache

Episode/task metadata comes from `LeRobotDatasetMetadata`. Frames are read straight from the parquet files
(PNG bytes per frame) instead of `LeRobotDataset.__getitem__`, so only the frames we need are decoded:

  episodes.parquet   episode_index, task_index, task, suite, suite_task_id, length, rep (representative demo)
  lowdim.npz         state (N, 8), action (N, 7), ep (N,), frame (N,) for every frame, sorted by (ep, frame)
  progress.npz       agentview at progress fractions FRACS for every episode, 128 px  (E, F, 128, 128, 3)
  first256.npz       agentview frame 0 for every episode, 256 px                       (E, 256, 256, 3)
  last256.npz        agentview last frame for every episode, 256 px                    (E, 256, 256, 3)
  rep/ep{E}.npz      agentview + wrist, every frame, 256 px, for the 40 representative demos

Orientation: the stored images are already upright (robot at the top, table in front), the same view as our
eval videos, which flip the raw MuJoCo render by 180° (`run_eval.compose_frame`). No flip is applied here.
"""

from __future__ import annotations

import argparse
import io
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

FRACS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SMALL = 128
TRAIN_TASKS = "/outputs/VLA-shortcut-exps/analysis/p1_train_overlap/train_tasks.csv"

_WANT: dict[
    int, dict[int, tuple[bool, bool, bool, int]]
] = {}  # ep → frame → (small, first/last 256, rep, frac idx)


def episode_table(root: Path) -> pd.DataFrame:
    meta = LeRobotDatasetMetadata("HuggingFaceVLA/libero", root=root)
    eps = meta.episodes.to_pandas()
    tasks = meta.tasks.reset_index()  # task, task_index
    tt = pd.read_csv(TRAIN_TASKS)[["instruction", "suite", "task_id", "scene_id"]]
    df = eps[["episode_index", "tasks", "length", "dataset_from_index"]].copy()
    df["task"] = df["tasks"].map(lambda t: t[0])
    df = df.drop(columns="tasks").merge(tasks, on="task", how="left")
    df = df.merge(
        tt.rename(columns={"instruction": "task", "task_id": "suite_task_id"}), on="task", how="left"
    )
    assert df["suite"].notna().all() and len(df) == meta.total_episodes
    med = df.groupby("task")["length"].transform("median")
    df["dist_med"] = (df["length"] - med).abs()
    rep = df.sort_values(["dist_med", "episode_index"]).groupby("task").head(1)["episode_index"]
    df["rep"] = df["episode_index"].isin(set(rep))
    return df.drop(columns="dist_med")


def plan(df: pd.DataFrame) -> dict[int, dict[int, tuple[bool, bool, bool, int]]]:
    want: dict[int, dict[int, tuple[bool, bool, bool, int]]] = {}
    for r in df.itertuples():
        L = int(r.length)
        w: dict[int, tuple[bool, bool, bool, int]] = {}
        for i, f in enumerate(FRACS):
            w[round(f * (L - 1))] = (True, f in (0.0, 1.0), bool(r.rep), i)
        if r.rep:
            for t in range(L):
                w.setdefault(t, (False, False, True, -1))
        want[int(r.episode_index)] = w
    return want


def _decode(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def work(path: str) -> dict:
    t = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"])
    ep = t["episode_index"].to_numpy()
    fr = t["frame_index"].to_numpy()
    state = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
    action = np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
    rows = [i for i in range(len(ep)) if int(fr[i]) in _WANT.get(int(ep[i]), {})]
    out: dict = {"ep": ep, "frame": fr, "state": state, "action": action, "imgs": {}}
    if rows:
        img = pq.read_table(path, columns=["observation.images.image", "observation.images.image2"])
        a_col, w_col = img["observation.images.image"], img["observation.images.image2"]
        for i in rows:
            e, f = int(ep[i]), int(fr[i])
            small, big, rep, fi = _WANT[e][f]
            a = _decode(a_col[i]["bytes"].as_py())
            rec: dict = {}
            if small:
                rec["small"] = (fi, np.asarray(Image.fromarray(a).resize((SMALL, SMALL), Image.BILINEAR)))
            if big:
                rec["big"] = a
            if rep:
                rec["rep"] = (a, _decode(w_col[i]["bytes"].as_py()))
            out["imgs"][(e, f)] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    (args.cache / "rep").mkdir(exist_ok=True)

    df = episode_table(args.root)
    df.to_parquet(args.cache / "episodes.parquet")
    global _WANT
    _WANT = plan(df)
    files = sorted(str(p) for p in (args.root / "data").rglob("*.parquet"))
    print(f"{len(df)} episodes, {len(files)} parquet files")

    E = int(df["episode_index"].max()) + 1
    lengths = dict(zip(df["episode_index"], df["length"], strict=True))
    progress = np.zeros((E, len(FRACS), SMALL, SMALL, 3), np.uint8)
    first = np.zeros((E, 256, 256, 3), np.uint8)
    last = np.zeros((E, 256, 256, 3), np.uint8)
    rep_frames: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    low = {k: [] for k in ("ep", "frame", "state", "action")}
    seen = np.zeros(E, int)
    with Pool(args.workers) as pool:
        for n, res in enumerate(pool.imap_unordered(work, files)):
            for k in low:
                low[k].append(res[k])
            for (e, f), rec in res["imgs"].items():
                if "small" in rec:
                    progress[e, rec["small"][0]] = rec["small"][1]
                    seen[e] += 1
                if "big" in rec:
                    if f == 0:
                        first[e] = rec["big"]
                    if f == lengths[e] - 1:
                        last[e] = rec["big"]
                if "rep" in rec:
                    rep_frames.setdefault(e, {})[f] = rec["rep"]
            if n % 20 == 0:
                print(f"  {n + 1}/{len(files)}", flush=True)
    lowc = {k: np.concatenate(v) for k, v in low.items()}
    order = np.lexsort((lowc["frame"], lowc["ep"]))
    lowc = {k: v[order] for k, v in lowc.items()}
    assert len(lowc["ep"]) == int(df["length"].sum()), "frame count mismatch"
    n_expected = np.array([len({round(f * (lengths[e] - 1)) for f in FRACS}) for e in range(E)])
    assert (seen == n_expected).all(), "missing progress frames"
    np.savez(args.cache / "lowdim.npz", **lowc)
    np.savez(args.cache / "progress.npz", progress=progress, fracs=np.array(FRACS))
    np.savez(args.cache / "first256.npz", first=first)
    np.savez(args.cache / "last256.npz", last=last)
    for e, fr in rep_frames.items():
        L = lengths[e]
        assert len(fr) == L, (e, len(fr), L)
        np.savez_compressed(
            args.cache / "rep" / f"ep{e}.npz",
            agent=np.stack([fr[t][0] for t in range(L)]),
            wrist=np.stack([fr[t][1] for t in range(L)]),
        )
    (args.cache / "extract_meta.json").write_text(
        json.dumps(
            {"fracs": FRACS, "small": SMALL, "n_files": len(files), "n_frames": len(lowc["ep"])}, indent=1
        )
    )
    print("done")


if __name__ == "__main__":
    main()
