"""T009: compare eval init renders before / after the LIBERO env fix, and both against the training first frames.

    PYTHONPATH=src:scripts/shortcut python scripts/shortcut/analysis/env_fix_compare.py \
        --before /outputs/VLA-shortcut-exps/data/hf_libero_viz_cache \
        --after /outputs/VLA-shortcut-exps/data/hf_libero_viz_cache_T009 \
        --out /outputs/VLA-shortcut-exps/analysis/env_fix

Per task (render/<suite>_t<id>.npz from train_data_render.py, 50 init states each):
  pixel diff after vs before (max |Δ|, fraction of pixels changed) — Spatial/Goal must be identical;
  movable-object position change after vs before;
  mean colour distance of the renders to the task's training first frames (64 px, RGB L2 per pixel);
  objects more than 0.3 m above the median object height of their init state (thrown off the table; the
  thrown ones sit 0.4-0.6 m up, objects on a cabinet or stove top 0.1-0.2 m);
  match = no such object and the mean render is within 2x the training frames' own spread of the mean
  training first frame (train_dist <= 2 * train_self_spread; the dark Object floor is ~20x). The figures
  are the check by eye.
Writes env_fix_compare.csv and before_after_<suite>.png (train 2 | before 2 | after 2 per task).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def small(img: np.ndarray, side: int = 64) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((side, side), Image.BILINEAR)).astype(np.float32)


def airborne(obj: np.ndarray) -> int:
    """Objects (n_init, K, 3) more than 0.3 m above the median object height of their init state."""
    z = obj[..., 2]
    return int((z > np.median(z, axis=1, keepdims=True) + 0.3).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ep = pd.read_parquet(args.before / "episodes.parquet")
    first = np.load(args.before / "first256.npz")["first"]
    rows = []
    tiles: dict[str, list] = {s: [] for s in SUITES}
    for s in SUITES:
        for t in range(10):
            b = np.load(args.before / "render" / f"{s}_t{t}.npz")
            a = np.load(args.after / "render" / f"{s}_t{t}.npz")
            d = np.abs(a["img"].astype(np.int16) - b["img"].astype(np.int16))
            es = ep[(ep.suite == s) & (ep.suite_task_id == t)].episode_index.to_numpy()
            tr = np.stack([small(first[e]) for e in es])
            tr_mean = tr.mean(0)

            def dist(imgs: np.ndarray, tr_mean: np.ndarray = tr_mean) -> float:
                """Per-pixel RGB L2 distance of the mean render to the mean training first frame."""
                m = np.stack([small(i) for i in imgs]).mean(0)
                return float(np.linalg.norm(m - tr_mean, axis=-1).mean())

            # spread of the training frames around their own mean, for scale
            tr_self = float(np.linalg.norm(tr - tr_mean, axis=-1).mean())
            names = list(b["obj_names"])
            assert names == list(a["obj_names"])
            dpos = np.linalg.norm(a["obj"] - b["obj"], axis=-1)
            rows.append(
                {
                    "suite": s,
                    "task_id": t,
                    "n_init": len(a["img"]),
                    "px_max_absdiff": int(d.max()),
                    "px_frac_changed": float((d.max(-1) > 0).mean()),
                    "obj_max_shift_m": float(dpos.max()),
                    "n_obj_airborne_before": airborne(b["obj"]),
                    "n_obj_airborne_after": airborne(a["obj"]),
                    "train_dist_before": dist(b["img"]),
                    "train_dist_after": dist(a["img"]),
                    "train_self_spread": tr_self,
                }
            )
            tiles[s].append((t, [first[e] for e in es[:2]], b["img"][:2], a["img"][:2]))
    df = pd.DataFrame(rows)
    for k in ("before", "after"):
        df[f"match_{k}"] = (df[f"n_obj_airborne_{k}"] == 0) & (
            df[f"train_dist_{k}"] <= 2 * df.train_self_spread
        )
    print(df.groupby("suite")[["match_before", "match_after"]].sum().to_string())
    df.to_csv(args.out / "env_fix_compare.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(df.round(4).to_string(index=False))
    for s in SUITES:
        tw, gap, left = 128, 3, 40
        width = left + 6 * (tw + gap) + 2 * 8
        height = 24 + 10 * (tw + gap)
        c = Image.new("RGB", (width, height), (252, 252, 251))
        dr = ImageDraw.Draw(c)
        dr.text(
            (4, 4),
            f"{s}: training first frames (2) | eval init before fix (2) | after fix (2)",
            fill=(11, 11, 11),
        )
        for t, tr, bf, af in tiles[s]:
            y = 24 + t * (tw + gap)
            dr.text((4, y + tw // 2), f"t{t}", fill=(11, 11, 11))
            for j, img in enumerate([*tr, *bf, *af]):
                x = left + j * (tw + gap) + 8 * (j // 2)
                c.paste(Image.fromarray(np.asarray(img)).resize((tw, tw)), (x, y))
        c.save(args.out / f"before_after_{s}.png")


if __name__ == "__main__":
    main()
