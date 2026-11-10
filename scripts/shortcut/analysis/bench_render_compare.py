"""T019: do LIBERO-Plus / LIBERO-PRO clean-task first frames match the training data? (T009 check, other venvs)

    uv run python scripts/shortcut/analysis/bench_render_compare.py --renders <dir with plus/ pro/> --out <dir>

Inputs: <renders>/{plus,pro}/render/<suite>_t<id>.npz (bench_render.py), the T008 cache (training first frames,
raw-env renders) and the T009 cache (our fixed env), all at the same init indices.
Per (bench, suite, task), with env_fix_compare.py's measures:
  train_dist          mean render vs mean training first frame (64 px, RGB L2 per pixel); train_self_spread for scale
  n_obj_airborne      objects > 0.3 m above the median object height after settle (thrown off the table)
  px_vs_fix / px_vs_raw   mean |Δ| (0–255) to our T009-fixed / raw env render at the same init
  match               T009 rule: no airborne object and train_dist <= 2 × train_self_spread
  like                fix | raw | other: which of our envs the bench render equals (mean |Δ| < 2)
Writes bench_render_compare.csv, summary.md (suite × bench match counts) and tiles_<suite>.png
(training 1 | ours fixed 1 | Plus 1 | PRO 1 per task).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_fix_compare import SUITES, airborne, small  # noqa: E402

T008 = Path("/outputs/VLA-shortcut-exps/data/hf_libero_viz_cache")
T009 = Path("/outputs/VLA-shortcut-exps/data/hf_libero_viz_cache_T009")
BENCHES = ("plus", "pro")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--raw", type=Path, default=T008, help="T008 cache: episodes.parquet, first256.npz, raw renders"
    )
    ap.add_argument("--fix", type=Path, default=T009, help="T009 cache: renders of our fixed env")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ep = pd.read_parquet(args.raw / "episodes.parquet")
    first = np.load(args.raw / "first256.npz")["first"]
    rows: list[dict[str, Any]] = []
    tiles: dict[str, list[Any]] = {s: [] for s in SUITES}
    for s in SUITES:
        for t in range(10):
            es = ep[(ep.suite == s) & (ep.suite_task_id == t)].episode_index.to_numpy()
            tr = np.stack([small(first[e]) for e in es])
            tr_mean = tr.mean(0)
            spread = float(np.linalg.norm(tr - tr_mean, axis=-1).mean())
            fix = np.load(args.fix / "render" / f"{s}_t{t}.npz")
            raw = np.load(args.raw / "render" / f"{s}_t{t}.npz")
            tile = [first[es[0]], fix["img"][0]]
            for b in BENCHES:
                p = args.renders / b / "render" / f"{s}_t{t}.npz"
                if not p.exists():
                    tile.append(None)
                    continue
                r = np.load(p)
                inits = r["inits"] if "inits" in r.files else np.arange(len(r["img"]))
                m = np.stack([small(i) for i in r["img"]]).mean(0)
                d_fix = float(np.abs(r["img"].astype(np.int16) - fix["img"][inits].astype(np.int16)).mean())
                d_raw = float(np.abs(r["img"].astype(np.int16) - raw["img"][inits].astype(np.int16)).mean())
                row: dict[str, Any] = {
                    "bench": b,
                    "suite": s,
                    "task_id": t,
                    "n_init": len(inits),
                    "train_dist": float(np.linalg.norm(m - tr_mean, axis=-1).mean()),
                    "train_self_spread": spread,
                    "n_obj_airborne": airborne(r["obj"]),
                    "n_obj_airborne_fix": airborne(fix["obj"][inits]),
                    "px_vs_fix": d_fix,
                    "px_vs_raw": d_raw,
                }
                row["match"] = row["n_obj_airborne"] == 0 and row["train_dist"] <= 2 * spread
                row["like"] = "fix" if d_fix < 2 else "raw" if d_raw < 2 else "other"
                rows.append(row)
                tile.append(r["img"][0])
            tiles[s].append((t, tile))
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "bench_render_compare.csv", index=False)
    summ = df.groupby(["suite", "bench"]).agg(
        tasks=("task_id", "count"),
        match=("match", "sum"),
        airborne_tasks=("n_obj_airborne", lambda x: int((x > 0).sum())),
        like_fix=("like", lambda x: int((x == "fix").sum())),
        like_raw=("like", lambda x: int((x == "raw").sum())),
        mean_train_dist=("train_dist", "mean"),
        mean_px_vs_fix=("px_vs_fix", "mean"),
    )
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summ.round(2).to_string())
        print(df.round(2).to_string(index=False))
    (args.out / "summary.md").write_text(
        "# Plus / PRO clean-task render vs training first frames (T019)\n\n"
        + "```\n"
        + summ.round(2).to_string()
        + "\n```"
        + "\n\nmatch = T009 rule (no airborne object, train_dist <= 2 × train_self_spread); like = which of our envs "
        "the render equals (mean |Δ| < 2 of 255).\n"
    )
    for s in SUITES:
        tw, gap, left = 128, 3, 40
        c = Image.new("RGB", (left + 4 * (tw + gap), 24 + 10 * (tw + gap)), (252, 252, 251))
        dr = ImageDraw.Draw(c)
        dr.text(
            (4, 4), f"{s}: training | ours (T009 fix) | LIBERO-Plus | LIBERO-PRO  (init 0)", fill=(11, 11, 11)
        )
        for t, imgs in tiles[s]:
            y = 24 + t * (tw + gap)
            dr.text((4, y + tw // 2), f"t{t}", fill=(11, 11, 11))
            for j, img in enumerate(imgs):
                if img is not None:
                    c.paste(Image.fromarray(np.asarray(img)).resize((tw, tw)), (left + j * (tw + gap), y))
        c.save(args.out / f"tiles_{s}.png")


if __name__ == "__main__":
    main()
