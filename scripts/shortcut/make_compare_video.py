"""Side-by-side comparison of several conditions at the same (task, init): mp4 + a GIF ≤ 3 MB.

    uv run python scripts/shortcut/make_compare_video.py RUN_DIR [RUN_DIR ...] --suite libero_goal --task 8 \
        --init 0 --conds C0,C4,C5 --out results/compare/goal_t08_init00

Clips are found through episodes.jsonl `video` paths. `--conds` takes conditions (all their variants) or
`cond/variant` items (e.g. C4/C4-a). `--model` restricts to one model tag when several runs are given.
`--views agent` (default) keeps only the agentview half of each clip so many conditions fit in one row.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import load_episodes  # noqa: E402

GIF_MAX_BYTES = 3 * 1024 * 1024
FPS = 10


def read_clip(path: Path, views: str) -> list[np.ndarray]:
    frames = [np.asarray(f) for f in imageio.mimread(path, memtest=False)]
    if views == "agent":
        frames = [f[:, : f.shape[1] // 2] for f in frames]
    return frames


def hstack_clips(clips: list[list[np.ndarray]], gap: int = 4) -> list[np.ndarray]:
    n = max(len(c) for c in clips)
    h = max(c[0].shape[0] for c in clips)
    out = []
    for t in range(n):
        parts = []
        for i, c in enumerate(clips):
            f = c[min(t, len(c) - 1)]  # shorter (early-stopped) clips hold their last frame
            if f.shape[0] < h:
                f = np.pad(f, ((0, h - f.shape[0]), (0, 0), (0, 0)))
            parts.append(f)
            if i < len(clips) - 1:
                parts.append(np.full((h, gap, 3), 255, np.uint8))
        frame = np.concatenate(parts, axis=1)
        pad = -frame.shape[1] % 8  # libx264 macro blocks
        out.append(np.pad(frame, ((0, 0), (0, pad), (0, 0)), constant_values=255) if pad else frame)
    return out


def make_gif(mp4: Path, gif: Path, width: int) -> None:
    """Palette GIF, shrinking width / fps until it is under GIF_MAX_BYTES."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for scale, fps in [(1.0, 10), (0.75, 10), (0.75, 7), (0.5, 7), (0.5, 5), (0.35, 5), (0.25, 4)]:
        w = int(width * scale) // 2 * 2
        vf = f"fps={fps},scale={w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(mp4), "-vf", vf, str(gif)], check=True)  # nosec B603
        if gif.stat().st_size <= GIF_MAX_BYTES:
            print(f"gif {gif} {gif.stat().st_size / 1e6:.2f} MB (width {w}, {fps} fps)")
            return
    print(f"[warn] gif still {gif.stat().st_size / 1e6:.2f} MB > 3 MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs / episodes.jsonl / globs")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--init", type=int, required=True)
    ap.add_argument("--conds", default="C0,C1,C2,C3,C4,C5,C6")
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeat-idx", type=int, default=0)
    ap.add_argument("--views", choices=["agent", "both"], default="agent")
    ap.add_argument("--out", required=True, help="output path without extension")
    args = ap.parse_args()

    eps = [
        r
        for r in load_episodes(args.runs)
        if r["suite"] == args.suite
        and r["task_id"] == args.task
        and r["init_idx"] == args.init
        and r.get("repeat_idx", 0) == args.repeat_idx
        and r.get("video")
        and (args.model is None or r["model"] == args.model)
    ]
    picked = []
    for item in args.conds.split(","):
        cond, _, var = item.partition("/")
        for r in sorted(eps, key=lambda r: (r["model"], r["variant"])):
            if r["condition"] == cond and (not var or r["variant"] == var):
                picked.append(r)
    if not picked:
        raise SystemExit("no clips match")
    models = {r["model"] for r in picked}
    clips = [read_clip(Path(r["_root"]) / r["video"], args.views) for r in picked]
    frames = hstack_clips(clips)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mp4 = out.with_suffix(".mp4")
    imageio.mimwrite(mp4, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    make_gif(mp4, out.with_suffix(".gif"), frames[0].shape[1])
    manifest = [
        {
            k: r.get(k)
            for k in ("model", "condition", "variant", "instruction", "success_orig", "success_new", "video")
        }
        for r in picked
    ]
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(
        f"{mp4}: {len(picked)} clips ({', '.join(f'{r["condition"]}/{r["variant"]}' for r in picked)}), models {models}"
    )


if __name__ == "__main__":
    main()
