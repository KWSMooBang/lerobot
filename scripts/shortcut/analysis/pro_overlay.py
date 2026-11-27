"""T036: attention overlay videos + stills for the LIBERO-PRO evaluation of T023 (CPU only, no rollout, no replay).

Same look as the cand31 overlays (kubectl/cand31/attn/attn_overlay.py, libero_goal_composition/4_tasks/*):
rows agentview / wrist, columns original | early L0–5 | mid L6–11 | late L12–17, label above each panel = that
camera's share of the action queries' attention, one frame per policy call, 3 fps, a title bar.

Inputs (T023 run, default runs/bench-pi05lib-pro_all):
  jobs/<job>/episodes.jsonl   one row per episode: outcome, `video` (relative to the job dir, null = no video),
                              `activations` (npz path)
  activations/<job>/<suite>/<task>/<variant>_init<NN>.npz   written by serve/activations.py during the run:
      attn_img  (C, 3, 512) f16  action queries' attention on the 2 × 256 image keys per layer group, mean over the
                                 50 action queries, 8 heads, 10 denoising steps and the group's 6 layers. Keys
                                 0–255 = observation.images.image (agentview), 256–511 = image2 (wrist); each is the
                                 SigLIP 16 × 16 patch grid, row-major. Raw attention probability (no renormalisation).
      attn_mass (C, 10, 18, 7)   share per input type; camera share label = mean over the 10 denoising steps and the
                                 group's layers of attn_mass[..., cam] (== attn_img[c, g, cam].sum() up to f16 rounding)
      t         (C,)             env step of each call (call index × 10)
  jobs/<job>/videos/<suite>/<task>/<variant>_init<NN>.mp4   512 × 256 (agentview | wrist), 10 fps. Frames are the
      same upright() (180° rotated raw render) images that went to the policy (bench/libero_pro/sim.py, client.py),
      written at t = 0, 2, 4, …; then the final step, then 10 hold frames. So the frame for call c is index t[c] // 2.
      The eval header text (dimmed box on the top rows) is baked into these frames.

Outputs (--out, default analysis/pro_traj/attention_overlay):
  <job>/<suite>/<task>/<variant>_init<NN>_<success|fail>_attention.mp4   every episode with a video
  <job>/<suite>/<task>/attention_overlay_<variant>_init<NN>.png          one per entry (init 0 if it has a video)
  index.csv          one row per episode (overlay path or "no video")
  timing_check.csv   per rendered episode: calls, n_steps, video frames, last frame used
  orientation_check.json / orientation_check.png   PRO vs cand31 mean patch maps, as-is vs rotated 180°

Run: OPENBLAS_CORETYPE=ZEN uv run python scripts/shortcut/analysis/pro_overlay.py [--run DIR] [--out DIR]
        [--workers 8] [--only <substring>] [--limit N] [--stage orient,render,index]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = Path("/outputs/VLA-shortcut-exps/runs/bench-pi05lib-pro_all")
OUT = Path("/outputs/VLA-shortcut-exps/analysis/pro_traj/attention_overlay")
CAND31 = Path("/workspace/VLA-shortcut-exps-kubectl/cand31/attn/out/ours-goal-custom-train10")

GROUPS = ["early L0–5", "mid L6–11", "late L12–17"]
GROUP_LAYERS = [(0, 6), (6, 12), (12, 18)]
CAMS = ["agentview", "wrist"]
P = 16  # SigLIP patches per side
SIDE = 256  # per-camera side in the eval video
S = 240  # tile side (cand31)
FPS = 3
CRF = 28
STEP_PER_FRAME = 2  # eval video: 10 fps of a 20 Hz control loop
HOLD = 10  # eval video: hold frames after the final step
N_STILL = 6
CMAP = plt.get_cmap("inferno")
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- data


def episodes(run: Path) -> list[dict[str, Any]]:
    """All episodes of the run, with paths and the fields the overlay needs."""
    out = []
    for f in sorted((run / "jobs").glob("*/episodes.jsonl")):
        job = f.parent.name
        for line in f.read_text().splitlines():
            r = json.loads(line)
            variant = f"{r['axis']}_{r['level']}"
            task = f"task{r['task_id']:02d}"
            out.append(
                {
                    "job": job,
                    "suite": r["suite"],
                    "perturbation": r["axis"],
                    "level": r["level"],
                    "variant": variant,
                    "task": task,
                    "task_id": r["task_id"],
                    "init": r["init_idx"],
                    "outcome": "success" if r["success"] else "fail",
                    "dual": bool(r["dual"]),
                    "success_orig": r["success_orig"],
                    "n_steps": r["n_steps"],
                    "instruction": r.get("instruction_given") or r["instruction"],
                    "npz": r["activations"],
                    "video": str(f.parent / r["video"]) if r.get("video") else None,
                    "rel": f"{job}/{r['suite']}/{task}",
                    "name": f"{variant}_init{r['init_idx']:02d}",
                }
            )
    return out


def shares(am: np.ndarray) -> np.ndarray:
    """attn_mass (C, n, 18, 7) → (C, 3 groups, 2 cams): mean over denoising steps and the group's layers."""
    return np.stack([am[:, :, lo:hi, :2].mean((1, 2)) for lo, hi in GROUP_LAYERS], 1)


def patch_map(ai: np.ndarray, call: int, g: int, c: int) -> np.ndarray:
    return ai[call, g, c * P * P : (c + 1) * P * P].astype(np.float32).reshape(P, P)


def read_frames(path: str) -> np.ndarray:
    gen = imageio_ffmpeg.read_frames(path, output_params=["-threads", "1"])
    meta = next(gen)
    w, h = meta["size"]
    frames = [np.frombuffer(b, np.uint8).reshape(h, w, 3) for b in gen]
    return np.stack(frames)


# ---------------------------------------------------------------- drawing (cand31 look)


def overlay(img: np.ndarray, m: np.ndarray, q: float = 97) -> np.ndarray:
    """cand31: heat scaled to the frame's q-th percentile patch, alpha follows the value."""
    v = np.clip(m / max(np.percentile(m, q), 1e-12), 0, 1)
    v = cv2.resize(v, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    rgb = CMAP(v)[..., :3] * 255
    a = (0.15 + 0.7 * v)[..., None]
    return (img * (1 - a) + rgb * a).astype(np.uint8)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 13), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit(text: str, width: int, scale: float = 0.45) -> str:
    if cv2.getTextSize(text, FONT, scale, 1)[0][0] <= width:
        return text
    while text and cv2.getTextSize(text + "...", FONT, scale, 1)[0][0] > width:
        text = text[:-1]
    return text + "..."


def title_bar(lines: list[str], width: int) -> np.ndarray:
    bar: np.ndarray = np.zeros((4 + 18 * len(lines), width, 3), np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(bar, fit(s, width - 8), (4, 15 + 18 * i), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def head(ep: dict[str, Any]) -> str:
    s = f"{ep['suite']} {ep['task']} | PRO {ep['perturbation']}/{ep['level']} | init {ep['init']:02d}"
    s += f" | outcome: {ep['outcome']}"
    if ep["dual"]:
        s += f" (orig goal: {'yes' if ep['success_orig'] else 'no'})"
    return s


def cam_views(frame: np.ndarray) -> list[np.ndarray]:
    return [frame[:, :SIDE], frame[:, SIDE : 2 * SIDE]]


def video_frames(ep: dict[str, Any], ai: np.ndarray, sh: np.ndarray, t: np.ndarray, vid: np.ndarray) -> list:
    bar = title_bar([head(ep), f'"{ep["instruction"]}"'], 4 * S)
    out = []
    for call in range(len(t)):
        views = cam_views(vid[int(t[call]) // STEP_PER_FRAME])
        rows = []
        for c, cam in enumerate(CAMS):
            img = cv2.resize(views[c], (S, S), interpolation=cv2.INTER_AREA)
            tiles = [label(img, f"{cam} step {int(t[call])}")]
            for g, gname in enumerate(GROUPS):
                tiles.append(
                    label(
                        overlay(img, patch_map(ai, call, g, c)),
                        f"{gname.split()[0]} {sh[call, g, c] * 100:.0f}%",
                    )
                )
            rows.append(np.concatenate(tiles, 1))
        out.append(np.concatenate([bar, *rows], 0))
    return out


def write_mp4(path: Path, frames: list[np.ndarray]) -> None:
    h, w = frames[0].shape[:2]
    pad = (-h) % 8
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.mp4")
    gen = imageio_ffmpeg.write_frames(
        str(tmp),
        (w, h + pad),
        fps=FPS,
        codec="libx264",
        pix_fmt_out="yuv420p",
        quality=None,
        macro_block_size=1,
        output_params=["-crf", str(CRF), "-preset", "medium", "-movflags", "+faststart", "-threads", "1"],
    )
    gen.send(None)
    for f in frames:
        if pad:
            f = np.concatenate([f, np.zeros((pad, w, 3), np.uint8)], 0)
        gen.send(np.ascontiguousarray(f))
    gen.close()
    os.replace(tmp, path)


def still(
    ep: dict[str, Any], ai: np.ndarray, sh: np.ndarray, t: np.ndarray, vid: np.ndarray, path: Path
) -> None:
    """cand31 attention_overlay_i00.png: cameras × layer groups rows, N_STILL calls spread over the episode."""
    calls = sorted(set(np.linspace(0, len(t) - 1, N_STILL).round().astype(int).tolist()))
    rows = [(c, g) for c in range(2) for g in range(3)]
    fig, axes = plt.subplots(
        len(rows), len(calls), figsize=(1.75 * len(calls), 1.85 * len(rows)), squeeze=False
    )
    for k, call in enumerate(calls):
        views = cam_views(vid[int(t[call]) // STEP_PER_FRAME])
        for r, (c, g) in enumerate(rows):
            ax = axes[r, k]
            ax.imshow(overlay(views[c], patch_map(ai, call, g, c)))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{sh[call, g, c] * 100:.1f}%", fontsize=7, pad=2)
            if k == 0:
                ax.set_ylabel(f"{CAMS[c]}\n{GROUPS[g]}", fontsize=7)
            if r == len(rows) - 1:
                ax.set_xlabel(f"step {int(t[call])} (call {call})", fontsize=7)
    fig.suptitle(
        f'{head(ep)}\n"{ep["instruction"]}"\n'
        "(% above each panel: share of the action queries' attention on that camera)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- per episode


def out_name(ep: dict[str, Any]) -> str:
    return f"{ep['name']}_{ep['outcome']}_attention.mp4"


def render(job: tuple[dict[str, Any], Path, bool, bool]) -> dict[str, Any]:
    ep, out, want_still, force = job
    cv2.setNumThreads(1)
    d = out / ep["rel"]
    d.mkdir(parents=True, exist_ok=True)
    mp4 = d / out_name(ep)
    png = d / f"attention_overlay_{ep['name']}.png"
    z = np.load(ep["npz"])
    ai, t = z["attn_img"], z["t"]
    sh = shares(z["attn_mass"])
    meta = json.loads(str(z["meta"]))
    vid = read_frames(ep["video"])
    n_calls_expected = -(-ep["n_steps"] // 10)
    last_step_frame = len(vid) - HOLD - 2  # index of the frame at the last even step before the final step
    idx = t // STEP_PER_FRAME
    rec = {
        "job": ep["job"],
        "suite": ep["suite"],
        "task": ep["task"],
        "name": ep["name"],
        "n_calls": len(t),
        "n_calls_meta": meta.get("n_calls"),
        "skipped_calls": len(meta.get("skipped_calls", [])),
        "n_steps": ep["n_steps"],
        "calls_x_chunk": len(t) * 10,
        "calls_match": n_calls_expected == len(t),
        "video_frames": len(vid),
        "video_frames_expected": 1 + ep["n_steps"] // STEP_PER_FRAME + 1 + HOLD,
        "last_t": int(t[-1]),
        "last_frame_used": int(idx[-1]),
        "t_past_video": bool(idx[-1] > last_step_frame),
        "share_vs_img_maxdiff": float(
            np.abs(
                sh
                - np.stack(
                    [ai[:, :, c * P * P : (c + 1) * P * P].astype(np.float32).sum(-1) for c in range(2)], -1
                )
            ).max()
        ),
    }
    if idx[-1] >= len(vid):  # never seen; would crash: clamp and flag
        idx = np.minimum(idx, len(vid) - 1)
        t = idx * STEP_PER_FRAME
    if force or not mp4.exists():
        write_mp4(mp4, video_frames(ep, ai, sh, t, vid))
    rec["mp4_bytes"] = mp4.stat().st_size
    if want_still and (force or not png.exists()):
        still(ep, ai, sh, t, vid, png)
    return rec


# ---------------------------------------------------------------- orientation check


def mean_maps(paths: Iterator[Path] | list[Path]) -> np.ndarray:
    """(3 groups, 2 cams, 16, 16) mean patch map over episodes and calls, each normalised to sum 1."""
    acc = []
    for p in paths:
        ai = np.load(p)["attn_img"].astype(np.float32).mean(0)  # (3, 512); cand31 also has the empty camera
        acc.append(ai[:, : 2 * P * P].reshape(3, 2, P, P))
    m = np.mean(acc, 0)
    return m / m.sum((-1, -2), keepdims=True)


def orientation_check(run: Path, out: Path) -> None:
    """PRO goal `none` (original tasks) vs cand31 train10 (same 10 LIBERO-Goal tasks, upright 360² frames):
    correlation of mean patch maps per task, PRO as-is vs PRO rotated 180° (raw render orientation)."""
    res: dict[str, Any] = {"per_task": {}}
    rows = []
    for k in range(10):
        pro = sorted((run / "activations/pro-goal-none/libero_goal" / f"task{k:02d}").glob("*.npz"))
        c31 = sorted(CAND31.glob(f"ctl_t{k:02d}_train_i*.npz"))
        if not pro or not c31:
            continue
        a, b = mean_maps(pro), mean_maps(c31)
        r = {}
        for g in range(3):
            for c in range(2):
                x, y = a[g, c].ravel(), b[g, c].ravel()
                xr = a[g, c][::-1, ::-1].ravel()
                r[f"{GROUPS[g].split()[0]}_{CAMS[c]}"] = {
                    "as_is": float(np.corrcoef(x, y)[0, 1]),
                    "rot180": float(np.corrcoef(xr, y)[0, 1]),
                }
        res["per_task"][f"task{k:02d}"] = r
        rows.append((k, a, b))
    keys = list(next(iter(res["per_task"].values())))
    res["mean"] = {
        key: {o: float(np.mean([v[key][o] for v in res["per_task"].values()])) for o in ("as_is", "rot180")}
        for key in keys
    }
    (out / "orientation_check.json").write_text(json.dumps(res, indent=1))
    # figure: task 0 and pooled maps, PRO vs cand31
    a = np.mean([r[1] for r in rows], 0)
    b = np.mean([r[2] for r in rows], 0)
    fig, axes = plt.subplots(4, 3, figsize=(8, 10.5))
    for c in range(2):
        for g in range(3):
            for j, (m, name) in enumerate([(a, "PRO goal none (T023)"), (b, "cand31 train10")]):
                ax = axes[2 * c + j, g]
                ax.imshow(m[g, c], cmap="inferno")
                key = f"{GROUPS[g].split()[0]}_{CAMS[c]}"
                ax.set_title(
                    f"{name}\n{CAMS[c]} {GROUPS[g]}"
                    + (
                        f"\nr as-is {res['mean'][key]['as_is']:.2f} / rot180 {res['mean'][key]['rot180']:.2f}"
                        if j == 0
                        else ""
                    ),
                    fontsize=7,
                )
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(
        "mean patch attention, 10 LIBERO-Goal tasks: PRO (recorded during eval) vs cand31 (replay)\n"
        "r = per-task correlation of the 16×16 maps, mean over tasks",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / "orientation_check.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[orient]", json.dumps(res["mean"]))


# ---------------------------------------------------------------- main


def pick_stills(eps: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """One episode per entry (job · task · variant): init 0 if it has a video, else the lowest init that has one."""
    best: dict[tuple, dict[str, Any]] = {}
    for ep in eps:
        if not ep["video"]:
            continue
        key = (ep["job"], ep["task"], ep["variant"])
        if key not in best or ep["init"] < best[key]["init"]:
            best[key] = ep
    return {(e["rel"], e["name"]) for e in best.values()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="regex on <job>/<suite>/<task>/<name>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stage", default="orient,render,index")
    ap.add_argument("--force", action="store_true", help="re-render files that exist")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stages = a.stage.split(",")
    eps = episodes(a.run)
    if a.only:
        eps = [e for e in eps if re.search(a.only, f"{e['rel']}/{e['name']}")]
    stills = pick_stills(eps)
    if "orient" in stages:
        orientation_check(a.run, a.out)
    todo = [e for e in eps if e["video"]]
    if a.limit:
        todo = todo[: a.limit]
    recs: list[dict[str, Any]] = []
    if "render" in stages:
        jobs = [(e, a.out, (e["rel"], e["name"]) in stills, a.force) for e in todo]
        with Pool(a.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(render, jobs, chunksize=2), 1):
                recs.append(r)
                if i % 100 == 0 or i == len(jobs):
                    print(f"[render] {i}/{len(jobs)}", flush=True)
        recs.sort(key=lambda r: (r["job"], r["suite"], r["task"], r["name"]))
        with open(a.out / "timing_check.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
        bad = [r for r in recs if r["t_past_video"] or not r["calls_match"] or r["skipped_calls"]]
        print(f"[render] {len(recs)} episodes; timing issues: {len(bad)}")
        for r in bad[:20]:
            print("  ", r)
        print(f"[render] share vs attn_img max |diff| = {max(r['share_vs_img_maxdiff'] for r in recs):.2e}")
    if "index" in stages:
        with open(a.out / "index.csv", "w", newline="") as f:
            iw = csv.writer(f)
            iw.writerow(
                [
                    "job",
                    "suite",
                    "perturbation",
                    "level",
                    "task",
                    "init",
                    "outcome",
                    "success_orig",
                    "instruction",
                    "video",
                    "overlay",
                    "still",
                ]
            )
            for e in eps:
                mp4 = a.out / e["rel"] / out_name(e)
                png = a.out / e["rel"] / f"attention_overlay_{e['name']}.png"
                iw.writerow(
                    [
                        e["job"],
                        e["suite"],
                        e["perturbation"],
                        e["level"],
                        e["task"],
                        e["init"],
                        e["outcome"],
                        e["success_orig"] if e["dual"] else "",
                        e["instruction"],
                        e["video"] or "no video",
                        str(mp4.relative_to(a.out)) if e["video"] and mp4.exists() else "no video",
                        str(png.relative_to(a.out)) if png.exists() else "",
                    ]
                )
        n_vid = sum(1 for e in eps if e["video"])
        print(f"[index] {len(eps)} episodes, {n_vid} with video, {len(eps) - n_vid} without")


if __name__ == "__main__":
    main()
