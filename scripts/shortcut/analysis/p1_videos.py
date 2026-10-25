"""P1 pilot comparison videos (T005): pick episodes by rule, render mp4 + GIF ≤ 3 MB + last-frame png, manifest.json.

    uv run python scripts/shortcut/analysis/p1_videos.py /outputs/VLA-shortcut-exps/runs/sc-p1-* \
        --out /outputs/VLA-shortcut-exps/analysis/p1

Categories (videos only exist for init 0–2 of pi05lib; pi05base uses the same inits):
  A  language swap   C0 | C4 | C5, same model and init. Per model × suite one video per available type
                     (shortcut = original goal under C4/C5, follows = new goal, fail = neither).
  B  language drop   C0 | C1, same model and init. Per model × suite 1–2.
  C  model contrast  pi05lib C0 | pi05lib C4 | pi05base C0 | pi05base C4, same init and C4 variant. One per suite.
Picks maximise how typical the shown episode is for its (task, condition, variant) cell over all 10 inits
(`why` gives the counts). Each panel keeps the clip's own overlay (live t, orig/new goal yes@t) and gets a
header with model, condition, the full instruction and the first grasped object.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess  # nosec B404
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from metrics import load_episodes  # noqa: E402
from p1_tables import SUITE_SHORT, VIDEO_INITS, classify, default_bddl_root, task_objects  # noqa: E402

GIF_MAX_BYTES = 3 * 1024 * 1024
GIF_MAX_WIDTH = 1040
SRC_FPS = 10
PANEL = 256
HEADER_H = 92
TITLE_H = 20
GAP = 4
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MODEL_LABEL = {"pi05lib": "π0.5 LIBERO", "pi05base": "π0.5 base"}
SUITE_LABEL = {"libero_spatial": "LIBERO-Spatial", "libero_goal": "LIBERO-Goal"}
COND_LABEL = {
    "C0": "original",
    "C1": "empty instruction",
    "C4": "object swap",
    "C5": "target swap",
}
TYPE_KO = {
    "shortcut": "원래 goal",
    "follows": "새 goal",
    "fail": "둘 다 실패",
    "success": "성공",
    "both": "둘 다",
}

Ep = dict[str, Any]


# ---------------------------------------------------------------- episode index


def vtype(r: Ep) -> str:
    so, sn = bool(r["success_orig"]), bool(r.get("success_new"))
    if r["condition"] in ("C0", "C1", "C2", "C3"):
        return "success" if so else "fail"
    return "both" if so and sn else "shortcut" if so else "follows" if sn else "fail"


class Index:
    def __init__(self, eps: list[Ep], objs: dict[tuple[str, int], str]):
        self.objs = objs
        self.by_key: dict[tuple, Ep] = {}
        self.cells: dict[tuple, list[Ep]] = defaultdict(list)
        for r in eps:
            if r.get("repeat_idx", 0) != 0:
                continue
            r["_vtype"] = vtype(r)
            r["_qtype"] = classify(r, objs[(r["suite"], r["task_id"])])
            self.by_key[
                (r["model"], r["suite"], r["task_id"], r["condition"], r["variant"], r["init_idx"])
            ] = r
            self.cells[(r["model"], r["suite"], r["task_id"], r["condition"], r["variant"])].append(r)

    def get(self, model: str, suite: str, task: int, cond: str, var: str, init: int) -> Ep | None:
        r = self.by_key.get((model, suite, task, cond, var, init))
        return r if r and r.get("video") else None

    def variants(self, model: str, suite: str, task: int, cond: str) -> list[str]:
        return sorted(v for (m, s, t, c, v) in self.cells if (m, s, t, c) == (model, suite, task, cond))

    def cell(self, r: Ep) -> list[Ep]:
        return self.cells[(r["model"], r["suite"], r["task_id"], r["condition"], r["variant"])]

    def frac(self, r: Ep, key: str = "_vtype") -> float:
        cell = self.cell(r)
        return sum(e[key] == r[key] for e in cell) / len(cell)

    def describe(self, r: Ep) -> str:
        """'spat t3 C4-a: 10 ep 중 원래 goal 9/10, 새 goal 0/10 · 이 영상(init 1) = 원래 goal (다수 유형)'."""
        cell = self.cell(r)
        n = len(cell)
        so = sum(bool(e["success_orig"]) for e in cell)
        parts = [f"{SUITE_SHORT[r['suite']]} t{r['task_id']} {r['model']} {r['variant']}: {n} ep 중 "]
        if r["condition"] in ("C4", "C5", "C6"):
            sn = sum(bool(e.get("success_new")) for e in cell)
            parts.append(f"원래 goal {so}/{n}, 새 goal {sn}/{n}")
        else:
            parts.append(f"성공 {so}/{n}")
        q = Counter(e["_qtype"] for e in cell).most_common(2)
        parts.append(" (" + ", ".join(f"{k} {v}" for k, v in q) + ")")
        k = sum(e["_vtype"] == r["_vtype"] for e in cell)
        kq = sum(e["_qtype"] == r["_qtype"] for e in cell)
        rep = "다수" if 2 * k > n else "동률" if 2 * k == n else "소수"
        parts.append(
            f" · 이 영상(init {r['init_idx']}) = {TYPE_KO[r['_vtype']]} ({rep} {k}/{n}), 행동 {r['_qtype']} {kq}/{n}"
        )
        return "".join(parts)


# ---------------------------------------------------------------- selection


def pick_a(ix: Index, model: str, suite: str) -> list[tuple[str, list[Ep]]]:
    """C0 | C4 | C5 per (task, init), one video per type; types ordered so the hypothesis-relevant one wins
    task ties (Spatial: shortcut first, Goal: follows first)."""
    order = ["shortcut", "follows", "fail"] if suite == "libero_spatial" else ["follows", "shortcut", "fail"]
    used_tasks: set[int] = set()

    def best_for(kind: str, grasp_bonus: float) -> tuple[float, list[Ep]] | None:
        best: tuple[float, list[Ep]] | None = None
        for task in range(10):
            if task in used_tasks:
                continue
            for init in VIDEO_INITS:
                c0 = ix.get(model, suite, task, "C0", "orig", init)
                if c0 is None:
                    continue
                v4s = [
                    ix.get(model, suite, task, "C4", v, init) for v in ix.variants(model, suite, task, "C4")
                ]
                v5s = [
                    ix.get(model, suite, task, "C5", v, init) for v in ix.variants(model, suite, task, "C5")
                ]
                for r4 in [r for r in v4s if r] or [None]:
                    for r5 in [r for r in v5s if r] or [None]:
                        panels = [r for r in (r4, r5) if r]
                        hit = [r for r in panels if r["_vtype"] == kind]
                        if not hit:
                            continue
                        # typical hit, and every other panel showing its own cell's usual outcome
                        score = sum(ix.frac(r) for r in hit) / len(hit) + 0.8 * sum(
                            ix.frac(r) for r in panels
                        ) / len(panels)
                        score += 0.1 * len(hit) + 0.5 * bool(c0["success_orig"]) + 0.05 * len(panels)
                        if kind == "fail":  # failures: a visible wrong grasp vs idling, typicality by subtype
                            score += sum(ix.frac(r, "_qtype") for r in hit) / len(hit)
                            score += grasp_bonus * sum(r["_qtype"] != "no_grasp" for r in [c0, *panels])
                        if best is None or score > best[0] + 1e-9:
                            best = (score, [c0, *panels])
        return best

    out = []
    for kind in order:
        best = best_for(kind, 0.4)
        if best is not None:
            used_tasks.add(best[1][0]["task_id"])
            out.append((kind, best[1]))
    if (
        len(out) < 2
    ):  # pi05base: every episode fails → add a second failure, now favouring the typical idle one
        best = best_for("fail", -0.4)
        if best is not None:
            out.append(("fail", best[1]))
    return out


def pick_b(ix: Index, model: str, suite: str) -> list[tuple[str, list[Ep]]]:
    """C0 | C1. pi05lib: Spatial C1 success (most typical) then C1 failure; Goal C1 failure then success.
    pi05base: one (all fail), preferring a C1 episode that grasps something."""
    if model == "pi05base":
        order = ["fail"]
    else:
        order = ["success", "fail"] if suite == "libero_spatial" else ["fail", "success"]
    out, used = [], set()
    for kind in order:
        best = None
        for task in range(10):
            if task in used:
                continue
            for init in VIDEO_INITS:
                c0 = ix.get(model, suite, task, "C0", "orig", init)
                c1 = ix.get(model, suite, task, "C1", "blank", init)
                if not (c0 and c1) or c1["_vtype"] != kind:
                    continue
                # most typical C1 outcome; failures also favour a visible wrong grasp; C0 at the same init succeeds
                score = ix.frac(c1) + 0.3 * (kind == "fail" and c1["_qtype"] != "no_grasp")
                score += 0.5 * bool(c0["success_orig"])
                if best is None or score > best[0] + 1e-9:
                    best = (score, [c0, c1])
        if best is not None:
            used.add(best[1][0]["task_id"])
            out.append((kind, best[1]))
    return out


def pick_c(ix: Index, suite: str) -> tuple[str, list[Ep]] | None:
    """pi05lib C0 | pi05lib C4 | pi05base C0 | pi05base C4, same (task, init, C4 variant).
    Spatial shows the pi05lib shortcut, Goal the pi05lib follow."""
    kind = "shortcut" if suite == "libero_spatial" else "follows"
    best = None
    for task in range(10):
        for var in ix.variants("pi05lib", suite, task, "C4"):
            for init in VIDEO_INITS:
                rs = [
                    ix.get("pi05lib", suite, task, "C0", "orig", init),
                    ix.get("pi05lib", suite, task, "C4", var, init),
                    ix.get("pi05base", suite, task, "C0", "orig", init),
                    ix.get("pi05base", suite, task, "C4", var, init),
                ]
                if not all(rs) or rs[1]["_vtype"] != kind or not rs[0]["success_orig"]:
                    continue
                score = ix.frac(rs[1]) + 0.5 * ix.frac(rs[3], "_qtype") + 0.1 * ix.frac(rs[0])
                if best is None or score > best[0] + 1e-9:
                    best = (score, rs)
    return (kind, best[1]) if best else None


# ---------------------------------------------------------------- rendering


def short_obj(name: str | None) -> str:
    if not name:
        return "none"
    for pre in ("akita_", "glazed_rim_porcelain_", "wooden_", "flat_"):
        name = name.removeprefix(pre)
    return name


def header(r: Ep, fonts: tuple[Any, Any]) -> np.ndarray:
    f, fb = fonts
    img = Image.new("RGB", (PANEL, HEADER_H), (24, 24, 24))
    d = ImageDraw.Draw(img)
    cond = r["condition"] if r["variant"] in ("orig", "blank") else r["variant"]
    d.text(
        (5, 3),
        f"{MODEL_LABEL[r['model']]} · {cond} {COND_LABEL.get(r['condition'], '')}",
        font=fb,
        fill="white",
    )
    instr = f'"{r["instruction"]}"' if r["instruction"] else '"" (no instruction)'
    lines = textwrap.wrap(instr, 42)[:3]
    for i, line in enumerate(lines):
        d.text((5, 20 + 14 * i), line, font=f, fill=(215, 215, 215))
    y = 20 + 14 * 3 + 2
    green, red = (90, 230, 90), (255, 100, 100)
    ok_o = bool(r["success_orig"])
    if r["condition"] in ("C4", "C5", "C6"):
        ok_n = bool(r.get("success_new"))
        d.text((5, y), f"final: orig {'✓' if ok_o else '✗'}", font=fb, fill=green if ok_o else red)
        d.text((122, y), f"new {'✓' if ok_n else '✗'}", font=fb, fill=green if ok_n else red)
    else:
        d.text((5, y), f"final: success {'✓' if ok_o else '✗'}", font=fb, fill=green if ok_o else red)
    d.text(
        (5, y + 15), f"first grasp: {short_obj(r.get('first_grasp_object'))}", font=f, fill=(170, 200, 255)
    )
    return np.asarray(img)


def title_bar(text: str, width: int, font: Any) -> np.ndarray:
    img = Image.new("RGB", (width, TITLE_H), (255, 255, 255))
    ImageDraw.Draw(img).text((5, 3), text, font=font, fill=(20, 20, 20))
    return np.asarray(img)


def render(rs: list[Ep], title: str, out: Path, ffmpeg: str) -> dict[str, Any]:
    fonts = (ImageFont.truetype(FONT, 11), ImageFont.truetype(FONT_BOLD, 11))
    clips = []
    for r in rs:
        frames = [
            np.asarray(f)[:, :PANEL] for f in imageio.mimread(Path(r["_root"]) / r["video"], memtest=False)
        ]
        clips.append((header(r, fonts), frames))
    n = max(len(c) for _, c in clips)
    width = len(clips) * PANEL + (len(clips) - 1) * GAP
    width += -width % 8
    tbar = title_bar(title, width, ImageFont.truetype(FONT_BOLD, 12))
    frames_out = []
    for t in range(n):
        cols = []
        for i, (hdr, c) in enumerate(clips):
            cols.append(np.concatenate([hdr, c[min(t, len(c) - 1)]], axis=0))
            if i < len(clips) - 1:
                cols.append(np.full((HEADER_H + PANEL, GAP, 3), 255, np.uint8))
        body = np.concatenate(cols, axis=1)
        body = np.pad(body, ((0, 0), (0, width - body.shape[1]), (0, 0)), constant_values=255)
        frames_out.append(np.concatenate([tbar, body], axis=0))
    out.parent.mkdir(parents=True, exist_ok=True)
    mp4 = out.with_suffix(".mp4")
    imageio.mimwrite(mp4, frames_out, fps=SRC_FPS, codec="libx264", quality=7, macro_block_size=8)
    gif = out.with_suffix(".gif")
    gif_info = make_gif(mp4, gif, width, ffmpeg)
    thumb = out.parent.parent / "thumbs" / (out.name + ".png")
    thumb.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(thumb, frames_out[-1])
    return {
        "mp4": str(mp4),
        "gif": str(gif),
        "gif_bytes": gif.stat().st_size,
        "thumb": str(thumb),
        **gif_info,
    }


def make_gif(mp4: Path, gif: Path, width: int, ffmpeg: str) -> dict[str, Any]:
    """Keep width ≤ 1040 and 8–10 fps; drop frames (step) and colours before shrinking."""
    w0 = min(width, GIF_MAX_WIDTH)
    tries = [(s, fps, col, 1.0) for s in (1, 2, 3) for fps in (10, 8) for col in (128, 64)]
    tries += [(3, 8, 64, 0.85), (4, 8, 64, 0.85), (4, 8, 48, 0.75)]
    for step, fps, colors, scale in tries:
        w = int(w0 * scale) // 2 * 2
        speed = SRC_FPS / (step * fps)  # source frame i lands at i / (step · fps) seconds
        vf = (
            f"setpts={speed:.4f}*PTS,fps={fps},scale={w}:-1:flags=lanczos,"
            f"split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        )
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(mp4), "-vf", vf, str(gif)], check=True)  # nosec B603
        if gif.stat().st_size <= GIF_MAX_BYTES:
            print(
                f"  gif {gif.name} {gif.stat().st_size / 1e6:.2f} MB (w {w}, step {step}, {fps} fps, {colors} col)"
            )
            return {"gif_width": w, "gif_fps": fps, "gif_frame_step": step, "gif_colors": colors}
    raise SystemExit(f"gif still {gif.stat().st_size / 1e6:.2f} MB > 3 MB: {gif}")


# ---------------------------------------------------------------- main


def entry(cat: str, vid: str, kind: str, rs: list[Ep], ix: Index, why_extra: str = "") -> dict[str, Any]:
    models = sorted({r["model"] for r in rs})
    seen, why = set(), []
    for r in rs:
        if r["condition"] == "C0" and cat != "C" and r["success_orig"]:
            continue  # C0 successes are the expected baseline; only listed when they fail or in C
        key = (r["model"], r["condition"], r["variant"])
        if key not in seen:
            seen.add(key)
            why.append(ix.describe(r))
    return {
        "id": vid,
        "category": cat,
        "model": models[0] if len(models) == 1 else "+".join(models),
        "suite": rs[0]["suite"],
        "task_id": rs[0]["task_id"],
        "init": rs[0]["init_idx"],
        "conditions": [
            (f"{r['model']} " if cat == "C" else "")
            + (r["variant"] if r["condition"] in ("C4", "C5") else r["condition"])
            for r in rs
        ],
        "instructions": [r["instruction"] for r in rs],
        "outcomes": [
            {
                **({"model": r["model"]} if cat == "C" else {}),
                "cond": r["variant"] if r["condition"] in ("C4", "C5") else r["condition"],
                "success_orig": bool(r["success_orig"]),
                "success_new": None if r.get("success_new") is None else bool(r["success_new"]),
                "t_orig": r.get("t_orig"),
                "t_new": r.get("t_new"),
                "first_grasp": r.get("first_grasp_object"),
                "qual_type": r["_qtype"],
                "video_src": str(Path(r["_root"]) / r["video"]),
            }
            for r in rs
        ],
        "type": kind,
        "why": " | ".join(([why_extra] if why_extra else []) + why),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bddl-root", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    eps = load_episodes(args.runs)
    bddl_root = Path(args.bddl_root) if args.bddl_root else default_bddl_root()
    ix = Index(eps, task_objects({r["suite"] for r in eps}, bddl_root))

    plan: list[tuple[str, str, str, list[Ep], str]] = []  # (cat, id, type, episodes, title)
    for model in ("pi05lib", "pi05base"):
        for suite in ("libero_spatial", "libero_goal"):
            s = SUITE_SHORT[suite]
            for kind, rs in pick_a(ix, model, suite):
                t, i = rs[0]["task_id"], rs[0]["init_idx"]
                vid = f"A_{model}_{s}_t{t:02d}_i{i}_{kind}"
                title = f"A · language swap · {MODEL_LABEL[model]} · {SUITE_LABEL[suite]} task {t} · init {i}"
                plan.append(("A", vid, kind, rs, title))
            for kind, rs in pick_b(ix, model, suite):
                t, i = rs[0]["task_id"], rs[0]["init_idx"]
                vid = f"B_{model}_{s}_t{t:02d}_i{i}_C1{kind}"
                title = (
                    f"B · language removed · {MODEL_LABEL[model]} · {SUITE_LABEL[suite]} task {t} · init {i}"
                )
                plan.append(("B", vid, f"C1-{kind}", rs, title))
    for suite in ("libero_spatial", "libero_goal"):
        picked = pick_c(ix, suite)
        if picked:
            kind, rs = picked
            t, i = rs[0]["task_id"], rs[0]["init_idx"]
            vid = f"C_{SUITE_SHORT[suite]}_t{t:02d}_i{i}_{rs[1]['variant']}"
            title = f"C · model contrast · {SUITE_LABEL[suite]} task {t} · init {i} · same {rs[1]['variant']} instruction"
            plan.append(("C", vid, f"pi05lib-{kind}", rs, title))

    manifest = []
    for cat, vid, kind, rs, title in plan:
        print(f"{vid}: {[r['variant'] + ':' + r['_qtype'] for r in rs]}")
        files = render(rs, title, out / "videos" / vid, ffmpeg)
        e = entry(cat, vid, kind, rs, ix)
        e.update(files)
        manifest.append(e)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    cnt = Counter(e["category"] for e in manifest)
    big = [e["id"] for e in manifest if e["gif_bytes"] > GIF_MAX_BYTES]
    print(
        f"{len(manifest)} videos (A {cnt['A']}, B {cnt['B']}, C {cnt['C']}) → {out / 'manifest.json'}; gif > 3 MB: {big}"
    )


if __name__ == "__main__":
    main()
