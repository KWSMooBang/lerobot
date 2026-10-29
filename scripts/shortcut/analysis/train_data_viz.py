# ruff: noqa: N803, N806  (D = data, M = manifest, X/W = matrices)
"""T008: visualize the π0.5 LIBERO training data (HuggingFaceVLA/libero, 4 suites, 40 tasks, 1,693 episodes).

    # 1. parquet → cache (CPU)                      scripts/shortcut/analysis/train_data_extract.py
    # 2. eval init-state renders + camera (GPU 0)   scripts/shortcut/analysis/train_data_render.py
    # 3. figures (CPU)
    PYTHONPATH=src:scripts/shortcut/analysis python scripts/shortcut/analysis/train_data_viz.py \
        --cache /outputs/VLA-shortcut-exps/data/hf_libero_viz_cache \
        --out /outputs/VLA-shortcut-exps/analysis/train_data_viz

Sections (manifest.json → section):
  A scale      stats_tasks.csv, episode_lengths.csv, stats_counts.png, stats_length.png
  B language   instructions.csv, slot_coverage_<suite>.png (+ slot_coverage.csv), vocab_<suite>.png (+ vocab_freq.csv)
  C scene      scene_task_map.png, grid_first_/grid_last_<suite>.png, contact_<suite>_t<id>.png,
               layout_distance.png/.csv (physical layout per task from the eval init renders → classifier ceiling),
               first_frame_task_acc.csv, first_frame_confusion.png, frame_task_acc_by_time.csv/.png
  D trajectory traj_xy_<suite>.png, grasp_place_<suite>.png (+ grasp_place.csv), overlay_<suite>_t<id>.png
  E action     actions_hist.png, actions_progress.png, actions_summary.csv
  F video      demo_<suite>_t<id>.gif, filmstrip_<suite>_t<id>.png
  G eval init  init_compare_<suite>.png, init_vs_train.csv, init_vs_train.png

Conventions. Images are shown as stored (upright, the same view as our eval videos; see train_data_extract.py).
observation.state = [eef xyz (world, m), eef axis-angle (3), gripper finger qpos (2)]; action = [Δxyz, Δaxis-angle,
gripper] with gripper −1 = open, +1 = close. Grasp = first frame with gripper action > 0; place = first later frame
with gripper action < 0 (first pick-and-place only). Top-down plots are drawn as seen from the agentview camera
(world +x down the page, world +y to the left). The agentview projection uses robosuite's
get_camera_transform_matrix; for the upright 256 px image, pixel = (255 − p0/p2, p1/p2) (checked against object
positions in the eval renders).

First-frame → task classifier: per suite, features of the agentview frame at a progress fraction (pixels: 32×32 RGB,
z-scored; dino: DINOv2-small CLS on the 128 px frame resized to 224), leave-one-episode-out 1-NN / 5-NN (cosine),
and 5-fold logistic regression on dino. chance = 1/10.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from slots import parse  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SNAME = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}
SKO = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}
# reference categorical palette (dataviz skill), slots 1–8 in fixed order; suites take slots 1–4
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TASK10 = CAT + ["#52514e", "#a0662a"]  # 10 tasks in one panel: always paired with a direct task-id label
SCOL = dict(zip(SUITES, CAT[:4], strict=True))
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INSTR_DIR = Path(__file__).resolve().parents[1] / "instructions"
TRAIN_TASKS = "/outputs/VLA-shortcut-exps/analysis/p1_train_overlap/train_tasks.csv"

plt.rcParams.update(
    {
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
    }
)


class Manifest:
    def __init__(self, out: Path):
        self.out, self.items = out, []

    def add(
        self, rel: str, section: str, title: str, caption: str, suite: str = "", task_id: int | None = None
    ) -> None:
        p = self.out / rel
        assert p.exists(), p
        self.items.append(
            {
                "id": Path(rel).stem,
                "section": section,
                "suite": suite,
                "task_id": task_id,
                "title": title,
                "caption": caption,
                "path": str(p),
                "bytes": p.stat().st_size,
            }
        )

    def write(self) -> None:
        (self.out / "manifest.json").write_text(json.dumps(self.items, ensure_ascii=False, indent=1))


def savefig(fig: Any, path: Path, dpi: int = 100) -> None:
    w = fig.get_size_inches()[0] * dpi
    assert w <= 1600 + 1, (path, w)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    assert path.stat().st_size <= 3e6, (path, path.stat().st_size)


def font(size: int) -> ImageFont.FreeTypeFont:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_png(img: Image.Image, path: Path) -> None:
    assert img.width <= 1600, (path, img.width)
    img.save(path, optimize=True)
    if path.stat().st_size > 3e6:
        img.convert("RGB").quantize(256).save(path, optimize=True)
    assert path.stat().st_size <= 3e6, path


def wrap(s: str, n: int) -> str:
    return "\n".join(textwrap.wrap(s, n))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------- data


class Data:
    def __init__(self, cache: Path):
        self.cache = cache
        ep = pd.read_parquet(cache / "episodes.parquet")
        self.ep = ep.sort_values("episode_index").reset_index(drop=True)
        tt = pd.read_csv(TRAIN_TASKS)
        self.tasks = tt.rename(columns={"task_id": "suite_task_id"})
        low = np.load(cache / "lowdim.npz")
        self.state, self.action, self.fep, self.fframe = low["state"], low["action"], low["ep"], low["frame"]
        starts = np.r_[0, np.nonzero(np.diff(self.fep))[0] + 1]
        self.slice = {
            int(self.fep[s]): (int(s), int(e))
            for s, e in zip(starts, np.r_[starts[1:], len(self.fep)], strict=True)
        }
        pr = np.load(cache / "progress.npz")
        self.progress, self.fracs = pr["progress"], pr["fracs"]
        self._first = self._last = None

    @property
    def first(self) -> np.ndarray:
        if self._first is None:
            self._first = np.load(self.cache / "first256.npz")["first"]
        return self._first

    @property
    def last(self) -> np.ndarray:
        if self._last is None:
            self._last = np.load(self.cache / "last256.npz")["last"]
        return self._last

    def eps(self, suite: str, tid: int | None = None) -> pd.DataFrame:
        d = self.ep[self.ep.suite == suite]
        return d if tid is None else d[d.suite_task_id == tid]

    def rep(self, suite: str, tid: int) -> int:
        d = self.eps(suite, tid)
        return int(d[d.rep].episode_index.iloc[0])

    def traj(self, e: int) -> tuple[np.ndarray, np.ndarray]:
        s, t = self.slice[e]
        return self.state[s:t], self.action[s:t]

    def instr(self, suite: str, tid: int) -> str:
        return str(self.eps(suite, tid).task.iloc[0])


def grasp_place(action: np.ndarray) -> tuple[int | None, int | None]:
    g = np.nonzero(action[:, 6] > 0)[0]
    if len(g) == 0:
        return None, None
    after = np.nonzero(action[g[0] :, 6] < 0)[0]
    return int(g[0]), (int(g[0] + after[0]) if len(after) else None)


# ---------------------------------------------------------------- A scale


def section_a(D: Data, out: Path, M: Manifest) -> dict:
    rows = []
    for (suite, tid), g in D.ep.groupby(["suite", "suite_task_id"]):
        L = g.length
        rows.append(
            {
                "suite": suite,
                "task_id": int(tid),
                "dataset_task_index": int(g.task_index.iloc[0]),
                "instruction": g.task.iloc[0],
                "scene_id": g.scene_id.iloc[0],
                "n_episodes": len(g),
                "n_frames": int(L.sum()),
                "len_mean": round(float(L.mean()), 1),
                "len_median": float(L.median()),
                "len_std": round(float(L.std()), 1),
                "len_min": int(L.min()),
                "len_max": int(L.max()),
                "rep_episode": int(g[g.rep].episode_index.iloc[0]),
            }
        )
    st = pd.DataFrame(rows)
    st["suite"] = pd.Categorical(st.suite, SUITES, ordered=True)
    st = st.sort_values(["suite", "task_id"])
    st.to_csv(out / "stats_tasks.csv", index=False)
    D.ep[["episode_index", "suite", "suite_task_id", "task_index", "length", "rep"]].rename(
        columns={"suite_task_id": "task_id", "task_index": "dataset_task_index"}
    ).to_csv(out / "episode_lengths.csv", index=False)
    M.add(
        "stats_tasks.csv",
        "A",
        "suite × task 별 학습 규모 표",
        "40 task: ep 수, frame 수, ep 길이 요약 (차트용 tidy)",
    )
    M.add(
        "episode_lengths.csv", "A", "episode 별 길이", "1,693 ep 의 suite, task, 길이(frame), 대표 demo 여부"
    )

    # counts
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4), sharey="row")
    for j, s in enumerate(SUITES):
        d = st[st.suite == s]
        for i, (col, lab) in enumerate((("n_episodes", "episodes"), ("n_frames", "frames"))):
            ax = axes[i, j]
            ax.bar(d.task_id, d[col], color=SCOL[s], width=0.7)
            for x, y in zip(d.task_id, d[col], strict=True):
                ax.text(
                    x,
                    y,
                    f"{y:,}" if col == "n_frames" else str(y),
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=INK2,
                )
            ax.set_xticks(range(10))
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f"{SNAME[s]}  ·  {d.n_episodes.sum()} ep, {d.n_frames.sum():,} frames")
            else:
                ax.set_xlabel("task id")
    fig.suptitle(
        "Training episodes and frames per task (HuggingFaceVLA/libero)", x=0.01, ha="left", fontsize=12
    )
    fig.tight_layout()
    savefig(fig, out / "stats_counts.png")
    ep_rng = (int(st.n_episodes.min()), int(st.n_episodes.max()))
    fr_by_suite = st.groupby("suite", observed=True).n_frames.sum()
    M.add(
        "stats_counts.png",
        "A",
        "task 별 학습 episode·frame 수",
        f"task 당 {ep_rng[0]}–{ep_rng[1]} ep; suite frame: "
        + ", ".join(f"{SNAME[s]} {fr_by_suite[s]:,}" for s in SUITES)
        + " (LIBERO-10 은 긴 태스크라 frame 이 가장 많다)",
    )

    # length
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for j, s in enumerate(SUITES):
        ax = axes[j]
        d = D.eps(s)
        data = [d[d.suite_task_id == t].length.to_numpy() for t in range(10)]
        ax.boxplot(
            data,
            positions=range(10),
            widths=0.55,
            showfliers=False,
            medianprops={"color": INK},
            boxprops={"color": MUTED},
            whiskerprops={"color": MUTED},
            capprops={"color": MUTED},
        )
        for t, v in enumerate(data):
            ax.scatter(t + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=SCOL[s], alpha=0.55, lw=0)
        ax.set_xticks(range(10))
        ax.set_xlabel("task id")
        ax.set_title(f"{SNAME[s]}  ·  median {int(d.length.median())} frames")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("episode length (frames, 10 fps)")
    fig.suptitle("Episode length per task (each dot = one demo)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    savefig(fig, out / "stats_length.png")
    med = D.ep.groupby("suite").length.median()
    M.add(
        "stats_length.png",
        "A",
        "task 별 episode 길이 분포",
        "suite 중앙값 "
        + ", ".join(f"{SNAME[s]} {int(med[s])}" for s in SUITES)
        + " frame (10 fps); 길이 범위 "
        f"{int(D.ep.length.min())}–{int(D.ep.length.max())}",
    )
    return {
        "n_ep": len(D.ep),
        "n_frames": int(D.ep.length.sum()),
        "ep_range": ep_rng,
        "len_median": med.to_dict(),
    }


# ---------------------------------------------------------------- B language


def col_key(a: Any) -> str:
    if a.verb in ("open", "close", "turn on", "turn off"):
        return f"[{a.verb}]"
    if a.verb == "push":
        return f"push→{a.dest_str}"
    return a.dest_str


def load_swaps() -> list[dict]:
    out = []
    for s in SUITES:
        d = json.loads((INSTR_DIR / f"{s}.json").read_text())
        for t in d["tasks"]:
            for w in t["swaps"]:
                out.append(
                    {
                        "suite": s,
                        "task_id": t["task_id"],
                        **{k: w[k] for k in ("id", "type", "instruction")},
                        "feasible": w.get("feasible", True),
                    }
                )
    return out


def section_b(D: Data, out: Path, M: Manifest) -> dict:
    st = pd.read_csv(out / "stats_tasks.csv")
    tt = D.tasks.set_index("instruction")
    rows = []
    for r in st.itertuples():
        acts = parse(r.instruction)
        rows.append(
            {
                "suite": r.suite,
                "task_id": r.task_id,
                "dataset_task_index": r.dataset_task_index,
                "instruction": r.instruction,
                "n_episodes": r.n_episodes,
                "n_frames": r.n_frames,
                "scene_id": r.scene_id,
                "bddl": tt.loc[r.instruction, "bddl"],
                "actions": " ; ".join(f"{a.verb}({a.obj_ref} → {a.dest_str})" for a in acts),
            }
        )
    pd.DataFrame(rows).to_csv(out / "instructions.csv", index=False)
    M.add(
        "instructions.csv",
        "B",
        "학습 instruction 40개 전체 표",
        "suite, task id, instruction, ep 수, scene id, bddl, slot 분해 (T007 slots 규칙)",
    )

    # slot coverage: (object × destination/action) cells, value = training episodes over all 4 suites
    cell = Counter()
    cell_suite: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for a in parse(r["instruction"]):
            cell[(a.obj_ref, col_key(a))] += r["n_episodes"]
            cell_suite[r["suite"]][(a.obj_ref, col_key(a))] += r["n_episodes"]
    swaps = load_swaps()
    cov_rows = []
    for w in swaps:
        for k, a in enumerate(parse(w["instruction"])):
            cov_rows.append(
                {
                    "suite": w["suite"],
                    "swap_id": f"{w['suite']}-t{w['task_id']}-{w['id']}",
                    "type": w["type"],
                    "feasible": w["feasible"],
                    "instruction": w["instruction"],
                    "action_idx": k,
                    "obj": a.obj_ref,
                    "col": col_key(a),
                    "train_ep_all": cell.get((a.obj_ref, col_key(a)), 0),
                    "train_ep_suite": cell_suite[w["suite"]].get((a.obj_ref, col_key(a)), 0),
                }
            )
    cov = pd.DataFrame(cov_rows)
    tr_rows = [
        {"suite": s, "obj": o, "col": c, "train_ep": n}
        for s, cc in cell_suite.items()
        for (o, c), n in cc.items()
    ]
    pd.DataFrame(tr_rows).to_csv(out / "slot_train_cells.csv", index=False)
    cov.to_csv(out / "slot_coverage.csv", index=False)
    M.add(
        "slot_train_cells.csv",
        "B",
        "학습 slot 칸 (물체 × 목적지/동작) ep 수",
        "suite 별 (조작 물체, 목적지 또는 동작) 칸의 학습 ep 수",
    )
    M.add(
        "slot_coverage.csv",
        "B",
        "평가 swap 의 slot 칸과 학습 ep 수",
        f"swap {len(swaps)}개 × action: 그 칸의 학습 ep 수 (전체 / 같은 suite)",
    )
    nums = {}
    for s in SUITES:
        sw = cov[cov.suite == s]
        objs = sorted({o for (o, _c) in cell_suite[s]} | set(sw.obj))
        cols = sorted(
            {c for (_o, c) in cell_suite[s]} | set(sw.col),
            key=lambda c: (c.startswith("["), c.startswith("push"), c),
        )
        mat = np.array([[cell.get((o, c), 0) for c in cols] for o in objs], float)
        fig_w = min(16, max(10, 4.5 + 1.0 * len(cols)))
        fig, ax = plt.subplots(figsize=(fig_w, 2.4 + 0.5 * len(objs)))
        vmax = max(mat.max(), 1)
        shown = np.where(mat > 0, mat, np.nan)
        ax.imshow(
            np.log1p(shown),
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", BLUE_RAMP),
            vmin=0,
            vmax=np.log1p(vmax),
            aspect="auto",
        )
        ax.grid(False)
        for i, o in enumerate(objs):
            for j, c in enumerate(cols):
                v = int(mat[i, j])
                if v:
                    own = cell_suite[s].get((o, c), 0)
                    lab = f"{v}" if own == v else f"{v}\n({own} here)"
                    ax.text(
                        j,
                        i,
                        lab,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if np.log1p(v) > 0.55 * np.log1p(vmax) else INK,
                    )
        # swaps
        hit = sw.groupby(["obj", "col"]).size()
        for (o, c), n in hit.items():
            i, j = objs.index(o), cols.index(c)
            zero = cell.get((o, c), 0) == 0
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.47, i - 0.42),
                    0.94,
                    0.84,
                    fill=False,
                    lw=1.8,
                    ec="#e34948" if zero else INK,
                    ls="--" if zero else "-",
                )
            )
            ax.text(
                j + 0.44,
                i - 0.36,
                f"{n}×" + ("  0 train" if zero else ""),
                ha="right",
                va="top",
                fontsize=6.5,
                color="#e34948" if zero else INK,
                weight="bold",
            )
        ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
        ax.set_yticks(range(len(objs)), objs)
        ax.set_xlabel("destination (relation:landmark) or [action without destination]")
        ax.set_ylabel("manipulated object (with its locating phrase)")
        n_zero = int((sw.train_ep_all == 0).sum())
        ax.set_title(
            f"{SNAME[s]}: training episodes per (object × destination) cell, all 4 suites "
            f"('(n here)' = this suite).\nBoxes = eval swaps ({len(sw)} actions); red dashed = cell absent from training ({n_zero})",
            loc="left",
            fontsize=9,
        )
        fig.tight_layout()
        savefig(fig, out / f"slot_coverage_{s}.png")
        nums[s] = {"swap_actions": len(sw), "zero": n_zero, "cells": int((mat > 0).sum())}
        M.add(
            f"slot_coverage_{s}.png",
            "B",
            f"{SKO[s]}: 물체 × 목적지 학습 칸과 평가 swap 위치",
            f"학습 칸 {int((mat > 0).sum())}개; swap action {len(sw)}개 중 {n_zero}개가 학습에 없는 칸 (빨간 점선)",
            suite=s,
        )

    # vocabulary
    vrows = []
    for r in rows:
        acts = parse(r["instruction"])
        toks = normalize_tokens(r["instruction"])
        cats = {
            "verb": [a.surface for a in acts],
            "object": [a.obj for a in acts],
            "destination": [a.dest[1] for a in acts if a.dest],
            "spatial": spatial_terms(r["instruction"], acts, toks),
        }
        for cat, words in cats.items():
            for w in set(words):
                vrows.append(
                    {
                        "suite": r["suite"],
                        "category": cat,
                        "term": w,
                        "n_instructions": 1,
                        "n_episodes": r["n_episodes"],
                    }
                )
    vf = (
        pd.DataFrame(vrows)
        .groupby(["suite", "category", "term"], as_index=False)[["n_instructions", "n_episodes"]]
        .sum()
    )
    vf.to_csv(out / "vocab_freq.csv", index=False)
    M.add(
        "vocab_freq.csv",
        "B",
        "suite 별 어휘 빈도",
        "동사·물체·목적지·공간 수식어별 등장 instruction 수와 ep 수",
    )
    for s in SUITES:
        d = vf[vf.suite == s]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        for ax, cat in zip(axes, ["verb", "object", "destination", "spatial"], strict=True):
            c = d[d.category == cat].sort_values("n_episodes")
            ax.barh(range(len(c)), c.n_episodes, color=SCOL[s], height=0.65)
            ax.set_yticks(range(len(c)), c.term)
            ax.set_ylim(-0.6, max(len(c), 6) - 0.4)
            for y, (ne, ni) in enumerate(zip(c.n_episodes, c.n_instructions, strict=True)):
                ax.text(ne, y, f" {ne} ep / {ni} instr", va="center", fontsize=7, color=INK2)
            ax.set_xlim(0, max(1, c.n_episodes.max()) * 1.45)
            ax.set_title(cat)
            ax.set_xlabel("training episodes")
            ax.grid(axis="y", visible=False)
            if c.empty:
                ax.text(0.5, 0.5, "none", transform=ax.transAxes, ha="center", color=MUTED)
        fig.suptitle(
            f"{SNAME[s]}: vocabulary in training instructions (bars = episodes whose instruction contains the term)",
            x=0.01,
            ha="left",
            fontsize=11,
        )
        fig.tight_layout()
        savefig(fig, out / f"vocab_{s}.png")
        top = d[d.category == "object"].sort_values("n_episodes", ascending=False).head(3)
        M.add(
            f"vocab_{s}.png",
            "B",
            f"{SKO[s]}: 동사·물체·목적지·공간 수식어 빈도",
            f"{SNAME[s]} 동사 {len(d[d.category == 'verb'])}종, 물체 {len(d[d.category == 'object'])}종, 목적지 {len(d[d.category == 'destination'])}종; "
            f"가장 흔한 물체 {', '.join(f'{t} ({n} ep)' for t, n in zip(top.term, top.n_episodes, strict=True))}",
            suite=s,
        )
    return {"slot": nums}


def normalize_tokens(s: str) -> list[str]:
    return re.sub(r"[^a-z ]", " ", s.lower()).split()


def spatial_terms(instr: str, acts: list, toks: list[str]) -> list[str]:
    """Spatial words: locating qualifiers of the object ('next to', 'between', 'on', …), the destination relation
    and side words (left/right/top/middle/bottom/back)."""
    terms = []
    for a in acts:
        q = a.obj_ref[len(a.obj) :].strip()
        if q:
            head = q.split(" ")[0]
            terms.append("obj: " + {"next": "next to", "from": "from table center"}.get(head, head))
        if a.dest:
            terms.append("dest: " + a.dest[0])
    for w in ("left", "right", "top", "middle", "bottom", "back"):
        if w in toks:
            terms.append("side: " + w)
    return terms


# ---------------------------------------------------------------- C scene ↔ task


def section_c_map(D: Data, out: Path, M: Manifest) -> dict:
    fig, axes = plt.subplots(1, 4, figsize=(16, 6.2))
    res = {}
    for ax, s in zip(axes, SUITES, strict=True):
        tk = D.tasks[D.tasks.suite == s].sort_values("suite_task_id")
        scenes = list(dict.fromkeys(tk.scene_id))
        ys = {
            sc: (len(scenes) - 1 - i) / max(1, len(scenes) - 1) * 9 if len(scenes) > 1 else 4.5
            for i, sc in enumerate(scenes)
        }
        for r in tk.itertuples():
            yt = 9 - r.suite_task_id
            ax.plot(
                [0, 1],
                [ys[r.scene_id], yt],
                color=SCOL[s],
                lw=1.2 + r.n_episodes / 25,
                alpha=0.7,
                solid_capstyle="round",
            )
            ax.text(
                1.04,
                yt,
                f"t{r.suite_task_id} {wrap(r.instruction, 34)}  ({r.n_episodes} ep)",
                va="center",
                fontsize=6.5,
                color=INK,
                linespacing=0.95,
            )
        for sc, y in ys.items():
            ax.scatter([0], [y], s=70, color=INK2, zorder=3)
            ax.text(
                -0.05, y, sc.replace("-scene-", " scene "), ha="right", va="center", fontsize=7, color=INK2
            )
        if s == "libero_spatial":  # different BDDL, same physical layout (see layout_distance.png)
            y7, y9 = ys["spat-scene-t07"], ys["spat-scene-t09"]
            ax.plot([-0.62, -0.68, -0.68, -0.62], [y7, y7, y9, y9], color="#e34948", lw=1.2)
            ax.text(
                -0.7,
                (y7 + y9) / 2,
                "same\nphysical\nlayout",
                ha="right",
                va="center",
                fontsize=6.5,
                color="#e34948",
            )
        ax.scatter([1] * len(tk), 9 - tk.suite_task_id, s=40, color=SCOL[s], zorder=3)
        ax.set_xlim(-0.95, 2.6)
        ax.set_ylim(-0.6, 9.6)
        ax.axis("off")
        k = len(tk) / len(scenes)
        ax.set_title(f"{SNAME[s]}: {len(scenes)} scene(s) → {len(tk)} tasks", loc="left")
        res[s] = {"scenes": len(scenes), "tasks": len(tk), "tasks_per_scene": round(k, 2)}
    fig.suptitle(
        "Scene (BDDL layout: objects, fixtures, init regions) → training task.  Line width ∝ episodes",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, out / "scene_task_map.png")
    M.add(
        "scene_task_map.png",
        "C",
        "장면(BDDL 배치) ↔ 학습 태스크 대응",
        "BDDL 기준: Spatial 10 장면→10 task, Object 10→10 (1:1); Goal 1 장면→10 task (1:10); LIBERO-10 9 장면→10 task "
        "(t0·t1 이 LIVING_ROOM_SCENE2 공유). 단 Spatial t7·t9 는 BDDL 은 다르지만 물리 배치가 같다 (layout_distance.png)",
    )
    return res


def label_tile(img: np.ndarray, text: str, size: int, fs: int = 13) -> Image.Image:
    im = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    dr = ImageDraw.Draw(im)
    f = font(fs)
    dr.rectangle([0, 0, size, fs + 6], fill=(20, 20, 20))
    dr.text((4, 2), text, fill=(255, 255, 255), font=f)
    return im


def titled_grid(
    tiles: list[Image.Image], captions: list[str], ncol: int, title: str, cap_lines: int = 3, fs: int = 12
) -> Image.Image:
    w, h = tiles[0].size
    ch = cap_lines * (fs + 3) + 6
    th = 34
    nrow = (len(tiles) + ncol - 1) // ncol
    gap = 6
    W = ncol * w + (ncol + 1) * gap
    H = th + nrow * (h + ch + gap) + gap
    canvas = Image.new("RGB", (W, H), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    dr.text((gap, 8), title, fill=(11, 11, 11), font=font(16))
    f = font(fs)
    for k, (t, c) in enumerate(zip(tiles, captions, strict=True)):
        r, q = divmod(k, ncol)
        x, y = gap + q * (w + gap), th + r * (h + ch + gap)
        lines = textwrap.wrap(c, int(w / (fs * 0.55)))[:cap_lines]
        for i, line in enumerate(lines):
            dr.text((x, y + i * (fs + 3)), line, fill=(82, 81, 78), font=f)
        canvas.paste(t, (x, y + ch))
    return canvas


def section_c_grids(D: Data, out: Path, M: Manifest) -> None:
    for s in SUITES:
        for kind in ("first", "last"):
            tiles, caps = [], []
            for t in range(10):
                e = D.rep(s, t)
                img = D.first[e] if kind == "first" else D.last[e]
                tiles.append(label_tile(img, f"t{t}  ep {e}", 256))
                caps.append(f"t{t}: {D.instr(s, t)}")
            g = titled_grid(
                tiles,
                caps,
                5,
                f"{SNAME[s]}: {kind} frame of the representative demo per task (agentview, as stored)",
            )
            save_png(g, out / f"grid_{kind}_{s}.png")
            M.add(
                f"grid_{kind}_{s}.png",
                "C",
                f"{SKO[s]}: task 별 대표 demo 의 {'첫' if kind == 'first' else '마지막'} 프레임 (2×5)",
                ("Goal 은 10 task 첫 장면이 같다; " if s == "libero_goal" and kind == "first" else "")
                + "대표 demo = 길이가 task 중앙값에 가장 가까운 ep",
                suite=s,
            )


def section_c_contact(D: Data, out: Path, M: Manifest) -> None:
    th = 128
    for s in SUITES:
        for t in range(10):
            d = D.eps(s, t)
            es = d.episode_index.to_numpy()
            ims = D.progress[es, 0]  # frame 0, 128 px
            std = ims.astype(np.float32).std(0).mean(-1)
            std_img = (np.clip(std / 40.0, 0, 1) * 255).astype(np.uint8)
            heat = np.asarray(matplotlib.colormaps["magma"](std_img / 255.0)[..., :3] * 255, np.uint8)
            tiles = [label_tile(ims[i], f"ep {e}", th, 10) for i, e in enumerate(es)]
            tiles.append(label_tile(heat, "pixel std", th, 10))
            ncol = 10
            gap = 3
            nrow = (len(tiles) + ncol - 1) // ncol
            W = ncol * th + (ncol + 1) * gap
            H = 50 + nrow * (th + gap) + gap
            c = Image.new("RGB", (W, H), (252, 252, 251))
            dr = ImageDraw.Draw(c)
            dr.text(
                (gap, 4),
                f"{SNAME[s]} t{t}: first frame of all {len(es)} training demos",
                fill=(11, 11, 11),
                font=font(14),
            )
            dr.text((gap, 24), wrap(D.instr(s, t), 150), fill=(82, 81, 78), font=font(12))
            for k, tile in enumerate(tiles):
                r, q = divmod(k, ncol)
                c.paste(tile, (gap + q * (th + gap), 50 + r * (th + gap)))
            save_png(c, out / f"contact_{s}_t{t}.png")
            M.add(
                f"contact_{s}_t{t}.png",
                "C",
                f"{SKO[s]} t{t}: 모든 학습 ep 의 첫 프레임",
                f"{len(es)} ep 의 초기 배치; 마지막 칸 = 픽셀 표준편차 (밝을수록 ep 간 변동 큼, 평균 {std.mean():.1f}/255)",
                suite=s,
                task_id=t,
            )


# ---------------------------------------------------------------- C classifier


def dino_features(imgs: np.ndarray, batch: int = 256) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    torch.set_num_threads(64)
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            x = proc(
                images=list(imgs[i : i + batch]),
                return_tensors="pt",
                do_resize=True,
                size={"height": 224, "width": 224},
                do_center_crop=False,
            )
            feats.append(model(**x).last_hidden_state[:, 0].numpy())
    return np.concatenate(feats).astype(np.float32)


def pixel_features(imgs: np.ndarray) -> np.ndarray:
    x = (
        np.stack([np.asarray(Image.fromarray(i).resize((32, 32), Image.BILINEAR)) for i in imgs])
        .reshape(len(imgs), -1)
        .astype(np.float32)
    )
    return x


def knn_loo(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    if k == 1:
        return y[nn[:, 0]]
    pred = []
    for row in nn:
        c = Counter(y[row])
        best = max(c.values())
        pred.append(next(lbl for lbl in y[row] if c[lbl] == best))  # tie → nearest
    return np.array(pred)


def logreg_cv(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    import torch

    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % folds
    pred = np.zeros_like(y)
    classes = np.unique(y)
    for f in range(folds):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.tensor((X[tr] - mu) / sd)
        yt = torch.tensor(np.searchsorted(classes, y[tr]))
        W = torch.zeros(X.shape[1], len(classes), requires_grad=True)
        b = torch.zeros(len(classes), requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

        def closure() -> Any:  # called inside opt.step below, in this iteration
            opt.zero_grad()  # noqa: B023
            loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + 1e-2 * (W**2).sum()  # noqa: B023
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            pred[te] = classes[(torch.tensor((X[te] - mu) / sd) @ W + b).argmax(1).numpy()]
    return pred


LAYOUT_SAME_CM = 3.0


def layout_distance(cache: Path, s: str) -> np.ndarray:
    """Task × task physical-layout distance (cm) from the eval init renders' object positions (median over the 50
    init states): objects matched by category (Hungarian on xy), distance = the largest matched displacement.
    NaN when the two tasks have different object sets (different scene contents)."""
    from scipy.optimize import linear_sum_assignment

    lay = {}
    for t in range(10):
        r = load_render(cache, s, t)
        assert r is not None, (s, t)
        lay[t] = ([re.sub(r"_\d+$", "", str(n)) for n in r["obj_names"]], np.median(r["obj"], 0))
    dist = np.full((10, 10), np.nan)
    for a in range(10):
        for b in range(10):
            (na, pa), (nb, pb) = lay[a], lay[b]
            if sorted(na) != sorted(nb):
                continue
            cost = np.array(
                [
                    [np.linalg.norm(pa[i, :2] - pb[j, :2]) if na[i] == nb[j] else 9.0 for j in range(len(nb))]
                    for i in range(len(na))
                ]
            )
            r_, c_ = linear_sum_assignment(cost)
            dist[a, b] = cost[r_, c_].max() * 100
    return dist


def layout_groups(dist: np.ndarray) -> list[list[int]]:
    """Connected components of 'same physical layout' (distance < LAYOUT_SAME_CM)."""
    groups: list[list[int]] = []
    seen: set[int] = set()
    for a in range(10):
        if a in seen:
            continue
        comp, stack = [], [a]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack += [y for y in range(10) if y not in seen and dist[x, y] < LAYOUT_SAME_CM]
        groups.append(sorted(comp))
    return groups


def section_c_layout(D: Data, out: Path, M: Manifest, cache: Path) -> dict:
    rows, res = [], {}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    for ax, s in zip(axes, SUITES, strict=True):
        dist = layout_distance(cache, s)
        groups = layout_groups(dist)
        n = D.eps(s).groupby("suite_task_id").size()
        ceiling = float(sum(max(n[t] for t in g) for g in groups) / n.sum())
        res[s] = {"groups": [g for g in groups if len(g) > 1], "n_layouts": len(groups), "ceiling": ceiling}
        for a in range(10):
            for b in range(10):
                rows.append({"suite": s, "task_a": a, "task_b": b, "layout_dist_cm": dist[a, b]})
        shown = np.where(np.isnan(dist), np.nan, np.minimum(dist, 30))
        ax.imshow(
            shown,
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", BLUE_RAMP[::-1] + [SURF]),
            vmin=0,
            vmax=30,
        )
        ax.grid(False)
        for a in range(10):
            for b in range(10):
                v = dist[a, b]
                ax.text(
                    b,
                    a,
                    "≠" if np.isnan(v) else f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=MUTED if np.isnan(v) else ("white" if v < 12 else INK),
                )
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("task")
        ax.set_ylabel("task")
        multi = [g for g in groups if len(g) > 1]
        sub = ", ".join("/".join(f"t{t}" for t in g) for g in multi) if multi and len(multi[0]) < 10 else ""
        ax.set_title(
            f"{SNAME[s]}: {len(groups)} distinct layout(s)" + (f"\nsame layout: {sub}" if sub else ""),
            loc="left",
            fontsize=9,
        )
    pd.DataFrame(rows).to_csv(out / "layout_distance.csv", index=False)
    fig.suptitle(
        "Physical layout distance between tasks (cm; largest object displacement, objects matched by category; "
        f"≠ = different object set).  < {LAYOUT_SAME_CM:g} cm = same starting scene",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    savefig(fig, out / "layout_distance.png")
    M.add(
        "layout_distance.csv",
        "C",
        "task 간 물리적 초기 배치 거리 표",
        "평가 init 렌더 물체 위치(50개 중앙값)로 계산한 task 쌍별 최대 물체 변위 (cm), 물체 집합이 다르면 NaN",
    )
    M.add(
        "layout_distance.png",
        "C",
        "task 간 초기 배치 거리: 어떤 task 들이 같은 장면에서 시작하나",
        "Spatial 은 t7·t9 가 같은 배치 (bowl 두 개의 역할만 바뀜); Object 는 물체 집합이 모두 다름; "
        "Goal 10 task 한 배치; LIBERO-10 t0·t1 한 배치. 첫 프레임 분류 상한: "
        + ", ".join(f"{SNAME[s]} {res[s]['ceiling']:.0%}" for s in SUITES),
    )
    return res


def section_c_classify(D: Data, out: Path, M: Manifest, cache: Path, layout: dict) -> dict:
    fpath = cache / "features.npz"
    E = D.progress.shape[0]
    F = len(D.fracs)
    if fpath.exists():
        z = np.load(fpath)
        dino, pix = z["dino"], z["pix"]
    else:
        flat = D.progress.reshape(E * F, 128, 128, 3)
        dino = dino_features(flat).reshape(E, F, -1)
        pix = pixel_features(flat).reshape(E, F, -1)
        np.savez(fpath, dino=dino, pix=pix)
    rows, conf = [], {}
    ep = D.ep.set_index("episode_index")
    for s in SUITES:
        idx = D.eps(s).episode_index.to_numpy()
        y = ep.loc[idx, "suite_task_id"].to_numpy()
        for fi, frac in enumerate(D.fracs):
            for feat_name, X in (("dino", dino[idx, fi]), ("pixels", pix[idx, fi])):
                Xz = X if feat_name == "dino" else (X - X.mean(0)) / (X.std(0) + 1e-6)
                methods = {
                    "1nn": knn_loo(Xz, y, 1),
                    "5nn": knn_loo(Xz, y, 5),
                    "logreg5fold": logreg_cv(X, y),
                }
                for m, p in methods.items():
                    k, n = int((p == y).sum()), len(y)
                    lo, hi = wilson(k, n)
                    rows.append(
                        {
                            "suite": s,
                            "frac": float(frac),
                            "feature": feat_name,
                            "method": m,
                            "acc": k / n,
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "n_ep": n,
                            "chance": 0.1,
                            "layout_ceiling": layout[s]["ceiling"],
                        }
                    )
                    if fi == 0 and feat_name == "dino" and m == "logreg5fold":
                        cm = np.zeros((10, 10), int)
                        for a, b in zip(y, p, strict=True):
                            cm[a, b] += 1
                        conf[s] = cm
    acc = pd.DataFrame(rows)
    acc.to_csv(out / "frame_task_acc_by_time.csv", index=False)
    a0 = acc[acc.frac == 0].drop(columns="frac")
    a0.to_csv(out / "first_frame_task_acc.csv", index=False)
    main = a0[(a0.feature == "dino") & (a0.method == "logreg5fold")].set_index("suite").acc
    ceil = {s: layout[s]["ceiling"] for s in SUITES}
    M.add(
        "first_frame_task_acc.csv",
        "C",
        "첫 프레임만으로 task 맞히기: suite 별 정확도",
        "dino logreg 5-fold: "
        + ", ".join(f"{SNAME[s]} {main[s]:.1%}" for s in SUITES)
        + " (chance 10%; 같은 배치 task 를 고려한 상한 "
        + ", ".join(f"{ceil[s]:.0%}" for s in SUITES)
        + "); pixels·1-NN·5-NN 포함",
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, s in zip(axes, SUITES, strict=True):
        cm = conf[s]
        ax.imshow(
            cm / cm.sum(1, keepdims=True),
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", [SURF] + BLUE_RAMP),
            vmin=0,
            vmax=1,
        )
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if cm[i, j] / cm[i].sum() > 0.55 else INK,
                    )
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("predicted task")
        ax.set_ylabel("true task")
        ax.set_title(f"{SNAME[s]}: {main[s]:.1%}  (chance 10%, layout ceiling {ceil[s]:.0%})", fontsize=9)
    fig.suptitle(
        "Task from the first agentview frame alone (DINOv2-small CLS, 5-fold logistic regression). Cells = episodes.\n"
        "Ceiling = best possible accuracy when tasks that share a starting layout are indistinguishable",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out / "first_frame_confusion.png")
    M.add(
        "first_frame_confusion.png",
        "C",
        "첫 프레임 → task 혼동행렬 (suite 별)",
        "대각선 = 맞힘. "
        + ", ".join(f"{SNAME[s]} {main[s]:.1%}" for s in SUITES)
        + "; Goal 은 대각선이 없다 (장면 하나), Spatial t7↔t9 · LIBERO-10 t0↔t1 는 같은 배치라 섞인다",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, (feat, meth) in zip(axes, (("dino", "logreg5fold"), ("pixels", "1nn")), strict=True):
        for s in SUITES:
            d = acc[(acc.suite == s) & (acc.feature == feat) & (acc.method == meth)]
            ax.fill_between(d.frac * 100, d.ci_lo, d.ci_hi, color=SCOL[s], alpha=0.12, lw=0)
            ax.plot(d.frac * 100, d.acc, color=SCOL[s], lw=2, marker="o", ms=4, label=SNAME[s])
        ends = sorted(
            (acc[(acc.suite == s) & (acc.feature == feat) & (acc.method == meth)].acc.iloc[-1], s)
            for s in SUITES
        )
        ly = -1.0
        for v, s in ends:  # end labels, dodged ≥ 0.045 apart
            ly = max(v, ly + 0.045)
            ax.text(101.5, ly, SNAME[s], color=INK2, va="center", fontsize=8)
        ax.axhline(0.1, color=MUTED, lw=1, ls="--")
        ax.text(1, 0.115, "chance 1/10", color=MUTED, fontsize=8)
        ax.set_xlim(0, 112)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("episode progress at the classified frame (%)")
        ax.set_title(
            f"{feat} "
            + ("logistic regression (5-fold)" if meth == "logreg5fold" else "1-NN (leave-one-episode-out)"),
            loc="left",
        )
    axes[0].set_ylabel("task accuracy within suite")
    axes[0].legend(loc="lower right")
    fig.suptitle(
        "When does the scene alone determine the task?  Accuracy of frame → task by progress (band = Wilson 95% CI)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out / "frame_task_acc_by_time.png")
    g = (
        acc[(acc.suite == "libero_goal") & (acc.feature == "dino") & (acc.method == "logreg5fold")]
        .set_index("frac")
        .acc
    )
    M.add(
        "frame_task_acc_by_time.csv",
        "C",
        "진행률별 프레임 → task 정확도 표",
        f"{F} 시점 × suite × (dino, pixels) × (1-NN, 5-NN, logreg)",
    )
    M.add(
        "frame_task_acc_by_time.png",
        "C",
        "진행률에 따라 장면만으로 task 가 정해지는 시점",
        f"Goal (dino logreg): 0% {g.iloc[0]:.0%} → 10% {g.loc[0.1]:.0%} → 20% {g.loc[0.2]:.0%} → 60% {g.loc[0.6]:.0%} "
        f"→ 100% {g.iloc[-1]:.0%}; 나머지 suite 는 t=0 부터 배치 상한 근처",
    )
    curves = {
        s: acc[(acc.suite == s) & (acc.feature == "dino") & (acc.method == "logreg5fold")]
        .set_index("frac")
        .acc.to_dict()
        for s in SUITES
    }
    return {"acc_t0": a0.to_dict("records"), "curves_dino_logreg": curves}


# ---------------------------------------------------------------- D trajectories


def topdown(ax: Any) -> None:
    """Axes as seen from the agentview camera: world +x down the page, world +y to the left."""
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_aspect("equal")


def load_render(cache: Path, s: str, t: int) -> dict | None:
    p = cache / "render" / f"{s}_t{t}.npz"
    return dict(np.load(p)) if p.exists() else None


def section_d(D: Data, out: Path, M: Manifest, cache: Path) -> dict:
    gp_rows = []
    for e in D.ep.episode_index:
        st, ac = D.traj(int(e))
        g, p = grasp_place(ac)
        r = D.ep.loc[D.ep.episode_index == e].iloc[0]
        gp_rows.append(
            {
                "episode_index": int(e),
                "suite": r.suite,
                "task_id": int(r.suite_task_id),
                "length": int(r.length),
                "grasp_t": g,
                "grasp_x": st[g, 0] if g is not None else np.nan,
                "grasp_y": st[g, 1] if g is not None else np.nan,
                "grasp_z": st[g, 2] if g is not None else np.nan,
                "place_t": p,
                "place_x": st[p, 0] if p is not None else np.nan,
                "place_y": st[p, 1] if p is not None else np.nan,
                "place_z": st[p, 2] if p is not None else np.nan,
                "n_close_events": int(((ac[1:, 6] > 0) & (ac[:-1, 6] <= 0)).sum() + (ac[0, 6] > 0)),
            }
        )
    gp = pd.DataFrame(gp_rows)
    gp.to_csv(out / "grasp_place.csv", index=False)
    M.add(
        "grasp_place.csv",
        "D",
        "ep 별 첫 파지·놓기 시점과 eef 위치",
        "grasp = gripper action 이 처음 닫힘(+), place = 그 뒤 처음 열림(−); world 좌표 (m)",
    )
    res = {}
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("t", BLUE_RAMP[1:])
    for s in SUITES:
        sub = D.eps(s)
        allxy = np.concatenate([D.traj(int(e))[0][:, :2] for e in sub.episode_index])
        lo, hi = allxy.min(0) - 0.03, allxy.max(0) + 0.03
        fig, axes = plt.subplots(2, 5, figsize=(16, 7.6), sharex=True, sharey=True)
        for t in range(10):
            ax = axes.flat[t]
            rnd = load_render(cache, s, t)
            if rnd is not None:  # object positions at the eval init states (same init distribution)
                for k, nm in enumerate(rnd["obj_names"]):
                    o = rnd["obj"][:, k]
                    ax.scatter(o[:, 1], o[:, 0], s=4, color=MUTED, alpha=0.35, lw=0)
                    ax.text(
                        np.median(o[:, 1]),
                        np.median(o[:, 0]),
                        re.sub(r"_\d+$", "", str(nm)).replace("_", " ")[:16],
                        fontsize=5.5,
                        color=MUTED,
                        ha="center",
                        va="bottom",
                    )
            for e in D.eps(s, t).episode_index:
                st = D.traj(int(e))[0]
                pts = st[:, [1, 0]]
                seg = np.stack([pts[:-1], pts[1:]], 1)
                lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(0, 1), lw=0.8, alpha=0.5)
                lc.set_array(np.linspace(0, 1, len(seg)))
                ax.add_collection(lc)
            gt = gp[(gp.suite == s) & (gp.task_id == t)]
            ax.scatter(
                gt.grasp_y,
                gt.grasp_x,
                s=10,
                marker="o",
                facecolor="none",
                edgecolor="#eb6834",
                lw=0.8,
                label="grasp",
            )
            ax.scatter(gt.place_y, gt.place_x, s=12, marker="x", color=INK, lw=0.8, label="place")
            ax.set_xlim(lo[1], hi[1])
            ax.set_ylim(lo[0], hi[0])
            topdown(ax)
            ax.set_title(f"t{t}: {wrap(D.instr(s, t), 42)}", fontsize=7, loc="left")
            if t >= 5:
                ax.set_xlabel("world y (m)  ← left")
            if t % 5 == 0:
                ax.set_ylabel("world x (m)  ↓ toward camera")
        axes.flat[0].legend(loc="lower left", fontsize=6.5)
        fig.suptitle(
            f"{SNAME[s]}: end-effector xy of every training demo, seen from the agentview side (light → dark = time). "
            "o grasp, x place; grey = object positions at eval init states",
            x=0.01,
            ha="left",
            fontsize=10,
        )
        fig.tight_layout()
        savefig(fig, out / f"traj_xy_{s}.png")
        M.add(
            f"traj_xy_{s}.png",
            "D",
            f"{SKO[s]}: task 별 모든 ep 의 eef xy 궤적",
            f"{len(sub)} ep, 2×5 small multiples, 시간에 따라 진해짐; 주황 o = 첫 파지, × = 놓기, 회색 = 물체 초기 위치",
            suite=s,
        )

        # grasp/place
        g = gp[gp.suite == s]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharex=True, sharey=True)
        for ax, (kind, mk) in zip(axes, (("grasp", "o"), ("place", "^")), strict=True):
            for t in range(10):
                d = g[g.task_id == t]
                ax.scatter(
                    d[f"{kind}_y"],
                    d[f"{kind}_x"],
                    s=16,
                    marker=mk,
                    color=TASK10[t],
                    alpha=0.75,
                    lw=0,
                    label=f"t{t}",
                )
                if d[f"{kind}_x"].notna().any():
                    ax.text(
                        d[f"{kind}_y"].median(),
                        d[f"{kind}_x"].median(),
                        f"t{t}",
                        fontsize=8,
                        weight="bold",
                        color=INK,
                        ha="center",
                        va="center",
                        bbox={
                            "boxstyle": "round,pad=0.15",
                            "fc": "white",
                            "ec": TASK10[t],
                            "lw": 1,
                            "alpha": 0.85,
                        },
                    )
            ax.set_xlim(lo[1], hi[1])
            ax.set_ylim(lo[0], hi[0])
            topdown(ax)
            ax.set_title(
                f"first {kind} (eef xy when gripper {'closes' if kind == 'grasp' else 're-opens'})",
                loc="left",
            )
            ax.set_xlabel("world y (m)  ← left")
        axes[0].set_ylabel("world x (m)  ↓ toward camera")
        axes[1].legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.3), fontsize=7)
        # numbers
        cg = g.groupby("task_id")[["grasp_x", "grasp_y"]].median().to_numpy()
        cp = g.groupby("task_id")[["place_x", "place_y"]].median().to_numpy()
        dg = np.nanmean([np.hypot(*(a - b)) for i, a in enumerate(cg) for j, b in enumerate(cg) if i < j])
        dpl = np.nanmean([np.hypot(*(a - b)) for i, a in enumerate(cp) for j, b in enumerate(cp) if i < j])
        fig.suptitle(
            f"{SNAME[s]}: where demos grasp and place, coloured by task (label at task median).  "
            f"Mean distance between task medians: grasp {dg * 100:.0f} cm, place {dpl * 100:.0f} cm",
            x=0.01,
            ha="left",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        savefig(fig, out / f"grasp_place_{s}.png")
        res[s] = {
            "grasp_between_task_cm": round(dg * 100, 1),
            "place_between_task_cm": round(dpl * 100, 1),
            "no_grasp": int(g.grasp_t.isna().sum()),
        }
        M.add(
            f"grasp_place_{s}.png",
            "D",
            f"{SKO[s]}: 첫 파지 지점과 놓는 지점 (task 색)",
            f"task 중앙값 간 평균 거리: 파지 {dg * 100:.0f} cm, 놓기 {dpl * 100:.0f} cm"
            + (" — Spatial 은 집는 곳만 다르고 놓는 곳(plate)은 같다" if s == "libero_spatial" else ""),
            suite=s,
        )

        # overlays
        for t in range(10):
            rnd = load_render(cache, s, t)
            if rnd is None:
                continue
            P = rnd["P"]
            e0 = D.rep(s, t)
            base = Image.fromarray(D.first[e0]).resize((512, 512), Image.BILINEAR)
            ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(ov)
            for e in D.eps(s, t).episode_index:
                st, ac = D.traj(int(e))
                uv = project(P, st[:, :3]) * 2
                n = len(uv)
                for i in range(n - 1):
                    c = matplotlib.colors.to_rgb(cmap(i / max(1, n - 1)))
                    dr.line([tuple(uv[i]), tuple(uv[i + 1])], fill=(*(int(255 * v) for v in c), 110), width=2)
                gi, pi = grasp_place(ac)
                for ti, col in ((gi, (235, 104, 52, 230)), (pi, (11, 11, 11, 230))):
                    if ti is not None:
                        x, y = uv[ti]
                        dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=col, width=2)
            img = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
            canvas = Image.new("RGB", (512, 562), (252, 252, 251))
            d2 = ImageDraw.Draw(canvas)
            d2.text(
                (4, 2),
                f"{SNAME[s]} t{t}: eef path of all {len(D.eps(s, t))} demos on ep {e0}'s first frame",
                fill=(11, 11, 11),
                font=font(13),
            )
            d2.text((4, 20), wrap(D.instr(s, t), 78), fill=(82, 81, 78), font=font(11), spacing=2)
            canvas.paste(img, (0, 50))
            save_png(canvas, out / f"overlay_{s}_t{t}.png")
            M.add(
                f"overlay_{s}_t{t}.png",
                "D",
                f"{SKO[s]} t{t}: eef 궤적을 agentview 첫 프레임에 투영",
                "LIBERO env 의 agentview 카메라 행렬로 투영 (밝→진 = 시간, 주황 원 = 첫 파지, 검은 원 = 놓기)",
                suite=s,
                task_id=t,
            )
    return res


def project(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    p = np.c_[xyz, np.ones(len(xyz))] @ P.T
    return np.c_[255 - p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]]


# ---------------------------------------------------------------- E actions

DIMS = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


def section_e(D: Data, out: Path, M: Manifest) -> dict:
    suite_of = D.ep.set_index("episode_index").suite
    fs = suite_of.loc[D.fep].to_numpy()
    rows = []
    for s in SUITES:
        a = D.action[fs == s]
        for k, dname in enumerate(DIMS):
            v = a[:, k]
            rows.append(
                {
                    "suite": s,
                    "dim": dname,
                    "n_frames": len(v),
                    "mean": float(v.mean()),
                    "std": float(v.std()),
                    "min": float(v.min()),
                    "p01": float(np.quantile(v, 0.01)),
                    "median": float(np.median(v)),
                    "p99": float(np.quantile(v, 0.99)),
                    "max": float(v.max()),
                    "frac_zero": float((v == 0).mean()),
                    "frac_abs_ge_0.9": float((np.abs(v) >= 0.9).mean()),
                    "frac_positive": float((v > 0).mean()),
                }
            )
    sm = pd.DataFrame(rows)
    sm.to_csv(out / "actions_summary.csv", index=False)
    close = sm[sm.dim == "gripper"].set_index("suite").frac_positive
    M.add(
        "actions_summary.csv",
        "E",
        "suite × action 차원 요약 통계",
        "평균·표준편차·분위수·0 비율·|a|≥0.9 비율; gripper 는 frac_positive = 닫힘 비율",
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    bins = np.linspace(-1, 1, 81)
    for k, dname in enumerate(DIMS[:6]):
        ax = axes.flat[k]
        for s in SUITES:
            v = D.action[fs == s, k]
            h, _ = np.histogram(v, bins=bins, density=True)
            ax.step(bins[:-1], h, where="post", color=SCOL[s], lw=1.4, label=SNAME[s])
        ax.set_yscale("log")
        ax.set_title(dname, loc="left")
        ax.set_xlabel("action value")
    axes.flat[0].set_ylabel("density (log)")
    axes.flat[4].set_ylabel("density (log)")
    axes.flat[0].legend()
    idle = {s: float((np.abs(D.action[fs == s, :6]).max(1) == 0).mean()) for s in SUITES}
    for ax, vals, ttl in (
        (axes.flat[6], close, "gripper action: fraction of frames closed (+1)"),
        (axes.flat[7], idle, "fraction of frames with all 6 motion dims = 0"),
    ):
        ax.barh(range(4), [vals[s] for s in SUITES], color=[SCOL[s] for s in SUITES], height=0.6)
        ax.set_yticks(range(4), [SNAME[s] for s in SUITES])
        for i, s in enumerate(SUITES):
            ax.text(vals[s], i, f" {vals[s]:.1%}", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, 1 if vals is close else max(vals.values()) * 1.4)
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Action distributions per suite (7-d delta EEF + gripper)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    savefig(fig, out / "actions_hist.png")
    M.add(
        "actions_hist.png",
        "E",
        "suite 별 7차원 action 분포와 gripper 닫힘 비율",
        "gripper 닫힘 비율 "
        + ", ".join(f"{SNAME[s]} {close[s]:.0%}" for s in SUITES)
        + "; 정지(6차원 모두 0) frame 비율 "
        + ", ".join(f"{idle[s]:.1%}" for s in SUITES),
    )

    # |action| by progress
    nb = 20
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    prog_rows = []
    for s in SUITES:
        acc = {
            "trans": [[] for _ in range(nb)],
            "rot": [[] for _ in range(nb)],
            "grip": [[] for _ in range(nb)],
        }
        for e in D.eps(s).episode_index:
            _, a = D.traj(int(e))
            b = np.minimum((np.arange(len(a)) / len(a) * nb).astype(int), nb - 1)
            for name, v in (
                ("trans", np.linalg.norm(a[:, :3], axis=1)),
                ("rot", np.linalg.norm(a[:, 3:6], axis=1)),
                ("grip", (a[:, 6] > 0).astype(float)),
            ):
                for i in range(nb):
                    acc[name][i].append(v[b == i].mean())
        x = (np.arange(nb) + 0.5) / nb * 100
        for ax, name, lab in zip(
            axes,
            ("trans", "rot", "grip"),
            ("mean |Δxyz| (action units)", "mean |Δrot| (action units)", "fraction gripper closed"),
            strict=True,
        ):
            y = np.array([np.mean(v) for v in acc[name]])
            ax.plot(x, y, color=SCOL[s], lw=2, label=SNAME[s])
            ax.set_xlabel("episode progress (%)")
            ax.set_title(lab, loc="left")
            for i in range(nb):
                prog_rows.append({"suite": s, "quantity": name, "progress_bin_center": x[i], "mean": y[i]})
    axes[0].legend()
    fig.suptitle(
        "Action magnitude over the episode (mean over demos, 20 progress bins)",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, out / "actions_progress.png")
    pd.DataFrame(prog_rows).to_csv(out / "actions_progress.csv", index=False)
    M.add(
        "actions_progress.png",
        "E",
        "ep 진행률에 따른 |action| 평균",
        "이동·회전 크기와 gripper 닫힘 비율을 진행률 20 구간 평균으로",
    )
    M.add("actions_progress.csv", "E", "진행률별 |action| 평균 표", "suite × (trans, rot, grip) × 20 구간")
    return {"close_frac": close.to_dict(), "idle_frac": idle}


# ---------------------------------------------------------------- F videos


def section_f(D: Data, out: Path, M: Manifest, cache: Path) -> None:
    for s in SUITES:
        for t in range(10):
            e = D.rep(s, t)
            z = np.load(cache / "rep" / f"ep{e}.npz")
            ag, wr = z["agent"], z["wrist"]
            L = len(ag)
            title = f"{SNAME[s]} t{t} · ep {e} · {L} frames"
            instr = D.instr(s, t)
            path = out / f"demo_{s}_t{t}.gif"
            for side, stride, colors in (
                (256, 2, 128),
                (224, 2, 96),
                (192, 2, 64),
                (192, 3, 64),
                (160, 3, 48),
                (160, 4, 32),
            ):
                fps = 10 / stride
                if fps > 8:
                    continue
                frames = []
                for i in range(0, L, stride):
                    frames.append(demo_frame(ag[i], wr[i], side, title + f" · t={i / 10:.1f}s", instr))
                pal = frames[len(frames) // 2].quantize(colors, method=Image.Quantize.MEDIANCUT)
                q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
                q[0].save(
                    path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True
                )
                if path.stat().st_size <= 1.5e6:
                    break
            assert path.stat().st_size <= 1.5e6, path
            M.add(
                f"demo_{s}_t{t}.gif",
                "F",
                f"{SKO[s]} t{t}: 대표 demo (agentview | wrist)",
                f"ep {e}, {L} frame (task 중앙값에 가장 가까운 길이), {side * 2}×{side} px, {fps:g} fps (실시간)",
                suite=s,
                task_id=t,
            )
            # filmstrip
            ks = np.linspace(0, L - 1, 8).round().astype(int)
            tw = 195
            gap = 4
            W = 8 * tw + 9 * gap
            c = Image.new("RGB", (W, 48 + 2 * (tw + gap) + 18), (252, 252, 251))
            dr = ImageDraw.Draw(c)
            dr.text(
                (gap, 4),
                f"{title} · 8 keyframes (top agentview, bottom wrist)",
                fill=(11, 11, 11),
                font=font(14),
            )
            dr.text((gap, 24), wrap(instr, 170), fill=(82, 81, 78), font=font(12))
            for j, k in enumerate(ks):
                x = gap + j * (tw + gap)
                c.paste(Image.fromarray(ag[k]).resize((tw, tw), Image.BILINEAR), (x, 48))
                c.paste(Image.fromarray(wr[k]).resize((tw, tw), Image.BILINEAR), (x, 48 + tw + gap))
                dr.text(
                    (x, 48 + 2 * (tw + gap)),
                    f"t={k / 10:.1f}s ({k / (L - 1):.0%})",
                    fill=(82, 81, 78),
                    font=font(11),
                )
            save_png(c, out / f"filmstrip_{s}_t{t}.png")
            M.add(
                f"filmstrip_{s}_t{t}.png",
                "F",
                f"{SKO[s]} t{t}: 대표 demo 8 keyframe",
                f"ep {e} 의 0–100% 균등 8 시점, agentview 와 wrist",
                suite=s,
                task_id=t,
            )


def demo_frame(a: np.ndarray, w: np.ndarray, side: int, title: str, instr: str) -> Image.Image:
    head = 30
    im = Image.new("RGB", (2 * side, side + head), (20, 20, 20))
    im.paste(Image.fromarray(a).resize((side, side), Image.BILINEAR), (0, head))
    im.paste(Image.fromarray(w).resize((side, side), Image.BILINEAR), (side, head))
    dr = ImageDraw.Draw(im)
    dr.text((3, 1), title, fill=(255, 255, 255), font=font(11))
    dr.text((3, 15), textwrap.shorten(instr, int(2 * side / 5.6)), fill=(200, 200, 200), font=font(10))
    return im


# ---------------------------------------------------------------- G eval init vs train

# what the side-by-side images show (checked by eye, init_compare_<suite>.png)
RENDER_NOTE = {
    "libero_spatial": "배치·조명 모두 학습과 같아 보인다",
    "libero_goal": "배치·조명 모두 학습과 같아 보인다",
    "libero_object": "물체 배치는 같지만 평가 렌더의 바닥이 더 어둡고 갈색이다 (텍스처·조명 차이)",
    "libero_10": "t0·t1·t7 (LIVING_ROOM_SCENE1/2) 은 학습 프레임의 milk·orange juice·ketchup 이 평가 렌더의 테이블 위에 없다 "
    "(sim 위치 z≈0.9–1.05 m, 테이블 위 물체 z≈0.45 m); 나머지 task 는 같아 보인다",
}


def section_g(D: Data, out: Path, M: Manifest, cache: Path) -> dict:
    feats = np.load(cache / "features.npz")["dino"][:, 0]  # train frame 0 (128 px → 224)
    rows = []
    rimgs = {}
    for s in SUITES:
        for t in range(10):
            r = load_render(cache, s, t)
            if r is None:
                continue
            rimgs[(s, t)] = r
    if not rimgs:
        return {}
    keys = list(rimgs)
    allimgs = np.concatenate(
        [
            np.stack(
                [np.asarray(Image.fromarray(i).resize((128, 128), Image.BILINEAR)) for i in rimgs[k]["img"]]
            )
            for k in keys
        ]
    )
    ef = dino_features(allimgs)
    off = 0
    for k in keys:
        n = len(rimgs[k]["img"])
        rimgs[k]["feat"] = ef[off : off + n]
        off += n
    ep = D.ep.set_index("episode_index")
    for s in SUITES:
        idx = D.eps(s).episode_index.to_numpy()
        y = ep.loc[idx, "suite_task_id"].to_numpy()
        Xt = feats[idx] / np.linalg.norm(feats[idx], axis=1, keepdims=True)
        S = Xt @ Xt.T
        np.fill_diagonal(S, -np.inf)
        for t in range(10):
            if (s, t) not in rimgs:
                continue
            r = rimgs[(s, t)]
            Xe = r["feat"] / np.linalg.norm(r["feat"], axis=1, keepdims=True)
            Se = Xe @ Xt.T  # eval init × train eps (whole suite)
            pred = y[Se.argmax(1)]
            same = y == t
            d_eval = 1 - Se[:, same].max(1)  # cosine distance to the nearest training demo of the same task
            d_train = 1 - S[np.ix_(same, same)].max(1)  # train LOO, same task
            thr = np.quantile(d_train, 0.95)
            # eef start position
            st0 = np.stack([D.traj(int(e))[0][0, :3] for e in idx[same]])
            rows.append(
                {
                    "suite": s,
                    "task_id": t,
                    "n_eval_init": len(Xe),
                    "n_train_ep": int(same.sum()),
                    "eval_task_acc_1nn": float((pred == t).mean()),
                    "eval_nn_dist_median": float(np.median(d_eval)),
                    "train_loo_nn_dist_median": float(np.median(d_train)),
                    "frac_eval_within_train_p95": float((d_eval <= thr).mean()),
                    "eef_start_train_mean": np.round(st0.mean(0), 3).tolist(),
                    "eef_start_eval_mean": np.round(r["eef"].mean(0), 3).tolist(),
                }
            )
            r["d_eval"], r["d_train"] = d_eval, d_train
    iv = pd.DataFrame(rows)
    iv.to_csv(out / "init_vs_train.csv", index=False)
    acc_s = iv.groupby("suite").apply(
        lambda d: (d.eval_task_acc_1nn * d.n_eval_init).sum() / d.n_eval_init.sum(), include_groups=False
    )
    within = iv.groupby("suite").apply(
        lambda d: (d.frac_eval_within_train_p95 * d.n_eval_init).sum() / d.n_eval_init.sum(),
        include_groups=False,
    )
    M.add(
        "init_vs_train.csv",
        "G",
        "평가 init 렌더 vs 학습 첫 프레임: 거리와 task 예측",
        "task 별: 평가 init 50개의 학습 최근접 거리(DINOv2 cosine), 학습 LOO 거리, 학습 p95 안 비율, 학습 kNN 으로 예측한 task 정확도",
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True)
    for ax, s in zip(axes, SUITES, strict=True):
        for t in range(10):
            r = rimgs.get((s, t))
            if r is None:
                continue
            ax.boxplot(
                [r["d_train"]],
                positions=[t - 0.18],
                widths=0.3,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "#e1e0d9", "color": MUTED},
                medianprops={"color": INK},
                whiskerprops={"color": MUTED},
                capprops={"color": MUTED},
            )
            ax.boxplot(
                [r["d_eval"]],
                positions=[t + 0.18],
                widths=0.3,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": SCOL[s], "color": SCOL[s]},
                medianprops={"color": INK},
                whiskerprops={"color": SCOL[s]},
                capprops={"color": SCOL[s]},
            )
        ax.set_xticks(range(10), [str(t) for t in range(10)])
        ax.set_xlim(-0.6, 9.6)
        ax.set_xlabel("task id")
        ax.set_title(
            f"{SNAME[s]}: eval→train task acc {acc_s[s]:.0%}, within p95 {within[s]:.0%}",
            loc="left",
            fontsize=9,
        )
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("cosine distance to nearest training\nfirst frame of the same task (DINOv2)")
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[
            Patch(fc="#e1e0d9", ec=MUTED, label="training demo (leave-one-out)"),
            Patch(fc=CAT[0], label="eval init state render"),
        ],
        loc="upper left",
    )
    fig.suptitle(
        "Are the eval init scenes inside the training distribution?  (grey = train vs train, colour = eval init vs train)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out / "init_vs_train.png")
    M.add(
        "init_vs_train.png",
        "G",
        "평가 init 장면이 학습 분포 안에 있는가 (거리 분포)",
        "학습 첫 프레임 kNN 으로 평가 init task 예측: "
        + ", ".join(f"{SNAME[s]} {acc_s[s]:.0%}" for s in SUITES)
        + "; 학습 LOO p95 거리 안: "
        + ", ".join(f"{within[s]:.0%}" for s in SUITES)
        + " (렌더 조명·바닥 차이도 거리에 들어간다)",
    )

    for s in SUITES:
        tw = 150
        gap = 4
        cols = 6  # 3 train, 3 eval
        W = 230 + cols * (tw + gap)
        H = 46 + 10 * (tw + gap)
        c = Image.new("RGB", (W, H), (252, 252, 251))
        dr = ImageDraw.Draw(c)
        dr.text(
            (4, 4),
            f"{SNAME[s]}: training demo first frames (left 3) vs eval init-state renders 0–2 (right 3)",
            fill=(11, 11, 11),
            font=font(14),
        )
        dr.text(
            (4, 24),
            "renders: LiberoEnv hard reset + set_init_state + 10 no-op steps, 256 px, flipped 180° like the eval videos",
            fill=(82, 81, 78),
            font=font(11),
        )
        for t in range(10):
            y0 = 46 + t * (tw + gap)
            dr.text(
                (4, y0 + 2),
                "\n".join(textwrap.wrap(f"t{t}: {D.instr(s, t)}", 30)[:7]),
                fill=(11, 11, 11),
                font=font(11),
            )
            es = D.eps(s, t).episode_index.to_numpy()[:3]
            for j, e in enumerate(es):
                c.paste(label_tile(D.first[e], f"train ep {e}", tw, 10), (230 + j * (tw + gap), y0))
            r = rimgs.get((s, t))
            if r is not None:
                for j in range(3):
                    c.paste(
                        label_tile(r["img"][j], f"eval init {j}", tw, 10),
                        (230 + (3 + j) * (tw + gap) + 6, y0),
                    )
        save_png(c, out / f"init_compare_{s}.png")
        M.add(
            f"init_compare_{s}.png",
            "G",
            f"{SKO[s]}: 학습 첫 프레임 vs 평가 init 렌더 나란히",
            f"task 당 학습 3장 | 평가 init 3장; 학습 p95 거리 안 {within[s]:.0%}, 최근접 학습 ep 의 task 일치 {acc_s[s]:.0%}. "
            + RENDER_NOTE[s],
            suite=s,
        )
    return {"eval_task_acc": acc_s.to_dict(), "within_p95": within.to_dict()}


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sections", default="ABCDEFG")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    D = Data(args.cache)
    M = Manifest(args.out)
    numbers: dict[str, Any] = {}
    secs = args.sections
    if "A" in secs:
        numbers["A"] = section_a(D, args.out, M)
    if "B" in secs:
        numbers["B"] = section_b(D, args.out, M)
    if "C" in secs:
        numbers["C_map"] = section_c_map(D, args.out, M)
        section_c_grids(D, args.out, M)
        section_c_contact(D, args.out, M)
        numbers["C_layout"] = section_c_layout(D, args.out, M, args.cache)
        numbers["C_cls"] = section_c_classify(D, args.out, M, args.cache, numbers["C_layout"])
    if "D" in secs:
        numbers["D"] = section_d(D, args.out, M, args.cache)
    if "E" in secs:
        numbers["E"] = section_e(D, args.out, M)
    if "F" in secs:
        section_f(D, args.out, M, args.cache)
    if "G" in secs:
        numbers["G"] = section_g(D, args.out, M, args.cache)
    if secs == "ABCDEFG":
        M.write()
    else:  # partial run: merge into an existing manifest
        mp = args.out / "manifest.json"
        old = json.loads(mp.read_text()) if mp.exists() else []
        new_ids = {i["id"] for i in M.items}
        M.items = [i for i in old if i["id"] not in new_ids] + M.items
        M.write()
    np_ = args.out / "numbers.json"
    old_n = json.loads(np_.read_text()) if np_.exists() else {}
    np_.write_text(json.dumps({**old_n, **numbers}, indent=1, default=str))
    # summary.md is written by hand from numbers.json and the figures
    M.items = [i for i in M.items if i["section"] != "meta"]
    M.add("numbers.json", "meta", "요약 수치 (기계용)", "각 섹션 함수가 반환한 수치")
    if (args.out / "summary.md").exists():
        M.add(
            "summary.md",
            "meta",
            "T008 요약",
            "섹션별 핵심 관찰 (숫자 포함), 첫 프레임 → task 정확도, 못 한 것",
        )
    M.write()
    tot = sum(p.stat().st_size for p in args.out.iterdir())
    print(f"{len(M.items)} manifest items, {tot / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
