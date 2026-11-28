# ruff: noqa: N803, N806  (S = suite state, Y = projections, as in pro_traj.py)
"""T037: LIBERO-PRO (T023) trajectories per entry, one colour per trial (init 0–9).

Input: the T023 feature cache (<bundle>/_cache/<suite>.npz + .json, written by pro_traj.py --stages load).
For every suite and each of the six T023 features (expert, gemma, siglip, patch_attention, attention, action;
z-scored per suite over all calls exactly as pro_traj.Suite does), a PCA-2 plane is fitted on that suite's
`none` calls (pro_traj.pca_fit, the 7_trajectories recipe) and every call of every episode is projected on it.

Outputs (under --out, default <bundle>/9_trial_traj):
  <suite>/task<NN>/<axis>_<level>.png   one figure per entry (suite × task × variant), 2 × 3 panels = features;
                                        10 trials coloured by init (fixed palette), light → full colour = time,
                                        call number every 10 calls, ○ start, × first grasp, ■ first release,
                                        end ● success / ✕ fail / ◆ fail that did the original task (task axis);
                                        thin grey = the 10 none trials of the same task (left out for none)
  summary/<suite>_<axis>_<level>_<feat>.png   all 10 tasks of one variant for one feature (default expert,
                                        patch_attention), same colouring
  index.csv                             entry → figure, successes, overlay folder (8_attention_overlay)

Run: OPENBLAS_CORETYPE=ZEN uv run --with scikit-learn --with matplotlib python \
        scripts/shortcut/analysis/pro_trial_traj.py --bundle /outputs/VLA-shortcut-exps/experiments/libero_pro
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pro_traj import (  # noqa: E402
    FEATS6,
    HUES10,
    INK,
    POS_LEVELS,
    SHORT,
    SUITES,
    SURFACE,
    call_of,
    grad_line,
    pca_fit,
)

BUNDLE = Path("/outputs/VLA-shortcut-exps/experiments/libero_pro")
JOB = {"libero_spatial": "spat", "libero_object": "obj", "libero_goal": "goal", "libero_10": "l10"}
PANEL = ["expert", "gemma", "siglip", "patch_attention", "attention", "action"]  # 2 × 3 order
assert sorted(PANEL) == sorted(FEATS6)
PTITLE = {
    "expert": "activations · action expert",
    "gemma": "activations · VLM (Gemma)",
    "siglip": "activations · SigLIP",
    "patch_attention": "image-patch attention",
    "attention": "attention shares by input type",
    "action": "predicted actions",
}
REF = "#b9b8b3"
AXIS_ORDER = ["none", "object", "position", "semantic", "task", "environment"]

# per-suite state, filled in the parent before forking the plotting workers
G: dict[str, Any] = {}


def load_suite(bundle: Path, suite: str) -> dict[str, Any]:
    """meta + 2-D projections (calls × 2) per feature on the none PCA-2 plane, z-scored like pro_traj.Suite."""
    meta = json.loads((bundle / "_cache" / f"{suite}.json").read_text())
    z = np.load(bundle / "_cache" / f"{suite}.npz")
    off = z["offsets"]
    none_rows = np.concatenate(
        [np.arange(off[i], off[i + 1]) for i, m in enumerate(meta) if m["axis"] == "none"]
    )
    Y, evr = {}, {}
    for f in FEATS6:
        x = z[f]
        mu, sd = x.mean(0), x.std(0)
        keep = sd > 1e-6
        xz = ((x - mu) / np.where(keep, sd, 1))[:, keep].astype(np.float32)
        pm, pcs, ev = pca_fit(xz[none_rows], 2)
        Y[f] = ((xz - pm) @ pcs.T).astype(np.float32)
        evr[f] = ev
    for i, m in enumerate(meta):
        m["i"] = i
    return {"suite": suite, "meta": meta, "off": off, "Y": Y, "evr": evr}


def traj(S: dict, f: str, i: int) -> np.ndarray:
    return S["Y"][f][S["off"][i] : S["off"][i + 1]]


def outcome(m: dict) -> str:
    if m["success"]:
        return "success"
    if m["axis"] == "task" and m.get("success_orig"):
        return "shortcut"
    return "fail"


TICK = {"success": "✓", "fail": "✗", "shortcut": "✗↺"}


def draw_trial(ax: Any, y: np.ndarray, m: dict, color: str, small: bool = False) -> None:
    lw = 1.0 if small else 1.5
    grad_line(ax, y, color, lw=lw)
    ms = 0.6 if small else 1.0
    ax.plot(y[0, 0], y[0, 1], "o", ms=4 * ms, color=color, mfc="white", mew=1.0, zorder=4)
    if not small:
        for k in range(10, len(y), 10):
            ax.text(y[k, 0], y[k, 1], str(k), fontsize=5, color=color, ha="left", va="bottom", zorder=5)
    g, r = call_of(m.get("first_grasp_t")), call_of(m.get("first_release_t"))
    if g is not None and g < len(y):
        ax.plot(y[g, 0], y[g, 1], "x", ms=7 * ms, mew=1.8 * ms, color=color, zorder=6)
    if r is not None and r < len(y):
        ax.plot(y[r, 0], y[r, 1], "s", ms=5 * ms, color=color, mec=INK, mew=0.6, zorder=6)
    o = outcome(m)
    e = y[-1]
    if o == "success":
        ax.plot(e[0], e[1], "o", ms=7 * ms, color=color, mec=INK, mew=0.7, zorder=7)
    elif o == "shortcut":
        ax.plot(e[0], e[1], "D", ms=6.5 * ms, color=color, mec=INK, mew=0.7, zorder=7)
    else:
        ax.plot(e[0], e[1], "X", ms=8.5 * ms, color=color, mec=INK, mew=0.6, zorder=7)


def entry_trials(S: dict, task: int, axis: str, level: str) -> list[dict]:
    t = [m for m in S["meta"] if m["task_id"] == task and m["axis"] == axis and m["level"] == level]
    return sorted(t, key=lambda m: m["init_idx"])


def draw_panel(ax: Any, S: dict, f: str, trials: list[dict], ref: list[dict], small: bool = False) -> None:
    for m in ref:
        y = traj(S, f, m["i"])
        ax.plot(y[:, 0], y[:, 1], color=REF, lw=0.45 if small else 0.6, alpha=0.7, zorder=1)
    for m in trials:
        draw_trial(ax, traj(S, f, m["i"]), m, HUES10[m["init_idx"] % 10], small=small)
    ax.autoscale()
    ax.margins(0.04)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6)


def marker_handles(task_axis: bool) -> list:
    L = plt.Line2D
    h = [
        L([], [], color=REF, lw=1, label="none trials of this task"),
        L([], [], ls="", marker="o", ms=4, mfc="white", color=INK, label="start (call 0)"),
        L([], [], ls="", marker="x", ms=6, mew=1.6, color=INK, label="first grasp"),
        L([], [], ls="", marker="s", ms=5, color="#999", mec=INK, label="first release"),
        L([], [], ls="", marker="o", ms=6, color="#999", mec=INK, label="end: success"),
        L([], [], ls="", marker="X", ms=7, color="#999", mec=INK, label="end: fail"),
    ]
    if task_axis:
        h.append(L([], [], ls="", marker="D", ms=5.5, color="#999", mec=INK, label="end: did original task"))
    return h


def entry_title(S: dict, trials: list[dict]) -> str:
    m0 = trials[0]
    k = sum(bool(m["success"]) for m in trials)
    s = (
        f"{SHORT[S['suite']]} · task{m0['task_id']:02d} · {m0['axis']} ({m0['level']}) · "
        f"success {k}/{len(trials)}"
    )
    if m0["axis"] == "task":
        k2 = sum(outcome(m) == "shortcut" for m in trials)
        s += f" · did original task {k2}/{len(trials)}"
    instr = f"“{m0['instruction']}”"
    if m0.get("original_instruction") and m0["original_instruction"] != m0["instruction"]:
        instr += f"   (trained: “{m0['original_instruction']}”)"
    return s + "\n" + "\n".join(textwrap.wrap(instr, 150))


def plot_entry(job: tuple[str, int, str, str, str]) -> tuple[str, str]:
    suite, task, axis, level, path = job
    S = G[suite]
    trials = entry_trials(S, task, axis, level)
    ref = [] if axis == "none" else entry_trials(S, task, "none", "original")
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2), facecolor=SURFACE)
    for ax, f in zip(axes.flat, PANEL, strict=True):
        draw_panel(ax, S, f, trials, ref)
        ev = S["evr"][f]
        ax.set_title(PTITLE[f], fontsize=9)
        ax.set_xlabel(f"none PC1 ({ev[0]:.0%})", fontsize=7)
        ax.set_ylabel(f"none PC2 ({ev[1]:.0%})", fontsize=7)
    init_h = [
        plt.Line2D(
            [], [], color=HUES10[m["init_idx"]], lw=2.5, label=f"init {m['init_idx']} {TICK[outcome(m)]}"
        )
        for m in trials
    ]
    fig.legend(
        handles=init_h, loc="lower center", ncol=10, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.03)
    )
    mh = marker_handles(axis == "task")
    if axis == "none":
        mh = mh[1:]
    fig.legend(
        handles=mh, loc="lower center", ncol=len(mh), fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0)
    )
    fig.suptitle(entry_title(S, trials), fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    return suite, path


def plot_summary(job: tuple[str, str, str, str, str]) -> str:
    suite, axis, level, f, path = job
    S = G[suite]
    tasks = sorted({m["task_id"] for m in S["meta"] if m["axis"] == axis and m["level"] == level})
    fig, axes = plt.subplots(2, 5, figsize=(22, 9.4), facecolor=SURFACE)
    tot = n = 0
    for ax, t in zip(axes.flat, range(10), strict=True):
        if t not in tasks:
            ax.set_axis_off()
            continue
        trials = entry_trials(S, t, axis, level)
        ref = [] if axis == "none" else entry_trials(S, t, "none", "original")
        draw_panel(ax, S, f, trials, ref, small=True)
        k = sum(bool(m["success"]) for m in trials)
        tot, n = tot + k, n + len(trials)
        lab = f"task{t:02d} · {k}/{len(trials)}"
        if axis == "task":
            lab += f" · orig {sum(outcome(m) == 'shortcut' for m in trials)}"
        ax.set_title(lab + "\n" + textwrap.shorten(trials[0]["instruction"], 60), fontsize=7.5)
    ev = S["evr"][f]
    init_h = [plt.Line2D([], [], color=HUES10[i], lw=2.5, label=f"init {i}") for i in range(10)]
    mh = marker_handles(axis == "task")
    if axis == "none":
        mh = mh[1:]
    fig.legend(handles=init_h + mh, loc="lower center", ncol=len(init_h + mh), fontsize=7.5, frameon=False)
    fig.suptitle(
        f"{SHORT[suite]} · {axis} ({level}) · {PTITLE[f]} — all tasks, success {tot}/{n}; "
        f"axes = suite none PC1 ({ev[0]:.0%}) / PC2 ({ev[1]:.0%}); light → full colour = early → late",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=95, facecolor=SURFACE)
    plt.close(fig)
    return path


def variants(meta: list[dict]) -> list[tuple[str, str]]:
    have = {(m["axis"], m["level"]) for m in meta}
    lv_rank = {lv: i for i, lv in enumerate(POS_LEVELS)}
    return sorted(have, key=lambda v: (AXIS_ORDER.index(v[0]), lv_rank.get(v[1], -1), v[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=BUNDLE)
    ap.add_argument("--out", type=Path, default=None, help="default <bundle>/9_trial_traj")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--summary-feats", default="expert,patch_attention")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="entries per suite (smoke test)")
    a = ap.parse_args()
    out = a.out or a.bundle / "9_trial_traj"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for suite in a.suites.split(","):
        G.clear()
        S = G[suite] = load_suite(a.bundle, suite)
        print(
            f"[{suite}] {len(S['meta'])} episodes; PC1+2 var "
            + " ".join(f"{f}={S['evr'][f].sum():.2f}" for f in PANEL),
            flush=True,
        )
        jobs, sjobs = [], []
        for axis, level in variants(S["meta"]):
            for f in a.summary_feats.split(","):
                sjobs.append(
                    (suite, axis, level, f, str(out / "summary" / f"{suite}_{axis}_{level}_{f}.png"))
                )
            for t in sorted({m["task_id"] for m in S["meta"] if m["axis"] == axis and m["level"] == level}):
                fig = out / suite / f"task{t:02d}" / f"{axis}_{level}.png"
                jobs.append((suite, t, axis, level, str(fig)))
                tr = entry_trials(S, t, axis, level)
                ov = f"8_attention_overlay/pro-{JOB[suite]}-{axis}/{suite}/task{t:02d}"
                rows.append(
                    {
                        "suite": suite,
                        "task": f"task{t:02d}",
                        "variant": f"{axis}_{level}",
                        "perturbation": axis,
                        "level": level,
                        "instruction": tr[0]["instruction"],
                        "original_instruction": tr[0].get("original_instruction"),
                        "n_trials": len(tr),
                        "successes": sum(bool(m["success"]) for m in tr),
                        "did_original_task": sum(outcome(m) == "shortcut" for m in tr)
                        if axis == "task"
                        else "",
                        "per_init": " ".join(f"{m['init_idx']}{TICK[outcome(m)]}" for m in tr),
                        "figure": str(fig.relative_to(out)),
                        "summary_figures": " ".join(
                            f"summary/{suite}_{axis}_{level}_{f}.png" for f in a.summary_feats.split(",")
                        ),
                        "overlay_folder": ov if (a.bundle / ov).is_dir() else "",
                    }
                )
        if a.limit:
            jobs, sjobs = jobs[: a.limit], sjobs[:1]
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
            n = sum(1 for _ in ex.map(plot_entry, jobs, chunksize=2))
            ns = sum(1 for _ in ex.map(plot_summary, sjobs))
        print(f"[{suite}] {n} entry figures, {ns} summary figures", flush=True)
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"index.csv: {len(rows)} entries", flush=True)


if __name__ == "__main__":
    main()
