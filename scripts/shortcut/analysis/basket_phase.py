"""Basket experiment, trajectories split at the grasp (T024): when in each trial does the grasp happen, and how far
does each condition sit from its paired ctl before and after it. CPU only, from the already-extracted npz.

LIBERO-Object training tasks, pi05 (lerobot/pi05_libero_finetuned), task × init 0–9, 28 policy calls per episode
(steps 0, 10, …, 270). Conditions, paired only within a machine:
  A100 (ferrari):  ctl (A100) = libero3 ours-obj-custom-train10; opp (0.01, −0.33), near (0.10, 0.26) = ours-obj-custom-basket
  4090 (VGI-102):  ctl (4090); far20 (0.20, 0.21), far25 (0.25, 0.20)
Grasp / release steps (g, r) from basket/far/1_results/outcomes.csv (T022; first step holding the object, first step
after letting go). Call index of an event = floor(step / 10): the last observation taken before / at the event.

Features (identical to T021 basket_traj.py / T022 basket_far.py): expert, gemma, siglip, attention, patch_attention,
action; z-scored per dimension (NaN-aware).

Outputs (under --out, default /outputs/VLA-shortcut-exps/kubectl/basket/phase2):
  1_traj_tsne/<feat>_{A100,4090}.png   10 task panels; each trial's line gets darker with time, thin before the grasp
                                       call and thick after; ● first call, × grasp call, ■ release call, dotted = failed.
                                       Embedding = T021 trajectories(): z-score + PCA-30 + t-SNE (perplexity 50) of all
                                       300 × 28 calls of the machine (fit per machine, so the two machines' maps differ).
  2_task_zoom/<feat>/O<k>.png          one task: rows A100 / 4090, a panel per condition with its 10 trials, call numbers
                                       every 5 calls, the condition mean trajectory (mean 2D point per call), the other
                                       conditions of the same task in gray. Same embedding as 1_traj_tsne.
  3_grasp_aligned/summary.png, <feat>.png
                                       distance to the same task·init ctl of the same machine ÷ ctl other-init baseline,
                                       both trials aligned to their own grasp call (relative call −10 … +15).
                                       z-score over all 600 trials (T022). Baseline = the ctl trial's distance to the
                                       other inits of the same task, also grasp-aligned (T021 boot_ratio_pairs: ratio of
                                       means, task·init resampled, baseline recomputed inside each resample).
  4_object_space/                      T024 follow-up, same coordinates as the original Object results:
    A_summary_<feat>{,_pc13}.png, A_task_zoom/<feat>/O<k>.png
                                       space A = z-score + PCA fitted on the calls of the 100 ctl (A100) Object episodes
                                       only; every condition of both machines projected (task-mean / per-trial
                                       trajectories, same markers as above). Numbers: (a) grasp-aligned distance to the
                                       own task's ctl centre (same-init twin left out) ÷ ctl spread; (b) share of a
                                       trial's variance in the top-k ctl PCs (k = 90% of ctl variance, 10-fold by init).
    B_tasks40_<feat>.png               space B = the 400 ctl A100 training episodes of all 40 tasks (libero3 S/O/L +
                                       cand31 Goal), 20 time bins per trial, z-score + PCA-50 fitted on them, basket
                                       conditions projected; joint t-SNE; nearest of the 40 task centroids.
    tables.md                          all space A / B tables
  numbers.json                         pre-grasp (rel −10 … 0) / post-grasp (rel +1 … +15) ratios with 95% CI, counts;
                                       object_space_A, task40_space_B.
Run: OMP_NUM_THREADS=8 OPENBLAS_CORETYPE=ZEN $W/cand31/.venv-viz/bin/python basket_phase.py [--out DIR] [--recompute]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")  # task rule: ≤ 8 CPU workers

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

K = Path("/outputs/VLA-shortcut-exps/kubectl")
OUTCOMES = K / "basket/far/1_results/outcomes.csv"
ATT_A = K / "libero-spatial-object-long/_raw/attn/out"
ATT_V = K / "basket/far/_raw/attn/out"
CTLJ, BSKJ = "ours-obj-custom-train10", "ours-obj-custom-basket"

# kind → (machine, npz dir, file prefix, kind name in outcomes.csv, label, hue)
COND = {
    "ctlA": ("A100", ATT_A / CTLJ, "ctl_t{:02d}_train", "ctlF", "ctl (A100)", "#2a78d6"),
    "opp": ("A100", ATT_A / BSKJ, "bsk_opp_t{:02d}", "opp", "opp (0.01, −0.33) [A100]", "#eb6834"),
    "near": ("A100", ATT_A / BSKJ, "bsk_near_t{:02d}", "near", "near (0.10, 0.26) [A100]", "#1baf7a"),
    "ctlV": ("4090", ATT_V / CTLJ, "ctl_t{:02d}_train", "ctl", "ctl (4090)", "#2a78d6"),
    "far20": ("4090", ATT_V / BSKJ, "bsk_c20y21_t{:02d}", "far20", "far20 (0.20, 0.21) [4090]", "#eda100"),
    "far25": ("4090", ATT_V / BSKJ, "bsk_c25y20_t{:02d}", "farM", "far25 (0.25, 0.20) [4090]", "#e34948"),
}
MACHINES = {"A100": ["ctlA", "opp", "near"], "4090": ["ctlV", "far20", "far25"]}
CTL = {"A100": "ctlA", "4090": "ctlV"}
KINDS = [k for ks in MACHINES.values() for k in ks]
MOVED = ["near", "far20", "far25", "opp"]  # by distance; opp = other side
NC = 28
STEPS = np.arange(NC) * 10
REL = np.arange(-10, 16)  # grasp-aligned relative calls
PRE, POST = REL <= 0, REL >= 1  # rel 0 = the last observation at or before the grasp step
MUTED, INK, INK2, SURFACE, GRID = "#898781", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
RNG = np.random.default_rng(0)
NBOOT = 2000
MIN_PAIRS = 20  # curve points with fewer task·init pairs are not drawn (still in numbers.json)
FEATS = {  # per call; identical to libero3 l3_analysis.py / basket_traj.py / basket_far.py
    "expert": lambda d: np.log(np.concatenate([d["exp_mlp"].mean(1), d["exp_h"].mean(1)], 1)),
    "gemma": lambda d: np.log(np.concatenate([d["vlm_mlp"].reshape(len(d["vlm_mlp"]), -1),
                                              d["vlm_h"].reshape(len(d["vlm_h"]), -1)], 1)),
    "siglip": lambda d: np.log(d["sig_h"][:, :2].reshape(len(d["sig_h"]), -1)),
    "attention": lambda d: d["attn_mass"].mean(1).reshape(len(d["attn_mass"]), -1),
    "patch_attention": lambda d: np.sqrt(d["attn_img"][:, :, :512].astype(np.float32).reshape(len(d["attn_img"]), -1)),
    "action": lambda d: d["act_pred"].reshape(len(d["act_pred"]), -1),
}  # reshape by each episode's own call count: identical to NC for the 28-call Object episodes
TITLE = {"expert": "action expert activations", "gemma": "VLM (Gemma) activations", "siglip": "SigLIP activations",
         "attention": "attention shares (layer × input)", "patch_attention": "image-patch attention",
         "action": "predicted actions"}


# ---------------------------------------------------------------- helpers


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def keyorder():
    return [(t, i) for t in range(10) for i in range(10)]


def num(s):
    return None if s in ("", None) else int(float(s))


def nanmean(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x, axis)


def rms(a, b):
    """RMS over the last axis, ignoring non-finite coordinates (NaN where nothing is finite)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.sqrt(np.nanmean((a - b) ** 2, -1))


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def shade(hue, s):
    """s in [0, 1]: 0 = pale (hue mixed 70% with white), 0.5 = the hue, 1 = the hue darkened 35%."""
    h = np.array(to_rgb(hue))
    if s < 0.5:
        w = 0.7 * (1 - s / 0.5)
        return tuple(h * (1 - w) + w)
    return tuple(h * (1 - 0.35 * (s - 0.5) / 0.5))


def dark(hue):
    return shade(hue, 1.0)


# ---------------------------------------------------------------- data


def load():
    rows = {(r["kind"], int(r["task"]), int(r["init"])): r for r in csv.DictReader(open(OUTCOMES))}
    eps = {}
    for k, (mach, d, pre, ck, _, _) in COND.items():
        for t, i in keyorder():
            r = rows[ck, t, i]
            z = np.load(d / f"{pre.format(t)}_i{i:02d}.npz")
            assert len(z["t"]) == NC and (z["t"] == STEPS).all(), (k, t, i)
            g, rel = num(r["grasp_t"]), num(r["release_t"])
            e = dict(kind=k, machine=mach, task=t, init=i, success=r["success"] == "1", grasp=g, release=rel,
                     gc=None if g is None else min(g // 10, NC - 1), rc=None if rel is None else min(rel // 10, NC - 1),
                     failure=r["failure"] or "success", instruction=r["instruction"])
            e.update({key: z[key] for key in ("attn_mass", "attn_img", "exp_mlp", "exp_h", "vlm_h", "vlm_mlp", "sig_h",
                                              "act_pred")})
            eps[k, t, i] = e
    return eps


def feature_arrays(eps, kinds, fit):
    """feat → kind → (100, NC, D) z-scored per dimension with statistics from the trials of `fit` (NaN-aware, T021)."""
    F = {}
    for f, fn in FEATS.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = {k: np.stack([fn(eps[k, t, i]) for t, i in keyorder()]).astype(np.float32) for k in kinds}
        for k in kinds:
            raw[k][~np.isfinite(raw[k])] = np.nan
        flat = np.concatenate([raw[k].reshape(-1, raw[k].shape[-1]) for k in fit])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sd = np.nanmean(flat, 0), np.nanstd(flat, 0)
        keep = np.nan_to_num(sd) > 1e-6
        F[f] = {k: ((raw[k] - mu) / np.where(keep, sd, 1))[..., keep] for k in kinds}
    return F


def embed(eps, out, recompute):
    """machine → feat → kind → (100, NC, 2): T021 trajectories() embedding, fit per machine on its 300 × 28 calls."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    E = {}
    for mach, ks in MACHINES.items():
        F = None
        E[mach] = {}
        for f in FEATS:
            p = cache / f"tsne_{f}_{mach}.npz"
            if p.exists() and not recompute:
                z = np.load(p)
                E[mach][f] = {k: z[k] for k in ks}
                continue
            if F is None:
                F = feature_arrays(eps, ks, ks)
            X = np.concatenate([F[f][k].reshape(-1, F[f][k].shape[-1]) for k in ks])
            Pc = PCA(n_components=min(30, X.shape[1]), random_state=0).fit_transform(np.nan_to_num(X))
            Tc = TSNE(n_components=2, perplexity=50, init="pca", learning_rate="auto", random_state=0,
                      n_jobs=8).fit_transform(Pc).reshape(len(ks), 100, NC, 2)
            E[mach][f] = {k: Tc[a] for a, k in enumerate(ks)}
            np.savez(p, **E[mach][f])
            print(f"[tsne] {mach} {f}", flush=True)
    return E


# ---------------------------------------------------------------- trajectory drawing


def draw_trial(ax, xy, e, hue, thin=0.55, thick=1.7, z=3, marker_scale=1.0, numbers=False):
    """one trial: segment c→c+1 shaded by time (pale → dark), thin while c < grasp call, thick after; dotted if failed."""
    gc = e["gc"] if e["gc"] is not None else NC
    seg = np.stack([xy[:-1], xy[1:]], 1)
    cols = [shade(hue, c / (NC - 2)) for c in range(NC - 1)]
    lws = [thin if c < gc else thick for c in range(NC - 1)]
    lc = LineCollection(seg, colors=cols, linewidths=lws, zorder=z, capstyle="round",
                        linestyles="-" if e["success"] else (0, (1.2, 1.4)))
    ax.add_collection(lc)
    ms = marker_scale
    ax.scatter(*xy[0], s=16 * ms, color=shade(hue, 0.1), edgecolor=dark(hue), lw=0.6, zorder=z + 1)
    if e["gc"] is not None:
        ax.scatter(*xy[e["gc"]], s=70 * ms, marker="x", color=dark(hue), lw=1.9, zorder=z + 2,
                   path_effects=[pe.withStroke(linewidth=3.2, foreground="white")])
    if e["rc"] is not None:
        ax.scatter(*xy[e["rc"]], s=26 * ms, marker="s", color=dark(hue), edgecolor="white", lw=0.7, zorder=z + 2)
    if numbers:
        for c in range(0, NC, 5):
            ax.text(*xy[c], str(c), fontsize=4.8, color=INK2, zorder=z + 3, ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])


def legend_handles(ks):
    h = [plt.Line2D([], [], color=COND[k][5], lw=2, label=COND[k][4]) for k in ks]
    h += [plt.Line2D([], [], color=INK, lw=0.6, label="before grasp call (reach & grasp)"),
          plt.Line2D([], [], color=INK, lw=1.8, label="after grasp call (move & place)"),
          plt.Line2D([], [], color=INK, lw=1, ls=":", label="failed episode"),
          plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=5, label="first call (pale end)"),
          plt.Line2D([], [], ls="", marker="x", color=INK, ms=8, mew=1.9, label="grasp call = floor(grasp_t/10)"),
          plt.Line2D([], [], ls="", marker="s", color=INK, ms=5, label="release call = floor(release_t/10)")]
    return h


def fig_traj_tsne(eps, E, out):
    d = out / "1_traj_tsne"
    d.mkdir(parents=True, exist_ok=True)
    ko = keyorder()
    for mach, ks in MACHINES.items():
        for f in FEATS:
            T = E[mach][f]
            fig, axes = plt.subplots(2, 5, figsize=(19, 8.6), facecolor=SURFACE)
            allseg = np.concatenate([np.stack([T[k][j, :-1], T[k][j, 1:]], 1) for k in ks for j in range(100)])
            tasks = np.repeat(np.tile(np.repeat(np.arange(10), 10), len(ks)), NC - 1)
            for t, ax in enumerate(axes.flat):
                ax.set_facecolor(SURFACE)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.add_collection(LineCollection(allseg[tasks != t], colors=MUTED, linewidths=0.3, alpha=0.07, zorder=1))
                for k in ks:
                    for j in range(t * 10, t * 10 + 10):
                        draw_trial(ax, T[k][j], eps[k, *ko[j]], COND[k][5], thin=0.45, thick=1.25, marker_scale=0.6)
                allxy = np.concatenate([T[k].reshape(-1, 2) for k in ks])
                pad = 0.03 * (allxy.max(0) - allxy.min(0))
                ax.set_xlim(allxy[:, 0].min() - pad[0], allxy[:, 0].max() + pad[0])
                ax.set_ylim(allxy[:, 1].min() - pad[1], allxy[:, 1].max() + pad[1])
                ax.set_title(f"O{t + 1}: {eps[ks[0], t, 0]['instruction'][12:50]}", fontsize=7.5, loc="left")
            fig.legend(handles=legend_handles(ks), loc="lower center", ncol=5, fontsize=7.5, frameon=False)
            fig.suptitle(f"[{mach}] {TITLE[f]}: t-SNE of all 300 × 28 calls of this machine; one line per trial, pale → "
                         f"dark with time (gray = other tasks)", fontsize=10)
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.1, top=0.93, wspace=0.04, hspace=0.12)
            fig.savefig(d / f"{f}_{mach}.png", dpi=110, facecolor=SURFACE)
            plt.close(fig)
        print(f"[fig1] {mach}", flush=True)


def fig_task_zoom(eps, E, out):
    ko = keyorder()
    for f in FEATS:
        d = out / "2_task_zoom" / f
        d.mkdir(parents=True, exist_ok=True)
        for t in range(10):
            fig, axes = plt.subplots(2, 3, figsize=(16, 10.6), facecolor=SURFACE)
            for r, (mach, ks) in enumerate(MACHINES.items()):
                T = E[mach][f]
                js = range(t * 10, t * 10 + 10)
                allxy = np.concatenate([T[k][js].reshape(-1, 2) for k in ks])
                pad = 0.06 * (allxy.max(0) - allxy.min(0))
                for c_, k in enumerate(ks):
                    ax = axes[r, c_]
                    ax.set_facecolor(SURFACE)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    others = [np.stack([T[o][j, :-1], T[o][j, 1:]], 1) for o in ks if o != k for j in js]
                    ax.add_collection(LineCollection(np.concatenate(others), colors="#b9b7b0", linewidths=0.8,
                                                     alpha=0.45, zorder=1))
                    hue = COND[k][5]
                    for j in js:
                        draw_trial(ax, T[k][j], eps[k, *ko[j]], hue, thin=0.6, thick=1.6, numbers=True)
                    m = T[k][js].mean(0)  # condition mean trajectory: mean 2D point per call over the 10 trials
                    ax.plot(m[:, 0], m[:, 1], color=dark(hue), lw=3.2, zorder=6, solid_capstyle="round",
                            path_effects=[pe.withStroke(linewidth=5, foreground="white")])
                    for c in range(0, NC, 5):
                        ax.text(*m[c], str(c), fontsize=7, fontweight="bold", color=INK, zorder=7, ha="center",
                                va="center", path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
                    E_ = [eps[k, *ko[j]] for j in js]
                    gcs = [e["gc"] for e in E_ if e["gc"] is not None]
                    nfail = sum(not e["success"] for e in E_)
                    ax.set_xlim(allxy[:, 0].min() - pad[0], allxy[:, 0].max() + pad[0])
                    ax.set_ylim(allxy[:, 1].min() - pad[1], allxy[:, 1].max() + pad[1])
                    ax.set_title(f"{COND[k][4]}\nsuccess {10 - nfail}/10 · grasp call median "
                                 f"{np.median(gcs) if gcs else '–'} (range {min(gcs) if gcs else '–'}–"
                                 f"{max(gcs) if gcs else '–'})", fontsize=8.5, loc="left")
                axes[r, 0].set_ylabel(f"{mach} embedding", fontsize=9)
            h = legend_handles([]) + [plt.Line2D([], [], color=INK, lw=3.2, label="condition mean (2D mean per call)"),
                                      plt.Line2D([], [], color="#b9b7b0", lw=1, label="other conditions, same task")]
            fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8, frameon=False)
            fig.suptitle(f"O{t + 1} ({eps['ctlA', t, 0]['instruction']}) · {TITLE[f]} · numbers = policy call (×10 "
                         f"steps); rows use separate t-SNE fits (A100 / 4090)", fontsize=10)
            fig.subplots_adjust(left=0.03, right=0.99, bottom=0.08, top=0.92, wspace=0.03, hspace=0.12)
            fig.savefig(d / f"O{t + 1}.png", dpi=105, facecolor=SURFACE)
            plt.close(fig)
        print(f"[fig2] {f}", flush=True)


# ---------------------------------------------------------------- grasp-aligned distance


def align(X, eps, k):
    """X (100, NC, D) → (100, len(REL), D) at calls gc + REL (NaN outside 0..NC-1 or without a grasp)."""
    out = np.full((100, len(REL), X.shape[-1]), np.nan, np.float32)
    for j, (t, i) in enumerate(keyorder()):
        gc = eps[k, t, i]["gc"]
        if gc is None:
            continue
        c = gc + REL
        ok = (c >= 0) & (c < NC)
        out[j, ok] = X[j, c[ok]]
    return out


def ctl_pairwise(Xc):
    """Xc (100, B, D) ctl → (10 tasks, 10, 10, B) RMS distance between inits a, b of the same task (NaN on the diagonal)."""
    out = np.full((10, 10, 10, Xc.shape[1]), np.nan)
    for t in range(10):
        for a in range(10):
            for b in range(10):
                if a != b:
                    out[t, a, b] = rms(Xc[t * 10 + a], Xc[t * 10 + b])
    return out


def boot_ratio_pairs(dm, pw, n=NBOOT, groups=None):
    """T021 basket_traj.py (unchanged): ratio of means distance(moved, own-init ctl) / baseline per column, baseline of
    trial (t, a) = mean distance to the other ctl inits of task t; task·init pairs resampled, baseline recomputed inside
    each resample from the resampled ctl trials. groups: bin slices / masks; per-bin baseline first, then the group mean."""
    dm = np.asarray(dm, float)
    V = np.isfinite(pw).astype(float)
    W = np.nan_to_num(pw)

    def gmean(x):
        if groups is None:
            return x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.stack([np.nanmean(x[..., g], -1) for g in groups], -1)

    dmg = gmean(dm).reshape(10, 10, -1)

    def ratio(c):
        with np.errstate(divide="ignore", invalid="ignore"):
            base = gmean(np.einsum("ntb,tabk->ntak", c, W) / np.einsum("ntb,tabk->ntak", c, V))
            R = np.isfinite(base) & np.isfinite(dmg)[None]
            w = c[..., None] * R
            out = (w * np.nan_to_num(dmg)[None]).sum((1, 2)) / (w * np.nan_to_num(base)).sum((1, 2))
            return np.where(np.isfinite(out), out, np.nan), (w > 0).sum((1, 2))

    m, cnt = ratio(np.ones((1, 10, 10)))
    idx = RNG.integers(0, 100, (n, 100))
    c = np.zeros((n, 100))
    np.add.at(c, (np.arange(n)[:, None], idx), 1)
    rb, _ = ratio(c.reshape(n, 10, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo, hi = np.nanpercentile(np.where(np.isfinite(rb), rb, np.nan), [2.5, 97.5], 0)
    return m[0], lo, hi, cnt[0]


def grasp_aligned(eps, out, res):
    d = out / "3_grasp_aligned"
    d.mkdir(parents=True, exist_ok=True)
    F = feature_arrays(eps, KINDS, KINDS)  # T022: z-score over all 600 trials
    COMP = {k: (k, CTL[COND[k][0]]) for k in MOVED}
    COMP["machine"] = ("ctlV", "ctlA")  # machine floor: 4090 ctl vs A100 ctl, same task·init (baseline: 4090 ctl)
    groups = [PRE, POST]
    R, curves = {}, {}
    nogr = {k: sum(eps[k, t, i]["gc"] is None for t, i in keyorder()) for k in KINDS}
    for f in FEATS:
        A = {k: align(F[f][k], eps, k) for k in KINDS}
        pw = {m: ctl_pairwise(A[m]) for m in ("ctlA", "ctlV")}
        R[f], curves[f] = {}, {}
        for name, (k, ref) in COMP.items():
            pref = "ctlV" if name == "machine" else ref
            dm = rms(A[k], A[ref])  # (100, len(REL))
            m, lo, hi, n = boot_ratio_pairs(dm, pw[pref], groups=groups)
            cm, clo, chi, cn = boot_ratio_pairs(dm, pw[pref])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base_trial = np.nanmean(pw[pref], 2).reshape(100, -1)  # per trial, per bin
            few = cn < MIN_PAIRS  # rel calls with few pairs (grasp call median 5 → rel ≤ −6 is rare): not drawn
            curves[f][name] = dict(m=np.where(few, np.nan, cm), lo=np.where(few, np.nan, clo),
                                   hi=np.where(few, np.nan, chi), n=cn, trial=dm / base_trial)
            R[f][name] = dict(pre=[r3(m[0]), r3(lo[0]), r3(hi[0])], post=[r3(m[1]), r3(lo[1]), r3(hi[1])],
                              n_pairs_pre=int(n[0]), n_pairs_post=int(n[1]),
                              by_rel_call=dict(rel=REL.tolist(), m=[r3(x) for x in cm], lo=[r3(x) for x in clo],
                                               hi=[r3(x) for x in chi], n_pairs=[int(x) for x in cn]))
        print(f"[grasp] {f}: " + " ".join(f"{nm}={R[f][nm]['pre'][0]}/{R[f][nm]['post'][0]}" for nm in COMP), flush=True)
    res["grasp_aligned"] = dict(
        definition=dict(rel_calls=[int(REL[0]), int(REL[-1])], pre="rel −10 … 0 (calls up to the grasp call)",
                        post="rel +1 … +15", ratio="mean distance / mean baseline over task·init pairs with both trials "
                        "present at the bin (ratio of means, per-bin baseline then window mean), 95% bootstrap CI",
                        zscore="per dimension over all 600 trials (T022)"),
        no_grasp_trials_excluded=nogr, ratios=R)

    hue = {k: COND[k][5] for k in MOVED} | {"machine": MUTED}
    lab = {k: f"{COND[k][4]} vs {COND[CTL[COND[k][0]]][4]}" for k in MOVED} | {
        "machine": "ctl (4090) vs ctl (A100): machine floor"}

    def frame(ax):
        ax.axvline(0.5, color=INK2, lw=0.9, ls="--")
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("policy call relative to the grasp call (×10 steps)", fontsize=7)
        ax.set_ylabel("distance ÷ ctl other-init baseline", fontsize=7)
        clean(ax)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    for ax, f in zip(axes.flat, FEATS):
        for nm in MOVED + ["machine"]:
            c = curves[f][nm]
            ax.plot(REL, c["m"], color=hue[nm], lw=1.8, ls="--" if nm == "machine" else "-", label=lab[nm])
            ax.fill_between(REL, c["lo"], c["hi"], color=hue[nm], alpha=0.15, lw=0)
        frame(ax)
        ax.set_title(TITLE[f], fontsize=9)
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle("grasp-aligned: distance to the same task·init ctl of the same machine ÷ ctl-vs-other-inits baseline "
                 "(each trial aligned to its own grasp call; dashed vertical = grasp; 95% CI; points with < 20 pairs not drawn)", fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "summary.png", dpi=115, facecolor=SURFACE)
    plt.close(fig)

    for f in FEATS:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.3), sharey=True, facecolor=SURFACE)
        for ax, nm in zip(axes, MOVED):
            c, fl = curves[f][nm], curves[f]["machine"]
            for tr in c["trial"]:
                ax.plot(REL, tr, color=hue[nm], lw=0.5, alpha=0.18)
            ax.plot(REL, fl["m"], color=MUTED, lw=1.5, ls="--", label=lab["machine"])
            ax.plot(REL, c["m"], color=dark(hue[nm]), lw=2.4, label="ratio of means (95% CI)")
            ax.fill_between(REL, c["lo"], c["hi"], color=hue[nm], alpha=0.3, lw=0)
            r = R[f][nm]
            ax.set_title(f"{lab[nm]}\nbefore {r['pre'][0]} [{r['pre'][1]}, {r['pre'][2]}] · after {r['post'][0]} "
                         f"[{r['post'][1]}, {r['post'][2]}]", fontsize=7.5, loc="left")
            frame(ax)
        top = np.nanpercentile(np.concatenate([curves[f][nm]["trial"].ravel() for nm in MOVED]), 99)
        axes[0].set_ylim(0, max(top, 2))
        axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
        fig.suptitle(f"{TITLE[f]}: grasp-aligned distance ÷ baseline; thin = one task·init pair (distance ÷ that ctl "
                     f"trial's own baseline), thick = ratio of means", fontsize=10)
        fig.tight_layout()
        fig.savefig(d / f"{f}.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)


def grasp_timing(eps, res):
    """grasp / release call per condition (for reading the figures)."""
    T = {}
    for k in KINDS:
        E = [eps[k, t, i] for t, i in keyorder()]
        gc = [e["gc"] for e in E if e["gc"] is not None]
        rc = [e["rc"] for e in E if e["rc"] is not None]
        T[k] = dict(machine=COND[k][0], label=COND[k][4], success=sum(e["success"] for e in E),
                    grasp_call_median=float(np.median(gc)), grasp_call_range=[int(min(gc)), int(max(gc))],
                    release_call_median=float(np.median(rc)) if rc else None, n_grasp=len(gc), n_release=len(rc),
                    n_no_grasp=100 - len(gc), n_grasp_no_release=sum(e["gc"] is not None and e["rc"] is None for e in E))
    res["timing"] = T


# ---------------------------------------------------------------- 4. fixed Object space (A) and 40-task space (B)

RUNS = Path("/workspace/VLA-shortcut-exps-kubectl/out/runs")
SUITES = [  # (tag, npz dir, episodes.jsonl dir), ctl A100 training-task episodes (libero3 / cand31 train10)
    ("S", ATT_A / "ours-spat-custom-train10", RUNS / "bench-pi05lib-libero3/jobs/ours-spat-custom-train10"),
    ("O", ATT_A / CTLJ, RUNS / "bench-pi05lib-libero3/jobs" / CTLJ),
    ("G", K / "libero_goal_composition/_raw/attn/out/ours-goal-custom-train10",
     RUNS / "bench-pi05lib-train10/jobs/ours-goal-custom-train10"),
    ("L", ATT_A / "ours-l10-custom-train10", RUNS / "bench-pi05lib-libero3/jobs/ours-l10-custom-train10"),
]
SUITE_NAME = {"S": "Spatial", "O": "Object", "G": "Goal", "L": "Long"}
SUITE_HUE = {"S": "#6250d6", "O": "#52514e", "G": "#3aa6b9", "L": "#7a5c3e"}  # not used by any basket condition
NBIN_B = 20  # libero3 l3_analysis.py: every trial resampled to 20 time bins so suites of different length mix


def boot_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [r3(x.mean()) if len(x) else None, None, None, int(len(x))]
    b = x[RNG.integers(0, len(x), (NBOOT, len(x)))].mean(1)
    return [r3(x.mean()), r3(np.percentile(b, 2.5)), r3(np.percentile(b, 97.5)), int(len(x))]


def boot_ratio_rows(a, b):
    """mean(a) / mean(b) over rows where both are finite; rows resampled together (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if not len(a) or b.mean() <= 0:
        return [None, None, None, 0]
    idx = RNG.integers(0, len(a), (NBOOT, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return [r3(a.mean() / b.mean()), r3(np.percentile(r, 2.5)), r3(np.percentile(r, 97.5)), int(len(a))]


def raw_feats(eps, kinds):
    """feat → kind → (100, NC, D) raw features, non-finite → NaN."""
    R = {}
    for f, fn in FEATS.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            R[f] = {k: np.stack([fn(eps[k, t, i]) for t, i in keyorder()]).astype(np.float32) for k in kinds}
        for k in kinds:
            R[f][k][~np.isfinite(R[f][k])] = np.nan
    return R


def zfit(flat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sd = np.nanmean(flat, 0), np.nanstd(flat, 0)
    keep = np.nan_to_num(sd) > 1e-6
    return lambda x: np.nan_to_num(((x - mu) / np.where(keep, sd, 1))[..., keep])  # NaN → 0 = the fit mean


def mean_episode(E):
    gcs = [e["gc"] for e in E if e["gc"] is not None]
    rcs = [e["rc"] for e in E if e["rc"] is not None]
    return dict(gc=int(round(np.median(gcs))) if gcs else None, rc=int(round(np.median(rcs))) if rcs else None,
                success=True)


def space_a(eps, out, res):
    """Space A: z-score and PCA fitted on the calls of the 100 ctl (A100) Object episodes only; every condition (both
    machines) projected onto the same axes."""
    from sklearn.decomposition import PCA

    d = out / "4_object_space"
    (d / "A_task_zoom").mkdir(parents=True, exist_ok=True)
    R = raw_feats(eps, KINDS)
    ko = keyorder()
    task_of = np.repeat(np.arange(10), 10)
    init_of = np.tile(np.arange(10), 10)
    RA = {"variance": {}, "cluster_distance": {}, "manifold": {}}
    for f in FEATS:
        z = zfit(R[f]["ctlA"].reshape(-1, R[f]["ctlA"].shape[-1]))
        Z = {k: z(R[f][k]) for k in KINDS}  # (100, NC, D')
        D = Z["ctlA"].shape[-1]
        pca = PCA(n_components=min(D, 100 * NC), random_state=0).fit(Z["ctlA"].reshape(-1, D))
        cum = np.cumsum(pca.explained_variance_ratio_)
        k90 = int(np.searchsorted(cum, 0.9) + 1)
        evr = pca.explained_variance_ratio_[:3]
        RA["variance"][f] = dict(dims=int(D), pc_var_pct=[round(float(v) * 100, 1) for v in evr], k90=k90)
        P = {k: pca.transform(Z[k].reshape(-1, D)).reshape(100, NC, -1)[..., :3] for k in KINDS}

        # (a) distance to the own task's ctl (A100) cluster ÷ ctl spread, grasp-aligned, pre / post grasp.
        # cluster centre at each relative call = mean of the task's ctl trials EXCEPT the same init (so no condition is
        # compared with its own paired ctl twin; for ctl A100 itself this is leave-one-out)
        A = {k: align(Z[k], eps, k) for k in KINDS}  # (100, len(REL), D')
        c = A["ctlA"].reshape(10, 10, len(REL), D)
        fin = np.isfinite(c[..., 0])
        sm, cnt = np.nansum(c, 1), fin.sum(1)  # (10, R, D'), (10, R)

        def loo_centre(j):
            t, a = task_of[j], init_of[j]
            x = c[t, a]
            with np.errstate(invalid="ignore", divide="ignore"):
                return (sm[t] - np.nan_to_num(x)) / (cnt[t] - fin[t, a])[:, None]

        cent = np.stack([loo_centre(j) for j in range(100)])  # (100, R, D')
        spread_trial = rms(A["ctlA"], cent)  # (100, R): ctl trial to its own task's leave-one-out centre
        spread = nanmean(spread_trial.reshape(10, 10, -1), 1)  # (10, R) per task
        rc = {}
        for k in KINDS:
            dist = rms(A[k], cent)  # (100, R)
            sp = spread[task_of]
            sp = np.where(np.isfinite(dist), sp, np.nan)
            rc[k] = {w: boot_ratio_rows(nanmean(np.where(m, dist, np.nan), 1), nanmean(np.where(m, sp, np.nan), 1))
                     for w, m in (("pre", PRE), ("post", POST))}
        RA["cluster_distance"][f] = rc

        # (b) inside the ctl manifold? share of a trial's variance (around the ctl mean) explained by the top-k90 ctl
        # PCs; 10-fold by init: the PCs are refitted without init i, then the init-i trials of every condition are
        # scored (so ctl A100 is out-of-sample like the others and no trial's paired ctl twin is in the fit)
        expl = {k: np.full((100, 2), np.nan) for k in KINDS}
        resid = {k: np.full((100, 2), np.nan) for k in KINDS}
        for i in range(10):
            tr = init_of != i
            pf = PCA(n_components=k90, random_state=0).fit(Z["ctlA"][tr].reshape(-1, D))
            for k in KINDS:
                for j in np.where(~tr)[0]:
                    x = A[k][j]
                    ok_r = np.isfinite(x[:, 0])
                    for w, m in enumerate((PRE, POST)):
                        xm = x[ok_r & m]
                        if not len(xm):
                            continue
                        xc = xm - pf.mean_
                        tot = (xc ** 2).sum()
                        prj = ((xc @ pf.components_.T) ** 2).sum()
                        expl[k][j, w] = prj / tot
                        resid[k][j, w] = (tot - prj) / len(xm) / D  # residual mean square per call and dimension
        mres = {}
        for k in KINDS:
            ref = "ctlA" if k == "ctlV" else CTL[COND[k][0]]
            mres[k] = {}
            for w, wn in enumerate(("pre", "post")):
                e_k, e_r = expl[k][:, w], expl[ref][:, w]
                diff = e_k - e_r
                mres[k][wn] = dict(explained=boot_ci(e_k),
                                   minus_ref=None if k == ref else boot_ci(diff),
                                   residual_over_ref=None if k == ref else boot_ratio_rows(resid[k][:, w], resid[ref][:, w]),
                                   ref=COND[ref][4])
        RA["manifold"][f] = mres
        print(f"[spaceA] {f}: k90={k90} PC1-3 {RA['variance'][f]['pc_var_pct']} | cluster post " +
              " ".join(f"{k}={rc[k]['post'][0]}" for k in KINDS) + " | explained post " +
              " ".join(f"{k}={mres[k]['post']['explained'][0]}" for k in KINDS), flush=True)

        # figures
        pv = RA["variance"][f]["pc_var_pct"]
        allp = np.concatenate([P[k].reshape(-1, 3) for k in KINDS])
        lim = [(np.percentile(allp[:, a], 0.5), np.percentile(allp[:, a], 99.5)) for a in range(3)]
        for axes_pc, tag in (((0, 1), ""), ((0, 2), "_pc13")):
            fig, axes = plt.subplots(2, 3, figsize=(17, 10.8), facecolor=SURFACE)
            ref_means = {t: P["ctlA"][t * 10:(t + 1) * 10].mean(0) for t in range(10)}
            for r, (mach, ks) in enumerate(MACHINES.items()):
                for cc, k in enumerate(ks):
                    ax = axes[r, cc]
                    ax.set_facecolor(SURFACE)
                    pa, pb = axes_pc
                    bg = P["ctlA"].reshape(-1, 3)
                    ax.scatter(bg[:, pa], bg[:, pb], s=2, color="#c9c7c0", alpha=0.35, lw=0, zorder=1)
                    for t in range(10):
                        m = P[k][t * 10:(t + 1) * 10].mean(0)
                        if k != "ctlA":
                            rm = ref_means[t]
                            ax.plot(rm[:, pa], rm[:, pb], color="#8fb6e8", lw=1.2, ls="--", zorder=2)
                        E_ = [eps[k, t, i] for i in range(10)]
                        draw_trial(ax, m[:, [pa, pb]], mean_episode(E_), COND[k][5], thin=0.9, thick=2.2,
                                   marker_scale=0.8, numbers=True)
                        ax.text(*m[-1, [pa, pb]], f" O{t + 1}", fontsize=6, color=INK2, zorder=8, clip_on=True)
                    ax.set_xlim(*lim[pa])
                    ax.set_ylim(*lim[pb])
                    ax.set_xlabel(f"PC{pa + 1} ({pv[pa]}% of ctl A100 variance)", fontsize=7)
                    ax.set_ylabel(f"PC{pb + 1} ({pv[pb]}%)", fontsize=7)
                    nf = sum(not eps[k, t, i]["success"] for t, i in ko)
                    ax.set_title(f"{COND[k][4]}: mean trajectory per task (10 inits){'' if k == 'ctlA' else ''}"
                                 f"\nsuccess {100 - nf}/100", fontsize=8.5, loc="left")
                    clean(ax)
            h = legend_handles([]) + [
                plt.Line2D([], [], color="#8fb6e8", lw=1.2, ls="--", label="ctl (A100) mean of the same task (reference)"),
                plt.Line2D([], [], ls="", marker="o", color="#c9c7c0", ms=4, label="all ctl (A100) calls")]
            fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
            fig.suptitle(f"space A (fixed by the original Object episodes): {TITLE[f]} — z-score + PCA fitted on the 100 ctl "
                         f"(A100) episodes only, all conditions projected; same axes for both machines", fontsize=10)
            fig.subplots_adjust(left=0.04, right=0.99, bottom=0.1, top=0.92, wspace=0.18, hspace=0.25)
            fig.savefig(d / f"A_summary_{f}{tag}.png", dpi=105, facecolor=SURFACE)
            plt.close(fig)
        fd = d / "A_task_zoom" / f
        fd.mkdir(parents=True, exist_ok=True)
        for t in range(10):
            js = range(t * 10, t * 10 + 10)
            pts = np.concatenate([P[k][js].reshape(-1, 3) for k in KINDS])
            pad = 0.05 * (pts.max(0) - pts.min(0))
            rm = P["ctlA"][js].mean(0)
            fig, axes = plt.subplots(2, 3, figsize=(16, 10.6), facecolor=SURFACE)
            for r, (mach, ks) in enumerate(MACHINES.items()):
                for cc, k in enumerate(ks):
                    ax = axes[r, cc]
                    ax.set_facecolor(SURFACE)
                    others = [np.stack([P[o][j, :-1, :2], P[o][j, 1:, :2]], 1) for o in KINDS if o != k for j in js]
                    ax.add_collection(LineCollection(np.concatenate(others), colors="#b9b7b0", linewidths=0.7,
                                                     alpha=0.35, zorder=1))
                    if k != "ctlA":
                        ax.plot(rm[:, 0], rm[:, 1], color="#8fb6e8", lw=2, ls="--", zorder=2)
                    for j in js:
                        draw_trial(ax, P[k][j, :, :2], eps[k, *ko[j]], COND[k][5], thin=0.6, thick=1.6, numbers=True)
                    m = P[k][js].mean(0)
                    ax.plot(m[:, 0], m[:, 1], color=dark(COND[k][5]), lw=3.2, zorder=6,
                            path_effects=[pe.withStroke(linewidth=5, foreground="white")])
                    for c_ in range(0, NC, 5):
                        ax.text(*m[c_, :2], str(c_), fontsize=7, fontweight="bold", color=INK, zorder=7, ha="center",
                                va="center", path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
                    ax.set_xlim(pts[:, 0].min() - pad[0], pts[:, 0].max() + pad[0])
                    ax.set_ylim(pts[:, 1].min() - pad[1], pts[:, 1].max() + pad[1])
                    ax.set_xlabel(f"PC1 ({pv[0]}%)", fontsize=7)
                    ax.set_ylabel(f"PC2 ({pv[1]}%)", fontsize=7)
                    nf = sum(not eps[k, t, i]["success"] for i in range(10))
                    ax.set_title(f"{COND[k][4]} · success {10 - nf}/10", fontsize=8.5, loc="left")
                    clean(ax)
            h = legend_handles([]) + [
                plt.Line2D([], [], color=INK, lw=3.2, label="condition mean (mean per call)"),
                plt.Line2D([], [], color="#8fb6e8", lw=2, ls="--", label="ctl (A100) mean (reference)"),
                plt.Line2D([], [], color="#b9b7b0", lw=1, label="other conditions, same task")]
            fig.legend(handles=h, loc="lower center", ncol=5, fontsize=7.5, frameon=False)
            fig.suptitle(f"space A · O{t + 1} ({eps['ctlA', t, 0]['instruction']}) · {TITLE[f]} · PC1–PC2 of the ctl "
                         f"(A100) fit; all six panels share the same axes", fontsize=10)
            fig.subplots_adjust(left=0.04, right=0.99, bottom=0.09, top=0.93, wspace=0.15, hspace=0.2)
            fig.savefig(fd / f"O{t + 1}.png", dpi=100, facecolor=SURFACE)
            plt.close(fig)
    res["object_space_A"] = RA


def ptrans(pca, X):
    """PCA.transform without BLAS (see space_b)."""
    return np.einsum("nd,kd->nk", np.asarray(X, np.float64) - pca.mean_, pca.components_, optimize=False)


def resample(x, n=NBIN_B):
    t = np.linspace(0, 1, len(x))
    tt = np.linspace(0, 1, n)
    return np.stack([np.interp(tt, t, x[:, j]) for j in range(x.shape[1])], 1)


def load_suites():
    """40 training tasks, ctl A100, init 0–9: tag, task label, instruction, raw features per feat."""
    T = []
    for tag, nd, jd in SUITES:
        instr = {}
        for line in (jd / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            instr[int(e["task_id"])] = e["instruction"]
        for t in range(10):
            for i in range(10):
                z = dict(np.load(nd / f"ctl_t{t:02d}_train_i{i:02d}.npz"))
                with np.errstate(divide="ignore", invalid="ignore"):
                    fe = {f: fn(z).astype(np.float32) for f, fn in FEATS.items()}
                T.append(dict(tag=tag, task=f"{tag}{t + 1}", instruction=instr[t], feats=fe))
    return T


def space_b(eps, out, res):
    """Space B: whole-trial vectors (20 time bins, as libero3 l3_analysis.py maps) of the 400 ctl A100 training
    episodes (40 tasks); z-score + PCA-50 fitted on them; basket conditions projected; joint t-SNE of all trials."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    d = out / "4_object_space"
    d.mkdir(parents=True, exist_ok=True)
    tr = load_suites()
    print("[spaceB] loaded", len(tr), flush=True)
    labels = [x["task"] for x in tr]
    tasks = list(dict.fromkeys(labels))
    lab = np.array(labels)
    instr = {x["task"]: x["instruction"] for x in tr}
    ko = keyorder()
    proj_k = [k for k in KINDS if k != "ctlA"]  # ctl A100 = the O trials of the fit set
    RB = {}
    for f, fn in FEATS.items():
        Xf = [resample(np.nan_to_num(x["feats"][f], nan=np.nan)) for x in tr]
        with np.errstate(divide="ignore", invalid="ignore"):
            Xk = {k: [resample(fn(eps[k, t, i]).astype(np.float32)) for t, i in ko] for k in proj_k}
        for k in proj_k:
            for x in Xk[k]:
                x[~np.isfinite(x)] = np.nan
        for x in Xf:
            x[~np.isfinite(x)] = np.nan
        z = zfit(np.concatenate(Xf))  # l3_analysis.py zscore_calls: per dimension over all calls of the fit trials
        Zf = np.stack([z(x).ravel() for x in Xf])
        Zk = {k: np.stack([z(x).ravel() for x in Xk[k]]) for k in proj_k}
        # numpy's OpenBLAS dgemm segfaulted here on VGI-102 (randomized SVD and PCA.transform); the full SVD runs in
        # scipy and the projection is done with einsum (no BLAS), same result
        pca = PCA(n_components=50, svd_solver="full").fit(Zf.astype(np.float64))
        Pf = ptrans(pca, Zf)
        Pk = {k: ptrans(pca, Zk[k]) for k in proj_k}
        cent = np.stack([Pf[lab == u].mean(0) for u in tasks])
        # "inside a training task's range": distance to the nearest centroid ≤ the 95th percentile of the 400 ctl
        # trials' leave-one-out distance to their own task centroid (otherwise the nearest task is only "least far")
        own_d = np.array([np.linalg.norm(Pf[j] - Pf[(lab == lab[j]) & (np.arange(len(lab)) != j)].mean(0))
                          for j in range(len(lab))])
        thr = float(np.percentile(own_d, 95))
        r = {}
        for k in ["ctlA"] + proj_k:
            near_t, near_d = [], []
            for j, (t, i) in enumerate(ko):
                if k == "ctlA":
                    row = np.where(lab == f"O{t + 1}")[0][i]
                    x = Pf[row]
                    c = cent.copy()
                    c[tasks.index(f"O{t + 1}")] = Pf[(lab == f"O{t + 1}") & (np.arange(len(lab)) != row)].mean(0)
                else:
                    x = Pk[k][j]
                    c = cent.copy()  # the same-init ctl twin is 1 of 10 trials in its task centroid (kept, as T021)
                dd = np.linalg.norm(c - x, axis=1)
                near_t.append(tasks[int(np.argmin(dd))])
                near_d.append(float(dd.min()))
            own = [n == f"O{t + 1}" for n, (t, i) in zip(near_t, ko)]
            others = {}
            for n, (t, i) in zip(near_t, ko):
                if n != f"O{t + 1}":
                    others[n] = others.get(n, 0) + 1
            r[k] = dict(own_task=boot_ci(np.array(own, float)),
                        object_suite=r3(np.mean([n.startswith("O") for n in near_t])),
                        within_ctl_range=boot_ci((np.array(near_d) <= thr).astype(float)),
                        nearest_dist_over_ctl_median=r3(np.median(near_d) / np.median(own_d)),
                        by_suite={SUITE_NAME[s]: int(sum(n.startswith(s) for n in near_t)) for s in "SOGL"},
                        other_nearest=dict(sorted(others.items(), key=lambda kv: -kv[1])),
                        other_nearest_instr={u: instr[u] for u in others})
        RB[f] = dict(ctl_own_dist_p95=r3(thr), pca_var_pct_pc1_pc2=[round(float(v) * 100, 1) for v in pca.explained_variance_ratio_[:2]], nearest=r)
        print(f"[spaceB] {f}: own " + " ".join(f"{k}={r[k]['own_task'][0]}" for k in r) + " | in range " +
              " ".join(f"{k}={r[k]['within_ctl_range'][0]}/{r[k]['nearest_dist_over_ctl_median']}" for k in r) +
              " | suite " +
              " ".join(f"{k}={r[k]['by_suite']}" for k in r), flush=True)
        # joint t-SNE of the 400 + 500 trials (PCA-50 coordinates; t-SNE cannot place new points on a fixed map)
        allP = np.concatenate([Pf] + [Pk[k] for k in proj_k])
        T2 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0,
                  n_jobs=8).fit_transform(allP)
        Tf, Tk = T2[:len(Pf)], {k: T2[len(Pf) + 100 * a: len(Pf) + 100 * (a + 1)] for a, k in enumerate(proj_k)}
        fig, axes = plt.subplots(2, 2, figsize=(17, 14), facecolor=SURFACE)
        pv = RB[f]["pca_var_pct_pc1_pc2"]
        for r_, (name, Yf, Yk) in enumerate((("PCA (fitted on the 400 ctl trials)", Pf[:, :2], {k: Pk[k][:, :2] for k in proj_k}),
                                             ("joint t-SNE (all 900 trials)", Tf, Tk))):
            for c_, zoom in enumerate((False, True)):
                ax = axes[r_, c_]
                ax.set_facecolor(SURFACE)
                for s in "SOGL":
                    m = np.array([x["tag"] == s for x in tr])
                    ax.scatter(Yf[m, 0], Yf[m, 1], s=14 if s != "O" else 22, color=SUITE_HUE[s], alpha=0.35 if s != "O" else 0.6,
                               lw=0, zorder=2, label=f"{SUITE_NAME[s]} ctl (A100), 100 trials")
                for u in tasks:
                    cx, cy = np.median(Yf[lab == u], 0)
                    ax.text(cx, cy, u, fontsize=6.5 if not zoom else 8, color=SUITE_HUE[u[0]], ha="center", va="center",
                            zorder=6, fontweight="bold", clip_on=True,
                            bbox={"boxstyle": "round,pad=0.1", "fc": SURFACE, "ec": "none", "alpha": 0.6})
                for k in proj_k:
                    ok = np.array([eps[k, t, i]["success"] for t, i in ko])
                    ax.scatter(Yk[k][ok, 0], Yk[k][ok, 1], s=26, marker="o", facecolor="none", edgecolor=COND[k][5],
                               lw=1.2, zorder=4, label=f"{COND[k][4]} (○ success, × failed)")
                    ax.scatter(Yk[k][~ok, 0], Yk[k][~ok, 1], s=30, marker="x", color=COND[k][5], lw=1.3, zorder=4)
                if zoom:
                    obj = np.concatenate([Yf[lab == u] for u in tasks if u[0] == "O"] + [Yk[k] for k in proj_k])
                    lo_, hi_ = np.percentile(obj, 1, 0), np.percentile(obj, 99, 0)
                    pad = 0.1 * (hi_ - lo_)
                    ax.set_xlim(lo_[0] - pad[0], hi_[0] + pad[0])
                    ax.set_ylim(lo_[1] - pad[1], hi_[1] + pad[1])
                if r_ == 0:
                    ax.set_xlabel(f"PC1 ({pv[0]}% of the 400 ctl trials' variance)", fontsize=7)
                    ax.set_ylabel(f"PC2 ({pv[1]}%)", fontsize=7)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.set_title(f"{name}{' — zoom on the Object region' if zoom else ''}", fontsize=9, loc="left")
                clean(ax)
        axes[0, 0].legend(fontsize=7, frameon=False, loc="best")
        fig.suptitle(f"space B (40 training tasks): {TITLE[f]} — one point per trial (20 time bins); basket conditions "
                     f"projected; labels = task (S/O/G/L + number) at the median of its 10 ctl trials", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(d / f"B_tasks40_{f}.png", dpi=100, facecolor=SURFACE)
        plt.close(fig)
    res["task40_space_B"] = dict(
        definition="whole trial → 20 time bins (np.interp over normalized episode time), z-score per dimension over all "
                   "calls of the 400 ctl A100 training trials, PCA-50 fitted on them; nearest of the 40 task centroids "
                   "(ctl A100 itself: leave-one-out)",
        tasks={u: instr[u] for u in tasks}, by_feature=RB)


def space_tables(res, out):
    """markdown tables for the README (also handy in the log)."""
    A, B = res["object_space_A"], res["task40_space_B"]["by_feature"]
    L = ["| 특징 | 구간 | " + " | ".join(COND[k][4] for k in KINDS) + " |", "|---|---|" + "---|" * len(KINDS)]
    for f in FEATS:
        for w in ("pre", "post"):
            L.append(f"| {f} | {'전' if w == 'pre' else '후'} | " + " | ".join(
                "{} [{}, {}]".format(*A["cluster_distance"][f][k][w][:3]) for k in KINDS) + " |")
    L += ["", "| 특징 (k90) | 구간 | " + " | ".join(COND[k][4] for k in KINDS) + " |", "|---|---|" + "---|" * len(KINDS)]
    for f in FEATS:
        for w in ("pre", "post"):
            row = []
            for k in KINDS:
                m = A["manifold"][f][k][w]
                s = "{} [{}, {}]".format(*m["explained"][:3])
                if m["minus_ref"]:
                    s += " (Δ {})".format(m["minus_ref"][0])
                row.append(s)
            L.append(f"| {f} ({A['variance'][f]['k90']}) | {'전' if w == 'pre' else '후'} | " + " | ".join(row) + " |")
    L += ["", "| 특징 | " + " | ".join(COND[k][4] for k in KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for f in FEATS:
        L.append(f"| {f} | " + " | ".join(
            "{} · {} ({})".format(B[f]["nearest"][k]["own_task"][0], B[f]["nearest"][k]["within_ctl_range"][0],
                             ", ".join(f"{u} {n}" for u, n in list(B[f]["nearest"][k]["other_nearest"].items())[:3]) or "–")
            for k in KINDS) + " |")
    (out / "4_object_space" / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(K / "basket/phase2"))
    ap.add_argument("--recompute", action="store_true", help="recompute the cached t-SNE embeddings")
    ap.add_argument("--only", default="1234", help="which parts to run (1, 2, 3, 4 = fixed Object / 40-task spaces)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = load()
    print("loaded", len(eps), flush=True)
    res = {"conditions": {k: dict(machine=v[0], npz=str(v[1]), prefix=v[2], outcomes_kind=v[3], label=v[4])
                          for k, v in COND.items()}}
    grasp_timing(eps, res)
    if "1" in a.only or "2" in a.only:
        E = embed(eps, out, a.recompute)
        if "1" in a.only:
            fig_traj_tsne(eps, E, out)
        if "2" in a.only:
            fig_task_zoom(eps, E, out)
    if "3" in a.only:
        grasp_aligned(eps, out, res)
    if "4" in a.only:
        space_a(eps, out, res)
        space_b(eps, out, res)
        space_tables(res, out)
    p = out / "numbers.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    (p).write_text(json.dumps(old | res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
