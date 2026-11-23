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
  numbers.json                         pre-grasp (rel −10 … 0) / post-grasp (rel +1 … +15) ratios with 95% CI, counts.
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
    "gemma": lambda d: np.log(np.concatenate([d["vlm_mlp"].reshape(NC, -1), d["vlm_h"].reshape(NC, -1)], 1)),
    "siglip": lambda d: np.log(d["sig_h"][:, :2].reshape(NC, -1)),
    "attention": lambda d: d["attn_mass"].mean(1).reshape(NC, -1),
    "patch_attention": lambda d: np.sqrt(d["attn_img"][:, :, :512].astype(np.float32).reshape(NC, -1)),
    "action": lambda d: d["act_pred"].reshape(NC, -1),
}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(K / "basket/phase2"))
    ap.add_argument("--recompute", action="store_true", help="recompute the cached t-SNE embeddings")
    ap.add_argument("--only", default="123", help="which figure groups to draw (1, 2, 3)")
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
    p = out / "numbers.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    (p).write_text(json.dumps(old | res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
