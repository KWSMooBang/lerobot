"""Basket experiment (T021): internal representations beyond patch attention, CPU only, from already-extracted npz.

LIBERO-Object training tasks, pi05 (lerobot/pi05_libero_finetuned); per task × init 0–9:
  ctl   original scene (libero3 `ours-obj-custom-train10`)
  opp   basket at world (0.01, -0.33), the other side
  near  basket 10 cm toward the agentview camera
Every episode has 28 policy calls (steps 0, 10, …, 270), so calls align by index across conditions.

Features (same as libero3 l3_analysis.py FEATS): expert, gemma, siglip, attention, patch_attention, action; each z-scored
per dimension over all 300×28 calls. Phases from basket/1_results/outcomes.csv (g = first grasp, r = first release step):
  approach [0, g-20) · grasp [g-20, g+20) · carry [g+20, r-20) · release [r-20, r+20) · after [r+20, 280)
(missing phases, e.g. no release, are NaN). Phase-aligned bins: 5 per phase, features linearly interpolated over steps.

Questions → outputs (under --out, default /outputs/VLA-shortcut-exps/kubectl/basket):
  Q1 task identity     7_trial_map/<feat>/trialmap_{tsne,pca}.png, dendrogram.png, 7_trial_map/cluster_numbers.json
  Q2 when they diverge 6_trajectories/divergence_by_call.png, divergence_by_phase.png, <feat>/traj_tsne.png
  Q3 attention by type 2_attention/attention_by_type.png, attention_by_type_diff.png; 3_activation/*.png
  Q4 failures, near    5_analysis/failure_divergence.png, near_between.png; all numbers in 5_analysis/numbers.json
Bootstrap: 2000 resamples of episodes (paired = resample task·init pairs), 95% percentile CI.

Run:  OPENBLAS_CORETYPE=ZEN python basket_traj.py [--out DIR]   (numpy, scipy, scikit-learn, matplotlib)
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

K = Path("/outputs/VLA-shortcut-exps/kubectl")
ATT = K / "libero-spatial-object-long/_raw/attn/out"
OUTCOMES = K / "basket/1_results/outcomes.csv"
JOB = {"ctl": "ours-obj-custom-train10", "opp": "ours-obj-custom-basket", "near": "ours-obj-custom-basket"}
FILE = {"ctl": "ctl_t{t:02d}_train_i{i:02d}.npz", "opp": "bsk_opp_t{t:02d}_i{i:02d}.npz", "near": "bsk_near_t{t:02d}_i{i:02d}.npz"}
KINDS = ["ctl", "opp", "near"]
MOVED = ["opp", "near"]
LABEL = {"ctl": "ctl (original)", "opp": "opp (other side)", "near": "near (10 cm toward camera)"}
HUE = {"ctl": "#2a78d6", "opp": "#eb6834", "near": "#1baf7a"}
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#7a5c3e", "#3aa6b9"]
MUTED, INK, INK2, SURFACE = "#898781", "#0b0b0b", "#52514e", "#fcfcfb"
SEGMENTS = ["img_agentview", "img_wrist", "bos", "task", "state", "template", "action"]
SEG_LABEL = ["agentview", "wrist", "BOS", "sentence", "state", "template", "action"]
LGROUPS = [("L0–5", 0, 6), ("L6–11", 6, 12), ("L12–17", 12, 18)]
PHASES = ["approach", "grasp", "carry", "release", "after"]
NB = 5  # bins per phase
NC = 28
STEPS = np.arange(NC) * 10
T_END = 280
FAIL_SHORT = {"held the object to the end": "held to timeout", "released, not in basket": "released outside",
              "never grasped": "never grasped", "grasped the basket": "never grasped"}  # 1 basket grasp merged
FAIL_TYPES = ["held to timeout", "released outside", "never grasped"]
FAIL_HUE = {"success": INK, "held to timeout": "#e34948", "released outside": "#eda100", "never grasped": "#6250d6"}
RNG = np.random.default_rng(0)
NBOOT = 2000

FEATS = {  # per call; identical to libero3 l3_analysis.py
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


# ---------------------------------------------------------------- stats


def boot_mean(x, n=NBOOT):
    """mean and 95% CI of the mean over axis 0 (NaN rows dropped per column)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    m = np.nanmean(x, 0)
    idx = RNG.integers(0, len(x), (n, len(x)))
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = np.nanmean(x[idx], 1)
    lo, hi = np.nanpercentile(b, [2.5, 97.5], 0)
    return m, lo, hi


def boot_ratio(a, b, n=NBOOT):
    """ratio of means mean(a)/mean(b) for paired rows, with CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    idx = RNG.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), int(len(a))


def dz(diff):
    """paired effect size (mean / sd of the differences) with bootstrap CI."""
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return dict(n=int(len(d)))
    idx = RNG.integers(0, len(d), (NBOOT, len(d)))
    bm = d[idx].mean(1)
    bd = bm / np.maximum(d[idx].std(1, ddof=1), 1e-12)
    return dict(mean=r3(d.mean()), ci=[r3(np.percentile(bm, 2.5)), r3(np.percentile(bm, 97.5))],
                dz=r3(d.mean() / max(d.std(ddof=1), 1e-12)), dz_ci=[r3(np.percentile(bd, 2.5)), r3(np.percentile(bd, 97.5))],
                n=int(len(d)))


def hedges(a, b):
    """Hedges g of a − b (unpaired), bootstrap CI."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(na=int(len(a)), nb=int(len(b)))

    def g(x, y):
        s = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
        j = 1 - 3 / (4 * (len(x) + len(y)) - 9)
        return j * (x.mean() - y.mean()) / max(s, 1e-12)

    bs = [g(a[RNG.integers(0, len(a), len(a))], b[RNG.integers(0, len(b), len(b))]) for _ in range(NBOOT)]
    return dict(diff=r3(a.mean() - b.mean()), g=r3(g(a, b)), g_ci=[r3(np.nanpercentile(bs, 2.5)), r3(np.nanpercentile(bs, 97.5))],
                na=int(len(a)), nb=int(len(b)))


def r3(x):
    return round(float(x), 3)


# ---------------------------------------------------------------- data


def num(s):
    return None if s in ("", None) else int(float(s))


def load():
    eps = {}
    for r in csv.DictReader(open(OUTCOMES)):
        k, t, i = r["kind"], int(r["task"]), int(r["init"])
        d = np.load(ATT / JOB[k] / FILE[k].format(t=t, i=i))
        assert len(d["t"]) == NC and (d["t"] == STEPS).all()
        e = dict(kind=k, task=t, init=i, success=r["success"] == "1", grasp=num(r["grasp_t"]), release=num(r["release_t"]),
                 failure=FAIL_SHORT.get(r["failure"], "") if r["success"] != "1" else "success",
                 instruction=r["instruction"], first_grasp=r["first_grasp"])
        e.update({key: d[key] for key in ("attn_mass", "attn_img", "exp_mlp", "exp_h", "vlm_h", "vlm_mlp", "sig_h", "act_pred", "seg_ntok")})
        eps[k, t, i] = e
    return eps


def bounds(e):
    """phase → (start, end) in steps, or None."""
    g, r = e["grasp"], e["release"]
    b = dict.fromkeys(PHASES)
    if g is None:
        b["approach"] = (0, T_END)
        return b
    b["approach"] = (0, max(g - 20, 1))
    b["grasp"] = (max(g - 20, 0), g + 20)
    if r is None:
        b["carry"] = (g + 20, T_END) if g + 20 < T_END - 10 else None
        return b
    b["carry"] = (g + 20, r - 20) if r - 20 > g + 20 else None
    b["release"] = (r - 20, min(r + 20, T_END))
    b["after"] = (r + 20, T_END) if r + 20 < T_END - 10 else None
    return b


def call_phase(e):
    """phase label per call (by call step)."""
    b = bounds(e)
    out = []
    for s in STEPS:
        lab = None
        for p in PHASES:
            if b[p] is not None and b[p][0] <= s < b[p][1]:
                lab = p
        out.append(lab or ("after" if b["after"] else "carry" if b["carry"] else "approach"))
    return out


def phase_interp(X, e):
    """X (NC, D) → (len(PHASES)*NB, D) features at phase-relative bins; NaN where the phase is missing."""
    b = bounds(e)
    out = np.full((len(PHASES) * NB, X.shape[1]), np.nan, np.float32)
    for p, ph in enumerate(PHASES):
        if b[ph] is None:
            continue
        s, en = b[ph]
        pos = s + (np.arange(NB) + 0.5) / NB * (en - s)
        f = np.clip(pos / 10, 0, NC - 1)
        lo = np.floor(f).astype(int)
        hi = np.minimum(lo + 1, NC - 1)
        w = (f - lo)[:, None]
        out[p * NB:(p + 1) * NB] = (1 - w) * X[lo] + w * X[hi]
    return out


def keyorder():
    return [(t, i) for t in range(10) for i in range(10)]


def feature_arrays(eps):
    """feat → kind → (100, NC, D) z-scored over all 300×NC calls."""
    F = {}
    for f, fn in FEATS.items():
        raw = {k: np.stack([fn(eps[k, t, i]) for t, i in keyorder()]).astype(np.float32) for k in KINDS}
        flat = np.concatenate([raw[k].reshape(-1, raw[k].shape[-1]) for k in KINDS])
        mu, sd = flat.mean(0), flat.std(0)
        keep = sd > 1e-6
        F[f] = {k: ((raw[k] - mu) / np.where(keep, sd, 1))[..., keep] for k in KINDS}
    return F


def rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2, -1))


# ---------------------------------------------------------------- figure helpers


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def phase_axis(ax):
    for p in range(1, len(PHASES)):
        ax.axvline(p * NB - 0.5, color="#cfcdc6", lw=0.8)
    ax.set_xticks([p * NB + (NB - 1) / 2 for p in range(len(PHASES))], PHASES, fontsize=7)


# ---------------------------------------------------------------- Q1: trial maps


def q1(eps, F, out, res):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    ko = keyorder()
    res["q1"] = {}
    for f in FEATS:
        Z = np.concatenate([F[f][k].reshape(100, -1) for k in KINDS])  # 300 × (NC·D); order: kind, task, init
        kind = np.repeat(KINDS, 100)
        task = np.tile(np.repeat(np.arange(10), 10), 3)
        ok = np.concatenate([[eps[k, t, i]["success"] for t, i in ko] for k in KINDS])
        P = PCA(n_components=50, random_state=0).fit_transform(Z)
        ctl = kind == "ctl"
        cent = np.stack([P[ctl & (task == t)].mean(0) for t in range(10)])
        rows = {}
        for k in KINDS:
            own, d_own, d_other, rel = [], [], [], []
            for j in np.where(kind == k)[0]:
                c = cent.copy()
                if k == "ctl":  # leave the trial itself out of its centroid
                    c[task[j]] = P[ctl & (task == task[j]) & (np.arange(300) != j)].mean(0)
                d = np.linalg.norm(c - P[j], axis=1)
                own.append(float(np.argmin(d) == task[j]))
                d_own.append(d[task[j]])
                d_other.append(np.delete(d, task[j]).min())
            rows[k] = dict(own=np.array(own), d_own=np.array(d_own), d_other=np.array(d_other))
        # per-task ctl spread: mean LOO distance of ctl trials to their own ctl centroid
        spread = np.array([rows["ctl"]["d_own"][task[ctl] == t].mean() for t in range(10)])
        r = {}
        for k in KINDS:
            m, lo, hi = boot_mean(rows[k]["own"])
            rel = rows[k]["d_own"] / spread[task[kind == k]]
            mr, lr, hr = boot_mean(rel)
            mo, lo2, ho2 = boot_mean(rows[k]["d_own"] / rows[k]["d_other"])
            r[k] = dict(nearest_ctl_centroid_is_own_task=[r3(m[0]), r3(lo[0]), r3(hi[0])],
                        dist_to_own_ctl_centroid_over_ctl_spread=[r3(mr[0]), r3(lr[0]), r3(hr[0])],
                        dist_own_over_nearest_other=[r3(mo[0]), r3(lo2[0]), r3(ho2[0])])
        # 30 centroids (task × condition), leave-one-out: own cell / own task other condition / other task
        cells = {(t, k): np.where((task == t) & (kind == k))[0] for t in range(10) for k in KINDS}
        cnt = defaultdict(lambda: defaultdict(int))
        for j in range(300):
            best, bd = None, np.inf
            for (t, k), ii in cells.items():
                ii = ii[ii != j]
                dd = np.linalg.norm(P[ii].mean(0) - P[j])
                if dd < bd:
                    best, bd = (t, k), dd
            cnt[kind[j]]["own cell" if best == (task[j], kind[j]) else "own task, other condition" if best[0] == task[j]
                         else "other task"] += 1
        for k in KINDS:
            r[k]["nearest_of_30_cells"] = {a: round(b / 100, 2) for a, b in cnt[k].items()}
        res["q1"][f] = r
        # maps
        fd = out / "7_trial_map" / f
        fd.mkdir(parents=True, exist_ok=True)
        T = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(P)
        MK = {"ctl": "o", "opp": "^", "near": "s"}
        for emb, Y in (("tsne", T), ("pca", P[:, :2])):
            fig, axes = plt.subplots(1, 2, figsize=(17, 8), facecolor=SURFACE)
            for c, colour_by in enumerate(["task", "condition"]):
                ax = axes[c]
                ax.set_facecolor(SURFACE)
                ax.set_xticks([])
                ax.set_yticks([])
                for j in range(300):
                    col = HUES[task[j]] if colour_by == "task" else HUE[kind[j]]
                    ax.scatter(Y[j, 0], Y[j, 1], s=34, marker=MK[kind[j]], color=col,
                               edgecolors=SURFACE if ok[j] else INK, linewidths=0.4 if ok[j] else 1.3, zorder=3)
                for t in range(10):
                    x, y = np.median(Y[ctl & (task == t)], 0)
                    ax.text(x, y, f"O{t + 1}", fontsize=8, ha="center", va="center", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.75))
                ax.set_title(f"colour = {colour_by}; label = ctl median of the task", fontsize=9)
            h = [plt.Line2D([], [], ls="", marker=MK[k], ms=7, mfc=MUTED, mec=SURFACE, label=LABEL[k]) for k in KINDS]
            h += [plt.Line2D([], [], ls="", marker="o", ms=7, mfc=SURFACE, mec=INK, mew=1.3, label="dark edge = failed")]
            h += [plt.Line2D([], [], ls="", marker="o", ms=7, mfc=HUES[t], mec=SURFACE,
                             label=f"O{t + 1} {eps['ctl', t, 0]['instruction'][12:40]}") for t in range(10)]
            fig.legend(handles=h, loc="lower center", ncol=7, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
            fig.suptitle(f"{TITLE[f]} — one point per trial (28 calls × {F[f]['ctl'].shape[-1]} dims, PCA 50) — {emb.upper()}",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0.06, 1, 0.97))
            fig.savefig(fd / f"trialmap_{emb}.png", dpi=110, facecolor=SURFACE)
            plt.close(fig)
        C = np.stack([P[cells[t, k]].mean(0) for t in range(10) for k in KINDS])
        labs = [f"O{t + 1}-{k}" for t in range(10) for k in KINDS]
        fig, ax = plt.subplots(figsize=(12, 4), facecolor=SURFACE)
        dendrogram(linkage(C, "average"), labels=labs, ax=ax, color_threshold=0, above_threshold_color=INK2, leaf_font_size=7)
        for lab in ax.get_xticklabels():
            lab.set_color(HUE[lab.get_text().split("-")[1]])
        ax.set_title(f"{TITLE[f]}: task × condition centroids, average linkage (colour = condition)", fontsize=9, loc="left")
        fig.tight_layout()
        fig.savefig(fd / "dendrogram.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)
        print(f"[q1] {f}: " + "  ".join(f"{k} own={r[k]['nearest_ctl_centroid_is_own_task'][0]} "
                                        f"rel={r[k]['dist_to_own_ctl_centroid_over_ctl_spread'][0]}" for k in KINDS), flush=True)
    (out / "7_trial_map").mkdir(parents=True, exist_ok=True)
    (out / "7_trial_map" / "cluster_numbers.json").write_text(json.dumps(res["q1"], indent=1))
    # summary bar
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), facecolor=SURFACE)
    x = np.arange(len(FEATS))
    for ax, key, yl in [(axes[0], "nearest_ctl_centroid_is_own_task", "nearest ctl task centroid = own task"),
                        (axes[1], "dist_to_own_ctl_centroid_over_ctl_spread", "distance to own ctl centroid\n÷ ctl trials' own spread")]:
        for j, k in enumerate(KINDS):
            v = np.array([res["q1"][f][k][key] for f in FEATS])
            ax.bar(x + (j - 1) * 0.27, v[:, 0], 0.25, color=HUE[k], label=LABEL[k])
            ax.errorbar(x + (j - 1) * 0.27, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor=INK, lw=0.8)
        ax.set_xticks(x, list(FEATS), fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        clean(ax)
    axes[1].axhline(1, color=MUTED, ls=":", lw=1)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Q1: do moved-basket trials stay with their task's ctl cluster? (PCA-50 of whole trials; 95% bootstrap CI)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "7_trial_map" / "task_identity.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- Q2: divergence over time


def baseline_ctl(Xc):
    """Xc (100, B, D) ctl: per trial, mean distance to the other 9 inits of the same task at the same bin."""
    out = np.full(Xc.shape[:2], np.nan)
    for t in range(10):
        blk = Xc[t * 10:(t + 1) * 10]
        for a in range(10):
            ds = [rms(blk[a], blk[b]) for b in range(10) if b != a]
            with np.errstate(all="ignore"):
                out[t * 10 + a] = np.nanmean(ds, 0)
    return out


def q2(eps, F, out, res):
    ko = keyorder()
    PI = {f: {k: np.stack([phase_interp(F[f][k][j], eps[k, t, i]) for j, (t, i) in enumerate(ko)]) for k in KINDS} for f in FEATS}
    D = {"call": {}, "phase": {}}
    for f in FEATS:
        for axis, A in (("call", F[f]), ("phase", PI[f])):
            D[axis][f] = {k: rms(A[k], A["ctl"]) for k in MOVED}
            D[axis][f]["base"] = baseline_ctl(A["ctl"])
            D[axis][f]["opp_near"] = rms(A["opp"], A["near"])
    res["q2"] = {}
    for axis, nb, xl in (("call", NC, "policy call (×10 steps)"), ("phase", len(PHASES) * NB, None)):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), facecolor=SURFACE)
        for ax, f in zip(axes.flat, FEATS, strict=True):
            for k in MOVED + ["base"]:
                m, lo, hi = boot_mean(D[axis][f][k])
                col = HUE.get(k, MUTED)
                lab = {"opp": "opp vs ctl, same init", "near": "near vs ctl, same init",
                       "base": "ctl vs ctl, other inits (same task)"}[k]
                ax.plot(m, color=col, lw=1.8, ls="--" if k == "base" else "-", label=lab)
                ax.fill_between(np.arange(nb), lo, hi, color=col, alpha=0.18, lw=0)
            ax.set_title(TITLE[f], fontsize=9)
            ax.set_ylabel("RMS distance (z units)", fontsize=7)
            clean(ax)
            if axis == "phase":
                phase_axis(ax)
            else:
                ax.set_xlabel(xl, fontsize=7)
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle("Q2: distance to the ctl episode of the same task·init, " +
                     ("by policy call" if axis == "call" else "phase-aligned (5 bins per phase; grasp/release = ±20 steps)") +
                     "; mean of 100 pairs, 95% bootstrap CI", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "6_trajectories" / f"divergence_by_{axis}.png", dpi=115, facecolor=SURFACE)
        plt.close(fig)
    # numbers: per phase, ratio of means vs the ctl baseline, and opp − near paired
    for f in FEATS:
        r = {}
        for p, ph in enumerate(PHASES):
            sl = slice(p * NB, (p + 1) * NB)
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    v = {k: np.nanmean(D["phase"][f][k][:, sl], 1) for k in MOVED + ["base"]}
            r[ph] = {f"{k}_over_base": [r3(x) for x in boot_ratio(v[k], v["base"])[:3]] for k in MOVED}
            r[ph]["opp_minus_near"] = dz(v["opp"] - v["near"])
        # by call: first call where the ratio's lower CI exceeds 1 (moved trial as far from its own-init ctl as other inits)
        for k in MOVED:
            first = None
            for c in range(NC):
                m, lo, hi, _ = boot_ratio(D["call"][f][k][:, c], D["call"][f]["base"][:, c])
                if lo > 1 and first is None:
                    first = c
            r[f"{k}_first_call_above_base"] = first
            r[f"{k}_call0_over_base"] = [r3(x) for x in boot_ratio(D["call"][f][k][:, 0], D["call"][f]["base"][:, 0])[:3]]
        res["q2"][f] = r
        print(f"[q2] {f}: " + " ".join(f"{ph}:{r[ph]['opp_over_base'][0]}/{r[ph]['near_over_base'][0]}" for ph in PHASES), flush=True)
    return PI, D


def trajectories(eps, F, out):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    ko = keyorder()
    for f in FEATS:
        X = np.concatenate([F[f][k].reshape(-1, F[f][k].shape[-1]) for k in KINDS])
        Pc = PCA(n_components=min(30, X.shape[1]), random_state=0).fit_transform(X)
        Tc = TSNE(n_components=2, perplexity=50, init="pca", learning_rate="auto", random_state=0).fit_transform(Pc)
        Tc = Tc.reshape(3, 100, NC, 2)
        fig, axes = plt.subplots(2, 5, figsize=(17, 7.5), facecolor=SURFACE)
        for t, ax in enumerate(axes.flat):
            ax.set_facecolor(SURFACE)
            ax.set_xticks([])
            ax.set_yticks([])
            for a in range(3):
                for j in range(100):
                    if j // 10 != t:
                        ax.plot(Tc[a, j, :, 0], Tc[a, j, :, 1], color=MUTED, lw=0.3, alpha=0.08, zorder=1)
            for a, k in enumerate(KINDS):
                for j in range(t * 10, t * 10 + 10):
                    e = eps[k, *ko[j]]
                    ax.plot(Tc[a, j, :, 0], Tc[a, j, :, 1], color=HUE[k], lw=0.9, alpha=0.8, ls="-" if e["success"] else ":",
                            zorder=3)
                    ax.scatter(Tc[a, j, 0, 0], Tc[a, j, 0, 1], s=12, color=HUE[k], zorder=4)
                    if e["grasp"] is not None:
                        c = min(e["grasp"] // 10, NC - 1)
                        ax.scatter(Tc[a, j, c, 0], Tc[a, j, c, 1], s=14, marker="x", color=HUE[k], lw=1, zorder=4)
            ax.set_title(f"O{t + 1}: {eps['ctl', t, 0]['instruction'][12:50]}", fontsize=7.5, loc="left")
        h = [plt.Line2D([], [], color=HUE[k], label=LABEL[k]) for k in KINDS]
        h += [plt.Line2D([], [], color=INK, ls=":", label="failed"),
              plt.Line2D([], [], ls="", marker="o", color=INK, ms=4, label="first call"),
              plt.Line2D([], [], ls="", marker="x", color=INK, ms=5, label="first grasp")]
        fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8, frameon=False)
        fig.suptitle(f"{TITLE[f]} — t-SNE of all 300×28 calls; one line per trial (gray = other tasks)", fontsize=10)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.91, wspace=0.05, hspace=0.15)
        (out / "6_trajectories" / f).mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "6_trajectories" / f / "traj_tsne.png", dpi=105, facecolor=SURFACE)
        plt.close(fig)
        print(f"[traj] {f}", flush=True)


# ---------------------------------------------------------------- Q3: attention by input type


def q3(eps, out, res):
    ko = keyorder()
    # A[k] (100, 5 phases, 3 groups, 7 segments), share in %
    A = {}
    for k in KINDS:
        arr = np.full((100, len(PHASES), 3, 7), np.nan)
        for j, (t, i) in enumerate(ko):
            e = eps[k, t, i]
            m = e["attn_mass"].mean(1)  # NC × 18 × 7
            lab = np.array(call_phase(e))
            for p, ph in enumerate(PHASES):
                if (lab == ph).any():
                    for g, (_, lo, hi) in enumerate(LGROUPS):
                        arr[j, p, g] = m[lab == ph, lo:hi].mean((0, 1)) * 100
        A[k] = arr
    res["q3"] = {"share_pct": {}, "diff_pp": {}}
    fig, axes = plt.subplots(3, len(PHASES), figsize=(18, 9), sharey="row", facecolor=SURFACE)
    x = np.arange(7)
    for g, (gn, _, _) in enumerate(LGROUPS):
        for p, ph in enumerate(PHASES):
            ax = axes[g, p]
            for j, k in enumerate(KINDS):
                m, lo, hi = boot_mean(A[k][:, p, g])
                ax.bar(x + (j - 1) * 0.27, m, 0.25, color=HUE[k], label=LABEL[k])
                ax.errorbar(x + (j - 1) * 0.27, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, lw=0.6)
                res["q3"]["share_pct"][f"{gn}|{ph}|{k}"] = {s: r3(v) for s, v in zip(SEG_LABEL, m, strict=True)}
            ax.set_xticks(x, SEG_LABEL, fontsize=7, rotation=30)
            ax.set_title(f"{gn} · {ph}", fontsize=9)
            clean(ax)
        axes[g, 0].set_ylabel("% of action-expert attention", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Q3: action-expert attention share by input type, layer group × phase (mean over denoising steps and the "
                 "phase's calls; 95% bootstrap CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "2_attention" / "attention_by_type.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    # paired differences
    fig, axes = plt.subplots(3, len(PHASES), figsize=(18, 9), sharey="row", facecolor=SURFACE)
    for g, (gn, _, _) in enumerate(LGROUPS):
        for p, ph in enumerate(PHASES):
            ax = axes[g, p]
            for j, k in enumerate(MOVED):
                d = A[k][:, p, g] - A["ctl"][:, p, g]
                m, lo, hi = boot_mean(d)
                ax.bar(x + (j - 0.5) * 0.36, m, 0.34, color=HUE[k], label=f"{k} − ctl")
                ax.errorbar(x + (j - 0.5) * 0.36, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, lw=0.6)
                for s in range(7):
                    res["q3"]["diff_pp"][f"{gn}|{ph}|{k}-ctl|{SEG_LABEL[s]}"] = dz(d[:, s])
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(x, SEG_LABEL, fontsize=7, rotation=30)
            ax.set_title(f"{gn} · {ph}", fontsize=9)
            clean(ax)
        axes[g, 0].set_ylabel("paired difference (pp)", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Q3: moved − ctl attention share at the same task·init (percentage points; 95% bootstrap CI). "
                 "Phase windows differ per episode (aligned to its own grasp / release).", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "2_attention" / "attention_by_type_diff.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return A


def activation(eps, out, res):
    """3_activation: relative change moved/ctl − 1 per layer × phase bin, paired, for several magnitude signals."""
    ko = keyorder()
    sigs = [("expert |MLP|", lambda e: e["exp_mlp"].mean(1)),
            ("expert hidden norm", lambda e: e["exp_h"].mean(1)),
            ("Gemma |MLP| agentview tokens", lambda e: e["vlm_mlp"][..., 0]),
            ("Gemma |MLP| wrist tokens", lambda e: e["vlm_mlp"][..., 1]),
            ("Gemma |MLP| sentence tokens", lambda e: e["vlm_mlp"][..., 3]),
            ("SigLIP hidden norm, agentview", lambda e: e["sig_h"][:, 0]),
            ("SigLIP hidden norm, wrist", lambda e: e["sig_h"][:, 1])]
    res["activation"] = {}
    (out / "3_activation").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(sigs), 2, figsize=(13, 2.3 * len(sigs)), facecolor=SURFACE)
    for r, (name, fn) in enumerate(sigs):
        V = {k: np.stack([phase_interp(fn(eps[k, t, i]), eps[k, t, i]) for t, i in ko]) for k in KINDS}  # 100 × 25 × L
        for c, k in enumerate(MOVED):
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rel = np.nanmean(V[k] / V["ctl"] - 1, 0) * 100  # 25 × L
            ax = axes[r, c]
            lim = max(np.nanmax(np.abs(rel)), 0.5)
            im = ax.imshow(rel.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            phase_axis(ax)
            ax.set_ylabel("layer", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_title(f"{name}: {k} / ctl − 1 (%)", fontsize=8.5)
            fig.colorbar(im, ax=ax, pad=0.01)
            res["activation"][f"{name}|{k}"] = {ph: r3(np.nanmean(np.abs(rel[p * NB:(p + 1) * NB]))) for p, ph in enumerate(PHASES)}
    fig.suptitle("relative activation change at the same task·init, phase-aligned (mean over 100 pairs); "
                 "numbers in 5_analysis/numbers.json are the mean |change| per phase", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "3_activation" / "activation_change_by_phase.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- Q4: failures and near-between


def q4(eps, F, PI, D, A, out, res):
    ko = keyorder()
    res["q4"] = {"failure_vs_success": {}, "near_between": {}, "attention_fail_vs_success": {}}
    # distance-to-ctl (phase-aligned) by outcome group
    fig, axes = plt.subplots(len(FEATS), 2, figsize=(13, 2.5 * len(FEATS)), sharex=True, facecolor=SURFACE)
    for r, f in enumerate(FEATS):
        for c, k in enumerate(MOVED):
            ax = axes[r, c]
            fl = np.array([eps[k, t, i]["failure"] for t, i in ko])
            ctl_ok = np.array([eps["ctl", t, i]["success"] for t, i in ko])
            d = D["phase"][f][k]
            for grp in ["success"] + FAIL_TYPES:
                sel = (fl == grp) & ctl_ok
                if sel.sum() == 0:
                    continue
                m, lo, hi = boot_mean(d[sel])
                ax.plot(m, color=FAIL_HUE[grp], lw=1.6, label=f"{grp} (n={sel.sum()})")
                if sel.sum() >= 5:
                    ax.fill_between(np.arange(len(m)), lo, hi, color=FAIL_HUE[grp], alpha=0.13, lw=0)
            m, _, _ = boot_mean(D["phase"][f]["base"])
            ax.plot(m, color=MUTED, ls="--", lw=1, label="ctl vs ctl, other inits")
            phase_axis(ax)
            clean(ax)
            ax.set_title(f"{TITLE[f]} — {LABEL[k]}", fontsize=8.5)
            if c == 0:
                ax.set_ylabel("RMS dist. to own-init ctl", fontsize=7)
            # effect sizes per phase: each failure type (and all failures) − success
            ok = fl == "success"
            for p, ph in enumerate(PHASES):
                with np.errstate(all="ignore"):
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        v = np.nanmean(d[:, p * NB:(p + 1) * NB], 1)
                key = f"{f}|{k}|{ph}"
                res["q4"]["failure_vs_success"][key] = {"all failures": hedges(v[(fl != "success") & ctl_ok], v[ok & ctl_ok])}
                # failures cluster in a few tasks: subtract each task's mean (this condition) to compare within task
                with np.errstate(all="ignore"):
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        vt = v - np.repeat([np.nanmean(v[t * 10:(t + 1) * 10]) for t in range(10)], 10)
                res["q4"]["failure_vs_success"][key]["all failures, task-demeaned"] = hedges(
                    vt[(fl != "success") & ctl_ok], vt[ok & ctl_ok])
                for grp in FAIL_TYPES:
                    res["q4"]["failure_vs_success"][key][grp] = hedges(v[(fl == grp) & ctl_ok], v[ok & ctl_ok])
        axes[r, 0].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.suptitle("Q4: distance to the own-init ctl episode, by outcome (pairs whose ctl episode succeeded; phase-aligned; "
                 "band = 95% CI, shown for n ≥ 5)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "5_analysis" / "failure_divergence.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)
    # early-warning: does the distance in the first calls (approach) already differ? (by call, calls 0–4)
    res["q4"]["early_calls_0_4"] = {}
    for f in FEATS:
        for k in MOVED:
            fl = np.array([eps[k, t, i]["failure"] for t, i in ko])
            v = D["call"][f][k][:, :5].mean(1)
            res["q4"]["early_calls_0_4"][f"{f}|{k}"] = {"all failures": hedges(v[fl != "success"], v[fl == "success"])} | {
                grp: hedges(v[fl == grp], v[fl == "success"]) for grp in FAIL_TYPES}
    # attention by type: failure − success within condition (carry and release phases, late layers)
    for k in MOVED:
        fl = np.array([eps[k, t, i]["failure"] for t, i in ko])
        for p, ph in enumerate(PHASES):
            for g, (gn, _, _) in enumerate(LGROUPS):
                for s in range(7):
                    v = A[k][:, p, g, s] - A["ctl"][:, p, g, s]
                    res["q4"]["attention_fail_vs_success"][f"{k}|{ph}|{gn}|{SEG_LABEL[s]}"] = hedges(v[fl != "success"], v[fl == "success"])
    # near between ctl and opp? per phase bin: λ = <near−ctl, opp−ctl>/|opp−ctl|², cos, |near−ctl|/|opp−ctl|
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor=SURFACE)
    for f in FEATS:
        c0, o, n = PI[f]["ctl"], PI[f]["opp"], PI[f]["near"]
        u, v = n - c0, o - c0
        with np.errstate(all="ignore"):
            lam = (u * v).sum(-1) / (v * v).sum(-1)
            cos = (u * v).sum(-1) / np.sqrt((u * u).sum(-1) * (v * v).sum(-1))
        dn, do, don = rms(n, c0), rms(o, c0), rms(o, n)
        r = {}
        for p, ph in enumerate(PHASES):
            sl = slice(p * NB, (p + 1) * NB)
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vv = {nm: np.nanmean(x[:, sl], 1) for nm, x in
                          (("lam", lam), ("cos", cos), ("dn", dn), ("do", do), ("don", don))}
            r[ph] = dict(lambda_on_ctl_to_opp=[r3(x) for x in np.concatenate(boot_mean(vv["lam"]))],
                         cos_near_shift_vs_opp_shift=[r3(x) for x in np.concatenate(boot_mean(vv["cos"]))],
                         d_near_ctl_over_d_opp_ctl=[r3(x) for x in boot_ratio(vv["dn"], vv["do"])[:3]],
                         d_near_opp_over_d_opp_ctl=[r3(x) for x in boot_ratio(vv["don"], vv["do"])[:3]])
        res["q4"]["near_between"][f] = r
        rb = np.array([boot_ratio(dn[:, b], do[:, b])[:3] for b in range(dn.shape[1])]).T  # ratio of means per bin
        for ax, arr, yl in ((axes[0], lam, "λ: projection of near−ctl on opp−ctl"),
                            (axes[1], cos, "cos(near−ctl, opp−ctl)"),
                            (axes[2], None, "mean |near−ctl| / mean |opp−ctl| (RMS)")):
            m, lo, hi = boot_mean(arr) if arr is not None else rb
            col = HUES[list(FEATS).index(f)]
            ax.plot(m, color=col, lw=1.6, label=f)
            ax.fill_between(np.arange(len(m)), lo, hi, color=col, alpha=0.12, lw=0)
            ax.set_title(yl, fontsize=9)
    for ax in axes:
        phase_axis(ax)
        clean(ax)
        ax.axhline(0, color=INK2, lw=0.7)
    axes[2].axhline(1, color=MUTED, ls=":", lw=1)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Q4: is near between ctl and opp? per task·init, phase-aligned (λ = 0: like ctl, 1: like opp along the "
                 "ctl→opp direction); 95% bootstrap CI", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "5_analysis" / "near_between.png", dpi=115, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=K / "basket")
    ap.add_argument("--skip-tsne", action="store_true", help="skip the slow per-call t-SNE trajectory figures")
    a = ap.parse_args()
    out = a.out
    for d in ("2_attention", "3_activation", "5_analysis", "6_trajectories", "7_trial_map"):
        (out / d).mkdir(parents=True, exist_ok=True)
    eps = load()
    print(f"[load] {len(eps)} episodes", flush=True)
    F = feature_arrays(eps)
    res = {"n": {k: dict(success=int(sum(eps[k, t, i]["success"] for t, i in keyorder()))) for k in KINDS},
           "dims": {f: int(F[f]["ctl"].shape[-1]) for f in FEATS}}
    for k in KINDS:
        fl = [eps[k, t, i]["failure"] for t, i in keyorder()]
        res["n"][k]["outcomes"] = {g: fl.count(g) for g in ["success"] + FAIL_TYPES}
    A = q3(eps, out, res)
    activation(eps, out, res)
    PI, D = q2(eps, F, out, res)
    q4(eps, F, PI, D, A, out, res)
    q1(eps, F, out, res)
    if not a.skip_tsne:
        trajectories(eps, F, out)
    (out / "5_analysis" / "numbers.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("[done]", out, flush=True)


if __name__ == "__main__":
    main()
