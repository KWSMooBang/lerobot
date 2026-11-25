"""T032 figures + numbers.json (stages `figs`, `numbers` of sae_vlm_traj.py)."""

from __future__ import annotations

import json
import pickle
import time

import layer_pop as lp
import numpy as np
import sae_vlm_traj as sv
from sae_vlm_traj import (
    CACHE,
    FB_CALLS,
    FIG,
    LAYERS,
    N_L,
    OUT,
    PHASE4,
    SET_OF,
    TRAJ_LAYERS,
    lab_rows,
    lname,
    load_json,
    tag_arr,
)

CAT = lp.CAT
TAGS = ("task", "scene", "suite", "phase", "always-on", "mixed", "rare")


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as p

    p.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.5,
            "lines.linewidth": 2,
        }
    )
    return p


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as p

    p.close(fig)


# ---------------------------------------------------------------- A1
def fig_recon() -> None:
    p = plt()
    E = load_json("encode")["layers"]
    x = [r["layer"] for r in E]
    fig, ax = p.subplots(1, 4, figsize=(16, 3.2))
    ax[0].plot(x, [r["train_ve"] for r in E], color=CAT[0], marker="o", ms=4, label="training (T015)")
    ax[0].plot(x, [r["ve"] for r in E], color=CAT[1], marker="o", ms=4, label="T028 (all calls)")
    ax[0].axhline(0.7, color="#999", ls=":", lw=1)
    ax[0].set(title="variance explained (raw space)", xlabel="layer pgNN", ylim=(0.6, 1.0))
    ax[0].legend(frameon=False)
    ax[1].plot(x, [r["train_nmse"] for r in E], color=CAT[0], marker="o", ms=4, label="training")
    ax[1].plot(x, [r["nmse"] for r in E], color=CAT[1], marker="o", ms=4, label="T028")
    ax[1].set(title="normalized MSE", xlabel="layer pgNN")
    ax[1].legend(frameon=False)
    for i, k_ in enumerate(("train", "cand", "l90", "bskA", "bskV")):
        ax[2].plot(x, [r["per_set"][k_]["ve"] for r in E], color=CAT[i], lw=1.5, label=SET_OF[k_])
    ax[2].set(title="T028 VE by episode set", xlabel="layer pgNN", ylim=(0.6, 1.0))
    ax[2].legend(frameon=False, fontsize=7)
    ax[3].plot(
        x,
        [r["fire_rho"] for r in E],
        color=CAT[2],
        marker="o",
        ms=4,
        label="Spearman ρ, fire freq train vs T028",
    )
    ax[3].plot(x, [r["never_fire"] for r in E], color=CAT[3], marker="o", ms=4, label="never-fire, T028")
    ax[3].plot(
        x, [r["train_never_fire"] for r in E], color=CAT[4], marker="o", ms=4, label="never-fire, training"
    )
    ax[3].set(title="feature usage", xlabel="layer pgNN", ylim=(0, 1))
    ax[3].legend(frameon=False, fontsize=7)
    _save(fig, "1_reconstruction")


# ---------------------------------------------------------------- A2
def fig_tags() -> None:
    p = plt()
    L = load_json("labels")
    lay = list(L)
    fig, ax = p.subplots(1, 2, figsize=(16, 3.6), gridspec_kw={"width_ratios": [2, 1], "wspace": 0.55})
    bottom = np.zeros(len(lay))
    parts = [
        ("task: instruction (same scene)", lambda v: v["sub"].get("instruction", 0)),
        ("task: same-scene group, weak", lambda v: v["sub"].get("task-shared-scene", 0)),
        ("task: scene+instruction confounded", lambda v: v["sub"].get("task-confounded", 0)),
    ]
    parts += [(t, (lambda t_: (lambda v: v["tags"].get(t_, 0)))(t)) for t in TAGS[1:]]
    cols = list(CAT[:8]) + ["#bdbdbd"]  # 9th category (rare) = neutral gray, not a cycled hue
    for i, (nm, fn) in enumerate(parts):
        h = np.array([fn(L[l_]) for l_ in lay])
        ax[0].bar(
            range(len(lay)), h, bottom=bottom, color=cols[i], width=0.8, label=nm, edgecolor="white", lw=0.6
        )
        bottom += h
    ax[0].set_xticks(range(len(lay)), lay, rotation=60)
    ax[0].set(title="primary tag per feature (2048 per layer, labels from training data)", ylabel="features")
    ax[0].legend(frameon=False, fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ph = np.array([[L[l_]["pref_phase_of_phase"].get(q, 0) for q in PHASE4] for l_ in lay])
    for i, q in enumerate(PHASE4):
        ax[1].plot(range(len(lay)), ph[:, i], color=CAT[i], label=f"phase tag, prefers {q}", lw=1.5)
    ax[1].set_xticks(range(len(lay)), lay, rotation=60)
    ax[1].set(title="phase-tagged features by preferred phase", ylabel="features")
    ax[1].legend(frameon=False, fontsize=7)
    _save(fig, "2_tags")
    info = load_json("trainlab")
    for li in (4, 12):
        z = np.load(CACHE / f"lab_{lname(li)}.npz")
        rows = lab_rows(li)
        sel = [int(r["feature"]) for r in rows if r["tag"] == "task"]
        pick = []
        for t in info["codes"]:
            ft = sorted(
                [f for f in sel if rows[f]["top_task"] == t], key=lambda f: -float(rows[f]["top_task_rate"])
            )
            pick += ft[:3]
        M = z["r_t"][:, pick]
        fig, ax = p.subplots(figsize=(14, 4.2))
        im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=0.6, interpolation="nearest")
        ax.set_yticks(range(40), info["codes"], fontsize=5)
        ax.set_xticks(
            range(len(pick)),
            [f"{f}{'*' if rows[f]['sub'] == 'instruction' else ''}" for f in pick],
            rotation=90,
            fontsize=4,
        )
        ax.set(
            title=f"{lname(li)}: training fire rate per task, task-specific features (top 3 per task; * = instruction, same scene)",
            xlabel="feature",
            ylabel="training task",
        )
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.02, label="fire rate")
        _save(fig, f"2_task_rates_{lname(li)}")


# ---------------------------------------------------------------- A3
def _grad_line(ax, xy, color, gc=None, lw=1.0, alpha=1.0):
    from matplotlib.colors import to_rgb

    n = len(xy)
    base = np.array(to_rgb(color))
    for i in range(n - 1):
        f = 0.25 + 0.75 * i / max(n - 2, 1)
        ax.plot(xy[i : i + 2, 0], xy[i : i + 2, 1], color=1 - f * (1 - base), lw=lw, alpha=alpha)
    ax.plot(*xy[0], "o", mfc="white", mec=color, ms=5, mew=1.2)
    ax.plot(*xy[-1], "s", color=color, ms=4)
    if gc is not None and gc < n:
        ax.plot(*xy[gc], "x", color=color, ms=6, mew=1.5)


def fig_traj() -> None:
    p = plt()
    from matplotlib.lines import Line2D

    T = pickle.load(open(CACHE / "traj.pkl", "rb"))
    byl = {t["layer"]: t for t in T}
    for case in T[0]["cases"]:
        conds = list(T[0]["cases"][case]["heat"])
        fig, axs = p.subplots(
            len(TRAJ_LAYERS),
            len(conds),
            figsize=(2.6 * len(conds) + 1.5, 2.3 * len(TRAJ_LAYERS)),
            squeeze=False,
        )
        for r, li in enumerate(TRAJ_LAYERS):
            C = byl[li]["cases"][case]
            vmax = max(float(np.quantile(C["heat"][c], 0.99)) for c in conds) or 1.0
            tg = tag_arr(li)
            rows = lab_rows(li)
            for j, c in enumerate(conds):
                ax = axs[r, j]
                ax.imshow(
                    C["heat"][c], aspect="auto", cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest"
                )
                ax.grid(False)
                if j == 0:
                    ax.set_yticks(
                        range(30),
                        [
                            f"{f} {tg[f][:5]}{' ' + rows[f]['top_task'] if tg[f] == 'task' else ''}"
                            for f in C["top"]
                        ],
                        fontsize=4,
                    )
                    ax.set_ylabel(lname(li) + (" (ref)" if li == 17 else ""))
                else:
                    ax.set_yticks([])
                if r == 0:
                    ax.set_title(c, fontsize=7)
                if r == len(TRAJ_LAYERS) - 1:
                    ax.set_xlabel("call")
        fig.suptitle(
            f"{case}: top-30 varying SAE features (condition-mean activation, feature × call). "
            "Row label = feature id, tag, top training task",
            fontsize=9,
        )
        _save(fig, f"3_heat_{case}")
        fig, axs = p.subplots(3, 6, figsize=(18, 8.5))
        for li in LAYERS:
            ax = axs.flat[li]
            for j, c in enumerate(conds):
                tr = byl[li]["cases"][case]["pc"][c]
                for t in tr[:10]:
                    _grad_line(ax, t["pc"], CAT[j], t["gc"], lw=0.5, alpha=0.5)
                n = min(len(t["pc"]) for t in tr)
                mean = np.mean([t["pc"][:n] for t in tr], 0)
                gcs = [t["gc"] for t in tr if t["gc"] is not None]
                _grad_line(ax, mean, CAT[j], int(np.median(gcs)) if gcs else None, lw=2)
            ax.set_title(lname(li) + (" (ref)" if li == 17 else ""), fontsize=8)
            ax.tick_params(labelsize=6)
        fig.legend(
            [Line2D([], [], color=CAT[j]) for j in range(len(conds))],
            conds,
            loc="lower center",
            ncol=len(conds),
            frameon=False,
        )
        fig.suptitle(
            f"{case}: SAE code trajectories, PC1–PC2 of a per-layer PCA fitted on U40 training episodes "
            "(light→dark = time, ○ start, × first grasp, ■ end; thick = condition mean)",
            fontsize=9,
        )
        _save(fig, f"3_pca_{case}")


# ---------------------------------------------------------------- A4
def fallback_matrix(R, key, src=None, calls=FB_CALLS, rec="recs", kind=None):
    A = np.full((N_L, calls), np.nan)
    for r in R:
        for c in range(calls):
            v = [
                x[key][c]
                for x in r[rec]
                if (src is None or x["src"] == src)
                and (kind is None or x.get("kind") == kind)
                and len(x[key]) > c
                and np.isfinite(x[key][c])
            ]
            if len(v) >= 3:
                A[r["layer"], c] = np.median(v)
    return A


def fig_fallback() -> None:
    p = plt()
    R = pickle.load(open(CACHE / "fallback.pkl", "rb"))
    fig, axs = p.subplots(2, 4, figsize=(19, 7))
    cols = (
        ("cand", None, "recs", "task features, cand31"),
        ("l90", None, "recs", "task features, L90 lscene"),
        (None, None, "recs", "task features, all"),
        (None, "object", "odrec", "object features (n=8)"),
    )
    for i, (src, kind, rec, nm) in enumerate(cols):
        for j, key in enumerate(("idx_I", "idx_E")):
            ax = axs[j, i]
            A = fallback_matrix(R, key, src, rec=rec, kind=kind)
            im = ax.imshow(
                A, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
            )
            ax.grid(False)
            ax.set(
                title=f"{'INSTRUCTED' if key == 'idx_I' else 'EXECUTED'} — {nm}",
                xlabel="call",
                ylabel="layer pgNN",
            )
    fig.colorbar(
        im,
        ax=axs,
        fraction=0.02,
        label="feature index (median over trials): 0 = as off as when doing the other task, "
        "1 = as on as when really doing it",
    )
    fig.suptitle(
        "Fallback trials: training-labelled features of the instructed (top) and executed (bottom) task / object",
        fontsize=10,
    )
    _save(fig, "4_fallback_index")
    fig, ax = p.subplots(1, 2, figsize=(11, 3.2))
    for c in (0, 1, 2, 3):
        ax[0].plot(
            LAYERS,
            [r["pairs"]["calls"][c]["n_sig"] for r in R],
            color=CAT[c],
            marker="o",
            ms=3,
            label=f"call {c}",
        )
        ax[1].plot(
            LAYERS,
            [r["pairs"]["calls"][c]["frac_fire_changed"] / 2 for r in R],
            color=CAT[c],
            marker="o",
            ms=3,
            label=f"call {c}",
        )
    ax[0].set(
        title="same-observation pairs (cand vs base, n=150):\nfeatures differing, BH q<0.05",
        xlabel="layer pgNN",
        ylabel="features",
    )
    ax[1].set(
        title="share of the 100 active features that swap\n(call 0 = identical observation, only the instruction differs)",
        xlabel="layer pgNN",
        ylim=(0, 0.7),
    )
    for a in ax:
        a.legend(frameon=False, fontsize=7)
    _save(fig, "4_pairs")


# ---------------------------------------------------------------- A5
def fig_basket() -> None:
    p = plt()
    B = pickle.load(open(CACHE / "basket.pkl", "rb"))
    fig, axs = p.subplots(1, 3, figsize=(15, 3.6))
    for k_, win in enumerate(("call0", "carry")):
        ax = axs[k_]
        bottom = np.zeros(N_L)
        for i, cl in enumerate(("graded", "on/off", "other")):
            h = np.array([b[win]["class"].get(cl, 0) for b in B])
            ax.bar(LAYERS, h, bottom=bottom, color=CAT[i], label=cl, edgecolor="white", lw=0.6)
            bottom += h
        ax.set(
            title=f"basket-sensitive features ({win}; ctl4090 vs far25, BH q<0.05)",
            xlabel="layer pgNN",
            ylabel="features",
        )
        ax.legend(frameon=False)
    b = B[8]["call0"]
    ax = axs[2]
    cl = np.array(b["cls"])
    for i, c in enumerate(("graded", "on/off", "other")):
        m = cl == c
        ax.scatter(np.array(b["m25"])[m], np.array(b["m20"])[m], s=10, color=CAT[i], label=c)
    lim = float(np.abs(np.concatenate([b["m25"], b["m20"]])).max())
    ax.plot([-lim, lim], [-lim, lim], color="#999", lw=1, ls=":", label="λ = 1")
    ax.plot([-lim, lim], [-0.8 * lim, 0.8 * lim], color="#555", lw=1, ls="--", label="geometric λ = 0.80")
    ax.set(
        title="pg08 call 0: mean Δ per feature",
        xlabel="Δ activation, far25 − ctl",
        ylabel="Δ activation, far20 − ctl",
    )
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "5_basket")


# ---------------------------------------------------------------- A6
def fig_kauf() -> None:
    p = plt()
    K = pickle.load(open(CACHE / "kauf.pkl", "rb"))
    fig, ax = p.subplots(figsize=(12, 3.4))
    for i, t in enumerate(("task", "instruction", "phase", "mixed")):
        med, lo, hi = [], [], []
        for k in K:
            tg = tag_arr(k["layer"])
            sub = np.array([r["sub"] for r in lab_rows(k["layer"])])
            v = k["pf"][sub == "instruction"] if t == "instruction" else k["pf"][tg == t]
            med.append(np.median(v) if len(v) else np.nan)
            lo.append(np.quantile(v, 0.1) if len(v) else np.nan)
            hi.append(np.quantile(v, 0.9) if len(v) else np.nan)
        med, lo, hi = map(np.array, (med, lo, hi))
        x = np.array(LAYERS) + (i - 1.5) * 0.15
        ax.errorbar(x, med, yerr=[med - lo, hi - med], color=CAT[i], fmt="o", ms=3, lw=1, label=t)
    ax.axhline(0.25, color="#555", ls="--", lw=1, label="chance (512/2048)")
    ax.set(
        title="read-in potent fraction of SAE decoder directions under the next layer's K/V "
        "(median, 10–90%; pg17 = own read-in, proxy only)",
        xlabel="layer pgNN (reader = pgNN+1)",
        ylabel="‖Pᵀd‖²",
    )
    ax.legend(frameon=False, fontsize=7, ncol=5)
    _save(fig, "6_kaufman")


# ---------------------------------------------------------------- A7
def t030_dec() -> dict:
    d = pickle.load(open(lp.CACHE / "dec_parts.pkl", "rb"))
    out = {}
    for rep in ("v_all6", "v_task"):
        e, i_, t40 = [], [], []
        for li in LAYERS:
            o = d[(rep, li, "z")]
            ex = [np.mean([v for v in col if v is not None]) for col in o["xgen"]["task_exec"]]
            ins = [np.mean([v for v in col if v is not None]) for col in o["xgen"]["task_instr"]]
            e.append(float(np.mean(ex[3:])))
            i_.append(float(np.mean(ins[3:])))
            t40.append(float(np.mean(o["acc"]["task40"])))
        out[rep] = {"exec": e, "instr": i_, "task40": t40}
    return out


def fig_dec() -> None:
    p = plt()
    d = pickle.load(open(CACHE / "dec_parts.pkl", "rb"))
    T = t030_dec()
    fig, ax = p.subplots(1, 4, figsize=(18, 3.6))
    for i, (rep, nm) in enumerate((("sae", "SAE codes"), ("x", "pooled vector (SAE input)"))):
        ax[0].plot(
            LAYERS,
            [np.mean(d[(rep, li)]["acc"]["task40"]) for li in LAYERS],
            color=CAT[i],
            marker="o",
            ms=3,
            label=nm,
        )
        ax[1].plot(
            LAYERS,
            [d[(rep, li)]["acc"]["within_G"][0] for li in LAYERS],
            color=CAT[i],
            marker="o",
            ms=3,
            label=nm,
        )
        ax[2].plot(
            LAYERS, [d[(rep, li)]["phase_bacc"] for li in LAYERS], color=CAT[i], marker="o", ms=3, label=nm
        )
        ax[3].plot(
            LAYERS,
            [np.mean(d[(rep, li)]["xgen_exec"][3:]) for li in LAYERS],
            color=CAT[i],
            marker="o",
            ms=3,
            label=f"{nm}: executed",
        )
        ax[3].plot(
            LAYERS,
            [np.mean(d[(rep, li)]["xgen_instr"][3:]) for li in LAYERS],
            color=CAT[i],
            marker="o",
            ms=3,
            ls="--",
            label=f"{nm}: instructed",
        )
    ax[0].plot(LAYERS, T["v_all6"]["task40"], color=CAT[2], lw=1, label="T030 v_all6 (z)")
    ax[3].plot(LAYERS, T["v_all6"]["exec"], color=CAT[2], lw=1, label="T030 v_all6: executed")
    ax[3].plot(LAYERS, T["v_all6"]["instr"], color=CAT[2], lw=1, ls="--", label="T030 v_all6: instructed")
    ax[0].set(title="task (40-way), mean over calls", xlabel="layer pgNN", ylim=(0.5, 1.02))
    ax[1].set(title="Goal within-scene task (10-way), call 0", xlabel="layer pgNN", ylim=(0, 1.05))
    ax[2].set(title="phase (4-way balanced acc)", xlabel="layer pgNN", ylim=(0.5, 1.0))
    ax[3].set(
        title="fallback trials: share decoded as executed / instructed (calls ≥ 3)",
        xlabel="layer pgNN",
        ylim=(0, 1),
    )
    for a in ax:
        a.legend(frameon=False, fontsize=6)
    _save(fig, "7_decoding")


def stage_figs() -> None:
    for f in (fig_recon, fig_tags, fig_traj, fig_fallback, fig_basket, fig_kauf, fig_dec):
        t = time.time()
        f()
        print(f"  {f.__name__} {time.time() - t:.0f}s", flush=True)


# ---------------------------------------------------------------- numbers
CALL_BINS = ((0, 1), (1, 3), (3, 10), (10, 20), (20, 28))


def _bin_median(recs, key, a, b, src=None, kind=None):
    v = []
    for x in recs:
        if (src and x["src"] != src) or (kind and x.get("kind") != kind):
            continue
        arr = np.array(x[key][a:b], float)
        if np.isfinite(arr).any():
            v.append(np.nanmean(arr))
    if not v:
        return None
    rng = np.random.default_rng(0)
    bs = [np.median(rng.choice(v, len(v))) for _ in range(1000)]
    return {
        "median": float(np.median(v)),
        "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "n": len(v),
    }


def stage_numbers() -> None:
    from scipy.stats import mannwhitneyu, spearmanr

    num: dict = {}
    E = load_json("encode")
    num["A1_reconstruction"] = {
        lname(r["layer"]): {
            k: r[k]
            for k in (
                "nmse",
                "ve",
                "ve_nomassive",
                "never_fire",
                "train_nmse",
                "train_ve",
                "train_never_fire",
                "fire_rho",
                "t028_active_rare_in_train",
                "massive_share_median",
                "massive_share_p90",
            )
        }
        | {"ve_by_set": {k: v["ve"] for k, v in r["per_set"].items()}}
        for r in E["layers"]
    }
    num["A1_reconstruction"]["_check"] = {
        "cache_vs_npz_maxdiff": E["cache_check_maxdiff"],
        "ntok_range": [E["ntok_min"], E["ntok_max"]],
    }
    L = load_json("labels")
    num["A2_labels"] = {
        k: {
            "tags": v["tags"],
            "task_sub": v["sub"],
            "phase_flag": v["phase_flag"],
            "phase_tag_pref": v["pref_phase_of_phase"],
            "FT_size_median": float(np.median([len(x) for x in v["FT"].values()])),
            "object_dest_sets": {a: len(b) for a, b in v["od"].items()},
        }
        for k, v in L.items()
    }
    R = pickle.load(open(CACHE / "fallback.pkl", "rb"))
    A4 = {}
    for r in R:
        e = {}
        for src in ("cand", "l90", None):
            for key in ("idx_I", "idx_E"):
                e[f"{src or 'all'}_{key}"] = {
                    f"calls{a}-{b - 1}": _bin_median(r["recs"], key, a, b, src) for a, b in CALL_BINS
                }
        for key in ("idx_I", "idx_E"):
            e[f"object_{key}"] = {
                f"calls{a}-{b - 1}": _bin_median(r["odrec"], key, a, b, kind="object") for a, b in CALL_BINS
            }
        e["n_trials"] = {
            "cand": sum(x["src"] == "cand" for x in r["recs"]),
            "l90": sum(x["src"] == "l90" for x in r["recs"]),
            "object": sum(x["kind"] == "object" for x in r["odrec"]),
            "dest": sum(x["kind"] == "dest" for x in r["odrec"]),
        }
        e["pairs"] = {
            c: {k: v for k, v in pc.items() if k != "sig_list"} for c, pc in r["pairs"]["calls"].items()
        }
        e["pairs_n"] = r["pairs"]["n_pairs"]
        A4[lname(r["layer"])] = e
    nsig0 = [R[li]["pairs"]["calls"][0]["n_sig"] for li in range(17)]
    rho = spearmanr(range(17), nsig0)
    A4["_H4_spearman_nsig_call0_pg00_16"] = {"rho": float(rho[0]), "p": float(rho[1])}
    num["A4_fallback"] = A4
    B = pickle.load(open(CACHE / "basket.pkl", "rb"))
    num["A5_basket"] = {
        lname(b["layer"]): {
            w: {
                k: b[w][k]
                for k in (
                    "n_tested",
                    "n_sig",
                    "class",
                    "lambda_median",
                    "tags",
                    "near_same_sign_smaller",
                    "opp_same_sign",
                    "dist_by_cond",
                    "nfire_changed",
                )
            }
            for w in ("call0", "carry")
        }
        | {"n_pairs": b["n"]}
        for b in B
    }
    K = pickle.load(open(CACHE / "kauf.pkl", "rb"))
    A6 = {}
    ntest, pvs = 0, []
    for k in K:
        tg = tag_arr(k["layer"])
        sub = np.array([r["sub"] for r in lab_rows(k["layer"])])
        ent = {
            "chance": k["chance"],
            "reader": k["reader"],
            "median_by_tag": k["median_by_tag"],
            "n_by_tag": k["n_by_tag"],
            "median_all": float(np.median(k["pf"])),
        }
        # H6 as registered needs always-on features (none / 1 per layer) -> report the closest contrast: task vs non-task
        a, b_ = k["pf"][tg == "task"], k["pf"][(tg != "task") & (tg != "rare")]
        if len(a) > 5 and len(b_) > 5:
            pv = mannwhitneyu(a, b_).pvalue
            ent["task_vs_nontask_mwu_p"] = float(pv)
            pvs.append(pv)
            ntest += 1
        a, b_ = k["pf"][sub == "instruction"], k["pf"][sub == "task-confounded"]
        ent["instr_vs_confounded_mwu_p"] = float(mannwhitneyu(a, b_).pvalue) if len(a) > 5 else None
        A6[lname(k["layer"])] = ent
    if pvs:
        q = sv.bh(np.array(pvs))
        A6["_task_vs_nontask_BH"] = {"n_tests": ntest, "n_q_lt_0.05": int((q < 0.05).sum())}
    num["A6_kaufman"] = A6
    d = pickle.load(open(CACHE / "dec_parts.pkl", "rb"))
    T = t030_dec()
    num["A7_decoding"] = {
        lname(li): {
            rep: {
                "task40": float(np.mean(d[(rep, li)]["acc"]["task40"])),
                "within": float(np.mean(d[(rep, li)]["acc"]["within"])),
                "within_G_call0": d[(rep, li)]["acc"]["within_G"][0],
                "phase_bacc": d[(rep, li)]["phase_bacc"],
                "xgen_exec_calls3plus": float(np.mean(d[(rep, li)]["xgen_exec"][3:])),
                "xgen_instr_calls3plus": float(np.mean(d[(rep, li)]["xgen_instr"][3:])),
                "xgen_exec_call0": d[(rep, li)]["xgen_exec"][0],
                "xgen_instr_call0": d[(rep, li)]["xgen_instr"][0],
            }
            for rep in ("sae", "x")
        }
        | {
            "T030_v_all6": {
                "task40": T["v_all6"]["task40"][li],
                "xgen_exec_calls3plus": T["v_all6"]["exec"][li],
                "xgen_instr_calls3plus": T["v_all6"]["instr"][li],
            }
        }
        for li in LAYERS
    }
    if (CACHE / "expert.pkl").exists():
        X = pickle.load(open(CACHE / "expert.pkl", "rb"))
        num["A7_expert_link"] = [{k: v for k, v in x.items()} for x in X]
    (OUT / "numbers.json").write_text(json.dumps(num, indent=1, default=sv._js) + "\n")
    print("numbers.json written")
