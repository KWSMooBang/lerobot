#!/usr/bin/env python
"""T033 figures (static PNG). Usage: sae_popdyn_figs.py a|b|c|d|e"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_popdyn as P  # noqa: E402

# reference categorical palette (fixed order), text inks, diverging poles
CAT = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "lines.linewidth": 2})
TAGC = {"instruction": CAT[0], "task-shared-scene": CAT[2], "task-confounded": "#9c9c9c", "phase": CAT[1], "scene": CAT[6],
        "mixed": "#cccccc", "task": CAT[0], "other": "#bbbbbb"}
GROUPS = ["U40 Spatial", "U40 Object", "U40 Goal", "U40 Long", "cand FALLBACK", "cand NEW", "cand other", "L90 lscene", "L90 part",
          "basket near", "basket opp", "basket ctl4090", "basket far20", "basket far25"]


def save(fig, name):
    P.FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(P.FIG / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_a():
    sel = P.load_json("a_select")
    blocks = [b for b in P.VLM_SUB + P.EXP_SUB + ["whole_sub"] if b in sel]
    fig, ax = plt.subplots(2, len(blocks), figsize=(2.1 * len(blocks), 5), sharey="row")
    for j, b in enumerate(blocks):
        s = sel[b]
        Rs = sorted(int(r) for r in s["ranks"])
        for R in Rs:
            e = s["ranks"][str(R)]["err"]
            ax[0, j].plot([R] * len(e), e, "o", ms=3, color=CAT[0], alpha=0.6)
            sm = s["ranks"][str(R)]["sim"]
            ax[1, j].plot([R] * len(sm), sm, "o", ms=3, color=CAT[1], alpha=0.6)
        ax[1, j].plot(Rs, [s["ranks"][str(R)]["sim_median"] for R in Rs], color=CAT[1])
        ax[1, j].axhline(0.75, color=INK2, lw=1, ls="--")
        for a_ in ax[:, j]:
            a_.axvline(s["R_star"], color=INK, lw=1, ls=":")
        ax[0, j].set_title(f"{P.pretty(b)}  R*={s['R_star']}", fontsize=9)
        ax[1, j].set_xlabel("rank")
    ax[0, 0].set_ylabel("reconstruction error\n‖X−X̂‖/‖X‖")
    ax[1, 0].set_ylabel("stability (similarity\nto best fit)")
    save(fig, "a1_rank.png")

    res = {r["block"]: r for r in pickle.load(open(P.CACHE / "a_res.pkl", "rb"))}
    M = P.meta()
    cond = np.array([P.cond_label(m) for m in M])
    gc = np.array([m["grasp_call"] if m["grasp_call"] is not None else np.nan for m in M], float)
    rc = np.array([m["release_call"] if m["release_call"] is not None else np.nan for m in M], float)
    for b, r in res.items():
        R = r["R_star"]
        fit = P.load_fits(b, R)
        fit = fit[int(np.argmin([f["err"] for f in fit]))]
        inf = P.info(b)
        fig, ax = plt.subplots(R, 3, figsize=(13, 1.25 * R + 0.6), gridspec_kw={"width_ratios": [1.3, 1, 1.6]})
        ax = np.atleast_2d(ax)
        for k in range(R):
            w = fit["B"][:, k]
            cols = [TAGC.get(t, "#bbbbbb") for t in inf["tag"]]
            ax[k, 0].bar(np.arange(len(w)), w, color=cols, width=1.0)
            ax[k, 0].set_xlim(0, len(w))
            ax[k, 0].set_yticks([])
            c = r["comps"][k]
            ax[k, 0].text(0.99, 0.95, f"#{k} {c['tag_mode']} ({c['tag_purity']:.0%})  task {c['task_mode']} ({c['task_purity']:.0%})",
                          transform=ax[k, 0].transAxes, ha="right", va="top", fontsize=7, color=INK)
            ax[k, 1].plot(np.arange(P.NC), fit["C"][:, k], color=CAT[0])
            ax[k, 1].axvline(np.nanmedian(gc), color=CAT[1], lw=1, ls="--")
            ax[k, 1].axvline(np.nanmedian(rc), color=CAT[2], lw=1, ls="--")
            ax[k, 1].set_yticks([])
            a = r["trial_loadings"][:, k]
            for gi, g in enumerate(GROUPS):
                v = a[cond == g]
                if len(v):
                    ax[k, 2].plot(gi + (np.random.default_rng(gi).random(len(v)) - 0.5) * 0.6, v, ".", ms=2, color=CAT[gi % 8], alpha=0.5)
                    ax[k, 2].plot([gi - 0.35, gi + 0.35], [np.median(v)] * 2, color=INK, lw=1.5)
            ax[k, 2].set_xticks([])
            ax[k, 2].set_yticks([])
        for k in range(R - 1):
            ax[k, 0].set_xticklabels([])
            ax[k, 1].set_xticklabels([])
        ax[-1, 0].set_xlabel("feature (colour = training tag)")
        ax[-1, 1].set_xlabel("call (orange = median grasp, aqua = median release)")
        ax[-1, 2].set_xticks(range(len(GROUPS)))
        ax[-1, 2].set_xticklabels(GROUPS, rotation=60, ha="right", fontsize=7)
        ax[0, 0].set_title("feature loadings")
        ax[0, 1].set_title("time profile")
        ax[0, 2].set_title("trial loadings (bar = median)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=TAGC[t]) for t in (P.TAGS_V if b[0] != "e" else P.TAGS_E)]
        H = 1.25 * R + 0.6
        fig.subplots_adjust(top=1 - 0.75 / H, hspace=0.35)
        fig.legend(handles, list(P.TAGS_V if b[0] != "e" else P.TAGS_E), loc="upper center", ncol=6, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.3, 1 - 0.2 / H))
        fig.suptitle(f"TCA {P.pretty(b)}, rank {R}", y=1 - 0.02 / H, x=0.8)
        save(fig, f"a2_components_{P.pretty(b)}.png")

    fig_a3()


def fig_a3():
    A = P.load_json("a")
    blocks = [b for b in P.VLM_SUB + P.EXP_SUB + ["whole_sub"] if b in A]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    x = np.arange(len(blocks))
    for k, (key, col, lab) in enumerate([("frac_closer_E_cand", CAT[0], "cand"), ("frac_closer_E_l90", CAT[1], "L90 lscene"),
                                         ("frac_closer_E", INK2, "all")]):
        ax[0].bar(x + (k - 1) * 0.27, [A[b]["fallback"][key] for b in blocks], 0.25, color=col, label=lab)
    ax[0].axhline(0.5, color=INK, lw=1, ls="--")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([P.pretty(b) for b in blocks], rotation=45)
    ax[0].set_ylabel("share of fallback trials whose\nloadings are closer to EXECUTED task")
    ax[0].set_ylim(0, 1.18)
    ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax[1].bar(x - 0.2, [A[b]["fallback"]["nn5_E_mean"] for b in blocks], 0.38, color=CAT[0], label="executed task")
    ax[1].bar(x + 0.2, [A[b]["fallback"]["nn5_I_mean"] for b in blocks], 0.38, color=CAT[1], label="instructed task")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([P.pretty(b) for b in blocks], rotation=45)
    ax[1].set_ylabel("of 5 nearest U40 trials,\nhow many are of …")
    ax[1].legend(frameon=False, fontsize=8)
    save(fig, "a3_fallback.png")


def _shade(ax, blocks):
    for j, b in enumerate(blocks):
        if P.flag(b) not in ("main", "whole"):
            ax.axvspan(j - 0.5, j + 0.5, color="#f3f3f3", zorder=0, lw=0)


def fig_b():
    B = P.load_json("b")
    FC = {"time": CAT[2], "I": CAT[1], "E": CAT[0], "scene": CAT[6]}
    FL = {"time": "time (call)", "I": "instructed task", "E": "executed task", "scene": "scene"}
    for key, ttl, fn in (("unique_fb", "unique R² on FALLBACK rows", "b1_unique_fallback.png"),
                         ("unique", "unique R² on all rows (U40 + fallback)", "b1_unique_all.png")):
        fig, ax = plt.subplots(3, 2, figsize=(13, 8.5), sharey="row", gridspec_kw={"width_ratios": [1, 1]})
        for i, w in enumerate(P.B_WINDOWS):
            for j, bl in enumerate((P.VLM_ALL, P.EXP_ALL)):
                a = ax[i, j]
                _shade(a, bl)
                for f in P.B_FACTORS:
                    a.plot(range(len(bl)), [B[b][f"I/{w}"][key][f] for b in bl], "o-", ms=4, color=FC[f], label=FL[f])
                full = "R2_full_fb" if key == "unique_fb" else "R2_full"
                a2v = [B[b][f"I/{w}"][full] for b in bl]
                a.plot(range(len(bl)), a2v, "--", lw=1.2, color=INK2, label="full model R²")
                a.set_xticks(range(len(bl)))
                a.set_xticklabels([P.pretty(b) for b in bl], rotation=60, fontsize=7)
                a.axhline(0, color=INK2, lw=0.8)
                a.set_title(f"{'VLM' if j == 0 else 'action expert'} — calls {w}", fontsize=9)
            ax[i, 0].set_ylabel("cross-validated R²")
        ax[0, 0].legend(frameon=False, fontsize=8, ncol=3)
        fig.suptitle(ttl + " (grey band = reference / caution / massive-channel layers)")
        fig.tight_layout()
        save(fig, fn)
    # dPCA marginalisation shares
    bl = P.VLM_ALL + P.EXP_ALL + ["vlm_all", "exp_all", "whole"]
    keys = [("time", CAT[2], "time"), ("condition", CAT[1], "condition (base / FALLBACK / NEW, incl. ×time)"),
            ("variant", CAT[6], "variant (nuisance)"), ("cond_x_variant", "#bbbbbb", "condition × variant")]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    bottom = np.zeros(len(bl))
    for k, c, lab in keys:
        v = np.array([B[b]["dpca"]["share"][k] for b in bl])
        ax.bar(range(len(bl)), v, bottom=bottom, color=c, label=lab, width=0.8, edgecolor="white", linewidth=1)
        bottom += v
    ax.set_xticks(range(len(bl)))
    ax.set_xticklabels([P.pretty(b) for b in bl], rotation=60, fontsize=7)
    ax.set_ylabel("share of variance")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.set_title(f"dPCA marginalisation, Goal cand variants with both FALLBACK and NEW trials (n = {B['v08']['dpca']['V']})", fontsize=9)
    save(fig, "b2_dpca_shares.png")


def fig_c():
    Cr = P.load_json("c")
    fig, ax = plt.subplots(2, 2, figsize=(13, 7), sharey="row")
    for j, bl in enumerate((P.VLM_ALL, P.EXP_ALL)):
        a = ax[0, j]
        _shade(a, bl)
        x = np.arange(len(bl))
        a.plot(x, [Cr[b]["r2_median"] for b in bl], "o-", ms=4, color=CAT[0], label="source tasks (base, object, destination)")
        a.plot(x, [Cr[b]["r2_rand_median"] for b in bl], "o-", ms=4, color=INK2, label="random non-source Goal tasks")
        for k, b in enumerate(bl):
            if Cr[b].get("q_source_gt_random", 1) < 0.05:
                a.plot(k, Cr[b]["r2_median"] + 0.04, "*", color=INK, ms=6)
        a.axhline(0.48, color=CAT[1], lw=1, ls="--", label="T029 raw vectors (source R² 0.48)")
        a.set_xticks(x)
        a.set_xticklabels([P.pretty(b) for b in bl], rotation=60, fontsize=7)
        a.set_title(("VLM" if j == 0 else "action expert") + ": NNLS fit R² (median over cand trials; * = BH q < 0.05)", fontsize=9)
        a = ax[1, j]
        _shade(a, bl)
        a.plot(x, [Cr[b]["slope_NEW_median"] for b in bl], "o-", ms=4, color=CAT[0], label="NEW")
        a.plot(x, [Cr[b]["slope_FB_median"] for b in bl], "o-", ms=4, color=CAT[1], label="FALLBACK")
        for k, b in enumerate(bl):
            if Cr[b]["p_slope_NEW_vs_FB"] < 0.05:
                a.plot(k, max(Cr[b]["slope_NEW_median"], Cr[b]["slope_FB_median"]) + 0.05, "*", color=INK, ms=6)
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xticks(x)
        a.set_xticklabels([P.pretty(b) for b in bl], rotation=60, fontsize=7)
        a.set_title("slope of (w_destination − w_object) over the episode (median; * = uncorrected p < 0.05)", fontsize=9)
    ax[0, 0].set_ylabel("R²")
    ax[1, 0].set_ylabel("slope per episode")
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[1, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "c1_r2_slope.png")
    show = ["v04", "v08", "v12", "e08", "e11", "e17", "whole"]
    fig, ax = plt.subplots(1, len(show), figsize=(2.6 * len(show), 3.2), sharey=True)
    for a, b in zip(ax, show):
        t = Cr[b]["traj"]
        for cls, ls in (("NEW", "-"), ("FALLBACK", "--")):
            if f"{cls}/w_obj" in t:
                a.plot(t[f"{cls}/w_obj"], ls, color=CAT[1], label=f"object source, {cls}")
                a.plot(t[f"{cls}/w_dest"], ls, color=CAT[0], label=f"destination source, {cls}")
        a.set_title(P.pretty(b), fontsize=9)
        a.set_xlabel("call")
    ax[0].set_ylabel("NNLS weight (mean over trials)")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "c2_weights.png")


def fig_d():
    D = P.load_json("d")
    keys = list(D)
    kind = {k: ("whole" if "vlm_all" in k else "ref" if (k.startswith("pg") and "->pg" in k) or k.startswith("L") else "vlm->exp") for k in keys}
    KC = {"vlm->exp": CAT[0], "ref": CAT[1], "whole": CAT[6]}
    n = len(keys)
    cols = 7
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.2 * rows), sharex=True)
    for a, k in zip(ax.flat, keys):
        r = D[k]
        y = np.array(r["R2_rrr"])
        se = np.array(r["R2_rrr_se"])
        x = np.arange(1, len(y) + 1)
        a.fill_between(x, y - se, y + se, color=KC[kind[k]], alpha=0.2, lw=0)
        a.plot(x, y, color=KC[kind[k]])
        a.axhline(r["R2_full"], color=INK2, lw=1, ls="--")
        a.axvline(r["comm_dim"], color=INK, lw=1, ls=":")
        a.set_title(f"{k}  dim {r['comm_dim']}", fontsize=8)
        a.set_ylim(0, max(0.05, r["R2_full"] * 1.15))
    for a in ax.flat[n:]:
        a.axis("off")
    for a in ax[-1]:
        a.set_xlabel("rank")
    fig.suptitle("reduced-rank regression: CV R² vs rank (band = ±1 SE over 10 init folds; dashed = full ridge; dotted = communication dim)\n"
                 "blue = VLM → action expert, orange = within-model reference, violet = whole network", fontsize=9)
    fig.tight_layout()
    save(fig, "d1_rrr_curves.png")
    # tags and pairs
    vk = [k for k in keys if kind[k] != "ref" or k.startswith("pg")]
    vk = [k for k in vk if k.startswith("pg") or "vlm_all" in k]
    tags = [t for t in P.TAGS_V if any(t in D[k].get("lev_by_tag", {}) for k in vk)]
    M = np.array([[D[k]["lev_by_tag"].get(t, {}).get("ratio", np.nan) for t in tags] for k in vk])
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.32 * len(vk) + 1.8), gridspec_kw={"width_ratios": [1, 1.3]})
    im = ax[0].imshow(np.log2(M), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax[0].set_xticks(range(len(tags)))
    ax[0].set_xticklabels(tags, rotation=30, ha="right")
    ax[0].set_yticks(range(len(vk)))
    ax[0].set_yticklabels(vk, fontsize=8)
    ax[0].grid(False)
    for i, k in enumerate(vk):
        for j, t in enumerate(tags):
            d = D[k]["lev_by_tag"].get(t)
            if d and min(d["p_hi"], d["p_lo"]) < 0.01:
                ax[0].text(j, i, "*", ha="center", va="center", color=INK, fontsize=9)
    fig.colorbar(im, ax=ax[0], label="log2 (mean loading of tag / mean of all)")
    ax[0].set_title("which VLM features load on the communication subspace (* = perm p < 0.01)", fontsize=9)
    y = np.arange(len(vk))
    ax[1].barh(y - 0.27, [D[k]["pairs"]["frac_comm"] for k in vk], 0.26, color=CAT[0], label="communication subspace")
    ax[1].barh(y, [D[k]["pairs"]["frac_topPC"] for k in vk], 0.26, color=CAT[2], label="top-PC subspace (same dim)")
    ax[1].barh(y + 0.27, [D[k]["pairs"]["frac_rand_p95"] for k in vk], 0.26, color="#bbbbbb", label="random subspaces (95th pct)")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(vk, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("share of the instruction-only call-0 difference (same-observation pairs) inside the subspace")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "d2_tags_pairs.png")


def fig_e():
    E = pickle.load(open(P.CACHE / "e_res.pkl", "rb"))
    bl = P.VLM_ALL + P.EXP_ALL
    # 1. fallback: median d_I - d_E per call, layers x calls
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 1]})
    Mx = np.array([E[b]["fallback"]["median_dI_minus_dE"] for b in bl + ["vlm_all", "exp_all", "whole"]])
    lim = np.nanmax(np.abs(Mx))
    im = ax[0].imshow(Mx, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    names = [P.pretty(b) for b in bl + ["vlm_all", "exp_all", "whole"]]
    for a in ax:
        a.set_yticks(range(len(names)))
        a.set_yticklabels(names, fontsize=7)
        a.set_xlabel("call")
        a.grid(False)
    fig.colorbar(im, ax=ax[0], label="median [d(instructed) − d(executed)]  (red = closer to executed)")
    ax[0].set_title("fallback trials: cosine distance to U40 mean trajectory", fontsize=9)
    F = np.array([E[b]["fallback"]["frac_closer_E"] for b in bl + ["vlm_all", "exp_all", "whole"]])
    im = ax[1].imshow(F, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax[1], label="share of fallback trials closer to executed task")
    ax[1].set_title("share closer to EXECUTED task (0.5 = undecided)", fontsize=9)
    fig.tight_layout()
    save(fig, "e1_fallback_distance.png")
    # 2. cand vs base divergence call
    vs = sorted(E["v08"]["cand_vs_base"])
    Dv = np.array([[E[b]["cand_vs_base"][v]["all"]["div_call"] if E[b]["cand_vs_base"][v]["all"]["div_call"] is not None else np.nan
                    for b in bl] for v in vs], float)
    fig, ax = plt.subplots(figsize=(13, 0.24 * len(vs) + 1.8))
    im = ax.imshow(Dv, cmap="viridis_r", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(len(bl)))
    ax.set_xticklabels([P.pretty(b) for b in bl], rotation=60, fontsize=7)
    ylab = []
    for v in vs:
        c = E["v08"]["cand_vs_base"][v]["cls"]
        ylab.append(f"{v} vs {E['v08']['cand_vs_base'][v]['base']}  (FB {c.get('FALLBACK', 0)}, NEW {c.get('NEW', 0)})")
    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels(ylab, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="divergence call (white = never diverges in calls 0–27)")
    ax.set_title("first call where cand trials (same start, different sentence) differ from their base task beyond the split-half floor (perm p < 0.05)", fontsize=9)
    save(fig, "e2_cand_divergence.png")
    # 3. speed
    show = ["v01", "v08", "v15", "e05", "e11", "e17", "whole"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for k, b in enumerate(show):
        for a, nm in zip(ax, ("grasp", "release")):
            sdict = E[b]["speed"][nm]
            lags = sorted(sdict)
            v = np.array([sdict[l_] for l_ in lags])
            a.plot(lags, v / v.max(), color=CAT[k], label=P.pretty(b))
    for a, nm in zip(ax, ("grasp", "release")):
        a.axvline(0, color=INK2, lw=1, ls="--")
        a.set_xlabel(f"call − {nm} call")
    ax[0].set_ylabel("speed |x(c+1) − x(c)| (÷ max)")
    ax[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "e3_speed.png")
    # 4. basket geometry
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    G = [E[b]["basket_geom"]["mean0-27"] for b in bl]
    x = np.arange(len(bl))
    for k, (nm, c) in enumerate((("near", CAT[2]), ("far20", CAT[0]), ("far25", CAT[6]), ("opp", CAT[1]), ("machine", "#bbbbbb"))):
        ax[0].plot(x, [g["norm"][nm] for g in G], "o-", ms=3, color=c, label=nm if nm != "machine" else "ctl4090 − ctl A100 (machine)")
    ax[0].plot(x, [g["floor"]["ctl4090"] for g in G], ":", color=INK2, label="noise floor (split-half)")
    ax[1].plot(x, [g["path_ratio"] for g in G], "o-", ms=3, color=CAT[0], label="path ratio near→far20→far25")
    ax[1].axhline(1, color=INK2, lw=1)
    ax[2].plot(x, [g["opp_dev"] for g in G], "o-", ms=3, color=CAT[1], label="opp distance from 0→far25 line ÷ |far25|")
    ax[2].plot(x, [g["cos"]["opp,far25"] for g in G], "o-", ms=3, color=CAT[6], label="cos(opp, far25)")
    ax[2].plot(x, [g["cos"]["near,far25"] for g in G], "o-", ms=3, color=CAT[2], label="cos(near, far25)")
    for a in ax:
        _shade(a, bl)
        a.set_xticks(x)
        a.set_xticklabels([P.pretty(b) for b in bl], rotation=60, fontsize=6)
        a.legend(frameon=False, fontsize=7)
    ax[0].set_ylabel("|Δ| (episode-mean vector, calls 0–27)")
    save(fig, "e4_basket_path.png")
    # 5. participation ratio
    fig, ax = plt.subplots(figsize=(12, 3.4))
    for k, su in enumerate("SOGL"):
        ax.plot(x, [E[b]["PR"][su] for b in bl], "o-", ms=3, color=CAT[k], label={"S": "Spatial", "O": "Object", "G": "Goal", "L": "Long"}[su])
    _shade(ax, bl)
    ax.set_xticks(x)
    ax.set_xticklabels([P.pretty(b) for b in bl], rotation=60, fontsize=7)
    ax.set_ylabel("participation ratio")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    save(fig, "e5_participation_ratio.png")


if __name__ == "__main__":
    {"a": fig_a, "a3": fig_a3, "b": fig_b, "c": fig_c, "d": fig_d, "e": fig_e}[sys.argv[1]]()
