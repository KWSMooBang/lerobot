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


if __name__ == "__main__":
    {"a": fig_a, "a3": fig_a3}[sys.argv[1]]()
