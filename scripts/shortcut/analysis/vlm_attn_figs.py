"""T035 basic analysis of the VLM prefix self-attention (CPU, ≤ 8 workers). Run with the matplotlib venv ($PV).

  aggregate  every ep/*.npz → <BIG>/seg_headmean_all.npz (① head mean per call, all episodes) and
             <SMALL>/agg.npz (layer means overall / by set / by group, per-head means, event-aligned curves)
  figs       figures in <SMALL>/figures/ + <SMALL>/headline.json

Quantities (head mean of ①, 6 segments 0 agentview, 1 wrist, 2 BOS, 3 task, 4 state, 5 template):
  img→instr  attention of an image-patch query to all instruction tokens, mean over the 512 patches
             = (A[0,3] + A[1,3]) / 2  (both cameras have 256 patches)
  instr→img  attention of an instruction-token query to all 512 patches, mean over instruction tokens
             = A[3,0] + A[3,1]
  BOS share  attention to BOS averaged over all valid query tokens (token-weighted over the 6 query segments)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SMALL = Path("/outputs/VLA-shortcut-exps/vlm_attn")
BIG = Path("/data0/woosung/datasets/VLA-shortcut-exps/vlm_attn")
SEGS = ["agentview", "wrist", "BOS", "instr", "state", "template"]
NL = 18
WIN = 8  # event-aligned window: calls −WIN … +WIN


def manifest() -> list[dict[str, str]]:
    with open(SMALL / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def group(r: dict[str, str]) -> str:
    """Display group: libero3 by suite, Goal by pool, basket sets, LIBERO-90."""
    if r["set"] == "libero3":
        return {"spatial": "Spatial", "object": "Object", "long": "Long"}[r["condition"]]
    if r["set"] == "goal":
        return {"train": "Goal train", "cand_new": "Goal cand-NEW", "cand_fallback": "Goal cand-FALLBACK",
                "cand_other": "Goal cand-other"}[r["goal_pool"]]  # fmt: skip
    return {"basket_a100": "basket A100", "basket_far": "basket far (4090)", "libero90": "LIBERO-90"}[r["set"]]


GROUPS = ["Spatial", "Object", "Long", "Goal train", "Goal cand-NEW", "Goal cand-FALLBACK", "Goal cand-other",
          "basket A100", "basket far (4090)", "LIBERO-90"]  # fmt: skip
SETS = ["libero3", "goal", "basket_a100", "basket_far", "libero90"]


def derived(A: np.ndarray, ntok: np.ndarray) -> dict[str, np.ndarray]:  # noqa: N803
    """A: (..., 6, 6) head-mean ① → img→instr, instr→img, img→BOS, instr→instr (ntok unused, kept for callers)."""
    return {
        "img2instr": (A[..., 0, 3] + A[..., 1, 3]) / 2,
        "instr2img": A[..., 3, 0] + A[..., 3, 1],
        "img2bos": (A[..., 0, 2] + A[..., 1, 2]) / 2,
        "instr2instr": A[..., 3, 3],
    }


def _load(r: dict[str, str]) -> dict[str, Any]:
    with np.load(BIG / r["path"]) as z:
        sa = z["seg_attn"]  # (C, 18, 8, 6, 6)
        ntok = z["seg_ntok"].astype(np.float64)
    return {"hm": sa.mean(2), "head": sa.mean(0), "ntok": ntok}


def cmd_aggregate(a: argparse.Namespace) -> None:
    rows = manifest()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_load, rows, chunksize=8))
    hm_all = np.concatenate([x["hm"] for x in res]).astype(np.float32)  # (calls, 18, 6, 6)
    ntok_all = np.concatenate([x["ntok"] for x in res]).astype(np.int16)
    ep_idx = np.concatenate([np.full(len(x["hm"]), i, np.int32) for i, x in enumerate(res)])
    call = np.concatenate([np.arange(len(x["hm"]), dtype=np.int16) for x in res])
    np.savez(BIG / "seg_headmean_all.npz", seg_attn_headmean=hm_all, seg_ntok=ntok_all, ep_index=ep_idx, call=call,
             episode=np.asarray([r["path"] for r in rows]))  # fmt: skip
    agg: dict[str, np.ndarray] = {}
    # call-weighted layer means (every call counts once)
    agg["mean/all"] = hm_all.mean(0)
    for s in SETS:
        m = np.isin(ep_idx, [i for i, r in enumerate(rows) if r["set"] == s])
        agg[f"mean/set/{s}"] = hm_all[m].mean(0)
    for g in GROUPS:
        ids = [i for i, r in enumerate(rows) if group(r) == g]
        agg[f"mean/group/{g}"] = hm_all[np.isin(ep_idx, ids)].mean(0)
        agg[f"head/group/{g}"] = np.mean([res[i]["head"] for i in ids], 0)  # (18, 8, 6, 6)
    agg["head/all"] = np.mean([x["head"] for x in res], 0)
    # BOS share token-weighted, per call → mean per layer
    w = ntok_all / ntok_all.sum(-1, keepdims=True)
    bos = (w[:, None, :] * hm_all[..., 2]).sum(-1)  # (calls, 18)
    agg["bos_share/all"] = bos.mean(0)
    for g in GROUPS:
        ids = [i for i, r in enumerate(rows) if group(r) == g]
        agg[f"bos_share/group/{g}"] = bos[np.isin(ep_idx, ids)].mean(0)
    # time courses: per group, by call index (absolute) and aligned to grasp / release
    for g in GROUPS:
        ids = [i for i, r in enumerate(rows) if group(r) == g]
        C = max(len(res[i]["hm"]) for i in ids)  # noqa: N806
        for q in ("img2instr", "instr2img"):
            acc = np.zeros((C, NL))
            n = np.zeros(C)
            for i in ids:
                d = derived(res[i]["hm"], res[i]["ntok"])[q]
                acc[: len(d)] += d
                n[: len(d)] += 1
            agg[f"time/{q}/{g}"] = acc / n[:, None].clip(min=1)
            agg[f"time_n/{g}"] = n
            for ev in ("grasp_call", "release_call"):
                acc = np.zeros((2 * WIN + 1, NL))
                n = np.zeros(2 * WIN + 1)
                for i in ids:
                    if not rows[i][ev]:
                        continue
                    e = int(float(rows[i][ev]))
                    d = derived(res[i]["hm"], res[i]["ntok"])[q]
                    for k in range(-WIN, WIN + 1):
                        if 0 <= e + k < len(d):
                            acc[k + WIN] += d[e + k]
                            n[k + WIN] += 1
                agg[f"ev/{ev}/{q}/{g}"] = acc / n[:, None].clip(min=1)
                agg[f"ev_n/{ev}/{g}"] = n
    np.savez(SMALL / "agg.npz", **agg)
    print(f"aggregate: {len(rows)} episodes, {len(hm_all)} calls → {SMALL / 'agg.npz'}")


# ----------------------------------------------------------------------------------------------- figures


def cmd_figs(a: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fd = SMALL / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    g = dict(np.load(SMALL / "agg.npz"))
    head: dict[str, Any] = {}

    # 1. mean ① per layer, overall and by set
    def heat_grid(A: np.ndarray, title: str, path: Path) -> None:  # noqa: N803
        fig, axs = plt.subplots(3, 6, figsize=(21, 11), gridspec_kw={"wspace": 0.45, "hspace": 0.5})
        for li, ax in enumerate(axs.flat):
            im = ax.imshow(A[li], vmin=0, vmax=1, cmap="viridis")
            for i in range(6):
                for j in range(6):
                    v = A[li, i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if v < 0.55 else "black",
                            fontweight="bold" if (i, j) in ((0, 3), (1, 3), (3, 0), (3, 1)) else "normal")  # fmt: skip
            for i, j in ((0, 3), (1, 3), (3, 0), (3, 1)):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=1.2))
            ax.set_title(f"layer {li}", fontsize=9)
            ax.set_xticks(range(6), SEGS, rotation=60, fontsize=6.5)
            ax.set_yticks(range(6), SEGS, fontsize=6.5)
        fig.colorbar(im, ax=axs, shrink=0.6, label="attention share (row sums to 1)")
        fig.suptitle(title + "\nrow = query segment, column = key segment; head mean; red = image↔instruction", fontsize=11)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)

    heat_grid(g["mean/all"], "VLM self-attention ①, all 1,560 episodes (mean over all calls)", fd / "1_seg_heatmap_all.png")
    for s in SETS:
        heat_grid(g[f"mean/set/{s}"], f"VLM self-attention ①, set {s}", fd / f"1_seg_heatmap_set_{s}.png")

    # 2. layer profiles: img→instr, instr→img, BOS share, by group
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, axs = plt.subplots(1, 4, figsize=(21, 4.6))
    L = np.arange(NL)  # noqa: N806
    for k, gr in enumerate(GROUPS):
        A = g[f"mean/group/{gr}"]  # noqa: N806
        d = derived(A, np.ones(6))
        axs[0].plot(L, d["img2instr"], "-o", ms=3, color=cols[k], label=gr)
        axs[1].plot(L, d["instr2img"], "-o", ms=3, color=cols[k])
        axs[2].plot(L, g[f"bos_share/group/{gr}"], "-o", ms=3, color=cols[k])
        axs[3].plot(L, d["img2bos"], "-o", ms=3, color=cols[k])
    for ax, t in zip(axs, ("image patch → instruction (share)", "instruction → image patches (share)",
                           "BOS share (all valid queries, token-weighted)", "image patch → BOS (share)"), strict=True):  # fmt: skip
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("VLM layer")
        ax.set_xticks(L[::2])
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.suptitle("Layer profile of VLM self-attention (head mean, mean over calls)")
    fig.savefig(fd / "2_layer_profile.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 2b. per head: img→instr and instr→img, layer × head (all episodes)
    H = g["head/all"]  # noqa: N806
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (t, M) in zip(axs, (("image → instruction", (H[..., 0, 3] + H[..., 1, 3]) / 2),
                                ("instruction → image", H[..., 3, 0] + H[..., 3, 1]),
                                ("image → BOS", (H[..., 0, 2] + H[..., 1, 2]) / 2)), strict=True):  # fmt: skip
        im = ax.imshow(M, aspect="auto", cmap="magma")
        ax.set_title(t)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax)
    fig.suptitle("① per head (all episodes, mean over calls)")
    fig.savefig(fd / "2b_per_head.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3. over time: layer × call heatmaps (absolute call) and event-aligned lines
    tgroups = ["Spatial", "Object", "Long", "Goal train", "Goal cand-NEW", "Goal cand-FALLBACK", "basket A100",
               "basket far (4090)", "LIBERO-90"]  # fmt: skip
    for q, qt in (("img2instr", "image → instruction"), ("instr2img", "instruction → image")):
        fig, axs = plt.subplots(3, 3, figsize=(18, 13), gridspec_kw={"hspace": 0.4})
        vmax = max(np.percentile(g[f"time/{q}/{gr}"], 99) for gr in tgroups)
        for ax, gr in zip(axs.flat, tgroups, strict=True):
            M = g[f"time/{q}/{gr}"]  # noqa: N806
            n = g[f"time_n/{gr}"]
            im = ax.imshow(M.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(f"{gr} (n ep = {int(n[0])})", fontsize=10)
            ax.set_xlabel("call (env step = 10·call)")
            ax.set_ylabel("layer")
        fig.colorbar(im, ax=axs, shrink=0.6, label=qt)
        fig.suptitle(f"{qt} attention by layer and call (head mean, mean over episodes)")
        fig.savefig(fd / f"3_time_{q}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        for ev in ("grasp_call", "release_call"):
            show = (0, 4, 8, 12, 16, 17)
            fig, axs = plt.subplots(1, len(show), figsize=(24, 4.2), sharex=True)
            x = np.arange(-WIN, WIN + 1)
            for k, gr in enumerate(tgroups):
                n = g[f"ev_n/{ev}/{gr}"]
                if n.max() < 5:
                    continue
                M = g[f"ev/{ev}/{q}/{gr}"]  # noqa: N806
                ok = n >= 5
                for ax, li in zip(axs, show, strict=True):
                    ax.plot(x[ok], M[ok, li], "-", color=cols[k], label=f"{gr} ({int(n.max())})")
            for ax, li in zip(axs, show, strict=True):
                ax.axvline(0, color="k", lw=0.8, ls="--")
                ax.set_title(f"layer {li}")
                ax.set_xlabel(f"call − {ev.split('_')[0]} call")
                ax.grid(alpha=0.3)
            axs[0].set_ylabel(qt)
            axs[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
            fig.suptitle(f"{qt}, aligned to {ev.split('_')[0]} (T025 events; points with ≥ 5 episodes)")
            fig.savefig(fd / f"3_{q}_aligned_{ev.split('_')[0]}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

    # headline numbers
    A = g["mean/all"]  # noqa: N806
    d = derived(A, np.ones(6))
    head["img2instr_by_layer"] = [round(float(v), 4) for v in d["img2instr"]]
    head["instr2img_by_layer"] = [round(float(v), 4) for v in d["instr2img"]]
    head["bos_share_by_layer"] = [round(float(v), 4) for v in g["bos_share/all"]]
    head["img2bos_by_layer"] = [round(float(v), 4) for v in d["img2bos"]]
    head["img2instr_by_group_layer"] = {gr: [round(float(v), 4) for v in derived(g[f"mean/group/{gr}"], np.ones(6))["img2instr"]]
                                        for gr in GROUPS}  # fmt: skip
    # call-0 vs last-call change, per group, layers 8–14
    head["img2instr_first_vs_last_call_L8_14"] = {
        gr: [round(float(g[f"time/img2instr/{gr}"][0, 8:15].mean()), 4),
             round(float(g[f"time/img2instr/{gr}"][int((g[f"time_n/{gr}"] > 0).sum()) - 1, 8:15].mean()), 4)]
        for gr in GROUPS}  # fmt: skip
    (SMALL / "headline.json").write_text(json.dumps(head, indent=1))
    print(json.dumps(head, indent=1))

    examples_overlay(plt, fd)
    example_full(plt, fd)


def examples_overlay(plt: Any, fd: Path) -> None:
    """② example: object-noun instruction tokens → agentview patches at layers 0 / 8 / 16 over a few calls."""
    rows = [r for r in manifest() if r["goal_pool"] == "train"]
    picks = []
    for nouns in (("cream", "cheese"), ("stove",), ("bowl",)):
        for r in rows:
            if all(n in r["instruction"] for n in nouns) and r["task"] not in [p[0]["task"] for p in picks]:
                if r["grasp_call"]:
                    picks.append((r, nouns))
                    break
    for r, nouns in picks:
        z = np.load(BIG / r["path"])
        obs = np.load(r["obs_path"])
        toks = [str(t) for t in z["instr_tok"]]
        ti = [i for i, t in enumerate(toks) if t.strip("▁") in nouns or any(n in t for n in nouns)]
        C = len(z["t"])  # noqa: N806
        gc = int(float(r["grasp_call"]))
        rc = int(float(r["release_call"])) if r["release_call"] else C - 1
        calls = sorted({0, max(gc - 1, 0), min(rc, C - 1), C - 1})
        layers = (0, 8, 16)
        fig, axs = plt.subplots(len(layers), len(calls), figsize=(3.3 * len(calls), 3.3 * len(layers)))
        for i, li in enumerate(layers):
            for j, c in enumerate(calls):
                ax = axs[i, j]
                m = z["i2p"][c, li][ti].astype(np.float32).sum(0)[:256].reshape(16, 16)  # summed over the noun tokens
                ax.imshow(obs["image"][c])
                h, w = obs["image"].shape[1:3]
                im = ax.imshow(m, extent=(0, w, h, 0), cmap="inferno", alpha=0.55, interpolation="nearest",
                               vmin=0, vmax=max(float(m.max()), 1e-6))  # fmt: skip
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"L{li} call {c}  max {m.max():.3f}\nagentview total {m.sum():.3f}", fontsize=8)
                plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"② {r['task']} '{r['instruction']}' ({r['episode']})\nattention from tokens "
                     f"{[toks[k] for k in ti]} (head mean, summed over these tokens) → agentview patches; "
                     f"grasp call {gc}, release call {rc}", fontsize=9)  # fmt: skip
        fig.savefig(fd / f"4_i2p_overlay_{r['task']}_{'_'.join(nouns)}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def example_full(plt: Any, fd: Path) -> None:
    """③ example: the full head-mean matrix of one call (Goal train episode), every layer, segment boundaries."""
    from matplotlib.colors import LogNorm

    r = next(r for r in manifest() if r["goal_pool"] == "train" and r["grasp_call"])
    z = np.load(BIG / r["full_path"])
    c = int(float(r["grasp_call"]))
    v = z["full_valid"][c]
    seg = z["full_seg"][c][v]
    F = z["full"][c][:, v][:, :, v].astype(np.float32)  # noqa: N806
    bounds = [int(np.nonzero(seg == s)[0][0]) for s in range(6)] + [len(seg)]
    fig, axs = plt.subplots(3, 6, figsize=(24, 12.5))
    for li, ax in enumerate(axs.flat):
        im = ax.imshow(F[li].clip(min=1e-5), norm=LogNorm(1e-5, 1), cmap="magma", interpolation="nearest")
        for b in bounds[1:-1]:
            ax.axhline(b - 0.5, color="cyan", lw=0.4)
            ax.axvline(b - 0.5, color="cyan", lw=0.4)
        ax.set_title(f"layer {li}", fontsize=9)
        mids = [(bounds[s] + bounds[s + 1]) / 2 for s in range(6)]
        ax.set_xticks(mids, ["agent", "wrist", "B", "instr", "st", "tmpl"], fontsize=6, rotation=90)
        ax.set_yticks(mids, ["agent", "wrist", "B", "instr", "st", "tmpl"], fontsize=6)
    fig.colorbar(im, ax=axs, shrink=0.6, label="attention (head mean, log scale)")
    fig.suptitle(f"③ full token × token attention, {r['episode']} ({r['task']}: '{r['instruction']}'), call {c} "
                 f"(grasp); row = query, column = key; {len(seg)} valid tokens", fontsize=11)  # fmt: skip
    fig.savefig(fd / "5_full_matrix_example.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("aggregate")
    p.add_argument("--workers", type=int, default=8)
    sp.add_parser("figs")
    a = ap.parse_args()
    {"aggregate": cmd_aggregate, "figs": cmd_figs}[a.cmd](a)


if __name__ == "__main__":
    main()
