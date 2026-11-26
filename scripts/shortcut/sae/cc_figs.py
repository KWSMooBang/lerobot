#!/usr/bin/env python
"""T034 figures + numbers.json (CPU).  python cc_figs.py <expert run> <vlm run> [<joint run>]

Out: /outputs/VLA-shortcut-exps/sae/crosscoder/fig/*.png, numbers.json.
Colours: reference palette slots (blue #257de0, orange #f47230, aqua #00b894); magnitudes in one hue (Blues).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path("/outputs/VLA-shortcut-exps/sae/crosscoder")
AN = ROOT / "analysis"
FIG = ROOT / "fig"
W = Path("/data0/woosung/datasets/VLA-shortcut-exps/sae/crosscoder")
T031Q = Path("/outputs/VLA-shortcut-exps/sae/action_expert/weights/quality.json")
T015 = Path("/data0/woosung/datasets/VLA-shortcut-exps/sae/vlm")
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID = "#191919", "#666666", "#e5e5e5"
CATS = ("shallow-only", "deep-only", "middle-only", "carried", "born-middle", "shallow-to-middle", "other")
CAT_KO = {"shallow-only": "shallow-only", "deep-only": "deep-only", "middle-only": "middle-only", "carried": "carried (all layers)",
          "born-middle": "born in middle", "shallow-to-middle": "shallow-to-middle", "other": "other"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"):
    from matplotlib import font_manager

    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break


def sae_ref() -> dict[str, float]:
    ref = {}
    for r in json.loads(T031Q.read_text()):
        if r["job"].endswith("_s9") and not r["job"].endswith("tok"):
            ref[f"ae{int(r['job'][1:3]):02d}"] = r["ve"]
    for i in range(18):
        ref[f"pg{i:02d}"] = json.loads((T015 / f"pg{i:02d}" / "metrics.json").read_text())["var_explained"]
    return ref


def run_metrics(name: str) -> dict:
    return json.loads((W / name / "metrics.json").read_text())


def fig_quality(runs: dict[str, str], ref: dict) -> dict:
    panels = []
    for kind, name in runs.items():
        m = run_metrics(name)
        per = m["heldout"]["per_layer"]
        if kind == "joint":
            panels.append(("joint: VLM layers", name, [p for p in per if p["layer"].startswith("pg")]))
            panels.append(("joint: expert layers", name, [p for p in per if p["layer"].startswith("ae")]))
        else:
            panels.append(({"expert": "expert crosscoder", "vlm": "VLM crosscoder"}[kind], name, per))
    fig, axs = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.0), sharey=True)
    axs = np.atleast_1d(axs)
    out = {}
    for ax, (title, name, per) in zip(axs, panels, strict=True):
        x = np.arange(len(per))
        cc = [p["var_explained"] for p in per]
        sr = [ref.get(p["layer"], np.nan) for p in per]
        ax.plot(x, sr, color=ORANGE, marker="o", ms=4, label="per-layer SAE (T015 / T031)")
        ax.plot(x, cc, color=BLUE, marker="o", ms=4, label="crosscoder (held-out)")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks(x[::3], [p["layer"][2:] for p in per][::3])
        ax.set_xlabel("layer")
        out[title] = {p["layer"]: {"cc": c, "sae": s} for p, c, s in zip(per, cc, sr, strict=True)}
    axs[0].set_ylabel("variance explained")
    axs[0].legend(frameon=False, loc="lower left")
    lo = min(min(v["cc"] for v in d.values()) for d in out.values())
    axs[0].set_ylim(max(0.0, lo - 0.05), 1.005)
    fig.savefig(FIG / "quality_per_layer.png")
    plt.close(fig)
    return out


def fig_profiles(runs: dict[str, str]) -> None:
    fig, axs = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 4.2), layout="constrained")
    axs = np.atleast_1d(axs)
    for ax, (kind, name) in zip(axs, runs.items(), strict=True):
        f = pd.read_parquet(AN / name / "features.parquet")
        p = np.load(W / name / "profile.npy")[f["live"].to_numpy()]
        c = f.loc[f.live, "centroid"].to_numpy()
        P = p[np.argsort(c)]
        layers = run_metrics(name)["layers"]
        im = ax.imshow(P, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=np.quantile(P, 0.99))
        ax.set_xticks(range(0, len(layers), 3 if len(layers) <= 18 else 6),
                      [layers[i] for i in range(0, len(layers), 3 if len(layers) <= 18 else 6)], rotation=90, fontsize=7)
        if kind == "joint":
            ax.axvline(17.5, color=ORANGE, lw=1.5)
        ax.set_ylabel(f"{len(P):,} live features (by depth centroid)", fontsize=8)
        ax.set_title({"expert": "expert", "vlm": "VLM", "joint": "joint (left VLM | right expert)"}[kind], fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="layer share")
    fig.savefig(FIG / "layer_profiles.png")
    plt.close(fig)


def fig_categories(runs: dict[str, str]) -> None:
    items = []
    for kind, name in runs.items():
        pr = json.loads((AN / name / "profiles.json").read_text())
        if kind == "joint":
            items.append((f"joint shared ({pr['kinds']['shared']:,}): VLM part", pr["categories_shared_vlm_part"]))
            items.append((f"joint shared: expert part", pr["categories_shared_expert_part"]))
        else:
            items.append(({"expert": "expert", "vlm": "VLM"}[kind], pr["categories"]))
    fig, axs = plt.subplots(1, len(items), figsize=(3.2 * len(items), 2.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (title, cnt) in zip(axs, items, strict=True):
        tot = max(sum(cnt.values()), 1)
        v = [cnt.get(c, 0) / tot for c in CATS]
        y = np.arange(len(CATS))
        ax.barh(y, v, color=BLUE, height=0.6)
        for yy, vv, c in zip(y, v, CATS, strict=True):
            ax.text(vv + 0.01, yy, f"{cnt.get(c, 0):,}", va="center", fontsize=7, color=INK)
        ax.set_yticks(y, [CAT_KO[c] for c in CATS])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("share of live features")
        ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "profile_categories.png")
    plt.close(fig)


def fig_joint_share(name: str) -> None:
    f = pd.read_parquet(AN / name / "features.parquet")
    v = f.loc[f.live, "vlm_share"].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 2.8))
    bins = np.linspace(0, 1, 41)
    ax.hist(v, bins=bins, color=BLUE, edgecolor="white", linewidth=0.8)
    for x in (0.1, 0.9):
        ax.axvline(x, color=INK2, lw=1, ls="--")
    ax.text(0.02, ax.get_ylim()[1] * 0.92, "expert-only", fontsize=8, color=INK)
    ax.text(0.45, ax.get_ylim()[1] * 0.92, "shared", fontsize=8, color=INK)
    ax.text(0.91, ax.get_ylim()[1] * 0.92, "VLM-only", fontsize=8, color=INK)
    ax.set_xlabel("VLM share of the feature (sum of ||d_i^l||^2 over VLM layers)")
    ax.set_ylabel("features")
    fig.savefig(FIG / "joint_vlm_share.png")
    plt.close(fig)


def fig_tag_depth(runs: dict[str, str]) -> None:
    groups = [("instruction", "instr."), ("task-confounded", "task"), ("scene", "scene"),
              ("phase", "phase"), ("mixed", "mixed")]
    cols = []
    for kind, name in runs.items():
        if kind == "joint":
            cols.append(("joint: centroid in VLM part", name, "centroid_vlm"))
            cols.append(("joint: VLM share", name, "vlm_share"))
        else:
            cols.append(({"expert": "expert: depth centroid", "vlm": "VLM: depth centroid"}[kind], name, "centroid"))
    fig, axs = plt.subplots(1, len(cols), figsize=(3.8 * len(cols), 3.2), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, (title, name, col) in zip(axs, cols, strict=True):
        f = pd.read_parquet(AN / name / "features.parquet")
        f = f[f.live]
        data, labs = [], []
        for g, lab in groups:
            d = f[(f.tag == "task") & (f["sub"] == g)] if g in ("instruction", "task-confounded") else f[f.tag == g]
            data.append(d[col].to_numpy())
            labs.append(f"{lab}\n(n={len(d)})")
        ok = [i for i, d in enumerate(data) if len(d)]
        bp = ax.boxplot([data[i] for i in ok], positions=ok, widths=0.55, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor="#cfe2f8", edgecolor=BLUE)
        for k in ("medians",):
            for b in bp[k]:
                b.set(color=BLUE, linewidth=2)
        ax.set_xticks(range(len(groups)), labs, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("0 = first layer, 1 = last layer" if "share" not in title else "0 = expert-only, 1 = VLM-only")
    fig.savefig(FIG / "tag_depth.png")
    plt.close(fig)


def capacity(runs: dict[str, str]) -> dict:
    """Like-for-like size table (lead request): features and active features per sample vs mean VE, crosscoder
    vs the 18 per-layer SAEs it is compared with (T031 expert: n 2 048, k 64; T015 VLM: n 2 048, k 100)."""
    ref = sae_ref()
    out = {}
    for kind, name in runs.items():
        m = run_metrics(name)
        per = m["heldout"]["per_layer"]
        rows = {"crosscoder": {"features": m["n_latents"], "live_features": int(round((1 - m["never_fire_all"]) * m["n_latents"])),
                               "active_per_sample": m["k"], "ve_mean": m["heldout"]["ve_mean"]}}
        for fam, n_sae, k_sae, src in (("ae", 2048, 64, "T031 held-out"), ("pg", 2048, 100, "T015 training data")):
            ls = [p["layer"] for p in per if p["layer"].startswith(fam)]
            if ls:
                rows[f"per_layer_sae_{fam}"] = {"features": n_sae * len(ls), "active_per_sample": k_sae * len(ls),
                                                "ve_mean": float(np.mean([ref[x] for x in ls])), "ve_source": src,
                                                "crosscoder_ve_mean_same_layers": float(np.mean([p["var_explained"] for p in per if p["layer"] in ls]))}
        out[kind] = rows
    return out


TAG_GROUPS = (("instruction", "instruction (same scene)", BLUE), ("task-confounded", "task (scene+instr.)", ORANGE),
              ("phase", "phase", AQUA))


def tag_profiles(name: str) -> dict:
    """Mean layer profile of each tag group divided by the mean profile of all live features (enrichment by layer)."""
    f = pd.read_parquet(AN / name / "features.parquet")
    p = np.load(W / name / "profile.npy")
    live = f["live"].to_numpy()
    base = p[live].mean(0)
    out = {"layers": run_metrics(name)["layers"], "all": base.tolist()}
    for g, _, _ in TAG_GROUPS + (("scene", "", ""),):
        m = live & ((f["tag"] == "task") & (f["sub"] == g) if g in ("instruction", "task-confounded") else (f["tag"] == g)).to_numpy()
        if m.sum():
            out[g] = {"n": int(m.sum()), "mean_profile": p[m].mean(0).tolist(), "enrichment": (p[m].mean(0) / base).tolist()}
    return out


def fig_tag_profiles(runs: dict[str, str]) -> dict:
    fig, axs = plt.subplots(1, len(runs), figsize=(3.8 * len(runs), 3.0), squeeze=False)
    res = {}
    for ax, (kind, name) in zip(axs[0], runs.items(), strict=True):
        tp = tag_profiles(name)
        res[kind] = tp
        x = np.arange(len(tp["layers"]))
        for g, lab, col in TAG_GROUPS:
            if g in tp:
                ax.plot(x, tp[g]["enrichment"], color=col, label=f"{lab} (n={tp[g]['n']})")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        if kind == "joint":
            ax.axvline(17.5, color=INK2, lw=1)
        st = 3 if len(x) <= 18 else 6
        ax.set_xticks(x[::st], [tp["layers"][i] for i in x[::st]], rotation=90, fontsize=7)
        ax.set_title({"expert": "expert", "vlm": "VLM", "joint": "joint"}[kind], fontsize=9)
        ax.set_ylabel("mean layer share / all-feature mean")
        ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "tag_layer_enrichment.png")
    plt.close(fig)
    return res


def fig_fallback(runs: dict[str, str]) -> None:
    fig, axs = plt.subplots(2, len(runs), figsize=(3.8 * len(runs), 6.0), squeeze=False)
    for j, (kind, name) in enumerate(runs.items()):
        fb = json.loads((AN / name / "fallback.json").read_text())
        ax = axs[0, j]
        for key, col, lab in (("on_E", BLUE, "executed-task features"), ("on_I", ORANGE, "instructed-task features")):
            c = [r["call"] for r in fb[key] if r["mean"] is not None]
            v = [r["mean"] for r in fb[key] if r["mean"] is not None]
            ax.plot(c, v, color=col, label=lab)
        ax.set_title({"expert": "expert", "vlm": "VLM", "joint": "joint"}[kind] + f" ({fb['n_trials']} fallback trials)", fontsize=9)
        ax.set_xlabel("call (10 env step)")
        ax.set_ylabel("fraction active (mean)")
        if j == 0:
            ax.legend(frameon=False)
        ax = axs[1, j]
        E = np.array(fb["contrib"]["E_fb"])
        I_ = np.array(fb["contrib"]["I_fb"])
        D = np.stack([E.mean(0), I_.mean(0)])  # (2, L): mean over calls
        layers = fb["layers"]
        x = np.arange(len(layers))
        ax.plot(x, D[0], color=BLUE, label="executed-task features")
        ax.plot(x, D[1], color=ORANGE, label="instructed-task features")
        ax.set_xticks(x[:: 3 if len(x) <= 18 else 6], [layers[i] for i in x[:: 3 if len(x) <= 18 else 6]], rotation=90, fontsize=7)
        if kind == "joint":
            ax.axvline(17.5, color=INK2, lw=1, ls="--")
        ax.set_ylabel("per-layer contribution sum z_i*||d_i^l||")
        ax.set_title("where in depth (mean over calls)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fallback.png")
    plt.close(fig)


def main() -> None:
    names = sys.argv[1:]
    runs = dict(zip(("expert", "vlm", "joint"), names, strict=False))
    FIG.mkdir(parents=True, exist_ok=True)
    ref = sae_ref()
    num: dict = {"runs": runs, "quality": fig_quality(runs, ref)}
    fig_profiles(runs)
    fig_categories(runs)
    if "joint" in runs:
        fig_joint_share(runs["joint"])
    fig_tag_depth(runs)
    fig_fallback(runs)
    num["tag_profiles"] = fig_tag_profiles(runs)
    num["capacity"] = capacity(runs)
    for kind, name in runs.items():
        m = run_metrics(name)
        num[kind] = {"name": name, "n_latents": m["n_latents"], "k": m["k"], "heldout_ve_mean": m["heldout"]["ve_mean"],
                     "heldout_ve_min": m["heldout"]["ve_min"], "never_fire": m["never_fire_all"],
                     "per_layer": m["heldout"]["per_layer"], "train_seconds": m["train_seconds"],
                     "peak_mem_gb": m["peak_mem_gb"]}
        for key in ("profiles", "labels", "fallback", "t028_quality"):
            p = AN / name / f"{key}.json"
            if p.exists():
                d = json.loads(p.read_text())
                if key == "labels":
                    d = {k: v for k, v in d.items() if k != "FT"}
                if key == "fallback":
                    d = {k: v for k, v in d.items() if k != "contrib"}
                num[kind][key] = d
    sweep = {}
    for p in sorted((ROOT / "runs").glob("*.json")):
        m = json.loads(p.read_text())
        sweep[m["name"]] = {"n_latents": m["n_latents"], "k": m["k"], "ve_mean": m["heldout"]["ve_mean"],
                            "ve_min": m["heldout"]["ve_min"], "never_fire": m["never_fire_all"],
                            "n_live": int(round((1 - m["never_fire_all"]) * m["n_latents"])),
                            "train_seconds": m["train_seconds"], "steps_per_s": m["steps_per_s"]}
    num["sweep"] = sweep
    gl = ROOT / "gpu_time.tsv"
    if gl.exists():
        rows = [ln.split("\t") for ln in gl.read_text().splitlines() if ln.strip()]
        num["gpu_h"] = sum(int(r[3]) for r in rows) / 3600
        num["gpu_jobs"] = [{"tag": r[0], "start": r[1], "seconds": int(r[3]), "rc": int(r[4])} for r in rows]
    (ROOT / "numbers.json").write_text(json.dumps(num, indent=1, default=str) + "\n")
    print("ok", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
