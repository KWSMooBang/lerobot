# ruff: noqa: N803, N806, B023  (matrix names X/S/P/U as in the papers; closures are called within their loop iteration)
"""T023: pi05 (lerobot/pi05_libero_finetuned) on all of LIBERO-PRO — performance and activation trajectories.

Input: one sc-bench run with --record-activations (default runs/bench-pi05lib-pro_all): jobs/pro-*/episodes.jsonl
(one row per episode, `activations` = npz path written by the policy server, serve/activations.py).
Every episode is paired with the `none` episode of the same suite · task · init (same seed).

Features per policy call (T021 / libero3 definitions; z-scored per suite over all calls):
  expert, gemma, siglip, attention, patch_attention, action   (T021 six; scalar summaries per layer / input)
  expert_vec  action-expert hidden state (layer 17, last denoising step, mean over the 50 action tokens), 1024-d
  gemma_vec   VLM hidden state (layer 17; agentview / wrist / task / state token means concatenated), 8192-d
  (the *_vec ones are PCA-reduced per suite before the population analyses)

Stages (all CPU; outputs under --out, default /outputs/VLA-shortcut-exps/analysis/pro_traj):
  load      episodes + features → _cache/ (npz per suite) — reused by the others
  perf      1_performance/: success per suite × perturbation (Wilson 95% CI), Object position by level,
            task-axis shortcut rate (success_orig), failure types
  t021      2_trial_map/, 3_divergence/: trial maps of the six features (PCA-50 / t-SNE), share of episodes
            nearest to their own task's none centroid (and, task axis, to the task the new sentence names),
            divergence from the paired none episode by call
  jpca      4_jpca/: Churchland 2012 — dx/dt = M x on PCA-6 states, R² of skew-symmetric vs unconstrained M,
            rotation plane / frequency, none-fit M applied to each perturbation; action expert also along the
            10 denoising steps
  manifold  5_manifold/: Gallego 2018 — per-condition top-k PCA manifold, principal angles to none with random
            and split-half baselines, CCA-aligned latent correlations vs unaligned and vs other-task pairs
  traj      7_trajectories/: example trajectories (none PCA-2 plane; colour depth = time, call numbers every 5,
            × first grasp, ■ first release) and distance to the paired none episode aligned on the first grasp
  attractor 6_attractor/: Inagaki 2019 — task / position axes: projection of each episode on the axis between two
            reference trajectories (own trained task → task named by the new sentence, or → successful
            perturbed episodes); bimodality (BC, GMM ΔBIC, dip test) over time; endpoint vs outcome
Numbers: <out>/numbers.json (every stage adds its block).

Run: OPENBLAS_CORETYPE=ZEN uv run --with scikit-learn --with diptest python scripts/shortcut/analysis/pro_traj.py \
        [--run DIR] [--out DIR] [--stages load,perf,t021,jpca,manifold,attractor] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = Path("/outputs/VLA-shortcut-exps/runs/bench-pi05lib-pro_all")
OUT = Path("/outputs/VLA-shortcut-exps/analysis/pro_traj")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SHORT = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}
AXES = ["none", "object", "position", "semantic", "task", "environment"]
POS_LEVELS = ["swap"] + [f"{a}0.{k}" for a in "xy" for k in range(1, 6)]
FEATS6 = ["expert", "gemma", "siglip", "attention", "patch_attention", "action"]
VEC = ["expert_vec", "gemma_vec"]
TITLE = {
    "expert": "action expert activations",
    "gemma": "VLM (Gemma) activations",
    "siglip": "SigLIP activations",
    "attention": "attention shares (layer × input)",
    "patch_attention": "image-patch attention",
    "action": "predicted actions",
    "expert_vec": "action-expert hidden state (L17)",
    "gemma_vec": "Gemma hidden state (L17)",
}
HUE = {
    "none": "#52514e",
    "object": "#2a78d6",
    "position": "#eb6834",
    "semantic": "#1baf7a",
    "task": "#e34948",
    "environment": "#6250d6",
}
HUES10 = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
    "#7a5c3e",
    "#3aa6b9",
]
MUTED, INK, INK2, SURFACE, GRID = "#898781", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
K_TRIAL = (
    12  # calls (t = 0..110) per trial vector for trial maps: every episode has ≥ 12 calls or holds its last
)
RNG = np.random.default_rng(0)
NBOOT = 1000
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------- small stats


def r3(x: Any) -> float:
    return round(float(x), 3)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [r3(p), r3(max(0.0, c - h)), r3(min(1.0, c + h))]


def boot_ci(x: np.ndarray, fn=np.mean, n: int = NBOOT) -> list[float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan")] * 3
    b = [fn(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return [r3(fn(x)), r3(np.percentile(b, 2.5)), r3(np.percentile(b, 97.5))]


def clean(ax: Any) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def cond_of(e: dict) -> str:
    """condition label: axis, or position:<level> for libero_object's 11 position levels."""
    return (
        f"position:{e['level']}" if e["axis"] == "position" and e["suite"] == "libero_object" else e["axis"]
    )


# ---------------------------------------------------------------- load


def read_episodes(run: Path) -> list[dict]:
    eps = []
    for f in sorted(run.glob("jobs/pro-*/episodes.jsonl")):
        seen = set()
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["suite"], r["axis"], r["level"], r["task_id"], r["init_idx"], r.get("repeat_idx", 0))
            if key in seen:
                continue
            seen.add(key)
            r["steps_path"] = str(f.parent / r["steps_file"])
            eps.append(r)
    return eps


def release_t(steps_path: str, grasp_t: int | None) -> int | None:
    """first release after the first grasp: first env step (1-based, like first_grasp_t) after it where no
    object is held between both fingers while the gripper is commanded open; None if never."""
    if not grasp_t:
        return None
    d = np.load(steps_path)
    g, a = d["grasped"], d["action"][:, 6]
    for i in range(int(grasp_t), len(g)):
        if g[i] == "" and a[i] < 0:
            return i + 1
    return None


def episode_features(path: str) -> dict[str, np.ndarray]:
    d = np.load(path)
    c = len(d["t"])
    f = {
        "expert": np.log(np.concatenate([d["exp_mlp"].mean(1), d["exp_h"].mean(1)], 1)),
        "gemma": np.log(np.concatenate([d["vlm_mlp"].reshape(c, -1), d["vlm_h"].reshape(c, -1)], 1)),
        "siglip": np.log(d["sig_h"][:, :2].reshape(c, -1)),
        "attention": d["attn_mass"].mean(1).reshape(c, -1),
        "patch_attention": np.sqrt(d["attn_img"][:, :, :512].astype(np.float32).reshape(c, -1)),
        "action": d["act_pred"].reshape(c, -1),
        "expert_vec": d["exp_vec"][:, -1, 2].astype(np.float32),
        "gemma_vec": d["vlm_vec"][:, 2].reshape(c, -1).astype(np.float32),
        "expert_denoise": d["exp_vec"][:, :, 2].astype(np.float16),  # (C, 10, 1024)
        "attn_mass": d["attn_mass"].astype(np.float32),
    }
    return {k: v.astype(np.float16 if k in ("expert_denoise",) else np.float32) for k, v in f.items()}


def stage_load(run: Path, out: Path, workers: int) -> None:
    eps = read_episodes(run)
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"[load] {len(eps)} episodes from {run}", flush=True)
    for suite in SUITES:
        se = [e for e in eps if e["suite"] == suite and e.get("activations")]
        if not se:
            continue
        with ProcessPoolExecutor(workers) as ex:
            feats = list(ex.map(episode_features, [e["activations"] for e in se], chunksize=8))
            rel = list(
                ex.map(
                    release_t,
                    [e["steps_path"] for e in se],
                    [e.get("first_grasp_t") for e in se],
                    chunksize=16,
                )
            )
        for e, r in zip(se, rel, strict=True):
            e["first_release_t"] = r
        arrays: dict[str, Any] = {}
        lens = np.array([len(f["action"]) for f in feats])
        arrays["calls"] = lens
        arrays["offsets"] = np.concatenate([[0], np.cumsum(lens)])
        for k in feats[0]:
            arrays[k] = np.concatenate([f[k] for f in feats])
        meta = [
            {
                k: e.get(k)
                for k in (
                    "suite",
                    "axis",
                    "level",
                    "task_id",
                    "init_idx",
                    "success",
                    "success_orig",
                    "success_new",
                    "n_steps",
                    "first_grasp_object",
                    "first_grasp_t",
                    "first_release_t",
                    "instruction",
                    "original_instruction",
                    "t_orig",
                    "t_new",
                    "activations",
                    "video",
                )
            }
            for e in se
        ]
        np.savez(cache / f"{suite}.npz", **arrays)
        (cache / f"{suite}.json").write_text(json.dumps(meta))
        print(f"[load] {suite}: {len(se)} episodes, {int(lens.sum())} calls", flush=True)
    (cache / "episodes_all.json").write_text(json.dumps(eps))


def present(out: Path) -> list[str]:
    """suites with cached episodes (a partial run may lack some)."""
    return [s for s in SUITES if (out / "_cache" / f"{s}.json").exists()]


class Suite:
    """Cached episodes of one suite: meta list + per-episode feature views (z-scored per feature)."""

    def __init__(self, out: Path, suite: str):
        self.suite = suite
        self.meta = json.loads((out / "_cache" / f"{suite}.json").read_text())
        z = np.load(out / "_cache" / f"{suite}.npz")
        self.off = z["offsets"]
        self.calls = z["calls"]
        self.raw = {k: z[k] for k in z.files if k not in ("offsets", "calls")}
        self.z: dict[str, np.ndarray] = {}
        for f in FEATS6:
            x = self.raw[f]
            mu, sd = x.mean(0), x.std(0)
            keep = sd > 1e-6
            self.z[f] = ((x - mu) / np.where(keep, sd, 1))[:, keep].astype(np.float32)
        self.n = len(self.meta)
        for i, m in enumerate(self.meta):
            m["i"] = i
            m["cond"] = cond_of(m)
        self.none = {(m["task_id"], m["init_idx"]): m["i"] for m in self.meta if m["axis"] == "none"}

    def ep(self, feat: str, i: int, z: bool = True) -> np.ndarray:
        src = self.z if (z and feat in self.z) else self.raw
        return src[feat][self.off[i] : self.off[i + 1]]

    def fixed(self, feat: str, i: int, k: int = K_TRIAL, z: bool = True) -> np.ndarray:
        """first k calls, the last call held if the episode is shorter."""
        x = self.ep(feat, i, z)
        if len(x) >= k:
            return x[:k]
        return np.concatenate([x, np.repeat(x[-1:], k - len(x), 0)])

    def where(self, **kw: Any) -> list[int]:
        return [m["i"] for m in self.meta if all(m.get(k) == v for k, v in kw.items())]

    def conds(self) -> list[str]:
        order = (
            ["none", "object"]
            + [f"position:{lv}" for lv in POS_LEVELS]
            + ["position", "semantic", "task", "environment"]
        )
        have = {m["cond"] for m in self.meta}
        return [c for c in order if c in have]


def pca_fit(x: np.ndarray, k: int, max_rows: int = 20000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mean, components (k, D), explained variance ratio; randomized on a row sample when large."""
    mu = x.mean(0)
    xs = x if len(x) <= max_rows else x[RNG.choice(len(x), max_rows, replace=False)]
    xs = xs - mu
    _u, s, vt = np.linalg.svd(xs, full_matrices=False) if xs.shape[1] <= 4096 else _rsvd(xs, k + 20)
    var: np.ndarray = s**2
    return mu, vt[:k], var[:k] / var.sum()


def _rsvd(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.utils.extmath import randomized_svd

    u, s, vt = randomized_svd(x, k, random_state=0)
    tot = (x**2).sum()
    # keep var ratio honest: append the residual as one pseudo-component
    s_all = np.concatenate([s, [np.sqrt(max(tot - (s**2).sum(), 0))]])
    return u, s_all, vt


# ---------------------------------------------------------------- perf


def target_tokens(instr: str) -> set[str]:
    stop = {
        "the",
        "and",
        "put",
        "pick",
        "place",
        "it",
        "on",
        "in",
        "of",
        "to",
        "up",
        "both",
        "top",
        "inside",
        "into",
    }
    return {w for w in re.findall(r"[a-z]+", instr.lower()) if w not in stop and len(w) > 2}


def failure_type(e: dict) -> str:
    if e["axis"] == "task":
        if e["success_new"]:
            return "new goal done"
        if e["success_orig"]:
            return "original goal done (shortcut)"
        return "neither goal"
    if e["success"]:
        return "success"
    g = e.get("first_grasp_object")
    if not g:
        return "no grasp"
    toks = {t for t in g.lower().split("_") if t.isalpha() and t not in ("akita", "obj")}
    return (
        "grasped a named object, not done"
        if toks & target_tokens(e.get("instruction_original") or e["instruction"])
        else "grasped another object"
    )


def stage_perf(out: Path, res: dict) -> None:
    eps = json.loads((out / "_cache" / "episodes_all.json").read_text())
    od = out / "1_performance"
    od.mkdir(parents=True, exist_ok=True)
    tab: dict = {}
    for s in SUITES:
        for a in AXES:
            se = [e for e in eps if e["suite"] == s and e["axis"] == a]
            if not se:
                continue
            r: dict[str, Any] = {
                "n": len(se),
                "success": wilson(sum(bool(e["success"]) for e in se), len(se)),
            }
            if a == "task":
                r["success_orig"] = wilson(sum(bool(e["success_orig"]) for e in se), len(se))
                r["both"] = wilson(sum(bool(e["success_orig"] and e["success_new"]) for e in se), len(se))
            r["failure_types"] = dict(Counter(failure_type(e) for e in se))
            # paired vs none (same task · init)
            none = {(e["task_id"], e["init_idx"]): e for e in eps if e["suite"] == s and e["axis"] == "none"}
            pairs = [
                (none[(e["task_id"], e["init_idx"])]["success"], e["success"])
                for e in se
                if (e["task_id"], e["init_idx"]) in none
            ]
            r["paired_none_success_to_fail"] = int(sum(bool(x) and not y for x, y in pairs))
            r["paired_none_fail_to_success"] = int(sum((not x) and bool(y) for x, y in pairs))
            if a == "position" and s == "libero_object":
                r["by_level"] = {
                    lv: wilson(
                        sum(bool(e["success"]) for e in se if e["level"] == lv),
                        sum(e["level"] == lv for e in se),
                    )
                    for lv in POS_LEVELS
                }
                r["by_level_n"] = {lv: sum(e["level"] == lv for e in se) for lv in POS_LEVELS}
            r["by_task"] = {
                t: wilson(
                    sum(bool(e["success"]) for e in se if e["task_id"] == t),
                    sum(e["task_id"] == t for e in se),
                )
                for t in range(10)
            }
            tab[f"{s}|{a}"] = r
    res["perf"] = tab
    # table csv
    with open(od / "success_table.csv", "w") as f:
        f.write("suite,perturbation,n,success,ci_lo,ci_hi,success_orig,orig_lo,orig_hi\n")
        for key, r in tab.items():
            s, a = key.split("|")
            so = r.get("success_orig", ["", "", ""])
            f.write(
                f"{s},{a},{r['n']},{r['success'][0]},{r['success'][1]},{r['success'][2]},{so[0]},{so[1]},{so[2]}\n"
            )
    with open(od / "episodes.csv", "w") as f:
        f.write(
            "suite,axis,level,task,init,success,success_orig,success_new,failure_type,n_steps,first_grasp,video\n"
        )
        for e in eps:
            f.write(
                f"{e['suite']},{e['axis']},{e['level']},{e['task_id']},{e['init_idx']},{int(bool(e['success']))},"
                f"{'' if e.get('success_orig') is None else int(bool(e['success_orig']))},"
                f"{'' if e.get('success_new') is None else int(bool(e['success_new']))},{failure_type(e)},"
                f"{e['n_steps']},{e.get('first_grasp_object') or ''},{e.get('video') or ''}\n"
            )
    # figure: grouped bars suite × axis
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [2.3, 1.4, 1.2]}
    )
    ax = axes[0]
    w = 0.13
    for j, a in enumerate(AXES):
        xs, ys, lo, hi = [], [], [], []
        for i, s in enumerate(SUITES):
            rt = tab.get(f"{s}|{a}")
            if rt is None:
                continue
            xs.append(i + (j - 2.5) * w)
            ys.append(rt["success"][0])
            lo.append(rt["success"][0] - rt["success"][1])
            hi.append(rt["success"][2] - rt["success"][0])
        ax.bar(xs, ys, w * 0.92, color=HUE[a], label=a)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, lw=0.7)
    for i, s in enumerate(SUITES):  # task-axis shortcut rate as hatched overlay
        rt = tab.get(f"{s}|task")
        if rt:
            ax.scatter(i + 1.5 * w, rt["success_orig"][0], marker="_", s=180, color=INK, zorder=5)
    ax.set_xticks(range(4), [SHORT[s] for s in SUITES])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate (PRO goal), Wilson 95% CI", fontsize=8)
    ax.set_title(
        "success by suite × perturbation (black tick on task = original goal reached: shortcut rate)",
        fontsize=8.5,
        loc="left",
    )
    ax.legend(fontsize=7, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    clean(ax)
    ax = axes[1]
    rp = tab.get("libero_object|position")
    if rp:
        for prefix, col in (("x", "#eb6834"), ("y", "#2a78d6")):
            lv = [f"{prefix}0.{k}" for k in range(1, 6)]
            m = np.array([rp["by_level"][v] for v in lv], float)
            ax.plot(range(1, 6), m[:, 0], "-o", color=col, label=f"{prefix} shift", ms=4)
            ax.fill_between(range(1, 6), m[:, 1], m[:, 2], color=col, alpha=0.15, lw=0)
        sw = rp["by_level"]["swap"]
        ax.axhline(sw[0], color=MUTED, ls="--", lw=1, label=f"swap {sw[0]:.2f}")
        if "libero_object|none" in tab:
            ax.axhline(tab["libero_object|none"]["success"][0], color=INK, ls=":", lw=1, label="none")
        ax.set_xticks(range(1, 6), [f"0.{k}" for k in range(1, 6)])
        ax.set_xlabel("position level (PRO libero_object_temp_{x,y}<level>)", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title("Object: position level curve", fontsize=8.5, loc="left")
        clean(ax)
    ax = axes[2]
    cats = ["new goal done", "original goal done (shortcut)", "neither goal"]
    cols = ["#1baf7a", "#e34948", "#cfcdc6"]
    bottom = np.zeros(4)
    for c, col in zip(cats, cols, strict=True):
        v = np.array(
            [
                tab[f"{s}|task"]["failure_types"].get(c, 0) / tab[f"{s}|task"]["n"]
                if f"{s}|task" in tab
                else 0
                for s in SUITES
            ]
        )
        ax.bar(range(4), v, bottom=bottom, color=col, label=c, width=0.6)
        bottom += v
    ax.set_xticks(range(4), [SHORT[s] for s in SUITES])
    ax.set_title("task axis: which goal was reached", fontsize=8.5, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1)
    clean(ax)
    fig.tight_layout()
    fig.savefig(od / "success.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    # failure types (non-task axes)
    ftypes = ["success", "no grasp", "grasped another object", "grasped a named object, not done"]
    fcol = ["#cfcdc6", "#6250d6", "#eda100", "#e34948"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), facecolor=SURFACE, sharey=True)
    for ax, s in zip(axes, SUITES, strict=True):
        bottom = np.zeros(5)
        ax_list = [a for a in AXES if a != "task"]
        for c, col in zip(ftypes, fcol, strict=True):
            v = np.array(
                [
                    tab[f"{s}|{a}"]["failure_types"].get(c, 0) / tab[f"{s}|{a}"]["n"]
                    if f"{s}|{a}" in tab
                    else 0
                    for a in ax_list
                ]
            )
            ax.bar(range(5), v, bottom=bottom, color=col, label=c, width=0.65)
            bottom += v
        ax.set_xticks(range(5), ax_list, rotation=30, fontsize=7)
        ax.set_title(SHORT[s], fontsize=9)
        clean(ax)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(
        "outcome types (first grasped object vs objects named in the original sentence; heuristic)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(od / "failure_types.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("[perf] " + "  ".join(f"{k}={v['success'][0]}" for k, v in tab.items()), flush=True)


# ---------------------------------------------------------------- task-axis instruction → trained task


def norm_sentence(s: str) -> list[str]:
    s = (
        s.lower()
        .replace("akita ", "")
        .replace("pick the", "pick up the")
        .replace("on the top of the", "on the")
    )
    s = s.replace("top layer of the drawer", "top drawer").replace(
        "the top of the drawer", "top of the cabinet"
    )
    return re.findall(r"[a-z]+", s)


def named_task(new: str, train: list[str | None], own: int) -> int | None:
    """trained task (same suite) whose training sentence the new sentence names; None if none clearly does."""
    a = set(norm_sentence(new))
    best, bj = None, 0.0
    for t, s in enumerate(train):
        if s is None:
            continue
        b = set(norm_sentence(s))
        j = len(a & b) / len(a | b)
        if j > bj:
            best, bj = t, j
    return best if (best is not None and best != own and bj >= 0.85) else None


# ---------------------------------------------------------------- t021: trial maps + divergence


def stage_t021(out: Path, res: dict) -> None:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    res["t021"] = {}
    od = out / "2_trial_map"
    dd = out / "3_divergence"
    od.mkdir(parents=True, exist_ok=True)
    dd.mkdir(parents=True, exist_ok=True)
    for suite in present(out):
        S = Suite(out, suite)
        train: list[str | None] = [None] * 10
        for m in S.meta:
            train[m["task_id"]] = m["original_instruction"]  # LIBERO training sentence (same on every axis)
        rs: dict = {}
        for f in FEATS6:
            Z = np.stack([S.fixed(f, i).ravel() for i in range(S.n)])
            P = PCA(n_components=min(50, Z.shape[0] - 1), random_state=0).fit_transform(Z)
            task = np.array([m["task_id"] for m in S.meta])
            isnone = np.array([m["axis"] == "none" for m in S.meta])
            cent = np.stack(
                [
                    P[isnone & (task == t)].mean(0)
                    if (isnone & (task == t)).any()
                    else np.full(P.shape[1], np.inf)
                    for t in range(10)
                ]
            )
            own = defaultdict(list)
            named = defaultdict(list)
            for m in S.meta:
                i = m["i"]
                c = cent.copy()
                if m["axis"] == "none":
                    c[m["task_id"]] = P[isnone & (task == m["task_id"]) & (np.arange(S.n) != i)].mean(0)
                dist = np.linalg.norm(c - P[i], axis=1)
                nearest = int(np.argmin(np.where(np.isfinite(dist), dist, np.inf)))
                own[m["cond"]].append(float(nearest == m["task_id"]))
                if m["axis"] == "task":
                    nt = named_task(m["instruction"], train, m["task_id"])
                    if nt is not None:
                        named["task"].append(
                            {
                                "own": nearest == m["task_id"],
                                "named": nearest == nt,
                                "other": nearest not in (nt, m["task_id"]),
                            }
                        )
            rf: dict[str, Any] = {c: boot_ci(np.array(v)) for c, v in own.items()}
            if named["task"]:
                rf["task_axis_nearest"] = {
                    k: r3(np.mean([d[k] for d in named["task"]])) for k in ("own", "named", "other")
                } | {"n": len(named["task"])}
            rs[f] = {"nearest_none_centroid_is_own_task": rf}
            # map
            T = TSNE(
                n_components=2,
                perplexity=min(30, (S.n - 1) // 3),
                init="pca",
                learning_rate="auto",
                random_state=0,
            ).fit_transform(P)
            fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), facecolor=SURFACE)
            mk = {
                "none": "o",
                "object": "s",
                "position": "^",
                "semantic": "D",
                "task": "X",
                "environment": "P",
            }
            for ax, by in zip(axes, ("task", "perturbation"), strict=True):
                for m in S.meta:
                    i = m["i"]
                    col = HUES10[m["task_id"]] if by == "task" else HUE[m["axis"]]
                    ax.scatter(
                        T[i, 0],
                        T[i, 1],
                        s=16 if m["axis"] != "none" else 30,
                        marker=mk[m["axis"]],
                        color=col,
                        edgecolors=SURFACE if m["success"] else INK,
                        linewidths=0.3 if m["success"] else 0.9,
                        alpha=0.85,
                    )
                for t in range(10):
                    if not (isnone & (task == t)).any():
                        continue
                    x, y = np.median(T[isnone & (task == t)], 0)
                    ax.text(
                        x,
                        y,
                        f"t{t}",
                        fontsize=8,
                        ha="center",
                        va="center",
                        bbox={"boxstyle": "round,pad=0.12", "fc": SURFACE, "ec": "none", "alpha": 0.8},
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    f"colour = {by}; label = none median of the task; dark edge = failed", fontsize=8.5
                )
            h = [plt.Line2D([], [], ls="", marker=mk[a], ms=7, color=HUE[a], label=a) for a in AXES]
            fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8, frameon=False)
            fig.suptitle(
                f"{SHORT[suite]} — {TITLE[f]}: one point per episode (first {K_TRIAL} calls, PCA-50 → t-SNE)",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.96))
            (od / f).mkdir(exist_ok=True)
            fig.savefig(od / f / f"trialmap_{suite}.png", dpi=105, facecolor=SURFACE)
            plt.close(fig)
        # divergence by call vs the paired none episode
        div: dict = {}
        conds = [c for c in S.conds() if c != "none"]
        kmax = 30
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), facecolor=SURFACE)
        for ax, f in zip(axes.flat, FEATS6, strict=True):
            base = np.full((len(S.none), kmax), np.nan)
            tasks_inits = sorted(S.none)
            for j, (t, i0) in enumerate(tasks_inits):
                others = [S.none[(t, i1)] for i1 in range(10) if i1 != i0 and (t, i1) in S.none]
                a = S.ep(f, S.none[(t, i0)])
                ds = []
                for o in others:
                    b = S.ep(f, o)
                    n = min(len(a), len(b), kmax)
                    d = np.full(kmax, np.nan)
                    d[:n] = np.sqrt(((a[:n] - b[:n]) ** 2).mean(-1))
                    ds.append(d)
                base[j] = np.nanmean(ds, 0)
            bm, bs = np.nanmean(base, 0), np.nanstd(base, 0)
            ax.plot(bm, color=MUTED, ls="--", lw=1.5, label="none vs none, other init")
            div[f] = {}
            for c in conds:
                idx = [m["i"] for m in S.meta if m["cond"] == c]
                D = np.full((len(idx), kmax), np.nan)
                first = []
                for r, i in enumerate(idx):
                    m = S.meta[i]
                    jn = S.none.get((m["task_id"], m["init_idx"]))
                    if jn is None:
                        continue
                    a, b = S.ep(f, i), S.ep(f, jn)
                    n = min(len(a), len(b), kmax)
                    D[r, :n] = np.sqrt(((a[:n] - b[:n]) ** 2).mean(-1))
                    above = np.where(D[r, :n] > bm[:n] + 2 * bs[:n])[0]
                    first.append(int(above[0]) if len(above) else np.nan)
                m_ = np.nanmean(D, 0)
                if ":" not in c or c in ("position:swap", "position:x0.5", "position:y0.5"):
                    ax.plot(
                        m_,
                        color=HUE[c.split(":")[0]],
                        lw=1.6 if ":" not in c else 1.0,
                        ls="-" if ":" not in c or c.endswith("swap") else ":",
                        label=c,
                    )
                ratio = m_ / bm
                grp_first = next((k for k in range(kmax) if np.isfinite(ratio[k]) and ratio[k] > 1.5), None)
                div[f][c] = {
                    "per_episode_first_call_above_base": boot_ci(np.array(first, float), np.nanmedian),
                    "share_never_above": r3(np.mean(np.isnan(first))) if first else None,
                    "call0_ratio_to_base": r3(ratio[0]),
                    "mean_ratio_calls_0_11": r3(np.nanmean(ratio[:12])),
                    "group_first_call_ratio_gt_1.5": grp_first,
                }
            ax.set_title(TITLE[f], fontsize=9)
            ax.set_xlabel("policy call (×10 env steps)", fontsize=7)
            ax.set_ylabel("RMS distance to paired none (z)", fontsize=7)
            clean(ax)
        axes[0, 0].legend(fontsize=6.5, frameon=False)
        fig.suptitle(
            f"{SHORT[suite]}: distance to the none episode of the same task · init, by call (mean over episodes)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(dd / f"divergence_{suite}.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)
        res["t021"][suite] = {"trial_map": rs, "divergence": div}
        print(
            f"[t021] {suite}: expert own-task "
            + " ".join(
                f"{c}={v[0]}"
                for c, v in rs["expert"]["nearest_none_centroid_is_own_task"].items()
                if isinstance(v, list)
            ),
            flush=True,
        )
    # summary figure: own-task share per feature × axis (suite-pooled means)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), facecolor=SURFACE, sharey=True)
    for ax, suite in zip(axes, SUITES, strict=True):
        r = res["t021"].get(suite)
        if not r:
            continue
        x = np.arange(len(FEATS6))
        for j, axis_name in enumerate(AXES):
            v = [
                r["trial_map"][f]["nearest_none_centroid_is_own_task"].get(axis_name, [np.nan] * 3)
                for f in FEATS6
            ]
            v = np.array(v, float)
            ax.bar(x + (j - 2.5) * 0.13, v[:, 0], 0.12, color=HUE[axis_name], label=axis_name)
        ax.set_xticks(x, FEATS6, rotation=30, fontsize=7)
        ax.set_title(SHORT[suite], fontsize=9)
        clean(ax)
    axes[0].set_ylabel("nearest none task centroid = own task", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.suptitle(
        "do perturbed episodes stay with their own task's none cluster? (first 12 calls, PCA-50)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(od / "task_identity.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- jPCA (Churchland 2012)


def fit_lin(x: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, float]:
    """dx ≈ x M (rows = samples): least squares M and R²."""
    m, *_ = np.linalg.lstsq(x, dx, rcond=None)
    return m, r2(dx, x @ m)


def fit_skew(x: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, float]:
    """dx ≈ x M with M skew-symmetric (M = -Mᵀ): linear LS in the k(k-1)/2 free entries."""
    k = x.shape[1]
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    cols = []
    for i, j in pairs:
        e = np.zeros((k, k))
        e[i, j], e[j, i] = 1, -1
        cols.append((x @ e).ravel())
    a = np.stack(cols, 1)
    w, *_ = np.linalg.lstsq(a, dx.ravel(), rcond=None)
    m = np.zeros((k, k))
    for (i, j), v in zip(pairs, w, strict=True):
        m[i, j], m[j, i] = v, -v
    return m, r2(dx, x @ m)


def r2(y: np.ndarray, yh: np.ndarray) -> float:
    """1 - SSE / Σy² (uncentred: dx = Mx has no intercept; the states are already cross-condition-mean-subtracted)."""
    return float(1 - ((y - yh) ** 2).sum() / max((y**2).sum(), 1e-12))


def rot_plane(m: np.ndarray) -> tuple[np.ndarray, float]:
    """first jPCA plane (2 orthonormal columns in PCA space) and its rotation frequency (rad / step)."""
    ev, evec = np.linalg.eig(m)
    i = int(np.argmax(np.abs(ev.imag)))
    v = evec[:, i]
    p = np.stack([v.real, v.imag], 1)
    q, _ = np.linalg.qr(p)
    return q, float(abs(ev[i].imag))


def plane_angle(p: np.ndarray, q: np.ndarray) -> float:
    s = np.linalg.svd(p.T @ q, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1, 1))))


def jpca_data(
    S: Suite, feat: str, pcs: np.ndarray, mu: np.ndarray, cmean: np.ndarray, idx: list[int]
) -> tuple:
    xs, dxs = [], []
    for i in idx:
        x = S.ep(feat, i, z=False)
        k = min(len(x), len(cmean))
        y = (x[:k] - cmean[:k] - mu) @ pcs.T
        if len(y) < 3:
            continue
        xs.append(0.5 * (y[1:] + y[:-1]))
        dxs.append(y[1:] - y[:-1])
    return (np.concatenate(xs), np.concatenate(dxs)) if xs else (None, None)


def stage_jpca(out: Path, res: dict) -> None:
    od = out / "4_jpca"
    od.mkdir(parents=True, exist_ok=True)
    res["jpca"] = {}
    k = 6
    for suite in present(out):
        S = Suite(out, suite)
        rs: dict = {}
        for feat in VEC:
            X = S.raw[feat]
            kmax = int(S.calls.max())
            # cross-condition (cross-episode) mean at each call, Churchland's preprocessing; soft normalisation per dim
            cm = np.zeros((kmax, X.shape[1]), np.float32)
            cnt = np.zeros(kmax)
            for i in range(S.n):
                x = S.ep(feat, i, z=False)
                cm[: len(x)] += x
                cnt[: len(x)] += 1
            cm /= np.maximum(cnt, 1)[:, None]
            Y = np.concatenate([S.ep(feat, i, z=False) - cm[: S.calls[i]] for i in range(S.n)])
            mu, pcs, evr = pca_fit(Y, k)
            r: dict[str, Any] = {"pca_var_explained": r3(evr.sum())}
            xn, dxn = jpca_data(S, feat, pcs, mu, cm, S.where(axis="none"))
            m_none_skew, _ = fit_skew(xn, dxn)
            p_none, _ = rot_plane(m_none_skew)
            for c in S.conds():
                idx = [m["i"] for m in S.meta if m["cond"] == c]
                x, dx = jpca_data(S, feat, pcs, mu, cm, idx)
                if x is None:
                    continue
                _, r_unc = fit_lin(x, dx)
                m_sk, r_sk = fit_skew(x, dx)
                plane, freq = rot_plane(m_sk)
                # control: calls shuffled within each episode (breaks the temporal order, keeps the state cloud)
                xs_, dxs_ = [], []
                for i in idx:
                    y = (S.ep(feat, i, z=False) - cm[: S.calls[i]] - mu) @ pcs.T
                    y = y[RNG.permutation(len(y))]
                    if len(y) >= 3:
                        xs_.append(0.5 * (y[1:] + y[:-1]))
                        dxs_.append(y[1:] - y[:-1])
                _, r_sk_sh = fit_skew(np.concatenate(xs_), np.concatenate(dxs_))
                r[c] = {
                    "n_eps": len(idx),
                    "r2_unconstrained": r3(r_unc),
                    "r2_skew": r3(r_sk),
                    "skew_over_unc": r3(r_sk / r_unc) if r_unc > 0 else None,
                    "r2_skew_shuffled_calls": r3(r_sk_sh),
                    "rotation_freq_rad_per_call": r3(freq),
                    "plane_angle_to_none_deg": r3(plane_angle(plane, p_none)),
                    "r2_with_none_skew_M": r3(r2(dx, x @ m_none_skew)),
                }
            rs[feat] = r
        # action expert along the 10 denoising steps (within each call)
        E = S.raw["expert_denoise"].astype(np.float32)  # (calls, 10, 1024)
        dm = E.mean(0)  # cross-call mean per denoising step
        mu, pcs, evr = pca_fit((E - dm).reshape(-1, E.shape[-1]), k)
        rd: dict[str, Any] = {"pca_var_explained": r3(evr.sum())}
        for c in S.conds():
            idx = [m["i"] for m in S.meta if m["cond"] == c]
            rows = np.concatenate([np.arange(S.off[i], S.off[i + 1]) for i in idx])
            y = (E[rows] - dm - mu) @ pcs.T  # (n, 10, k)
            x = 0.5 * (y[:, 1:] + y[:, :-1])
            dx = y[:, 1:] - y[:, :-1]
            x, dx = x.reshape(-1, k), dx.reshape(-1, k)
            _, r_unc = fit_lin(x, dx)
            m_sk, r_sk = fit_skew(x, dx)
            rd[c] = {
                "r2_unconstrained": r3(r_unc),
                "r2_skew": r3(r_sk),
                "skew_over_unc": r3(r_sk / r_unc) if r_unc > 0 else None,
                "rotation_freq_rad_per_step": r3(rot_plane(m_sk)[1]),
            }
        rs["expert_denoise"] = rd
        res["jpca"][suite] = rs
        # example figure: none + task + position episodes in the none jPCA plane (gemma and expert)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=SURFACE)
        for ax, feat in zip(axes[:2], VEC, strict=True):
            X = S.raw[feat]
            kmax = int(S.calls.max())
            cm = np.zeros((kmax, X.shape[1]), np.float32)
            cnt = np.zeros(kmax)
            for i in range(S.n):
                x = S.ep(feat, i, z=False)
                cm[: len(x)] += x
                cnt[: len(x)] += 1
            cm /= np.maximum(cnt, 1)[:, None]
            Y = np.concatenate([S.ep(feat, i, z=False) - cm[: S.calls[i]] for i in range(S.n)])
            mu, pcs, _ = pca_fit(Y, k)
            xn, dxn = jpca_data(S, feat, pcs, mu, cm, S.where(axis="none"))
            p_none, _ = rot_plane(fit_skew(xn, dxn)[0])
            for c, ls in (("none", "-"), ("task", "-"), ("position", "-")):
                for t in range(0, 10, 3):
                    for i in [m["i"] for m in S.meta if m["axis"] == c and m["task_id"] == t][:2]:
                        y = (S.ep(feat, i, z=False) - cm[: S.calls[i]] - mu) @ pcs.T @ p_none
                        ax.plot(y[:, 0], y[:, 1], ls, color=HUE[c], lw=0.8, alpha=0.75)
                        ax.plot(y[0, 0], y[0, 1], "o", color=HUE[c], ms=3)
            ax.set_title(f"{TITLE[feat]}: none's jPCA plane (tasks 0,3,6,9 × 2 inits)", fontsize=8.5)
            clean(ax)
        ax = axes[2]
        cl = S.conds()
        v_sk = [rs["expert_vec"][c]["r2_skew"] if c in rs["expert_vec"] else np.nan for c in cl]
        v_un = [rs["expert_vec"][c]["r2_unconstrained"] if c in rs["expert_vec"] else np.nan for c in cl]
        g_sk = [rs["gemma_vec"][c]["r2_skew"] if c in rs["gemma_vec"] else np.nan for c in cl]
        g_un = [rs["gemma_vec"][c]["r2_unconstrained"] if c in rs["gemma_vec"] else np.nan for c in cl]
        x = np.arange(len(cl))
        ax.bar(x - 0.3, v_un, 0.2, color="#9ec5ef", label="expert R² unconstrained")
        ax.bar(x - 0.1, v_sk, 0.2, color="#2a78d6", label="expert R² skew")
        ax.bar(x + 0.1, g_un, 0.2, color="#f6b99a", label="gemma R² unconstrained")
        ax.bar(x + 0.3, g_sk, 0.2, color="#eb6834", label="gemma R² skew")
        ax.set_xticks(x, cl, rotation=60, fontsize=6.5)
        ax.legend(fontsize=6.5, frameon=False)
        ax.set_title("dx/dt = Mx on PCA-6 (call axis)", fontsize=8.5)
        clean(ax)
        fig.suptitle(f"{SHORT[suite]}: jPCA (Churchland 2012) on call-axis trajectories", fontsize=10)
        fig.tight_layout()
        fig.savefig(od / f"jpca_{suite}.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)
        print(
            f"[jpca] {suite}: expert none skew/unc={rs['expert_vec']['none']['r2_skew']}/{rs['expert_vec']['none']['r2_unconstrained']} "
            f"gemma none {rs['gemma_vec']['none']['r2_skew']}/{rs['gemma_vec']['none']['r2_unconstrained']} "
            f"denoise none {rd['none']['r2_skew']}/{rd['none']['r2_unconstrained']}",
            flush=True,
        )


# ---------------------------------------------------------------- manifold (Gallego 2018)


def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: (k, D) orthonormal rows → k principal angles in degrees (ascending)."""
    s = np.linalg.svd(a @ b.T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def cca_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """canonical correlations of paired rows (QR-based, Gallego's alignment)."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    qx, _ = np.linalg.qr(x)
    qy, _ = np.linalg.qr(y)
    return np.clip(np.linalg.svd(qx.T @ qy, compute_uv=False), 0, 1)


def stage_manifold(out: Path, res: dict, kdim: int = 10) -> None:
    od = out / "5_manifold"
    od.mkdir(parents=True, exist_ok=True)
    res["manifold"] = {"k": kdim}
    for suite in present(out):
        S = Suite(out, suite)
        rs: dict = {}
        for feat in VEC:
            X = S.raw[feat]
            # global reduction first (PCA-50 on the whole suite), manifolds inside it, like Gallego's 'neural space'
            mu0, P0, _ = pca_fit(X, 50)
            R = (X - mu0) @ P0.T
            rows = {
                c: np.concatenate(
                    [np.arange(S.off[m["i"]], S.off[m["i"] + 1]) for m in S.meta if m["cond"] == c]
                )
                for c in S.conds()
            }

            def man(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                m = R[r].mean(0)
                _u, _s, vt = np.linalg.svd(R[r] - m, full_matrices=False)
                return m, vt[:kdim]

            mans = {c: man(r) for c, r in rows.items()}
            mn, Un = mans["none"]
            # baselines: none split by init halves; isotropic random; data-covariance random
            none_eps = [m for m in S.meta if m["axis"] == "none"]
            h1 = np.concatenate(
                [np.arange(S.off[m["i"]], S.off[m["i"] + 1]) for m in none_eps if m["init_idx"] % 2 == 0]
            )
            h2 = np.concatenate(
                [np.arange(S.off[m["i"]], S.off[m["i"] + 1]) for m in none_eps if m["init_idx"] % 2 == 1]
            )
            split = principal_angles(man(h1)[1], man(h2)[1])
            rnd = []
            cov_rnd = []
            C = np.cov((R - R.mean(0)).T)
            Lc = np.linalg.cholesky(C + 1e-6 * np.eye(len(C)))
            for _ in range(200):
                q, _ = np.linalg.qr(RNG.normal(size=(R.shape[1], kdim)))
                rnd.append(principal_angles(q.T, Un))
                q2, _ = np.linalg.qr(Lc @ RNG.normal(size=(R.shape[1], kdim)))
                cov_rnd.append(principal_angles(q2.T, Un))
            rnd, cov_rnd = np.array(rnd), np.array(cov_rnd)
            r: dict = {
                "baseline_none_split_half_deg": [r3(v) for v in split],
                "baseline_random_deg_mean": [r3(v) for v in rnd.mean(0)],
                "baseline_random_deg_p1": [r3(v) for v in np.percentile(rnd, 1, 0)],
                "baseline_cov_random_deg_mean": [r3(v) for v in cov_rnd.mean(0)],
                "baseline_cov_random_deg_p1": [r3(v) for v in np.percentile(cov_rnd, 1, 0)],
            }
            sd_none = np.sqrt(((R[rows["none"]] - mn) ** 2).sum(1).mean())
            for c in S.conds():
                if c == "none":
                    continue
                mc, Uc = mans[c]
                ang = principal_angles(Uc, Un)
                xc = R[rows[c]] - mc
                align = float(((xc @ Un.T) ** 2).sum() / max(((xc @ Uc.T) ** 2).sum(), 1e-12))
                # CCA on paired (task, init, call) latents
                lc, ln, lo = [], [], []
                for m in S.meta:
                    if m["cond"] != c:
                        continue
                    j = S.none.get((m["task_id"], m["init_idx"]))
                    if j is None:
                        continue
                    a = R[S.off[m["i"]] : S.off[m["i"] + 1]]
                    b = R[S.off[j] : S.off[j + 1]]
                    n = min(len(a), len(b))
                    lc.append((a[:n] - mc) @ Uc.T)
                    ln.append((b[:n] - mn) @ Un.T)
                    # control: none episode of another task, same init
                    t2 = (m["task_id"] + 1 + RNG.integers(0, 9)) % 10
                    j2 = S.none.get((t2, m["init_idx"]))
                    if j2 is not None:
                        b2 = R[S.off[j2] : S.off[j2 + 1]]
                        n2 = min(len(a), len(b2))
                        lo.append(((a[:n2] - mc) @ Uc.T, (b2[:n2] - mn) @ Un.T))
                lc_, ln_ = np.concatenate(lc), np.concatenate(ln)
                cc = cca_corr(lc_, ln_)
                unal = [abs(np.corrcoef(lc_[:, i], ln_[:, i])[0, 1]) for i in range(kdim)]
                cc_other = (
                    cca_corr(np.concatenate([x for x, _ in lo]), np.concatenate([y for _, y in lo]))
                    if lo
                    else np.full(kdim, np.nan)
                )
                shift = float(np.linalg.norm(mc - mn) / sd_none)
                r[c] = {
                    "principal_angles_deg": [r3(v) for v in ang],
                    "mean_angle_deg": r3(ang.mean()),
                    "alignment_index": r3(align),  # c's variance captured by none's manifold / by its own
                    "cca_top4_mean": r3(cc[:4].mean()),
                    "cca_all": [r3(v) for v in cc],
                    "unaligned_corr_top4_mean": r3(np.mean(unal[:4])),
                    "cca_top4_mean_vs_other_task_none": r3(np.nanmean(cc_other[:4])),
                    "centroid_shift_over_none_sd": r3(shift),
                }
            # none vs none other task (aligned) as the reference for 'different behaviour'
            lo = []
            for (t, i0), j in S.none.items():
                j2 = S.none.get(((t + 1) % 10, i0))
                if j2 is None:
                    continue
                a, b = R[S.off[j] : S.off[j + 1]], R[S.off[j2] : S.off[j2 + 1]]
                n = min(len(a), len(b))
                lo.append(((a[:n] - mn) @ Un.T, (b[:n] - mn) @ Un.T))
            r["none_vs_none_other_task_cca_top4"] = (
                r3(
                    cca_corr(np.concatenate([x for x, _ in lo]), np.concatenate([y for _, y in lo]))[
                        :4
                    ].mean()
                )
                if lo
                else None
            )
            lo = []
            for (t, i0), j in S.none.items():
                j2 = S.none.get((t, (i0 + 1) % 10))
                if j2 is None:
                    continue
                a, b = R[S.off[j] : S.off[j + 1]], R[S.off[j2] : S.off[j2 + 1]]
                n = min(len(a), len(b))
                lo.append(((a[:n] - mn) @ Un.T, (b[:n] - mn) @ Un.T))
            r["none_vs_none_other_init_cca_top4"] = (
                r3(
                    cca_corr(np.concatenate([x for x, _ in lo]), np.concatenate([y for _, y in lo]))[
                        :4
                    ].mean()
                )
                if lo
                else None
            )
            rs[feat] = r
        res["manifold"][suite] = rs
        # figure: principal angles per condition with baselines
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), facecolor=SURFACE)
        for ax, feat in zip(axes[:2], VEC, strict=True):
            r = rs[feat]
            ax.plot(
                range(1, kdim + 1),
                r["baseline_random_deg_mean"],
                color=MUTED,
                ls="--",
                lw=1.5,
                label="random subspace",
            )
            ax.plot(
                range(1, kdim + 1),
                r["baseline_cov_random_deg_p1"],
                color=MUTED,
                ls=":",
                lw=1.5,
                label="cov-matched random (1st pct)",
            )
            ax.plot(
                range(1, len(r["baseline_none_split_half_deg"]) + 1),
                r["baseline_none_split_half_deg"],
                color=INK,
                lw=2,
                label="none split-half (floor)",
            )
            for c in S.conds():
                if c == "none" or c not in r:
                    continue
                ax.plot(
                    range(1, len(r[c]["principal_angles_deg"]) + 1),
                    r[c]["principal_angles_deg"],
                    color=HUE[c.split(":")[0]],
                    lw=1.4 if ":" not in c else 0.7,
                    alpha=1 if ":" not in c else 0.6,
                    label=c if ":" not in c or c.endswith("swap") else None,
                )
            ax.set_xlabel("principal angle #", fontsize=8)
            ax.set_ylabel("degrees (vs none manifold)", fontsize=8)
            ax.set_title(f"{TITLE[feat]}: top-{kdim} manifolds", fontsize=9)
            ax.legend(fontsize=6.5, frameon=False)
            clean(ax)
        ax = axes[2]
        cl = [c for c in S.conds() if c != "none"]
        x = np.arange(len(cl))
        for off_, feat, col in ((-0.2, "expert_vec", "#2a78d6"), (0.2, "gemma_vec", "#eb6834")):
            r = rs[feat]
            ax.bar(
                x + off_, [r[c]["cca_top4_mean"] for c in cl], 0.18, color=col, label=f"{feat} CCA-aligned"
            )
            ax.scatter(
                x + off_,
                [r[c]["unaligned_corr_top4_mean"] for c in cl],
                marker="_",
                s=90,
                color=INK,
                zorder=4,
            )
            ax.scatter(
                x + off_,
                [r[c]["cca_top4_mean_vs_other_task_none"] for c in cl],
                marker="x",
                s=18,
                color=col,
                zorder=4,
            )
            if r["none_vs_none_other_init_cca_top4"] is not None:
                ax.axhline(r["none_vs_none_other_init_cca_top4"], color=col, ls=":", lw=1)
        ax.set_xticks(x, cl, rotation=60, fontsize=6.5)
        ax.set_ylim(0, 1.02)
        ax.set_title(
            "latent dynamics vs paired none: CCA top-4 (bar), unaligned (—), other-task pair (x), none other-init (⋯)",
            fontsize=7.5,
        )
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        clean(ax)
        fig.suptitle(f"{SHORT[suite]}: preserved manifold? (Gallego 2018)", fontsize=10)
        fig.tight_layout()
        fig.savefig(od / f"manifold_{suite}.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)
        print(
            f"[manifold] {suite}: expert task angle {rs['expert_vec'].get('task', {}).get('mean_angle_deg')} "
            f"split {np.mean(rs['expert_vec']['baseline_none_split_half_deg']):.1f} "
            f"rand {np.mean(rs['expert_vec']['baseline_random_deg_mean']):.1f}",
            flush=True,
        )


# ---------------------------------------------------------------- attractor (Inagaki 2019)


def bimodality(x: np.ndarray) -> dict:
    from scipy.stats import kurtosis, skew
    from sklearn.mixture import GaussianMixture

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return {"n": n}
    g = skew(x)
    k = kurtosis(x)
    bc = (g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    xx = x[:, None]
    b1 = GaussianMixture(1, random_state=0).fit(xx).bic(xx)
    gm2 = GaussianMixture(2, random_state=0, n_init=3).fit(xx)
    b2 = gm2.bic(xx)
    out = {
        "n": n,
        "bimodality_coef": r3(bc),
        "gmm_delta_bic_1_minus_2": r3(b1 - b2),
        "gmm2_means": sorted(r3(v) for v in gm2.means_.ravel()),
    }
    try:
        import diptest

        d, p = diptest.diptest(x)
        out |= {"dip": r3(d), "dip_p": r3(p)}
    except ImportError:
        pass
    return out


def stage_attractor(out: Path, res: dict) -> None:
    od = out / "6_attractor"
    od.mkdir(parents=True, exist_ok=True)
    res["attractor"] = {}
    feats = ["expert_vec", "gemma_vec", "action"]
    kmax = 52
    for suite in present(out):
        S = Suite(out, suite)
        train: list[str | None] = [None] * 10
        for m in S.meta:
            train[m["task_id"]] = m["original_instruction"]  # LIBERO training sentence (same on every axis)
        rs: dict = {}
        for feat in feats:
            X = S.raw[feat] if feat != "action" else S.z["action"]
            mu0, P0, _ = pca_fit(X, 50)

            def traj(i: int, P0: np.ndarray = P0, mu0: np.ndarray = mu0, X: np.ndarray = X) -> np.ndarray:
                y = (X[S.off[i] : S.off[i + 1]] - mu0) @ P0.T
                return np.concatenate([y, np.repeat(y[-1:], kmax - len(y), 0)]) if len(y) < kmax else y[:kmax]

            def mean_traj(ids: list[int]) -> np.ndarray:
                return np.mean([traj(i) for i in ids], 0)

            rf: dict = {}
            for axis in ("task", "position"):
                proj, outc, info = [], [], []
                for t in range(10):
                    a_ids = [S.none[(t, i)] for i in range(10) if (t, i) in S.none]
                    lv_list = (
                        ["task"] if axis == "task" else (POS_LEVELS if suite == "libero_object" else ["swap"])
                    )
                    for lv in lv_list:
                        ids = [
                            m["i"]
                            for m in S.meta
                            if m["axis"] == axis and m["level"] == lv and m["task_id"] == t
                        ]
                        if not ids or not a_ids:
                            continue
                        ref_kind = None
                        nt = named_task(S.meta[ids[0]]["instruction"], train, t) if axis == "task" else None
                        good: list[int] = []
                        if nt is not None:
                            b_fixed = [S.none[(nt, i)] for i in range(10) if (nt, i) in S.none]
                            ref_kind = f"none of task {nt} (named by the new sentence)"
                        else:
                            good = [i for i in ids if S.meta[i]["success"]]
                            if len(good) < 2:
                                continue
                            b_fixed = []
                            ref_kind = "successful perturbed episodes (leave-one-out)"
                        A = mean_traj(a_ids)
                        for i in ids:
                            B = mean_traj(b_fixed or [g for g in good if g != i])
                            cd = B - A
                            nrm = (cd**2).sum(1)
                            p = ((traj(i) - A) * cd).sum(1) / np.maximum(nrm, 1e-9)
                            n_real = S.calls[i]
                            p[n_real:] = np.nan  # only the episode's own calls
                            proj.append(p)
                            m = S.meta[i]
                            if axis == "task":
                                o = (
                                    "new goal"
                                    if m["success_new"]
                                    else "original goal (shortcut)"
                                    if m["success_orig"]
                                    else "neither"
                                )
                            else:
                                o = "success" if m["success"] else "failure"
                            outc.append(o)
                            info.append({"task": t, "level": lv, "init": m["init_idx"], "ref": ref_kind})
                if not proj:
                    continue
                Pm = np.array(proj)
                end = np.array([np.nanmean(p[np.isfinite(p)][-3:]) for p in Pm])
                windows = {
                    "calls_0_2": (0, 3),
                    "calls_5_7": (5, 8),
                    "calls_10_12": (10, 13),
                    "calls_20_22": (20, 23),
                }
                r: dict[str, Any] = {
                    "n": len(Pm),
                    "references": dict(Counter(x["ref"] for x in info)),
                    "endpoint_last3": bimodality(end),
                    "by_window": {w: bimodality(np.nanmean(Pm[:, a:b], 1)) for w, (a, b) in windows.items()},
                    "endpoint_by_outcome": {o: boot_ci(end[np.array(outc) == o]) for o in sorted(set(outc))},
                    "endpoint_share_closer_to_B": r3(np.mean(end > 0.5)),
                }
                r["endpoint_by_outcome_n"] = dict(Counter(outc))
                rf[axis] = r
                # figure
                fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURFACE)
                ocol = {
                    "new goal": "#1baf7a",
                    "original goal (shortcut)": "#e34948",
                    "neither": MUTED,
                    "success": "#1baf7a",
                    "failure": "#e34948",
                }
                for p, o in zip(Pm, outc, strict=True):
                    axes[0].plot(p, color=ocol[o], lw=0.6, alpha=0.5)
                for o in sorted(set(outc)):
                    sel = np.array(outc) == o
                    axes[0].plot(np.nanmean(Pm[sel], 0), color=ocol[o], lw=2.4, label=f"{o} (n={sel.sum()})")
                axes[0].axhline(0, color=INK, lw=0.8)
                axes[0].axhline(1, color=INK, lw=0.8, ls="--")
                axes[0].set_ylim(-1.5, 2.5)
                axes[0].set_xlabel("policy call", fontsize=8)
                axes[0].set_ylabel("projection: 0 = own trained task (none), 1 = reference B", fontsize=7.5)
                axes[0].legend(fontsize=7, frameon=False)
                clean(axes[0])
                bins = np.linspace(-1.5, 2.5, 33)
                for o in sorted(set(outc)):
                    axes[1].hist(
                        np.clip(end[np.array(outc) == o], -1.5, 2.5),
                        bins=bins,
                        color=ocol[o],
                        alpha=0.6,
                        label=o,
                    )
                axes[1].set_xlabel("endpoint (mean of the last 3 calls)", fontsize=8)
                b = r["endpoint_last3"]
                axes[1].set_title(
                    f"BC {b.get('bimodality_coef')} · GMM ΔBIC {b.get('gmm_delta_bic_1_minus_2')} · dip p {b.get('dip_p')}",
                    fontsize=8,
                )
                clean(axes[1])
                fig.suptitle(
                    f"{SHORT[suite]} · {axis} axis · {TITLE[feat]}: coding-direction projection (Inagaki 2019)",
                    fontsize=9.5,
                )
                fig.tight_layout()
                (od / feat).mkdir(exist_ok=True)
                fig.savefig(od / feat / f"{axis}_{suite}.png", dpi=110, facecolor=SURFACE)
                plt.close(fig)
            rs[feat] = rf
        res["attractor"][suite] = rs
        t = rs.get("expert_vec", {}).get("task", {})
        print(
            f"[attractor] {suite}: task expert endpoint BC={t.get('endpoint_last3', {}).get('bimodality_coef')} "
            f"by outcome={ {k: v[0] for k, v in t.get('endpoint_by_outcome', {}).items()} }",
            flush=True,
        )


# ---------------------------------------------------------------- trajectories with time, grasp / release


def call_of(t: int | None) -> int | None:
    """env step (1-based) → index of the policy call whose chunk executed it (calls at steps 0, 10, 20, …)."""
    return None if not t else (int(t) - 1) // 10


def grad_line(ax: Any, y: np.ndarray, color: str, ls: str = "-", lw: float = 1.4) -> None:
    """trajectory with time as colour depth (light = early, dark = late)."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb

    rgb = np.array(to_rgb(color))
    n = len(y)
    if n < 2:
        return
    seg = np.stack([y[:-1], y[1:]], 1)
    w = np.linspace(0.12, 1.0, n - 1)[:, None]
    cols = np.concatenate([1 - w * (1 - rgb), np.ones((n - 1, 1))], 1)  # blend white → colour
    ax.add_collection(LineCollection(seg, colors=cols, linewidths=lw, linestyles=ls))


def mark_traj(ax: Any, y: np.ndarray, m: dict, color: str) -> None:
    ax.plot(y[0, 0], y[0, 1], "o", ms=3.5, color=color, mfc="white", zorder=4)
    for k in range(5, len(y), 5):
        ax.text(y[k, 0], y[k, 1], str(k), fontsize=5, color=color, ha="left", va="bottom", zorder=5)
    g, r = call_of(m.get("first_grasp_t")), call_of(m.get("first_release_t"))
    if g is not None and g < len(y):
        ax.plot(y[g, 0], y[g, 1], "x", ms=8, mew=2.2, color=color, zorder=6)
    if r is not None and r < len(y):
        ax.plot(y[r, 0], y[r, 1], "s", ms=6, color=color, mec=INK, mew=0.8, zorder=6)


def stage_traj(out: Path, res: dict) -> None:
    td = out / "7_trajectories"
    td.mkdir(parents=True, exist_ok=True)
    res["traj"] = {}
    pert_axes = ["object", "position", "semantic", "task", "environment"]
    for suite in present(out):
        S = Suite(out, suite)
        rs: dict = {}
        # ---- example trajectories in the none PCA-2 plane, time as colour depth, × grasp, ■ release
        for feat in ("expert_vec", "gemma_vec", "action"):
            X = S.raw[feat] if feat != "action" else S.z["action"]
            rows = np.concatenate([np.arange(S.off[i], S.off[i + 1]) for i in S.where(axis="none")])
            mu, pcs, evr = pca_fit(X[rows], 2)
            fig, axes = plt.subplots(2, 5, figsize=(22, 8.6), facecolor=SURFACE)
            for col, axis in enumerate(pert_axes):
                lv = "swap" if axis == "position" else None
                for row, t in enumerate((0, 5)):
                    ax = axes[row, col]
                    for init in (0, 1):
                        j = S.none.get((t, init))
                        cand = [
                            m
                            for m in S.meta
                            if m["axis"] == axis
                            and m["task_id"] == t
                            and m["init_idx"] == init
                            and (lv is None or m["level"] == lv)
                        ]
                        pair = []
                        if j is not None:
                            pair.append((S.meta[j], HUE["none"]))
                        if cand:
                            pair.append((cand[0], HUE[axis]))
                        for m, c in pair:
                            y = (
                                S.ep(feat, m["i"], z=False) if feat != "action" else S.ep("action", m["i"])
                            ) - mu
                            y = y @ pcs.T
                            grad_line(ax, y, c, ls="-" if m["success"] else "--")
                            mark_traj(ax, y, m, c)
                    ax.autoscale()
                    ax.set_title(f"{axis} · t{t} · inits 0–1", fontsize=8.5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for s in ("top", "right"):
                        ax.spines[s].set_visible(False)
            h = [
                plt.Line2D([], [], color=HUE["none"], lw=2, label="paired none episode"),
                *[plt.Line2D([], [], color=HUE[a], lw=2, label=a) for a in pert_axes],
                plt.Line2D([], [], color=MUTED, lw=1.5, ls="--", label="failed episode"),
                plt.Line2D([], [], ls="", marker="x", ms=7, mew=1.8, color=INK, label="first grasp"),
                plt.Line2D([], [], ls="", marker="s", ms=5, color=INK, label="first release"),
                plt.Line2D(
                    [],
                    [],
                    ls="",
                    marker="o",
                    ms=4,
                    mfc="white",
                    color=INK,
                    label="call 0 (numbers: every 5 calls)",
                ),
            ]
            fig.legend(handles=h, loc="lower center", ncol=10, fontsize=7.5, frameon=False)
            fig.suptitle(
                f"{SHORT[suite]} — {TITLE[feat]}: trajectories in the none PCA-2 plane ({evr.sum():.0%} var); "
                "colour light → dark = early → late",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.96))
            (td / feat).mkdir(exist_ok=True)
            fig.savefig(td / feat / f"traj_{suite}.png", dpi=105, facecolor=SURFACE)
            plt.close(fig)
        # ---- grasp-aligned distance to the paired none episode (both aligned on their own first grasp)
        ks = np.arange(-8, 16)
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), facecolor=SURFACE)
        for ax, f in zip(axes.flat, FEATS6, strict=True):

            def aligned(i: int, j: int, f: str = f) -> np.ndarray:
                gi, gj = call_of(S.meta[i]["first_grasp_t"]), call_of(S.meta[j]["first_grasp_t"])
                assert gi is not None and gj is not None  # callers pass grasping episodes only
                a, b = S.ep(f, i), S.ep(f, j)
                d = np.full(len(ks), np.nan)
                for n, k in enumerate(ks):
                    if 0 <= gi + k < len(a) and 0 <= gj + k < len(b):
                        d[n] = np.sqrt(((a[gi + k] - b[gj + k]) ** 2).mean())
                return d

            base = []
            for (t, i0), j in S.none.items():
                if not S.meta[j]["first_grasp_t"]:
                    continue
                for i1 in range(10):
                    j2 = S.none.get((t, i1))
                    if i1 != i0 and j2 is not None and S.meta[j2]["first_grasp_t"]:
                        base.append(aligned(j, j2))
            if base:
                ax.plot(
                    ks, np.nanmean(base, 0), color=MUTED, ls="--", lw=1.5, label="none vs none, other init"
                )
            rf: dict = {}
            for axis in pert_axes:
                D, rel = [], []
                for m in S.meta:
                    if m["axis"] != axis or not m["first_grasp_t"]:
                        continue
                    j = S.none.get((m["task_id"], m["init_idx"]))
                    if j is None or not S.meta[j]["first_grasp_t"]:
                        continue
                    D.append(aligned(m["i"], j))
                    rc, gc = call_of(m.get("first_release_t")), call_of(m["first_grasp_t"])
                    if rc is not None and gc is not None:
                        rel.append(rc - gc)
                if len(D) < 3:
                    continue
                mD = np.nanmean(D, 0)
                mD[np.isfinite(np.array(D)).sum(0) < 5] = np.nan  # too few episodes at this offset
                ax.plot(ks, mD, color=HUE[axis], lw=1.6, label=f"{axis} (n={len(D)})")
                if rel:
                    rk = int(np.median(rel))
                    if ks[0] <= rk <= ks[-1]:
                        ax.plot(rk, mD[rk - ks[0]], "s", ms=6, color=HUE[axis], mec=INK, zorder=5)
                pre, post = (ks >= -5) & (ks < 0), (ks > 0) & (ks <= 5)
                bm = np.nanmean(base, 0) if base else np.full(len(ks), np.nan)
                rf[axis] = {
                    "n": len(D),
                    "pre_grasp_5_calls": r3(np.nanmean(mD[pre])),
                    "post_grasp_5_calls": r3(np.nanmean(mD[post])),
                    "pre_over_base": r3(np.nanmean(mD[pre]) / np.nanmean(bm[pre])),
                    "post_over_base": r3(np.nanmean(mD[post]) / np.nanmean(bm[post])),
                    "median_release_minus_grasp_calls": int(np.median(rel)) if rel else None,
                }
            ax.axvline(0, color=INK, lw=0.8)
            ax.plot(0, ax.get_ylim()[0], "x", ms=8, mew=2, color=INK, clip_on=False)
            ax.set_title(TITLE[f], fontsize=9)
            ax.set_xlabel(
                "policy call relative to the first grasp (× = grasp, ■ = median first release)", fontsize=7
            )
            ax.set_ylabel("RMS distance to paired none (z)", fontsize=7)
            clean(ax)
            rs[f] = rf
        axes[0, 0].legend(fontsize=6.5, frameon=False)
        fig.suptitle(
            f"{SHORT[suite]}: distance to the paired none episode, both aligned on their own first grasp",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(td / f"grasp_aligned_{suite}.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)
        res["traj"][suite] = rs
        print(
            f"[traj] {suite}: expert post/base "
            + " ".join(f"{a}={v['post_over_base']}" for a, v in rs.get("expert", {}).items()),
            flush=True,
        )


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--stages", default="load,perf,t021,jpca,manifold,attractor,traj")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    num = a.out / "numbers.json"
    res = json.loads(num.read_text()) if num.exists() else {}
    stages = a.stages.split(",")
    if "load" in stages:
        stage_load(a.run, a.out, a.workers)
        res["source_run"] = str(a.run)
    for name, fn in (
        ("perf", stage_perf),
        ("t021", stage_t021),
        ("jpca", stage_jpca),
        ("manifold", stage_manifold),
        ("attractor", stage_attractor),
        ("traj", stage_traj),
    ):
        if name in stages:
            fn(a.out, res)
            num.write_text(json.dumps(res, indent=1, default=str))
    num.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
