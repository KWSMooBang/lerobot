"""Basket distance experiment (T022): LIBERO-Object, pi05 (lerobot/pi05_libero_finetuned), basket moved toward the
agentview camera by different amounts; behaviour, attention on the basket spots and T021-style internal representations
compared across distance. CPU only, from the eval runs and the extracted attention / activation npz.

Conditions (task × init 0–9 each, training sentences):
  ctlF   original scene, ferrari A100   (libero3 bench-pi05lib-libero3 / ours-obj-custom-train10)
  near   basket (0.10, 0.26), ferrari   (libero3 / ours-obj-custom-basket)
  opp    basket (0.01, -0.33), ferrari  (libero3 / ours-obj-custom-basket)
  ctl    original scene, VGI-102 RTX 4090 (bench-pi05lib-basket_far / ours-obj-custom-train10)
  far20, farM  basket moved further toward the camera, VGI-102 (bench-pi05lib-basket_far / ours-obj-custom-basket)
Paired comparisons are only made within a machine: near / opp vs ctlF, far* vs ctl (the two GPUs drift apart from the
first denoising step, so cross-machine pairs would mix model noise into the distances). ctl vs ctlF is reported as the
machine floor.

Outputs under --out (default /outputs/VLA-shortcut-exps/kubectl/basket/far):
  1_results/outcomes.csv, summary.json, release_xy.png, follow.png
  2_attention/basket_attention.png, carry_maps.png, attention_by_type_diff.png
  5_analysis/numbers.json, distance_summary.png, divergence_by_call.png, divergence_by_phase.png, task_identity.png,
             direction.png
Run (viz venv has sklearn): OPENBLAS_CORETYPE=ZEN $W/cand31/.venv-viz/bin/python basket_far.py \
     --far20 bsk_c20y22:0.20:0.22 --farM bsk_c25y20:0.25:0.20
"""

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

W = Path("/workspace/VLA-shortcut-exps-kubectl")
K = Path("/outputs/VLA-shortcut-exps/kubectl")
RUN_F = W / "out/runs/bench-pi05lib-libero3/jobs"
RUN_V = W / "out/runs/bench-pi05lib-basket_far/jobs"
ATT_F, OBS_F = W / "libero3/attn/out", K / "libero-spatial-object-long/_raw/attn/obs"
ATT_V, OBS_V = W / "basket/far/attn/out", W / "basket/far/attn/obs"
CTLJ, BSKJ = "ours-obj-custom-train10", "ours-obj-custom-basket"

P = 16  # attention patches per side
NC = 28
STEPS = np.arange(NC) * 10
T_END = 280
HALF = 0.07  # basket wall half extent
GROUPS = ["early L0–5", "mid L6–11", "late L12–17"]
LGROUPS = [("L0–5", 0, 6), ("L6–11", 6, 12), ("L12–17", 12, 18)]
SEG_LABEL = ["agentview", "wrist", "BOS", "sentence", "state", "template", "action"]
PHASES = ["approach", "grasp", "carry", "release", "after"]
NB = 5
RNG = np.random.default_rng(0)
NBOOT = 2000
MUTED, INK, INK2, SURFACE, GRID = "#898781", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
FEATS = {  # identical to libero3 l3_analysis.py / basket_traj.py
    "expert": lambda d: np.log(np.concatenate([d["exp_mlp"].mean(1), d["exp_h"].mean(1)], 1)),
    "gemma": lambda d: np.log(np.concatenate([d["vlm_mlp"].reshape(NC, -1), d["vlm_h"].reshape(NC, -1)], 1)),
    "siglip": lambda d: np.log(d["sig_h"][:, :2].reshape(NC, -1)),
    "attention": lambda d: d["attn_mass"].mean(1).reshape(NC, -1),
    "patch_attention": lambda d: np.sqrt(d["attn_img"][:, :, :512].astype(np.float32).reshape(NC, -1)),
    "action": lambda d: d["act_pred"].reshape(NC, -1),
}


def cond_table(a):
    def far(spec):
        vid, x, y = spec.split(":")
        return vid, (float(x), float(y))

    f20, fm = far(a.far20), far(a.farM)
    C = {
        "ctlF": dict(machine="A100", run=RUN_F / CTLJ, att=ATT_F / CTLJ, obs=OBS_F / CTLJ, pre="ctl_t{:02d}_train",
                     xy=(0.0, 0.26), ctl="ctlF"),
        "near": dict(machine="A100", run=RUN_F / BSKJ, att=ATT_F / BSKJ, obs=OBS_F / BSKJ, pre="bsk_near_t{:02d}",
                     xy=(0.10, 0.26), ctl="ctlF"),
        "opp": dict(machine="A100", run=RUN_F / BSKJ, att=ATT_F / BSKJ, obs=OBS_F / BSKJ, pre="bsk_opp_t{:02d}",
                    xy=(0.01, -0.33), ctl="ctlF"),
        "ctl": dict(machine="4090", run=RUN_V / CTLJ, att=ATT_V / CTLJ, obs=OBS_V / CTLJ, pre="ctl_t{:02d}_train",
                    xy=(0.0, 0.26), ctl="ctl"),
        "far20": dict(machine="4090", run=RUN_V / BSKJ, att=ATT_V / BSKJ, obs=OBS_V / BSKJ, pre=f20[0] + "_t{:02d}",
                      xy=f20[1], ctl="ctl"),
        "farM": dict(machine="4090", run=RUN_V / BSKJ, att=ATT_V / BSKJ, obs=OBS_V / BSKJ, pre=fm[0] + "_t{:02d}",
                     xy=fm[1], ctl="ctl"),
    }
    for k, c in C.items():
        dx = np.subtract(c["xy"], (0.0, 0.26))
        c["shift_cm"] = round(float(np.linalg.norm(dx)) * 100, 1)
        tag = " [A100]" if c["machine"] == "A100" else ""
        c["label"] = {"ctlF": "ctl (A100)", "ctl": "ctl (4090)"}.get(k, f"{k} ({c['xy'][0]:.2f}, {c['xy'][1]:.2f}){tag}")
    C["ctlF"]["hue"], C["near"]["hue"], C["opp"]["hue"] = "#8fb6e8", "#1baf7a", "#eb6834"
    C["ctl"]["hue"], C["far20"]["hue"], C["farM"]["hue"] = "#2a78d6", "#eda100", "#e34948"
    return C


KINDS = ["ctlF", "ctl", "near", "far20", "farM", "opp"]
MOVED = ["near", "far20", "farM", "opp"]  # ordered by distance toward the camera, opp last (other side)


# ---------------------------------------------------------------- stats


def r3(x):
    return round(float(x), 3)


def boot_mean(x, n=NBOOT):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    m = np.nanmean(x, 0)
    idx = RNG.integers(0, len(x), (n, len(x)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = np.nanmean(x[idx], 1)
    lo, hi = np.nanpercentile(b, [2.5, 97.5], 0)
    return m, lo, hi


def ci(x):
    m, lo, hi = boot_mean(np.asarray(x, float))
    return [r3(m[0]), r3(lo[0]), r3(hi[0])]


def boot_ratio(a, b, n=NBOOT):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    idx = RNG.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return [r3(a.mean() / b.mean()), r3(np.percentile(r, 2.5)), r3(np.percentile(r, 97.5))]


def nanmean(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x, axis)


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def phase_axis(ax):
    for p in range(1, len(PHASES)):
        ax.axvline(p * NB - 0.5, color="#cfcdc6", lw=0.8)
    ax.set_xticks([p * NB + (NB - 1) / 2 for p in range(len(PHASES))], PHASES, fontsize=7)


# ---------------------------------------------------------------- data


def load(C):
    eps = {}
    cache = {}
    for k in KINDS:
        c = C[k]
        if c["run"] not in cache:
            cache[c["run"]] = [json.loads(x) for x in (c["run"] / "episodes.jsonl").read_text().splitlines()]
        for e in cache[c["run"]]:
            t, i = e["task_id"], e["init_idx"]
            if e["variant"] != c["pre"].format(t) or i > 9:
                continue
            st = np.load(c["run"] / e["steps_file"])
            held = np.array([g != "" for g in st["grasped"]])
            g0 = int(np.argmax(held)) if held.any() else None
            rel = [s for s in range(1, len(held)) if held[s - 1] and not held[s]]
            r = rel[0] if rel else None
            name = f"{e['variant']}_i{i:02d}"
            d = np.load(c["att"] / f"{name}.npz")
            assert len(d["t"]) == NC and (d["t"] == STEPS).all(), name
            ee = dict(kind=k, task=t, init=i, name=name, instruction=e["instruction"], success=bool(e["success_new"]),
                      t_success=e["t_new"], first_grasp=e["first_grasp_object"], grasp=g0, release=r,
                      release_xy=st["eef_pos"][r - 1, :2] if r else None, held_end=bool(held[-1]),
                      action0=st["action"][0], obs=c["obs"] / f"{name}.npz")
            ee.update({key: d[key] for key in ("attn_mass", "attn_img", "exp_mlp", "exp_h", "vlm_h", "vlm_mlp", "sig_h",
                                               "act_pred")})
            ee["failure"] = failure(ee)
            eps[k, t, i] = ee
        n = sum(1 for kk in eps if kk[0] == k)
        assert n == 100, (k, n)
    return eps


def failure(e):
    if e["success"]:
        return "success"
    if e["first_grasp"] is None:
        return "never grasped"
    if e["first_grasp"] == "basket_1":
        return "grasped the basket"
    return "held to timeout" if e["held_end"] else "released outside"


def keyorder():
    return [(t, i) for t in range(10) for i in range(10)]


def bounds(e):
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
    b = bounds(e)
    out = []
    for s in STEPS:
        lab = None
        for p in PHASES:
            if b[p] is not None and b[p][0] <= s < b[p][1]:
                lab = p
        out.append(lab or ("after" if b["after"] else "carry" if b["carry"] else "approach"))
    return out


def coarse_phase(e, s):  # bsk_analysis.py phases (attention on basket spots)
    if e["grasp"] is None or s < e["grasp"]:
        return "approach"
    if e["release"] is None or s < e["release"]:
        return "carry"
    return "after release"


def phase_interp(X, e):
    b = bounds(e)
    out = np.full((len(PHASES) * NB, X.shape[1]), np.nan, np.float32)
    for p, ph in enumerate(PHASES):
        if b[ph] is None:
            continue
        s, en = b[ph]
        q = np.linspace(s, en, NB + 1)
        q = (q[:-1] + q[1:]) / 2
        idx = np.clip(q / 10, 0, NC - 1)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, NC - 1)
        w = (idx - lo)[:, None]
        out[p * NB:(p + 1) * NB] = X[lo] * (1 - w) + X[hi] * w
    return out


# ---------------------------------------------------------------- 1. behaviour


def behaviour(eps, C, out, res):
    d = out / "1_results"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "outcomes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "machine", "task", "init", "instruction", "success", "t_success", "first_grasp", "grasp_t",
                    "release_t", "release_x", "release_y", "held_end", "failure", "paired_ctl_success"])
        for k in KINDS:
            for t, i in keyorder():
                e = eps[k, t, i]
                rx, ry = e["release_xy"] if e["release_xy"] is not None else (None, None)
                w.writerow([k, C[k]["machine"], t, i, e["instruction"], int(e["success"]), e["t_success"], e["first_grasp"],
                            e["grasp"], e["release"], None if rx is None else round(float(rx), 4),
                            None if ry is None else round(float(ry), 4), int(e["held_end"]),
                            "" if e["success"] else e["failure"], int(eps[C[k]["ctl"], t, i]["success"])])
    S = {}
    for k in KINDS:
        E = [eps[k, t, i] for t, i in keyorder()]
        P_ = [eps[C[k]["ctl"], t, i] for t, i in keyorder()]
        fails = [e["failure"] for e in E if not e["success"]]
        s = dict(machine=C[k]["machine"], basket_xy=list(C[k]["xy"]), shift_cm=C[k]["shift_cm"],
                 success=sum(e["success"] for e in E), per_task=[sum(eps[k, t, i]["success"] for i in range(10)) for t in range(10)],
                 failures={f: fails.count(f) for f in sorted(set(fails))},
                 paired_ctl_ok_moved_fail=sum(p["success"] and not e["success"] for e, p in zip(E, P_)),
                 paired_ctl_fail_moved_ok=sum(e["success"] and not p["success"] for e, p in zip(E, P_)))
        ok = [e for e in E if e["success"]]
        if ok:
            s["success_release_xy_mean"] = np.mean([e["release_xy"] for e in ok], 0).round(3).tolist()
            s["carry_steps_success_mean"] = round(float(np.mean([e["release"] - e["grasp"] for e in ok])), 1)
            s["t_success_mean"] = round(float(np.mean([e["t_success"] for e in ok])), 1)
        # follow fraction: release-point shift along the basket shift, paired with the same machine's ctl episode
        if k in MOVED:
            sv = np.subtract(C[k]["xy"], C[C[k]["ctl"]]["xy"])
            for subset in ("both_success", "both_released"):
                fr = []
                for e, p in zip(E, P_):
                    if e["release_xy"] is None or p["release_xy"] is None:
                        continue
                    if subset == "both_success" and not (e["success"] and p["success"]):
                        continue
                    fr.append(float(np.dot(e["release_xy"] - p["release_xy"], sv) / np.dot(sv, sv)))
                s[f"follow_{subset}"] = ci(fr) + [len(fr)] if fr else None
            # perpendicular error and distance from the new basket centre (success)
            s["release_dist_to_new_centre_cm_success"] = ci(
                [np.linalg.norm(e["release_xy"] - np.array(C[k]["xy"])) * 100 for e in ok]) if ok else None
        S[k] = s
    res["behaviour"] = S
    (d / "summary.json").write_text(json.dumps(S, indent=1))

    # release points, top-down, policy view orientation (world +y on the left)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE)
    for ax, (ks, title) in zip(axes, [(["ctlF", "near", "opp"], "A100 (ferrari): ctl, near, opp"),
                                      (["ctl", "far20", "farM"], "RTX 4090 (VGI-102): ctl, far20, far_max")]):
        for k in ks:
            cx, cy = C[k]["xy"]
            ax.add_patch(plt.Rectangle((cy - HALF, cx - HALF), 2 * HALF, 2 * HALF, fill=False, ec=C[k]["hue"],
                                       ls="--" if k.startswith("ctl") else "-", lw=1.5))
            for okf, mk in [(True, "o"), (False, "x")]:
                xy = np.array([eps[k, t, i]["release_xy"] for t, i in keyorder() if eps[k, t, i]["success"] == okf
                               and eps[k, t, i]["release_xy"] is not None]).reshape(-1, 2)
                ax.scatter(xy[:, 1], xy[:, 0], s=18 if okf else 30, marker=mk, color=C[k]["hue"], alpha=0.7 if okf else 1,
                           lw=0 if okf else 1.4, label=f"{C[k]['label']}: {'success' if okf else 'failed'} ({len(xy)})")
        ax.set_xlim(0.42, -0.45)
        ax.set_ylim(-0.17, 0.42)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("world y (policy view: left ↔ right)", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
        clean(ax)
    axes[0].set_ylabel("world x (↑ toward the agentview camera)", fontsize=8)
    fig.suptitle("where the gripper let go of the first grasped object (eef xy one step before release); boxes = basket "
                 "walls, dashed = original spot", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(d / "release_xy.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # follow fraction and success vs distance
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), facecolor=SURFACE)
    for k in MOVED:
        x = C[k]["shift_cm"]
        a = S[k]
        axes[0].scatter(x, a["success"], color=C[k]["hue"], s=50, marker="s" if C[k]["machine"] == "A100" else "o")
        axes[0].annotate(k, (x, a["success"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        for ax, key in ((axes[1], "follow_both_success"), (axes[2], "follow_both_released")):
            if a[key]:
                m, lo, hi = a[key][:3]
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="s" if C[k]["machine"] == "A100" else "o",
                            color=C[k]["hue"], ms=7, capsize=3)
                ax.annotate(f"{k} (n={a[key][3]})", (x, m), fontsize=7, xytext=(5, 3), textcoords="offset points")
    for k in ("ctlF", "ctl"):
        axes[0].axhline(S[k]["success"], color=C[k]["hue"], ls=":", lw=1, label=f"{C[k]['label']}: {S[k]['success']}")
    axes[0].set_ylabel("success / 100", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_ylabel("release shift along basket shift ÷ basket shift", fontsize=8)
    axes[1].set_title("both episodes of the pair succeeded", fontsize=8)
    axes[2].set_title("both released (any outcome)", fontsize=8)
    for ax in axes:
        ax.set_xlabel("basket shift from the original spot (cm)", fontsize=8)
        clean(ax)
    for ax in axes[1:]:
        ax.axhline(1, color=MUTED, ls=":", lw=1)
        ax.axhline(0, color=INK2, lw=0.7)
    fig.suptitle("success and how far the release point follows the basket (paired with the same machine's ctl, same "
                 "task·init; square = A100 run, circle = 4090 run; 95% bootstrap CI)", fontsize=9)
    fig.tight_layout()
    fig.savefig(d / "follow.png", dpi=125, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- 2. attention on the basket spots


def changed(x, y):
    dd = np.abs(x.astype(np.int16) - y.astype(np.int16)).max(-1) > 40
    n = dd.shape[0] // P
    return dd[: n * P, : n * P].reshape(P, n, P, n).mean((1, 3)) > 0.25


def first_frame(e):
    return np.load(e["obs"])["image"][0]


def basket_masks(eps, C):
    """old spot / new basket patch masks per moved episode. Reference without a basket on the left = the opp frame of
    the same task·init (ferrari). opp: as in bsk_analysis.py (left / right halves of changed(ctlF, opp))."""
    left = np.zeros((P, P), bool)
    left[:, : P // 2] = True
    M = {}
    for t, i in keyorder():
        O = first_frame(eps["opp", t, i])
        ch = {k: changed(first_frame(eps[k, t, i]), O) for k in ("ctlF", "ctl", "near", "far20", "farM")}
        M["opp", t, i] = dict(old=ch["ctlF"] & left, new=ch["ctlF"] & ~left)
        for k in ("near", "far20", "farM"):
            old, new = ch[C[k]["ctl"]] & left, ch[k] & left
            M[k, t, i] = dict(old=old & ~new, new=new & ~old)
    return M


def agent_map(e, call, g):
    return e["attn_img"][call, g, : P * P].astype(np.float32).reshape(P, P)


def basket_attention(eps, C, out, res):
    d = out / "2_attention"
    d.mkdir(parents=True, exist_ok=True)
    M = basket_masks(eps, C)
    PH = ["approach", "carry", "after release"]
    share = defaultdict(list)
    for k in MOVED:
        for t, i in keyorder():
            m = M[k, t, i]
            for who, ee in (("ctl", eps[C[k]["ctl"], t, i]), ("moved", eps[k, t, i])):
                for g in range(3):
                    acc = defaultdict(list)
                    for c, s in enumerate(STEPS):
                        a = agent_map(ee, c, g)
                        a = a / a.sum()
                        ph = coarse_phase(ee, int(s))
                        acc[ph, "old"].append(a[m["old"]].sum())
                        acc[ph, "new"].append(a[m["new"]].sum())
                    for (ph, reg), v in acc.items():
                        share[k, who, reg, ph, g].append(np.mean(v))
    npatch = {k: [float(np.mean([M[k, t, i][r].sum() for t, i in keyorder()])) for r in ("old", "new")] for k in MOVED}
    res["basket_attention"] = dict(
        mean_patches_old_new={k: [round(x, 1) for x in v] for k, v in npatch.items()},
        share_pct={f"{k}|{who}|{reg}|{ph}|{GROUPS[g]}": ci(np.array(v) * 100) for (k, who, reg, ph, g), v in share.items()})
    fig, axes = plt.subplots(len(MOVED), 3, figsize=(12, 3.1 * len(MOVED)), sharey="row", facecolor=SURFACE)
    for r, k in enumerate(MOVED):
        for g in range(3):
            ax = axes[r, g]
            x = np.arange(len(PH))
            bars = [("ctl episode · old spot (basket there)", "ctl", "old", C[C[k]["ctl"]]["hue"], None),
                    ("ctl episode · new spot (empty floor)", "ctl", "new", C[C[k]["ctl"]]["hue"], "//"),
                    ("moved episode · old spot (empty floor)", "moved", "old", C[k]["hue"], "//"),
                    ("moved episode · new spot (basket there)", "moved", "new", C[k]["hue"], None)]
            for b, (lab, who, reg, col, hatch) in enumerate(bars):
                v = [np.mean(share[k, who, reg, ph, g]) * 100 if share[k, who, reg, ph, g] else np.nan for ph in PH]
                ax.bar(x + (b - 1.5) * 0.2, v, 0.17, color=col if hatch is None else "white", ec=col, hatch=hatch, lw=1.2,
                       label=lab)
            ax.set_xticks(x, PH, fontsize=8)
            ax.set_title(f"{C[k]['label']} · {GROUPS[g]}  (patches old/new {npatch[k][0]:.0f}/{npatch[k][1]:.0f})",
                         fontsize=8)
            clean(ax)
        axes[r, 0].set_ylabel("% of agentview attention", fontsize=8)
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle("action-expert attention on the old spot and the new basket, by distance (patches from first-frame "
                 "differences against the opp frame; overlap left out; pairs within a machine; [A100] = ferrari run)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(d / "basket_attention.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # carry-phase mean maps (late layers): ctl, moved, difference; per moved condition
    G_ = 2
    fig, axes = plt.subplots(len(MOVED), 4, figsize=(14, 3.6 * len(MOVED)), facecolor=SURFACE)
    for r, k in enumerate(MOVED):
        mean, bg = {}, {}
        for who in (C[k]["ctl"], k):
            ms, ims = [], []
            for t, i in keyorder():
                ee = eps[who, t, i]
                cs = [c for c, s in enumerate(STEPS) if coarse_phase(ee, int(s)) == "carry"]
                if cs:
                    a = np.mean([agent_map(ee, c, G_) for c in cs], 0)
                    ms.append(a / a.sum())
                if i == 0:
                    ims.append(first_frame(ee).astype(np.float32))
            mean[who], bg[who] = np.mean(ms, 0), np.mean(ims, 0) / 255
        cw = C[k]["ctl"]
        for c, who in enumerate((cw, k)):
            ax = axes[r, c]
            ax.imshow(bg[who])
            ax.imshow(mean[who], cmap="inferno", alpha=0.55, extent=(0, bg[who].shape[1], bg[who].shape[0], 0))
            ax.set_title(f"{C[who]['label']}\nmean carry attention", fontsize=8)
        diff = mean[k] - mean[cw]
        lim = np.abs(diff).max()
        ax = axes[r, 2]
        ax.imshow(bg[k], alpha=0.6)
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.75, extent=(0, bg[k].shape[1], bg[k].shape[0], 0))
        ax.set_title("moved − ctl (red: more when moved)", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax = axes[r, 3]
        mo = np.mean([M[k, t, i]["old"] for t, i in keyorder()], 0)
        mn = np.mean([M[k, t, i]["new"] for t, i in keyorder()], 0)
        ax.imshow(bg[k], alpha=0.6)
        ax.imshow(np.dstack([mn, np.zeros_like(mn), mo, np.maximum(mo, mn) * 0.8]),
                  extent=(0, bg[k].shape[1], bg[k].shape[0], 0))
        ax.set_title("patches: red = new basket, blue = old spot", fontsize=8)
        for a_ in axes[r]:
            a_.set_xticks([])
            a_.set_yticks([])
    fig.suptitle(f"policy view, {GROUPS[G_]}; background = mean first frame of init 0 over the 10 tasks", fontsize=9)
    fig.tight_layout()
    fig.savefig(d / "carry_maps.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- 2b. attention by input type (paired, carry phase)


def attention_by_type(eps, C, out, res):
    A = {}
    for k in KINDS:
        arr = np.full((100, len(PHASES), 3, 7), np.nan)
        for j, (t, i) in enumerate(keyorder()):
            e = eps[k, t, i]
            m = e["attn_mass"].mean(1)
            lab = np.array(call_phase(e))
            for p, ph in enumerate(PHASES):
                if (lab == ph).any():
                    for g, (_, lo, hi) in enumerate(LGROUPS):
                        arr[j, p, g] = m[lab == ph, lo:hi].mean((0, 1)) * 100
        A[k] = arr
    res["attention_by_type_diff_pp"] = {}
    fig, axes = plt.subplots(3, len(PHASES), figsize=(18, 9), sharey="row", facecolor=SURFACE)
    x = np.arange(7)
    wd = 0.8 / (len(MOVED) + 1)
    for g, (gn, _, _) in enumerate(LGROUPS):
        for p, ph in enumerate(PHASES):
            ax = axes[g, p]
            for j, k in enumerate(MOVED + ["ctl"]):
                ref = "ctlF" if k == "ctl" else C[k]["ctl"]
                dd = A[k][:, p, g] - A[ref][:, p, g]
                m, lo, hi = boot_mean(dd)
                lab = "ctl(4090) − ctl(A100): machine floor" if k == "ctl" else f"{k} − {ref}"
                ax.bar(x + (j - len(MOVED) / 2) * wd, m, wd * 0.9, color=C[k]["hue"] if k != "ctl" else MUTED, label=lab)
                ax.errorbar(x + (j - len(MOVED) / 2) * wd, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, lw=0.5)
                for s in range(7):
                    res["attention_by_type_diff_pp"][f"{gn}|{ph}|{lab.split(':')[0]}|{SEG_LABEL[s]}"] = [
                        r3(m[s]), r3(lo[s]), r3(hi[s])]
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(x, SEG_LABEL, fontsize=7, rotation=30)
            ax.set_title(f"{gn} · {ph}", fontsize=9)
            clean(ax)
        axes[g, 0].set_ylabel("paired difference (pp)", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("action-expert attention share by input type: moved − ctl of the same machine, same task·init (pp; 95% "
                 "bootstrap CI). Gray = 4090 ctl − A100 ctl (machine floor).", fontsize=10)
    fig.tight_layout()
    (out / "2_attention").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "2_attention" / "attention_by_type_diff.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- 3. internal representations (T021 measures)


def rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2, -1))


def feature_arrays(eps):
    F = {}
    for f, fn in FEATS.items():
        raw = {k: np.stack([fn(eps[k, t, i]) for t, i in keyorder()]).astype(np.float32) for k in KINDS}
        flat = np.concatenate([raw[k].reshape(-1, raw[k].shape[-1]) for k in KINDS])
        mu, sd = flat.mean(0), flat.std(0)
        keep = sd > 1e-6
        F[f] = {k: ((raw[k] - mu) / np.where(keep, sd, 1))[..., keep] for k in KINDS}
    return F


def baseline(Xc):
    out = np.full(Xc.shape[:2], np.nan)
    for t in range(10):
        blk = Xc[t * 10:(t + 1) * 10]
        for a in range(10):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[t * 10 + a] = np.nanmean([rms(blk[a], blk[b]) for b in range(10) if b != a], 0)
    return out


def representation(eps, C, out, res):
    from sklearn.decomposition import PCA

    d = out / "5_analysis"
    d.mkdir(parents=True, exist_ok=True)
    F = feature_arrays(eps)
    ko = keyorder()
    PI = {f: {k: np.stack([phase_interp(F[f][k][j], eps[k, t, i]) for j, (t, i) in enumerate(ko)]) for k in KINDS}
          for f in FEATS}
    R = {"task_identity": {}, "divergence": {}, "direction": {}}
    task = np.repeat(np.arange(10), 10)
    for f in FEATS:
        # Q1 task identity: PCA-50 of whole trials (all 600); nearest centroid among the 10 ctl task centroids of the
        # same machine (leave-one-out for ctl itself); ctl(4090) is also scored against the A100 centroids.
        Z = np.concatenate([F[f][k].reshape(100, -1) for k in KINDS])
        Pz = PCA(n_components=50, random_state=0).fit_transform(Z)
        Pk = {k: Pz[j * 100:(j + 1) * 100] for j, k in enumerate(KINDS)}
        r = {}
        for k in KINDS + ["ctl_vs_A100"]:
            src, cref = ("ctl", "ctlF") if k == "ctl_vs_A100" else (k, C[k]["ctl"])
            own = []
            for j in range(100):
                cent = np.stack([Pk[cref][task == t].mean(0) for t in range(10)])
                if src == cref:
                    cent[task[j]] = Pk[cref][(task == task[j]) & (np.arange(100) != j)].mean(0)
                own.append(float(np.argmin(np.linalg.norm(cent - Pk[src][j], axis=1)) == task[j]))
            r[k] = ci(own)
        R["task_identity"][f] = r
        # Q2 divergence: distance to the same machine's ctl at the same task·init, by call and phase-aligned;
        # baseline = that ctl vs its other inits (same task)
        rr = {}
        Dc = {k: rms(F[f][k], F[f][C[k]["ctl"]]) for k in MOVED}
        Dc["machine"] = rms(F[f]["ctl"], F[f]["ctlF"])
        Bc = {m: baseline(F[f][m]) for m in ("ctl", "ctlF")}
        Dp = {k: rms(PI[f][k], PI[f][C[k]["ctl"]]) for k in MOVED}
        Dp["machine"] = rms(PI[f]["ctl"], PI[f]["ctlF"])
        Bp = {m: baseline(PI[f][m]) for m in ("ctl", "ctlF")}
        for k in MOVED + ["machine"]:
            ref = "ctl" if k == "machine" else C[k]["ctl"]
            first = None
            for c in range(NC):
                if boot_ratio(Dc[k][:, c], Bc[ref][:, c])[1] > 1:
                    first = c
                    break
            ph = {}
            for p, pn in enumerate(PHASES):
                sl = slice(p * NB, (p + 1) * NB)
                ph[pn] = boot_ratio(nanmean(Dp[k][:, sl], 1), nanmean(Bp[ref][:, sl], 1))
            rr[k] = dict(first_call_above_base=first, call0_over_base=boot_ratio(Dc[k][:, 0], Bc[ref][:, 0]),
                         phase_over_base=ph, whole_over_base=boot_ratio(Dc[k].mean(1), Bc[ref].mean(1)))
        R["divergence"][f] = rr
        # direction: per task·init and phase bin, λ = <k − ctl, u> / |u|² onto
        #   u_far = farM − ctl (4090, same init): how much of the far_max shift far20 makes (far_max = 1 by definition)
        #   u_opp = opp − ctlF (A100, same init): T021's ctl→opp axis; for far*, the shift is taken against the 4090 ctl
        #   (cross-machine projection; the machine floor ctl − ctlF is projected too)
        dd = {}
        uf = PI[f]["farM"] - PI[f]["ctl"]
        uo = PI[f]["opp"] - PI[f]["ctlF"]
        for name, (v, u) in {"far20_on_ctl_to_farM": (PI[f]["far20"] - PI[f]["ctl"], uf),
                             "near_on_ctlF_to_opp": (PI[f]["near"] - PI[f]["ctlF"], uo),
                             "far20_on_ctlF_to_opp": (PI[f]["far20"] - PI[f]["ctl"], uo),
                             "farM_on_ctlF_to_opp": (PI[f]["farM"] - PI[f]["ctl"], uo),
                             "machine_on_ctlF_to_opp": (PI[f]["ctl"] - PI[f]["ctlF"], uo)}.items():
            with np.errstate(all="ignore"):
                lam = (v * u).sum(-1) / (u * u).sum(-1)
                cos = (v * u).sum(-1) / np.sqrt((v * v).sum(-1) * (u * u).sum(-1))
            dd[name] = {pn: dict(lam=ci(nanmean(lam[:, p * NB:(p + 1) * NB], 1)),
                                 cos=ci(nanmean(cos[:, p * NB:(p + 1) * NB], 1))) for p, pn in enumerate(PHASES)}
            dd[name]["_curve"] = [boot_mean(lam)[0].round(3).tolist(), boot_mean(cos)[0].round(3).tolist()]
        R["direction"][f] = dd
        R.setdefault("_curves", {})[f] = dict(
            call={k: [x.tolist() for x in boot_mean(Dc[k] / nanmean(Bc["ctl" if k == "machine" else C[k]["ctl"]], 0))]
                  for k in MOVED + ["machine"]},
            phase={k: [x.tolist() for x in boot_mean(Dp[k] / nanmean(Bp["ctl" if k == "machine" else C[k]["ctl"]], 0))]
                   for k in MOVED + ["machine"]})
        print(f"[rep] {f}: own " + " ".join(f"{k}={r[k][0]}" for k in r) + " | carry/base " +
              " ".join(f"{k}={rr[k]['phase_over_base']['carry'][0]}" for k in MOVED + ["machine"]), flush=True)
    curves = R.pop("_curves")
    res["representation"] = R

    # figures: divergence (ratio to the baseline) by call and by phase
    for axis, nb in (("call", NC), ("phase", len(PHASES) * NB)):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), facecolor=SURFACE)
        for ax, f in zip(axes.flat, FEATS):
            for k in MOVED + ["machine"]:
                m, lo, hi = (np.array(x) for x in curves[f][axis][k])
                col = MUTED if k == "machine" else C[k]["hue"]
                lab = "ctl(4090) vs ctl(A100): machine floor" if k == "machine" else f"{C[k]['label']} vs its ctl"
                ax.plot(m, color=col, lw=1.7, ls="--" if k == "machine" else "-", label=lab)
                ax.fill_between(np.arange(nb), lo, hi, color=col, alpha=0.15, lw=0)
            ax.axhline(1, color=INK2, lw=0.8, ls=":")
            ax.set_title(f, fontsize=9)
            ax.set_ylabel("distance ÷ ctl other-init baseline", fontsize=7)
            clean(ax)
            if axis == "phase":
                phase_axis(ax)
            else:
                ax.set_xlabel("policy call (×10 steps)", fontsize=7)
        axes[0, 0].legend(fontsize=6.5, frameon=False)
        fig.suptitle(f"distance to the same task·init ctl episode of the same machine ÷ ctl-vs-other-inits baseline, "
                     f"{'by call' if axis == 'call' else 'phase-aligned'}; 1 = as different as another init (95% CI)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(d / f"divergence_by_{axis}.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)

    # task identity bars
    fig, ax = plt.subplots(figsize=(12, 3.6), facecolor=SURFACE)
    ks = KINDS + ["ctl_vs_A100"]
    x = np.arange(len(FEATS))
    wd = 0.8 / len(ks)
    for j, k in enumerate(ks):
        v = np.array([R["task_identity"][f][k] for f in FEATS])
        col = MUTED if k == "ctl_vs_A100" else C[k]["hue"]
        lab = "ctl(4090) on A100 centroids" if k == "ctl_vs_A100" else C[k]["label"]
        ax.bar(x + (j - len(ks) / 2 + 0.5) * wd, v[:, 0], wd * 0.9, color=col, label=lab)
        ax.errorbar(x + (j - len(ks) / 2 + 0.5) * wd, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none",
                    ecolor=INK, lw=0.6)
    ax.set_xticks(x, list(FEATS), fontsize=8)
    ax.set_ylabel("nearest ctl task centroid = own task", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower left")
    clean(ax)
    fig.suptitle("do moved trials stay with their task's ctl cluster? (PCA-50 of whole trials, centroids from the same "
                 "machine's ctl; 95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "task_identity.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # direction curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), facecolor=SURFACE)
    HUEL = {"far20_on_ctl_to_farM": C["far20"]["hue"], "near_on_ctlF_to_opp": C["near"]["hue"],
            "far20_on_ctlF_to_opp": "#c98a00", "farM_on_ctlF_to_opp": C["farM"]["hue"], "machine_on_ctlF_to_opp": MUTED}
    for ax, f in zip(axes.flat, FEATS):
        for name, col in HUEL.items():
            ax.plot(R["direction"][f][name]["_curve"][0], color=col, lw=1.6, label=name,
                    ls="--" if name.startswith("machine") else "-")
        ax.axhline(0, color=INK2, lw=0.7)
        ax.axhline(1, color=MUTED, lw=0.7, ls=":")
        ax.set_title(f, fontsize=9)
        phase_axis(ax)
        clean(ax)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("direction of the representation shift: λ = projection of (moved − ctl) on a reference shift, per "
                 "task·init, phase-aligned (mean of 100)", fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "direction.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- distance summary


def distance_summary(C, out, res):
    B, Ra, Rr = res["behaviour"], res["basket_attention"]["share_pct"], res["representation"]
    rows = []
    for k in MOVED:
        b = B[k]
        row = dict(cond=k, machine=C[k]["machine"], xy=C[k]["xy"], shift_cm=C[k]["shift_cm"], success=b["success"],
                   follow_success=b["follow_both_success"], carry_steps=b.get("carry_steps_success_mean"),
                   att_late_carry={f"{who}|{reg}": Ra.get(f"{k}|{who}|{reg}|carry|late L12–17")
                                   for who in ("ctl", "moved") for reg in ("old", "new")},
                   expert_own=Rr["task_identity"]["expert"][k],
                   expert_carry_over_base=Rr["divergence"]["expert"][k]["phase_over_base"]["carry"],
                   expert_first_call=Rr["divergence"]["expert"][k]["first_call_above_base"],
                   action_first_call=Rr["divergence"]["action"][k]["first_call_above_base"])
        rows.append(row)
    res["distance_summary"] = rows
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), facecolor=SURFACE)
    for r in rows:
        k, x = r["cond"], r["shift_cm"]
        mk = "s" if r["machine"] == "A100" else "o"
        col = C[k]["hue"]
        for ax, val in ((axes[0], r["follow_success"]), (axes[2], r["expert_own"]), (axes[3], r["expert_carry_over_base"])):
            if val:
                ax.errorbar(x, val[0], yerr=[[val[0] - val[1]], [val[2] - val[0]]], fmt=mk, color=col, ms=7, capsize=3)
                ax.annotate(k, (x, val[0]), fontsize=7, xytext=(5, 3), textcoords="offset points")
        a = r["att_late_carry"]
        if a["moved|new"]:
            axes[1].scatter(x, a["moved|new"][0], marker=mk, color=col, s=45)
            axes[1].scatter(x, a["moved|old"][0], marker=mk, facecolor="white", edgecolor=col, s=45)
            axes[1].annotate(k, (x, a["moved|new"][0]), fontsize=7, xytext=(5, 3), textcoords="offset points")
    axes[0].set_title("release follows the basket (fraction)", fontsize=9)
    axes[1].set_title("late-layer carry attention, moved episode:\nfilled = new basket, open = old spot (%)", fontsize=9)
    axes[2].set_title("expert: nearest ctl centroid = own task", fontsize=9)
    axes[3].set_title("expert distance to own-init ctl ÷ baseline, carry", fontsize=9)
    for ax in axes:
        ax.set_xlabel("basket shift (cm); square = A100 run, circle = 4090 run", fontsize=7)
        clean(ax)
    axes[3].axhline(1, color=MUTED, ls=":", lw=1)
    fig.suptitle("by distance: near (10 cm, A100) → far20 → far_max (4090); opp (other side, A100) for reference",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "5_analysis" / "distance_summary.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--far20", required=True, help="variant prefix:x:y, e.g. bsk_far20:0.20:0.26")
    ap.add_argument("--farM", required=True, help="variant prefix:x:y for far_max")
    ap.add_argument("--out", default=str(K / "basket/far"))
    ap.add_argument("--test-ferrari", action="store_true",
                    help="pipeline test before the 4090 run: the 4090 slots read the ferrari run (pass ferrari variants)")
    a = ap.parse_args()
    if a.test_ferrari:
        global RUN_V, ATT_V, OBS_V
        RUN_V, ATT_V, OBS_V = RUN_F, ATT_F, OBS_F
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C = cond_table(a)
    eps = load(C)
    print("loaded", len(eps), flush=True)
    res = {"conditions": {k: {kk: (str(v) if isinstance(v, Path) else v) for kk, v in c.items()} for k, c in C.items()}}
    behaviour(eps, C, out, res)
    print(json.dumps({k: (v["success"], v.get("follow_both_success")) for k, v in res["behaviour"].items()}), flush=True)
    basket_attention(eps, C, out, res)
    attention_by_type(eps, C, out, res)
    representation(eps, C, out, res)
    distance_summary(C, out, res)
    (out / "5_analysis" / "numbers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["distance_summary"], indent=1))


if __name__ == "__main__":
    main()
