"""Trajectories with time, phase and grasp markers for the four LIBERO suites (T025). CPU only, from the already-extracted
npz (no GPU, no re-extraction); drawing follows T024 basket_phase.py.

Data (all A100 / ferrari, pi05 lerobot/pi05_libero_finetuned, task × init 0–9, no early stop):
  Spatial S1–S10, Object O1–O10, Long L1–L10   libero3 run   (28 / 28 / 52 policy calls per episode)
  Goal G1–G10 (train10) + cand31 g01–g31       cand31 runs   (30 calls; 410 trials)
Features per call: the six T021 definitions (expert, gemma, siglip, attention, patch_attention, action), same as
libero3 l3_analysis.py / cand31 attn_traj.py / basket_phase.py.

Events (step data recorded by run_eval.py at every env step: robosuite contacts of the gripper geoms, "grasped" = scene
object touched by both fingers, gripper command; the attention extraction replayed these actions with eef difference 0,
so the recorded data is the replay):
  grasp    both fingers touch the same movable object while the gripper is commanded closed; runs of such steps on the
           same object with gaps ≤ GAP_MERGE steps are merged, runs shorter than MIN_HOLD steps dropped.
           Fixtures (cabinets, stove, microwave, wine rack, desk caddy) are never a grasp.
  release  first step after a grasp run ends; kind "open" if the gripper command is open within RELEASE_OPEN steps,
           otherwise "slip" (object left the fingers while still closed).
  contact  first step any gripper geom touches a scene object/fixture; drawn (◆) only for episodes without a grasp.
Call of an event = min(step // 10, NC − 1): the last observation at or before the event.

Spaces (per suite; Goal = train10 + cand31 in one space):
  A  fixed PCA: z-score per dimension + PCA fitted on the calls of the suite's training trials only (Goal: train10);
     everything else projected. Axis labels give the explained-variance share.
  B  joint t-SNE: z-score over all calls of all trials in the suite, PCA-30, t-SNE (perplexity 50) of all calls.
  T40 summary: z-score + PCA fitted on all calls of the 400 training trials of the four suites.

Outputs:
  <libero3 out>/8_phase_traj/   Spatial, Object, Long, task40 summary
  <goal out>/8_phase_traj/      Goal + cand31
    events/events_<suite>.csv           one row per trial: all grasp / release / contact steps and calls
    a_overview/<feat>_<suite>_{tsne,pca}[_cand].png
    b_zoom/<feat>/<suite>/<task>.png    (expert, patch_attention, action)
    numbers_<suite>.json, tables_<suite>.md
Run: OPENBLAS_CORETYPE=ZEN $W/cand31/.venv-viz/bin/python suite_phase_traj.py [--suites SOLG] [--parts eabcs] [--recompute]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")  # task rule: ≤ 8 CPU workers

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import textwrap  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

K = Path("/outputs/VLA-shortcut-exps/kubectl")
L3 = K / "libero-spatial-object-long"
GC = K / "libero_goal_composition"
OUT = {"S": L3 / "8_phase_traj", "O": L3 / "8_phase_traj", "L": L3 / "8_phase_traj", "G": GC / "8_phase_traj",
       "T40": L3 / "8_phase_traj"}
J3 = L3 / "_raw/bench-pi05lib-libero3/jobs"
SUITES = {  # tag → list of (group, npz dir, job dir), NC
    "S": ([("train", L3 / "_raw/attn/out/ours-spat-custom-train10", J3 / "ours-spat-custom-train10")], 28),
    "O": ([("train", L3 / "_raw/attn/out/ours-obj-custom-train10", J3 / "ours-obj-custom-train10")], 28),
    "L": ([("train", L3 / "_raw/attn/out/ours-l10-custom-train10", J3 / "ours-l10-custom-train10")], 52),
    "G": ([("train", GC / "_raw/attn/out/ours-goal-custom-train10",
            GC / "_raw/runs/bench-pi05lib-train10/jobs/ours-goal-custom-train10"),
           ("cand", GC / "_raw/attn/out/ours-goal-custom-cand31",
            GC / "_raw/runs/bench-pi05lib-cand31/jobs/ours-goal-custom-cand31")], 30),
}
SUITE_NAME = {"S": "Spatial", "O": "Object", "G": "Goal", "L": "Long"}
SUITE_HUE = {"S": "#6250d6", "O": "#2a78d6", "G": "#1baf7a", "L": "#eb6834"}
JUDGE = GC / "1_tables/task_judgment.csv"
BASKET = K / "basket/1_results/outcomes.csv"
FIXTURES = {"wooden_cabinet_1", "white_cabinet_1", "flat_stove_1", "microwave_1", "wine_rack_1", "desk_caddy_1"}
GAP_MERGE, MIN_HOLD, RELEASE_OPEN = 10, 5, 3
ZOOM_FEATS = ("expert", "patch_attention", "action")
TRAIN_HUE, CAND, BASE, OTHER = "#2a78d6", "#2a78d6", "#eb6834", "#1baf7a"
MUTED, INK, INK2, SURFACE, GRID = "#898781", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SPLIT_RATE, SPLIT_RUN = 0.8, 3  # split call: first call from which the own-task rate stays ≥ 0.8 for 3 calls
FEATS = {  # per call; identical to libero3 l3_analysis.py / cand31 attn_traj.py / basket_phase.py
    "expert": lambda d: np.log(np.concatenate([d["exp_mlp"].mean(1), d["exp_h"].mean(1)], 1)),
    "gemma": lambda d: np.log(np.concatenate([d["vlm_mlp"].reshape(len(d["vlm_mlp"]), -1),
                                              d["vlm_h"].reshape(len(d["vlm_h"]), -1)], 1)),
    "siglip": lambda d: np.log(d["sig_h"][:, :2].reshape(len(d["sig_h"]), -1)),
    "attention": lambda d: d["attn_mass"].mean(1).reshape(len(d["attn_mass"]), -1),
    "patch_attention": lambda d: np.sqrt(d["attn_img"][:, :, :512].astype(np.float32).reshape(len(d["attn_img"]), -1)),
    "action": lambda d: d["act_pred"].reshape(len(d["act_pred"]), -1),
}
NPZ_KEYS = ("attn_mass", "attn_img", "exp_mlp", "exp_h", "vlm_h", "vlm_mlp", "sig_h", "act_pred", "t")
TITLE = {"expert": "action expert activations", "gemma": "VLM (Gemma) activations", "siglip": "SigLIP activations",
         "attention": "attention shares (layer × input)", "patch_attention": "image-patch attention",
         "action": "predicted actions"}


# ---------------------------------------------------------------- helpers (T024 basket_phase.py)


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def shade(hue, s):
    """s in [0, 1]: 0 = pale (hue mixed 70% with white), 0.5 = the hue, 1 = the hue darkened 35%."""
    h = np.array(to_rgb(hue))
    if s < 0.5:
        w = 0.7 * (1 - s / 0.5)
        return tuple(h * (1 - w) + w)
    return tuple(h * (1 - 0.35 * (s - 0.5) / 0.5))


def dark(hue):
    return shade(hue, 1.0)


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def blank(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#e1e0d9")


# ---------------------------------------------------------------- events


def goal_targets(goal):
    """objects named in the goal predicates (all arguments, region suffix stripped: wooden_cabinet_1_top_region →
    wooden_cabinet_1; table regions skipped)."""
    out = []
    for p in goal:
        for a in p[1:]:
            m = re.match(r"^(.*?_\d+)", a)
            if m and m.group(1) not in out and not a.startswith("main_table"):
                out.append(m.group(1))
    return out


def events_from_steps(st, nc, targets=()):
    """grasp / release / contact steps of one episode from the recorded step data (see module doc)."""
    g, cl, ct = st["grasped"], st["gripper_closing"].astype(bool), st["contacts"]
    n = len(g)
    held = []
    for t in range(n):
        objs = [o for o in str(g[t]).split("|") if o and o not in FIXTURES]
        held.append(objs[0] if objs and cl[t] else "")
    two = [set(str(x).split("|")) for x in g]
    runs = []  # [start, end_inclusive, obj]
    for t, o in enumerate(held):
        if not o:
            continue
        if runs and runs[-1][2] == o and t - runs[-1][1] - 1 <= GAP_MERGE:
            runs[-1][1] = t
        elif runs and runs[-1][2] == o and t <= runs[-1][1]:
            continue
        else:
            runs.append([t, t, o])
        while runs[-1][1] + 1 < n and o in two[runs[-1][1] + 1]:  # held until the object leaves both fingers
            runs[-1][1] += 1
    runs = [r for r in runs if r[1] - r[0] + 1 >= MIN_HOLD]
    grasps, releases = [], []
    for s, e, o in runs:
        grasps.append((s, o))
        r = e + 1
        if r < n:
            kind = "open" if not cl[r:r + RELEASE_OPEN].all() else "slip"
            releases.append((r, o, kind))
    contact = next(((t, f) for t, c in enumerate(ct) for f in str(c).split("|") if f in targets), (None, None))
    on_target = contact[0] is not None
    if not on_target:  # target never touched: first contact with anything
        contact = next(((t, str(c).split("|")[0]) for t, c in enumerate(ct) if c), (None, None))
    fixture = next(((t, f) for t, c in enumerate(ct) for f in str(c).split("|") if f in FIXTURES), (None, None))
    call = lambda t: None if t is None else min(t // 10, nc - 1)  # noqa: E731
    gcs = [call(s) for s, _ in grasps]
    return dict(grasp_t=[s for s, _ in grasps], grasp_obj=[o for _, o in grasps], gcs=gcs,
                release_t=[r for r, _, _ in releases], release_obj=[o for _, o, _ in releases],
                release_kind=[k for _, _, k in releases], rcs=[call(r) for r, _, _ in releases],
                contact_t=contact[0], contact_obj=contact[1], cc=call(contact[0]), contact_on_target=on_target,
                fixture_t=fixture[0], fixture_obj=fixture[1],
                gc=gcs[0] if gcs else None,  # first grasp call: thin → thick
                bound=gcs[0] if gcs else (call(contact[0]) if contact[0] is not None else None))


def load_suite(tag, arrays=True):
    """trials of one suite, ordered: training tasks 1–10 × init 0–9, then (Goal) candidates g01–g31 × init 0–9."""
    groups, nc = SUITES[tag]
    judge = {f"{r['cand']}_{r['ep']}": r for r in csv.DictReader(open(JUDGE))} if tag == "G" else {}
    eps = []
    for grp, nd, jd in groups:
        rows = [json.loads(x) for x in (jd / "episodes.jsonl").read_text().splitlines()]
        rows.sort(key=lambda e: (e["variant"], int(e["init_idx"])))
        for e in rows:
            v, i = e["variant"], int(e["init_idx"])
            name = f"{v}_i{i:02d}"
            st = np.load(jd / e["steps_file"])
            rec = dict(suite=tag, group=grp, variant=v, init=i, name=name, instruction=e["instruction"],
                       success=bool(e["success"]), base=int(e["task_id"]) + 1, nc=nc)
            rec["label"] = f"{tag}{rec['base']}" if grp == "train" else v
            rec["targets"] = goal_targets(e["variant_meta"]["goal"])
            rec.update(events_from_steps(st, nc, rec["targets"]))
            if grp == "cand":
                j = judge[name]
                rec["old"] = json.loads(j["old"])
                rec["cat"] = j["cat"]
                rec["success"] = j["cat"] in ("new", "new_byproduct")  # new goal reached (judge)
            if arrays:
                z = np.load(nd / f"{name}.npz")
                assert len(z["t"]) == nc and (z["t"] == np.arange(nc) * 10).all(), name
                rec["d"] = {k: z[k] for k in NPZ_KEYS}
            eps.append(rec)
    return eps


def write_events(eps, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "group", "task", "init", "instruction", "success", "n_grasp", "grasp_t", "grasp_call",
                    "grasp_obj", "release_t", "release_call", "release_kind", "contact_t", "contact_call",
                    "contact_obj", "contact_is_goal_object", "goal_objects", "first_fixture_contact_t", "fixture", "executed_training_tasks", "judge_cat"])
        j = lambda xs: ";".join(str(x) for x in xs)  # noqa: E731
        for e in eps:
            w.writerow([e["suite"], e["group"], e["label"], e["init"], e["instruction"], int(e["success"]),
                        len(e["grasp_t"]), j(e["grasp_t"]), j(e["gcs"]), j(e["grasp_obj"]), j(e["release_t"]),
                        j(e["rcs"]), j(e["release_kind"]), e["contact_t"], e["cc"], e["contact_obj"], int(e["contact_on_target"]),
                        j(e["targets"]), e["fixture_t"],
                        e["fixture_obj"], j(e.get("old", [])), e.get("cat", "")])


def basket_check():
    """first grasp step of libero3 Object ctl vs basket/1_results/outcomes.csv (same episodes, T021 definition =
    first step with both fingers on any object, gripper state ignored)."""
    eps = load_suite("O", arrays=False)
    rows = {(int(r["task"]), int(r["init"])): r for r in csv.DictReader(open(BASKET)) if r["kind"] == "ctl"}
    res = dict(n=0, exact=0, same_call=0, within_10=0, diffs=[])
    for e in eps:
        r = rows[e["base"] - 1, e["init"]]
        b = None if r["grasp_t"] == "" else int(r["grasp_t"])
        m = e["grasp_t"][0] if e["grasp_t"] else None
        res["n"] += 1
        res["exact"] += b == m
        bc = None if b is None else min(b // 10, 27)
        res["same_call"] += bc == e["gc"]
        res["within_10"] += (b is None and m is None) or (b is not None and m is not None and abs(b - m) <= 10)
        if b != m:
            res["diffs"].append(dict(task=e["label"], init=e["init"], basket=b, ours=m))
        rb = None if r["release_t"] == "" else int(r["release_t"])
        mr = e["release_t"][0] if e["release_t"] else None
        res.setdefault("release_exact", 0)
        res.setdefault("release_same_call", 0)
        res["release_exact"] += rb == mr
        res["release_same_call"] += (None if rb is None else min(rb // 10, 27)) == (None if mr is None else min(mr // 10, 27))
        if rb != mr:
            res.setdefault("release_diffs", []).append(dict(task=e["label"], init=e["init"], basket=rb, ours=mr))
    return res


# ---------------------------------------------------------------- features and spaces


def raw_features(eps, f):
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.stack([FEATS[f](e["d"]) for e in eps]).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X  # (N, NC, D)


def zfit(flat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sd = np.nanmean(flat, 0), np.nanstd(flat, 0)
    keep = np.nan_to_num(sd) > 1e-6
    return lambda x: np.nan_to_num(((x - mu) / np.where(keep, sd, 1))[..., keep])  # NaN → 0 = the fit mean


def spaces(eps, tag, cache, recompute):
    """feat → dict(tsne (N, NC, 2), pca (N, NC, 3), pca_var [3], P50 (N, NC, ≤50) fixed-PCA coordinates)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    cache.mkdir(parents=True, exist_ok=True)
    tr = np.array([e["group"] == "train" for e in eps])
    S = {}
    for f in FEATS:
        p = cache / f"space_{tag}_{f}.npz"
        if p.exists() and not recompute:
            z = np.load(p)
            S[f] = {k: z[k] for k in z.files}
            continue
        X = raw_features(eps, f)
        N, nc, D = X.shape
        # A: fixed on the training trials
        za = zfit(X[tr].reshape(-1, D))
        Za = za(X)
        Dk = Za.shape[-1]
        pca = PCA(n_components=min(50, Dk), random_state=0).fit(Za[tr].reshape(-1, Dk))
        P = pca.transform(Za.reshape(-1, Dk)).reshape(N, nc, -1)
        # B: joint t-SNE of all calls of all trials
        zb = zfit(X.reshape(-1, D))
        Zb = zb(X).reshape(N * nc, -1)
        Pc = PCA(n_components=min(30, Zb.shape[1]), random_state=0).fit_transform(Zb)
        T = TSNE(n_components=2, perplexity=50, init="pca", learning_rate="auto", random_state=0,
                 n_jobs=8).fit_transform(Pc).reshape(N, nc, 2)
        S[f] = dict(tsne=T.astype(np.float32), pca=P[..., :3].astype(np.float32), P50=P.astype(np.float32),
                    pca_var=pca.explained_variance_ratio_[:3].astype(np.float32))
        np.savez(p, **S[f])
        print(f"[space] {tag} {f}: D={D} PC1-3 {np.round(S[f]['pca_var'] * 100, 1).tolist()}", flush=True)
    return S


# ---------------------------------------------------------------- drawing


def draw_trial(ax, xy, e, hue, thin=0.55, thick=1.7, z=3, ms=1.0, numbers=False, evnum=False):
    """one trial: segment c→c+1 pale → dark with time, thin while c < first grasp call, thick after; dotted if failed.
    ○ first call, × grasp (numbered when > 1), ■ release (□ = slip), ◆ first contact (episodes without a grasp)."""
    nc = len(xy)
    gc = e["gc"] if e["gc"] is not None else nc
    seg = np.stack([xy[:-1], xy[1:]], 1)
    cols = [shade(hue, c / (nc - 2)) for c in range(nc - 1)]
    lws = [thin if c < gc else thick for c in range(nc - 1)]
    ax.add_collection(LineCollection(seg, colors=cols, linewidths=lws, zorder=z, capstyle="round",
                                     linestyles="-" if e["success"] else (0, (1.2, 1.4))))
    ax.scatter(*xy[0], s=16 * ms, color=shade(hue, 0.1), edgecolor=dark(hue), lw=0.6, zorder=z + 1)
    many_g, many_r = len(e["gcs"]) > 1, len(e["rcs"]) > 1
    txt = dict(fontsize=4.6 * max(ms, 0.9), color=dark(hue), zorder=z + 3, fontweight="bold",
               path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
    for k, c in enumerate(e["gcs"]):
        ax.scatter(*xy[c], s=60 * ms, marker="x", color=dark(hue), lw=1.7, zorder=z + 2,
                   path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
        if many_g and evnum:
            ax.annotate(str(k + 1), xy[c], xytext=(3, 2), textcoords="offset points", **txt)
    for k, (c, kind) in enumerate(zip(e["rcs"], e["release_kind"], strict=True)):
        ax.scatter(*xy[c], s=24 * ms, marker="s", color=dark(hue) if kind == "open" else "white",
                   edgecolor=dark(hue) if kind == "slip" else "white", lw=0.8, zorder=z + 2)
        if many_r and evnum:
            ax.annotate(str(k + 1), xy[c], xytext=(3, -6), textcoords="offset points", **txt)
    if not e["gcs"] and e["cc"] is not None:
        ax.scatter(*xy[e["cc"]], s=34 * ms, marker="D", color=dark(hue), edgecolor="white", lw=0.7, zorder=z + 2)
    if numbers:
        for c in range(0, nc, 5):
            ax.text(*xy[c], str(c), fontsize=4.8, color=INK2, zorder=z + 3, ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])


def mean_episode(E):
    """markers for a mean trajectory: median call of the k-th grasp / release over trials that have one."""
    def med(key, k):
        xs = [e[key][k] for e in E if len(e[key]) > k]
        return int(round(np.median(xs))) if len(xs) * 2 >= len(E) else None  # at least half of the trials have it
    ng = max((len(e["gcs"]) for e in E), default=0)
    nr = max((len(e["rcs"]) for e in E), default=0)
    gcs = [c for c in (med("gcs", k) for k in range(ng)) if c is not None]
    rcs = [c for c in (med("rcs", k) for k in range(nr)) if c is not None]
    ccs = [e["cc"] for e in E if not e["gcs"] and e["cc"] is not None]
    return dict(gcs=gcs, rcs=rcs, release_kind=["open"] * len(rcs), gc=gcs[0] if gcs else None,
                cc=int(np.median(ccs)) if (not gcs and ccs) else None, success=True)


def draw_mean(ax, m, E, hue, z=6):
    me = mean_episode(E)
    ax.plot(m[:, 0], m[:, 1], color=dark(hue), lw=3.0, zorder=z, solid_capstyle="round",
            path_effects=[pe.withStroke(linewidth=4.8, foreground="white")])
    draw_trial(ax, m, me, hue, thin=0, thick=0, z=z + 1, ms=1.5, evnum=True)
    for c in range(0, len(m), 5):
        ax.text(*m[c], str(c), fontsize=7, fontweight="bold", color=INK, zorder=z + 4, ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])


def legend_handles(extra=()):
    h = list(extra)
    h += [plt.Line2D([], [], color=INK, lw=0.6, label="before the first grasp (reach)"),
          plt.Line2D([], [], color=INK, lw=1.8, label="after the first grasp"),
          plt.Line2D([], [], color=INK, lw=1, ls=":", label="failed episode"),
          plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=5, label="first call (pale end; dark = late)"),
          plt.Line2D([], [], ls="", marker="x", color=INK, ms=8, mew=1.9, label="grasp (number = 1st, 2nd …)"),
          plt.Line2D([], [], ls="", marker="s", color=INK, ms=5, label="release, gripper opened"),
          plt.Line2D([], [], ls="", marker="s", mfc="white", mec=INK, ms=5, label="release, slipped while closed"),
          plt.Line2D([], [], ls="", marker="D", color=INK, ms=5, label="first contact (episodes without a grasp)")]
    return h


def emb_info(S, f, emb):
    if emb == "tsne":
        return S[f]["tsne"], None
    v = [round(float(x) * 100, 1) for x in S[f]["pca_var"]]
    return S[f]["pca"][..., :2], v


def limits(Y, q=(0.3, 99.7), pad=0.04):
    pts = Y.reshape(-1, 2)
    lo, hi = np.percentile(pts, q[0], 0), np.percentile(pts, q[1], 0)
    p = pad * (hi - lo)
    return (lo[0] - p[0], hi[0] + p[0]), (lo[1] - p[1], hi[1] + p[1])


def axis_labels(ax, v, fit):
    if v is None:
        return
    ax.set_xlabel(f"PC1 ({v[0]}% of {fit} variance)", fontsize=6.5, labelpad=1)
    ax.set_ylabel(f"PC2 ({v[1]}%)", fontsize=6.5, labelpad=1)


def segs(Y):
    return np.concatenate([np.stack([y[:-1], y[1:]], 1) for y in Y]) if len(Y) else np.zeros((0, 2, 2))


def fig_overview(job):
    """(a) one panel per training task (and, for Goal, one per candidate), all panels in the suite's shared space."""
    tag, f, emb, eps, S, out = job
    Y, v = emb_info(S, f, emb)
    xl, yl = limits(Y)
    fit = "train10" if tag == "G" else f"{SUITE_NAME[tag]} training-trial"
    tr = [j for j, e in enumerate(eps) if e["group"] == "train"]
    ename = "joint t-SNE of all calls of the suite" if emb == "tsne" else f"fixed PCA (fit on the {fit} calls)"
    allseg = segs(Y)
    seg_trial = np.repeat(np.arange(len(eps)), Y.shape[1] - 1)

    def panel(ax, idx_groups, title):
        if v is None:
            blank(ax)
        else:
            ax.set_facecolor(SURFACE)
            clean(ax)
        ax.add_collection(LineCollection(allseg[~np.isin(seg_trial, [j for g in idx_groups for j in g[0]])],
                                         colors=MUTED, linewidths=0.3, alpha=0.06, zorder=1))
        for k, (idx, hue, thin, thick) in enumerate(idx_groups):
            for j in idx:
                draw_trial(ax, Y[j], eps[j], hue, thin=thin, thick=thick, ms=0.55, z=3 + 3 * k, evnum=True)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_title(title, fontsize=7.2, loc="left")
        axis_labels(ax, v, fit)

    d = out / "a_overview"
    d.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.8), facecolor=SURFACE)
    for t, ax in enumerate(axes.flat, 1):
        idx = [j for j in tr if eps[j]["base"] == t]
        gc = [eps[j]["gc"] for j in idx if eps[j]["gc"] is not None]
        sub = (f"first grasp call med {np.median(gc):g}" if gc else "no grasp (◆ contact)") + \
              f" · success {sum(eps[j]['success'] for j in idx)}/10"
        panel(ax, [(idx, TRAIN_HUE, 0.45, 1.25)], textwrap.shorten(f"{tag}{t}: {eps[idx[0]]['instruction']}", 62) +
              f"\n{sub}")
    fig.legend(handles=legend_handles([plt.Line2D([], [], color=TRAIN_HUE, lw=2, label="this task's 10 trials")]),
               loc="lower center", ncol=5, fontsize=7.5, frameon=False)
    fig.suptitle(f"{SUITE_NAME[tag]} · {TITLE[f]} · {ename}; one line per trial, pale → dark with time "
                 f"(gray = the suite's other trials)", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.11, top=0.92, wspace=0.12 if v is not None else 0.04,
                        hspace=0.3 if v is not None else 0.18)
    fig.savefig(d / f"{f}_{SUITE_NAME[tag]}_{emb}.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)
    if tag != "G":
        return
    cands = sorted({e["variant"] for e in eps if e["group"] == "cand"})
    fig, axes = plt.subplots(5, 7, figsize=(22, 17), facecolor=SURFACE)
    for ax, c in zip(axes.flat, cands, strict=False):
        ci = [j for j, e in enumerate(eps) if e["variant"] == c]
        base, other, moved = cand_context(eps, ci)
        bi = [j for j in tr if eps[j]["base"] == base]
        oi = [j for j in tr if eps[j]["base"] == other] if other else []
        new = sum(eps[j]["cat"] == "new" for j in ci)
        sub = f"base G{base}" + (f" · {moved}/10 did G{other}" if other else "") + f" · new {new}/10"
        panel(ax, [(bi, BASE, 0.3, 0.8), (oi, OTHER, 0.3, 0.8), (ci, CAND, 0.5, 1.3)],
              textwrap.fill(f"{c}: {eps[ci[0]]['instruction']}", 46) + f"\n{sub}")
    for ax in list(axes.flat)[len(cands):]:
        ax.axis("off")
    h = legend_handles([plt.Line2D([], [], color=CAND, lw=2, label="candidate's trials"),
                        plt.Line2D([], [], color=BASE, lw=2, label="its base training task"),
                        plt.Line2D([], [], color=OTHER, lw=2, label="training task done instead (≥3 of 10)")])
    axes.flat[len(cands)].legend(handles=h, loc="center left", fontsize=8.5, frameon=False)
    fig.suptitle(f"Goal + cand31 · {TITLE[f]} · {ename} (same space as the training-task figure); "
                 f"one line per trial, pale → dark with time", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.12 if v is not None else 0.05,
                        hspace=0.42)
    fig.savefig(d / f"{f}_{SUITE_NAME[tag]}_{emb}_cand.png", dpi=100, facecolor=SURFACE)
    plt.close(fig)


def cand_context(eps, ci):
    base = eps[ci[0]]["base"]
    moved = Counter(t for j in ci for t in eps[j]["old"] if t != base)
    top = moved.most_common(1)
    other = top[0][0] if top and top[0][1] >= 3 else None
    return base, other, (top[0][1] if other else 0)


def fig_zoom(job):
    """(b) one page per task (Goal: also per candidate): columns = trial sets, rows = t-SNE / fixed PCA; call numbers
    every 5, set mean trajectory bold, the suite's other trials gray; zoomed to the drawn trials."""
    tag, f, eps, S, out, key = job
    tr = [j for j, e in enumerate(eps) if e["group"] == "train"]
    if key.startswith("g"):
        ci = [j for j, e in enumerate(eps) if e["variant"] == key]
        base, other, moved = cand_context(eps, ci)
        sets = [(ci, CAND, f"{key} (candidate): {eps[ci[0]]['instruction']}"),
                ([j for j in tr if eps[j]["base"] == base], BASE, f"base G{base}: "
                 f"{eps[[j for j in tr if eps[j]['base'] == base][0]]['instruction']}")]
        if other:
            oi = [j for j in tr if eps[j]["base"] == other]
            sets.append((oi, OTHER, f"done instead by {moved}/10: G{other}: {eps[oi[0]]['instruction']}"))
    else:
        t = int(key[1:])
        idx = [j for j in tr if eps[j]["base"] == t]
        sets = [(idx, TRAIN_HUE, f"{key}: {eps[idx[0]]['instruction']}")]
    ncol = len(sets)
    if ncol == 1:  # one training task: t-SNE | PCA side by side
        fig, ax_ = plt.subplots(1, 2, figsize=(17, 8.6), facecolor=SURFACE)
        axes = np.array([[ax_[0]], [ax_[1]]])
    else:
        fig, axes = plt.subplots(2, ncol, figsize=(7.2 * ncol + 0.6, 14), facecolor=SURFACE, squeeze=False)
    drawn = [j for s in sets for j in s[0]]
    for r, emb in enumerate(("tsne", "pca")):
        Y, v = emb_info(S, f, emb)
        sel = Y[drawn]
        lo, hi = sel.reshape(-1, 2).min(0), sel.reshape(-1, 2).max(0)
        pad = 0.08 * (hi - lo)
        for c, (idx, hue, title) in enumerate(sets):
            ax = axes[r, c]
            if v is None:
                blank(ax)
            else:
                ax.set_facecolor(SURFACE)
                clean(ax)
            rest = [j for j in range(len(eps)) if j not in idx and (eps[j]["group"] == "train" or key.startswith("g"))]
            rest = [j for j in rest if eps[j]["group"] == "train" or eps[j]["variant"] == key]
            ax.add_collection(LineCollection(segs(Y[rest]), colors="#b9b7b0", linewidths=0.5, alpha=0.25, zorder=1))
            for j in idx:
                draw_trial(ax, Y[j], eps[j], hue, thin=0.6, thick=1.6, numbers=True, evnum=True)
            draw_mean(ax, Y[idx].mean(0), [eps[j] for j in idx], hue)
            E_ = [eps[j] for j in idx]
            g1 = [e["gc"] for e in E_ if e["gc"] is not None]
            ng = Counter(len(e["gcs"]) for e in E_)
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            sub = (f"success {sum(e['success'] for e in E_)}/{len(E_)} · first grasp call "
                   + (f"median {np.median(g1):g} ({min(g1)}–{max(g1)})" if g1 else "– (no grasp)")
                   + " · grasps/episode " + ", ".join(f"{k}×{n}" for k, n in sorted(ng.items())))
            space = "joint t-SNE (B)" if emb == "tsne" else "fixed PCA (A)"
            ax.set_title(f"[{space}] " + textwrap.fill(title, 70) + f"\n{sub}", fontsize=8.2, loc="left")
            if v is not None:
                axis_labels(ax, v, "train10" if tag == "G" else "training-trial")
    h = legend_handles([plt.Line2D([], [], color=INK, lw=3, label="mean trajectory (2D mean per call; markers = "
                                   "median event calls)"),
                        plt.Line2D([], [], color="#b9b7b0", lw=1, label="other training tasks of the suite")])
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"{SUITE_NAME[tag]} · {key} · {TITLE[f]} · numbers = policy call (×10 steps); zoomed on the drawn "
                 f"trials (same embeddings as the overview)", fontsize=10.5)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.12 if ncol == 1 else 0.07, top=0.88 if ncol == 1 else 0.92,
                        wspace=0.12, hspace=0.2)
    d = out / "b_zoom" / f / SUITE_NAME[tag]
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{key}.png", dpi=95, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- numbers (step 4)


def dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return dict(n=0)
    q = np.percentile(xs, [25, 50, 75])
    return dict(n=len(xs), median=float(q[1]), q25=float(q[0]), q75=float(q[2]), min=int(min(xs)), max=int(max(xs)))


def fmt_dist(d):
    return "–" if not d.get("n") else f"{d['median']:g} [{d['q25']:g}–{d['q75']:g}] ({d['min']}–{d['max']}, n={d['n']})"


def nearest_task(eps, P):
    """per call, nearest training-task centroid (same call, fixed-PCA-50 coordinates; own trial left out).
    Returns (N, NC) training task numbers."""
    tr = np.array([e["group"] == "train" for e in eps])
    base = np.array([e["base"] for e in eps])
    tasks = sorted(set(base[tr]))
    sums = {t: P[tr & (base == t)].sum(0) for t in tasks}  # (NC, K)
    cnt = {t: (tr & (base == t)).sum() for t in tasks}
    N, nc, _ = P.shape
    out = np.zeros((N, nc), int)
    for j in range(N):
        C = np.stack([(sums[t] - (P[j] if (tr[j] and base[j] == t) else 0)) / (cnt[t] - (tr[j] and base[j] == t))
                      for t in tasks])  # (T, NC, K)
        out[j] = np.array(tasks)[np.argmin(((C - P[j][None]) ** 2).sum(-1), 0)]
    return out


def phase_rates(eps, near, target):
    """mean over trials of the share of calls whose nearest centroid == target(e), before (calls ≤ boundary) and after.
    boundary = first grasp call; contact call for episodes without a grasp; episodes with neither: all 'before'."""
    pre, post = [], []
    for j, e in enumerate(eps):
        tg = target(e)
        if tg is None:
            continue
        hit = np.isin(near[j], tg)
        b = e["bound"] if e["bound"] is not None else len(hit) - 1
        pre.append(hit[:b + 1].mean())
        if b + 1 < len(hit):
            post.append(hit[b + 1:].mean())
    return [r3(np.mean(pre)) if pre else None, len(pre)], [r3(np.mean(post)) if post else None, len(post)]


def split_call(rate):
    """first call from which the rate stays ≥ SPLIT_RATE for SPLIT_RUN consecutive calls."""
    ok = rate >= SPLIT_RATE
    for c in range(len(ok) - SPLIT_RUN + 1):
        if ok[c:c + SPLIT_RUN].all():
            return c
    return None


def numbers(tag, eps, S, out):
    tr = [j for j, e in enumerate(eps) if e["group"] == "train"]
    T = [e for e in eps if e["group"] == "train"]
    res = dict(suite=SUITE_NAME[tag], n_trials=len(eps), calls=eps[0]["nc"])
    res["events"] = dict(
        first_grasp_call=dist([e["gc"] for e in T]), first_release_call=dist([e["rcs"][0] if e["rcs"] else None for e in T]),
        last_release_call=dist([e["rcs"][-1] if e["rcs"] else None for e in T]),
        grasps_per_episode=dict(Counter(len(e["gcs"]) for e in T)),
        slip_releases=sum(k == "slip" for e in T for k in e["release_kind"]),
        open_releases=sum(k == "open" for e in T for k in e["release_kind"]),
        no_grasp_trials={e["name"]: e["contact_obj"] for e in T if not e["gcs"]},
        per_task={})
    for t in range(1, 11):
        E = [e for e in T if e["base"] == t]
        res["events"]["per_task"][f"{tag}{t}"] = dict(
            instruction=E[0]["instruction"], success=sum(e["success"] for e in E),
            first_grasp_call=dist([e["gc"] for e in E]), second_grasp_call=dist([e["gcs"][1] if len(e["gcs"]) > 1 else None for e in E]),
            first_release_call=dist([e["rcs"][0] if e["rcs"] else None for e in E]),
            grasps_per_episode=dict(Counter(len(e["gcs"]) for e in E)),
            grasped_objects=dict(Counter(o for e in E for o in e["grasp_obj"])),
            no_grasp=sum(not e["gcs"] for e in E), contact_call_no_grasp=dist([e["cc"] for e in E if not e["gcs"]]),
            first_contact_objects_no_grasp=dict(Counter(e["contact_obj"] for e in E if not e["gcs"])))
    if tag == "G":
        C = [e for e in eps if e["group"] == "cand"]
        res["events"]["cand"] = dict(first_grasp_call=dist([e["gc"] for e in C]),
                                     first_release_call=dist([e["rcs"][0] if e["rcs"] else None for e in C]),
                                     grasps_per_episode=dict(Counter(len(e["gcs"]) for e in C)),
                                     no_grasp=sum(not e["gcs"] for e in C))
    res["identity"] = {}
    for f in FEATS:
        near = nearest_task(eps, S[f]["P50"])
        own = near[tr] == np.array([eps[j]["base"] for j in tr])[:, None]  # (100, NC)
        rate = own.mean(0)
        pre, post = phase_rates([eps[j] for j in tr], near[tr], lambda e: [e["base"]])
        per_task = {}
        for t in range(1, 11):
            m = np.array([eps[j]["base"] == t for j in tr])
            sc = split_call(own[m].mean(0))
            g1 = [eps[j]["gc"] for j in np.array(tr)[m] if eps[j]["gc"] is not None]
            b1 = [eps[j]["bound"] for j in np.array(tr)[m] if eps[j]["bound"] is not None]
            per_task[f"{tag}{t}"] = dict(split_call=sc, first_grasp_call_median=float(np.median(g1)) if g1 else None,
                                         boundary_median=float(np.median(b1)) if b1 else None,
                                         split_minus_boundary=None if sc is None or not b1 else float(sc - np.median(b1)),
                                         pre=phase_rates([eps[j] for j in np.array(tr)[m]], near[tr][m], lambda e: [e["base"]])[0][0],
                                         post=phase_rates([eps[j] for j in np.array(tr)[m]], near[tr][m], lambda e: [e["base"]])[1][0])
        sc = split_call(rate)
        r = dict(own_rate_by_call=[r3(x) for x in rate], suite_split_call=sc,
                 suite_merge_call=None if sc is None else next(
                     (c for c in range(sc, len(rate) - SPLIT_RUN + 1) if (rate[c:c + SPLIT_RUN] < SPLIT_RATE).all()), None),
                 first_call_rate=r3(rate[0]), pre=pre, post=post, per_task=per_task)
        if tag == "G":
            ci = [j for j, e in enumerate(eps) if e["group"] == "cand"]
            CE = [eps[j] for j in ci]
            nb = phase_rates(CE, near[ci], lambda e: [e["base"]])
            nx = phase_rates(CE, near[ci], lambda e: [t for t in e["old"] if t != e["base"]] or None)
            r["cand"] = dict(nearest_is_base=dict(pre=nb[0], post=nb[1]),
                             nearest_is_executed_other=dict(pre=nx[0], post=nx[1]),
                             base_rate_by_call=[r3(x) for x in (near[ci] == np.array([e["base"] for e in CE])[:, None]).mean(0)])
        res["identity"][f] = r
        print(f"[numbers] {tag} {f}: split {r['suite_split_call']} pre {pre} post {post}", flush=True)
    res["space_A_pca_var_pct"] = {f: [round(float(x) * 100, 1) for x in S[f]["pca_var"]] for f in FEATS}
    (out / f"numbers_{SUITE_NAME[tag]}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    tables(tag, res, out)
    fig_identity(tag, res, eps, out)
    return res


def tables(tag, res, out):
    ev = res["events"]
    Lm = [f"## {SUITE_NAME[tag]} ({res['calls']} calls/episode)", "",
          "| task | 문장 | 성공 | 첫 grasp call 중앙값 [IQR] (범위) | 2번째 grasp | 첫 release call | grasp 횟수 (횟수×에피소드) | grasp 없음 |",
          "|---|---|---|---|---|---|---|---|"]
    for t, d in ev["per_task"].items():
        Lm.append(f"| {t} | {d['instruction']} | {d['success']}/10 | {fmt_dist(d['first_grasp_call'])} | "
                  f"{fmt_dist(d['second_grasp_call'])} | {fmt_dist(d['first_release_call'])} | "
                  + ", ".join(f"{k}×{n}" for k, n in sorted(d['grasps_per_episode'].items())) + f" | {d['no_grasp']} |")
    Lm.append(f"| 전체 | | | {fmt_dist(ev['first_grasp_call'])} | | {fmt_dist(ev['first_release_call'])} | "
              + ", ".join(f"{k}×{n}" for k, n in sorted(ev['grasps_per_episode'].items())) + f" | {len(ev['no_grasp_trials'])} |")
    Lm += ["", "| 특징 | 첫 call 자기 task 비율 | 갈라지는 call (suite) | 다시 섞이는 call | grasp 전 | grasp 후 | task별 갈라지는 call − 첫 grasp call |"
           + (" 후보: 가장 가까운 = 바탕 task (전/후) | 후보: = 대신 한 task (전/후) |" if tag == "G" else ""),
           "|---|---|---|---|---|---|---|" + ("---|---|" if tag == "G" else "")]
    for f, r in res["identity"].items():
        sm = ", ".join(f"{t} {'–' if d['split_minus_boundary'] is None else f'{d['split_minus_boundary']:+g}'}"
                       for t, d in r["per_task"].items())
        row = (f"| {f} | {r['first_call_rate']} | {r['suite_split_call']} | {r['suite_merge_call']} | {r['pre'][0]} | {r['post'][0]} | {sm} |")
        if tag == "G":
            c = r["cand"]
            row += (f" {c['nearest_is_base']['pre'][0]} / {c['nearest_is_base']['post'][0]} |"
                    f" {c['nearest_is_executed_other']['pre'][0]} / {c['nearest_is_executed_other']['post'][0]}"
                    f" (n={c['nearest_is_executed_other']['pre'][1]}) |")
        Lm.append(row)
    (out / f"tables_{SUITE_NAME[tag]}.md").write_text("\n".join(Lm) + "\n")


def fig_identity(tag, res, eps, out):
    """own-task nearest-centroid rate per call (training trials) with the first-grasp call distribution below."""
    T = [e for e in eps if e["group"] == "train"]
    nc = eps[0]["nc"]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 5.6), facecolor=SURFACE, sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1.2]))
    hues = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#6250d6", "#898781"]
    for f, h in zip(FEATS, hues, strict=True):
        r = res["identity"][f]
        ax.plot(np.arange(nc), [np.nan if x is None else x for x in r["own_rate_by_call"]], color=h, lw=1.6, label=f)
        if tag == "G":
            ax.plot(np.arange(nc), r["cand"]["base_rate_by_call"], color=h, lw=1, ls="--")
    ax.axhline(0.1, color=MUTED, lw=0.8, ls=":")
    ax.axhline(SPLIT_RATE, color=MUTED, lw=0.8, ls="--")
    ax.set_ylabel("nearest centroid = own task", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.set_title(f"{SUITE_NAME[tag]}: per call, nearest training-task centroid (fixed PCA-50, leave-one-out) is the "
                 f"trial's own task" + (" (dashed = candidates: nearest is their base task)" if tag == "G" else ""),
                 fontsize=8.5, loc="left")
    clean(ax)
    g = [e["gc"] for e in T if e["gc"] is not None]
    r_ = [e["rcs"][0] for e in T if e["rcs"]]
    bx.hist([g, r_], bins=np.arange(nc + 1) - 0.5, color=["#0b0b0b", "#b9b7b0"], label=["first grasp", "first release"])
    bx.legend(fontsize=7, frameon=False)
    bx.set_xlabel("policy call (×10 steps)", fontsize=8)
    bx.set_ylabel("trials", fontsize=8)
    clean(bx)
    fig.tight_layout()
    fig.savefig(out / f"identity_{SUITE_NAME[tag]}.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- 40-task summary


def task40(all_eps, out, recompute):
    """PCA fitted on all calls of the 400 training trials of the four suites; task mean trajectories."""
    from sklearn.decomposition import PCA

    eps = [e for tag in "SOGL" for e in all_eps[tag] if e["group"] == "train"]
    d = out / "a_overview"
    d.mkdir(parents=True, exist_ok=True)
    tasks = list(dict.fromkeys(e["label"] for e in eps))
    res = {}
    for f in FEATS:
        Xs = [np.nan_to_num(FEATS[f](e["d"]).astype(np.float32), nan=np.nan, posinf=np.nan, neginf=np.nan) for e in eps]
        z = zfit(np.concatenate(Xs))
        Z = [z(x) for x in Xs]
        pca = PCA(n_components=3, random_state=0).fit(np.concatenate(Z))
        P = [pca.transform(x) for x in Z]
        v = [round(float(x) * 100, 1) for x in pca.explained_variance_ratio_]
        res[f] = v
        M = {u: np.mean([P[j] for j, e in enumerate(eps) if e["label"] == u], 0) for u in tasks}
        E = {u: [e for e in eps if e["label"] == u] for u in tasks}
        allp = np.concatenate(P)
        xl, yl = limits(allp[None, :, :2])
        fig, axes = plt.subplots(1, 5, figsize=(26, 6), facecolor=SURFACE)
        for k, ax in enumerate(axes):
            ax.set_facecolor(SURFACE)
            clean(ax)
            ax.scatter(allp[:, 0], allp[:, 1], s=1, color="#d5d3cc", alpha=0.25, lw=0, zorder=0)
            show = "SOGL" if k == 0 else "SOGL"[k - 1]
            for u in tasks:
                if u[0] not in show:
                    ax.plot(M[u][:, 0], M[u][:, 1], color="#b9b7b0", lw=0.7, alpha=0.6, zorder=1)
                    continue
                draw_trial(ax, M[u][:, :2], mean_episode(E[u]), SUITE_HUE[u[0]], thin=0.8, thick=2.0, ms=0.7,
                           numbers=k > 0, evnum=k > 0)
                ax.text(*M[u][-1, :2], f" {u}", fontsize=6.5 if k else 5.5, color=dark(SUITE_HUE[u[0]]), zorder=9,
                        fontweight="bold", path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_xlabel(f"PC1 ({v[0]}% of the 400 training trials' variance)", fontsize=7)
            ax.set_ylabel(f"PC2 ({v[1]}%)", fontsize=7)
            ax.set_title("all four suites" if k == 0 else f"{SUITE_NAME[show]} (others gray)", fontsize=9, loc="left")
        h = legend_handles([plt.Line2D([], [], color=SUITE_HUE[s], lw=2, label=f"{SUITE_NAME[s]}: task mean (10 inits)")
                            for s in "SOGL"])
        fig.legend(handles=h, loc="lower center", ncol=6, fontsize=7.5, frameon=False)
        fig.suptitle(f"40 training tasks · {TITLE[f]} · z-score + PCA fitted on all calls of the 400 training trials; "
                     f"one line per task = mean of its 10 trials per call (Long has 52 calls, Goal 30, Spatial/Object 28); "
                     f"markers = median event calls", fontsize=10)
        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.2, top=0.9, wspace=0.18)
        fig.savefig(d / f"{f}_task40_pca.png", dpi=100, facecolor=SURFACE)
        plt.close(fig)
        print(f"[task40] {f} {v}", flush=True)
    return res


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="SOLG")
    ap.add_argument("--parts", default="eabns4", help="e events, s spaces, a overview, b zoom, n numbers, 4 task40")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    all_eps, summary = {}, {}
    for tag in a.suites:
        out = OUT[tag]
        out.mkdir(parents=True, exist_ok=True)
        eps = load_suite(tag)
        all_eps[tag] = eps
        print(f"[load] {tag}: {len(eps)} trials", flush=True)
        if "e" in a.parts:
            write_events(eps, out / "events" / f"events_{SUITE_NAME[tag]}.csv")
        if not set(a.parts) & set("sabn"):
            continue
        S = spaces(eps, tag, out / "_cache", a.recompute)
        if "n" in a.parts:
            summary[tag] = numbers(tag, eps, S, out)
        lite = [{k: v for k, v in e.items() if k != "d"} for e in eps]
        jobs = []
        if "a" in a.parts:
            jobs += [(fig_overview, (tag, f, emb, lite, S, out)) for f in FEATS for emb in ("tsne", "pca")]
        if "b" in a.parts:
            keys = [f"{tag}{t}" for t in range(1, 11)] + sorted({e["variant"] for e in eps if e["group"] == "cand"})
            jobs += [(fig_zoom, (tag, f, lite, S, out, k)) for f in ZOOM_FEATS for k in keys]
        with get_context("fork").Pool(a.workers) as pool:
            for _ in pool.imap_unordered(_run, jobs, chunksize=2):
                pass
        print(f"[figs] {tag}: {len(jobs)}", flush=True)
    if "e" in a.parts and "O" in a.suites:
        bc = basket_check()
        (OUT["O"] / "events" / "basket_check.json").write_text(json.dumps(bc, indent=1))
        print(f"[basket] grasp exact {bc['exact']}/{bc['n']}, same call {bc['same_call']}/{bc['n']}, ±10 steps "
              f"{bc['within_10']}/{bc['n']}; release exact {bc['release_exact']}, same call {bc['release_same_call']}",
              flush=True)
    if "4" in a.parts:
        for tag in "SOGL":
            if tag not in all_eps:
                all_eps[tag] = load_suite(tag)
        v = task40(all_eps, OUT["T40"], a.recompute)
        (OUT["T40"] / "task40_pca_var.json").write_text(json.dumps(v, indent=1))


def _run(job):
    fn, args = job
    os.environ["OMP_NUM_THREADS"] = "1"
    fn(args)


if __name__ == "__main__":
    main()
