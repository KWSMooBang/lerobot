"""T026: which properties of the training-task trajectories separate LIBERO-Goal composition candidates (cand31) that
succeed at the new task from those that fall back / fail. CPU only, no re-extraction.

Inputs (all A100 / ferrari, pi05 lerobot/pi05_libero_finetuned, Goal train10 + cand31, 410 trials × 30 calls):
  8_phase_traj/_cache/space_G_<feat>.npz   T025 fixed PCA-50 (z-score + PCA fitted on train10 calls only), joint t-SNE
  8_phase_traj/events/events_Goal.csv      grasp / release / contact calls (same trial order as the cache)
  1_tables/task_judgment.csv               per-trial outcome of the candidates
  _raw/attn/out/*/<trial>.npz              attn_img (A5 patch-location maps only)
Feature definitions, roles and statistics: see composition_traits_prereg.md (fixed before the results).

Run: OPENBLAS_CORETYPE=ZEN $W/cand31/.venv-viz/bin/python composition_traits.py [--parts nfs]
  n = numbers + csv, f = per-candidate figures, s = summary figures
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")  # task rule: ≤ 8 CPU workers

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

GC = Path("/outputs/VLA-shortcut-exps/kubectl/libero_goal_composition")
PT = GC / "8_phase_traj"
OUT = GC / "9_composition_traits"
EVENTS = PT / "events/events_Goal.csv"
JUDGE = GC / "1_tables/task_judgment.csv"
NPZ = {"train": GC / "_raw/attn/out/ours-goal-custom-train10", "cand": GC / "_raw/attn/out/ours-goal-custom-cand31"}
FEATS = ("patch_attention", "expert", "gemma", "siglip", "attention", "action")
NC = 30
SUCC_MIN = 3  # success label: ≥ 3/10 new-task trials
EXCLUDED = ("g18", "g28")  # "pick up": goal = height ≥ 1 m, reached as a by-product
INFEASIBLE = ("g01", "g22")  # scripted-policy feasibility check (task spec); g13 flagged in the cand31 table too
NPERM, NBOOT = 10000, 10000
RNG = np.random.default_rng(0)

# palette (dataviz reference, first three categorical slots validate all-pairs)
C_A, C_B, C_BASE, C_OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e4e3df"

# ---------------------------------------------------------------- roles (hand-made from cand31_results.md "바꾼 부분")
# A = component source 1 (object / "what", or the first task of a chain), B = component source 2 (destination /
# action / second task of a chain). Segment roles: P pick→release, L place (release−2..release), H handle/contact.
# type: obj_swap (new object into a trained destination), dst_swap, attr (drawer / referent), chain, novel_action.
R = [
    # cand, type, base, obj_src, dst_src, act_src, A, rA, B, rB, ambiguous note
    ("g01", "novel_referent", "1", "", "", "1", "", "", "", "", "bottom drawer never in training → no 2nd source"),
    ("g02", "attr", "1", "4", "", "1", "4", "H", "1", "H", "object = top drawer (G4 drawer phase), action = open (G1)"),
    ("g03", "attr", "4", "4", "4", "1", "1", "H", "4", "P",
     "ambiguous: middle drawer from G1 + bowl-into-drawer from G4; pair = G1 handle → G4 bowl carry"),
    ("g04", "attr+obj_swap", "4", "3;10", "4", "1", "3;10", "P", "4", "L",
     "ambiguous: three sources (drawer G1, object G3/G10, into-drawer G4); pair = object carry → G4 place"),
    ("g05", "attr+obj_swap", "4", "7", "4", "1", "7", "P", "4", "L",
     "ambiguous: three sources (drawer G1, object G7, into-drawer G4); pair = object carry → G4 place"),
    ("g06", "novel_action", "4", "4", "", "", "", "", "", "", "close never in training (starts with drawer open)"),
    ("g07", "chain", "1", "", "", "1;8", "1", "H", "8", "H", "chain G1 → G8"),
    ("g08", "obj_swap", "2", "3;10", "2", "", "3;10", "P", "2", "L", "base listed 2,3,10; table frames it as G2 + bottle"),
    ("g09", "dst_swap", "10", "2;5;9", "10", "", "2;5;9", "P", "10", "L", "base listed 2,5,9,10"),
    ("g10", "obj_swap", "2", "7", "2", "", "7", "P", "2", "L", ""),
    ("g11", "dst_swap", "2;5;9", "2;5;9", "6", "", "2;5;9", "P", "6", "H",
     "destination 'front of stove' only reached by pushing (G6 H segment)"),
    ("g12", "obj_swap", "2", "6", "2", "", "6", "H", "2", "L", "plate is never grasped in training (G6 pushes it)"),
    ("g13", "chain", "2;10", "", "", "10;2", "10", "L", "2", "P", "chain G10 → G2; cand31 table: script-infeasible"),
    ("g14", "obj_swap", "7", "3;10", "7", "", "3;10", "P", "7", "L", "base listed 3,7,10"),
    ("g15", "obj_swap", "9", "3;10", "9", "", "3;10", "P", "9", "L", "base listed 3,9,10"),
    ("g16", "obj_swap", "3;5", "7", "3;5", "", "7", "P", "3;5", "L", "base 3 or 5 (both 'top of cabinet')"),
    ("g17", "dst_swap", "3;10", "3;10", "6", "", "3;10", "P", "6", "H", "destination only pushed-to in G6"),
    ("g18", "novel_action", "3;10", "3;10", "", "", "", "", "", "", "excluded (pick up: height goal)"),
    ("g19", "obj_swap", "3;5", "6", "3;5", "", "6", "H", "3;5", "L", "plate never grasped in training"),
    ("g20", "obj_swap", "4", "3;10", "4", "", "3;10", "P", "4", "L", ""),
    ("g21", "obj_swap", "4", "7", "4", "", "7", "P", "4", "L", ""),
    ("g22", "novel_referent", "4", "", "", "4", "", "", "", "", "bottom drawer never in training → no 2nd source"),
    ("g23", "obj_swap", "6", "2;5;9", "6", "6", "2;5;9", "P", "6", "H", "push a bowl: bowl only ever grasped in training"),
    ("g24", "obj_swap", "6", "7", "6", "6", "7", "P", "6", "H", "push cream cheese: only ever grasped in training"),
    ("g25", "dst_swap", "7", "7", "6", "", "7", "P", "6", "H", "base listed 6,7"),
    ("g26", "dst_swap", "7", "7", "9", "", "7", "P", "9", "L", "base 7 or 9 (table gives both readings)"),
    ("g27", "dst_swap", "7", "7", "10", "", "7", "P", "10", "L", "base listed 7,10"),
    ("g28", "novel_action", "7", "7", "", "", "", "", "", "", "excluded (pick up: height goal)"),
    ("g29", "chain", "7;9", "", "", "7;9", "7", "L", "9", "P", "chain G7 → G9 (bowl now holds the cheese)"),
    ("g30", "novel_action", "8", "", "", "8", "", "", "", "", "turn off never in training (starts on)"),
    ("g31", "chain", "8;9", "", "", "9;8", "9", "L", "8", "H", "chain G9 → G8"),
]
TL = lambda s: [int(x) for x in s.split(";") if x]  # noqa: E731


def roles():
    out = {}
    for c, typ, base, o, d, a, A, rA, B, rB, note in R:
        out[c] = dict(cand=c, type=typ, base=TL(base), obj_src=TL(o), dst_src=TL(d), act_src=TL(a), A=TL(A), rA=rA,
                      B=TL(B), rB=rB, note=note, ambiguous=note.startswith("ambiguous") or "or" in note.split(":")[0])
    return out


# ---------------------------------------------------------------- data


def load():
    ev = list(csv.DictReader(open(EVENTS)))
    assert len(ev) == 410
    jd = {f"{r['cand']}_{r['ep']}": r for r in csv.DictReader(open(JUDGE))}
    T = []
    for r in ev:
        ints = lambda s: [int(x) for x in s.split(";") if x != ""]  # noqa: E731
        t = dict(group=r["group"], label=r["task"], init=int(r["init"]), instr=r["instruction"],
                 gcs=ints(r["grasp_call"]), rcs=ints(r["release_call"]),
                 cc=int(r["contact_call"]) if r["contact_call"] not in ("", "None") else None,
                 fix_t=int(r["first_fixture_contact_t"]) if r["first_fixture_contact_t"] not in ("", "None") else None,
                 fixture=r["fixture"])
        if r["group"] == "train":
            t["task"] = int(r["task"][1:])
            t["success"] = r["success"] == "1"
            t["name"] = f"ctl_t{t['task'] - 1:02d}_train_i{t['init']:02d}"
        else:
            j = jd[f"{r['task']}_i{t['init']:02d}"]
            t["cat"] = j["cat"]
            t["old"] = json.loads(j["old"])
            t["success"] = j["cat"] in ("new", "new_byproduct")
            t["name"] = f"{r['task']}_i{t['init']:02d}"
        t["bound"] = t["gcs"][0] if t["gcs"] else t["cc"]
        T.append(t)
    S = {}
    for f in FEATS:
        z = np.load(PT / f"_cache/space_G_{f}.npz")
        S[f] = {k: z[k] for k in z.files}
        assert S[f]["P50"].shape[:2] == (410, NC)
    return T, S


def segment(t, role, task):
    """calls of one role segment of a training trial (see prereg), or None."""
    clip = lambda a, b: list(range(max(0, a), min(NC - 1, b) + 1))  # noqa: E731
    if role == "H":
        if task == 4:
            c = None if t["fix_t"] is None else min(t["fix_t"] // 10, NC - 1)
        else:
            c = t["cc"]
        return None if c is None else clip(c - 1, c + 3)
    if not t["gcs"]:
        return None
    g = t["gcs"][0]
    r = next((x for x in t["rcs"] if x >= g), None)
    if r is None:
        return None
    return clip(g, r) if role == "P" else clip(r - 2, r)


def dist(a, b):
    """pairwise Euclidean distances (n, d) × (m, d) → (n, m) without BLAS (T024: dgemm segfault on VGI-102)."""
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


class Space:
    def __init__(self, T, S, f):
        self.f, self.X = f, S[f]["P50"].astype(np.float64)
        self.pca, self.tsne = S[f]["pca"], S[f]["tsne"]
        self.T = T
        self.tr = {k: [i for i, t in enumerate(T) if t["group"] == "train" and t["task"] == k and t["success"]]
                   for k in range(1, 11)}
        self.mu = {k: self.X[self.tr[k]].mean(0) for k in range(1, 11)}  # (NC, 50) task mean trajectory
        self.s = {}
        for k, ids in self.tr.items():
            Y = self.X[ids]  # (n, NC, 50)
            iu = np.triu_indices(len(ids), 1)
            d = np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))  # (n, n, NC)
            self.s[k] = float(np.median(d[iu[0], iu[1]]))
        self.s_med = float(np.median(list(self.s.values())))
        self.allpts = {k: self.X[ids].reshape(-1, self.X.shape[-1]) for k, ids in self.tr.items()}

    def segs(self, k, role):
        out = []
        for i in self.tr[k]:
            sg = segment(self.T[i], role, k)
            if sg:
                out.append((i, sg))
        return out

    def a1(self, a, ra, b, rb):
        """A1, A1c, A2 for one (a, b) source combination."""
        SA, SB = self.segs(a, ra), self.segs(b, rb)
        if not SA or not SB:
            return None
        s = (self.s[a] + self.s[b]) / 2
        m, cross, pq, d0, dpre = [], [], [], [], []
        for i, sa in SA:
            for j, sb in SB:
                if i == j:
                    continue
                D = dist(self.X[i, sa], self.X[j, sb]) / s
                p, q = np.unravel_index(np.argmin(D), D.shape)
                m.append(D[p, q])
                pq.append((sa[p], sb[q]))
                cross.append(int((D.min(1) < 1).sum()))
                d0.append(np.sqrt(((self.X[i, 0] - self.X[j, 0]) ** 2).sum()) / s)
                bi, bj = self.T[i]["bound"], self.T[j]["bound"]
                e = min(x for x in (bi, bj, NC - 1) if x is not None)
                dpre.append(np.sqrt(((self.X[i, :e + 1] - self.X[j, :e + 1]) ** 2).sum(-1)).mean() / s)
        pq = np.array(pq)
        return dict(A1=float(np.median(m)), A1c=float(np.median(cross)), A1_p=float(np.median(pq[:, 0])),
                    A1_q=float(np.median(pq[:, 1])), A2_0=float(np.median(d0)), A2_pre=float(np.median(dpre)),
                    n_pairs=len(m), a=a, b=b)

    def a3(self, b, rb):
        SB = self.segs(b, rb)
        if not SB:
            return None
        other = np.concatenate([self.allpts[k] for k in range(1, 11) if k != b])
        r = []
        for i, sb in SB:
            own = np.concatenate([self.X[j] for j in self.tr[b] if j != i])
            P = self.X[i, sb]
            r.extend(dist(P, other).min(1) / dist(P, own).min(1))
        return float(np.median(r))

    def a4(self, base):
        tight, mn, mean = [], [], []
        for k in base:
            d = [np.sqrt(((self.mu[k] - self.mu[l]) ** 2).sum(-1)).mean() / ((self.s[k] + self.s[l]) / 2)
                 for l in range(1, 11) if l != k]
            tight.append(self.s[k] / self.s_med)
            mn.append(min(d))
            mean.append(np.mean(d))
        return dict(A4_tight=float(np.mean(tight)), A4_min=float(np.mean(mn)), A4_mean=float(np.mean(mean)))

    def centroids(self):
        return np.stack([self.mu[k] for k in range(1, 11)])  # (10, NC, 50)


# ---------------------------------------------------------------- A5 patch maps


def patch_maps(T, roles_):
    """mean early-layer agentview / both-camera attention map per (task, role) and the train10 global mean."""
    cache = {}

    def get(i):
        if i not in cache:
            t = T[i]
            z = np.load(NPZ[t["group"]] / f"{t['name']}.npz")
            cache[i] = z["attn_img"][:, 0, :512].astype(np.float64)  # (NC, 512) layer group 0, two cameras
        return cache[i]

    train = [i for i, t in enumerate(T) if t["group"] == "train"]
    glob = np.mean([get(i) for i in train], axis=(0, 1))
    maps = {}
    for r in roles_.values():
        for ks, role in ((r["A"], r["rA"]), (r["B"], r["rB"])):
            for k in ks:
                if (k, role) in maps:
                    continue
                rows = []
                for i in train:
                    t = T[i]
                    if t["task"] != k or not t["success"]:
                        continue
                    sg = segment(t, role, k)
                    if sg:
                        rows.append(get(i)[sg].mean(0))
                maps[k, role] = np.mean(rows, 0) if rows else None
    return maps, glob


def cos(a, b):
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# ---------------------------------------------------------------- statistics


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):  # average ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def spearman_perm(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = spearman(x, y)
    null = np.array([spearman(x, RNG.permutation(y)) for _ in range(NPERM)])
    p = (1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (NPERM + 1)
    return dict(rho=r(rho), p=r(p), n=int(ok.sum()))


def group_diff(x, lab):
    x, lab = np.asarray(x, float), np.asarray(lab, bool)
    ok = np.isfinite(x)
    a, b = x[ok & lab], x[ok & ~lab]
    if len(a) < 2 or len(b) < 2:
        return None
    d = np.median(a) - np.median(b)
    bs = [np.median(RNG.choice(a, len(a))) - np.median(RNG.choice(b, len(b))) for _ in range(NBOOT)]
    # rank-based separation: AUC of "success has the larger value"
    auc = np.mean([(ai > bi) + 0.5 * (ai == bi) for ai in a for bi in b])
    return dict(med_succ=r(np.median(a)), med_fail=r(np.median(b)), diff=r(d),
                ci=[r(np.percentile(bs, 2.5)), r(np.percentile(bs, 97.5))], auc=r(auc), n_succ=len(a), n_fail=len(b))


def loo_threshold(x, lab):
    """leave-one-out single threshold (direction chosen on the training fold too)."""
    x, lab = np.asarray(x, float), np.asarray(lab, bool)
    ok = np.isfinite(x)
    x, lab = x[ok], lab[ok]
    hits = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        best = (-1, None, None)
        cands = np.unique(x[m])
        for th in np.concatenate([[-np.inf], (cands[:-1] + cands[1:]) / 2, [np.inf]]):
            for sgn in (1, -1):
                acc = np.mean((sgn * (x[m] - th) > 0) == lab[m])
                if acc > best[0]:
                    best = (acc, th, sgn)
        hits.append((best[2] * (x[i] - best[1]) > 0) == lab[i])
    return dict(loo_acc=r(np.mean(hits)), majority=r(max(lab.mean(), 1 - lab.mean())), n=len(x))


def r(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


# ---------------------------------------------------------------- per-candidate computations


def role_of(k, rl):
    """role colour index of training task k for a candidate: 0 A, 1 B, 2 base, 3 other (A wins over B over base)."""
    if k in rl["A"]:
        return 0
    if k in rl["B"]:
        return 1
    if k in rl["base"]:
        return 2
    return 3


def switch_data(sp, T, cand, rl):
    """nearest training-task centroid per call for each candidate trial."""
    C = sp.centroids()  # (10, NC, 50)
    ids = [i for i, t in enumerate(T) if t["group"] == "cand" and t["label"] == cand]
    near = np.zeros((len(ids), NC), int)
    dsrc = {}
    for n, i in enumerate(ids):
        d = np.sqrt(((sp.X[i][None] - C) ** 2).sum(-1))  # (10, NC)
        near[n] = d.argmin(0) + 1
        dsrc[i] = d
    return ids, near, dsrc


def switch_summary(T, ids, near, rl):
    """per trial: share of calls nearest A / B / base before & after bound; A→B switch call relative to grasp."""
    out = []
    for n, i in enumerate(ids):
        t = T[i]
        b = t["bound"] if t["bound"] is not None else 5
        pre, post = near[n, :b + 1], near[n, b + 1:]
        f = lambda seg, ks: float(np.isin(seg, ks).mean()) if len(seg) else np.nan  # noqa: E731
        sw = None
        if rl["A"] and rl["B"]:
            seenA = False
            for c in range(NC):
                if near[n, c] in rl["A"]:
                    seenA = True
                elif seenA and near[n, c] in rl["B"] and near[n, c] not in rl["A"]:
                    sw = c
                    break
        out.append(dict(trial=t["name"], success=t["success"], cat=t["cat"], bound=t["bound"],
                        pre_A=f(pre, rl["A"]), pre_B=f(pre, rl["B"]), pre_base=f(pre, rl["base"]),
                        post_A=f(post, rl["A"]), post_B=f(post, rl["B"]), post_base=f(post, rl["base"]),
                        post_old=f(post, t["old"]) if t["old"] else np.nan,
                        switch_call=sw, switch_rel=None if sw is None or t["bound"] is None else sw - t["bound"]))
    return out


# ---------------------------------------------------------------- figures


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def fig_candidate(c, rl, spaces, T, a1info, path):
    """rows: patch_attention, expert; cols: fixed PCA (PC1–2), joint t-SNE."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    cand_ids = [i for i, t in enumerate(T) if t["group"] == "cand" and t["label"] == c]
    groups = []
    for ks, col, nm in ((rl["A"], C_A, "A"), (rl["B"], C_B, "B"), ([k for k in rl["base"] if k not in rl["A"] + rl["B"]],
                                                                       C_BASE, "base")):
        for k in ks:
            groups.append((k, col, nm))
    for row, f in enumerate(("patch_attention", "expert")):
        sp = spaces[f]
        info = a1info.get(f)
        for col_i, kind in enumerate(("pca", "tsne")):
            ax = axes[row, col_i]
            Y = sp.pca[..., :2] if kind == "pca" else sp.tsne
            ax.scatter(Y[:100].reshape(-1, 2)[:, 0], Y[:100].reshape(-1, 2)[:, 1], s=1.5, c=C_OTHER, lw=0, zorder=1)
            for k, colr, nm in groups:
                for i in sp.tr[k]:
                    ax.plot(Y[i, :, 0], Y[i, :, 1], color=colr, lw=0.6, alpha=0.35, zorder=2)
                m = Y[sp.tr[k]].mean(0) if kind == "pca" else np.median(Y[sp.tr[k]], 0)
                ax.plot(m[:, 0], m[:, 1], color=colr, lw=2.4, zorder=4, label=f"{nm}: G{k}")
                ax.plot(*m[0], "o", ms=6, mfc="white", mec=colr, mew=1.5, zorder=5)
                for cc in range(5, NC, 5):
                    ax.annotate(str(cc), m[cc], fontsize=6, color=INK2, zorder=6)
                gc = [T[i]["gcs"][0] for i in sp.tr[k] if T[i]["gcs"]]
                rc = [x for i in sp.tr[k] for x in T[i]["rcs"][:1]]
                if gc:
                    gm = int(np.median(gc))
                    ax.plot(*m[gm], "x", ms=10, mew=2.4, color=INK, zorder=7)
                if rc:
                    rm = int(np.median(rc))
                    ax.plot(*m[rm], "s", ms=7, mfc=colr, mec=INK, mew=1.2, zorder=7)
            for i in cand_ids:
                t = T[i]
                ax.plot(Y[i, :, 0], Y[i, :, 1], color=INK, lw=0.8, alpha=0.75, ls="-" if t["success"] else ":",
                        zorder=3)
                ax.plot(*Y[i, 0], "o", ms=3, mfc="white", mec=INK, mew=0.6, zorder=3)
                if t["gcs"]:
                    ax.plot(*Y[i, t["gcs"][0]], "x", ms=5, mew=1.0, color=INK, zorder=3)
            if info is not None:
                a, b = info["a"], info["b"]
                p, q = int(round(info["A1_p"])), int(round(info["A1_q"]))
                ma = Y[sp.tr[a]].mean(0) if kind == "pca" else np.median(Y[sp.tr[a]], 0)
                mb = Y[sp.tr[b]].mean(0) if kind == "pca" else np.median(Y[sp.tr[b]], 0)
                ax.plot([ma[p, 0], mb[q, 0]], [ma[p, 1], mb[q, 1]], "--", color=INK, lw=1, zorder=8)
                for pt, cc in ((ma[p], p), (mb[q], q)):
                    ax.plot(*pt, "*", ms=17, mfc="#eda100", mec=INK, mew=0.8, zorder=9)
                    ax.annotate(f"c{cc}", pt, xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK,
                                fontweight="bold", zorder=10)
            clean(ax)
            ttl = f"{f} — {'fixed PCA (PC1–2)' if kind == 'pca' else 'joint t-SNE (reference)'}"
            if info is not None and kind == "pca":
                ttl += f"\nA1 = {info['A1']:.2f} (50-d, /within-task spread), crossing {info['A1c']:.0f} calls"
            ax.set_title(ttl, fontsize=9, color=INK)
            if row == 0 and col_i == 0:
                ax.legend(fontsize=7, loc="best", frameon=False)
    n_new = sum(T[i]["success"] for i in cand_ids)
    fig.suptitle(f"{c}: {T[cand_ids[0]]['instr']}   — new {n_new}/10   [{rl['type']}]\n"
                 f"A (object/1st) = G{'/G'.join(map(str, rl['A'])) or '–'} [{rl['rA'] or '–'}]  ·  "
                 f"B (destination/action/2nd) = G{'/G'.join(map(str, rl['B'])) or '–'} [{rl['rB'] or '–'}]  ·  "
                 f"base = G{'/G'.join(map(str, rl['base']))}\n"
                 "thick = source mean trajectory (○ call 0, numbers every 5 calls, × median grasp, ■ median release)\n"
                 "★ = closest approach (A1 argmin calls, computed in 50-d); black = candidate trials "
                 "(solid = new-task success, dotted = not)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_scatter(feat_rows, key, title, path, cands, rate, excluded_rate=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, f in zip(axes.ravel(), FEATS):
        x = np.array([feat_rows[c].get(f"{f}:{key}", np.nan) for c in cands], float)
        y = np.array([rate[c] for c in cands])
        ok = np.isfinite(x)
        succ = y >= SUCC_MIN / 10
        ax.scatter(x[ok & succ], y[ok & succ], s=55, c=C_B, edgecolor=INK, lw=0.6, zorder=3, label="success ≥ 3/10")
        ax.scatter(x[ok & ~succ], y[ok & ~succ], s=40, c=C_A, edgecolor=INK, lw=0.6, zorder=3, label="< 3/10")
        for c, xi, yi in zip(cands, x, y):
            if np.isfinite(xi):
                ax.annotate(c, (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK2)
        st = feat_rows["_stats"].get(f"{f}:{key}")
        if excluded_rate:
            for c, (xv, yv) in excluded_rate.items():
                v = xv.get(f"{f}:{key}", np.nan)
                if np.isfinite(v):
                    ax.scatter([v], [yv], s=40, marker="^", facecolor="none", edgecolor=MUTED, zorder=2)
                    ax.annotate(c + " (excl.)", (v, yv), xytext=(3, -9), textcoords="offset points", fontsize=7,
                                color=MUTED)
        if key == "A1":
            ax.axvline(1, color=MUTED, lw=1, ls="--")
        clean(ax)
        if st is None:
            s = ""
        elif st["rho"] is None:
            s = f"\nconstant across candidates (n = {st['n']})"
        else:
            s = f"\nSpearman ρ = {st['rho']:+.2f}, perm p = {st['p']:.3f}, n = {st['n']}"
        ax.set_title(f"{f}{s}", fontsize=9, color=INK)
        ax.set_xlabel(title, fontsize=8, color=INK2)
        ax.set_ylabel("new-task success rate", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{title} vs new-task success (one point per candidate; g18/g28 grey △, not in the statistics)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_switch(f, sw, rl_all, T, cands, path):
    cmap = ListedColormap([C_A, C_B, C_BASE, C_OTHER])
    n = len(cands)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.4 * nrow))
    for ax, c in zip(axes.ravel(), cands):
        ids, near = sw[c]["ids"], sw[c]["near"]
        rl = rl_all[c]
        order = sorted(range(len(ids)), key=lambda n_: (not T[ids[n_]]["success"], T[ids[n_]]["init"]))
        M = np.array([[role_of(k, rl) for k in near[o]] for o in order])
        ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto", interpolation="nearest")
        for y, o in enumerate(order):
            t = T[ids[o]]
            for x in range(NC):
                k = near[o, x]
                if role_of(k, rl) == 3:
                    ax.text(x, y, str(k), fontsize=4.2, ha="center", va="center", color=INK2)
            if t["gcs"]:
                ax.plot(t["gcs"][0], y, "x", ms=4, mew=1.0, color=INK)
            for rc in t["rcs"][:1]:
                ax.plot(rc, y, "s", ms=3, mfc="white", mec=INK, mew=0.7)
        ns = sum(T[i]["success"] for i in ids)
        if 0 < ns < len(ids):
            ax.axhline(ns - 0.5, color=INK, lw=1.4)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([("✓ " if T[ids[o]]["success"] else "") + f"i{T[ids[o]]['init']}" for o in order],
                           fontsize=5)
        ax.set_xticks(range(0, NC, 5))
        ax.tick_params(labelsize=6)
        ax.set_title(f"{c} new {ns}/10 · A=G{'/'.join(map(str, rl['A'])) or '–'} B=G{'/'.join(map(str, rl['B'])) or '–'}"
                     f" base=G{'/'.join(map(str, rl['base']))}", fontsize=7, color=INK)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C_A, label="A source (object / 1st task)"),
                        Patch(color=C_B, label="B source (destination / action / 2nd task)"),
                        Patch(color=C_BASE, label="base task (if not A/B)"),
                        Patch(color=C_OTHER, label="other training task (number = which)")],
               loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"{f}: nearest training-task centroid per call (rows = trials, ✓ = new-task success, on top; "
                 "× first grasp, □ first release)", y=0.995, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_mixed(mixed, curves, path):
    feats = ("patch_attention", "expert")
    tg = ("A", "B", "base")
    fig, axes = plt.subplots(len(mixed), 6, figsize=(20, 2.6 * len(mixed)), squeeze=False)
    for row, c in enumerate(mixed):
        for fi, f in enumerate(feats):
            for ti, tname in enumerate(tg):
                ax = axes[row, fi * 3 + ti]
                cv = curves[f][c].get(tname)
                if cv is None:
                    ax.axis("off")
                    continue
                x = np.arange(NC)
                for key, colr, ls, lab in (("succ", C_B, "-", "success"), ("fail", C_A, ":", "not success")):
                    m, lo, hi = cv[key]
                    ax.fill_between(x, lo, hi, color=colr, alpha=0.18, lw=0)
                    ax.plot(x, m, color=colr, lw=2, ls=ls, label=f"{lab} (n={cv['n_' + key]})")
                for xx in cv["sig"]:
                    ax.axvspan(xx - 0.5, xx + 0.5, color="#eda100", alpha=0.25, lw=0)
                ax.axvline(cv["grasp"], color=MUTED, lw=1, ls="--")
                clean(ax)
                ax.set_title(f"{c} · {f} · dist to {tname} (G{cv['task']})", fontsize=8, color=INK)
                if fi == 0 and ti == 0:
                    ax.legend(fontsize=6, frameon=False)
                    ax.set_ylabel("distance / spread", fontsize=7)
    fig.suptitle("Mixed-outcome candidates: distance to each source-task centroid per call (mean ± 95% bootstrap CI "
                 "over trials). yellow band = call where the success − failure CI excludes 0; dashed = median "
                 "grasp call", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="nfs")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    T, S = load()
    RL = roles()
    cands = [f"g{i:02d}" for i in range(1, 32)]
    main_c = [c for c in cands if c not in EXCLUDED]
    n_new = {c: sum(t["success"] for t in T if t["group"] == "cand" and t["label"] == c) for c in cands}
    rate = {c: n_new[c] / 10 for c in cands}
    spaces = {f: Space(T, S, f) for f in FEATS}

    # roles.csv
    with open(OUT / "roles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cand", "instruction", "type", "base_task", "object_source", "destination_source",
                    "action_source", "A_tasks", "A_segment", "B_tasks", "B_segment", "new_of_10", "success_ge3",
                    "excluded", "infeasible_flag", "ambiguous", "note"])
        for c in cands:
            rl = RL[c]
            instr = next(t["instr"] for t in T if t["label"] == c)
            j = lambda xs: ";".join(map(str, xs))  # noqa: E731
            w.writerow([c, instr, rl["type"], j(rl["base"]), j(rl["obj_src"]), j(rl["dst_src"]), j(rl["act_src"]),
                        j(rl["A"]), rl["rA"], j(rl["B"]), rl["rB"], n_new[c], int(n_new[c] >= SUCC_MIN),
                        int(c in EXCLUDED), int(c in INFEASIBLE or c == "g13"), int(rl["ambiguous"]), rl["note"]])

    # ---------------- prior features
    rows = {c: {} for c in cands}
    a1info = {c: {} for c in cands}
    for f in FEATS:
        sp = spaces[f]
        for c in cands:
            rl = RL[c]
            if rl["A"] and rl["B"]:
                best = None
                for a in rl["A"]:
                    for b in rl["B"]:
                        if a == b:
                            continue
                        v = sp.a1(a, rl["rA"], b, rl["rB"])
                        if v and (best is None or v["A1"] < best["A1"]):
                            best = v
                if best:
                    a1info[c][f] = best
                    for k in ("A1", "A1c", "A2_0", "A2_pre", "A1_p", "A1_q"):
                        rows[c][f"{f}:{k}"] = best[k]
                    rows[c][f"{f}:pair"] = f"G{best['a']}{rl['rA']}-G{best['b']}{rl['rB']}"
                    rows[c][f"{f}:A3"] = sp.a3(best["b"], rl["rB"])
            rows[c].update({f"{f}:{k}": v for k, v in sp.a4(rl["base"]).items()})
    maps, glob = patch_maps(T, RL)
    for c in cands:
        rl = RL[c]
        best = a1info[c].get("patch_attention")
        if best is None:
            continue
        ma, mb = maps.get((best["a"], rl["rA"])), maps.get((best["b"], rl["rB"]))
        if ma is None or mb is None:
            continue
        rows[c]["A5"] = cos(ma[:256] - glob[:256], mb[:256] - glob[:256])
        rows[c]["A5_both"] = cos(ma - glob, mb - glob)
        rows[c]["A5_raw"] = cos(ma[:256], mb[:256])

    PRIOR = ("A1", "A1c", "A2_0", "A2_pre", "A3", "A4_tight", "A4_min", "A4_mean")
    lab = np.array([n_new[c] >= SUCC_MIN for c in main_c])
    yv = np.array([rate[c] for c in main_c])
    stats = {}
    for f in FEATS:
        for k in PRIOR:
            x = np.array([rows[c].get(f"{f}:{k}", np.nan) for c in main_c], float)
            st = spearman_perm(x, yv)
            st["group"] = group_diff(x, lab)
            stats[f"{f}:{k}"] = st
    for k in ("A5", "A5_both", "A5_raw"):
        x = np.array([rows[c].get(k, np.nan) for c in main_c], float)
        stats[k] = spearman_perm(x, yv)
        stats[k]["group"] = group_diff(x, lab)
    xs = np.array([rows[c].get("patch_attention:A1", np.nan) for c in main_c], float)
    loo = loo_threshold(xs, lab)
    # sensitivity: pick-and-place swaps only (obj_swap / dst_swap, both P/L/H sources)
    pp = [c for c in main_c if RL[c]["type"] in ("obj_swap", "dst_swap")]
    stats_pp = {}
    for f in FEATS:
        for k in ("A1", "A1c", "A2_0", "A2_pre", "A3", "A4_min", "A4_tight"):
            x = np.array([rows[c].get(f"{f}:{k}", np.nan) for c in pp], float)
            stats_pp[f"{f}:{k}"] = spearman_perm(x, np.array([rate[c] for c in pp]))

    with open(OUT / "features.csv", "w", newline="") as fh:
        keys = ["pair"] + list(PRIOR) + ["A1_p", "A1_q"]
        cols = [f"{f}:{k}" for f in FEATS for k in keys] + ["A5", "A5_both", "A5_raw"]
        w = csv.writer(fh)
        w.writerow(["cand", "new_of_10", "success_ge3", "in_main"] + cols)
        for c in cands:
            w.writerow([c, n_new[c], int(n_new[c] >= SUCC_MIN), int(c in main_c)] +
                       [(v if isinstance(v, str) else r(v)) if (v := rows[c].get(k)) is not None else "" for k in cols])

    # ---------------- B6 switch pattern
    sw = {f: {} for f in FEATS}
    sw_rows = {f: [] for f in FEATS}
    for f in FEATS:
        for c in cands:
            ids, near, dsrc = switch_data(spaces[f], T, c, RL[c])
            sw[f][c] = dict(ids=ids, near=near, dsrc=dsrc)
            for d in switch_summary(T, ids, near, RL[c]):
                d["cand"] = c
                sw_rows[f].append(d)
    sw_stats = {}
    for f in FEATS:
        rs = [d for d in sw_rows[f] if d["cand"] in main_c and RL[d["cand"]]["A"] and RL[d["cand"]]["B"]]
        res = {}
        for key in ("pre_A", "pre_B", "pre_base", "post_A", "post_B", "post_base", "post_old"):
            a = np.array([d[key] for d in rs if d["success"]], float)
            b = np.array([d[key] for d in rs if not d["success"]], float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            res[key] = dict(succ=r(a.mean()) if len(a) else None, fail=r(b.mean()) if len(b) else None,
                            n_succ=len(a), n_fail=len(b))
        res["switch_A_to_B_share"] = dict(
            succ=r(np.mean([d["switch_call"] is not None for d in rs if d["success"]])),
            fail=r(np.mean([d["switch_call"] is not None for d in rs if not d["success"]])))
        rel = [d["switch_rel"] for d in rs if d["success"] and d["switch_rel"] is not None]
        res["switch_rel_to_grasp_success"] = dict(median=r(np.median(rel)) if rel else None,
                                                  q=[r(np.percentile(rel, 25)), r(np.percentile(rel, 75))] if rel
                                                  else None, n=len(rel))
        relf = [d["switch_rel"] for d in rs if not d["success"] and d["switch_rel"] is not None]
        res["switch_rel_to_grasp_fail"] = dict(median=r(np.median(relf)) if relf else None, n=len(relf))
        # failures that fell back to a training task: is the executed task already nearest before the bound?
        fb = [d for d in rs if not d["success"] and d["cat"] == "trained"]
        res["fallback_pre_old"] = r(np.mean([sw_pre_old(T, sw[f][d["cand"]], d) for d in fb])) if fb else None
        sw_stats[f] = res

    # ---------------- within-candidate success vs failure
    mixed = [c for c in main_c if 2 <= n_new[c] <= 8]  # ≥ 2 trials in each group
    curves = {f: {} for f in FEATS}
    within = {f: {} for f in FEATS}
    for f in FEATS:
        sp = spaces[f]
        for c in mixed:
            rl = RL[c]
            ids = sw[f][c]["ids"]
            succ = np.array([T[i]["success"] for i in ids])
            g = [T[i]["gcs"][0] for i in ids if T[i]["gcs"]]
            curves[f][c] = {}
            within[f][c] = {}
            targets = (("A", rl["A"] or rl["act_src"]), ("B", rl["B"]), ("base", rl["base"]))
            for tname, ks in targets:
                if not ks:
                    continue
                k = ks[0] if len(ks) == 1 else min(ks, key=lambda kk: np.mean(
                    [sw[f][c]["dsrc"][i][kk - 1].mean() for i in ids]))
                D = np.array([sw[f][c]["dsrc"][i][k - 1] for i in ids]) / sp.s[k]  # (10, NC)
                a, b = D[succ], D[~succ]
                diffs = np.array([RNG.choice(a, len(a)).mean(0) - RNG.choice(b, len(b)).mean(0)
                                  for _ in range(2000)])
                lo, hi = np.percentile(diffs, 2.5, 0), np.percentile(diffs, 97.5, 0)
                sig = [int(x) for x in np.where((lo > 0) | (hi < 0))[0]]

                def band(M):
                    bs = np.array([M[RNG.integers(0, len(M), len(M))].mean(0) for _ in range(2000)])
                    return M.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)

                curves[f][c][tname] = dict(succ=band(a), fail=band(b), n_succ=len(a), n_fail=len(b), sig=sig,
                                           grasp=int(np.median(g)) if g else 0, task=k)
                within[f][c][tname] = dict(task=k, first_split_call=sig[0] if sig else None, n_split_calls=len(sig),
                                           split_calls=sig,
                                           mean_diff_pre=r((a.mean(0) - b.mean(0))[:int(np.median(g)) + 1].mean()
                                                           if g else np.nan),
                                           mean_diff_post=r((a.mean(0) - b.mean(0))[int(np.median(g)) + 1:].mean()
                                                            if g else np.nan))
    # pooled over mixed candidates: per candidate success − failure mean distance to B (post) and base
    pooled = {}
    for f in FEATS:
        pooled[f] = {}
        for tname in ("A", "B", "base"):
            per = []
            for c in mixed:
                cv = curves[f][c].get(tname)
                if cv:
                    per.append(cv["succ"][0] - cv["fail"][0])
            P = np.array(per)
            bs = np.array([P[RNG.integers(0, len(P), len(P))].mean(0) for _ in range(NBOOT // 5)])
            lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
            pooled[f][tname] = dict(mean=[r(v) for v in P.mean(0)], ci_lo=[r(v) for v in lo],
                                    ci_hi=[r(v) for v in hi],
                                    sig_calls=[int(x) for x in np.where((lo > 0) | (hi < 0))[0]], n_cand=len(P))

    if "n" in args.parts:
        js = dict(
            meta=dict(feats=FEATS, success_min=SUCC_MIN, main=main_c, excluded=EXCLUDED, infeasible=INFEASIBLE,
                      mixed=mixed, n_new=n_new, spread={f: {k: r(v) for k, v in spaces[f].s.items()} for f in FEATS}),
            prior_stats=stats, prior_stats_pick_place_only=dict(cands=pp, stats=stats_pp), loo_patch_A1=loo,
            excluded_candidates={c: {k: (v if isinstance(v, str) else r(v)) for k, v in rows[c].items()}
                                 for c in EXCLUDED},
            switch=sw_stats, within=within, within_pooled=pooled,
            switch_trials={f: [{k: (r(v) if isinstance(v, float) else v) for k, v in d.items()} for d in sw_rows[f]]
                           for f in ("patch_attention", "expert")},
        )
        (OUT / "numbers.json").write_text(json.dumps(js, indent=1, ensure_ascii=False, default=str))
        print_summary(stats, stats_pp, loo, sw_stats, within, pooled, rows, main_c, n_new, mixed)

    if "f" in args.parts:
        d = OUT / "fig1_candidates"
        d.mkdir(exist_ok=True)
        for c in cands:
            fig_candidate(c, RL[c], spaces, T, a1info[c], d / f"{c}.png")
        print("[fig1] done", flush=True)
    if "s" in args.parts:
        rows["_stats"] = stats
        excl = {c: (rows[c], rate[c]) for c in EXCLUDED}
        for k, ttl in (("A1", "A1 closest approach (/ within-task spread)"), ("A1c", "A1c crossing length (calls < 1)"),
                       ("A2_0", "A2 distance at call 0 (/ spread)"), ("A3", "A3 destination exclusivity"),
                       ("A4_min", "A4 base task: min distance to another task (/ spread)")):
            fig_scatter(rows, k, ttl, OUT / f"fig2_scatter_{k}.png", main_c, rate, excl)
        d = OUT / "fig3_switch"
        d.mkdir(exist_ok=True)
        for f in FEATS:
            fig_switch(f, sw[f], RL, T, cands, d / f"switch_{f}.png")
        fig_mixed(mixed, curves, OUT / "fig4_mixed_success_vs_failure.png")
        fig_a5(rows, main_c, rate, stats, OUT / "fig2_scatter_A5.png")
        print("[figs] done", flush=True)


def sw_pre_old(T, swf, d):
    i = next(i for i in swf["ids"] if T[i]["name"] == d["trial"])
    n = swf["ids"].index(i)
    t = T[i]
    b = t["bound"] if t["bound"] is not None else 5
    return float(np.isin(swf["near"][n, :b + 1], t["old"]).mean())


def fig_a5(rows, cands, rate, stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, k in zip(axes, ("A5", "A5_both")):
        x = np.array([rows[c].get(k, np.nan) for c in cands], float)
        y = np.array([rate[c] for c in cands])
        ok = np.isfinite(x)
        s = y >= SUCC_MIN / 10
        ax.scatter(x[ok & s], y[ok & s], s=55, c=C_B, edgecolor=INK, lw=0.6, label="success ≥ 3/10")
        ax.scatter(x[ok & ~s], y[ok & ~s], s=40, c=C_A, edgecolor=INK, lw=0.6, label="< 3/10")
        for c, xi, yi in zip(cands, x, y):
            if np.isfinite(xi):
                ax.annotate(c, (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK2)
        st = stats[k]
        clean(ax)
        ax.set_title(f"{k}: ρ = {st['rho']:+.2f}, perm p = {st['p']:.3f}, n = {st['n']}", fontsize=9)
        ax.set_xlabel("cosine of source key-phase patch maps (minus train10 mean map)"
                      + (" — agentview" if k == "A5" else " — both cameras"), fontsize=8)
        ax.set_ylabel("new-task success rate", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("A5 patch-attention location overlap (early layers) vs new-task success", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def print_summary(stats, stats_pp, loo, sw_stats, within, pooled, rows, main_c, n_new, mixed):
    print("== prior features: Spearman rho (perm p) vs new-task rate, n; success-failure AUC")
    for k in ("A1", "A1c", "A2_0", "A2_pre", "A3", "A4_tight", "A4_min", "A4_mean"):
        print(k.ljust(9), "  ".join(f"{f[:6]} {stats[f'{f}:{k}']['rho']}({stats[f'{f}:{k}']['p']},"
                                     f"n{stats[f'{f}:{k}']['n']},auc{(stats[f'{f}:{k}']['group'] or {}).get('auc')})"
                                     for f in FEATS))
    for k in ("A5", "A5_both", "A5_raw"):
        print(k, stats[k])
    print("pick/place only:", {k: (v["rho"], v["p"], v["n"]) for k, v in stats_pp.items()})
    print("LOO patch A1:", loo)
    for f in FEATS:
        print("switch", f, json.dumps(sw_stats[f]))
    for f in ("patch_attention", "expert"):
        for c in mixed:
            print("within", f, c, {t: (v["task"], v["first_split_call"], v["n_split_calls"]) for t, v in
                                   within[f][c].items()})
    for f in FEATS:
        print("pooled", f, {t: v["sig_calls"] for t, v in pooled[f].items()})
    print("A1 patch per cand:", {c: (r(rows[c].get('patch_attention:A1', np.nan)), n_new[c]) for c in main_c})


if __name__ == "__main__":
    main()
