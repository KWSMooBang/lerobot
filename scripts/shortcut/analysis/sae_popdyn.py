#!/usr/bin/env python
"""T033 sae_popdyn: population analyses of SAE feature activations on T028 (CPU only).

SAE codes of each call are treated as a population vector of recorded neurons; an episode is a path of these vectors.
Definitions, metrics and hypotheses: sae_popdyn_PREREG.md (committed before results).

Stages (run in this order)
  prep   per-layer sparse blocks: rare / dead features removed, each feature divided by its T028 max
  a      tensor component analysis (non-negative CP, HALS) + rank selection + fallback trial loadings
  b      variance partition by cross-validated regression (time / instructed / executed / scene) + dPCA marginalisation
  c      composed task as NNLS mixture of its source tasks (T029 A7 in SAE space)
  d      communication subspace (reduced-rank regression VLM -> action expert)
  e      trajectory geometry (divergence, speed, participation ratio, basket path, fallback)
  figs_<x> figures for stage x; numbers -> numbers.json

Inputs: T032 VLM codes (vlm/analysis/_cache/codes_pgNN.npz), T031 expert codes (cache/t031_t028_codes/codes_LNN_s9.npz),
T030 layer_pop meta / labels / cases.

Run:  OPENBLAS_CORETYPE=SkylakeX .venv-pd/bin/python scripts/shortcut/analysis/sae_popdyn.py <stage>
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_pop as lp  # noqa: E402
from neuro_dyn import bh  # noqa: E402

OUT = Path(os.environ.get("T033_OUT", "/outputs/VLA-shortcut-exps/sae/population"))
FIG = OUT / "fig"
RES = OUT / "results"
CACHE = Path(os.environ.get("T033_CACHE", "/data0/woosung/datasets/VLA-shortcut-exps/cache/t033_popdyn"))
VLM_CACHE = Path("/outputs/VLA-shortcut-exps/sae/vlm/analysis/_cache")
VLM_LAB = Path("/outputs/VLA-shortcut-exps/sae/vlm/analysis/feature_labels.csv")
EXP_CODES = Path("/data0/woosung/datasets/VLA-shortcut-exps/cache/t031_t028_codes")
WORKERS = 8
NC = 28  # calls 0..27 (every T028 episode has >= 28 calls)
NF = 2048
N_TOT = 51100
PHASES = ("reach", "grasp", "carry", "release")

VLM_ALL = [f"v{i:02d}" for i in range(18)]
EXP_ALL = [f"e{i:02d}" for i in range(18)]
VLM_MAIN = [f"v{i:02d}" for i in range(16)]
EXP_MAIN = [f"e{i:02d}" for i in range(5, 18)]
VLM_SUB = ["v01", "v04", "v08", "v12", "v15"]
EXP_SUB = ["e05", "e08", "e11", "e14", "e17"]
WHOLE = {"whole": VLM_MAIN + EXP_MAIN, "vlm_all": VLM_MAIN, "exp_all": EXP_MAIN}
TAGS_V = ("instruction", "task-shared-scene", "task-confounded", "phase", "scene", "mixed")
TAGS_E = ("task", "phase", "other")
RNG = np.random.default_rng(0)


def pretty(b: str) -> str:
    if b in WHOLE:
        return b
    return ("pg" if b[0] == "v" else "L") + b[1:]


def flag(b: str) -> str:
    """layer status from the PREREG (main / caution / reference / massive)."""
    if b in WHOLE:
        return "whole"
    i = int(b[1:])
    if b[0] == "v":
        return "main" if i <= 15 else ("caution" if i == 16 else "reference")
    return "main" if i >= 5 else "massive"


# ============================================================================ io
def _js(o):
    if isinstance(o, dict):
        return {str(k): _js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_js(v) for v in o]
    if isinstance(o, np.ndarray):
        return _js(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_json(name: str, obj) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(_js(obj), indent=1, ensure_ascii=False))


def load_json(name: str):
    return json.loads((RES / f"{name}.json").read_text())


def pmap(fn, jobs, workers=WORKERS):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs))


# ============================================================================ meta
_M: list = []


def meta() -> list[dict]:
    if not _M:
        _M.extend(lp.meta())
    return _M


def ridx(M=None) -> np.ndarray:
    """(n_ep, NC) global row index of call c of episode i."""
    M = M or meta()
    return np.array([[m["off"] + c for c in range(NC)] for m in M])


def u40_ids(M=None) -> list[int]:
    M = M or meta()
    return [m["i"] for m in lp.u40(M)]


def pool() -> list[dict]:
    return lp.labels()["pool"]


def cand_class(m: dict) -> str:
    if m["outcome"] in lp.FALLBACK:
        return "FALLBACK"
    if m["outcome"] in lp.NEW:
        return "NEW"
    return "other"


def cond_label(m: dict) -> str:
    """trial condition for loadings: U40 suite, cand FALLBACK/NEW/other, l90 kind, basket condition."""
    s = m["t027_set"]
    if s == "train":
        return {"S": "U40 Spatial", "O": "U40 Object", "G": "U40 Goal", "L": "U40 Long"}[m["task"][0]]
    if s == "cand":
        return "cand " + cand_class(m)
    if s == "l90":
        return "L90 " + (m.get("kind") or m["condition"])
    return "basket " + m["condition"]


# ============================================================================ prep
def _raw_codes(b: str) -> tuple[np.ndarray, np.ndarray]:
    if b[0] == "v":
        z = np.load(VLM_CACHE / f"codes_pg{b[1:]}.npz")
        return z["idx"].astype(np.int64), z["vals"].astype(np.float32)
    z = np.load(EXP_CODES / f"codes_L{b[1:]}_s9.npz")
    return z["idx"].astype(np.int64), z["val"].astype(np.float32)


_VLAB: dict = {}
_SEL: dict = {}


def vlm_labels(li: int) -> list[dict]:
    if not _VLAB:
        for r in csv.DictReader(open(VLM_LAB)):
            _VLAB.setdefault(int(r["layer"][2:]), []).append(r)
    return _VLAB[li]


def exp_sel(li: int) -> dict:
    if not _SEL:
        _SEL.update(pickle.load(open(EXP_CODES / "select.pkl", "rb")))
    return _SEL[f"L{li}_s9"]


def feature_info(b: str) -> dict:
    """per original feature: keep-by-label flag, tag group, top training task, phase."""
    li = int(b[1:])
    if b[0] == "v":
        R = vlm_labels(li)
        assert len(R) == NF and all(int(r["feature"]) == j for j, r in enumerate(R))
        tag = []
        for r in R:
            t = r["tag"]
            if t == "task":
                tag.append(r["sub"] or "task-confounded")
            elif t in ("scene", "suite"):
                tag.append("scene")
            elif t == "phase":
                tag.append("phase")
            elif t == "rare":
                tag.append("rare")
            else:
                tag.append("mixed")
        return {"lab_keep": np.array([r["tag"] != "rare" for r in R]), "tag": np.array(tag),
                "top_task": np.array([r["top_task"] for r in R]), "phase": np.array([r["pref_phase"] for r in R])}
    s = exp_sel(li)
    tasks = np.array(s["tasks"])
    tag = np.where(s["t_sel"], "task", np.where(s["ph_sel"], "phase", "other"))
    return {"lab_keep": np.asarray(s["live"], bool), "tag": tag, "top_task": tasks[s["bt"]],
            "phase": np.array(PHASES)[s["bp"]]}


def prep_job(b: str) -> dict:
    idx, val = _raw_codes(b)
    rows = np.repeat(np.arange(N_TOT), idx.shape[1])
    X = sp.csr_matrix((val.ravel(), (rows, idx.ravel())), shape=(N_TOT, NF), dtype=np.float32)
    X.eliminate_zeros()
    fi = feature_info(b)
    mx = X.max(0).toarray().ravel()
    keep = fi["lab_keep"] & (mx > 0)
    X = X[:, keep] @ sp.diags((1.0 / mx[keep]).astype(np.float32))
    X = sp.csr_matrix(X, dtype=np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    sp.save_npz(CACHE / f"blk_{b}.npz", X)
    info = {"feat": np.flatnonzero(keep), "tag": fi["tag"][keep], "top_task": fi["top_task"][keep],
            "phase": fi["phase"][keep], "rms": float(np.sqrt(X.multiply(X).sum() / N_TOT))}
    pickle.dump(info, open(CACHE / f"info_{b}.pkl", "wb"))
    return {"block": b, "n_label_keep": int(fi["lab_keep"].sum()), "n_dead_t028": int((fi["lab_keep"] & (mx <= 0)).sum()),
            "n_keep": int(keep.sum()), "nnz_per_call": float(X.nnz / N_TOT), "rms": info["rms"],
            "tags": dict(Counter(info["tag"]))}


def stage_prep() -> None:
    res = pmap(prep_job, VLM_ALL + EXP_ALL)
    save_json("prep", {r["block"]: r for r in res})
    for r in res:
        print(r["block"], r["n_label_keep"], r["n_dead_t028"], r["n_keep"], round(r["nnz_per_call"], 1), r["tags"])


def block(b: str) -> sp.csr_matrix:
    if b in WHOLE:
        return sp.hstack([block(x) / info(x)["rms"] for x in WHOLE[b]], format="csr", dtype=np.float32)
    return sp.load_npz(CACHE / f"blk_{b}.npz").tocsr()


def info(b: str) -> dict:
    if b in WHOLE:
        parts = [info(x) for x in WHOLE[b]]
        return {"feat": np.concatenate([p["feat"] for p in parts]),
                "layer": np.concatenate([[x] * len(p["feat"]) for x, p in zip(WHOLE[b], parts)]),
                "tag": np.concatenate([p["tag"] for p in parts]), "top_task": np.concatenate([p["top_task"] for p in parts]),
                "phase": np.concatenate([p["phase"] for p in parts]), "rms": 1.0}
    d = pickle.load(open(CACHE / f"info_{b}.pkl", "rb"))
    d["layer"] = np.array([b] * len(d["feat"]))
    return d


def tensor_rows(X: sp.csr_matrix, eps: list[int] | None = None) -> sp.csr_matrix:
    """rows (trial i, call c) in order i*NC + c."""
    R = ridx()
    if eps is not None:
        R = R[eps]
    return X[R.ravel()]


def dense_traj(X: sp.csr_matrix, eps: list[int]) -> np.ndarray:
    """(n_eps, NC, nf) dense."""
    return tensor_rows(X, eps).toarray().reshape(len(eps), NC, -1)


# ============================================================================ a: TCA
A_RANKS = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30)
A_SEEDS = 10
A_MAXIT = 300
A_TOL = 1e-5


class SpTensor:
    """3-way sparse tensor (I trials, J features, K calls) stored as Y = (I*K x J) csr and its transpose."""

    def __init__(self, Y: sp.csr_matrix, I: int, K: int):
        self.Y = Y.tocsr()
        self.Yt = Y.T.tocsr()
        self.I, self.K, self.J = I, K, Y.shape[1]
        self.sq = float(Y.multiply(Y).sum())


def cp_hals(T: SpTensor, R: int, seed: int, maxit: int = A_MAXIT, tol: float = A_TOL) -> dict:
    rng = np.random.default_rng(seed)
    A = rng.random((T.I, R))
    B = rng.random((T.J, R))
    C = rng.random((T.K, R))
    # scale so that ||Xhat|| ~ ||X||
    nh = np.sqrt(((A.T @ A) * (B.T @ B) * (C.T @ C)).sum())
    A *= np.sqrt(T.sq) / nh
    err_hist = []
    eps = 1e-12

    def hals(F, M, G):
        for r in range(F.shape[1]):
            F[:, r] = np.maximum(eps, F[:, r] + (M[:, r] - F @ G[:, r]) / max(G[r, r], eps))

    for it in range(maxit):
        YB = T.Y @ B  # (I*K, R)
        YB3 = YB.reshape(T.I, T.K, R)
        MA = np.einsum("ikr,kr->ir", YB3, C)
        hals(A, MA, (B.T @ B) * (C.T @ C))
        AC = (A[:, None, :] * C[None, :, :]).reshape(T.I * T.K, R)
        MB = T.Yt @ AC
        hals(B, MB, (A.T @ A) * (C.T @ C))
        YB3 = (T.Y @ B).reshape(T.I, T.K, R)
        MC = np.einsum("ikr,ir->kr", YB3, A)
        hals(C, MC, (A.T @ A) * (B.T @ B))
        # MC is the mode-3 MTTKRP with the current A, B -> <X, Xhat> = sum(MC * C) exactly
        inner = float((MC * C).sum())
        nrm = float(((A.T @ A) * (B.T @ B) * (C.T @ C)).sum())
        err = np.sqrt(max(T.sq - 2 * inner + nrm, 0) / T.sq)
        # normalise B, C to unit columns; scale A (model unchanged)
        nb, nc = np.linalg.norm(B, axis=0), np.linalg.norm(C, axis=0)
        B /= nb
        C /= nc
        A *= nb * nc
        err_hist.append(err)
        if it > 5 and abs(err_hist[-2] - err) < tol:
            break
    lam = np.linalg.norm(A, axis=0)
    order = np.argsort(-lam)
    return {"R": R, "seed": seed, "err": float(err_hist[-1]), "n_iter": it + 1, "lam": lam[order],
            "A": (A / lam)[:, order].astype(np.float32), "B": B[:, order].astype(np.float32), "C": C[:, order].astype(np.float32)}


def kruskal_sim(m1: dict, m2: dict) -> float:
    """Williams 2018 / tensortools similarity: Hungarian-matched product of factor cosines x lambda penalty."""
    from scipy.optimize import linear_sum_assignment

    S = np.ones((m1["R"], m2["R"]))
    for k in ("A", "B", "C"):
        a = m1[k] / np.maximum(np.linalg.norm(m1[k], axis=0), 1e-12)
        b = m2[k] / np.maximum(np.linalg.norm(m2[k], axis=0), 1e-12)
        S *= a.T @ b
    l1, l2 = m1["lam"][:, None], m2["lam"][None, :]
    S *= 1 - np.abs(l1 - l2) / np.maximum(np.maximum(l1, l2), 1e-12)
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].sum() / max(m1["R"], m2["R"]))


_TT: dict = {}


def _tensor(b: str) -> SpTensor:
    if b not in _TT:
        _TT.clear()
        X = block(b)
        _TT[b] = SpTensor(tensor_rows(X).astype(np.float64), len(meta()), NC)
    return _TT[b]


def a_fit_job(job):
    b, R, seed = job
    t0 = time.time()
    T = _tensor(b)
    fit = cp_hals(T, R, seed)
    fit["sec"] = time.time() - t0
    fit["block"] = b
    d = CACHE / "tca" / b
    d.mkdir(parents=True, exist_ok=True)
    pickle.dump(fit, open(d / f"R{R:02d}_s{seed}.pkl", "wb"))
    return b, R, seed, fit["err"], fit["n_iter"], fit["sec"]


def a_blocks() -> list[str]:
    return VLM_SUB + EXP_SUB + ["whole_sub"]


def _a_block_tensor_name(b: str) -> str:
    return b


def stage_a_fit(blocks: list[str] | None = None, ranks=A_RANKS, seeds=A_SEEDS) -> None:
    blocks = blocks or (VLM_SUB + EXP_SUB)
    jobs = [(b, R, s) for b in blocks for R in ranks for s in range(seeds)
            if not (CACHE / "tca" / b / f"R{R:02d}_s{s}.pkl").exists()]
    # biggest ranks first; group by block so each worker keeps its tensor
    jobs.sort(key=lambda j: (j[0], -j[1], j[2]))
    print(f"TCA fits to run: {len(jobs)}", flush=True)
    with ProcessPoolExecutor(WORKERS) as ex:
        for n, r in enumerate(ex.map(a_fit_job, jobs, chunksize=1)):
            if n % 20 == 0:
                print(n, r, flush=True)


WHOLE["whole_sub"] = VLM_SUB + EXP_SUB  # whole-network TCA variant: the 10 subset layers, block-RMS scaled


def load_fits(b: str, R: int) -> list[dict]:
    return [pickle.load(open(p, "rb")) for p in sorted((CACHE / "tca" / b).glob(f"R{R:02d}_s*.pkl"))]


def a_select_job(b: str) -> dict:
    out = {"block": b, "ranks": {}}
    for R in A_RANKS:
        F = load_fits(b, R)
        if not F:
            continue
        errs = np.array([f["err"] for f in F])
        best = F[int(errs.argmin())]
        sims = [kruskal_sim(best, f) for j, f in enumerate(F) if j != int(errs.argmin())]
        out["ranks"][R] = {"err": errs.tolist(), "err_min": float(errs.min()), "sim": sims,
                           "sim_median": float(np.median(sims)) if sims else np.nan,
                           "n_iter": [f["n_iter"] for f in F]}
    ok = [R for R, v in out["ranks"].items() if v["sim_median"] >= 0.75]
    out["R_star"] = int(max(ok)) if ok else int(min(out["ranks"]))
    return out


def a_analyse_job(b: str) -> dict:
    """components of the best fit at R*: features, time profile, trial loadings, fallback question."""
    from scipy.stats import binomtest

    sel = load_json("a_select")[b]
    R = sel["R_star"]
    F = load_fits(b, R)
    fit = F[int(np.argmin([f["err"] for f in F]))]
    M = meta()
    inf = info(b)
    A = fit["A"] * fit["lam"]  # trial loadings incl. scale
    Bf, Cf = fit["B"], fit["C"]
    U = u40_ids(M)
    task_of = {i: M[i]["task"] for i in U}
    tasks = sorted(set(task_of.values()), key=lambda t: (t[0], int(t[1:])))
    # chance distribution for tag / task purity of 20 random features
    nf = len(inf["tag"])
    rp_tag, rp_task = [], []
    for _ in range(1000):
        s = RNG.choice(nf, 20, replace=False)
        rp_tag.append(Counter(inf["tag"][s]).most_common(1)[0][1] / 20)
        rp_task.append(Counter(inf["top_task"][s]).most_common(1)[0][1] / 20)
    comps = []
    labels_sets = {
        "suite": np.array([M[i]["task"][0] if M[i]["t027_set"] == "train" else "x" for i in range(len(M))]),
        "task": np.array([M[i]["task"] for i in range(len(M))]),
        "condition": np.array([cond_label(M[i]) for i in range(len(M))]),
    }
    gc = np.array([M[i]["grasp_call"] if M[i]["grasp_call"] is not None else np.nan for i in range(len(M))], float)
    rc = np.array([M[i]["release_call"] if M[i]["release_call"] is not None else np.nan for i in range(len(M))], float)
    for r in range(R):
        top = np.argsort(-Bf[:, r])[:20]
        tagc = Counter(inf["tag"][top])
        taskc = Counter(inf["top_task"][top])
        a = A[:, r]
        eta = {}
        for k, lab in labels_sets.items():
            idx = np.arange(len(M)) if k != "suite" else np.array(U)
            y, g = a[idx], lab[idx]
            tot = ((y - y.mean()) ** 2).sum()
            btw = sum(((y[g == v].mean() - y.mean()) ** 2) * (g == v).sum() for v in set(g))
            eta[k] = float(btw / tot) if tot > 0 else np.nan
        hi = np.argsort(-a)[: len(M) // 10]
        cond_mean = {c: float(a[labels_sets["condition"] == c].mean()) for c in sorted(set(labels_sets["condition"]))}
        task_mean = {t: float(np.mean([a[i] for i in U if task_of[i] == t])) for t in tasks}
        comps.append({
            "r": r, "lam": float(fit["lam"][r]),
            "top_feat": [{"layer": pretty(str(inf["layer"][j])), "f": int(inf["feat"][j]), "w": float(Bf[j, r]),
                          "tag": str(inf["tag"][j]), "top_task": str(inf["top_task"][j]), "phase": str(inf["phase"][j])} for j in top],
            "tag_purity": tagc.most_common(1)[0][1] / 20, "tag_mode": tagc.most_common(1)[0][0],
            "task_purity": taskc.most_common(1)[0][1] / 20, "task_mode": taskc.most_common(1)[0][0],
            "time": Cf[:, r].tolist(), "peak_call": int(Cf[:, r].argmax()),
            "hi_grasp_median": float(np.nanmedian(gc[hi])), "hi_release_median": float(np.nanmedian(rc[hi])),
            "hi_conditions": dict(Counter(labels_sets["condition"][hi]).most_common(5)),
            "hi_tasks": dict(Counter(labels_sets["task"][hi]).most_common(5)),
            "eta2": eta, "cond_mean": cond_mean, "task_mean": task_mean,
            "layers": dict(Counter(inf["layer"][top])) if len(set(inf["layer"])) > 1 else None,
        })
    # fallback question
    cent = {t: A[[i for i in U if task_of[i] == t]].mean(0) for t in tasks}

    def cos(x, y):
        return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))

    UA = A[U]
    Un = UA / np.maximum(np.linalg.norm(UA, axis=1, keepdims=True), 1e-12)
    Ut = np.array([task_of[i] for i in U])
    fb = []
    for p in pool():
        m = p["m"]
        I, E = p["instructed"], [e for e in p["executed"] if e in cent]
        if I not in cent or not E:
            continue
        a = A[m["i"]]
        cI, cE = cos(a, cent[I]), max(cos(a, cent[e]) for e in E)
        nn = np.argsort(-(Un @ (a / max(np.linalg.norm(a), 1e-12))))[:5]
        fb.append({"src": p["src"], "cI": cI, "cE": cE, "nnE": int(sum(Ut[j] in E for j in nn)), "nnI": int(sum(Ut[j] == I for j in nn))})
    closerE = np.array([f["cE"] > f["cI"] for f in fb])
    res_fb = {"n": len(fb), "frac_closer_E": float(closerE.mean()),
              "p_binom": float(binomtest(int(closerE.sum()), len(closerE), 0.5, alternative="greater").pvalue),
              "median_cE_minus_cI": float(np.median([f["cE"] - f["cI"] for f in fb])),
              "nn5_E_mean": float(np.mean([f["nnE"] for f in fb])), "nn5_I_mean": float(np.mean([f["nnI"] for f in fb]))}
    for src in ("cand", "l90"):
        s = np.array([f["cE"] > f["cI"] for f in fb if f["src"] == src])
        res_fb[f"frac_closer_E_{src}"] = float(s.mean()) if len(s) else np.nan
        res_fb[f"n_{src}"] = int(len(s))
    tp = np.array([c["task_purity"] for c in comps])
    gp = np.array([c["tag_purity"] for c in comps])
    return {"block": b, "R_star": R, "err": fit["err"], "n_feat": nf, "comps": comps, "fallback": res_fb, "fb_trials": fb,
            "task_purity_median": float(np.median(tp)), "task_purity_chance_p95": float(np.percentile(rp_task, 95)),
            "task_purity_chance_median": float(np.median(rp_task)),
            "tag_purity_median": float(np.median(gp)), "tag_purity_chance_median": float(np.median(rp_tag)),
            "frac_comp_task_purity_gt_p95": float(np.mean(tp > np.percentile(rp_task, 95))),
            "trial_loadings": A.astype(np.float32)}


def stage_a() -> None:
    blocks = VLM_SUB + EXP_SUB
    stage_a_fit(blocks)
    sel = pmap(a_select_job, blocks + (["whole_sub"] if (CACHE / "tca" / "whole_sub").exists() else []))
    save_json("a_select", {s["block"]: s for s in sel})
    res = pmap(a_analyse_job, [s["block"] for s in sel])
    pickle.dump(res, open(CACHE / "a_res.pkl", "wb"))
    out = {}
    for r in res:
        out[r["block"]] = {k: v for k, v in r.items() if k not in ("trial_loadings", "fb_trials")}
    save_json("a", out)
    from scipy.stats import spearmanr

    ps = [out[b]["fallback"]["p_binom"] for b in out]
    q = bh(ps)
    for b, qq in zip(out, q):
        out[b]["fallback"]["q_bh"] = float(qq)
    vi = [b for b in VLM_SUB if b in out]
    rho = spearmanr([int(b[1:]) for b in vi], [1 - out[b]["fallback"]["frac_closer_E"] for b in vi])
    out["_summary"] = {"H_a2_spearman_depth_vs_fracI": [float(rho.statistic), float(rho.pvalue)]}
    save_json("a", out)
    for b in out:
        if b.startswith("_"):
            continue
        o = out[b]
        print(b, "R*", o["R_star"], "err", round(o["err"], 3), "fb closerE", round(o["fallback"]["frac_closer_E"], 3),
              "q", round(o["fallback"]["q_bh"], 4), "task purity", o["task_purity_median"], "chance p95", o["task_purity_chance_p95"])


# ============================================================================ b: variance partition
B_WINDOWS = {"0-2": (0, 2), "3-9": (3, 9), "10-19": (10, 19)}
B_LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
B_FACTORS = ("time", "I", "E", "scene")


def b_trials() -> list[dict]:
    """U40 (I = E = own task) + fallback pool (T032 I, E)."""
    M = meta()
    out = []
    for i in u40_ids(M):
        m = M[i]
        out.append({"i": i, "I": m["task"], "E": [m["task"]], "sent": m["task"], "scene": m["scene"], "fb": False,
                    "init": m["init"], "src": "U40"})
    for p in pool():
        m = p["m"]
        out.append({"i": m["i"], "I": p["instructed"], "E": list(p["executed"]), "sent": m["variant"], "scene": m["scene"],
                    "fb": True, "init": m["init"], "src": p["src"]})
    return out


def b_design(trials: list[dict], c0: int, c1: int, inst_key: str = "I") -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """one-hot factor blocks (main effect + interaction with call); rows = (trial, call)."""
    calls = np.arange(c0, c1 + 1)
    W = len(calls)
    n = len(trials) * W
    cidx = np.tile(np.arange(W), len(trials))
    tr = np.repeat(np.arange(len(trials)), W)
    T1 = np.eye(W)[cidx]

    def cat(levels_per_trial: list[list[str]]) -> np.ndarray:
        lev = sorted({x for ls in levels_per_trial for x in ls})
        H = np.zeros((len(trials), len(lev)))
        for k, ls in enumerate(levels_per_trial):
            for x in ls:
                H[k, lev.index(x)] = 1.0
        Hr = H[tr]
        inter = (Hr[:, :, None] * T1[:, None, :]).reshape(n, -1)
        return np.concatenate([Hr, inter], 1)

    blocks = {"time": T1, "I": cat([[t[inst_key]] for t in trials]), "E": cat([t["E"] for t in trials]),
              "scene": cat([[t["scene"]] for t in trials])}
    rows = np.array([trials[k]["i"] for k in tr])
    rows = ridx()[rows, calls[cidx]]
    init = np.array([trials[k]["init"] for k in tr])
    fb = np.array([trials[k]["fb"] for k in tr])
    return blocks, rows, init, fb


def _ymat(b: str, rows: np.ndarray) -> sp.csr_matrix:
    if b in WHOLE:
        return sp.hstack([block(x)[rows] / info(x)["rms"] for x in WHOLE[b]], format="csr")
    return block(b)[rows]


def cv_r2_models(X: np.ndarray, Y: sp.csr_matrix, folds: np.ndarray, lams, sub: np.ndarray, models: dict) -> dict:
    """ridge CV R^2 for several column subsets of X and several lambdas, using only p x p statistics per fold.
    models: name -> column index array. returns {(name, lam): (sse, sst, sse_sub, sst_sub, per-fold R2 list)}."""
    acc: dict = {}
    for f in np.unique(folds):
        te, trn = folds == f, folds != f
        xm = X[trn].mean(0)
        Xc, Xt = X[trn] - xm, X[te] - xm
        Ytr = Y[trn]
        ym = np.asarray(Ytr.mean(0)).ravel()
        Ctr = np.asarray((Ytr.T @ Xc).T)  # p x nf  (Xc is centred so the mean of Y drops out)
        Ytc = Y[te].toarray().astype(np.float64) - ym
        s = sub[te]
        Cte, Cts = Xt.T @ Ytc, Xt[s].T @ Ytc[s]
        A = Ctr @ Ctr.T
        Bx, Bs = Ctr @ Cte.T, Ctr @ Cts.T
        Ste, Sts = Xt.T @ Xt, Xt[s].T @ Xt[s]
        Sxx = Xc.T @ Xc
        sst, ssts = float((Ytc**2).sum()), float((Ytc[s] ** 2).sum())
        for name, m in models.items():
            ix = np.ix_(m, m)
            for lam in lams:
                G = np.linalg.inv(Sxx[ix] + lam * np.eye(len(m)))
                cross, crs = float((G * Bx[ix]).sum()), float((G * Bs[ix]).sum())
                quad = float(((G @ Ste[ix] @ G) * A[ix]).sum())
                quads = float(((G @ Sts[ix] @ G) * A[ix]).sum())
                sse, sses = sst - 2 * cross + quad, ssts - 2 * crs + quads
                r = acc.setdefault((name, lam), [0.0, 0.0, 0.0, 0.0, []])
                r[0] += sse
                r[1] += sst
                r[2] += sses
                r[3] += ssts
                r[4].append(1 - sse / sst)
    return acc


def b_job(job) -> dict:
    b, wname, inst_key = job
    c0, c1 = B_WINDOWS[wname]
    trials = b_trials()
    blocks, rows, init, fb = b_design(trials, c0, c1, inst_key)
    Y = _ymat(b, rows)
    full = np.concatenate([blocks[k] for k in B_FACTORS], 1)
    edges = np.cumsum([0] + [blocks[k].shape[1] for k in B_FACTORS])
    cols = {k: np.arange(edges[j], edges[j + 1]) for j, k in enumerate(B_FACTORS)}
    models = {"full": np.arange(full.shape[1])}
    for k in B_FACTORS:
        models[f"-{k}"] = np.concatenate([cols[j] for j in B_FACTORS if j != k])
        models[f"only_{k}"] = cols[k]
    acc = cv_r2_models(full, Y, init, B_LAMBDAS, fb, models)

    def r2(name, lam):
        a = acc[(name, lam)]
        return 1 - a[0] / a[1], (1 - a[2] / a[3]) if a[3] > 0 else np.nan

    scores = {lam: r2("full", lam)[0] for lam in B_LAMBDAS}
    lam = max(scores, key=scores.get)
    r_all, r_fb = r2("full", lam)
    out = {"block": b, "window": wname, "inst": inst_key, "lambda": lam, "lambda_scores": scores, "R2_full": r_all,
           "R2_full_fb": r_fb, "R2_full_folds": acc[("full", lam)][4], "unique": {}, "unique_fb": {}, "alone": {}, "alone_fb": {},
           "n_rows": int(len(rows)), "n_fb_rows": int(fb.sum())}
    for k in B_FACTORS:
        ra, rf = r2(f"-{k}", lam)
        out["unique"][k], out["unique_fb"][k] = r_all - ra, r_fb - rf
        out["alone"][k], out["alone_fb"][k] = r2(f"only_{k}", lam)
    return out


def dpca_job(b: str) -> dict:
    """Kobak 2016 marginalisation on Goal cand variants with both FALLBACK and NEW trials: base / FALLBACK / NEW."""
    M = meta()
    byv = defaultdict(lambda: defaultdict(list))
    for m in M:
        if m["t027_set"] == "cand" and cand_class(m) in ("FALLBACK", "NEW"):
            byv[m["variant"]][cand_class(m)].append(m["i"])
    base_of = {m["variant"]: m["base"] for m in M if m["t027_set"] == "cand"}
    U = defaultdict(list)
    for i in u40_ids(M):
        U[M[i]["task"]].append(i)
    vs = [v for v in sorted(byv) if byv[v]["FALLBACK"] and byv[v]["NEW"] and base_of.get(v) in U]
    X = block(b)
    T = []
    for v in vs:
        T.append([dense_traj(X, U[base_of[v]]).mean(0), dense_traj(X, byv[v]["FALLBACK"]).mean(0),
                  dense_traj(X, byv[v]["NEW"]).mean(0)])
    T = np.array(T)  # (V, 3, NC, nf)
    mu = T.mean((0, 1, 2))
    Xc = T - mu
    tot = (Xc**2).sum()
    m_t = Xc.mean((0, 1))  # (NC, nf)
    m_kt = Xc.mean(0) - m_t  # condition + condition x time
    m_k = m_kt.mean(1)  # condition main effect (time-averaged)
    m_v = Xc.mean(1) - m_t  # variant + variant x time (nuisance)
    resid = Xc - m_t[None, None] - m_kt[None] - m_v[:, None]
    V = len(vs)
    share = {"time": float((m_t**2).sum() * V * 3 / tot), "condition": float((m_kt**2).sum() * V / tot),
             "condition_main": float((m_k**2).sum() * V * NC / tot),
             "variant": float((m_v**2).sum() * 3 / tot), "cond_x_variant": float((resid**2).sum() / tot)}
    # condition marginal: distance between condition means per call (which pair carries the condition variance)
    cm = Xc.mean(0)  # (3, NC, nf)
    pair = {f"{a}-{c}": np.linalg.norm(cm[i] - cm[j], axis=1).tolist()
            for (i, a), (j, c) in [((0, "base"), (1, "FB")), ((0, "base"), (2, "NEW")), ((1, "FB"), (2, "NEW"))]}
    # first demixed axis of the condition marginal (PCA of m_kt) and its explained share of the condition marginal
    s = np.linalg.svd(m_kt.reshape(-1, m_kt.shape[-1]), compute_uv=False)
    return {"block": b, "variants": vs, "V": V, "share": share, "pair_dist": pair,
            "cond_pc_share": (s[:3] ** 2 / (s**2).sum()).tolist()}


def stage_b_dpca() -> None:
    """recompute only the dPCA marginalisation (fix: condition main effect averaged over time, not over conditions)."""
    out = load_json("b")
    blocks = VLM_ALL + EXP_ALL + list(k for k in WHOLE if k != "whole_sub")
    for r in pmap(dpca_job, blocks):
        out[r["block"]]["dpca"] = r
    save_json("b", out)


def stage_b() -> None:
    blocks = VLM_ALL + EXP_ALL + list(k for k in WHOLE if k != "whole_sub")
    jobs = [(b, w, "I") for b in blocks for w in B_WINDOWS] + [(b, w, "sent") for b in VLM_SUB + EXP_SUB + ["whole"] for w in B_WINDOWS]
    # whole-network kernels are the most expensive -> first
    jobs.sort(key=lambda j: j[0] not in WHOLE)
    res = pmap(b_job, jobs)
    out = defaultdict(dict)
    for r in res:
        out[r["block"]][f"{r['inst']}/{r['window']}"] = r
    dres = pmap(dpca_job, blocks)
    for r in dres:
        out[r["block"]]["dpca"] = r
    from scipy.stats import spearmanr

    summ = {}
    for w in B_WINDOWS:
        vi = [b for b in VLM_MAIN]
        ui = [out[b][f"I/{w}"]["unique_fb"]["I"] for b in vi]
        rho = spearmanr([int(b[1:]) for b in vi], ui)
        summ[f"H_b2_vlm_uniqueI_fb_depth_spearman/{w}"] = [float(rho.statistic), float(rho.pvalue)]
        summ[f"H_b1_E_gt_I_fb_main_layers/{w}"] = {
            "n_main": len(VLM_MAIN + EXP_MAIN),
            "n_E_gt_I": int(sum(out[b][f"I/{w}"]["unique_fb"]["E"] > out[b][f"I/{w}"]["unique_fb"]["I"] for b in VLM_MAIN + EXP_MAIN))}
        tv = [out[b][f"I/{w}"]["unique"]["time"] / max(out[b][f"I/{w}"]["R2_full"], 1e-9) for b in VLM_MAIN]
        te = [out[b][f"I/{w}"]["unique"]["time"] / max(out[b][f"I/{w}"]["R2_full"], 1e-9) for b in EXP_MAIN]
        summ[f"H_b3_time_share_of_full/{w}"] = {"vlm_median": float(np.median(tv)), "exp_median": float(np.median(te))}
    out["_summary"] = summ
    save_json("b", out)
    for b in blocks:
        r = out[b]["I/3-9"]
        print(pretty(b), "R2", round(r["R2_full"], 3), "fb", round(r["R2_full_fb"], 3), "lam", r["lambda"],
              {k: round(v, 4) for k, v in r["unique_fb"].items()}, "dpca", {k: round(v, 3) for k, v in out[b]["dpca"]["share"].items()})
    print(json.dumps(_js(summ), indent=1))


# ============================================================================ c: composition NNLS
def roles() -> dict:
    return {r["cand"]: r for r in csv.DictReader(open(lp.ROLES))}


def c_sources(rl: dict) -> dict:
    out = {}
    for col, key in (("base_task", "base"), ("object_source", "object"), ("destination_source", "destination")):
        xs = [x for x in (rl.get(col) or "").split(";") if x]
        if xs:
            out[key] = f"G{xs[0]}"
    return out


def strat_perm(vals: np.ndarray, lab: np.ndarray, strata, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """mean(vals[lab]) - mean(vals[~lab]); two-sided p from label permutation within strata (T029 strat_perm)."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(vals)
    vals, lab, strata = vals[ok], lab[ok], np.asarray(strata, dtype=object)[ok]
    if lab.sum() < 2 or (~lab).sum() < 2:
        return float("nan"), float("nan")
    obs = vals[lab].mean() - vals[~lab].mean()
    groups = [np.where(strata == s)[0] for s in set(strata)]
    cnt = 0
    lp_ = lab.copy()
    for _ in range(n):
        for g in groups:
            lp_[g] = rng.permutation(lab[g])
        if lp_.sum() == 0 or lp_.sum() == len(lp_):
            continue
        cnt += abs(vals[lp_].mean() - vals[~lp_].mean()) >= abs(obs) - 1e-12
    return float(obs), float((cnt + 1) / (n + 1))


def _bin(traj: np.ndarray, nb: int = 20) -> np.ndarray:
    t = np.linspace(0, 1, len(traj))
    tb = np.linspace(0, 1, nb)
    return np.stack([np.interp(tb, t, traj[:, j]) for j in range(traj.shape[1])], 1)


def c_job(job) -> dict:
    from scipy.optimize import nnls

    b, align = job  # align: "call" (primary, calls 0..27) or "prog" (T029: whole episode, 20 progress bins)
    rng = np.random.default_rng(1)
    M = meta()
    X = block(b)
    goal = [f"G{k}" for k in range(1, 11)]
    U = defaultdict(list)
    for i in u40_ids(M):
        if M[i]["task"] in goal:
            U[M[i]["task"]].append(i)

    def traj(i):
        m = M[i]
        if align == "call":
            return X[m["off"]: m["off"] + NC].toarray().astype(np.float64)
        return _bin(X[m["off"]: m["off"] + m["C"]].toarray().astype(np.float64), 20)

    cent = {t: np.mean([traj(i) for i in U[t]], 0) for t in goal}
    gm = np.mean([cent[t] for t in goal], 0)
    ro = roles()
    recs = []
    for m in M:
        if m["t027_set"] != "cand":
            continue
        s = c_sources(ro.get(m["variant"], {}))
        uniq = sorted(set(s.values()))
        if not uniq:
            continue
        e = traj(m["i"])
        T_ = len(e)
        out = {}
        for cen in ("centred", "raw"):
            g = gm if cen == "centred" else np.zeros_like(gm)
            Wt, r2 = [], []
            for c in range(T_):
                A = np.stack([cent[t][c] - g[c] for t in uniq], 1)
                y = e[c] - g[c]
                w, rn = nnls(A, y)
                Wt.append(w)
                r2.append(1 - rn**2 / max(y @ y, 1e-12))
            others = [t for t in goal if t not in uniq]
            rr = []
            for _ in range(100):
                pick = list(rng.choice(others, len(uniq), replace=False))
                q = []
                for c in range(T_):
                    A = np.stack([cent[t][c] - g[c] for t in pick], 1)
                    y = e[c] - g[c]
                    q.append(1 - nnls(A, y)[1] ** 2 / max(y @ y, 1e-12))
                rr.append(np.mean(q))
            out[cen] = {"W": np.array(Wt), "r2": np.array(r2), "r2_rand": float(np.mean(rr))}
        Wt = out["centred"]["W"]
        tau = np.linspace(0, 1, T_)
        rec = {"i": m["i"], "variant": m["variant"], "outcome": m["outcome"], "cls": cand_class(m), "success": m["success"],
               "init": m["init"], "sources": s, "uniq": uniq,
               "r2": float(out["centred"]["r2"].mean()), "r2_rand": out["centred"]["r2_rand"],
               "r2_raw": float(out["raw"]["r2"].mean()), "r2_rand_raw": out["raw"]["r2_rand"],
               "r2_call": out["centred"]["r2"].astype(np.float32), "W": Wt.astype(np.float32)}
        if "object" in s and "destination" in s and s["object"] != s["destination"]:
            wo, wd = Wt[:, uniq.index(s["object"])], Wt[:, uniq.index(s["destination"])]
            rec["slope"] = float(np.polyfit(tau, wd - wo, 1)[0])
            rec["w_obj"], rec["w_dest"] = wo.astype(np.float32), wd.astype(np.float32)
        if "base" in s:
            rec["base_share"] = float(np.mean(Wt[:, uniq.index(s["base"])] / (Wt.sum(1) + 1e-12)))
            rec["w_base"] = Wt[:, uniq.index(s["base"])].astype(np.float32)
        recs.append(rec)
    from scipy.stats import wilcoxon

    res = {"block": b, "align": align, "n_trials": len(recs)}
    byv = defaultdict(list)
    for r in recs:
        byv[r["variant"]].append(r["r2"] - r["r2_rand"])
    cm = np.array([np.mean(v) for v in byv.values()])
    res["r2_median"] = float(np.median([r["r2"] for r in recs]))
    res["r2_rand_median"] = float(np.median([r["r2_rand"] for r in recs]))
    res["r2_raw_median"] = float(np.median([r["r2_raw"] for r in recs]))
    res["r2_rand_raw_median"] = float(np.median([r["r2_rand_raw"] for r in recs]))
    res["diff_by_variant_median"] = float(np.median(cm))
    res["p_source_gt_random"] = float(wilcoxon(cm, alternative="greater").pvalue)
    sel = [r for r in recs if r["cls"] in ("FALLBACK", "NEW")]
    lab = np.array([r["cls"] == "NEW" for r in sel])
    for k in ("slope", "base_share", "r2"):
        v = np.array([r.get(k, np.nan) for r in sel], float)
        diff, p = strat_perm(v, lab, [r["variant"] for r in sel])
        res[f"{k}_NEW_minus_FB"], res[f"p_{k}_NEW_vs_FB"] = diff, p
        res[f"{k}_NEW_median"] = float(np.nanmedian(v[lab])) if np.isfinite(v[lab]).any() else None
        res[f"{k}_FB_median"] = float(np.nanmedian(v[~lab])) if np.isfinite(v[~lab]).any() else None
        res[f"n_{k}_NEW"], res[f"n_{k}_FB"] = int(np.isfinite(v[lab]).sum()), int(np.isfinite(v[~lab]).sum())
    labs = np.array([r["success"] == 1 for r in recs])
    for k in ("slope", "r2"):
        v = np.array([r.get(k, np.nan) for r in recs], float)
        diff, p = strat_perm(v, labs, [r["variant"] for r in recs])
        res[f"{k}_succ_minus_fail"], res[f"p_{k}_succ_vs_fail"] = diff, p
    # mean weight trajectories by class
    traj_ = {}
    for cls in ("FALLBACK", "NEW"):
        for k in ("w_obj", "w_dest", "w_base", "r2_call"):
            vv = [r[k] for r in recs if r["cls"] == cls and k in r]
            if vv:
                traj_[f"{cls}/{k}"] = np.mean(vv, 0).tolist()
                traj_[f"{cls}/{k}/n"] = len(vv)
    res["traj"] = traj_
    res["per_trial"] = [{k: v for k, v in r.items() if k not in ("W", "r2_call", "w_obj", "w_dest", "w_base")} for r in recs]
    return res


def stage_c() -> None:
    blocks = VLM_ALL + EXP_ALL + [k for k in WHOLE if k != "whole_sub"]
    jobs = [(b, "call") for b in blocks] + [("e17", "prog"), ("whole", "prog")]
    res = pmap(c_job, jobs)
    out = {}
    for r in res:
        out[r["block"] if r["align"] == "call" else f"{r['block']}/prog"] = r
    fams = {"vlm": VLM_ALL, "exp": EXP_ALL, "whole": [k for k in WHOLE if k != "whole_sub"]}
    for fam, bl in fams.items():
        for key in ("p_source_gt_random", "p_slope_NEW_vs_FB", "p_slope_succ_vs_fail"):
            q = bh([out[b][key] if np.isfinite(out[b][key]) else 1.0 for b in bl])
            for b, qq in zip(bl, q):
                out[b][key.replace("p_", "q_")] = float(qq)
    save_json("c", out)
    for b in out:
        r = out[b]
        print(b, "R2", round(r["r2_median"], 3), "rand", round(r["r2_rand_median"], 3), "q", round(r.get("q_source_gt_random", np.nan), 4),
              "slope NEW", r["slope_NEW_median"], "FB", r["slope_FB_median"], "p", round(r["p_slope_NEW_vs_FB"], 3),
              "succ-fail", round(r["slope_succ_minus_fail"], 3), round(r["p_slope_succ_vs_fail"], 3))


# ============================================================================ d: communication subspace
D_PAIRS = [("v04", "e05"), ("v05", "e05"), ("v07", "e08"), ("v08", "e08"), ("v10", "e11"), ("v11", "e11"),
           ("v13", "e14"), ("v14", "e14"), ("v16", "e17"), ("v17", "e17"), ("v04", "e17"), ("v15", "e05")]
D_REF = [("v01", "v02"), ("v04", "v05"), ("v08", "v09"), ("v12", "v13"), ("e05", "e06"), ("e08", "e09"), ("e11", "e12"), ("e14", "e15")]
D_LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
D_RMAX = 60
D_PCA = 500


def _pca_basis(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(0)
    C = Xc.T @ Xc
    w, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :k]


def d_mats(name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """dense (N_TOT, p) matrix and, for whole-network inputs, the orthonormal basis back to feature space."""
    if name in ("vlm_all", "exp_all"):
        # two-stage PCA: each block -> top 64 PCs, concatenated (block-RMS scaled), joint PCA to D_PCA
        parts, bases = [], []
        for x in WHOLE[name]:
            Xb = block(x).toarray() / info(x)["rms"]
            Vb = _pca_basis(Xb, 64)
            parts.append(Xb @ Vb)
            bases.append(Vb)
        Z = np.concatenate(parts, 1)
        Vz = _pca_basis(Z, D_PCA)
        Bd = sp.block_diag([sp.csr_matrix(v) for v in bases]).tocsr()
        W = np.asarray(Bd @ Vz)  # feature space (sum nf) x D_PCA, orthonormal columns
        return (Z @ Vz).astype(np.float64), W
    return block(name).toarray().astype(np.float64), None


def d_job(job) -> dict:
    xn, yn = job
    M = meta()
    X, Wx = d_mats(xn)
    Y, _ = d_mats(yn)
    init = np.concatenate([[m["init"]] * m["C"] for m in M])
    p, q = X.shape[1], Y.shape[1]
    # per-init sufficient statistics
    S = {}
    for k in range(10):
        r = init == k
        S[k] = (r.sum(), X[r].sum(0), Y[r].sum(0), X[r].T @ X[r], X[r].T @ Y[r])

    def train_stats(excl):
        n = sum(S[k][0] for k in S if k != excl)
        sx = sum(S[k][1] for k in S if k != excl)
        sy = sum(S[k][2] for k in S if k != excl)
        xx = sum(S[k][3] for k in S if k != excl)
        xy = sum(S[k][4] for k in S if k != excl)
        xm, ym = sx / n, sy / n
        return xm, ym, xx - n * np.outer(xm, xm), xy - n * np.outer(xm, ym)

    folds = []
    for f in range(10):
        xm, ym, Cxx, Cxy = train_stats(f)
        w, V = np.linalg.eigh(Cxx)
        VtCxy = V.T @ Cxy
        te = init == f
        Xte, Yte = X[te] - xm, Y[te] - ym
        sst = float((Yte**2).sum())
        fr = {"sst": sst, "full": {}, "rrr": {}}
        for lam in D_LAMBDAS:
            B = V @ (VtCxy / (w + lam)[:, None])
            P = Xte @ B
            fr["full"][lam] = 1 - float(((Yte - P) ** 2).sum()) / sst
            # RRR: principal axes of training fitted values  Yhat^T Yhat = B^T Cxx B
            G = B.T @ (Cxx @ B)
            ev, Vy = np.linalg.eigh(G)
            Vy = Vy[:, ::-1][:, :D_RMAX]
            Z, Q = P @ Vy, Yte @ Vy
            res = sst - np.cumsum(2 * (Q * Z).sum(0) - (Z**2).sum(0))
            fr["rrr"][lam] = (1 - res / sst).tolist()
        folds.append(fr)
    lam = max(D_LAMBDAS, key=lambda l_: np.mean([fr["full"][l_] for fr in folds]))
    full = np.array([fr["full"][lam] for fr in folds])
    rrr = np.array([fr["rrr"][lam] for fr in folds])  # (10, RMAX)
    thr = full.mean() - full.std(ddof=1) / np.sqrt(len(full))
    ok = np.flatnonzero(rrr.mean(0) >= thr)
    dim = int(ok[0] + 1) if len(ok) else D_RMAX
    # subspace on all data
    xm, ym = X.mean(0), Y.mean(0)
    Xc = X - xm
    Cxx = Xc.T @ Xc
    Cxy = Xc.T @ (Y - ym)
    w, V = np.linalg.eigh(Cxx)
    B = V @ ((V.T @ Cxy) / (w + lam)[:, None])
    ev, Vy = np.linalg.eigh(B.T @ (Cxx @ B))
    Vy = Vy[:, ::-1][:, :dim]
    Qx, _ = np.linalg.qr(B @ Vy)  # p x dim, orthonormal (input coordinates)
    Qf = Wx @ Qx if Wx is not None else Qx  # feature coordinates
    out = {"x": xn, "y": yn, "lambda": lam, "R2_full_folds": full.tolist(), "R2_full": float(full.mean()),
           "R2_full_se": float(full.std(ddof=1) / np.sqrt(len(full))), "R2_rrr": rrr.mean(0).tolist(),
           "R2_rrr_se": (rrr.std(0, ddof=1) / np.sqrt(len(rrr))).tolist(), "comm_dim": dim, "p": p, "q": q}
    # which input features load on the subspace (feature-space leverage), by tag
    if xn in WHOLE or xn[0] == "v" or xn[0] == "e":
        inf = info(xn) if xn not in ("vlm_all", "exp_all") else info(xn)
        lev = (Qf**2).sum(1)
        tags = np.asarray(inf["tag"])
        base = lev.mean()
        by = {}
        rng = np.random.default_rng(2)
        perm = [rng.permutation(tags) for _ in range(1000)]
        for t in sorted(set(tags)):
            obs = lev[tags == t].mean() / base
            null = np.array([lev[pt == t].mean() / base for pt in perm])
            by[t] = {"n": int((tags == t).sum()), "ratio": float(obs), "p_hi": float((np.sum(null >= obs) + 1) / 1001),
                     "p_lo": float((np.sum(null <= obs) + 1) / 1001)}
        out["lev_by_tag"] = by
        top = np.argsort(-lev)[:20]
        out["top_lev"] = [{"layer": pretty(str(inf["layer"][j])), "f": int(inf["feat"][j]), "tag": str(inf["tag"][j]),
                           "top_task": str(inf["top_task"][j]), "lev": float(lev[j])} for j in top]
    # same-observation pairs at call 0: instruction-only difference in the input code
    pairs = json.load(open(lp.PAIRS))
    bypath = {m["path"]: m for m in M}
    pp = [(bypath[q_["cand"]], bypath[q_["base"]]) for q_ in pairs
          if q_.get("valid") and q_.get("rich_available") and q_["cand"] in bypath and q_["base"] in bypath]
    Dd = np.array([X[a["off"]] - X[b_["off"]] for a, b_ in pp])  # input coordinates
    if Wx is not None:  # use the full feature-space difference so that variance outside the PCA counts as outside
        Df = np.asarray(block_whole_rows(xn, np.array([a["off"] for a, _ in pp])) - block_whole_rows(xn, np.array([b_["off"] for _, b_ in pp])))
        Qs, dimspace = Qf, Df
    else:
        Qs, dimspace = Qx, Dd
    nrm = (dimspace**2).sum(1)
    okp = nrm > 0
    frac = ((dimspace[okp] @ Qs) ** 2).sum(1) / nrm[okp]
    rng = np.random.default_rng(3)
    rnd = []
    for _ in range(1000):
        Rq, _ = np.linalg.qr(rng.normal(size=(Qs.shape[0], dim)))
        rnd.append(float(np.mean(((dimspace[okp] @ Rq) ** 2).sum(1) / nrm[okp])))
    # top-dim PCs of the input
    if Wx is None:
        Vp = _pca_basis(X, dim)
    else:
        Vp = Wx[:, :dim]  # Wx columns are already ordered by joint-PCA variance
    pc = float(np.mean(((dimspace[okp] @ Vp) ** 2).sum(1) / nrm[okp]))
    out["pairs"] = {"n": int(okp.sum()), "frac_comm": float(frac.mean()), "frac_rand_mean": float(np.mean(rnd)),
                    "frac_rand_p95": float(np.percentile(rnd, 95)), "frac_topPC": pc,
                    "p_vs_rand": float((np.sum(np.array(rnd) >= frac.mean()) + 1) / 1001)}
    return out


def block_whole_rows(name: str, rows: np.ndarray) -> np.ndarray:
    return np.concatenate([block(x)[rows].toarray() / info(x)["rms"] for x in WHOLE[name]], 1)


def stage_d() -> None:
    jobs = [("vlm_all", "exp_all")] + D_PAIRS + D_REF
    res = pmap(d_job, jobs)
    out = {f"{pretty(r['x'])}->{pretty(r['y'])}": r for r in res}
    # H-d1: communication dim vs within-VLM reference at matched depth
    save_json("d", out)
    for k, r in out.items():
        print(k, "lam", r["lambda"], "R2 full", round(r["R2_full"], 3), "dim", r["comm_dim"], "R2@dim", round(r["R2_rrr"][r["comm_dim"] - 1], 3),
              "pairs", {kk: round(v, 3) if isinstance(v, float) else v for kk, v in r["pairs"].items()})


# ============================================================================ e: trajectory geometry
E_PERM = 1000


def _cosd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return 1 - (a * b).sum(-1) / np.maximum(na * nb, 1e-12)


def diverge(TA: np.ndarray, TB: np.ndarray, rng, nperm: int = E_PERM) -> dict:
    """TA (nA, NC, nf), TB (nB, NC, nf): per-call cosine distance of means, split-half floor of A, permutation p."""
    d = _cosd(TA.mean(0), TB.mean(0))
    nA = len(TA)
    fl = np.mean([_cosd(TA[s[: nA // 2]].mean(0), TA[s[nA // 2:]].mean(0)) for s in (rng.permutation(nA) for _ in range(200))], 0)
    allT = np.concatenate([TA, TB])
    cnt = np.zeros(NC)
    for _ in range(nperm):
        s = rng.permutation(len(allT))
        cnt += _cosd(allT[s[:nA]].mean(0), allT[s[nA:]].mean(0)) >= d - 1e-12
    p = (cnt + 1) / (nperm + 1)
    hit = np.flatnonzero((d > fl) & (p < 0.05))
    return {"d": d, "floor": fl, "p": p, "div_call": int(hit[0]) if len(hit) else None}


def e_job(b: str) -> dict:
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(4)
    M = meta()
    X = block(b)
    out = {"block": b}
    U = defaultdict(list)
    for i in u40_ids(M):
        U[M[i]["task"]].append(i)
    TR = {t: dense_traj(X, ids) for t, ids in U.items()}
    tmean = {t: v.mean(0) for t, v in TR.items()}
    # --- divergence: cand variant vs its base task
    byv = defaultdict(list)
    for m in M:
        if m["t027_set"] == "cand":
            byv[m["variant"]].append(m["i"])
    cv = {}
    for v, ids in sorted(byv.items()):
        base = M[ids[0]]["base"]
        if base not in TR:
            continue
        TC = dense_traj(X, ids)
        r = diverge(TR[base], TC, rng)
        cls = Counter(cand_class(M[i]) for i in ids)
        rec = {"base": base, "n": len(ids), "cls": dict(cls), "all": r}
        for c in ("FALLBACK", "NEW"):
            sub = [k for k, i in enumerate(ids) if cand_class(M[i]) == c]
            if len(sub) >= 3:
                rec[c] = diverge(TR[base], TC[sub], rng, nperm=500)
        cv[v] = rec
    out["cand_vs_base"] = cv
    # --- basket and g10 cases
    CS = lp.cases()
    bk = CS["basket_O9"]
    TB = {k: dense_traj(X, ids) for k, ids in bk.items()}
    out["basket_div"] = {f"{a}|{c}": diverge(TB[a], TB[c], rng) for a, c in
                         [("ctl (A100)", "near (A100)"), ("ctl (A100)", "opp (A100)"), ("ctl4090", "far20"), ("ctl4090", "far25"),
                          ("ctl (A100)", "ctl4090")]}
    g = CS["goal_g10"]
    TG = {k: dense_traj(X, ids) for k, ids in g.items()}
    out["g10_div"] = {f"g10|{t}": diverge(TG[t], TG["g10 (cand)"], rng) for t in ("G7", "G2")}
    # --- basket path geometry
    mean = {k: v.mean(0) for k, v in TB.items()}

    def half_floor(T):
        s = rng.permutation(len(T))
        h = len(T) // 2
        return T[s[:h]].mean(0) - T[s[h:]].mean(0)

    def geom(f):  # f maps a (NC, nf) mean to the vector used
        dn = f(mean["near (A100)"]) - f(mean["ctl (A100)"])
        d20 = f(mean["far20"]) - f(mean["ctl4090"])
        d25 = f(mean["far25"]) - f(mean["ctl4090"])
        dop = f(mean["opp (A100)"]) - f(mean["ctl (A100)"])
        dm = f(mean["ctl4090"]) - f(mean["ctl (A100)"])
        n = {k: float(np.linalg.norm(v)) for k, v in (("near", dn), ("far20", d20), ("far25", d25), ("opp", dop), ("machine", dm))}
        path = (np.linalg.norm(dn) + np.linalg.norm(d20 - dn) + np.linalg.norm(d25 - d20)) / max(np.linalg.norm(d25), 1e-12)
        t = np.clip(dop @ d25 / max(d25 @ d25, 1e-12), 0, 1)
        dev = float(np.linalg.norm(dop - t * d25) / max(np.linalg.norm(d25), 1e-12))
        cs = lambda a, c: float(a @ c / max(np.linalg.norm(a) * np.linalg.norm(c), 1e-12))  # noqa: E731
        fl = {k: float(np.mean([np.linalg.norm(f(half_floor(TB[k]) + 0 * mean[k])) for _ in range(50)]) / np.sqrt(2))
              for k in ("ctl (A100)", "ctl4090")}
        return {"norm": n, "ordered": bool(n["near"] < n["far20"] < n["far25"]), "path_ratio": float(path),
                "opp_dev": dev, "opp_proj_t": float(t), "cos": {"near,far25": cs(dn, d25), "far20,far25": cs(d20, d25), "opp,far25": cs(dop, d25),
                                                               "opp,near": cs(dop, dn), "machine,far25": cs(dm, d25)},
                "floor": fl}

    out["basket_geom"] = {"mean0-27": geom(lambda T: T.mean(0)), "mean3-19": geom(lambda T: T[3:20].mean(0)),
                          "per_call": [geom(lambda T, c=c: T[c]) for c in range(NC)]}
    # --- speed aligned to grasp / release
    sp_g, sp_r, sp_gU = defaultdict(list), defaultdict(list), defaultdict(list)
    uset = set(u40_ids(M))
    for m in M:
        Tm = X[m["off"]: m["off"] + m["C"]].toarray()
        v = np.linalg.norm(np.diff(Tm, axis=0), axis=1)  # v[c] = |x(c+1) - x(c)|
        for key, ev, dst in (("g", m["grasp_call"], sp_g), ("r", m["release_call"], sp_r)):
            if ev is None:
                continue
            for lag in range(-6, 11):
                c = ev + lag
                if 0 <= c < len(v):
                    dst[lag].append(v[c])
                    if key == "g" and m["i"] in uset:
                        sp_gU[lag].append(v[c])
    out["speed"] = {nm: {lag: float(np.mean(d[lag])) for lag in sorted(d)} for nm, d in (("grasp", sp_g), ("release", sp_r), ("grasp_U40", sp_gU))}
    for nm in ("grasp", "release", "grasp_U40"):
        s = out["speed"][nm]
        out["speed"][nm + "_peak_lag"] = int(max(s, key=s.get))
    # --- participation ratio per suite (U40 call vectors), via the Gram matrix
    pr = {}
    for su in "SOGL":
        ids = [i for t, v in U.items() if t[0] == su for i in v]
        Z = tensor_rows(X, ids).toarray().astype(np.float64)
        Z -= Z.mean(0)
        ev = np.linalg.eigvalsh(Z @ Z.T)
        ev = np.maximum(ev, 0)
        pr[su] = float(ev.sum() ** 2 / (ev**2).sum())
    out["PR"] = pr
    # --- fallback: distance to instructed vs executed task trajectory
    recs = []
    for p in pool():
        m = p["m"]
        I, E = p["instructed"], [e for e in p["executed"] if e in tmean]
        if I not in tmean or not E:
            continue
        x = X[m["off"]: m["off"] + NC].toarray()
        dI = _cosd(x, tmean[I])
        dE = np.min([_cosd(x, tmean[e]) for e in E], 0)
        recs.append({"src": p["src"], "dI": dI, "dE": dE})
    dd = np.array([r["dI"] - r["dE"] for r in recs])
    pv = np.array([wilcoxon(dd[:, c], alternative="greater").pvalue for c in range(NC)])
    hit = np.flatnonzero(pv < 0.05)
    fb = {"n": len(recs), "median_dI_minus_dE": np.median(dd, 0), "p": pv, "first_sig_call": int(hit[0]) if len(hit) else None,
          "frac_closer_E": (dd > 0).mean(0), "median_dI": np.median([r["dI"] for r in recs], 0), "median_dE": np.median([r["dE"] for r in recs], 0)}
    for src in ("cand", "l90"):
        s = np.array([r["dI"] - r["dE"] for r in recs if r["src"] == src])
        fb[f"frac_closer_E_{src}"] = (s > 0).mean(0)
    out["fallback"] = fb
    return out


def stage_e() -> None:
    blocks = [k for k in WHOLE if k != "whole_sub"] + VLM_ALL + EXP_ALL
    res = pmap(e_job, blocks)
    out = {r["block"]: r for r in res}
    pickle.dump(out, open(CACHE / "e_res.pkl", "wb"))
    summ = {}
    for b, r in out.items():
        cv = r["cand_vs_base"]
        dv = [v["all"]["div_call"] for v in cv.values()]
        summ[b] = {"cand_div_median": float(np.median([x if x is not None else NC for x in dv])), "cand_div_none": int(sum(x is None for x in dv)),
                   "cand_div_call0": int(sum(x == 0 for x in dv)), "n_variants": len(dv),
                   "fb_first_sig_call": r["fallback"]["first_sig_call"], "fb_frac_E_call0": float(r["fallback"]["frac_closer_E"][0]),
                   "fb_frac_E_mean3_19": float(np.mean(r["fallback"]["frac_closer_E"][3:20])),
                   "basket_ordered": r["basket_geom"]["mean0-27"]["ordered"], "basket_path": r["basket_geom"]["mean0-27"]["path_ratio"],
                   "basket_opp_dev": r["basket_geom"]["mean0-27"]["opp_dev"], "speed_peak_grasp": r["speed"]["grasp_peak_lag"],
                   "speed_peak_release": r["speed"]["release_peak_lag"], "PR": r["PR"]}
        for c in ("FALLBACK", "NEW"):
            xs = [v[c]["div_call"] for v in cv.values() if c in v]
            summ[b][f"cand_div_median_{c}"] = float(np.median([x if x is not None else NC for x in xs])) if xs else None
            summ[b][f"cand_div_n_{c}"] = len(xs)
    save_json("e", {"summary": summ, "detail": out})
    for b, s in summ.items():
        print(pretty(b), s)


def numbers_b() -> dict:
    B = load_json("b")
    out = {"summary": B["_summary"], "note": "unique R2 of time is 0 by construction (I/E/scene x call blocks span it); time is reported as R2 alone"}
    for b in VLM_ALL + EXP_ALL + [k for k in WHOLE if k != "whole_sub"]:
        o = {}
        for w in B_WINDOWS:
            r = B[b][f"I/{w}"]
            o[w] = {"R2_full_fb": r["R2_full_fb"], "uniq_fb_I": r["unique_fb"]["I"], "uniq_fb_E": r["unique_fb"]["E"],
                    "uniq_fb_scene": r["unique_fb"]["scene"], "alone_time": r["alone"]["time"]}
            if f"sent/{w}" in B[b]:
                o[w]["sentence_uniq_fb_I"] = B[b][f"sent/{w}"]["unique_fb"]["I"]
                o[w]["sentence_uniq_fb_E"] = B[b][f"sent/{w}"]["unique_fb"]["E"]
        o["dpca"] = B[b]["dpca"]["share"]
        out[pretty(b)] = o
    from scipy.stats import spearmanr

    for w in B_WINDOWS:
        tv = [B[b][f"I/{w}"]["alone"]["time"] for b in VLM_MAIN]
        te = [B[b][f"I/{w}"]["alone"]["time"] for b in EXP_MAIN]
        out["summary"][f"H_b3_time_alone_R2/{w}"] = {"vlm_median": float(np.median(tv)), "exp_median": float(np.median(te))}
        rho = spearmanr(range(len(EXP_MAIN)), [B[b][f"I/{w}"]["unique_fb"]["I"] for b in EXP_MAIN])
        out["summary"][f"exp_uniqueI_fb_depth_spearman/{w}"] = [float(rho.statistic), float(rho.pvalue)]
    return out


# ============================================================================ numbers
def stage_numbers() -> None:
    """compact numbers.json (all detail stays in results/*.json)."""
    N: dict = {"features_after_rare_removal": {pretty(b): r["n_keep"] for b, r in load_json("prep").items()}}
    if (RES / "a.json").exists():
        A = load_json("a")
        N["a"] = {"spearman_depth_vs_instructed_side": A["_summary"]["H_a2_spearman_depth_vs_fracI"]}
        for b, r in A.items():
            if b.startswith("_"):
                continue
            et = np.array([c["eta2"]["task"] for c in r["comps"]])
            modes = [c["tag_mode"] for c in r["comps"]]
            N["a"][pretty(b)] = {"R_star": r["R_star"], "err": r["err"], "fallback_frac_closer_executed": r["fallback"]["frac_closer_E"],
                                 "cand": r["fallback"]["frac_closer_E_cand"], "l90": r["fallback"]["frac_closer_E_l90"],
                                 "p_binom": r["fallback"]["p_binom"], "nn5_executed": r["fallback"]["nn5_E_mean"],
                                 "nn5_instructed": r["fallback"]["nn5_I_mean"], "n_comp_eta2_task_ge_0.8": int((et >= 0.8).sum()),
                                 "median_eta2_task": float(np.median(et)), "n_comp_phase_mode": modes.count("phase"),
                                 "n_comp_instruction_mode": modes.count("instruction"), "task_purity_median": r["task_purity_median"],
                                 "task_purity_chance_p95": r["task_purity_chance_p95"]}
    for st in "bcde":
        if (RES / f"{st}.json").exists() and f"numbers_{st}" in globals():
            N[st] = globals()[f"numbers_{st}"]()
    (OUT / "numbers.json").write_text(json.dumps(_js(N), indent=1, ensure_ascii=False))


STAGES = {"numbers": stage_numbers, "prep": stage_prep, "a_fit": stage_a_fit, "a": stage_a, "b": stage_b, "b_dpca": stage_b_dpca, "c": stage_c, "d": stage_d, "e": stage_e}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="+")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in args.stage:
        t0 = time.time()
        STAGES[s]()
        print(f"[{s}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
