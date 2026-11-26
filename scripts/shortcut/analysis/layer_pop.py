#!/usr/bin/env python
"""T030 layer_pop: layer-by-layer population-vector analysis of pi05lib on the T028 rich re-extraction (CPU only).

Stages (hypotheses / metrics / baselines: layer_pop_PREREG.md, also copied to the output folder)
  extract   per-layer memmap cache (all 1,560 episodes, all calls) + episode meta joined with T027 labels
  norm      per-channel z-score stats (U40 reference), massive-channel detection, fixed per-layer PCA bases
  dim       (2) participation ratio / top-PC variance per layer x representation
  traj      (1) per-layer trajectories: depth small multiples for representative cases + shape metrics
  dec       (3) decoding maps (task / suite / verb / object / destination / phase) + cross-generalisation
  beta      (4) T027 pair regression (semantics / motion / scene) per layer x phase
  div       (5) divergence maps (basket moved vs ctl, cand vs base)
  kauf      (6) Kaufman read-in potent fraction per VLM layer
  gal       (7) Gallego CCA / principal angles per layer (basket) + NNLS source-task combination (cand)
  mag       (8) magnitude-only vs vector summary (uses dec / beta / div outputs)
  numbers   numbers.json

Run (OPENBLAS ZEN/Haswell kernels segfault on this EPYC, see neuro_dyn.py):
  OPENBLAS_CORETYPE=SkylakeX .venv-nd/bin/python scripts/shortcut/analysis/layer_pop.py <stage|all>
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuro_dyn import bh, pf, row_space  # noqa: E402

RICH = Path("/outputs/VLA-shortcut-exps/rich_acts")
OUT = Path("/outputs/VLA-shortcut-exps/analysis/layer_pop")
CACHE = OUT / "_cache"
T027 = Path("/outputs/VLA-shortcut-exps/kubectl/rsa_decoding")
ROLES = Path("/outputs/VLA-shortcut-exps/kubectl/libero_goal_composition/9_composition_traits/roles.csv")
PAIRS = Path("/outputs/VLA-shortcut-exps/analysis/neuro_dyn/b/pairs.json")
WORKERS = 8
N_L = 18
MLP_LAYERS = (2, 5, 8, 11, 14, 17)
TOK_LAYERS = (0, 2, 5, 8, 11, 14, 17)
STEPS = (0, 4, 9)
SEGS = ("agentview", "wrist", "bos", "task", "state", "template")
N_TOK_PC = 32  # exp_tok_pc / vlm_tok_pc components kept in the cache
N_EDGE = 6  # first / last task tokens kept for the language token x channel representation


# ============================================================================ meta
def _int(x):
    return int(x) if x not in ("", None) else None


def build_meta() -> list[dict]:
    """rich manifest rows joined with T027 per-episode labels (events, motion, scene pixels, executed task)."""
    rows = list(csv.DictReader(open(RICH / "manifest.csv")))
    t027 = {n: pickle.load(open(T027 / "_cache" / f"eps_{n}.pkl", "rb")) for n in ("train", "cand", "l90", "bskA", "bskV")}
    idx = {}
    for e in t027["train"]:
        idx[("train", e["task"], e["init"])] = e
    for e in t027["cand"]:
        idx[("cand", e["task"], e["init"])] = e
    for n in ("l90", "bskA", "bskV"):
        for e in t027[n]:
            idx[(n, e["variant"], e["init"])] = e
    out, off = [], 0
    for r in rows:
        s, init = r["set"], int(r["init"])
        if s == "libero3" or (s == "goal" and r["condition"] == "train"):
            k = ("train", r["task"], init)
        elif s == "goal":
            k = ("cand", r["task"], init)
        elif s == "libero90":
            k = ("l90", r["variant"], init)
        elif s == "basket_a100":
            k = ("bskA", r["variant"], init)
        else:
            k = ("bskV", r["variant"], init)
        e = idx[k]
        C = int(r["n_calls"])
        m = {
            "i": len(out), "path": r["path"], "episode": r["episode"], "set": s, "condition": r["condition"],
            "task": r["task"], "variant": r["variant"], "init": init, "instruction": r["instruction"],
            "success": int(r["success"]), "outcome": r["outcome"], "C": C, "off": off,
            "machine": r["machine"], "replay_machine": r["replay_machine"],
            "t027_set": k[0], "pca_fit": int(r["pca_fit"]),
        }
        assert e["instruction"] == r["instruction"] and e["nc"] == C, (k, e["instruction"], r["instruction"])
        for f in ("gc", "rcs", "bound_call", "motion", "motion_act", "pix", "exec", "scene", "group"):
            m[f] = e.get(f)
        for f in ("base", "exec_other", "nearest_trained", "exec_L", "kind", "cond"):
            if f in e:
                m[f] = e[f]
        m["grasp_call"] = _int(r["grasp_call"])
        m["release_call"] = _int(r["release_call"])
        off += C
        out.append(m)
    return out


def meta() -> list[dict]:
    return pickle.load(open(CACHE / "meta.pkl", "rb"))


# ============================================================================ extract
SPEC = {  # name: (shape after Ntot, dtype); layout = (layer, Ntot, ...) so one layer is contiguous
    "vlm_seg": ((N_L, None, 6, 2048), np.float16),
    "exp_mean": ((N_L, None, 3, 1024), np.float16),
    "exp_mlp": ((len(MLP_LAYERS), None, 3, 4096), np.float16),
    "exp_tok": ((len(MLP_LAYERS), None, 3, 50, N_TOK_PC), np.float16),
    "vlm_tok": ((len(TOK_LAYERS), None, 2 * N_EDGE, N_TOK_PC), np.float32),
    "mag_vlm_h": ((None, N_L, 6), np.float32),
    "mag_vlm_mlp": ((None, N_L, 6), np.float32),
    "mag_exp_h": ((None, 10, N_L), np.float32),
    "mag_exp_mlp": ((None, 10, N_L), np.float32),
}


def _shape(name, ntot):
    return tuple(ntot if s is None else s for s in SPEC[name][0])


def _mm(name, mode="r"):
    return np.load(CACHE / f"{name}.npy", mmap_mode=mode)


def _edge_tokens(pc: np.ndarray, lseg: np.ndarray) -> np.ndarray:
    """pc (C, 7, N, k), lseg (C, N) -> (7, C, 2*N_EDGE, k): first / last N_EDGE task-segment tokens."""
    C = pc.shape[0]
    out = np.zeros((pc.shape[1], C, 2 * N_EDGE, pc.shape[-1]), np.float32)
    for c in range(C):
        ti = np.nonzero(lseg[c] == 3)[0]
        sel = np.concatenate([ti[:N_EDGE], ti[-N_EDGE:]])
        if len(sel) < 2 * N_EDGE:  # very short instruction: pad by repeating the last token
            sel = np.concatenate([sel, np.repeat(sel[-1:], 2 * N_EDGE - len(sel))])
        out[:, c] = pc[c, :, sel].transpose(1, 0, 2)
    return out


_BASES: dict = {}


def _project(x: np.ndarray, stream: str, layers, k: int) -> np.ndarray:
    """raw (C, nL, ..., D) -> (C, nL, ..., k) with the T028 bases."""
    if not _BASES:
        _BASES.update(np.load(RICH / "pca_bases.npz"))
    out = []
    for i, li in enumerate(layers):
        mean = _BASES[f"{stream}/L{li}/mean"]
        comp = _BASES[f"{stream}/L{li}/components"][:k]
        out.append(np.einsum("c...d,kd->c...k", x[:, i] - mean, comp))
    return np.stack(out, 1).astype(np.float32)


def extract_one(m: dict) -> int:
    z = np.load(RICH / m["path"])
    a, b = m["off"], m["off"] + m["C"]
    mm = {n: _mm(n, "r+") for n in SPEC}
    mm["vlm_seg"][:, a:b] = z["vlm_seg"].transpose(1, 0, 2, 3)
    mm["exp_mean"][:, a:b] = z["exp_mean"].transpose(2, 0, 1, 3)
    em = z["exp_mlp_mean"]
    if em.shape[2] == N_L:  # pilot files keep all 18 MLP layers
        em = em[:, :, list(MLP_LAYERS)]
    mm["exp_mlp"][:, a:b] = em.transpose(2, 0, 1, 3)
    if "exp_tok_pc" in z.files:
        etpc = z["exp_tok_pc"][..., :N_TOK_PC]
    else:  # pilot file: raw tokens, project with the T028 bases (same formula as rich_extract)
        etpc = _project(z["exp_tok"].astype(np.float32), "exp_tok", MLP_LAYERS, N_TOK_PC)
    mm["exp_tok"][:, a:b] = etpc.transpose(1, 0, 2, 3, 4)
    if "vlm_tok_pc" in z.files:
        vtpc = z["vlm_tok_pc"][..., :N_TOK_PC].astype(np.float32)
    else:
        vtpc = _project(z["vlm_tok"].astype(np.float32), "vlm_tok", TOK_LAYERS, N_TOK_PC)
    mm["vlm_tok"][:, a:b] = _edge_tokens(vtpc, z["lang_seg"])
    mm["mag_vlm_h"][a:b] = z["vlm_h"]
    mm["mag_vlm_mlp"][a:b] = z["vlm_mlp"]
    mm["mag_exp_h"][a:b] = z["exp_h"]
    mm["mag_exp_mlp"][a:b] = z["exp_mlp"]
    for v in mm.values():
        v.flush()
    assert np.isfinite(z["vlm_seg"][:, :, [0, 1, 3, 4]].astype(np.float32)).all(), m["path"]
    return m["C"]


def stage_extract() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    M = build_meta()
    ntot = M[-1]["off"] + M[-1]["C"]
    pickle.dump(M, open(CACHE / "meta.pkl", "wb"))
    for n, (_, dt) in SPEC.items():
        p = CACHE / f"{n}.npy"
        if not p.exists():
            np.lib.format.open_memmap(p, mode="w+", dtype=dt, shape=_shape(n, ntot)).flush()
    done = CACHE / "extract_done.txt"
    t = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        n = sum(ex.map(extract_one, M, chunksize=4))
    done.write_text(f"{len(M)} episodes, {n} calls, {time.time() - t:.0f} s\n")
    print(done.read_text())



# ============================================================================ representations
REPS: dict[str, dict] = {
    "v_task": {"src": "vlm_seg", "seg": [3], "layers": tuple(range(N_L))},
    "v_agent": {"src": "vlm_seg", "seg": [0], "layers": tuple(range(N_L))},
    "v_wrist": {"src": "vlm_seg", "seg": [1], "layers": tuple(range(N_L))},
    "v_state": {"src": "vlm_seg", "seg": [4], "layers": tuple(range(N_L))},
    "v_all6": {"src": "vlm_seg", "seg": [0, 1, 2, 3, 4, 5], "layers": tuple(range(N_L))},
    **{f"e_s{k}": {"src": "exp_mean", "step": i, "layers": tuple(range(N_L))} for i, k in enumerate(STEPS)},
    **{f"m_s{k}": {"src": "exp_mlp", "step": i, "layers": MLP_LAYERS} for i, k in enumerate(STEPS)},
    "et_s0": {"src": "exp_tok", "step": 0, "layers": MLP_LAYERS},
    "et_s9": {"src": "exp_tok", "step": 2, "layers": MLP_LAYERS},
    "vt": {"src": "vlm_tok", "layers": TOK_LAYERS},
}
REP_ORDER = list(REPS)
MASSIVE_RATIO = 20.0


def step_of(rep: str) -> int:
    return int(rep.split("_s")[1]) if "_s" in rep else 9


def raw(rep: str, layer: int, a0: int = 0, a1: int | None = None) -> np.ndarray:
    """(a1 - a0, D) float32 vectors of one representation at one layer (absolute layer number), rows a0:a1."""
    R = REPS[rep]
    li = R["layers"].index(layer)
    a = _mm(R["src"])[li]
    a1 = a.shape[0] if a1 is None else a1
    # basic slicing only: fancy indexing + dtype cast on a memmap is ~400x slower
    if R["src"] == "vlm_seg":
        seg = R["seg"]
        assert seg == list(range(seg[0], seg[-1] + 1))
        x = np.array(a[a0:a1, seg[0]:seg[-1] + 1]).astype(np.float32)
    elif R["src"] in ("exp_mean", "exp_mlp", "exp_tok"):
        x = np.array(a[a0:a1, R["step"]]).astype(np.float32)
    else:
        x = np.array(a[a0:a1]).astype(np.float32)
    return x.reshape(len(x), -1)


def _ranges(eps: list[dict] | None, chunk: int = 4096) -> list[tuple[int, int]]:
    """merged contiguous row ranges of the given episodes, split into chunks."""
    M = eps if eps is not None else meta()
    rs: list[list[int]] = []
    for a, b in sorted((m["off"], m["off"] + m["C"]) for m in M):
        if rs and a <= rs[-1][1]:
            rs[-1][1] = max(rs[-1][1], b)
        else:
            rs.append([a, b])
    return [(x, min(x + chunk, b)) for a, b in rs for x in range(a, b, chunk)]


def mag(rep: str, layer: int | None) -> np.ndarray:
    """T021 magnitude-only features. layer=None: all layers (the T021/T027 feature itself)."""
    R = REPS[rep]
    if R["src"] in ("vlm_seg", "vlm_tok"):
        seg = R.get("seg", [3])
        h, m = _mm("mag_vlm_h"), _mm("mag_vlm_mlp")
        sl = slice(None) if layer is None else [layer]
        x = np.concatenate([np.asarray(h[:, sl][:, :, seg]), np.asarray(m[:, sl][:, :, seg])], 1)
    else:
        k = step_of(rep)
        h, m = _mm("mag_exp_h"), _mm("mag_exp_mlp")
        sl = slice(None) if layer is None else [layer]
        x = np.concatenate([np.asarray(h[:, k, sl]), np.asarray(m[:, k, sl])], 1)
    return np.log(np.maximum(x.reshape(len(x), -1).astype(np.float32), 1e-6))


def rows_of(M: list[dict]) -> np.ndarray:
    return np.concatenate([np.arange(m["off"], m["off"] + m["C"]) for m in M])


def u40(M: list[dict]) -> list[dict]:
    return [m for m in M if m["t027_set"] == "train"]


_NORM: dict = {}


def norm_stats(rep: str, layer: int) -> dict:
    if not _NORM:
        z = np.load(CACHE / "norm.npz")
        _NORM.update({k: z[k] for k in z.files})
    k = f"{rep}/L{layer}"
    return {"mu": _NORM[k + "/mu"], "sd": _NORM[k + "/sd"], "massive": _NORM[k + "/massive"]}


def zview(x: np.ndarray, st: dict) -> np.ndarray:
    sd = np.where(st["sd"] > 1e-6, st["sd"], np.inf)
    y = (x - st["mu"]) / sd
    y[:, st["massive"]] = 0
    return y.astype(np.float32)


_LN = None


def rms_view(x: np.ndarray, rep: str, layer: int) -> np.ndarray:
    """what the next layer sees: per-vector RMSNorm (per segment for VLM), times (1 + w_ln^{l+1}) for VLM l < 17."""
    global _LN
    R = REPS[rep]
    if R["src"] == "vlm_seg":
        nseg = len(R["seg"])
        y = x.reshape(len(x), nseg, -1)
        y = y / np.sqrt((y**2).mean(-1, keepdims=True) + 1e-6)
        if layer < N_L - 1:
            if _LN is None:
                w = np.load(RICH / "weights.npz", allow_pickle=True)
                _LN = {j: w[f"vlm/L{j}/input_layernorm"].astype(np.float32) for j in range(N_L)}
            y = y * (1 + _LN[layer + 1])[None, None]
        return y.reshape(len(x), -1).astype(np.float32)
    return (x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6)).astype(np.float32)


def vectors(rep: str, layer: int, view: str = "z", eps: list[dict] | None = None) -> np.ndarray:
    """(Ntot, D) array; only the rows of `eps` (None = all) are filled, the rest is never touched (np.empty).

    view: z (main) | rms (robustness) | raw | mag (same-layer magnitude, z-scored on U40) | magall.
    Rows are converted chunk by chunk so untouched rows never become resident memory (the host is memory-pressured).
    """
    if view in ("mag", "magall"):
        x = mag(rep, layer if view == "mag" else None)
        r = rows_of(u40(meta()))
        mu, sd = x[r].mean(0), x[r].std(0)
        return ((x - mu) / np.where(sd > 1e-6, sd, np.inf)).astype(np.float32)
    rg = _ranges(eps)
    ntot = _mm("mag_exp_h").shape[0]
    D = raw(rep, layer, 0, 1).shape[1]
    X = np.empty((ntot, D), np.float32)
    st = norm_stats(rep, layer) if view == "z" else None
    for a, b in rg:
        x = raw(rep, layer, a, b)
        if view == "rms":
            x = rms_view(x, rep, layer)
        elif view == "z":
            x = zview(x, st)
        X[a:b] = x
    return X


def resample(x: np.ndarray, n: int) -> np.ndarray:
    """(T, D) -> (n, D), linear interpolation over normalised time."""
    T = len(x)
    if T == 1:
        return np.repeat(x, n, 0)
    pos = np.linspace(0, T - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (pos - i0)[:, None]
    return ((1 - w) * x[i0] + w * x[i1]).astype(np.float32)


def all_jobs(reps=None, with_layers=True):
    return [(r, li) for r in (reps or REP_ORDER) for li in REPS[r]["layers"]]


def pool_map(fn, jobs, workers=WORKERS):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


def _js(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return float(o) if np.isscalar(o) else str(o)


def save_json(name: str, obj) -> None:
    (OUT / "res").mkdir(parents=True, exist_ok=True)
    with open(OUT / "res" / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=1, default=_js)


def load_json(name: str):
    return json.load(open(OUT / "res" / f"{name}.json"))


# ============================================================================ norm
def norm_job(job):
    rep, layer = job
    x = raw(rep, layer)[rows_of(u40(meta()))].astype(np.float64)
    rms = np.sqrt((x**2).mean(0))
    massive = np.zeros(x.shape[1], bool)
    if REPS[rep]["src"] not in ("exp_tok", "vlm_tok"):
        massive = rms > MASSIVE_RATIO * np.median(rms)
    info = [{"ch": int(j), "rms": float(rms[j]), "max_abs": float(np.abs(x[:, j]).max()),
             "ratio_to_median_rms": float(rms[j] / np.median(rms))} for j in np.nonzero(massive)[0]]
    return rep, layer, x.mean(0).astype(np.float32), x.std(0).astype(np.float32), massive, info


def stage_norm() -> None:
    res = pool_map(norm_job, all_jobs())
    out, info = {}, {}
    for rep, layer, mu, sd, massive, inf in res:
        k = f"{rep}/L{layer}"
        out[k + "/mu"], out[k + "/sd"], out[k + "/massive"] = mu, sd, massive
        if inf:
            info[k] = inf
    np.savez(CACHE / "norm.npz", **out)
    save_json("massive_channels", info)
    print({k: [d["ch"] for d in v] for k, v in info.items() if not k.startswith("v_all6")})


# ============================================================================ labels
SEM_CSV = T027 / "semantics.csv"
PHASE4 = ("reach", "grasp", "carry", "release")
CALLS = (0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 24, 27)
FALLBACK = ("trained", "fail_trained")
NEW = ("new", "new_byproduct")


def read_sem() -> dict:
    return {r["task"]: {k: set(filter(None, r[k].split("|"))) for k in ("verb", "object", "destination")}
            for r in csv.DictReader(open(SEM_CSV))}


def primary(S: dict, t: str, key: str) -> str:
    v = sorted(S[t][key])
    return v[0] if v else "none"


def phase_labels(m: dict) -> list:
    """T027 per-call 4-phase label (training episodes with a grasp)."""
    lab: list = [None] * m["C"]
    gc = m["gc"]
    if gc is None:
        return lab
    rc = next((r for r in m["rcs"] if r is not None and r > gc), None)
    for c in range(m["C"]):
        if c < gc - 2:
            lab[c] = "reach"
        elif c <= gc + 1:
            lab[c] = "grasp"
        elif rc is not None and c < rc - 1:
            lab[c] = "carry"
        elif rc is not None and c <= rc + 2:
            lab[c] = "release"
        elif rc is None:
            lab[c] = "carry"
    return lab


_LAB: dict = {}


def labels() -> dict:
    if _LAB:
        return _LAB
    M = meta()
    S = read_sem()
    U = u40(M)
    L = {"U": U, "off": np.array([m["off"] for m in U]), "task": np.array([m["task"] for m in U]),
         "suite": np.array([m["task"][0] for m in U]), "init": np.array([m["init"] for m in U])}
    for key, nm in (("verb", "verb"), ("object", "object"), ("destination", "dest")):
        y = np.array([primary(S, t, key) for t in L["task"]])
        ntask = Counter(dict(zip(L["task"], y)).values())
        L[nm] = np.array([v if ntask[v] >= 2 else None for v in y], dtype=object)
    L["phase"] = [phase_labels(m) for m in U]
    # cross-generalisation pools (T027 definitions)
    pool = []
    for m in M:
        if m["t027_set"] == "cand" and m.get("exec_other"):
            pool.append({"m": m, "src": "cand", "executed": list(m["exec_other"]), "instructed": m["base"]})
        elif (m["t027_set"] == "l90" and m.get("kind") == "lscene" and m["outcome"] == "trained" and m.get("exec_L")
              and m["nearest_trained"] not in m["exec_L"]):
            pool.append({"m": m, "src": "l90", "executed": list(m["exec_L"]), "instructed": m["nearest_trained"]})
    for p in pool:
        t = p["m"]["task"]
        for key, nm in (("object", "object"), ("destination", "dest")):
            p[f"{nm}_instr"] = primary(S, t, key) if t in S else None
            p[f"{nm}_exec"] = sorted({primary(S, e, key) for e in p["executed"] if e in S})
    L["pool"] = pool
    _LAB.update(L)
    return L


# ============================================================================ (3) decoding
def _std(A):
    mu, sd = A.mean(0), A.std(0)
    return mu, np.where(sd > 1e-6, sd, np.inf)


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xtes: list, ncomp: int = 50) -> list:
    """StandardScaler -> PCA (train fold) -> StandardScaler -> LogisticRegression(C=1); T027 classifier."""
    from sklearn.linear_model import LogisticRegression

    if len(set(ytr)) < 2:
        return [np.full(len(X), ytr[0], dtype=object) for X in Xtes]
    mu, sd = _std(Xtr)
    A = (Xtr - mu) / sd
    k = min(ncomp, len(A) - 1, A.shape[1])
    if A.shape[1] <= k:
        V = np.eye(A.shape[1], dtype=np.float32)
    elif len(A) > 2000:
        from sklearn.utils.extmath import randomized_svd

        V = randomized_svd(A, k, random_state=0)[2].T
    else:
        V = np.linalg.svd(A, full_matrices=False)[2][:k].T
    Z = A @ V
    m2, s2 = _std(Z)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit((Z - m2) / s2, ytr)
    return [clf.predict((((X - mu) / sd) @ V - m2) / s2) for X in Xtes]


def cv_pred(X, y, groups, n_splits=None):
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    cv = LeaveOneGroupOut() if n_splits is None else GroupKFold(n_splits)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y, groups):
        pred[te] = fit_predict(X[tr], y[tr], [X[te]])[0]
    return pred


def bacc(y, pred):
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(list(y), list(pred)))


def dec_job(job):
    rep, layer, view = job
    t0 = time.time()
    L = labels()
    X = vectors(rep, layer, view, L["U"] + [p["m"] for p in L["pool"]])
    calls = CALLS if view != "rms" else (0, 10, 20)
    out = {"rep": rep, "layer": layer, "view": view, "calls": list(calls), "acc": defaultdict(list), "xgen": defaultdict(list)}
    pool = L["pool"]
    for c in calls:
        Xc = X[L["off"] + c]
        out["acc"]["task40"].append(float(np.mean(cv_pred(Xc, L["task"], L["init"]) == L["task"])))
        out["acc"]["suite"].append(float(np.mean(cv_pred(Xc, L["suite"], L["init"]) == L["suite"])))
        w = {}
        for su in "SOGL":
            msk = L["suite"] == su
            w[su] = float(np.mean(cv_pred(Xc[msk], L["task"][msk], L["init"][msk]) == L["task"][msk]))
        out["acc"]["within"].append(float(np.mean(list(w.values()))))
        for su in "SOGL":
            out["acc"][f"within_{su}"].append(w[su])
        for nm in ("verb", "object", "dest"):
            msk = L[nm] != None  # noqa: E711
            y = L[nm][msk].astype(str)
            out["acc"][nm].append(float(np.mean(cv_pred(Xc[msk], y, L["task"][msk], n_splits=10) == y)))
        # cross-generalisation to fallback trials (same call)
        ok = [i for i, p in enumerate(pool) if p["m"]["C"] > c]
        Xp = X[np.array([pool[i]["m"]["off"] + c for i in ok])]
        pt = fit_predict(Xc, L["task"], [Xp])[0]
        hits: dict = {"task_exec": [None] * len(pool), "task_instr": [None] * len(pool)}
        for j, i in enumerate(ok):
            hits["task_exec"][i] = int(pt[j] in pool[i]["executed"])
            hits["task_instr"][i] = int(pt[j] == pool[i]["instructed"])
        for nm in ("object", "dest"):
            msk = L[nm] != None  # noqa: E711
            cls = set(L[nm][msk])
            pr = fit_predict(Xc[msk], L[nm][msk].astype(str), [Xp])[0]
            he, hi = [None] * len(pool), [None] * len(pool)
            for j, i in enumerate(ok):
                p = pool[i]
                ins, exe = p[f"{nm}_instr"], p[f"{nm}_exec"]
                if ins in cls and ins not in exe and any(e in cls for e in exe):
                    he[i], hi[i] = int(pr[j] in exe), int(pr[j] == ins)
            hits[f"{nm}_exec"], hits[f"{nm}_instr"] = he, hi
        for k, v in hits.items():
            out["xgen"][k].append(v)
    if view in ("z", "mag"):  # phase decoding: every 2nd call of U40, pooled
        rr, yy, gg = [], [], []
        for m, lab in zip(L["U"], L["phase"]):
            for c in range(0, m["C"], 2):
                if lab[c] is not None:
                    rr.append(m["off"] + c)
                    yy.append(lab[c])
                    gg.append(m["init"])
        yy = np.array(yy)
        pred = cv_pred(X[np.array(rr)], yy, np.array(gg))
        out["phase_bacc"] = bacc(yy, pred)
        out["phase_recall"] = {ph: float(np.mean(pred[yy == ph] == ph)) for ph in PHASE4}
    out["sec"] = time.time() - t0
    out["acc"], out["xgen"] = dict(out["acc"]), dict(out["xgen"])
    return out


def dec_jobs() -> list:
    jobs = [(r, li, "z") for r, li in all_jobs()]
    jobs += [(r, li, "mag") for r, li in all_jobs() if not r.startswith("et_") and r != "vt"]
    jobs += [(r, li, "rms") for r, li in all_jobs(["v_task", "v_agent", "v_all6", "e_s9", "m_s9"])]
    jobs += [(r, -1, "magall") for r in ("v_task", "v_agent", "v_wrist", "v_state", "v_all6", "e_s0", "e_s4", "e_s9")]
    size = {"v_all6": 6, "m_s0": 2, "m_s4": 2, "m_s9": 2}  # big jobs first for load balance
    return sorted(jobs, key=lambda j: -(size.get(j[0], 1) * (j[2] == "z")))


def run_parts(name: str, fn, jobs: list) -> dict:
    """run jobs in the pool, checkpointing results to _cache/<name>_parts.pkl (resumable)."""
    from concurrent.futures import as_completed

    part = CACHE / f"{name}_parts.pkl"
    done = pickle.load(open(part, "rb")) if part.exists() else {}
    todo = [j for j in jobs if j not in done]
    print(f"{name}: {len(jobs)} jobs, {len(todo)} to run", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fn, j): j for j in todo}
        for n, f in enumerate(as_completed(futs)):
            done[futs[f]] = f.result()
            if n % 10 == 0 or n == len(todo) - 1:
                pickle.dump(done, open(part, "wb"))
                print(f"  {n + 1}/{len(todo)} {futs[f]} {time.time() - t0:.0f}s", flush=True)
    return done


def stage_dec() -> None:
    done = run_parts("dec", dec_job, dec_jobs())
    save_json("dec", {"pool": [{"episode": p["m"]["episode"], "src": p["src"], "task": p["m"]["task"],
                                "executed": p["executed"], "instructed": p["instructed"]} for p in labels()["pool"]],
                      "jobs": list(done.values())})



# ============================================================================ (1) trajectories + (2) dimensionality
TRAJ_REPS = ("v_task", "v_agent", "v_all6", "e_s9", "m_s9")


def cases() -> dict:
    """representative cases for the depth small multiples (PREREG analysis 1)."""
    M = meta()
    pick = lambda f: [m["i"] for m in M if f(m)]  # noqa: E731
    out = {
        "basket_O9": {
            "ctl (A100)": pick(lambda m: m["set"] == "libero3" and m["task"] == "O9"),
            "near (A100)": pick(lambda m: m["set"] == "basket_a100" and m["condition"] == "near" and m["task"] == "O9"),
            "opp (A100)": pick(lambda m: m["set"] == "basket_a100" and m["condition"] == "opp" and m["task"] == "O9"),
            "ctl4090": pick(lambda m: m["set"] == "basket_far" and m["condition"] == "ctl4090" and m["task"] == "O9"),
            "far20": pick(lambda m: m["set"] == "basket_far" and m["condition"] == "far20" and m["task"] == "O9"),
            "far25": pick(lambda m: m["set"] == "basket_far" and m["condition"] == "far25" and m["task"] == "O9"),
        },
        "goal_g10": {
            "g10 (cand)": pick(lambda m: m["t027_set"] == "cand" and m["task"] == "g10"),
            "G7": pick(lambda m: m["t027_set"] == "train" and m["task"] == "G7"),
            "G2": pick(lambda m: m["t027_set"] == "train" and m["task"] == "G2"),
        },
    }
    cnt = Counter()
    single = defaultdict(Counter)
    for m in M:
        if m["t027_set"] == "l90" and m.get("kind") == "lscene" and m["outcome"] == "trained" and len(m.get("exec_L") or []) == 1:
            cnt[m["task"]] += 1
            single[m["task"]][m["exec_L"][0]] += 1
    best = sorted(cnt, key=lambda t: (-cnt[t], int(t[1:])))[0]
    lt = single[best].most_common(1)[0][0]
    out[f"l90_{best}"] = {
        f"{best} lscene, did {lt}": pick(lambda m: m["t027_set"] == "l90" and m["task"] == best and m["outcome"] == "trained"
                                         and (m.get("exec_L") or []) == [lt]),
        f"{best} lscene, other outcome": pick(lambda m: m["t027_set"] == "l90" and m["task"] == best
                                             and not (m["outcome"] == "trained" and (m.get("exec_L") or []) == [lt])),
        f"{lt} (train)": pick(lambda m: m["t027_set"] == "train" and m["task"] == lt),
    }
    out["spatial_S1"] = {"S1": pick(lambda m: m["t027_set"] == "train" and m["task"] == "S1")}
    out["long_L1"] = {"L1": pick(lambda m: m["t027_set"] == "train" and m["task"] == "L1")}
    return out


def _spectrum(X: np.ndarray, kmax: int = 300) -> tuple[np.ndarray, float, float]:
    """eigenvalues (descending; top-kmax if D is large), trace, squared Frobenius norm of the covariance."""
    X = X - X.mean(0)
    n, D = X.shape
    if D <= 4096:
        C = (X.T.astype(np.float64) @ X) / n
        ev = np.linalg.eigvalsh(C)[::-1]
        return np.maximum(ev, 0), float(np.trace(C)), float((C**2).sum())
    from sklearn.utils.extmath import randomized_svd

    G = X @ X.T if n < D else None
    if G is not None:
        ev = np.linalg.eigvalsh(G.astype(np.float64) / n)[::-1]
        return np.maximum(ev, 0), float(ev.sum()), float((ev**2).sum())
    s = randomized_svd(X, kmax, random_state=0, n_iter=4)[1]
    tr = float((X.astype(np.float64) ** 2).sum() / n)
    # ||C||_F^2 = ||X^T X||_F^2 / n^2, accumulated in blocks of columns
    fro = 0.0
    for a in range(0, D, 2048):
        B = X[:, a:a + 2048].T.astype(np.float64) @ X
        fro += float((B**2).sum())
    return s.astype(np.float64) ** 2 / n, tr, fro / n**2


def _dimstats(X: np.ndarray) -> dict:
    ev, tr, fro = _spectrum(X)
    cum = np.cumsum(ev) / tr
    return {"PR": tr**2 / fro, "top1": float(cum[0]), "top5": float(cum[4]), "top10": float(cum[9]),
            "top50": float(cum[min(49, len(cum) - 1)]),
            "n90": int(np.searchsorted(cum, 0.9) + 1) if cum[-1] >= 0.9 else f">{len(cum)}", "D": int(X.shape[1])}


def dim_job(job):
    rep, layer = job
    t0 = time.time()
    M = meta()
    U = u40(M)
    rows = rows_of(U)
    ep = np.concatenate([np.full(m["C"], k) for k, m in enumerate(U)])
    out: dict = {"rep": rep, "layer": layer}
    case_eps = [M[i] for cs in cases().values() for ids in cs.values() for i in ids]
    Xall = vectors(rep, layer, "z", U + case_eps)
    X = Xall[rows]
    out["z"] = _dimstats(X)
    cnt = np.bincount(ep).astype(np.float32)[:, None]
    means = np.zeros((len(U), X.shape[1]), np.float32)
    np.add.at(means, ep, X)
    means /= cnt
    out["z_between"] = _dimstats(means)
    out["z_within"] = _dimstats(X - means[ep])
    if rep in ("v_task", "v_agent", "v_wrist", "v_state", "v_all6", "e_s0", "e_s9", "m_s9"):
        out["raw"] = _dimstats(vectors(rep, layer, "raw", U)[rows])
        out["rms"] = _dimstats(vectors(rep, layer, "rms", U)[rows])
    # shape metrics (z-space)
    tot = float(((X - X.mean(0)) ** 2).sum(1).mean())
    within = float(((X - means[ep]) ** 2).sum(1).mean())
    speed, straight = [], []
    for m in U:
        x = Xall[m["off"]:m["off"] + m["C"]]
        st = np.linalg.norm(np.diff(x, axis=0), axis=1)
        speed.append(st.mean() / np.sqrt(tot))
        straight.append(np.linalg.norm(x[-1] - x[0]) / max(st.sum(), 1e-9))
    mu = X.mean(0)
    if rep in TRAJ_REPS:
        from sklearn.utils.extmath import randomized_svd

        _, s, vt = randomized_svd(X - mu, 2, random_state=0)
        out["pc12_var"] = float((s**2).sum() / len(X) / tot)
        proj = {}
        for cs in cases().values():
            for ids in cs.values():
                for i in ids:
                    m = M[i]
                    proj[i] = ((Xall[m["off"]:m["off"] + m["C"]] - mu) @ vt.T).astype(np.float32)
        out["proj"] = proj
    out["shape"] = {"time_var_frac": within / tot, "speed": float(np.median(speed)), "straightness": float(np.median(straight))}
    out["sec"] = time.time() - t0
    return out


def stage_dim() -> None:
    size = {"v_all6": 6, "m_s0": 3, "m_s4": 3, "m_s9": 3}
    jobs = sorted(all_jobs(), key=lambda j: -size.get(j[0], 1))
    done = run_parts("dim", dim_job, jobs)
    pickle.dump({k: v.pop("proj") for k, v in done.items() if "proj" in v}, open(CACHE / "traj_proj.pkl", "wb"))
    save_json("dim", {"cases": cases(), "jobs": list(done.values())})



# ============================================================================ (4) T027 pair regression per layer x phase
BETA_REPS = ("v_task", "v_agent", "v_all6", "e_s9", "m_s9")
BSETS = ("L90_lscene", "cand31", "basket_4090", "basket_A100")
NPERM_B = 199
NBIN = 5


def read_sem_full() -> dict:
    return {r["task"]: {k: set(filter(None, r[k].split("|"))) for k in ("verb", "object", "destination", "qualifier")}
            for r in csv.DictReader(open(SEM_CSV))}


def _jacc(a, b):
    return 0.0 if not a and not b else 1 - len(a & b) / len(a | b)


def sem_dist(ta: str, tb: str, S: dict) -> float:
    """T027 `sem` = mean of verb / object / destination Jaccard distances and qualifier mismatch."""
    A, B = S[ta], S[tb]
    return (_jacc(A["verb"], B["verb"]) + _jacc(A["object"], B["object"]) + _jacc(A["destination"], B["destination"])
            + float(A["qualifier"] != B["qualifier"])) / 4


def bset(name: str) -> list[dict]:
    """T027 bset() on the rich episodes; adds `exec_b` (executed-behaviour label for the 2x2 table)."""
    M = meta()
    if name == "basket_A100":
        eps = [m for m in M if m["set"] == "libero3" and m["task"][0] == "O"] + [m for m in M if m["set"] == "basket_a100"]
        for m in eps:
            m["exec_b"] = f"{m['task']}:ctlA" if m["set"] == "libero3" else m["exec"]
    elif name == "basket_4090":
        eps = [m for m in M if m["set"] == "basket_far"]
    elif name == "L90_lscene":
        l9 = [m for m in M if m["set"] == "libero90"]
        scenes = {m["scene"] for m in l9}
        eps = [m for m in M if m["t027_set"] == "train" and m["task"][0] == "L" and m["scene"] in scenes] + l9
    else:
        eps = [m for m in M if m["t027_set"] == "train" and m["task"][0] == "G"] + [m for m in M if m["t027_set"] == "cand"]
    for m in eps:
        m.setdefault("exec_b", m["exec"])
    return eps


def phase_vec(X: np.ndarray, b: int, phase: str) -> np.ndarray:
    pre = X[:b + 1]
    post = X[b + 1:] if b + 1 < len(X) else X[-1:]
    if phase == "pre":
        return resample(pre, NBIN).ravel()
    if phase == "post":
        return resample(post, NBIN).ravel()
    return np.concatenate([resample(pre, NBIN).ravel(), resample(post, NBIN).ravel()])


def eucl(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float64)
    sq = (A * A).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * A @ A.T, 0))


_BD: dict = {}


def b_design(name: str, phase: str):
    k = (name, phase)
    if k not in _BD:
        S = read_sem_full()
        eps = bset(name)
        tasks = sorted({m["task"] for m in eps})
        R = np.array([[sem_dist(a, b, S) for b in tasks] for a in tasks])
        li = np.array([tasks.index(m["task"]) for m in eps])
        mv = {"pre": slice(0, 10), "post": slice(10, 20), "traj": slice(0, 20)}[phase]
        block = np.array([m["scene"] for m in eps])
        _BD[k] = dict(
            eps=eps, d_sem=R[li][:, li],
            d_mot=eucl(np.stack([m["motion_act"][mv].ravel() for m in eps])),
            d_sc=eucl(np.stack([m["pix"].ravel().astype(np.float32) for m in eps])),
            mask=np.triu(block[:, None] == block[None], 1), block=block)
    return _BD[k]


def b_cell(F: np.ndarray, D: dict, rng) -> dict:
    iu = np.nonzero(D["mask"])
    X = np.column_stack([D["d_sem"][iu], D["d_mot"][iu], D["d_sc"][iu]])
    sdx = X.std(0)
    use = sdx > 1e-9
    names = np.array(["sem", "motion", "scene"])[use]
    Xs = (X[:, use] - X[:, use].mean(0)) / sdx[use]
    y = F[iu]
    ys = (y - y.mean()) / y.std()
    X1 = np.column_stack([np.ones(len(ys)), Xs])
    pinv = np.linalg.pinv(X1)[1:]
    beta = pinv @ ys

    def r2(cols):
        if not cols:
            return 0.0
        A = np.column_stack([np.ones(len(ys))] + [Xs[:, c] for c in cols])
        e = ys - A @ np.linalg.lstsq(A, ys, rcond=None)[0]
        return 1 - (e @ e) / (ys @ ys)

    k = Xs.shape[1]
    full = r2(list(range(k)))
    uniq = {names[c]: full - r2([x for x in range(k) if x != c]) for c in range(k)}
    blocks = [np.nonzero(D["block"] == b)[0] for b in np.unique(D["block"])]
    cnt = np.zeros(k)
    n = len(D["block"])
    for _ in range(NPERM_B):
        p = np.arange(n)
        for bi in blocks:
            p[bi] = rng.permutation(bi)
        yp = F[p][:, p][iu]
        cnt += np.abs(pinv @ ((yp - yp.mean()) / yp.std())) >= np.abs(beta) - 1e-12
    pv = (cnt + 1) / (NPERM_B + 1)
    eps = D["eps"]
    lab = np.array([m["task"] for m in eps])
    ex = np.array([m["exec_b"] for m in eps])
    si, se = (lab[:, None] == lab[None])[iu], (ex[:, None] == ex[None])[iu]
    tab = {f"{'same' if a else 'diff'}_instr__{'same' if b else 'diff'}_exec":
           (float(y[(si == a) & (se == b)].mean() / y.mean()) if ((si == a) & (se == b)).any() else None)
           for a in (True, False) for b in (True, False)}
    return dict(n_eps=n, n_pairs=int(len(ys)), beta={names[c]: float(beta[c]) for c in range(k)},
                p={names[c]: float(pv[c]) for c in range(k)}, r2=float(full), unique=uniq, table2x2=tab)


def beta_job(job):
    rep, layer, view = job
    t0 = time.time()
    X = vectors(rep, layer, view, list({m["i"]: m for s_ in BSETS for m in bset(s_)}.values()))
    rng = np.random.default_rng(1)
    out = {"rep": rep, "layer": layer, "view": view, "cells": {}}
    for name in BSETS:
        for phase in ("pre", "post", "traj"):
            D = b_design(name, phase)
            V = np.stack([phase_vec(X[m["off"]:m["off"] + m["C"]], m["bound_call"], phase) for m in D["eps"]])
            F = eucl(V) / np.sqrt(V.shape[1])
            out["cells"][f"{name}|{phase}"] = b_cell(F, D, rng)
    out["sec"] = time.time() - t0
    return out


def stage_beta() -> None:
    jobs = [(r, li, v) for r, li in all_jobs(list(BETA_REPS)) for v in ("z", "mag")]
    jobs += [(r, -1, "magall") for r in ("v_task", "v_agent", "v_all6", "e_s9")]
    jobs = sorted(jobs, key=lambda j: -(j[0] == "v_all6") * (j[2] == "z"))
    done = run_parts("beta", beta_job, jobs)
    save_json("beta", list(done.values()))


# ============================================================================ (5) divergence maps
DIV_REPS = ("v_task", "v_agent", "v_wrist", "v_state", "v_all6", "e_s0", "e_s9", "m_s9")
DIV_CALLS = 28


def div_pairs() -> dict:
    """{group: [(a_index, ref_index, [other-init reference indices], cluster, outcome)]}"""
    M = meta()
    by = {}
    for m in M:
        by[(m["set"], m["condition"], m["task"], m["init"])] = m["i"]
    out: dict = defaultdict(list)
    for m in M:
        if m["set"] == "basket_a100":
            ref = ("libero3", "object", m["task"])
        elif m["set"] == "basket_far" and m["condition"] != "ctl4090":
            ref = ("basket_far", "ctl4090", m["task"])
        else:
            continue
        b = by[(*ref, m["init"])]
        others = [by[(*ref, j)] for j in range(10) if j != m["init"] and (*ref, j) in by]
        out[m["condition"]].append((m["i"], b, others, m["task"], m["outcome"]))
    path = {m["path"]: m for m in M}
    pairs = [p for p in json.load(open(PAIRS)) if p["valid"] and p.get("rich_available")]
    for p in pairs:
        a, b = path[p["cand"]], path[p["base"]]
        others = [by[("goal", "train", b["task"], j)] for j in range(10) if j != b["init"] and ("goal", "train", b["task"], j) in by]
        grp = "cand_fallback" if p["outcome"] in FALLBACK else "cand_new" if p["outcome"] in NEW else "cand_other"
        out[grp].append((a["i"], b["i"], others, p["variant"], p["outcome"]))
    return dict(out)


def div_job(job):
    rep, layer, view = job
    t0 = time.time()
    M = meta()
    need = sorted({i for pp in div_pairs().values() for a, b, oth, _, _ in pp for i in (a, b, *oth)})
    X = vectors(rep, layer, view, [M[i] for i in need])
    out = {"rep": rep, "layer": layer, "view": view, "groups": {}}
    for g, pp in div_pairs().items():
        R = []
        for a, b, others, _, _ in pp:
            ma, mb = M[a], M[b]
            n = min(ma["C"], mb["C"], DIV_CALLS)
            xa, xb = X[ma["off"]:ma["off"] + n], X[mb["off"]:mb["off"] + n]
            d = np.linalg.norm(xa - xb, axis=1)
            base = np.mean([np.linalg.norm(X[M[o]["off"]:M[o]["off"] + n] - xb, axis=1) for o in others], 0)
            r = np.full(DIV_CALLS, np.nan)
            r[:n] = d / np.maximum(base, 1e-9)
            R.append(r)
        R = np.array(R)
        out["groups"][g] = {"median": np.nanmedian(R, 0), "ratio": R.astype(np.float32),
                            "cluster": [p[3] for p in pp], "outcome": [p[4] for p in pp]}
    out["sec"] = time.time() - t0
    return out


def stage_div() -> None:
    jobs = [(r, li, v) for r, li in all_jobs(list(DIV_REPS)) for v in ("z", "mag", "rms")]
    jobs = [j for j in jobs if not (j[2] == "rms" and j[0] not in ("v_task", "v_all6", "e_s9"))]
    done = run_parts("div", div_job, jobs)
    pickle.dump(done, open(CACHE / "div.pkl", "wb"))



# ============================================================================ (6) Kaufman read-in per VLM layer
KAUF_SEGS = {"task": 3, "agentview": 0, "wrist": 1, "state": 4}


def readin(j: int) -> np.ndarray:
    w = np.load(RICH / "weights.npz", allow_pickle=True)
    g = 1.0 + w[f"vlm/L{j}/input_layernorm"].astype(np.float64)
    return np.concatenate([w[f"vlm/L{j}/k_proj"], w[f"vlm/L{j}/v_proj"]]).astype(np.float64) * g[None, :]


def kauf_job(layer: int) -> dict:
    """PF of call-0 differences of layer `layer`'s output under layer l+1's read-in (17: structural null + proxy)."""
    M = meta()
    a = _mm("vlm_seg")[layer]
    rd = readin(layer + 1 if layer < N_L - 1 else layer)
    P, _ = row_space(rd)
    out = {"layer": layer, "reader": layer + 1 if layer < N_L - 1 else None, "proxy": layer == N_L - 1,
           "chance": P.shape[1] / rd.shape[1], "segs": {}}
    pairs = div_pairs()
    U = u40(M)
    for sn, si in KAUF_SEGS.items():
        def v(i):
            x = np.asarray(a[M[i]["off"], si], np.float64)
            return x / np.sqrt((x**2).mean() + 1e-6)

        res = {}
        for g, pp in pairs.items():
            d = np.stack([v(p[0]) - v(p[1]) for p in pp])
            res[g] = {"pf": pf(d, P), "cluster": [p[3] for p in pp], "rel_norm": np.linalg.norm(d, axis=1) / np.sqrt(2048)}
        X0 = np.stack([v(m["i"]) for m in U])
        Xc = X0 - X0.mean(0)
        C = Xc.T @ Xc
        res["ACT"] = float(np.trace(P.T @ C @ P) / np.trace(C))
        out["segs"][sn] = res
    return out


def stage_kauf() -> None:
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(kauf_job, range(N_L)))
    pickle.dump(res, open(CACHE / "kauf.pkl", "wb"))


# ============================================================================ (7) Gallego per layer + Kim/NNLS per layer
GAL_REPS = ("e_s9", "v_all6", "v_task", "m_s9")
GAL_UNITS = (("basket_a100", "near"), ("basket_a100", "opp"), ("basket_far", "ctl4090"), ("basket_far", "far20"),
             ("basket_far", "far25"))


def _bin(x: np.ndarray, nb: int = 20) -> np.ndarray:
    return resample(x.astype(np.float64), nb).astype(np.float64)


def _angles(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def _cca_r(XA, YA, XB, YB, k=4) -> float:
    from sklearn.cross_decomposition import CCA

    try:
        m = CCA(n_components=k, max_iter=2000).fit(XA, YA)
        u, v = m.transform(XB, YB)
        return float(np.mean([np.corrcoef(u[:, i], v[:, i])[0, 1] for i in range(k)]))
    except Exception:
        return float("nan")


def gal_job(job):
    from scipy.optimize import nnls

    rep, layer = job
    t0 = time.time()
    M = meta()
    X = vectors(rep, layer, "z", [m for m in M if m["set"] in ("basket_a100", "basket_far", "goal")
                                  or (m["t027_set"] == "train" and m["task"][0] == "G")])
    dim = 10
    units = {}
    for s, c in GAL_UNITS:
        for t in sorted({m["task"] for m in M if m["set"] == s}):
            rr = sorted([m for m in M if m["set"] == s and m["condition"] == c and m["task"] == t], key=lambda m: m["init"])
            tr = [X[m["off"]:m["off"] + m["C"]] for m in rr]
            S = np.concatenate(tr)
            mu = S.mean(0)
            U = np.linalg.svd(S - mu, full_matrices=False)[2][:dim].T
            half = [np.concatenate(tr[i::2]) for i in (0, 1)]
            Uh = [np.linalg.svd(h - h.mean(0), full_matrices=False)[2][:dim].T for h in half]
            units[(c, t)] = {"mu": mu, "U": U, "UA": Uh[0], "self": float(np.mean(_angles(*Uh))),
                             "A": [_bin(x) for x in tr[0::2]], "B": [_bin(x) for x in tr[1::2]]}
    gal = {}
    for ref, oth in (("near", "far20"), ("near", "far25"), ("near", "opp"), ("near", "ctl4090"),
                     ("ctl4090", "far20"), ("ctl4090", "far25")):
        rows = []
        for t in sorted({k[1] for k in units}):
            a, b = units[(ref, t)], units[(oth, t)]
            ang = float(np.mean(_angles(a["U"], b["U"])))
            n, nB = min(len(a["A"]), len(b["A"])), min(len(a["B"]), len(b["B"]))
            cc = _cca_r(np.concatenate([(x - a["mu"]) @ a["UA"] for x in a["A"][:n]]),
                        np.concatenate([(x - b["mu"]) @ b["UA"] for x in b["A"][:n]]),
                        np.concatenate([(x - a["mu"]) @ a["UA"] for x in a["B"][:nB]]),
                        np.concatenate([(x - b["mu"]) @ b["UA"] for x in b["B"][:nB]]))
            rows.append({"task": t, "angle": ang, "angle_excess": ang - min(a["self"], b["self"]), "cca_r": cc})
        gal[f"{ref}|{oth}"] = rows
    # Kim / NNLS (T029 A7) on this representation
    ro = {r["cand"]: r for r in csv.DictReader(open(ROLES))}
    train = defaultdict(list)
    for m in M:
        if m["t027_set"] == "train" and m["task"][0] == "G":
            train[m["task"]].append(m)
    nb = 20
    cent = {t: np.mean([_bin(X[m["off"]:m["off"] + m["C"]], nb) for m in rr], 0) for t, rr in train.items()}
    gm = np.concatenate(list(cent.values())).mean(0)
    tasks = sorted(cent)
    rng = np.random.default_rng(0)
    recs = []
    for m in M:
        if m["t027_set"] != "cand":
            continue
        rl = ro.get(m["variant"], {})
        s = {}
        for col, key in (("base_task", "base"), ("object_source", "object"), ("destination_source", "destination"),
                         ("action_source", "action")):
            for x in (rl.get(col) or "").split(";"):
                if x and f"G{x}" in cent:
                    s.setdefault(key, f"G{x}")
        uniq = sorted(set(s.values()))
        if not uniq:
            continue
        e = X[m["off"]:m["off"] + m["C"]]
        C = len(e)
        bins = np.clip(np.round(np.linspace(0, nb - 1, C)).astype(int), 0, nb - 1)

        def fit(pick):
            W, r2 = [], []
            for c in range(C):
                A = np.stack([cent[t][bins[c]] - gm for t in pick], 1)
                y = e[c] - gm
                w, res_ = nnls(A, y)
                W.append(w)
                r2.append(1 - res_**2 / (y @ y))
            return np.array(W), float(np.mean(r2))

        Wt, r2 = fit(uniq)
        others = [t for t in tasks if t not in uniq]
        rr2 = [fit(list(rng.choice(others, len(uniq), replace=False)))[1] for _ in range(20)] if len(others) >= len(uniq) else []
        rec = {"variant": m["variant"], "outcome": m["outcome"], "r2": r2, "r2_random": float(np.mean(rr2)) if rr2 else np.nan}
        if "object" in s and "destination" in s and s["object"] != s["destination"]:
            tau = np.linspace(0, 1, C)
            wo, wd = Wt[:, uniq.index(s["object"])], Wt[:, uniq.index(s["destination"])]
            rec["slope_dest_minus_obj"] = float(np.polyfit(tau, wd - wo, 1)[0])
        recs.append(rec)
    return {"rep": rep, "layer": layer, "gal": gal, "nnls": recs, "sec": time.time() - t0}


def stage_gal() -> None:
    jobs = sorted(all_jobs(list(GAL_REPS)), key=lambda j: -(j[0] == "v_all6"))
    done = run_parts("gal", gal_job, jobs)
    save_json("gal", list(done.values()))



# ============================================================================ figures + numbers
CAT = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
TXT, TXT2 = "#191919", "#666666"
FIG = OUT / "fig"
DEC_TARGETS = ("task40", "within", "suite", "verb", "object", "dest")
TARGET_LABEL = {"task40": "task (40-way)", "within": "task within suite (10-way)", "suite": "suite (4-way)",
                "verb": "verb (unseen tasks)", "object": "object (unseen tasks)", "dest": "destination (unseen tasks)"}
DEC_REPS = ("v_task", "v_agent", "v_wrist", "v_state", "v_all6", "e_s0", "e_s4", "e_s9", "m_s0", "m_s9", "et_s0", "et_s9", "vt")
REP_LABEL = {"v_task": "VLM task seg", "v_agent": "VLM agentview", "v_wrist": "VLM wrist", "v_state": "VLM state",
             "v_all6": "VLM all 6 seg", "e_s0": "expert step0", "e_s4": "expert step4", "e_s9": "expert step9",
             "m_s0": "expert MLP step0", "m_s4": "expert MLP step4", "m_s9": "expert MLP step9",
             "et_s0": "expert 50 tok step0", "et_s9": "expert 50 tok step9", "vt": "VLM task tokens"}


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as p

    p.rcParams.update({"font.size": 8, "axes.edgecolor": TXT2, "axes.labelcolor": TXT, "xtick.color": TXT2,
                       "ytick.color": TXT2, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                       "axes.titlesize": 8.5, "legend.frameon": False})
    return p


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    import matplotlib.pyplot as p

    p.close(fig)


def dec_results() -> dict:
    return {(j["rep"], j["layer"], j["view"]): j for j in pickle.load(open(CACHE / "dec_parts.pkl", "rb")).values()}


def mag_key(rep: str, layer: int) -> tuple:
    if rep.startswith("et_"):
        return ("e_s" + rep.split("_s")[1], layer, "mag")
    if rep == "vt":
        return ("v_task", layer, "mag")
    return (rep, layer, "mag")


def magall_key(rep: str) -> tuple | None:
    base = {"vt": "v_task", "et_s0": "e_s0", "et_s9": "e_s9", "m_s0": "e_s0", "m_s4": "e_s4", "m_s9": "e_s9"}.get(rep, rep)
    return (base, -1, "magall")


def fig_dec(D: dict) -> None:
    p = plt()
    for view in ("z", "mag"):
        for tg in DEC_TARGETS:
            reps = [r for r in DEC_REPS if (r, REPS[r]["layers"][0], view) in D]
            fig, axs = p.subplots(1, len(reps), figsize=(1.55 * len(reps) + 1, 3.4), squeeze=False)
            for ax, r in zip(axs[0], reps):
                lay = REPS[r]["layers"]
                A = np.array([D[(r, li, view)]["acc"][tg] for li in lay])
                im = ax.imshow(A, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1)
                ax.set_yticks(range(len(lay)))
                ax.set_yticklabels(lay, fontsize=6)
                ax.set_xticks(range(len(CALLS)))
                ax.set_xticklabels(CALLS, fontsize=5.5, rotation=90)
                ax.set_title(REP_LABEL[r], fontsize=7)
                ax.set_xlabel("call")
            axs[0][0].set_ylabel("layer")
            fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8, label="CV accuracy")
            fig.suptitle(f"Decoding {TARGET_LABEL[tg]} — {'vector (z-scored)' if view == 'z' else 'magnitude only (T021, same layer)'}",
                         color=TXT)
            _save(fig, f"3_dec_{tg}_{view}.png")
    # phase: layer x phase recall
    reps = [r for r in DEC_REPS if (r, REPS[r]["layers"][0], "z") in D]
    fig, axs = p.subplots(2, len(reps), figsize=(1.4 * len(reps) + 1, 5), squeeze=False)
    for row, view in enumerate(("z", "mag")):
        for ax, r in zip(axs[row], reps):
            lay = REPS[r]["layers"]
            k = [(r, li, view) if view == "z" else mag_key(r, li) for li in lay]
            if not all(x in D and "phase_recall" in D[x] for x in k):
                ax.axis("off")
                continue
            A = np.array([[D[x]["phase_recall"][ph] for ph in PHASE4] for x in k])
            im = ax.imshow(A, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1)
            ax.set_yticks(range(len(lay)))
            ax.set_yticklabels(lay, fontsize=6)
            ax.set_xticks(range(4))
            ax.set_xticklabels(PHASE4, rotation=90, fontsize=6)
            ax.set_title(f"{REP_LABEL[r]}\n{'vector' if view == 'z' else 'magnitude'}", fontsize=6.5)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="recall")
    fig.suptitle("Phase decoding (reach / grasp / carry / release), layer × phase recall", color=TXT)
    _save(fig, "3_dec_phase.png")


def xgen_share(job: dict, kind: str, src: str | None, pool: list) -> tuple[np.ndarray, np.ndarray]:
    """per-call executed / instructed share (mean over trials with a value)."""
    ex, ins = [], []
    for ce, ci in zip(job["xgen"][f"{kind}_exec"], job["xgen"][f"{kind}_instr"]):
        idx = [i for i, v in enumerate(ce) if v is not None and (src is None or pool[i]["src"] == src)]
        ex.append(np.mean([ce[i] for i in idx]) if idx else np.nan)
        ins.append(np.mean([ci[i] for i in idx]) if idx else np.nan)
    return np.array(ex), np.array(ins)


def fig_xgen(D: dict) -> None:
    p = plt()
    pool = labels()["pool"]
    reps = ("v_task", "v_agent", "v_all6", "vt", "e_s0", "e_s9", "m_s9", "et_s9")
    for kind in ("task", "object", "dest"):
        fig, axs = p.subplots(3, len(reps), figsize=(1.6 * len(reps) + 1, 8), squeeze=False)
        for j, r in enumerate(reps):
            lay = REPS[r]["layers"]
            E = np.array([xgen_share(D[(r, li, "z")], kind, None, pool)[0] for li in lay])
            I = np.array([xgen_share(D[(r, li, "z")], kind, None, pool)[1] for li in lay])
            for i, (A, ttl, cm, lim) in enumerate(((E, "executed share", "Oranges", (0, 1)), (I, "instructed share", "Blues", (0, 1)),
                                                    (E - I, "executed − instructed", "RdBu_r", (-1, 1)))):
                ax = axs[i][j]
                im = ax.imshow(A, aspect="auto", origin="lower", cmap=cm, vmin=lim[0], vmax=lim[1])
                ax.set_yticks(range(len(lay)))
                ax.set_yticklabels(lay, fontsize=6)
                ax.set_xticks(range(len(CALLS)))
                ax.set_xticklabels(CALLS, fontsize=5.5, rotation=90)
                if i == 0:
                    ax.set_title(REP_LABEL[r], fontsize=7)
                if j == 0:
                    ax.set_ylabel(f"{ttl}\nlayer")
                if j == len(reps) - 1:
                    fig.colorbar(im, ax=axs[i].tolist(), shrink=0.8)
        n = sum(1 for v in D[("v_task", 0, "z")]["xgen"][f"{kind}_exec"][0] if v is not None)
        fig.suptitle(f"Cross-generalisation, {kind} decoder trained on 40 training tasks → fallback trials "
                     f"(cand exec_other + L90 lscene trained, n={n})", color=TXT)
        _save(fig, f"3_xgen_{kind}.png")


def fig_traj() -> None:
    p = plt()
    proj = pickle.load(open(CACHE / "traj_proj.pkl", "rb"))
    M = meta()
    cs = cases()
    for cname, conds in cs.items():
        for rep in TRAJ_REPS:
            lay = REPS[rep]["layers"]
            nc = 6
            nr = int(np.ceil(len(lay) / nc))
            fig, axs = p.subplots(nr, nc, figsize=(2.1 * nc, 2.0 * nr), squeeze=False)
            for k, li in enumerate(lay):
                ax = axs[k // nc][k % nc]
                P = proj[(rep, li)]
                for ci, (lab, ids) in enumerate(conds.items()):
                    col = CAT[ci % len(CAT)]
                    if not ids:
                        continue
                    T = np.stack([resample(P[i], 20) for i in ids])
                    for t in T[:4]:
                        ax.plot(t[:, 0], t[:, 1], color=col, lw=0.4, alpha=0.35)
                    mtr = T.mean(0)
                    for s in range(19):  # time gradient: light -> dark
                        ax.plot(mtr[s:s + 2, 0], mtr[s:s + 2, 1], color=col, lw=2, alpha=0.25 + 0.75 * s / 18,
                                solid_capstyle="round")
                    ax.plot(*mtr[0], "o", mfc="white", mec=col, ms=5, mew=1.3)
                    ax.plot(*mtr[-1], "s", color=col, ms=4.5)
                    gb = [M[i]["gc"] / max(M[i]["C"] - 1, 1) * 19 for i in ids if M[i]["gc"] is not None]
                    if gb:
                        g = int(round(np.mean(gb)))
                        ax.plot(*mtr[g], "x", color=TXT, ms=5, mew=1.3)
                    if k == 0:
                        ax.plot([], [], color=col, lw=2, label=f"{lab} (n={len(ids)})")
                ax.set_title(f"L{li}", fontsize=8)
                ax.tick_params(labelsize=5)
            for k in range(len(lay), nr * nc):
                axs[k // nc][k % nc].axis("off")
            fig.legend(loc="lower center", ncol=min(6, len(conds)), bbox_to_anchor=(0.5, -0.02 - 0.02 * (len(conds) > 6)))
            fig.suptitle(f"{cname}: {REP_LABEL[rep]}, PC1–PC2 per layer (PCA fit on 40 training tasks). "
                         "○ start  × grasp  ■ end, light→dark = time; thick = condition mean, thin = single trials",
                         color=TXT, fontsize=8.5)
            fig.tight_layout(rect=(0, 0.04, 1, 0.96))
            _save(fig, f"1_traj_{cname}_{rep}.png")


def fig_dim(dim: dict) -> None:
    p = plt()
    J = {(j["rep"], j["layer"]): j for j in dim}
    groups = (("VLM", ("v_task", "v_agent", "v_wrist", "v_state", "v_all6", "vt")),
              ("expert", ("e_s0", "e_s4", "e_s9", "m_s0", "m_s4", "m_s9", "et_s0", "et_s9")))
    fig, axs = p.subplots(2, 4, figsize=(14, 6))
    for row, (gn, reps) in enumerate(groups):
        for col, (key, lab) in enumerate((("PR", "participation ratio (z)"), ("top10", "variance in top-10 PCs (z)"),
                                          ("PR_raw", "participation ratio (raw / RMSNorm dashed)"), ("time_var_frac", "within-episode (time) variance fraction"))):
            ax = axs[row][col]
            for ci, r in enumerate(reps):
                lay = REPS[r]["layers"]
                if key == "PR_raw":
                    if "raw" not in J[(r, lay[0])]:
                        continue
                    ax.plot(lay, [J[(r, li)]["raw"]["PR"] for li in lay], color=CAT[ci % 8], lw=2, marker="o", ms=3, label=REP_LABEL[r])
                    ax.plot(lay, [J[(r, li)]["rms"]["PR"] for li in lay], color=CAT[ci % 8], lw=1.2, ls="--")
                    ax.set_yscale("log")
                elif key == "time_var_frac":
                    ax.plot(lay, [J[(r, li)]["shape"][key] for li in lay], color=CAT[ci % 8], lw=2, marker="o", ms=3, label=REP_LABEL[r])
                else:
                    ax.plot(lay, [J[(r, li)]["z"][key] for li in lay], color=CAT[ci % 8], lw=2, marker="o", ms=3, label=REP_LABEL[r])
                    if key == "PR":
                        ax.set_yscale("log")
            ax.set_title(f"{gn}: {lab}")
            ax.set_xlabel("layer")
            ax.grid(alpha=0.2)
        axs[row][0].legend(fontsize=6)
    fig.tight_layout()
    _save(fig, "2_dimensionality.png")
    fig, axs = p.subplots(1, 3, figsize=(12, 3.4))
    for ci, r in enumerate(("v_task", "v_agent", "v_all6", "e_s9", "m_s9", "et_s9", "vt")):
        lay = REPS[r]["layers"]
        for ax, (key, lab) in zip(axs, (("speed", "normalised speed (step / overall SD)"), ("straightness", "straightness (end-to-end / path)"),
                                        ("pc12_var", "variance in PC1–2"))):
            v = [J[(r, li)]["shape"].get(key, J[(r, li)].get(key)) for li in lay]
            if v[0] is None:
                continue
            ax.plot(lay, v, color=CAT[ci % 8], lw=2, marker="o", ms=3, label=REP_LABEL[r])
            ax.set_title(lab)
            ax.set_xlabel("layer")
            ax.grid(alpha=0.2)
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    _save(fig, "1_traj_shape_metrics.png")


def fig_beta(B: dict) -> None:
    p = plt()
    for view in ("z", "mag"):
        fig, axs = p.subplots(len(BSETS), len(BETA_REPS), figsize=(2.6 * len(BETA_REPS), 2.4 * len(BSETS)), squeeze=False)
        for i, s in enumerate(BSETS):
            for j, r in enumerate(BETA_REPS):
                ax = axs[i][j]
                lay = REPS[r]["layers"]
                for ci, (term, ls) in enumerate((("sem", "-"), ("motion", "-"), ("scene", "-"))):
                    for ph, mk, lw in (("traj", "o", 2), ("pre", None, 0.9), ("post", None, 0.9)):
                        v = [B[(r, li, view)]["cells"][f"{s}|{ph}"]["beta"].get(term, np.nan) for li in lay]
                        ax.plot(lay, v, color=CAT[ci], lw=lw, marker=mk, ms=2.5, ls=ls if ph != "pre" else ":",
                                alpha=1 if ph == "traj" else 0.7, label=f"β {term} ({ph})" if (i == 0 and j == 0) else None)
                ax.axhline(0, color=TXT2, lw=0.6)
                ax.set_ylim(-0.3, 0.9)
                ax.set_title(f"{s} — {REP_LABEL[r]}", fontsize=7)
                ax.grid(alpha=0.2)
        axs[-1][0].set_xlabel("layer")
        fig.legend(loc="lower center", ncol=9, fontsize=6)
        fig.suptitle(f"Pair regression (T027): feature distance ~ semantics + motion(action) + scene. "
                     f"{'vector' if view == 'z' else 'magnitude only'}. thick = whole trajectory, dotted = pre-grasp, thin = post",
                     color=TXT)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        _save(fig, f"4_beta_{view}.png")


def fig_div(DV: dict) -> None:
    p = plt()
    groups = ("near", "opp", "far20", "far25", "cand_fallback", "cand_new")
    for view in ("z", "mag"):
        fig, axs = p.subplots(len(DIV_REPS), len(groups), figsize=(2.2 * len(groups) + 1, 2.0 * len(DIV_REPS)), squeeze=False)
        for i, r in enumerate(DIV_REPS):
            lay = REPS[r]["layers"]
            for j, g in enumerate(groups):
                ax = axs[i][j]
                A = np.array([DV[(r, li, view)]["groups"][g]["median"] for li in lay])
                im = ax.imshow(A, aspect="auto", origin="lower", cmap="magma_r", vmin=0, vmax=1.5)
                on = [next((c for c in range(A.shape[1]) if A[k, c] >= 0.5), np.nan) for k in range(len(lay))]
                ax.plot(on, range(len(lay)), color="#00b894", lw=1.2)
                ax.set_yticks(range(len(lay)))
                ax.set_yticklabels(lay, fontsize=5)
                ax.tick_params(labelsize=5)
                if i == 0:
                    ax.set_title(g)
                if j == 0:
                    ax.set_ylabel(f"{REP_LABEL[r]}\nlayer", fontsize=6.5)
                if i == len(DIV_REPS) - 1:
                    ax.set_xlabel("call")
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.4, label="pair distance ÷ other-init baseline (median)")
        fig.suptitle(f"Divergence maps ({'vector z' if view == 'z' else 'magnitude only'}). green line = onset (first call with ratio ≥ 0.5)",
                     color=TXT)
        _save(fig, f"5_divergence_{view}.png")


def fig_kauf(K: list) -> None:
    p = plt()
    fig, axs = p.subplots(1, 4, figsize=(14, 3.4))
    for ax, sn in zip(axs, KAUF_SEGS):
        lay = [k["layer"] for k in K]
        for ci, (g, lab) in enumerate((("cand_fallback", "instr: cand − base, FALLBACK"), ("cand_new", "instr: cand − base, NEW"),
                                       ("near", "scene: near − ctl"), ("opp", "scene: opp − ctl"), ("far20", "scene: far20 − ctl"),
                                       ("far25", "scene: far25 − ctl"))):
            ax.plot(lay, [np.median(k["segs"][sn][g]["pf"]) for k in K], color=CAT[ci], lw=2 if ci < 2 else 1.2,
                    marker="o" if ci < 2 else None, ms=3, label=lab)
        ax.plot(lay, [k["segs"][sn]["ACT"] for k in K], color=TXT, lw=1.2, ls="--", label="ACT (all activity)")
        ax.axhline(0.25, color=TXT2, lw=0.8, ls=":", label="random (0.25)")
        ax.axvspan(16.5, 17.5, color="#dddddd", zorder=0)
        ax.text(17, 0.98, "L17: never read\n(proxy)", ha="center", va="top", fontsize=6, color=TXT2,
                transform=ax.get_xaxis_transform())
        ax.set_title(f"{sn} segment")
        ax.set_xlabel("layer l (read by l+1 K/V)")
        ax.grid(alpha=0.2)
    axs[0].set_ylabel("read-in potent fraction (median)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    _save(fig, "6_kaufman_readin.png")


def fig_gal(G: dict) -> None:
    p = plt()
    fig, axs = p.subplots(3, len(GAL_REPS), figsize=(3.2 * len(GAL_REPS), 8), squeeze=False)
    for j, r in enumerate(GAL_REPS):
        lay = REPS[r]["layers"]
        for ci, cmp in enumerate(("near|ctl4090", "near|far20", "near|far25", "near|opp")):
            cc = [np.nanmedian([x["cca_r"] for x in G[(r, li)]["gal"][cmp]]) for li in lay]
            an = [np.median([x["angle_excess"] for x in G[(r, li)]["gal"][cmp]]) for li in lay]
            axs[0][j].plot(lay, cc, color=CAT[ci], lw=2, marker="o", ms=3, label=cmp.replace("|", " vs "))
            axs[1][j].plot(lay, an, color=CAT[ci], lw=2, marker="o", ms=3)
        d = [np.nanmedian([x["r2"] - x["r2_random"] for x in G[(r, li)]["nnls"]]) for li in lay]
        sn = [np.nanmedian([x["slope_dest_minus_obj"] for x in G[(r, li)]["nnls"] if x["outcome"] in NEW and "slope_dest_minus_obj" in x]) for li in lay]
        sf = [np.nanmedian([x["slope_dest_minus_obj"] for x in G[(r, li)]["nnls"] if x["outcome"] in FALLBACK and "slope_dest_minus_obj" in x]) for li in lay]
        axs[2][j].plot(lay, d, color=CAT[0], lw=2, marker="o", ms=3, label="NNLS R² source − random")
        axs[2][j].plot(lay, sn, color=CAT[1], lw=1.5, label="slope dest−obj, NEW")
        axs[2][j].plot(lay, sf, color=CAT[2], lw=1.5, label="slope dest−obj, FALLBACK")
        axs[2][j].axhline(0, color=TXT2, lw=0.6)
        axs[0][j].set_title(REP_LABEL[r])
        for i in range(3):
            axs[i][j].grid(alpha=0.2)
        axs[2][j].set_xlabel("layer")
    axs[0][0].set_ylabel("held-out CCA r (median over tasks)")
    axs[1][0].set_ylabel("principal angle − self angle (°)")
    axs[2][0].set_ylabel("goal cand NNLS")
    axs[0][0].legend(fontsize=6)
    axs[2][0].legend(fontsize=6)
    fig.tight_layout()
    _save(fig, "7_gallego_kim.png")


def fig_magvec(D: dict) -> None:
    p = plt()
    reps = ("v_task", "v_agent", "v_all6", "e_s9", "m_s9", "et_s9", "vt")
    tg = (*DEC_TARGETS, "phase")
    fig, axs = p.subplots(len(reps), len(tg), figsize=(2.1 * len(tg), 1.7 * len(reps)), squeeze=False)
    for i, r in enumerate(reps):
        lay = REPS[r]["layers"]
        for j, t in enumerate(tg):
            ax = axs[i][j]
            get = (lambda k: D[k]["phase_bacc"]) if t == "phase" else (lambda k: float(np.mean(D[k]["acc"][t])))  # noqa: B023
            ax.plot(lay, [get((r, li, "z")) for li in lay], color=CAT[0], lw=2, marker="o", ms=2.5, label="vector")
            ax.plot(lay, [get(mag_key(r, li)) for li in lay], color=CAT[1], lw=1.5, marker="o", ms=2.5, label="magnitude (same layer)")
            ka = magall_key(r)
            if t != "phase" and ka in D:
                ax.axhline(get(ka), color=CAT[1], lw=1, ls="--", label="magnitude (all layers, T021)")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(TARGET_LABEL.get(t, "phase (balanced acc)"), fontsize=7)
            if j == 0:
                ax.set_ylabel(REP_LABEL[r], fontsize=7)
    axs[0][0].legend(fontsize=5.5)
    fig.suptitle("Vector vs magnitude-only decoding (call-mean accuracy) by layer", color=TXT)
    fig.tight_layout()
    _save(fig, "8_vector_vs_magnitude.png")


def stage_figs() -> None:
    D = dec_results()
    fig_dec(D)
    fig_xgen(D)
    fig_magvec(D)
    fig_dim(load_json("dim")["jobs"])
    fig_traj()
    fig_beta({(j["rep"], j["layer"], j["view"]): j for j in load_json("beta")})
    fig_div({k: v for k, v in pickle.load(open(CACHE / "div.pkl", "rb")).items()})
    fig_kauf(pickle.load(open(CACHE / "kauf.pkl", "rb")))
    fig_gal({(j["rep"], j["layer"]): j for j in load_json("gal")})



# ============================================================================ numbers + hypothesis tests
def _wil(x, alternative="two-sided"):
    from scipy.stats import wilcoxon

    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) < 5 or np.allclose(x, 0):
        return float("nan")
    return float(wilcoxon(x, alternative=alternative).pvalue)


def _mwu(a, b):
    from scipy.stats import mannwhitneyu

    a = np.asarray([v for v in a if np.isfinite(v)], float)
    b = np.asarray([v for v in b if np.isfinite(v)], float)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    return float(mannwhitneyu(a, b).pvalue)


class Family:
    def __init__(self, name):
        self.name, self.rows = name, []

    def add(self, test, p, **kw):
        self.rows.append({"test": test, "p": p, **kw})

    def done(self):
        q = bh([r["p"] for r in self.rows])
        for r, qq in zip(self.rows, q):
            r["q_bh"] = float(qq)
        return {"n_tests": len(self.rows), "n_sig_q05": int(np.nansum(q < 0.05)), "tests": self.rows}


def _cluster_means(vals, clusters):
    g = defaultdict(list)
    for v, c in zip(vals, clusters):
        if np.isfinite(v):
            g[c].append(v)
    return {c: float(np.mean(v)) for c, v in g.items()}


def _trial_diff(job, kind, calls_sel, pool):
    """per pool trial: mean over selected calls of (executed hit − instructed hit)."""
    ci = [job["calls"].index(c) for c in calls_sel if c in job["calls"]]
    out = []
    for i in range(len(pool)):
        v = [job["xgen"][f"{kind}_exec"][k][i] - job["xgen"][f"{kind}_instr"][k][i] for k in ci
             if job["xgen"][f"{kind}_exec"][k][i] is not None]
        out.append(float(np.mean(v)) if v else np.nan)
    return np.array(out)


def stage_numbers() -> None:  # noqa: C901
    N: dict = {"prereg": "PREREG.md (committed b11b1cc7 before results)"}
    D = dec_results()
    pool = labels()["pool"]
    dim = {(j["rep"], j["layer"]): j for j in load_json("dim")["jobs"]}
    DV = pickle.load(open(CACHE / "div.pkl", "rb"))
    K = pickle.load(open(CACHE / "kauf.pkl", "rb"))
    B = {(j["rep"], j["layer"], j["view"]): j for j in load_json("beta")}
    G = {(j["rep"], j["layer"]): j for j in load_json("gal")}
    mc = load_json("massive_channels")
    N["massive_channels"] = {k: [d["ch"] for d in v] for k, v in mc.items() if not k.startswith("v_all6")}
    N["massive_max"] = {k: max(d["max_abs"] for d in v) for k, v in mc.items() if k.startswith("v_") and not k.startswith("v_all6")}

    # ---- (1) trajectory shape / (2) dimensionality
    N["dim"] = {r: {li: {"PR_z": dim[(r, li)]["z"]["PR"], "top10_z": dim[(r, li)]["z"]["top10"], "n90_z": dim[(r, li)]["z"]["n90"],
                         "PR_between": dim[(r, li)]["z_between"]["PR"], "PR_within": dim[(r, li)]["z_within"]["PR"],
                         **({"PR_raw": dim[(r, li)]["raw"]["PR"], "PR_rms": dim[(r, li)]["rms"]["PR"],
                             "top1_raw": dim[(r, li)]["raw"]["top1"]} if "raw" in dim[(r, li)] else {}),
                         **dim[(r, li)]["shape"], "pc12_var": dim[(r, li)].get("pc12_var")}
                    for li in REPS[r]["layers"]} for r in REP_ORDER}

    # ---- (3) decoding
    dec = {}
    for r in REP_ORDER:
        dec[r] = {}
        for li in REPS[r]["layers"]:
            j = D[(r, li, "z")]
            m = D.get(mag_key(r, li))
            dec[r][li] = {"call_mean": {t: float(np.mean(j["acc"][t])) for t in DEC_TARGETS},
                          "call0": {t: j["acc"][t][0] for t in (*DEC_TARGETS, "within_G")},
                          "phase_bacc": j.get("phase_bacc"),
                          "mag_call_mean": {t: float(np.mean(m["acc"][t])) for t in DEC_TARGETS} if m else None,
                          "mag_phase_bacc": m.get("phase_bacc") if m else None}
            for kind in ("task", "object", "dest"):
                e, i = xgen_share(j, kind, None, pool)
                dec[r][li][f"xgen_{kind}"] = {"exec": e.tolist(), "instr": i.tolist()}
            for src in ("cand", "l90"):
                e, i = xgen_share(j, "task", src, pool)
                dec[r][li][f"xgen_task_{src}"] = {"exec": e.tolist(), "instr": i.tolist()}
    N["dec"] = dec
    N["dec_magall"] = {k[0]: {t: float(np.mean(v["acc"][t])) for t in DEC_TARGETS} for k, v in D.items() if k[2] == "magall"}
    N["dec_rms"] = {f"{k[0]}|L{k[1]}": {t: v["acc"][t] for t in ("task40", "within", "verb", "object", "dest")}
                    for k, v in D.items() if k[2] == "rms"}
    N["calls"] = list(CALLS)
    N["xgen_pool"] = dict(Counter(p["src"] for p in pool))
    # key question: instruction strongest where / when it turns into executed
    kq = {}
    for r in ("v_task", "v_all6", "vt", "e_s9", "m_s9"):
        lay = REPS[r]["layers"]
        I = np.array([dec[r][li]["xgen_task"]["instr"] for li in lay])
        E = np.array([dec[r][li]["xgen_task"]["exec"] for li in lay])
        li_i, ci_i = np.unravel_index(np.nanargmax(I), I.shape)
        kq[r] = {"max_instructed_share": float(I[li_i, ci_i]), "at_layer": int(lay[li_i]), "at_call": CALLS[ci_i],
                 "instructed_share_by_layer_callmean": dict(zip(map(int, lay), np.nanmean(I, 1).round(3).tolist())),
                 "executed_share_by_layer_callmean": dict(zip(map(int, lay), np.nanmean(E, 1).round(3).tolist())),
                 "first_call_exec_gt_instr_by_layer": {int(li): next((CALLS[c] for c in range(len(CALLS)) if E[k, c] > I[k, c]), None)
                                                       for k, li in enumerate(lay)}}
    N["key_question"] = kq

    # ---- hypotheses
    H: dict = {}
    # H1a
    fam = Family("H1")
    rat = {li: DV[("v_task", li, "z")]["groups"] for li in range(N_L)}
    per_var = defaultdict(lambda: np.full(N_L, np.nan))
    for li in range(N_L):
        vals, cl = [], []
        for g in ("cand_fallback", "cand_new", "cand_other"):
            vals += list(rat[li][g]["ratio"][:, 0])
            cl += rat[li][g]["cluster"]
        for c, v in _cluster_means(vals, cl).items():
            per_var[c][li] = v
    Rv = np.array(list(per_var.values()))
    mid, early, late = Rv[:, 6:12].mean(1), Rv[:, 0:6].mean(1), Rv[:, 12:18].mean(1)
    fam.add("H1a mid > early (v_task cand−base call-0 ratio)", _wil(mid - early), median_diff=float(np.median(mid - early)))
    fam.add("H1a mid > late", _wil(mid - late), median_diff=float(np.median(mid - late)))
    H["H1a"] = {"ratio_by_layer_median_over_variants": np.median(Rv, 0).round(4).tolist(), "n_variants": len(Rv),
                "peak_layer": int(np.argmax(np.median(Rv, 0))),
                "within_goal_call0_by_layer": [dec["v_task"][li]["call0"]["within_G"] for li in range(N_L)]}
    # H1b
    late_calls = [c for c in CALLS if c >= 4]
    dl = np.nanmean([_trial_diff(D[("v_task", li, "z")], "task", late_calls, pool) for li in range(12, 18)], 0)
    dm = np.nanmean([_trial_diff(D[("v_task", li, "z")], "task", late_calls, pool) for li in range(6, 12)], 0)
    fam.add("H1b late > mid (exec − instr share, v_task, calls ≥ 4)", _wil(dl - dm), median_diff=float(np.nanmedian(dl - dm)),
            mean_diff=float(np.nanmean(dl - dm)))
    H["H1b"] = {"mean_exec_minus_instr_mid": float(np.nanmean(dm)), "mean_exec_minus_instr_late": float(np.nanmean(dl)),
                "by_layer": [float(np.nanmean(_trial_diff(D[("v_task", li, "z")], "task", late_calls, pool))) for li in range(N_L)]}
    H["H1_tests"] = fam.done()
    # H2
    fam = Family("H2")
    cells, wins = 0, 0
    per = {}
    for r in REP_ORDER:
        for t in (*DEC_TARGETS, "phase"):
            dif = []
            for li in REPS[r]["layers"]:
                if t == "phase":
                    v, m = dec[r][li]["phase_bacc"], D[mag_key(r, li)].get("phase_bacc")
                else:
                    v, m = dec[r][li]["call_mean"][t], float(np.mean(D[mag_key(r, li)]["acc"][t]))
                dif.append(v - m)
            dif = np.array(dif)
            cells += len(dif)
            wins += int((dif > 0).sum())
            per[f"{r}|{t}"] = {"share_vec_gt_mag": float((dif > 0).mean()), "median_gain": float(np.median(dif)),
                               "min_gain": float(dif.min())}
            fam.add(f"{r}|{t}", _wil(dif) if len(dif) >= 6 else float("nan"), median_gain=float(np.median(dif)))
    H["H2"] = {"cells": cells, "vector_wins": wins, "share": wins / cells, "per": per}
    H["H2_tests"] = fam.done()
    # H3
    fam = Family("H3")
    h3 = {}
    for li in range(N_L):
        d = _trial_diff(D[("e_s9", li, "z")], "task", (0, 1, 2), pool)
        p = _wil(d)
        fam.add(f"e_s9 L{li}", p, mean_diff=float(np.nanmean(d)))
        e, i = xgen_share(D[("e_s9", li, "z")], "task", None, pool)
        h3[li] = {"exec_calls012": float(np.nanmean(e[:3])), "instr_calls012": float(np.nanmean(i[:3])), "mean_diff": float(np.nanmean(d))}
    for li in range(N_L):  # secondary: step 0
        d = _trial_diff(D[("e_s0", li, "z")], "task", (0, 1, 2), pool)
        h3[f"s0_L{li}"] = float(np.nanmean(d))
    H["H3"] = h3
    H["H3_tests"] = fam.done()
    t3 = H["H3_tests"]["tests"]
    H["H3_holds_all_layers"] = all(x["q_bh"] < 0.05 and x["mean_diff"] > 0 for x in t3)
    N["hypotheses"] = H

    # ---- (4) beta
    fam = Family("beta")
    bt = {}
    for (r, li, v), j in B.items():
        for cell, c in j["cells"].items():
            bt.setdefault(f"{r}|{v}|{cell}", {})[li] = {"beta": c["beta"], "p": c["p"], "unique": c["unique"]}
            if v == "z":
                for term in ("sem", "motion"):
                    if term in c["p"]:
                        fam.add(f"{r}|L{li}|{cell}|{term}", c["p"][term], beta=c["beta"][term])
    N["beta"] = bt
    N["beta_tests"] = {k: v for k, v in fam.done().items() if k != "tests"}

    # ---- (5) divergence
    fam = Family("div")
    dv = {}
    for (r, li, v), j in DV.items():
        for g, x in j["groups"].items():
            med = x["median"]
            on = next((c for c in range(len(med)) if med[c] >= 0.5), None)
            dv.setdefault(f"{r}|{v}|{g}", {})[li] = {"call0": float(med[0]), "onset": on, "median": np.round(med, 3).tolist()}
        if v == "z" and r in ("v_task", "e_s9"):
            fb = _cluster_means(list(j["groups"]["cand_fallback"]["ratio"][:, 0]), j["groups"]["cand_fallback"]["cluster"])
            nw = _cluster_means(list(j["groups"]["cand_new"]["ratio"][:, 0]), j["groups"]["cand_new"]["cluster"])
            fam.add(f"{r}|L{li} call0 FALLBACK vs NEW", _mwu(list(fb.values()), list(nw.values())),
                    fallback=float(np.median(list(fb.values()))), new=float(np.median(list(nw.values()))))
    first = {}
    for key, d in dv.items():
        ons = {li: x["onset"] for li, x in d.items()}
        valid = {li: o for li, o in ons.items() if o is not None}
        if valid:
            mo = min(valid.values())
            cand = [li for li, o in valid.items() if o == mo]
            first[key] = {"onset_call": mo, "layer": max(cand, key=lambda li: d[li]["call0"]), "layers_at_min": cand}
    N["div"] = dv
    N["div_first"] = first
    N["div_tests"] = fam.done()

    # ---- (6) Kaufman
    fam = Family("kauf")
    ka = {}
    for k in K:
        li = k["layer"]
        for sn, s in k["segs"].items():
            instr_c = _cluster_means(np.concatenate([s[g]["pf"] for g in ("cand_fallback", "cand_new", "cand_other")]),
                                     sum([s[g]["cluster"] for g in ("cand_fallback", "cand_new", "cand_other")], []))
            scene_c = {}
            for g in ("near", "opp", "far20", "far25"):
                scene_c.update({f"{g}|{c}": v for c, v in _cluster_means(s[g]["pf"], s[g]["cluster"]).items()})
            ka.setdefault(sn, {})[li] = {"instr": float(np.median(list(instr_c.values()))), "scene": float(np.median(list(scene_c.values()))),
                                         "ACT": s["ACT"], "chance": k["chance"], "proxy": k["proxy"],
                                         **{g: float(np.median(s[g]["pf"])) for g in s if g != "ACT"},
                                         "instr_rel_norm": float(np.median(np.concatenate([s[g]["rel_norm"] for g in ("cand_fallback", "cand_new")]))),
                                         "scene_rel_norm": float(np.median(np.concatenate([s[g]["rel_norm"] for g in ("near", "opp", "far20", "far25")])))}
            if not k["proxy"]:
                fam.add(f"{sn}|L{li} instr vs scene", _mwu(list(instr_c.values()), list(scene_c.values())),
                        diff=ka[sn][li]["instr"] - ka[sn][li]["scene"])
                fam.add(f"{sn}|L{li} instr − ACT", _wil(np.array(list(instr_c.values())) - s["ACT"]),
                        diff=ka[sn][li]["instr"] - s["ACT"])
    N["kauf"] = ka
    N["kauf_tests"] = fam.done()

    # ---- (7) Gallego / Kim
    fam = Family("gal")
    ga = {}
    for (r, li), j in G.items():
        g = j["gal"]
        row = {cmp: {"cca_r": float(np.nanmedian([x["cca_r"] for x in v])), "angle_excess": float(np.median([x["angle_excess"] for x in v]))}
               for cmp, v in g.items()}
        ok = {}
        for far in ("far20", "far25", "opp"):
            d = np.array([a["angle_excess"] - b["angle_excess"] for a, b in zip(g[f"near|{far}"], g["near|ctl4090"])])
            p = _wil(d)
            fam.add(f"{r}|L{li}|{far} angle excess > ctl4090", p, median=float(np.median(d)))
            ok[far] = {"cca_ok": row[f"near|{far}"]["cca_r"] >= row["near|ctl4090"]["cca_r"] - 0.1, "angle_diff": float(np.median(d)), "p": p}
        row["criterion"] = ok
        nn = j["nnls"]
        dd = _cluster_means([x["r2"] - x["r2_random"] for x in nn], [x["variant"] for x in nn])
        fam.add(f"{r}|L{li} NNLS R2 source > random", _wil(list(dd.values())), median=float(np.median(list(dd.values()))))
        sl = [x for x in nn if "slope_dest_minus_obj" in x and x["outcome"] in FALLBACK + NEW]
        if len(sl) > 5:
            from neuro_dyn_b import strat_perm

            v = np.array([x["slope_dest_minus_obj"] for x in sl])
            lab = np.array([x["outcome"] in NEW for x in sl])
            diff, p = strat_perm(v, lab, [x["variant"] for x in sl], n=2000)
            fam.add(f"{r}|L{li} slope NEW vs FALLBACK", p, diff=diff)
            row["slope_new_minus_fallback"] = diff
        row["nnls_r2_source"] = float(np.median([x["r2"] for x in nn]))
        row["nnls_r2_minus_random"] = float(np.median(list(dd.values())))
        ga.setdefault(r, {})[li] = row
    gt = fam.done()
    qmap = {t["test"]: t["q_bh"] for t in gt["tests"]}
    for r, d in ga.items():
        start = None
        for li in sorted(d):
            c = d[li]["criterion"]
            for far in ("far20", "far25", "opp"):
                c[far]["holds"] = bool(c[far]["cca_ok"] and c[far]["angle_diff"] > 0
                                       and qmap.get(f"{r}|L{li}|{far} angle excess > ctl4090", 1) < 0.05)
            if start is None and c["far20"]["holds"] and c["far25"]["holds"]:
                start = li
        ga[r]["first_layer_same_dynamics_diff_readout_far"] = start
    N["gal"] = ga
    N["gal_tests"] = gt

    # ---- (8) magnitude vs vector summary
    mv = {}
    for r in REP_ORDER:
        lay = REPS[r]["layers"]
        for t in DEC_TARGETS:
            vec = np.array([dec[r][li]["call_mean"][t] for li in lay])
            mg = np.array([float(np.mean(D[mag_key(r, li)]["acc"][t])) for li in lay])
            ka_ = magall_key(r)
            mv[f"{r}|{t}"] = {"vec_mean": float(vec.mean()), "mag_mean": float(mg.mean()), "gain_mean": float((vec - mg).mean()),
                              "vec_best": float(vec.max()), "vec_best_layer": int(lay[int(vec.argmax())]),
                              "mag_all_layers": float(np.mean(D[ka_]["acc"][t])) if ka_ in D else None}
    N["mag_vs_vec_decoding"] = mv
    N["mag_vs_vec_beta"] = {f"{r}|{s}": {li: {"vec": B[(r, li, "z")]["cells"][f"{s}|traj"]["beta"],
                                              "mag": B[(r, li, "mag")]["cells"][f"{s}|traj"]["beta"]} for li in REPS[r]["layers"]}
                            for r in BETA_REPS for s in BSETS}
    N["n_tests_total"] = sum(N[k]["n_tests"] for k in ("beta_tests", "div_tests", "kauf_tests", "gal_tests")) + sum(
        H[k]["n_tests"] for k in ("H1_tests", "H2_tests", "H3_tests"))
    with open(OUT / "numbers.json", "w") as f:
        json.dump(N, f, indent=1, default=_js)
    print("numbers.json written; total tests", N["n_tests_total"])


# ============================================================================ main
STAGES: dict = {"extract": stage_extract, "norm": stage_norm, "dec": stage_dec, "dim": stage_dim, "beta": stage_beta, "div": stage_div, "kauf": stage_kauf, "gal": stage_gal, "figs": stage_figs, "numbers": stage_numbers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=[*STAGES, "all"])
    a = ap.parse_args()
    for s in STAGES if a.stage == "all" else [a.stage]:
        t = time.time()
        STAGES[s]()
        print(f"[{s}] {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
