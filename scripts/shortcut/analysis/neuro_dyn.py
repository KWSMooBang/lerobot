#!/usr/bin/env python
"""T029 neuro_dyn: neuroscience population analyses on pi05lib activations (CPU only).

Stages
  weights  read the pi05lib read-in / readout weights from the HF safetensors (pure numpy, bf16 ok)
  load     cache per-episode vectors of the LIBERO-PRO run (runs/bench-pi05lib-pro_all)
  a1       Kaufman output-potent / output-null split (VLM -> expert read-in, expert -> action readout)

Hypotheses, metrics and baselines: scripts/shortcut/analysis/neuro_dyn_PREREG.md (committed before results).

(OPENBLAS_CORETYPE=ZEN and Haswell segfault in small-n dgemm with this numpy build (scipy-openblas 0.3.34)
 on the EPYC 9554; SkylakeX passes all checks, so use it.)

Run with the viz venv:
  OPENBLAS_CORETYPE=SkylakeX /workspace/VLA-shortcut-exps-kubectl/cand31/.venv-viz/bin/python \
      scripts/shortcut/analysis/neuro_dyn.py all
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RUN = Path("/outputs/VLA-shortcut-exps/runs/bench-pi05lib-pro_all")
OUT = Path("/outputs/VLA-shortcut-exps/analysis/neuro_dyn")
CACHE = OUT / "_cache"
CKPT = Path(
    "/cache/huggingface/hub/models--lerobot--pi05_libero_finetuned/snapshots/"
    "7152f13e34c6c115d808aea0e5d08aca74eac14b/model.safetensors"
)
VEC_LAYERS = (5, 11, 17)
SEGS = ("agentview", "wrist", "task", "state")
N_VLM_LAYERS = 18
D_VLM, D_EXP = 2048, 1024
N_STEPS = 10
ACT_DIMS = 7
MAX_CALLS = 11
WORKERS = 8
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SCENE_AXES = ("environment", "position", "object")

VLM_P = "model.paligemma_with_expert.paligemma.model.language_model."
EXP_P = "model.paligemma_with_expert.gemma_expert.model."


# ----------------------------------------------------------------------------- weights
def _read_safetensors(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name in names:
        h = header[name]
        a, b = h["data_offsets"]
        raw = np.asarray(mm[base + a : base + b])
        if h["dtype"] == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif h["dtype"] == "F32":
            arr = raw.view(np.float32)
        else:
            raise ValueError(h["dtype"])
        out[name] = arr.reshape(h["shape"]).copy()
    return out


def _sinusoid(t: float, dim: int = D_EXP, min_p: float = 4e-3, max_p: float = 4.0) -> np.ndarray:
    frac = np.linspace(0.0, 1.0, dim // 2)
    period = min_p * (max_p / min_p) ** frac
    x = t * (2 * math.pi / period)
    return np.concatenate([np.sin(x), np.cos(x)])


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def stage_weights() -> None:
    names = []
    for j in range(N_VLM_LAYERS):
        names += [f"{VLM_P}layers.{j}.self_attn.k_proj.weight", f"{VLM_P}layers.{j}.self_attn.v_proj.weight"]
        names += [f"{VLM_P}layers.{j}.input_layernorm.weight"]
    names += ["model.action_out_proj.weight", "model.action_out_proj.bias"]
    names += ["model.time_mlp_in.weight", "model.time_mlp_in.bias"]
    names += ["model.time_mlp_out.weight", "model.time_mlp_out.bias"]
    names += [f"{EXP_P}norm.dense.weight", f"{EXP_P}norm.dense.bias"]
    w = _read_safetensors(CKPT, names)
    out: dict[str, np.ndarray] = {}
    for j in range(N_VLM_LAYERS):
        g = 1.0 + w[f"{VLM_P}layers.{j}.input_layernorm.weight"]
        kv = np.concatenate(
            [w[f"{VLM_P}layers.{j}.self_attn.k_proj.weight"], w[f"{VLM_P}layers.{j}.self_attn.v_proj.weight"]]
        )
        out[f"readin_{j}"] = (kv * g[None, :]).astype(np.float32)  # 512 x 2048, acts on x/rms(x)
    # expert final AdaRMS scale per denoise step (t = 1.0, 0.9, ..., 0.1)
    scales = []
    for k in range(N_STEPS):
        te = _sinusoid(1.0 - 0.1 * k)
        h = _silu(w["model.time_mlp_in.weight"] @ te + w["model.time_mlp_in.bias"])
        c = _silu(w["model.time_mlp_out.weight"] @ h + w["model.time_mlp_out.bias"])
        mod = w[f"{EXP_P}norm.dense.weight"] @ c + w[f"{EXP_P}norm.dense.bias"]
        scales.append(mod[:D_EXP])
    out["exp_final_scale"] = np.stack(scales).astype(np.float32)  # 10 x 1024
    out["action_out_w"] = w["model.action_out_proj.weight"].astype(np.float32)  # 32 x 1024
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "weights.npz", **out)
    print("weights ->", CACHE / "weights.npz", {k: v.shape for k, v in list(out.items())[-3:]})


# ----------------------------------------------------------------------------- load
def _episodes() -> list[dict]:
    eps = []
    for f in sorted(glob.glob(str(RUN / "jobs/*/episodes.jsonl"))):
        for line in open(f):
            e = json.loads(line)
            if e.get("activations"):
                eps.append(e)
    return eps


def _load_one(path: str):
    z = np.load(path)
    vv = z["vlm_vec"].astype(np.float32)  # C,3,4,2048
    ev = z["exp_vec"].astype(np.float32)  # C,10,3,1024
    c = min(MAX_CALLS, vv.shape[0])
    vlm = np.full((MAX_CALLS, 3, 4, D_VLM), np.nan, np.float32)
    vlm[:c] = vv[:c]
    exp = np.full((MAX_CALLS, N_STEPS, 3, D_EXP), np.nan, np.float16)
    exp[:c] = ev[:c]
    # within-episode movement variance of exp_vec (layer 17, per step): centred second moment over calls
    e17 = ev[:, :, 2, :]  # C,10,1024
    cen = e17 - e17.mean(0, keepdims=True)
    move_cov_proj = None  # computed later from stored centred vectors
    return vlm, exp, cen.astype(np.float16), z["sig_h"][0].astype(np.float32), vv.shape[0], move_cov_proj


def stage_load() -> None:
    eps = _episodes()
    print("episodes with activations:", len(eps))
    meta = []
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(_load_one, [e["activations"] for e in eps], chunksize=16))
    vlm = np.stack([r[0] for r in res])
    exp = np.stack([r[1] for r in res])
    sig0 = np.stack([r[3] for r in res])
    ncalls = np.array([r[4] for r in res])
    # movement samples: concatenate centred per-call exp vectors (layer 17), keep episode index
    mv = np.concatenate([r[2] for r in res])
    mv_ep = np.concatenate([np.full(len(r[2]), i) for i, r in enumerate(res)])
    for e in eps:
        meta.append(
            {
                k: e.get(k)
                for k in (
                    "suite", "task_id", "init_idx", "axis", "level", "instruction", "original_instruction",
                    "success", "success_orig", "success_new", "first_grasp_t", "n_steps", "activations",
                )
            }
        )
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "vlm.npy", vlm)
    np.save(CACHE / "exp.npy", exp)
    np.save(CACHE / "sig0.npy", sig0)
    np.save(CACHE / "ncalls.npy", ncalls)
    np.save(CACHE / "move17.npy", mv)
    np.save(CACHE / "move17_ep.npy", mv_ep)
    json.dump(meta, open(CACHE / "meta.json", "w"))
    print("cached", vlm.shape, exp.shape, mv.shape)


# ----------------------------------------------------------------------------- helpers
def row_space(w: np.ndarray, energy: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis (D x r) of the row space of w, and singular values."""
    _, s, vt = np.linalg.svd(w.astype(np.float64), full_matrices=False)
    tol = s.max() * max(w.shape) * np.finfo(np.float32).eps
    r = int((s > tol).sum())
    if energy is not None:
        c = np.cumsum(s**2) / (s**2).sum()
        r = int(np.searchsorted(c, energy) + 1)
    return vt[:r].T, s


def pf(d: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Potent fraction ||P d||^2/||d||^2 for rows of d."""
    p = np.einsum("nd,dr->nr", d, basis)
    return (p**2).sum(1) / np.maximum((d**2).sum(1), 1e-30)


def gr(d: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gain ratio relative to an isotropic direction."""
    g = np.einsum("nd,rd->nr", d, w)
    iso = (w.astype(np.float64) ** 2).sum() / w.shape[1]
    return (g**2).sum(1) / np.maximum((d**2).sum(1), 1e-30) / iso


def rmsnorm(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6)


def wilcoxon(x: np.ndarray) -> float:
    from scipy.stats import wilcoxon as w

    x = x[np.isfinite(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return float("nan")
    return float(w(x).pvalue)


def bh(p: list[float]) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    o = np.argsort(pv)
    r = pv[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(n)
    qq[o] = np.minimum(r, 1)
    q[ok] = qq
    return q


def cluster_mean(vals: np.ndarray, keys: list[tuple]) -> tuple[np.ndarray, list[tuple]]:
    ks = sorted(set(keys))
    idx = {k: i for i, k in enumerate(ks)}
    s = np.zeros(len(ks))
    n = np.zeros(len(ks))
    for v, k in zip(vals, keys):
        if np.isfinite(v):
            s[idx[k]] += v
            n[idx[k]] += 1
    with np.errstate(invalid="ignore"):
        return s / n, ks


# ----------------------------------------------------------------------------- pairing
def build_pairs(meta: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """Return contrast -> list of (a, b) episode indices, b is the reference (none)."""
    none = {}
    by_axis: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        key = (m["suite"], m["task_id"], m["init_idx"])
        if m["axis"] == "none":
            none[key] = i
        by_axis.setdefault(m["axis"], []).append(i)
    pairs: dict[str, list[tuple[int, int]]] = {}

    def add(name, axis, level_filter=None):
        out = []
        for i in by_axis.get(axis, []):
            m = meta[i]
            if level_filter and m["level"] != level_filter:
                continue
            j = none.get((m["suite"], m["task_id"], m["init_idx"]))
            if j is not None:
                out.append((i, j))
        pairs[name] = out

    add("instr_para", "semantic")
    add("instr_goal", "task")
    add("scene_environment", "environment")
    add("scene_position", "position", "swap")
    add("scene_object", "object")
    pairs["scene_pooled"] = pairs["scene_environment"] + pairs["scene_position"] + pairs["scene_object"]
    btw = []
    for (s, t, k), i in none.items():
        for (s2, t2, k2), j in none.items():
            if s2 == s and k2 == k and t2 > t:
                btw.append((i, j))
    pairs["instr_between"] = btw
    return pairs


def pair_cluster(meta, p):
    a, b = p
    return (meta[b]["suite"], meta[b]["task_id"])


# ----------------------------------------------------------------------------- A1
def stage_a1(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    W = dict(np.load(CACHE / "weights.npz"))
    meta = json.load(open(CACHE / "meta.json"))
    vlm = np.load(CACHE / "vlm.npy", mmap_mode="r")
    exp = np.load(CACHE / "exp.npy", mmap_mode="r")
    sig0 = np.load(CACHE / "sig0.npy")
    pairs = build_pairs(meta)
    res: dict = {"n_pairs": {k: len(v) for k, v in pairs.items()}}

    # validity: identical observation at call 0 for semantic/task vs none (siglip is text-independent)
    val = {}
    for c in ("instr_para", "instr_goal", "scene_environment"):
        dd = [float(np.abs(sig0[a] - sig0[b]).max()) for a, b in pairs[c]]
        val[c] = {"max_abs_sig_h_diff_median": float(np.median(dd)), "frac_identical": float(np.mean(np.array(dd) == 0))}
    res["validity_sig_h_call0"] = val

    v0 = np.asarray(vlm[:, 0], dtype=np.float64)  # N,3,4,2048
    none_idx = np.array([i for i, m in enumerate(meta) if m["axis"] == "none"])
    contrasts = ["instr_para", "instr_goal", "instr_between", "scene_environment", "scene_position", "scene_object"]

    # ---------------- VLM read-in
    vlm_res: dict = {}
    tests = []  # (name, p)
    clus_store: dict = {}
    for li, l in enumerate(VEC_LAYERS):
        for wmode in ("next", "same"):
            j = l + 1 if wmode == "next" else l
            structural_null = wmode == "next" and j >= N_VLM_LAYERS
            if structural_null:
                j = l  # proxy
            Wt = W[f"readin_{j}"].astype(np.float64)
            B, s = row_space(Wt)
            B90, _ = row_space(Wt, 0.9)
            r, r90 = B.shape[1], B90.shape[1]
            # random-subspace null for PF (Haar): Beta(r/2,(D-r)/2) -> use 1000 draws on one vector set
            key = f"L{l}_{wmode}"
            ent = {"read_layer": j, "proxy_structural_null": structural_null, "rank": r, "rank90": r90,
                   "chance_pf": r / D_VLM, "chance_pf90": r90 / D_VLM, "segments": {}}
            beta = rng.beta(r / 2, (D_VLM - r) / 2, 1000)
            ent["random_pf_95ci"] = [float(np.quantile(beta, 0.025)), float(np.quantile(beta, 0.975))]
            for si, seg in enumerate(SEGS):
                X = v0[:, li, si]
                Xn = rmsnorm(X)
                sres = {}
                # ACT baseline: covariance of call-0 none vectors
                for norm_name, XX in (("norm", Xn), ("raw", X)):
                    C = XX[none_idx] - XX[none_idx].mean(0)
                    tot = (C**2).sum()
                    sres[f"act_pf_{norm_name}"] = float((np.einsum("nd,dr->nr", C, B) ** 2).sum() / tot)
                    sres[f"act_pf90_{norm_name}"] = float((np.einsum("nd,dr->nr", C, B90) ** 2).sum() / tot)
                    sres[f"act_gr_{norm_name}"] = float(
                        (np.einsum("nd,rd->nr", C, Wt) ** 2).sum() / tot / ((Wt**2).sum() / D_VLM)
                    )
                for c in contrasts:
                    pr = np.array(pairs[c])
                    if len(pr) == 0:
                        continue
                    keys = [pair_cluster(meta, p) for p in pairs[c]]
                    out = {}
                    for norm_name, XX in (("norm", Xn), ("raw", X)):
                        d = XX[pr[:, 0]] - XX[pr[:, 1]]
                        ok = (d**2).sum(1) > 0
                        pfv = np.where(ok, pf(d, B), np.nan)
                        pf90 = np.where(ok, pf(d, B90), np.nan)
                        grv = np.where(ok, gr(d, Wt), np.nan)
                        out[norm_name] = {
                            "pf_median": float(np.nanmedian(pfv)),
                            "pf90_median": float(np.nanmedian(pf90)),
                            "gr_median": float(np.nanmedian(grv)),
                            "frac_zero_diff": float(1 - ok.mean()),
                            "rel_norm_median": float(np.median(np.linalg.norm(d, axis=1) / np.linalg.norm(XX[pr[:, 1]], axis=1))),
                        }
                        if norm_name == "norm":
                            cm, ks = cluster_mean(pfv, keys)
                            cg, _ = cluster_mean(grv, keys)
                            clus_store[(key, seg, c)] = (dict(zip(ks, cm)), dict(zip(ks, cg)))
                    # scale channel
                    lr = np.log(np.sqrt((X[pr[:, 0]] ** 2).mean(1)) / np.sqrt((X[pr[:, 1]] ** 2).mean(1)))
                    out["dlog_rms_median_abs"] = float(np.median(np.abs(lr)))
                    sres[c] = out
                ent["segments"][seg] = sres
            vlm_res[key] = ent
    res["vlm_readin"] = vlm_res

    # tests (primary family H1 uses wmode=next, layers 5,11; all segs reported, task seg is primary)
    for (key, seg, c), (cpf, cgr) in clus_store.items():
        if not c.startswith("instr"):
            continue
        ent = vlm_res[key]
        # vs scene pooled (cluster-matched)
        scene = [clus_store.get((key, seg, f"scene_{s}")) for s in SCENE_AXES]
        for metric, ix in (("pf", 0), ("gr", 1)):
            mine = (cpf, cgr)[ix]
            comp = {}
            for k in mine:
                vs = [sc[ix].get(k) for sc in scene if sc is not None and k in sc[ix]]
                vs = [v for v in vs if np.isfinite(v)]
                if vs and np.isfinite(mine[k]):
                    comp[k] = mine[k] - np.mean(vs)
            diff = np.array(list(comp.values()))
            tests.append((f"A1.vlm.{key}.{seg}.{c}.{metric}_vs_scene", wilcoxon(diff), float(np.median(diff)) if len(diff) else float("nan")))
            act = ent["segments"][seg]["act_pf_norm" if metric == "pf" else "act_gr_norm"]
            diff2 = np.array([v - act for v in mine.values() if np.isfinite(v)])
            tests.append((f"A1.vlm.{key}.{seg}.{c}.{metric}_vs_act", wilcoxon(diff2), float(np.median(diff2)) if len(diff2) else float("nan")))
            chance = ent["chance_pf"] if metric == "pf" else 1.0
            diff3 = np.array([v - chance for v in mine.values() if np.isfinite(v)])
            tests.append((f"A1.vlm.{key}.{seg}.{c}.{metric}_vs_chance", wilcoxon(diff3), float(np.median(diff3)) if len(diff3) else float("nan")))

    # time course of instr_goal PF (task seg, layers 5/11, read layer l+1)
    tc = {}
    vlm_all = vlm
    pr = np.array(pairs["instr_goal"])
    for li, l in enumerate(VEC_LAYERS[:2]):
        B, _ = row_space(W[f"readin_{l + 1}"])
        for si, seg in enumerate(SEGS):
            row = []
            for c in range(MAX_CALLS):
                a = rmsnorm(np.asarray(vlm_all[pr[:, 0], c, li, si], np.float64))
                b = rmsnorm(np.asarray(vlm_all[pr[:, 1], c, li, si], np.float64))
                d = a - b
                ok = np.isfinite(d).all(1) & ((d**2).sum(1) > 0)
                row.append(float(np.median(pf(d[ok], B))) if ok.sum() > 10 else float("nan"))
            tc[f"L{l}.{seg}"] = row
    res["vlm_instr_goal_pf_by_call"] = tc

    # ---------------- expert readout
    e0 = np.asarray(exp[:, 0], dtype=np.float64)  # N,10,3,1024
    mv = np.load(CACHE / "move17.npy", mmap_mode="r")
    mv_ep = np.load(CACHE / "move17_ep.npy")
    sel = rng.choice(len(mv), size=min(40000, len(mv)), replace=False)
    sel.sort()
    mvs = np.asarray(mv[sel], np.float64)  # n,10,1024
    exp_res: dict = {"chance_pf": ACT_DIMS / D_EXP, "steps": {}}
    Wo_raw = W["action_out_w"][:ACT_DIMS].astype(np.float64)
    exp_clus = {}
    for k in range(N_STEPS):
        Wo = Wo_raw * (1 + W["exp_final_scale"][k].astype(np.float64))[None, :]
        Bo, _ = row_space(Wo)
        stp = {}
        for li, l in enumerate(VEC_LAYERS):
            X = e0[:, k, li]
            lres = {}
            M = mvs[:, k] if l == 17 else None
            if M is not None:
                ok = (M**2).sum(1) > 0
                lres["move_pf_median"] = float(np.median(pf(M[ok], Bo)))
                lres["move_pf_pooled"] = float((np.einsum("nd,dr->nr", M[ok], Bo) ** 2).sum() / (M[ok] ** 2).sum())
                lres["move_gr_pooled"] = float(
                    (np.einsum("nd,rd->nr", M[ok], Wo) ** 2).sum() / (M[ok] ** 2).sum() / ((Wo**2).sum() / D_EXP)
                )
            C = X[none_idx] - X[none_idx].mean(0)
            lres["act_pf"] = float((np.einsum("nd,dr->nr", C, Bo) ** 2).sum() / (C**2).sum())
            for c in contrasts:
                prr = np.array(pairs[c])
                d = X[prr[:, 0]] - X[prr[:, 1]]
                ok = (d**2).sum(1) > 0
                pfv = np.where(ok, pf(d, Bo), np.nan)
                grv = np.where(ok, gr(d, Wo), np.nan)
                lres[c] = {"pf_median": float(np.nanmedian(pfv)), "gr_median": float(np.nanmedian(grv)),
                           "pf_pooled": float(np.nansum((np.einsum("nd,dr->nr", d[ok], Bo) ** 2)) / (d[ok] ** 2).sum())}
                if l == 17:
                    keys = [pair_cluster(meta, p) for p in pairs[c]]
                    cm, ks = cluster_mean(pfv, keys)
                    cg, _ = cluster_mean(grv, keys)
                    exp_clus[(k, c)] = (cm, cg, ks)
            stp[f"L{l}"] = lres
        exp_res["steps"][k] = stp
        # H2 tests at step k (layer 17): instr_goal / instr_between vs MOVE (pooled constant)
        for c in ("instr_goal", "instr_between", "instr_para"):
            cm, cg, _ = exp_clus[(k, c)]
            tests.append((f"A1.exp.step{k}.{c}.pf_vs_move", wilcoxon(cm - stp["L17"]["move_pf_pooled"]),
                          float(np.nanmedian(cm - stp["L17"]["move_pf_pooled"]))))
            tests.append((f"A1.exp.step{k}.{c}.gr_vs_move", wilcoxon(cg - stp["L17"]["move_gr_pooled"]),
                          float(np.nanmedian(cg - stp["L17"]["move_gr_pooled"]))))
    res["expert_readout"] = exp_res

    # ---------------- decoding: task identity from potent vs dimension-matched null (leave-one-init-out)
    res["decode"] = decode_tasks(meta, v0, e0, W, rng)

    q = bh([t[1] for t in tests])
    res["tests"] = [{"name": n, "p": p, "q_bh": float(qq), "median_diff": md} for (n, p, md), qq in zip(tests, q)]
    res["n_tests"] = len(tests)
    res["n_sig_q05"] = int(np.nansum(q < 0.05))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a1").mkdir(exist_ok=True)
    json.dump(res, open(OUT / "a1" / "a1_results.json", "w"), indent=1, default=float)
    plot_a1(res)
    return res


def _lda_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_pc: int | None = None) -> float:
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    acc = []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        Xtr, Xte = X[tr], X[te]
        if n_pc is not None and Xtr.shape[1] > n_pc:
            p = PCA(n_pc).fit(Xtr)
            Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, y[tr])
        acc.append((clf.predict(Xte) == y[te]).mean())
    return float(np.mean(acc))


def _null_basis(B: np.ndarray, dim: int, r: int, rng) -> np.ndarray:
    g = rng.standard_normal((dim, r))
    g -= B @ (B.T @ g)
    q, _ = np.linalg.qr(g)
    return q


def decode_tasks(meta, v0, e0, W, rng, n_draw: int = 50) -> dict:
    out = {}
    Wo = W["action_out_w"][:ACT_DIMS].astype(np.float64) * (1 + W["exp_final_scale"][0].astype(np.float64))[None, :]
    Bo, _ = row_space(Wo)
    Bv, _ = row_space(W["readin_12"])  # vlm L11 output -> read by layer 12
    for suite in SUITES:
        idx = np.array([i for i, m in enumerate(meta) if m["axis"] == "none" and m["suite"] == suite])
        y = np.array([meta[i]["task_id"] for i in idx])
        g = np.array([meta[i]["init_idx"] for i in idx])
        so = {}
        X = e0[idx, 0, 2]  # step 0, layer 17
        so["exp_L17_potent7"] = _lda_cv(X @ Bo, y, g)
        nulls = [_lda_cv(X @ _null_basis(Bo, D_EXP, ACT_DIMS, rng), y, g) for _ in range(n_draw)]
        so["exp_L17_null7_mean"] = float(np.mean(nulls))
        so["exp_L17_null7_q"] = [float(np.quantile(nulls, 0.05)), float(np.quantile(nulls, 0.95))]
        for si, seg in ((0, "agentview"), (2, "task")):
            Xv = rmsnorm(v0[idx, 1, si])
            so[f"vlm_L11_{seg}_potent"] = _lda_cv(Xv @ Bv, y, g, 20)
            nb = [_lda_cv(Xv @ _null_basis(Bv, D_VLM, Bv.shape[1], rng), y, g, 20) for _ in range(5)]
            so[f"vlm_L11_{seg}_null_mean"] = float(np.mean(nb))
        out[suite] = so
    return out


# ----------------------------------------------------------------------------- plots
def plot_a1(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    od = OUT / "a1"
    contrasts = ["instr_para", "instr_goal", "instr_between", "scene_environment", "scene_position", "scene_object"]
    labels = ["paraphrase", "new goal", "between tasks", "environment", "position", "object"]
    cols = ["#2a6fdb", "#1b3f8b", "#7aa6ef", "#d9822b", "#b5541c", "#e8b07a"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row")
    for row, key in enumerate(("L5_next", "L11_next")):
        ent = res["vlm_readin"][key]
        for si, seg in enumerate(SEGS):
            ax = axes[row, si]
            sres = ent["segments"][seg]
            vals = [sres[c]["norm"]["pf_median"] if c in sres else np.nan for c in contrasts]
            ax.bar(range(len(vals)), vals, color=cols)
            ax.axhline(ent["chance_pf"], color="k", ls="--", lw=1, label="random subspace")
            ax.axhline(sres["act_pf_norm"], color="gray", ls=":", lw=1.5, label="all activity (ACT)")
            ax.set_xticks(range(len(vals)), labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"VLM out L{key[1:].split('_')[0]} → read by L{ent['read_layer']} · {seg}", fontsize=9)
            if si == 0:
                ax.set_ylabel("potent fraction (median)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("A1 Kaufman: share of the difference vector inside the expert's K/V read-in subspace (rms-normalised, call 0)")
    fig.tight_layout()
    fig.savefig(od / "a1_vlm_potent_fraction.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    st = res["expert_readout"]["steps"]
    ks = sorted(st, key=int)
    for c, lab, col in zip(contrasts, labels, cols):
        axes[0].plot([int(k) for k in ks], [st[k]["L17"][c]["pf_pooled"] for k in ks], "-o", color=col, label=lab, ms=3)
    axes[0].plot([int(k) for k in ks], [st[k]["L17"]["move_pf_pooled"] for k in ks], "-", color="k", lw=2, label="movement (within ep)")
    axes[0].axhline(res["expert_readout"]["chance_pf"], color="k", ls="--", lw=1, label="random 7-d")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("denoise step k (t = 1 − 0.1k)")
    axes[0].set_ylabel("potent fraction (pooled), log")
    axes[0].set_title("expert L17 → action_out_proj (7 dims)")
    axes[0].legend(fontsize=7)
    dec = res["decode"]
    su = list(dec)
    x = np.arange(len(su))
    axes[1].bar(x - 0.2, [dec[s]["exp_L17_potent7"] for s in su], 0.4, label="potent 7-d", color="#1b3f8b")
    axes[1].bar(x + 0.2, [dec[s]["exp_L17_null7_mean"] for s in su], 0.4, label="null 7-d (random, mean)", color="#aaaaaa")
    axes[1].axhline(0.1, color="k", ls="--", lw=1)
    axes[1].set_xticks(x, [s.replace("libero_", "") for s in su])
    axes[1].set_ylabel("task decoding accuracy (leave-one-init-out)")
    axes[1].set_title("task identity at call 0, expert L17 step 0")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(od / "a1_expert_potent.png", dpi=130)
    plt.close(fig)

    tc = res["vlm_instr_goal_pf_by_call"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, v in tc.items():
        ax.plot(range(len(v)), v, "-o", ms=3, label=k)
    ax.axhline(res["vlm_readin"]["L5_next"]["chance_pf"], color="k", ls="--", lw=1)
    ax.set_xlabel("call index")
    ax.set_ylabel("potent fraction of (new goal − none)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("A1: new-goal difference over the episode")
    fig.tight_layout()
    fig.savefig(od / "a1_instr_goal_by_call.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["weights", "load", "a1", "all"])
    a = ap.parse_args()
    if a.stage in ("weights", "all"):
        stage_weights()
    if a.stage in ("load", "all"):
        stage_load()
    if a.stage in ("a1", "all"):
        r = stage_a1()
        print(json.dumps({k: r[k] for k in ("n_pairs", "validity_sig_h_call0", "n_tests", "n_sig_q05")}, indent=1))


if __name__ == "__main__":
    main()
