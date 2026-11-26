#!/usr/bin/env python
"""T032 sae_vlm_traj: T015 VLM SAEs (pooled_all, pg00-pg17) applied to the T028 rich re-extraction (CPU only).

Stages (definitions / thresholds / hypotheses: sae_vlm_traj_PREREG.md, copied to the output folder)
  trainlab  training-frame table (HF libero: task code, scene, phase from the gripper command, progress)
  encode    T028 pooled_all inputs (token-count weighted mean of the 6 segment means) -> sparse SAE codes + A1 metrics
  labels    (A2) per-feature training fire rates per task / scene / suite / phase, tags, top-activating frames
  traj      (A3) per-case heatmaps (feature x call) and fixed-PCA trajectories of the code vector
  fallback  (A4) instructed vs executed feature index on fallback trials + same-observation pairs
  basket    (A5) basket-position features: graded vs on/off
  kauf      (A6) decoder direction potent fraction under the next layer's K/V read-in
  dec       (A7) decoding with SAE codes vs the pooled input vector (T030 classifier)
  expert    (A7 opt.) VLM feature x T031 expert feature correlation over U40 calls
  numbers   numbers.json
  figs      figures

Run (OPENBLAS ZEN/Haswell kernels segfault on this EPYC, see neuro_dyn.py):
  OPENBLAS_CORETYPE=SkylakeX .venv-sv/bin/python scripts/shortcut/analysis/sae_vlm_traj.py <stage|all>
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
import layer_pop as lp  # noqa: E402
from neuro_dyn import bh, row_space  # noqa: E402

RICH = lp.RICH
OUT = Path("/outputs/VLA-shortcut-exps/analysis/sae_vlm_traj")
CACHE = OUT / "_cache"
FIG = OUT / "fig"
SAE_ROOT = Path("/data0/woosung/datasets/VLA-shortcut-exps/vlm_sae")
HF_LIBERO = Path("/data0/woosung/datasets/VLA-shortcut-exps/hf_libero")
EXP_SAE = Path("/outputs/VLA-shortcut-exps/sae_ae/stage1")
WORKERS = 8
N_L = 18
LAYERS = tuple(range(N_L))  # pg00..pg17 (emb is not in T028)
K = 100
NF = 2048
MASSIVE_PG17 = (1577, 1959)
SET_OF = {
    "train": "U40 train",
    "cand": "goal cand31",
    "l90": "L90",
    "bskA": "basket A100",
    "bskV": "basket 4090",
}
PHASE4 = lp.PHASE4


def lname(li: int) -> str:
    return f"pg{li:02d}"


def sae_dir(li: int) -> Path:
    b = "a" if li <= 5 else ("b" if li <= 11 else "c")
    return SAE_ROOT / f"sae-vlm-{b}" / "pooled_all" / lname(li) / "seed0"


def _js(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


def save_json(name: str, obj) -> None:
    p = CACHE / f"{name}.json"
    p.write_text(json.dumps(obj, default=_js, indent=1) + "\n")


def load_json(name: str):
    return json.loads((CACHE / f"{name}.json").read_text())


# ============================================================================ SAE (numpy re-implementation of model.TopKSAE)
class NpSAE:
    def __init__(self, d: Path):
        from safetensors.numpy import load_file

        w = load_file(d / "sae.safetensors")
        self.W_enc = w["W_enc"].astype(np.float32)  # (n, d)
        self.W_dec = w["W_dec"].astype(np.float32)  # (d, n), unit columns
        self.b_pre = w["b_pre"].astype(np.float32)
        self.cfg = json.loads((d / "config.json").read_text())
        self.k = int(self.cfg["k"])
        self.eps = float(self.cfg.get("eps", 1e-6))
        self.c_mse = float(self.cfg.get("c_mse", 1.0))

    def normalize(self, x: np.ndarray):
        xc = x.astype(np.float32) - self.b_pre
        mu = xc.mean(-1, keepdims=True)
        xc = xc - mu
        nrm = np.maximum(np.linalg.norm(xc, axis=-1, keepdims=True), self.eps)
        return xc / nrm, mu, nrm

    def encode(self, x: np.ndarray, k: int | None = None):
        """x (N, d) -> (vals (N,k) >= 0, idx (N,k)) sorted by value descending (torch.topk order)."""
        k = k or self.k
        xt, _, _ = self.normalize(x)
        pre = xt @ self.W_enc.T
        idx = np.argpartition(-pre, k - 1, axis=1)[:, :k]
        v = np.take_along_axis(pre, idx, 1)
        o = np.argsort(-v, axis=1, kind="stable")
        idx, v = np.take_along_axis(idx, o, 1), np.take_along_axis(v, o, 1)
        return np.maximum(v, 0.0), idx

    def decode_norm(self, vals: np.ndarray, idx: np.ndarray) -> np.ndarray:
        z = np.zeros((len(vals), self.W_dec.shape[1]), np.float32)
        np.put_along_axis(z, idx, vals, 1)
        return z @ self.W_dec.T


# ============================================================================ (trainlab) training frames
def task_codes() -> list[str]:
    S = list(csv.DictReader(open(lp.SEM_CSV)))
    codes = [r["task"] for r in S if r["set"] == "train"]
    order = {"S": 0, "O": 1, "G": 2, "L": 3}
    return sorted(codes, key=lambda t: (order[t[0]], int(t[1:])))


def _norm_txt(s: str) -> str:
    return " ".join(s.lower().replace(".", " ").split())


def train_scene() -> dict:
    """T027 scene label per training task code (from the T030 meta of U40 episodes)."""
    sc = {}
    for m in lp.meta():
        if m["t027_set"] == "train":
            sc[m["task"]] = m["scene"]
    return sc


def frame_phase(a6: np.ndarray) -> tuple[list, int | None, int | None]:
    """PREREG A2: g = first close command (a6 > 0), r = first open command after g; T027 call windows x 10."""
    n = len(a6)
    lab: list = [None] * n
    close = np.where(a6 > 0)[0]
    if len(close) == 0:
        return lab, None, None
    g = int(close[0])
    op = np.where((a6 <= 0) & (np.arange(n) > g))[0]
    r = int(op[0]) if len(op) else None
    for f in range(n):
        if f < g - 20:
            lab[f] = "reach"
        elif f < g + 20:
            lab[f] = "grasp"
        elif r is None or f < r - 10:
            lab[f] = "carry"
        elif f < r + 30:
            lab[f] = "release"
    return lab, g, r


def stage_trainlab() -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    CACHE.mkdir(parents=True, exist_ok=True)
    codes = task_codes()
    S = {r["task"]: r["instruction"] for r in csv.DictReader(open(lp.SEM_CSV)) if r["set"] == "train"}
    bytxt = {_norm_txt(S[c]): c for c in codes}
    tasks = pd.read_parquet(HF_LIBERO / "meta" / "tasks.parquet")
    ti2code = {}
    for txt, row in tasks.iterrows():
        c = bytxt.get(_norm_txt(str(txt)))
        assert c is not None, txt
        ti2code[int(row["task_index"])] = c
    assert len(set(ti2code.values())) == 40
    files = sorted((HF_LIBERO / "data").rglob("*.parquet"))
    cols = ["index", "episode_index", "frame_index", "task_index", "action"]
    parts = [pq.read_table(f, columns=cols).to_pandas() for f in files]
    df = pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)
    meta = pd.read_parquet(sae_dir(0) / "codes_meta.parquet")
    assert len(df) == len(meta) and (df["index"].values == meta["index"].values).all()
    assert (df["episode_index"].values == meta["episode_index"].values).all()
    assert (df["frame_index"].values == meta["frame_index"].values).all()
    a6 = np.stack(df["action"].values)[:, 6].astype(np.float32)
    ep = df["episode_index"].values
    phase = np.full(len(df), -1, np.int8)
    prog = np.zeros(len(df), np.float32)
    gfr = {}
    for e in np.unique(ep):
        rows = np.where(ep == e)[0]
        rows = rows[np.argsort(df["frame_index"].values[rows])]
        lab, g, r = frame_phase(a6[rows])
        phase[rows] = [PHASE4.index(x) if x else -1 for x in lab]
        prog[rows] = np.arange(len(rows)) / max(len(rows) - 1, 1)
        gfr[int(e)] = (g, r, len(rows))
    code = np.array([ti2code[int(t)] for t in df["task_index"].values])
    tidx = np.array([codes.index(c) for c in code], np.int16)
    np.savez(
        CACHE / "train_frames.npz",
        task=tidx,
        episode=ep.astype(np.int32),
        frame=df["frame_index"].values.astype(np.int32),
        phase=phase,
        prog=prog,
        a6=a6,
    )
    nog = sum(1 for v in gfr.values() if v[0] is None)
    info = {
        "codes": codes,
        "scene": train_scene(),
        "n_frames": len(df),
        "n_eps": len(gfr),
        "eps_without_close": nog,
        "phase_counts": {p: int((phase == i).sum()) for i, p in enumerate(PHASE4)},
        "unlabelled": int((phase < 0).sum()),
        "ti2code": ti2code,
    }
    save_json("trainlab", info)
    print(json.dumps({k: v for k, v in info.items() if k not in ("ti2code", "scene")}, indent=1))


# ============================================================================ (encode) T028 -> codes
def _seg_ntok_one(m: dict) -> np.ndarray:
    z = np.load(RICH / m["path"])
    return z["seg_ntok"][:, :6].astype(np.int32)


def seg_ntok() -> np.ndarray:
    p = CACHE / "seg_ntok.npy"
    if p.exists():
        return np.load(p)
    M = lp.meta()
    with ProcessPoolExecutor(WORKERS) as ex:
        parts = list(ex.map(_seg_ntok_one, M, chunksize=8))
    a = np.concatenate(parts)
    assert len(a) == M[-1]["off"] + M[-1]["C"]
    np.save(p, a)
    return a


def _check_cache_one(m: dict) -> float:
    z = np.load(RICH / m["path"])
    a = lp._mm("vlm_seg")[:, m["off"] : m["off"] + m["C"]]
    b = z["vlm_seg"].transpose(1, 0, 2, 3)
    return float(np.abs(a.astype(np.float32) - b.astype(np.float16).astype(np.float32)).max())


def pooled(li: int, n: np.ndarray | None = None) -> np.ndarray:
    """PREREG 1: x = sum_s n_s seg_s / sum_s n_s over the 6 segments (fp32)."""
    n = seg_ntok() if n is None else n
    a = lp._mm("vlm_seg")[li]  # (Ntot, 6, 2048) fp16
    out = np.empty((a.shape[0], a.shape[2]), np.float32)
    w = n.astype(np.float32)
    for s in range(0, a.shape[0], 8192):
        seg = np.asarray(a[s : s + 8192], np.float32)
        ww = w[s : s + 8192]
        out[s : s + 8192] = (seg * ww[..., None]).sum(1) / np.maximum(ww.sum(1, keepdims=True), 1.0)
    return out


def encode_job(li: int) -> dict:
    sae = NpSAE(sae_dir(li))
    x = pooled(li)
    N = len(x)
    vals = np.empty((N, K), np.float32)
    idx = np.empty((N, K), np.int16)
    sq_n = 0.0
    sq_raw = np.zeros(N, np.float64)
    mean = x.mean(0, dtype=np.float64)
    xt_m = np.zeros(NF, np.float64)
    mshare = np.zeros(N, np.float32)
    sq_raw_nm = np.zeros(N, np.float64)
    keep = np.setdiff1d(np.arange(NF), MASSIVE_PG17)
    for s in range(0, N, 4096):
        xb = x[s : s + 4096]
        xt, mu, nrm = sae.normalize(xb)
        v, ix = sae.encode(xb)
        xth = sae.decode_norm(v, ix)
        xh = xth * nrm + mu + sae.b_pre
        vals[s : s + 4096], idx[s : s + 4096] = v, ix
        sq_n += float(((xt - xth) ** 2).sum())
        sq_raw[s : s + 4096] = ((xb - xh) ** 2).sum(1)
        sq_raw_nm[s : s + 4096] = ((xb - xh)[:, keep] ** 2).sum(1)
        xc = (xt * nrm) ** 2
        mshare[s : s + 4096] = xc[:, list(MASSIVE_PG17)].sum(1) / np.maximum(xc.sum(1), 1e-30)
        xt_m += xt.sum(0)
    var = ((x - mean) ** 2).sum(1)
    var_nm = ((x[:, keep] - mean[keep]) ** 2).sum(1)
    np.savez(CACHE / f"codes_{lname(li)}.npz", vals=vals, idx=idx)
    fire = np.zeros(NF, np.int64)
    np.add.at(fire, idx[vals > 0].astype(np.int64), 1)
    M = lp.meta()
    setrows = defaultdict(list)
    for m in M:
        setrows[m["t027_set"]].extend(range(m["off"], m["off"] + m["C"]))
    tr = json.loads((sae_dir(li) / "metrics.json").read_text())
    ff_tr = np.load(sae_dir(li) / "fire_freq.npy")
    from scipy.stats import spearmanr

    ff = fire / N
    act = fire > 0
    per_set = {}
    for k_, rows in setrows.items():
        r = np.array(rows)
        per_set[k_] = {"ve": float(1 - sq_raw[r].sum() / var[r].sum()), "n": len(r)}
    return {
        "layer": li,
        "n": N,
        "nmse": sq_n / N / sae.c_mse,
        "ve": float(1 - sq_raw.sum() / var.sum()),
        "ve_nomassive": float(1 - sq_raw_nm.sum() / var_nm.sum()),
        "never_fire": float((fire == 0).mean()),
        "train_nmse": tr["nmse"],
        "train_ve": tr["var_explained"],
        "train_never_fire": tr["stats"]["features_never_fire"] / NF,
        "fire_rho": float(spearmanr(ff, ff_tr)[0]),
        "t028_active_rare_in_train": float((act & (ff_tr < 1e-3)).sum() / max(act.sum(), 1)),
        "t028_fire_mass_on_train_rare": float(ff[ff_tr < 1e-3].sum() / ff.sum()),
        "massive_share_median": float(np.median(mshare)),
        "massive_share_p90": float(np.quantile(mshare, 0.9)),
        "norm_median": float(np.median(np.linalg.norm(x, axis=1))),
        "per_set": per_set,
        "val_zero_frac": float((vals <= 0).mean()),
    }


def stage_encode() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    M = lp.meta()
    rng = np.random.default_rng(0)
    pick = [M[i] for i in rng.choice(len(M), 20, replace=False)]
    with ProcessPoolExecutor(WORKERS) as ex:
        diffs = list(ex.map(_check_cache_one, pick))
    print("cache vs npz max |diff| over 20 ep:", max(diffs), flush=True)
    assert max(diffs) == 0.0
    n = seg_ntok()
    tot = n.sum(1)
    print("seg_ntok total tokens: min", tot.min(), "max", tot.max(), flush=True)
    with ProcessPoolExecutor(6) as ex:  # each job holds ~1.5 GB
        res = list(ex.map(encode_job, LAYERS))
    save_json(
        "encode",
        {
            "cache_check_maxdiff": max(diffs),
            "ntok_min": int(tot.min()),
            "ntok_max": int(tot.max()),
            "layers": res,
        },
    )
    for r in res:
        print(
            f"{lname(r['layer'])} nmse {r['nmse']:.4f} (train {r['train_nmse']:.4f}) VE {r['ve']:.3f} "
            f"(train {r['train_ve']:.3f}) never {r['never_fire']:.3f} rho {r['fire_rho']:.2f}",
            flush=True,
        )


def codes(li: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(CACHE / f"codes_{lname(li)}.npz")
    return z["vals"], z["idx"]


def dense(li: int, rows: np.ndarray | None = None) -> np.ndarray:
    v, ix = codes(li)
    if rows is not None:
        v, ix = v[rows], ix[rows]
    z = np.zeros((len(v), NF), np.float32)
    np.put_along_axis(z, ix.astype(np.int64), v, 1)
    return z


# ============================================================================ (labels) A2
def train_codes(li: int):
    d = sae_dir(li)
    return np.load(d / "codes_val.npy", mmap_mode="r"), np.load(d / "codes_idx.npy", mmap_mode="r")


def label_job(li: int) -> dict:
    TF = np.load(CACHE / "train_frames.npz")
    info = load_json("trainlab")
    codes_ = info["codes"]
    scene = info["scene"]
    v, ix = train_codes(li)
    v = np.asarray(v, np.float32)
    ix = np.asarray(ix, np.int64)
    N = len(v)
    on = v > 0
    rows = np.broadcast_to(np.arange(N)[:, None], ix.shape)[on]
    feats = ix[on]
    vals = v[on]

    def grate(g: np.ndarray, ng: int) -> np.ndarray:
        cnt = np.zeros((ng, NF), np.float64)
        np.add.at(cnt, (g[rows], feats), 1.0)
        n = np.bincount(g[g >= 0], minlength=ng).astype(np.float64)
        return cnt / np.maximum(n, 1)[:, None]

    task = TF["task"].astype(np.int64)
    r_t = grate(task, 40)  # (40, NF)
    ph = TF["phase"].astype(np.int64)
    phm = ph >= 0
    cntp = np.zeros((4, NF))
    sel = phm[rows]
    np.add.at(cntp, (ph[rows][sel], feats[sel]), 1.0)
    rho_p = cntp / np.bincount(ph[phm], minlength=4)[:, None]
    pb = np.minimum((TF["prog"] * 5).astype(np.int64), 4)
    r_prog = grate(pb, 5)
    phi = np.bincount(feats, minlength=NF) / N
    # scene / suite groups (rates = mean of member task rates)
    sc_names = sorted({scene[c] for c in codes_})
    sc_of = np.array([sc_names.index(scene[c]) for c in codes_])
    su_names = ["S", "O", "G", "L"]
    su_of = np.array([su_names.index(c[0]) for c in codes_])
    r_sc = np.stack([r_t[sc_of == g].mean(0) for g in range(len(sc_names))])
    r_su = np.stack([r_t[su_of == g].mean(0) for g in range(4)])
    sc_size = np.bincount(sc_of, minlength=len(sc_names))
    same_layout = {g for g, nm in enumerate(sc_names) if sc_size[g] >= 2 and nm not in ("SPATIAL", "OBJECT")}
    tags, rows_out = [], []
    for f in range(NF):
        rt = r_t[:, f]
        tag, sub = "mixed", ""
        ts = int(rt.argmax())
        oth = np.delete(rt, ts).mean()
        ratio_t = rt[ts] / max(oth, 1e-9)
        g = int(r_sc[:, f].argmax())
        ratio_g = r_sc[g, f] / max(np.delete(r_sc[:, f], g).mean(), 1e-9)
        memb = rt[sc_of == g]
        s = int(r_su[:, f].argmax())
        ratio_s = r_su[s, f] / max(np.delete(r_su[:, f], s).mean(), 1e-9)
        membs = rt[su_of == s]
        rp = rho_p[:, f]
        phase_flag = bool(rp.max() >= 0.05 and rp.max() >= 3 * rp.min())
        if phi[f] < 1e-3:
            tag = "rare"
        elif rt.min() >= 0.5:
            tag = "always-on"
        elif rt[ts] >= 0.05 and ratio_t >= 5:
            tag = "task"
            gs = sc_of[ts]
            if gs in same_layout:
                mates = rt[(sc_of == gs) & (np.arange(40) != ts)]
                sub = "instruction" if rt[ts] >= 5 * max(mates.mean(), 1e-9) else "task-shared-scene"
            else:
                sub = "task-confounded"
        elif sc_size[g] >= 2 and r_sc[g, f] >= 0.05 and ratio_g >= 5 and memb.min() >= memb.max() / 3:
            tag = "scene"
        elif r_su[s, f] >= 0.05 and ratio_s >= 5 and membs.min() >= membs.max() / 3:
            tag = "suite"
        elif phase_flag:
            tag = "phase"
        tags.append(tag)
        rows_out.append(
            {
                "layer": lname(li),
                "feature": f,
                "tag": tag,
                "sub": sub,
                "phase_flag": int(phase_flag),
                "pref_phase": PHASE4[int(rp.argmax())],
                "fire_rate": phi[f],
                "top_task": codes_[ts],
                "top_task_rate": rt[ts],
                "task_ratio": ratio_t,
                "top_scene": sc_names[g],
                "scene_ratio": ratio_g,
                "top_suite": su_names[s],
                "suite_ratio": ratio_s,
                "min_task_rate": rt.min(),
                **{f"ph_{p}": rp[i] for i, p in enumerate(PHASE4)},
                "prog_peak": int(r_prog[:, f].argmax()),
            }
        )
    # top-activating frames per feature (top 100 by value)
    o = np.lexsort((-vals, feats))
    fs = feats[o]
    starts = np.searchsorted(fs, np.arange(NF))
    ends = np.searchsorted(fs, np.arange(NF), side="right")
    for f in range(NF):
        rr = rows[o[starts[f] : min(ends[f], starts[f] + 100)]]
        if len(rr) == 0:
            rows_out[f].update(
                top100_task="", top100_task_share=0.0, top100_prog_median=np.nan, top100_example=""
            )
            continue
        tc = Counter(task[rr].tolist()).most_common(1)[0]
        r0 = rr[0]
        rows_out[f].update(
            top100_task=codes_[tc[0]],
            top100_task_share=tc[1] / len(rr),
            top100_prog_median=float(np.median(TF["prog"][rr])),
            top100_example=f"ep{int(TF['episode'][r0])}/f{int(TF['frame'][r0])}",
        )
    # object / destination feature sets (PREREG A2 last paragraph)
    S = lp.read_sem()
    od = {}
    for key, nm in (("object", "object"), ("destination", "dest")):
        y = [lp.primary(S, c, key) for c in codes_]
        cnt = Counter(y)
        for val in cnt:
            if cnt[val] < 2 or val == "none":
                continue
            msk = np.array([yy == val for yy in y])
            ro = r_t[msk].mean(0)
            rn = r_t[~msk].mean(0)
            ok = (
                (ro >= 0.05)
                & (ro >= 3 * np.maximum(rn, 1e-9))
                & ((r_t[msk] >= 2 * np.maximum(rn, 1e-9)).sum(0) >= 2)
            )
            od[f"{nm}:{val}"] = np.where(ok)[0].tolist()
    # per-task feature sets F_T (PREREG A4)
    FT = {}
    for t in range(40):
        oth = (r_t.sum(0) - r_t[t]) / 39
        FT[codes_[t]] = np.where((r_t[t] >= 0.05) & (r_t[t] >= 5 * np.maximum(oth, 1e-9)))[0].tolist()
    np.savez(
        CACHE / f"lab_{lname(li)}.npz",
        r_t=r_t.astype(np.float32),
        rho_p=rho_p.astype(np.float32),
        r_prog=r_prog.astype(np.float32),
        phi=phi,
        r_sc=r_sc,
        r_su=r_su,
    )
    return {
        "layer": li,
        "rows": rows_out,
        "od": od,
        "FT": FT,
        "sc_names": sc_names,
        "tags": dict(Counter(tags)),
        "same_layout": [sc_names[g] for g in sorted(same_layout)],
    }


def stage_labels() -> None:
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(label_job, LAYERS))
    rows = [r for x in res for r in x["rows"]]
    keys = list(rows[0])
    for r in rows:
        for k_ in keys:
            if isinstance(r[k_], float):
                r[k_] = round(r[k_], 5)
    with open(OUT / "feature_labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    save_json(
        "labels",
        {
            lname(x["layer"]): {
                "tags": x["tags"],
                "od": x["od"],
                "FT": x["FT"],
                "sc_names": x["sc_names"],
                "same_layout": x["same_layout"],
                "sub": dict(Counter(r["sub"] for r in x["rows"] if r["tag"] == "task")),
                "phase_flag": int(sum(r["phase_flag"] for r in x["rows"] if r["tag"] != "rare")),
                "pref_phase_of_phase": dict(
                    Counter(r["pref_phase"] for r in x["rows"] if r["tag"] == "phase")
                ),
            }
            for x in res
        },
    )
    for x in res:
        print(lname(x["layer"]), x["tags"], flush=True)


_LABROWS: dict = {}


def lab_rows(li: int) -> list[dict]:
    if not _LABROWS:
        for r in csv.DictReader(open(OUT / "feature_labels.csv")):
            _LABROWS.setdefault(r["layer"], []).append(r)
    return _LABROWS[lname(li)]


def tag_arr(li: int) -> np.ndarray:
    return np.array([r["tag"] for r in lab_rows(li)])


# ============================================================================ (traj) A3
TRAJ_LAYERS = (1, 4, 8, 12, 16, 17)


def traj_job(li: int) -> dict:
    M = lp.meta()
    U = lp.u40(M)
    urows = np.concatenate([np.arange(m["off"], m["off"] + m["C"]) for m in U])
    Z = dense(li)
    mu = Z[urows].mean(0)
    _, _, vt = np.linalg.svd(Z[urows] - mu, full_matrices=False)
    comp = vt[:2]
    out = {"layer": li, "cases": {}}
    for case, conds in lp.cases().items():
        cc = {}
        for cname, eps in conds.items():
            trials = []
            for i in eps:
                m = M[i]
                X = Z[m["off"] : m["off"] + m["C"]]
                trials.append(
                    {
                        "pc": ((X - mu) @ comp.T).astype(np.float32),
                        "gc": m.get("grasp_call"),
                        "mean": X.astype(np.float16),
                    }
                )
            if trials:
                cc[cname] = trials
        # top-30 features by variance of condition-mean activation over (condition, call)
        cm = []
        for cname, trials in cc.items():
            n = min(len(t["mean"]) for t in trials)
            cm.append(np.mean([t["mean"][:n].astype(np.float32) for t in trials], 0))
        allc = np.concatenate(cm)
        top = np.argsort(-allc.var(0))[:30]
        out["cases"][case] = {
            "top": top.tolist(),
            "heat": {cname: m_[:, top].T.astype(np.float32) for cname, m_ in zip(cc, cm)},
            "pc": {cname: [{"pc": t["pc"], "gc": t["gc"]} for t in trials] for cname, trials in cc.items()},
        }
    return out


def stage_traj() -> None:
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(traj_job, LAYERS))
    pickle.dump(res, open(CACHE / "traj.pkl", "wb"))


# ============================================================================ (fallback) A4
FB_CALLS = 28


def fallback_job(li: int) -> dict:
    M = lp.meta()
    L = lp.labels()
    U = L["U"]
    labs = load_json("labels")[lname(li)]
    FT = {t: set(v) for t, v in labs["FT"].items()}
    v, ix = codes(li)
    fire = np.zeros((len(v), NF), bool)
    np.put_along_axis(fire, ix.astype(np.int64), v > 0, 1)
    by_task = defaultdict(list)
    for m in U:
        by_task[m["task"]].append(m)

    def ref_on(tasks, F, c):
        vals = [fire[m["off"] + c, F].mean() for t in tasks for m in by_task.get(t, []) if m["C"] > c]
        return float(np.mean(vals)) if vals else np.nan

    recs = []
    for p in L["pool"]:
        m = p["m"]
        I, E = p["instructed"], [e for e in p["executed"] if e in FT]
        if I not in FT or not E:
            continue
        FE = set().union(*[FT[e] for e in E])
        FI_only = sorted(FT[I] - FE)
        FE_only = sorted(FE - FT[I])
        rec = {
            "episode": m["episode"],
            "src": p["src"],
            "task": m["task"],
            "I": I,
            "E": E,
            "nFI": len(FI_only),
            "nFE": len(FE_only),
            "idx_I": [],
            "idx_E": [],
            "on_I": [],
            "on_E": [],
        }
        for c in range(min(FB_CALLS, m["C"])):
            iI = iE = oI = oE = np.nan
            if FI_only:
                oI = float(fire[m["off"] + c, FI_only].mean())
                a, b = ref_on([I], FI_only, c), ref_on(E, FI_only, c)
                if np.isfinite(a) and np.isfinite(b) and a - b >= 0.02:
                    iI = (oI - b) / (a - b)
            if FE_only:
                oE = float(fire[m["off"] + c, FE_only].mean())
                a, b = ref_on(E, FE_only, c), ref_on([I], FE_only, c)
                if np.isfinite(a) and np.isfinite(b) and a - b >= 0.02:
                    iE = (oE - b) / (a - b)
            rec["idx_I"].append(iI)
            rec["idx_E"].append(iE)
            rec["on_I"].append(oI)
            rec["on_E"].append(oE)
        recs.append(rec)
    # object / destination features
    od = {k_: v_ for k_, v_ in labs["od"].items()}
    odrec = []
    for p in L["pool"]:
        m = p["m"]
        for nm in ("object", "dest"):
            ins, exe = p[f"{nm}_instr"], p[f"{nm}_exec"]
            kI = f"{nm}:{ins}"
            kE = [f"{nm}:{e}" for e in exe if f"{nm}:{e}" in od]
            if kI not in od or ins in exe or not kE:
                continue
            FEo = set().union(*[set(od[k_]) for k_ in kE])
            FIo = sorted(set(od[kI]) - FEo)
            FEo = sorted(FEo - set(od[kI]))
            # reference: U40 trials whose primary value is ins / is one of exe
            S = lp.read_sem()
            ti = [
                t for t in by_task if lp.primary(S, t, "object" if nm == "object" else "destination") == ins
            ]
            te = [
                t for t in by_task if lp.primary(S, t, "object" if nm == "object" else "destination") in exe
            ]
            r = {
                "episode": m["episode"],
                "src": p["src"],
                "kind": nm,
                "ins": ins,
                "exe": exe,
                "nFI": len(FIo),
                "nFE": len(FEo),
                "idx_I": [],
                "idx_E": [],
            }
            for c in range(min(FB_CALLS, m["C"])):
                iI = iE = np.nan
                if FIo:
                    a, b = ref_on(ti, FIo, c), ref_on(te, FIo, c)
                    if np.isfinite(a) and np.isfinite(b) and a - b >= 0.02:
                        iI = (float(fire[m["off"] + c, FIo].mean()) - b) / (a - b)
                if FEo:
                    a, b = ref_on(te, FEo, c), ref_on(ti, FEo, c)
                    if np.isfinite(a) and np.isfinite(b) and a - b >= 0.02:
                        iE = (float(fire[m["off"] + c, FEo].mean()) - b) / (a - b)
                r["idx_I"].append(iI)
                r["idx_E"].append(iE)
            odrec.append(r)
    # same-observation pairs (T029 pairs.json)
    pairs = json.load(open(lp.PAIRS))
    bypath = {m["path"]: m for m in M}
    pp = [
        (bypath[q["cand"]], bypath[q["base"]], q)
        for q in pairs
        if q.get("valid") and q.get("rich_available") and q["cand"] in bypath and q["base"] in bypath
    ]
    Z0 = None
    pres = {"n_pairs": len(pp), "calls": {}}
    tags = tag_arr(li)
    from scipy.stats import wilcoxon

    for c in (0, 1, 2, 3):
        A = dense(li, np.array([a["off"] + c for a, b, _ in pp]))
        B = dense(li, np.array([b["off"] + c for a, b, _ in pp]))
        D = A - B
        act = ((A > 0).sum(0) >= 5) | ((B > 0).sum(0) >= 5)
        tested = np.where(act)[0]
        pv = np.ones(NF)
        for f in tested:
            d = D[:, f]
            if np.all(d == 0):
                continue
            pv[f] = wilcoxon(d, zero_method="wilcox").pvalue
        q = np.ones(NF)
        q[tested] = bh(pv[tested])
        sig = np.where(q < 0.05)[0]
        up = sig[D[:, sig].mean(0) > 0]
        dn = sig[D[:, sig].mean(0) < 0]

        # which trained-task feature sets do the cand-up / cand-down features belong to?
        def membership(fs, which):
            cnt = Counter()
            for a, b, qq in pp:
                if which == "base":
                    t = b["task"]
                else:
                    t = (a.get("exec_other") or [None])[0]
                if t in FT:
                    cnt["in"] += len(set(fs) & FT[t])
                    cnt["n"] += len(fs)
            return cnt["in"] / max(cnt["n"], 1)

        anyG = set().union(*[FT[t] for t in FT if t.startswith("G")])
        pres["calls"][c] = {
            "n_tested": int(len(tested)),
            "n_sig": int(len(sig)),
            "n_up": int(len(up)),
            "n_dn": int(len(dn)),
            "tags_up": dict(Counter(tags[up])),
            "tags_dn": dict(Counter(tags[dn])),
            "dn_in_base_FT": membership(dn, "base"),
            "up_in_exec_FT": membership(up, "exec"),
            "up_in_anyG": float(np.mean([f in anyG for f in up])) if len(up) else np.nan,
            "dn_in_anyG": float(np.mean([f in anyG for f in dn])) if len(dn) else np.nan,
            "sig_list": sig.tolist(),
            "mean_abs_diff": float(np.abs(D).sum(1).mean()),
            "frac_fire_changed": float(((A > 0) != (B > 0)).sum(1).mean() / K),
        }
    del Z0
    return {"layer": li, "recs": recs, "odrec": odrec, "pairs": pres}


def stage_fallback() -> None:
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(fallback_job, LAYERS))
    pickle.dump(res, open(CACHE / "fallback.pkl", "wb"))


# ============================================================================ (basket) A5
def basket_pairs() -> dict:
    M = lp.meta()
    key = lambda m: (m["task"], m["init"])  # noqa: E731
    ctlV = {key(m): m for m in M if m["set"] == "basket_far" and m["condition"] == "ctl4090"}
    ctlA = {key(m): m for m in M if m["set"] == "libero3" and m["task"].startswith("O")}
    out = {}
    for cond, ctl, st in (
        ("far20", ctlV, "basket_far"),
        ("far25", ctlV, "basket_far"),
        ("near", ctlA, "basket_a100"),
        ("opp", ctlA, "basket_a100"),
    ):
        out[cond] = [
            (ctl[key(m)], m) for m in M if m["set"] == st and m["condition"] == cond and key(m) in ctl
        ]
    return out


def basket_job(li: int) -> dict:
    from scipy.stats import wilcoxon

    P = basket_pairs()
    Z = dense(li)
    res = {"layer": li, "n": {k_: len(v_) for k_, v_ in P.items()}}
    tags = tag_arr(li)

    def carry_rows(a, b):
        ra = (
            [c for c, ph in enumerate(lp.phase_labels(a)) if ph == "carry"] if a.get("gc") is not None else []
        )
        rb = (
            [c for c, ph in enumerate(lp.phase_labels(b)) if ph == "carry"] if b.get("gc") is not None else []
        )
        return ra, rb

    for win in ("call0", "carry"):
        D, F = {}, {}
        for cond, pp in P.items():
            d, fd = [], []
            for a, b in pp:
                if win == "call0":
                    xa, xb = Z[a["off"]], Z[b["off"]]
                else:
                    ra, rb = carry_rows(a, b)
                    if not ra or not rb:
                        continue
                    xa = Z[a["off"] + np.array(ra)].mean(0)
                    xb = Z[b["off"] + np.array(rb)].mean(0)
                d.append(xb - xa)
                fd.append((xb > 0).astype(np.float32) - (xa > 0).astype(np.float32))
            D[cond], F[cond] = np.array(d), np.array(fd)
        d25 = D["far25"]
        act = (np.abs(d25) > 0).sum(0) >= 5
        tested = np.where(act)[0]
        pv = np.ones(NF)
        for f in tested:
            pv[f] = wilcoxon(d25[:, f]).pvalue
        q = np.ones(NF)
        q[tested] = bh(pv[tested])
        sig = np.where(q < 0.05)[0]
        m20, m25 = D["far20"].mean(0), D["far25"].mean(0)
        mn = D["near"].mean(0) if len(D["near"]) else np.zeros(NF)
        mo = D["opp"].mean(0) if len(D["opp"]) else np.zeros(NF)
        f20, f25 = F["far20"].mean(0), F["far25"].mean(0)
        lam = m20 / np.where(np.abs(m25) > 1e-9, m25, np.nan)
        cls = []
        for f in sig:
            lf = lam[f]
            if (lf > 0.95) or (abs(f25[f]) >= 0.5 and lf > 0.8):
                cls.append("on/off")
            elif 0.3 <= lf <= 0.95:
                cls.append("graded")
            else:
                cls.append("other")
        cls = np.array(cls)
        near_same = [(np.sign(mn[f]) == np.sign(m25[f])) and (abs(mn[f]) < abs(m25[f])) for f in sig]
        res[win] = {
            "n_tested": int(len(tested)),
            "n_sig": int(len(sig)),
            "class": dict(Counter(cls.tolist())),
            "lambda_median": float(np.nanmedian(lam[sig])) if len(sig) else np.nan,
            "tags": dict(Counter(tags[sig])),
            "near_same_sign_smaller": float(np.mean(near_same)) if len(sig) else np.nan,
            "opp_same_sign": float(np.mean([np.sign(mo[f]) == np.sign(m25[f]) for f in sig]))
            if len(sig)
            else np.nan,
            "sig": sig.tolist(),
            "cls": cls.tolist(),
            "m20": m20[sig],
            "m25": m25[sig],
            "mnear": mn[sig],
            "mopp": mo[sig],
            "f20": f20[sig],
            "f25": f25[sig],
            "dist_by_cond": {c: float(np.linalg.norm(D[c].mean(0))) if len(D[c]) else np.nan for c in D},
            "nfire_changed": {c: float(np.abs(F[c]).sum(1).mean()) if len(F[c]) else np.nan for c in F},
        }
    return res


def stage_basket() -> None:
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(basket_job, LAYERS))
    pickle.dump(res, open(CACHE / "basket.pkl", "wb"))


# ============================================================================ (kauf) A6
def kauf_job(li: int) -> dict:
    sae = NpSAE(sae_dir(li))
    reader = li + 1 if li < N_L - 1 else li
    P, _ = row_space(lp.readin(reader))
    d = sae.W_dec.astype(np.float64)  # (2048, NF)
    d = d / np.linalg.norm(d, axis=0, keepdims=True)
    pfv = ((P.T @ d) ** 2).sum(0)
    tags = tag_arr(li)
    rows = lab_rows(li)
    sub = np.array([r["sub"] for r in rows])
    by = {t: pfv[tags == t] for t in set(tags)}
    by["instruction"] = pfv[sub == "instruction"]
    return {
        "layer": li,
        "reader": reader if li < N_L - 1 else None,
        "chance": P.shape[1] / 2048,
        "pf": pfv,
        "median_by_tag": {t: float(np.median(v)) if len(v) else np.nan for t, v in by.items()},
        "n_by_tag": {t: int(len(v)) for t, v in by.items()},
    }


def stage_kauf() -> None:
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(kauf_job, LAYERS))
    pickle.dump(res, open(CACHE / "kauf.pkl", "wb"))


# ============================================================================ (dec) A7
DEC_CALLS = lp.CALLS


def dec_job(job):
    rep, li = job
    t0 = time.time()
    L = lp.labels()
    eps = L["U"] + [p["m"] for p in L["pool"]]
    rows = np.concatenate([np.arange(m["off"], m["off"] + m["C"]) for m in eps])
    X = dense(li, rows) if rep == "sae" else pooled(li)[rows]
    pos = {}
    k_ = 0
    for m in eps:
        pos[m["i"]] = k_
        k_ += m["C"]
    off = np.array([pos[m["i"]] for m in L["U"]])
    pool = L["pool"]
    out = {"rep": rep, "layer": li, "acc": defaultdict(list), "xgen_exec": [], "xgen_instr": []}
    for c in DEC_CALLS:
        Xc = X[off + c]
        out["acc"]["task40"].append(float(np.mean(lp.cv_pred(Xc, L["task"], L["init"]) == L["task"])))
        w = []
        for su in "SOGL":
            msk = L["suite"] == su
            w.append(float(np.mean(lp.cv_pred(Xc[msk], L["task"][msk], L["init"][msk]) == L["task"][msk])))
        out["acc"]["within"].append(float(np.mean(w)))
        out["acc"]["within_G"].append(w[2])
        ok = [i for i, p in enumerate(pool) if p["m"]["C"] > c]
        Xp = X[np.array([pos[pool[i]["m"]["i"]] + c for i in ok])]
        pt = lp.fit_predict(Xc, L["task"], [Xp])[0]
        out["xgen_exec"].append(float(np.mean([pt[j] in pool[i]["executed"] for j, i in enumerate(ok)])))
        out["xgen_instr"].append(float(np.mean([pt[j] == pool[i]["instructed"] for j, i in enumerate(ok)])))
    rr, yy, gg = [], [], []
    for m, lab in zip(L["U"], L["phase"]):
        for c in range(0, m["C"], 2):
            if lab[c] is not None:
                rr.append(pos[m["i"]] + c)
                yy.append(lab[c])
                gg.append(m["init"])
    yy = np.array(yy)
    pred = lp.cv_pred(X[np.array(rr)], yy, np.array(gg))
    out["phase_bacc"] = lp.bacc(yy, pred)
    out["acc"] = dict(out["acc"])
    out["sec"] = time.time() - t0
    return out


def stage_dec() -> None:
    jobs = [(r, li) for li in LAYERS for r in ("sae", "x")]
    part = CACHE / "dec_parts.pkl"
    done = pickle.load(open(part, "rb")) if part.exists() else {}
    todo = [j for j in jobs if j not in done]
    from concurrent.futures import as_completed

    with ProcessPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(dec_job, j): j for j in todo}
        for n, f in enumerate(as_completed(futs)):
            done[futs[f]] = f.result()
            pickle.dump(done, open(part, "wb"))
            print(f"  dec {n + 1}/{len(todo)} {futs[f]} {done[futs[f]]['sec']:.0f}s", flush=True)


# ============================================================================ (expert) A7 optional
EXP_LAYERS = (2, 5, 8, 11, 14, 17)


def expert_job(job) -> dict:
    li, el = job
    M = lp.meta()
    U = lp.u40(M)
    urows = np.concatenate([np.arange(m["off"], m["off"] + m["C"]) for m in U])
    sae = NpSAE(EXP_SAE / f"L{el:02d}_s9")
    xe = np.asarray(lp._mm("exp_mean")[el][urows, 2], np.float32)  # step 9
    v, ix = sae.encode(xe)
    E = np.zeros((len(xe), sae.W_dec.shape[1]), np.float32)
    np.put_along_axis(E, ix, v, 1)
    V = dense(li, urows)
    va = (V > 0).mean(0) >= 0.01
    ea = (E > 0).mean(0) >= 0.01
    Vs, Es = V[:, va], E[:, ea]
    Vs = (Vs - Vs.mean(0)) / (Vs.std(0) + 1e-9)
    Es = (Es - Es.mean(0)) / (Es.std(0) + 1e-9)
    C = Vs.T @ Es / len(Vs)
    best = C.max(1)
    tags = tag_arr(li)[va]
    return {
        "layer": li,
        "exp_layer": el,
        "n_vlm": int(va.sum()),
        "n_exp": int(ea.sum()),
        "best_r_median": float(np.median(best)),
        "best_r_p90": float(np.quantile(best, 0.9)),
        "best_r_by_tag": {t: float(np.median(best[tags == t])) for t in set(tags)},
        "frac_best_r_gt_0.5": float((best > 0.5).mean()),
        "top_pairs": [
            (int(np.where(va)[0][i]), int(np.where(ea)[0][j]), float(C[i, j]))
            for i, j in zip(*np.unravel_index(np.argsort(-C, axis=None)[:15], C.shape))
        ],
    }


def stage_expert() -> None:
    if not EXP_SAE.exists():
        print("T031 expert SAE not found; skip")
        return
    jobs = [(li, el) for li in (1, 4, 8, 12, 16) for el in (2, 8, 14, 17)]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(expert_job, jobs))
    pickle.dump(res, open(CACHE / "expert.pkl", "wb"))


# ============================================================================ main
STAGES = {
    "trainlab": stage_trainlab,
    "encode": stage_encode,
    "labels": stage_labels,
    "traj": stage_traj,
    "fallback": stage_fallback,
    "basket": stage_basket,
    "kauf": stage_kauf,
    "dec": stage_dec,
    "expert": stage_expert,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="+")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    names = list(STAGES) if a.stage == ["all"] else a.stage
    for n in names:
        t = time.time()
        print(f"== {n}", flush=True)
        if n in ("figs", "numbers"):
            import sae_vlm_traj_figs as F

            getattr(F, f"stage_{n}")()
        else:
            STAGES[n]()
        print(f"== {n} done {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
