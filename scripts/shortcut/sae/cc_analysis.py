#!/usr/bin/env python
"""T034 crosscoder analysis (CPU). PREREG: cc_PREREG.md §6.

  python cc_analysis.py profiles <name>   layer-profile categories, depth centroid, joint VLM share
  python cc_analysis.py labels   <name>   T032 A2 tags from the training-data codes (fire = code > 0)
  python cc_analysis.py t028     <name>   encode the 1 560 T028 episodes (all calls)
  python cc_analysis.py fallback <name>   fallback trials: instructed vs executed task features, per call and per layer
  python cc_analysis.py all      <name>

<name> = a run under /data0/.../sae/crosscoder/. Results: /outputs/VLA-shortcut-exps/sae/crosscoder/analysis/<name>/.
Caches: /data0/.../cache/t034_<name>/ (T028 codes).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "analysis"))
import torch  # noqa: E402
from cc_data import CACHE as DCACHE  # noqa: E402
from cc_data import cache_dir  # noqa: E402
from crosscoder import CrossCoder  # noqa: E402

WEIGHTS = Path("/data0/woosung/datasets/VLA-shortcut-exps/sae/crosscoder")
OUT = Path("/outputs/VLA-shortcut-exps/sae/crosscoder/analysis")
T032 = Path("/outputs/VLA-shortcut-exps/sae/vlm/analysis/_cache")
PHASE4 = ("reach", "grasp", "carry", "release")
CATS = ("shallow-only", "deep-only", "middle-only", "carried", "born-middle", "shallow-to-middle", "other")


def out_dir(name: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(name: str, key: str, obj) -> None:
    def default(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return str(v)

    (out_dir(name) / f"{key}.json").write_text(json.dumps(obj, indent=1, default=default) + "\n")


def load(name: str, key: str):
    return json.loads((OUT / name / f"{key}.json").read_text())


class Run:
    def __init__(self, name: str):
        self.name = name
        self.w = WEIGHTS / name
        self.cfg = json.loads((self.w / "config.json").read_text())
        self.metrics = json.loads((self.w / "metrics.json").read_text())
        self.layers: list[str] = self.cfg["layers"]
        self.prof = np.load(self.w / "profile.npy")  # (n, L)
        self.fire = np.load(self.w / "fire_count.npy")
        self.live = self.fire > 0
        self.n = len(self.fire)
        self.k = int(self.cfg["k"])
        self.data = self.cfg["data"]

    def codes(self) -> tuple[np.ndarray, np.ndarray]:
        return np.load(self.w / "codes_idx.npy"), np.load(self.w / "codes_val.npy")

    def train_meta(self) -> pd.DataFrame:
        return pd.read_parquet(cache_dir("expert" if self.data != "vlm" else "vlm") / "meta.parquet")

    def model(self) -> CrossCoder:
        return CrossCoder.load(self.w)

    def block_ids(self, fam: str) -> np.ndarray:
        return np.array([i for i, nm in enumerate(self.layers) if nm.startswith(fam)])


# ============================================================================ profiles
def categorize(p: np.ndarray) -> np.ndarray:
    """p (n, 18) profiles (rows sum to 1) → category index per PREREG §6 (thirds 0–5, 6–11, 12–17)."""
    L = p.shape[1]
    assert L == 18, L
    mS, mM, mD = p[:, :6].sum(1), p[:, 6:12].sum(1), p[:, 12:].sum(1)
    cat = np.full(len(p), CATS.index("other"))
    rules = [
        ("shallow-to-middle", mD < 0.15),
        ("born-middle", (mS < 0.15) & (mM >= 0.3)),
        ("carried", np.minimum(np.minimum(mS, mM), mD) >= 0.15),
        ("middle-only", mM >= 0.7),
        ("deep-only", mD >= 0.7),
        ("shallow-only", mS >= 0.7),
    ]
    for nm, m in rules:  # reverse priority: later rules overwrite → first PREREG rule wins
        cat[m] = CATS.index(nm)
    return cat


def centroid(p: np.ndarray) -> np.ndarray:
    L = p.shape[1]
    return (p * np.arange(L)).sum(1) / p.sum(1).clip(1e-12) / (L - 1)


def layer_scales(R: Run) -> tuple[np.ndarray, np.ndarray]:
    """Training rows: s² (N, L) = ‖x_l − b_pre_l − μ_l‖² per row, Var_l = mean ‖x_l − mean_l‖² (raw)."""
    from safetensors.numpy import load_file

    b_pre = load_file(R.w / "crosscoder.safetensors")["b_pre"].astype(np.float32)
    srcs = []
    for fam, nm in (("pg", "vlm"), ("ae", "expert")):
        if any(l.startswith(fam) for l in R.layers):
            d = cache_dir(nm)
            info = json.loads((d / "info.json").read_text())
            x = np.load(d / "x.npy", mmap_mode="r")
            srcs += [(x, info["layers"].index(l)) for l in R.layers if l.startswith(fam)]
    o = np.cumsum([0] + list(R.cfg["dims"]))
    N = srcs[0][0].shape[0]
    s2 = np.zeros((N, len(R.layers)), np.float32)
    var = np.zeros(len(R.layers))
    for li, (x, j) in enumerate(srcs):
        acc, acc2 = 0.0, 0.0
        mean = None
        for pas in (0, 1):
            for s in range(0, N, 16384):
                a = np.asarray(x[s : s + 16384, j], np.float32)
                if pas == 0:
                    c = a - b_pre[o[li] : o[li + 1]]
                    c = c - c.mean(1, keepdims=True)
                    s2[s : s + 16384, li] = (c**2).sum(1)
                    acc = acc + a.astype(np.float64).sum(0)
                else:
                    acc2 += float(((a - mean) ** 2).sum())
            if pas == 0:
                mean = (acc / N).astype(np.float32)
        var[li] = acc2 / N
    return s2, var


def stage_profiles(name: str) -> dict:
    R = Run(name)
    p = R.prof
    res: dict = {"n_latents": R.n, "n_live": int(R.live.sum()), "never_fire": float(1 - R.live.mean())}
    feat = pd.DataFrame({"feature": np.arange(R.n), "live": R.live, "fire_count": R.fire})
    if R.data in ("vlm", "expert"):
        cat = categorize(p)
        feat["cat"] = [CATS[c] for c in cat]
        feat["centroid"] = centroid(p)
        cnt = Counter(feat.loc[feat.live, "cat"])
        res["categories"] = {c: int(cnt.get(c, 0)) for c in CATS}
        res["category_share"] = {c: cnt.get(c, 0) / max(R.live.sum(), 1) for c in CATS}
        res["mean_profile_live"] = p[R.live].mean(0).tolist()
    else:
        vb, eb = R.block_ids("pg"), R.block_ids("ae")
        v = p[:, vb].sum(1)
        feat["vlm_share"] = v
        kind = np.where(v >= 0.9, "vlm-only", np.where(v <= 0.1, "expert-only", "shared"))
        feat["kind"] = kind
        pv = p[:, vb] / p[:, vb].sum(1, keepdims=True).clip(1e-12)
        pe = p[:, eb] / p[:, eb].sum(1, keepdims=True).clip(1e-12)
        feat["cat_vlm"] = [CATS[c] for c in categorize(pv)]
        feat["cat_expert"] = [CATS[c] for c in categorize(pe)]
        feat["centroid_vlm"] = centroid(pv)
        feat["centroid_expert"] = centroid(pe)
        feat["centroid"] = centroid(p)  # over the 36 blocks (VLM 0–17 then expert 0–17)
        lv = feat[feat.live]
        cnt = Counter(lv["kind"])
        res["kinds"] = {k: int(cnt.get(k, 0)) for k in ("vlm-only", "shared", "expert-only")}
        res["kind_share"] = {k: cnt.get(k, 0) / max(len(lv), 1) for k in res["kinds"]}
        res["vlm_share_hist"] = np.histogram(lv["vlm_share"], bins=20, range=(0, 1))[0].tolist()
        for k, col in (("vlm-only", "cat_vlm"), ("expert-only", "cat_expert")):
            c2 = Counter(lv.loc[lv.kind == k, col])
            res[f"categories_{k}"] = {c: int(c2.get(c, 0)) for c in CATS}
        sh = lv[lv.kind == "shared"]
        res["categories_shared_vlm_part"] = {c: int(Counter(sh["cat_vlm"]).get(c, 0)) for c in CATS}
        res["categories_shared_expert_part"] = {c: int(Counter(sh["cat_expert"]).get(c, 0)) for c in CATS}
        # where does a shared feature leave the VLM / enter the expert (mass-weighted layer)
        res["shared_vlm_centroid_median"] = float(sh["centroid_vlm"].median()) if len(sh) else None
        res["shared_expert_centroid_median"] = float(sh["centroid_expert"].median()) if len(sh) else None
        res["mean_profile_live"] = p[R.live].mean(0).tolist()
    # ---- variance-aware view (lead request, added after the expert results; reported next to the decoder-norm view)
    # q_il ∝ ‖d_i^l‖² / C_l: feature i's expected explained variance in layer l is E[z_i²]·‖d_i^l‖² / C_l (diagonal
    # term; E[z_i²] is common to all layers of a feature), so q is the feature's share of explained variance by layer.
    # C_l ≈ 1 in every layer (mean of the unit x̃ ≈ 0), so ‖d‖²/C_l equals the decoder view. The variance-aware view is
    # therefore taken in the raw space: a_il = Σ_rows z_i² · s_l² · ‖d_i^l‖² / (N · Var_l), s_l = per-sample norm of
    # (x_l − b_pre_l − μ_l), Var_l = across-sample variance of raw x_l: feature i's (diagonal) share of layer l's
    # variance. q_il = a_il / Σ_l a_il.
    C = np.asarray(R.cfg["c_mse"], np.float64)
    res["c_mse"] = C.tolist()
    ix, vv = R.codes()
    vv = vv.astype(np.float32)
    s2, var = layer_scales(R)
    flat = ix.reshape(-1).astype(np.int64)
    ev = np.zeros_like(p, dtype=np.float64)
    for li in range(p.shape[1]):
        w = (vv**2 * s2[:, li : li + 1]).reshape(-1)
        ev[:, li] = np.bincount(flat, weights=w, minlength=R.n) / len(ix) * p[:, li] / var[li]
    q = ev / ev.sum(1, keepdims=True).clip(1e-30)
    res["layer_var_raw"] = var.tolist()
    res["diag_ev_sum_per_layer"] = ev[R.live].sum(0).tolist()  # ≈ VE without cross terms (sanity)
    np.save(out_dir(name) / "ev_share.npy", ev.astype(np.float32))
    if R.data in ("vlm", "expert"):
        cv = categorize(q)
        feat["cat_var"] = [CATS[c] for c in cv]
        feat["centroid_var"] = centroid(q)
        cnt = Counter(feat.loc[feat.live, "cat_var"])
        res["categories_var"] = {c: int(cnt.get(c, 0)) for c in CATS}
        res["category_share_var"] = {c: cnt.get(c, 0) / max(R.live.sum(), 1) for c in CATS}
        # per layer: share of that layer's (diagonal) explained variance carried by each decoder-norm category
        cat_arr = feat["cat"].to_numpy()
        tot = ev[R.live].sum(0).clip(1e-12)
        res["layer_ev_share_by_cat"] = {c: (ev[R.live & (cat_arr == c)].sum(0) / tot).tolist() for c in CATS}
    else:
        qv = q[:, vb].sum(1)
        feat["vlm_share_var"] = qv
        kv = np.where(qv >= 0.9, "vlm-only", np.where(qv <= 0.1, "expert-only", "shared"))
        feat["kind_var"] = kv
        lv = feat[feat.live]
        cnt = Counter(lv["kind_var"])
        res["kinds_var"] = {k: int(cnt.get(k, 0)) for k in ("vlm-only", "shared", "expert-only")}
        qvv = q[:, vb] / q[:, vb].sum(1, keepdims=True).clip(1e-12)
        qee = q[:, eb] / q[:, eb].sum(1, keepdims=True).clip(1e-12)
        feat["cat_vlm_var"] = [CATS[c] for c in categorize(qvv)]
        feat["cat_expert_var"] = [CATS[c] for c in categorize(qee)]
        lv = feat[feat.live]
        for k, col in (("vlm-only", "cat_vlm_var"), ("expert-only", "cat_expert_var")):
            c2 = Counter(lv.loc[lv.kind_var == k, col])
            res[f"categories_var_{k}"] = {c: int(c2.get(c, 0)) for c in CATS}
        kind_arr = feat["kind"].to_numpy()
        tot = ev[R.live].sum(0).clip(1e-12)
        res["layer_ev_share_by_kind"] = {k: (ev[R.live & (kind_arr == k)].sum(0) / tot).tolist()
                                         for k in ("vlm-only", "shared", "expert-only")}
    # ---- never-fire on train and held-out separately (lead request)
    hold = np.load(R.w / "heldout.npy")
    for split, msk in (("train", ~hold), ("heldout", hold)):
        f = np.bincount(ix[msk][vv[msk] > 0].astype(np.int64), minlength=R.n)
        res[f"never_fire_{split}"] = float((f == 0).mean())
    feat.to_parquet(out_dir(name) / "features.parquet", index=False)
    save(name, "profiles", res)
    print(json.dumps({k: v for k, v in res.items() if "hist" not in k and "mean_profile" not in k}, indent=1))
    return res


# ============================================================================ labels (T032 A2 rules)
def train_frames_for(meta: pd.DataFrame) -> dict:
    """T032 training-frame table (all 273 465 frames, sorted by dataset index) restricted to our rows."""
    TF = np.load(T032 / "train_frames.npz")
    info = json.loads((T032 / "trainlab.json").read_text())
    idx = meta["index"].to_numpy()
    assert len(TF["task"]) == 273465
    ep, fr = TF["episode"][idx], TF["frame"][idx]
    assert (ep == meta["episode_index"].to_numpy()).all() and (fr == meta["frame_index"].to_numpy()).all()
    return {k: TF[k][idx] for k in ("task", "phase", "prog", "episode", "frame")} | {
        "codes": info["codes"],
        "scene": info["scene"],
    }


def label_features(ix: np.ndarray, v: np.ndarray, TF: dict, NF: int) -> tuple[pd.DataFrame, dict]:
    """sae_vlm_traj.label_job tag rules on codes (N, k) for NF features. Returns per-feature table + FT sets."""
    codes_, scene = TF["codes"], TF["scene"]
    N = len(v)
    on = v > 0
    rows = np.broadcast_to(np.arange(N)[:, None], ix.shape)[on]
    feats = ix[on].astype(np.int64)

    def grate(g: np.ndarray, ng: int) -> np.ndarray:
        cnt = np.zeros((ng, NF), np.float64)
        sel = g[rows] >= 0
        np.add.at(cnt, (g[rows][sel], feats[sel]), 1.0)
        n = np.bincount(g[g >= 0], minlength=ng).astype(np.float64)
        return cnt / np.maximum(n, 1)[:, None]

    task = TF["task"].astype(np.int64)
    r_t = grate(task, 40)
    rho_p = grate(TF["phase"].astype(np.int64), 4)
    phi = np.bincount(feats, minlength=NF) / N
    sc_names = sorted({scene[c] for c in codes_})
    sc_of = np.array([sc_names.index(scene[c]) for c in codes_])
    su_names = ["S", "O", "G", "L"]
    su_of = np.array([su_names.index(c[0]) for c in codes_])
    r_sc = np.stack([r_t[sc_of == g].mean(0) for g in range(len(sc_names))])
    r_su = np.stack([r_t[su_of == g].mean(0) for g in range(4)])
    sc_size = np.bincount(sc_of, minlength=len(sc_names))
    same_layout = {g for g, nm in enumerate(sc_names) if sc_size[g] >= 2 and nm not in ("SPATIAL", "OBJECT")}
    out = []
    for f in range(NF):
        rt = r_t[:, f]
        tag, sub = "mixed", ""
        ts = int(rt.argmax())
        ratio_t = rt[ts] / max(np.delete(rt, ts).mean(), 1e-9)
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
        out.append({"feature": f, "tag": tag, "sub": sub, "phase_flag": int(phase_flag),
                    "pref_phase": PHASE4[int(rp.argmax())], "fire_rate": phi[f], "top_task": codes_[ts],
                    "top_task_rate": rt[ts], "task_ratio": ratio_t, "top_scene": sc_names[g]})
    FT = {}
    for t in range(40):
        oth = (r_t.sum(0) - r_t[t]) / 39
        FT[codes_[t]] = np.where((r_t[t] >= 0.05) & (r_t[t] >= 5 * np.maximum(oth, 1e-9)))[0].tolist()
    # T031-style selectivity (task: P(fire|best task) ≥ 0.5 and ≤ 0.05 elsewhere; phase: ≥ 0.5 and ≥ 3 × mean other)
    srt = np.sort(r_t, 0)
    task_sel = (srt[-1] >= 0.5) & (srt[-2] <= 0.05)
    pbest = rho_p.max(0)
    pother = (rho_p.sum(0) - pbest) / 3
    phase_sel = (pbest >= 0.5) & (pbest >= 3 * pother)
    df = pd.DataFrame(out)
    df["t031_task_sel"] = task_sel
    df["t031_phase_sel"] = phase_sel
    return df, {"FT": FT, "same_layout": [sc_names[g] for g in sorted(same_layout)]}


def stage_labels(name: str) -> dict:
    R = Run(name)
    ix, v = R.codes()
    TF = train_frames_for(R.train_meta())
    t0 = time.time()
    df, extra = label_features(ix, v.astype(np.float32), TF, R.n)
    feat = pd.read_parquet(OUT / name / "features.parquet")
    feat = feat.merge(df, on="feature")
    feat.to_parquet(out_dir(name) / "features.parquet", index=False)
    lv = feat[feat.live]
    res = {"tags": dict(Counter(lv["tag"])), "task_sub": dict(Counter(lv.loc[lv.tag == "task", "sub"])),
           "t031_task_sel": int(lv["t031_task_sel"].sum()), "t031_phase_sel": int(lv["t031_phase_sel"].sum()),
           "phase_flag": int(lv["phase_flag"].sum()), "FT": extra["FT"], "same_layout": extra["same_layout"],
           "seconds": round(time.time() - t0, 1)}
    # depth of instruction vs scene vs task features (PREREG §6 question)
    from scipy.stats import mannwhitneyu

    def grp(d):
        return {"instruction": d[(d.tag == "task") & (d["sub"] == "instruction")],
                "task-shared-scene": d[(d.tag == "task") & (d["sub"] == "task-shared-scene")],
                "task-confounded": d[(d.tag == "task") & (d["sub"] == "task-confounded")],
                "scene": d[d.tag == "scene"], "suite": d[d.tag == "suite"], "phase": d[d.tag == "phase"],
                "always-on": d[d.tag == "always-on"], "mixed": d[d.tag == "mixed"]}

    cols = ["centroid"] if R.data != "joint" else ["centroid_vlm", "centroid_expert", "vlm_share"]
    depth = {}
    for col in cols:
        G = grp(lv)
        depth[col] = {k: {"n": len(d), "median": float(d[col].median()) if len(d) else None,
                          "q25": float(d[col].quantile(0.25)) if len(d) else None,
                          "q75": float(d[col].quantile(0.75)) if len(d) else None} for k, d in G.items()}
        tests = {}
        for a, b in (("instruction", "scene"), ("instruction", "task-confounded"), ("scene", "task-confounded"),
                     ("instruction", "always-on")):
            if len(G[a]) >= 3 and len(G[b]) >= 3:
                tests[f"{a}_vs_{b}"] = float(mannwhitneyu(G[a][col], G[b][col]).pvalue)
        depth[col]["mwu_p"] = tests
        if "cat" in lv:
            depth["cat_by_tag"] = {k: dict(Counter(d["cat"])) for k, d in G.items()}
    if R.data == "joint":
        depth["kind_by_tag"] = {k: dict(Counter(d["kind"])) for k, d in grp(lv).items()}
    res["depth"] = depth
    save(name, "labels", res)
    print(json.dumps({k: v for k, v in res.items() if k != "FT"}, indent=1, default=str)[:3000])
    return res


# ============================================================================ T028 encode
def t028_inputs(R: Run) -> np.ndarray:
    """(51 100, D) fp32 in the crosscoder's block order: pg pooled_all as T032 (`pooled`), ae = exp_mean step 9."""
    import layer_pop as lp
    import sae_vlm_traj as sv

    n_tot = lp.meta()[-1]["off"] + lp.meta()[-1]["C"]
    X = np.empty((n_tot, sum(R.cfg["dims"])), np.float32)
    col = 0
    ntok = sv.seg_ntok()
    em = lp._mm("exp_mean")  # (18, N, 3, 1024), steps (0, 4, 9)
    for nm, d in zip(R.layers, R.cfg["dims"], strict=True):
        if nm.startswith("pg"):
            X[:, col : col + d] = sv.pooled(int(nm[2:]), ntok)
        elif nm.startswith("ae"):
            X[:, col : col + d] = np.asarray(em[int(nm[2:]), :, 2], np.float32)
        else:
            raise SystemExit(f"{nm} not in T028")
        col += d
        print(f"[t028] {nm}", flush=True)
    return X


def stage_t028(name: str) -> None:
    R = Run(name)
    cdir = DCACHE / f"t034_{name}"
    cdir.mkdir(parents=True, exist_ok=True)
    if (cdir / "t028_codes.npz").exists():
        print("exists")
        return
    torch.set_num_threads(32)
    X = t028_inputs(R)
    cc = R.model().eval()
    ii, vv = [], []
    sq = np.zeros(len(R.layers))
    var = np.zeros(len(R.layers))
    mean = X.mean(0)
    o = cc.cfg.offsets
    with torch.no_grad():
        for s in range(0, len(X), 4096):
            xb = torch.from_numpy(X[s : s + 4096])
            xt, mu, nrm = cc.normalize(xb)
            v, i = cc.topk(cc.pre_acts(xt))
            xh = cc.denormalize(cc.decode_sparse(v, i), mu, nrm).numpy()
            for li in range(len(R.layers)):
                sl = slice(o[li], o[li + 1])
                sq[li] += ((X[s : s + 4096, sl] - xh[:, sl]) ** 2).sum()
                var[li] += ((X[s : s + 4096, sl] - mean[sl]) ** 2).sum()
            ii.append(i.numpy().astype(np.int16))
            vv.append(v.numpy().astype(np.float16))
    np.savez(cdir / "t028_codes.npz", idx=np.concatenate(ii), val=np.concatenate(vv))
    ve = 1 - sq / var
    save(name, "t028_quality", {"layers": R.layers, "ve": ve.tolist(), "ve_mean": float(ve.mean())})
    print("T028 VE", np.round(ve, 3).tolist())


# ============================================================================ fallback
FB_CALLS = 28


def stage_fallback(name: str) -> dict:
    import layer_pop as lp

    R = Run(name)
    lab = load(name, "labels")
    FT = {t: set(v) for t, v in lab["FT"].items()}
    z = np.load(DCACHE / f"t034_{name}" / "t028_codes.npz")
    ix, vv = z["idx"].astype(np.int64), z["val"].astype(np.float32)
    n_tot = len(ix)
    fire = np.zeros((n_tot, R.n), bool)
    np.put_along_axis(fire, ix, vv > 0, 1)
    amp = np.sqrt(R.prof)  # (n, L) ‖d_i^l‖
    L = lp.labels()
    by_task = defaultdict(list)
    for m in L["U"]:
        by_task[m["task"]].append(m)
    feat = pd.read_parquet(OUT / name / "features.parquet")
    shared = feat["kind"].to_numpy() == "shared" if "kind" in feat else None

    def ref_on(tasks, F, c):
        vals = [fire[m["off"] + c, F].mean() for t in tasks for m in by_task.get(t, []) if m["C"] > c]
        return float(np.mean(vals)) if vals else np.nan

    def contrib(row: int, F: list[int]) -> np.ndarray:
        """per-layer contribution Σ_{i∈F active} z_i·‖d_i^l‖ at one call."""
        sel = np.isin(ix[row], F) & (vv[row] > 0)
        return (vv[row][sel][:, None] * amp[ix[row][sel]]).sum(0)

    recs = []
    Lb = len(R.layers)
    agg = {k: np.zeros((FB_CALLS, Lb)) for k in ("I_fb", "E_fb", "I_ref", "E_ref")}
    aggn = {k: np.zeros(FB_CALLS) for k in agg}
    for p in L["pool"]:
        m = p["m"]
        I, E = p["instructed"], [e for e in p["executed"] if e in FT]
        if I not in FT or not E:
            continue
        FE = set().union(*[FT[e] for e in E])
        FI_only, FE_only = sorted(FT[I] - FE), sorted(FE - FT[I])
        rec = {"episode": m["episode"], "src": p["src"], "I": I, "E": E, "nFI": len(FI_only), "nFE": len(FE_only),
               "idx_I": [], "idx_E": [], "on_I": [], "on_E": []}
        if shared is not None:
            rec["nFI_shared"] = int(shared[FI_only].sum()) if FI_only else 0
            rec["nFE_shared"] = int(shared[FE_only].sum()) if FE_only else 0
            rec["on_I_shared"], rec["on_E_shared"] = [], []
        for c in range(min(FB_CALLS, m["C"])):
            row = m["off"] + c
            iI = iE = oI = oE = np.nan
            if FI_only:
                oI = float(fire[row, FI_only].mean())
                a, b = ref_on([I], FI_only, c), ref_on(E, FI_only, c)
                if np.isfinite(a) and np.isfinite(b) and a - b >= 0.02:
                    iI = (oI - b) / (a - b)
                agg["I_fb"][c] += contrib(row, FI_only)
                aggn["I_fb"][c] += 1
                for t in [I]:
                    for mm in by_task.get(t, []):
                        if mm["C"] > c:
                            agg["I_ref"][c] += contrib(mm["off"] + c, FI_only) / len(by_task[t])
                    aggn["I_ref"][c] += 1
            if FE_only:
                oE = float(fire[row, FE_only].mean())
                a, b = ref_on(E, FE_only, c), ref_on([I], FE_only, c)
                if np.isfinite(a) and np.isfinite(b) and a - b >= 0.02:
                    iE = (oE - b) / (a - b)
                agg["E_fb"][c] += contrib(row, FE_only)
                aggn["E_fb"][c] += 1
                n_e = sum(len(by_task.get(t, [])) for t in E)
                for t in E:
                    for mm in by_task.get(t, []):
                        if mm["C"] > c:
                            agg["E_ref"][c] += contrib(mm["off"] + c, FE_only) / max(n_e, 1)
                aggn["E_ref"][c] += 1
            rec["idx_I"].append(iI)
            rec["idx_E"].append(iE)
            rec["on_I"].append(oI)
            rec["on_E"].append(oE)
            if shared is not None:
                fs = [f for f in FI_only if shared[f]]
                fe = [f for f in FE_only if shared[f]]
                rec["on_I_shared"].append(float(fire[row, fs].mean()) if fs else np.nan)
                rec["on_E_shared"].append(float(fire[row, fe].mean()) if fe else np.nan)
        recs.append(rec)

    def med(key, calls=range(FB_CALLS)):
        out = []
        for c in calls:
            vals = [r[key][c] for r in recs if len(r[key]) > c and np.isfinite(r[key][c])]
            out.append({"call": c, "n": len(vals), "median": float(np.median(vals)) if vals else None,
                        "mean": float(np.mean(vals)) if vals else None})
        return out

    contrib_maps = {k: (agg[k] / np.maximum(aggn[k], 1)[:, None]).tolist() for k in agg}
    res = {"n_trials": len(recs), "n_cand": sum(r["src"] == "cand" for r in recs),
           "n_l90": sum(r["src"] == "l90" for r in recs),
           "nFI_median": float(np.median([r["nFI"] for r in recs])) if recs else None,
           "nFE_median": float(np.median([r["nFE"] for r in recs])) if recs else None,
           "idx_I": med("idx_I"), "idx_E": med("idx_E"), "on_I": med("on_I"), "on_E": med("on_E"),
           "contrib": contrib_maps, "layers": R.layers}
    if shared is not None:
        res["on_I_shared"] = med("on_I_shared")
        res["on_E_shared"] = med("on_E_shared")
        res["nFI_shared_total"] = int(sum(r["nFI_shared"] for r in recs))
        res["nFE_shared_total"] = int(sum(r["nFE_shared"] for r in recs))
    # summary over calls ≥ 2 (T032 convention)
    for key in ("idx_I", "idx_E", "on_I", "on_E") + (("on_I_shared", "on_E_shared") if shared is not None else ()):
        vals = [r[key][c] for r in recs for c in range(2, len(r[key])) if np.isfinite(r[key][c])]
        res[f"{key}_calls2plus_median"] = float(np.median(vals)) if vals else None
    save(name, "fallback", res)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list) or k in ("E",)} for r in recs]).to_csv(
        out_dir(name) / "fallback_trials.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k.endswith("median") or k.startswith("n")}, indent=1))
    return res


STAGES = {"profiles": stage_profiles, "labels": stage_labels, "t028": stage_t028, "fallback": stage_fallback}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=(*STAGES, "all"))
    ap.add_argument("name")
    a = ap.parse_args()
    for s in STAGES if a.stage == "all" else [a.stage]:
        print(f"== {s} {a.name}", flush=True)
        STAGES[s](a.name)
