#!/usr/bin/env python
"""T029 neuro_dyn stage B: analyses on the T028 rich re-extraction (CPU only).

Stages (see neuro_dyn_PREREG.md, addendum B, for hypotheses / metrics / baselines)
  extract  per-episode compact cache from rich_acts (8 workers); pilot raw streams are PCA-projected
  pairs    cand <-> base-task train pairs at identical call-0 observation (validity from obs files)
  b1       confirmatory retest: read-in / K,V / expert differences, FALLBACK vs NEW
  a2       token axis: (a) language tokens of cand vs base, (b) 50 action tokens vs planned / executed motion
  a3       Elsayed preparation vs movement subspace alignment
  a4       Churchland jPCA on token / denoise / call axes
  a5       Gallego manifolds: principal angles + CCA-aligned latent dynamics
  a6       Inagaki multi-noise probe: coding direction, bimodality, outcome
  a7       combinatorial code: NNLS on source-task states (+ Kim-style source CD)

Run (OPENBLAS ZEN/Haswell kernels segfault on this EPYC, see neuro_dyn.py):
  OPENBLAS_CORETYPE=SkylakeX .venv-nd/bin/python scripts/shortcut/analysis/neuro_dyn_b.py all
(.venv-nd = numpy scipy scikit-learn matplotlib diptest; diptest is not in the shared viz venv)
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuro_dyn import ACT_DIMS, CACHE, OUT, bh, gr, pf, rmsnorm, row_space  # noqa: E402

RICH = Path("/outputs/VLA-shortcut-exps/rich_acts")
ROLES = Path("/outputs/VLA-shortcut-exps/kubectl/libero_goal_composition/9_composition_traits/roles.csv")
SEMANTICS = Path("/outputs/VLA-shortcut-exps/kubectl/rsa_decoding/semantics.csv")
BC = CACHE / "b"
BO = OUT / "b"
TOK_LAYERS = (0, 2, 5, 8, 11, 14, 17)
EXP_TOK_LAYERS = (2, 5, 8, 11, 14, 17)
KEEP_EXP_TOK = (11, 17)
STEPS = (0, 4, 9)
MASSIVE = (1577, 1959)
SEG = {"agentview": 0, "wrist": 1, "bos": 2, "task": 3, "state": 4, "template": 5}
N_VLM = 18
WORKERS = 8
FALLBACK = ("trained", "fail_trained")
NEW = ("new", "new_byproduct")
RNG = np.random.default_rng(0)


# ============================================================================ io
def manifest(all_rows: bool = False, cached: bool = True) -> list[dict]:
    """all_rows: every manifest row; else rows with a rich file (cached=False) or with a compact cache (cached=True)."""
    rows = list(csv.DictReader(open(RICH / "manifest.csv")))
    out = []
    for r in rows:
        if not all_rows and not ((BC / r["path"]).exists() if cached else (RICH / r["path"]).exists()):
            continue
        for k in ("init", "n_calls", "pilot", "pca_fit", "success"):
            r[k] = int(r[k]) if r[k] not in ("", None) else -1
        for k in ("grasp_call", "release_call", "contact_call"):
            r[k] = int(r[k]) if r[k] not in ("", None) else None
        r["probe_calls"] = [int(x) for x in r["probe_calls"].split(";")] if r["probe_calls"] else []
        out.append(r)
    return out


def f32(z, k):
    return z[k].astype(np.float32)


_BASES = None


def _bases():
    global _BASES
    if _BASES is None:
        _BASES = dict(np.load(RICH / "pca_bases.npz"))
    return _BASES


def _project(x: np.ndarray, stream: str, layers, k: int) -> np.ndarray:
    """raw (C, nL, ..., D) -> (C, nL, ..., k) with the T028 bases (same formula as rich_extract)."""
    b = _bases()
    out = []
    for i, li in enumerate(layers):
        mean = b[f"{stream}/L{li}/mean"]
        comp = b[f"{stream}/L{li}/components"][:k]
        out.append(np.einsum("c...d,kd->c...k", x[:, i] - mean, comp))
    return np.stack(out, 1).astype(np.float32)


def extract_one(r: dict) -> str:
    src = RICH / r["path"]
    dst = BC / r["path"]
    if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
        return "skip"
    z = np.load(src)
    C = z["exp_vec"].shape[0]
    d: dict[str, np.ndarray] = {}
    if "vlm_tok_pc" in z.files:
        vtpc = f32(z, "vlm_tok_pc")[:1]
    else:
        vtpc = _project(f32(z, "vlm_tok")[:1], "vlm_tok", TOK_LAYERS, 256)
        vtpc *= z["lang_valid"][:1, None, :, None]
    if "exp_tok_pc" in z.files:
        etpc = f32(z, "exp_tok_pc")
    else:
        etpc = _project(f32(z, "exp_tok"), "exp_tok", EXP_TOK_LAYERS, 128)
    vseg = f32(z, "vlm_seg")
    d["c0_vseg"] = vseg[0][:, [0, 1, 3, 4]]  # 18, 4 (agentview, wrist, task, state), 2048
    d["c0_kseg"] = f32(z, "k_seg")[0]
    d["c0_vvseg"] = f32(z, "v_seg")[0]
    d["c0_attn"] = f32(z, "attn_mass")[0]  # 10, 18, 7
    em = f32(z, "exp_mean")  # C, 3, 18, 1024
    d["c0_expmean"] = em[0]
    d["c0_ktok"] = f32(z, "k_tok")[0]
    d["c0_vtok"] = f32(z, "v_tok")[0]
    d["c0_vtokpc"] = vtpc[0]
    d["c0_ids"] = z["lang_ids"][0]
    d["c0_lseg"] = z["lang_seg"][0]
    d["c0_valid"] = z["lang_valid"][0]
    d["expL17"] = z["exp_vec"][:, :, 2].astype(np.float16)  # C, 10, 1024
    d["expmean17"] = em[:, :, 17].astype(np.float16)  # C, 3, 1024
    d["vseg11"] = vseg[:, 11][:, [0, 3]]  # C, 2, 2048 (agentview, task)
    d["exptok"] = etpc[:, [EXP_TOK_LAYERS.index(li) for li in KEEP_EXP_TOK]].astype(np.float16)  # C,2,3,50,128
    d["denx"] = z["den_x"][:, :, :, :ACT_DIMS].astype(np.float16)  # C, 11, 50, 7
    d["t"] = z["t"].astype(np.int32)
    for k in ("mn_calls", "mn_act"):
        d[k] = z[k]
    d["mn_exp_last"] = f32(z, "mn_exp_last")
    try:
        s = np.load(r["steps_path"], allow_pickle=False)
        d["eef"] = s["eef_pos"].astype(np.float32)
    except Exception:
        d["eef"] = np.zeros((0, 3), np.float32)
    d["C"] = np.array(C)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(dst)
    return "ok"


def stage_extract() -> None:
    rows = manifest(cached=False)
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(extract_one, rows, chunksize=4))
    print("extract:", {k: res.count(k) for k in set(res)}, "of", len(rows))


def load(r: dict, keys=None) -> dict:
    z = np.load(BC / r["path"])
    return {k: z[k] for k in (keys or z.files)}


def heldout(rows):
    return [r for r in rows if r["pca_fit"] == 0]


# ============================================================================ weights
_W = None


def weights() -> dict:
    global _W
    if _W is None:
        w = dict(np.load(RICH / "weights.npz", allow_pickle=True))
        a = dict(np.load(CACHE / "weights.npz"))  # stage A: expert final AdaRMS scale per step
        out = {}
        for j in range(N_VLM):
            g = 1.0 + w[f"vlm/L{j}/input_layernorm"].astype(np.float64)
            kv = np.concatenate([w[f"vlm/L{j}/k_proj"], w[f"vlm/L{j}/v_proj"]]).astype(np.float64)
            out[f"readin_{j}"] = kv * g[None, :]
        # stage-A readin built from the HF checkpoint must match T028's weights.npz
        out["check_readin6_maxdiff"] = float(np.abs(out["readin_6"] - a["readin_6"]).max())
        for k in STEPS:
            out[f"Wo_{k}"] = w["action_out_proj_w"][:ACT_DIMS].astype(np.float64) * (
                1 + a["exp_final_scale"][k].astype(np.float64)
            )[None, :]
        _W = out
    return _W


def norm_nomassive(x: np.ndarray) -> np.ndarray:
    y = rmsnorm(np.asarray(x, np.float64))
    y[..., list(MASSIVE)] = 0
    return y


# ============================================================================ stats helpers
def strat_perm(vals: np.ndarray, lab: np.ndarray, strata: list, n: int = 10000) -> tuple[float, float]:
    """mean(vals[lab]) - mean(vals[~lab]); p from label permutation within strata (two-sided)."""
    ok = np.isfinite(vals)
    vals, lab, strata = vals[ok], lab[ok], np.asarray(strata, dtype=object)[ok]
    if lab.sum() < 2 or (~lab).sum() < 2:
        return float("nan"), float("nan")
    obs = vals[lab].mean() - vals[~lab].mean()
    groups = [np.where(strata == s)[0] for s in set(strata)]
    cnt = 0
    lp = lab.copy()
    for _ in range(n):
        for g in groups:
            lp[g] = RNG.permutation(lab[g])
        if lp.sum() == 0 or lp.sum() == len(lp):
            continue
        st = vals[lp].mean() - vals[~lp].mean()
        cnt += abs(st) >= abs(obs) - 1e-12
    return float(obs), float((cnt + 1) / (n + 1))


def tost_boot(vals, lab, strata, margin_sd=0.25, n=2000):
    ok = np.isfinite(vals)
    vals, lab, strata = vals[ok], lab[ok], np.asarray(strata, dtype=object)[ok]
    if lab.sum() < 2 or (~lab).sum() < 2:
        return {"equivalent": None}
    sd = vals.std(ddof=1)
    margin = margin_sd * sd
    us = sorted(set(strata))
    bs = []
    for _ in range(n):
        pick = RNG.choice(len(us), len(us))
        idx = np.concatenate([np.where(strata == us[i])[0] for i in pick])
        a, b = vals[idx][lab[idx]], vals[idx][~lab[idx]]
        if len(a) and len(b):
            bs.append(a.mean() - b.mean())
    lo, hi = np.quantile(bs, [0.05, 0.95])
    return {"diff_ci90": [float(lo), float(hi)], "margin": float(margin), "equivalent": bool(lo > -margin and hi < margin)}


def wil(x):
    from scipy.stats import wilcoxon

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return float("nan")
    return float(wilcoxon(x).pvalue)


def mwu(a, b):
    from scipy.stats import mannwhitneyu

    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    return float(mannwhitneyu(a, b).pvalue)


class Tests:
    def __init__(self, family: str):
        self.family = family
        self.rows: list[dict] = []

    def add(self, name, p, **kw):
        self.rows.append({"name": f"{self.family}.{name}", "p": p, **kw})

    def finish(self) -> dict:
        q = bh([r["p"] for r in self.rows])
        for r, qq in zip(self.rows, q):
            r["q_bh"] = float(qq)
        return {"n_tests": len(self.rows), "n_sig_q05": int(np.nansum(q < 0.05)), "tests": self.rows}


def save(name: str, res: dict) -> None:
    BO.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(BO / f"{name}.json", "w"), indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as p

    return p


# ============================================================================ pairs
def roles() -> dict:
    return {r["cand"]: r for r in csv.DictReader(open(ROLES))}


def stage_pairs() -> list[dict]:
    rows = manifest(all_rows=True)
    have = {r["path"] for r in manifest()}
    ro = roles()
    train = {(r["task"], r["init"]): r for r in rows if r["set"] == "goal" and r["condition"] == "train"}
    out = []
    for r in rows:
        if r["set"] != "goal" or r["condition"] != "cand":
            continue
        bt = ro.get(r["variant"], {}).get("base_task", "")
        rec = {"cand": r["path"], "variant": r["variant"], "init": r["init"], "outcome": r["outcome"], "base_task": bt}
        if ";" in bt or not bt:
            rec["valid"], rec["why"] = False, "no single base task"
            out.append(rec)
            continue
        t = train.get(("G" + bt, r["init"]))
        if t is None:
            rec["valid"], rec["why"] = False, "no base train trial"
            out.append(rec)
            continue
        rec["base"] = t["path"]
        a, b = np.load(r["obs_path"]), np.load(t["obs_path"])
        di = max(float(np.abs(a[k][0].astype(np.int16) - b[k][0].astype(np.int16)).max()) for k in ("image", "image2"))
        ds = float(np.abs(a["state"][0] - b["state"][0]).max())
        rec.update({"img_maxdiff": di, "state_maxdiff": ds, "valid": bool(di <= 2 and ds < 1e-3)})
        rec["why"] = "" if rec["valid"] else "call-0 obs differ (scene modified)"
        rec["rich_available"] = r["path"] in have and t["path"] in have
        out.append(rec)
    BO.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(BO / "pairs.json", "w"), indent=1)
    v = [p for p in out if p["valid"]]
    print(f"pairs: {len(out)} cand trials, valid {len(v)}, rich-available {sum(p.get('rich_available', False) for p in v)}")
    return out


def valid_pairs() -> list[dict]:
    if not (BO / "pairs.json").exists():
        stage_pairs()
    return [p for p in json.load(open(BO / "pairs.json")) if p["valid"] and p.get("rich_available")]


# ============================================================================ B1
def stage_b1() -> dict:
    W = weights()
    pairs = valid_pairs()
    rows = {r["path"]: r for r in manifest()}
    T = Tests("B1")
    recs = []
    for p in pairs:
        a, b = load(rows[p["cand"]]), load(rows[p["base"]])
        rec = {"variant": p["variant"], "outcome": p["outcome"], "init": p["init"]}
        ti = 2  # task index in c0_vseg (agentview, wrist, task, state)
        pfs, pfs_nm, relx = [], [], []
        for l in range(N_VLM - 1):
            xa, xb = rmsnorm(a["c0_vseg"][l, ti].astype(np.float64)), rmsnorm(b["c0_vseg"][l, ti].astype(np.float64))
            d = (xa - xb)[None]
            B, _ = row_space(W[f"readin_{l + 1}"]) if f"B_{l + 1}" not in W else (W[f"B_{l + 1}"], None)
            W[f"B_{l + 1}"] = B
            pfs.append(float(pf(d, B)[0]))
            dn = d.copy()
            dn[:, list(MASSIVE)] = 0
            pfs_nm.append(float(pf(dn, B)[0]))
            relx.append(float(np.linalg.norm(d) / np.linalg.norm(xb)))
        rec["readin_pf"] = pfs
        rec["readin_pf_nomassive"] = pfs_nm
        rec["resid_rel"] = relx
        st = SEG["task"]
        rec["k_rel"] = [float(np.linalg.norm(a["c0_kseg"][l, st] - b["c0_kseg"][l, st]) / np.linalg.norm(b["c0_kseg"][l, st])) for l in range(N_VLM)]
        rec["v_rel"] = [float(np.linalg.norm(a["c0_vvseg"][l, st] - b["c0_vvseg"][l, st]) / np.linalg.norm(b["c0_vvseg"][l, st])) for l in range(N_VLM)]
        rec["attn_task_cand"] = a["c0_attn"][0, :, st].tolist()
        rec["attn_task_base"] = b["c0_attn"][0, :, st].tolist()
        de = (a["c0_expmean"][0, 17].astype(np.float64) - b["c0_expmean"][0, 17].astype(np.float64))[None]
        Bo, _ = row_space(W["Wo_0"])
        rec["exp_norm"] = float(np.linalg.norm(de))
        rec["exp_pf"] = float(pf(de, Bo)[0])
        rec["exp_gr"] = float(gr(de, W["Wo_0"])[0])
        recs.append(rec)
    res = {"n_pairs": len(recs), "outcomes": {o: sum(r["outcome"] == o for r in recs) for o in set(r["outcome"] for r in recs)},
           "weights_check_readin6_maxdiff_vs_stageA": W["check_readin6_maxdiff"]}
    sel = [r for r in recs if r["outcome"] in FALLBACK + NEW]
    lab = np.array([r["outcome"] in FALLBACK for r in sel])
    strata = [r["variant"] for r in sel]
    metrics = {
        "readin_pf_mean_L1_16": np.array([np.mean(r["readin_pf"][1:17]) for r in sel]),
        "resid_rel_mean": np.array([np.mean(r["resid_rel"][1:17]) for r in sel]),
        "k_rel_mean": np.array([np.mean(r["k_rel"][1:]) for r in sel]),
        "v_rel_mean": np.array([np.mean(r["v_rel"][1:]) for r in sel]),
        "attn_task_mean_cand": np.array([np.mean(r["attn_task_cand"]) for r in sel]),
        "exp_norm": np.array([r["exp_norm"] for r in sel]),
        "exp_pf": np.array([r["exp_pf"] for r in sel]),
    }
    summ = {}
    for k, v in metrics.items():
        diff, p = strat_perm(v, lab, strata)
        ent = {"fallback_median": float(np.nanmedian(v[lab])) if lab.any() else None,
               "new_median": float(np.nanmedian(v[~lab])) if (~lab).any() else None,
               "diff_mean_fallback_minus_new": diff, "p_perm": p}
        if k in ("readin_pf_mean_L1_16", "resid_rel_mean", "k_rel_mean", "v_rel_mean", "attn_task_mean_cand"):
            ent["tost"] = tost_boot(v, lab, strata)
        summ[k] = ent
        T.add(k, p, diff=diff)
    res["fallback_vs_new"] = summ
    res["n_fallback"], res["n_new"] = int(lab.sum()), int((~lab).sum())
    # secondary: lscene trained vs nothing (unpaired)
    ls = [r for r in manifest() if r["condition"] == "lscene"]
    att = defaultdict(list)
    for r in ls:
        if r["outcome"] in ("trained", "nothing"):
            d = load(r, ["c0_attn", "c0_vvseg"])
            att[r["outcome"]].append((float(d["c0_attn"][0, :, SEG["task"]].mean()),
                                      float(np.mean(d["c0_attn"][0, :, SEG["task"]] * np.linalg.norm(d["c0_vvseg"][:, SEG["task"]], axis=-1)))))
    if att:
        res["lscene_secondary"] = {o: {"n": len(v), "attn_task_median": float(np.median([x[0] for x in v])),
                                       "mass_x_vnorm_median": float(np.median([x[1] for x in v]))} for o, v in att.items()}
        p = mwu([x[1] for x in att.get("trained", [])], [x[1] for x in att.get("nothing", [])])
        res["lscene_secondary"]["p_mwu_mass_x_vnorm"] = p
        T.add("lscene_trained_vs_nothing_mass_x_vnorm", p)
    res["per_pair"] = recs
    res.update(T.finish())
    save("b1", res)
    plot_b1(recs)
    print(json.dumps({k: res[k] for k in ("n_pairs", "n_fallback", "n_new", "n_tests", "n_sig_q05")}))
    return res


def plot_b1(recs):
    p = plt()
    groups = {"FALLBACK": [r for r in recs if r["outcome"] in FALLBACK], "NEW": [r for r in recs if r["outcome"] in NEW]}
    cols = {"FALLBACK": "#c0392b", "NEW": "#1b3f8b"}
    fig, ax = p.subplots(1, 4, figsize=(18, 4))
    for g, rr in groups.items():
        if not rr:
            continue
        for i, (key, lab) in enumerate((("readin_pf", "read-in PF (task seg, out l → read l+1)"), ("v_rel", "|ΔV| / |V| (task seg)"), ("k_rel", "|ΔK| / |K| (task seg)"))):
            m = np.array([r[key] for r in rr])
            x = np.arange(m.shape[1]) + (1 if key == "readin_pf" else 0)
            ax[i].plot(x, np.median(m, 0), "-o", ms=3, color=cols[g], label=f"{g} (n={len(rr)})")
            ax[i].fill_between(x, np.quantile(m, 0.25, 0), np.quantile(m, 0.75, 0), color=cols[g], alpha=0.15)
            ax[i].set_title(lab, fontsize=9)
            ax[i].set_xlabel("VLM layer read by expert")
        ax[3].scatter([r["exp_norm"] for r in rr], [r["exp_pf"] for r in rr], color=cols[g], s=12, label=g, alpha=0.7)
    ax[0].axhline(0.25, color="k", ls="--", lw=1)
    ax[3].set_xlabel("|Δ expert L17| (step 0)")
    ax[3].set_ylabel("action-potent fraction")
    ax[3].set_yscale("log")
    ax[0].legend(fontsize=8)
    ax[3].legend(fontsize=8)
    fig.suptitle("B1: cand vs base task, identical call-0 observation — what changes where (median, IQR)")
    fig.tight_layout()
    fig.savefig(BO / "b1_fallback_vs_new.png", dpi=130)
    p.close(fig)


# ============================================================================ A2
def _recon_tok(pc: np.ndarray, li: int) -> np.ndarray:
    b = _bases()
    return pc.astype(np.float64) @ b[f"vlm_tok/L{li}/components"][: pc.shape[-1]].astype(np.float64) + b[f"vlm_tok/L{li}/mean"]


def stage_a2() -> dict:
    W = weights()
    T = Tests("A2")
    res: dict = {}
    rows = {r["path"]: r for r in manifest()}
    # ---- (a) language tokens
    pairs = valid_pairs()
    OFF = range(-6, 7)
    acc = {f"{s}_L{li}": defaultdict(list) for s in ("k", "v", "x") for li in TOK_LAYERS}
    pf_match, pf_diff = defaultdict(list), defaultdict(list)
    slopes = defaultdict(list)
    first_side = defaultdict(list)
    for p in pairs:
        a, b = load(rows[p["cand"]]), load(rows[p["base"]])
        ia = np.where(a["c0_valid"] & (a["c0_lseg"] == SEG["task"]))[0]
        ib = np.where(b["c0_valid"] & (b["c0_lseg"] == SEG["task"]))[0]
        sm = difflib.SequenceMatcher(a=a["c0_ids"][ia].tolist(), b=b["c0_ids"][ib].tolist(), autojunk=False)
        match = []
        for blk in sm.get_matching_blocks():
            match += [(ia[blk.a + t], ib[blk.b + t]) for t in range(blk.size)]
        unmatched_a = sorted(set(ia) - {m[0] for m in match})
        if not match or not unmatched_a:
            continue
        ua = np.array(unmatched_a)
        first_diff = ua.min()
        per_layer_dist = []
        for i, li in enumerate(TOK_LAYERS):
            xa = norm_nomassive(_recon_tok(a["c0_vtokpc"][i], li))
            xb = norm_nomassive(_recon_tok(b["c0_vtokpc"][i], li))
            dl = []
            for (ta, tb) in match:
                off = int(np.clip(ta - ua[np.argmin(np.abs(ua - ta))], -6, 6))
                for s, A, Bm in (("k", a["c0_ktok"][i], b["c0_ktok"][i]), ("v", a["c0_vtok"][i], b["c0_vtok"][i]), ("x", xa, xb)):
                    u, w = A[ta].astype(np.float64), Bm[tb].astype(np.float64)
                    cd = 1 - u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-12)
                    acc[f"{s}_L{li}"][off].append(cd)
                    if s == "k":
                        dl.append(cd)
                        if ta < first_diff:
                            first_side[li].append(cd)
            per_layer_dist.append(np.mean(dl))
            if li < 17:
                B = row_space(W[f"readin_{li + 1}"])[0]
                xa_f = rmsnorm(_recon_tok(a["c0_vtokpc"][i], li))
                xb_f = rmsnorm(_recon_tok(b["c0_vtokpc"][i], li))
                dm = np.stack([xa_f[ta] - xb_f[tb] for ta, tb in match])
                pf_match[li].append(float(np.median(pf(dm, B))))
                rep = [op for op in sm.get_opcodes() if op[0] == "replace"]
                if rep:
                    dd = np.stack([xa_f[ia[o[1]:o[2]]].mean(0) - xb_f[ib[o[3]:o[4]]].mean(0) for o in rep])
                    pf_diff[li].append(float(np.median(pf(dd, B))))
                else:
                    pf_diff[li].append(np.nan)
        from scipy.stats import spearmanr

        slopes[p["variant"]].append(spearmanr(TOK_LAYERS, per_layer_dist).correlation)
    res["a_lang"] = {
        "n_pairs_used": int(sum(len(v) for v in slopes.values())),
        "dist_by_offset": {k: {int(o): float(np.mean(v[o])) for o in OFF if v[o]} for k, v in acc.items()},
        "pre_diff_token_k_cosdist_median": {int(li): float(np.median(v)) for li, v in first_side.items()},
        "pf_matched_median": {int(li): float(np.nanmedian(v)) for li, v in pf_match.items()},
        "pf_differing_median": {int(li): float(np.nanmedian(v)) for li, v in pf_diff.items()},
    }
    cand_slope = np.array([np.mean(v) for v in slopes.values()])
    T.add("lang_divergence_grows_with_depth", wil(cand_slope), median_rho=float(np.median(cand_slope)) if len(cand_slope) else None)
    for li in pf_match:
        d = np.array(pf_match[li]) - np.array(pf_diff[li])
        T.add(f"lang_pf_matched_minus_differing_L{li}", wil(d), median=float(np.nanmedian(d)))
    # ---- (b) action tokens
    from scipy.stats import spearmanr

    recs = defaultdict(list)
    traj_by_cond = defaultdict(list)
    for r in heldout(manifest()):
        z = load(r, ["exptok", "denx", "t", "eef"])
        et = z["exptok"][:, 1].astype(np.float64)[..., :10]  # C, 3 steps, 50, 10 (L17)
        L = np.linalg.norm(np.diff(et, axis=2), axis=-1).sum(-1)  # C, 3
        ch = z["denx"][:, -1].astype(np.float64)  # C, 50, 7
        Lch = np.linalg.norm(np.diff(ch, axis=1), axis=-1).sum(-1)
        eef = z["eef"]
        for c in range(len(L)):
            t0 = int(z["t"][c])
            ex = float(np.linalg.norm(np.diff(eef[t0 : t0 + 11], axis=0), axis=-1).sum()) if len(eef) > t0 + 1 else np.nan
            recs[r["task"] + "|" + r["set"]].append((L[c, 0], L[c, 1], L[c, 2], Lch[c], ex))
        traj_by_cond[(r["set"], r["condition"])].append(et[0, 2])
    rho_ch, rho_ex, grow = [], [], []
    for k, v in recs.items():
        v = np.array(v)
        if len(v) < 6:
            continue
        rho_ch.append(spearmanr(v[:, 2], v[:, 3]).correlation)
        ok = np.isfinite(v[:, 4])
        rho_ex.append(spearmanr(v[ok, 2], v[ok, 4]).correlation if ok.sum() > 5 else np.nan)
        grow.append(np.median(v[:, 2] - v[:, 0]))
    allv = np.concatenate([np.array(v) for v in recs.values()]) if recs else np.zeros((0, 5))
    res["b_action"] = {
        "n_tasks": len(rho_ch),
        "pathlen_median_by_step": [float(np.median(allv[:, i])) for i in range(3)] if len(allv) else None,
        "rho_pathlen_vs_chunk_median_over_tasks": float(np.nanmedian(rho_ch)) if rho_ch else None,
        "rho_pathlen_vs_executed_median_over_tasks": float(np.nanmedian(rho_ex)) if rho_ex else None,
        "rho_pooled_chunk": float(spearmanr(allv[:, 2], allv[:, 3]).correlation) if len(allv) > 5 else None,
    }
    T.add("action_pathlen_grows_step0_to_9", wil(grow), median=float(np.median(grow)) if grow else None)
    T.add("action_rho_chunk_gt0", wil(rho_ch), median=float(np.nanmedian(rho_ch)) if rho_ch else None)
    T.add("action_rho_executed_gt0", wil(rho_ex), median=float(np.nanmedian(rho_ex)) if rho_ex else None)
    res.update(T.finish())
    save("a2", res)
    plot_a2(res, traj_by_cond)
    return res


def plot_a2(res, traj_by_cond):
    p = plt()
    fig, ax = p.subplots(1, 3, figsize=(17, 4.5))
    dbo = res["a_lang"]["dist_by_offset"]
    cmap = p.get_cmap("viridis")
    for i, li in enumerate(TOK_LAYERS):
        d = dbo.get(f"k_L{li}", {})
        if d:
            o = sorted(d)
            ax[0].plot(o, [d[k] for k in o], "-o", ms=3, color=cmap(i / 6), label=f"K L{li}")
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("signed token distance to nearest differing word (clipped ±6)")
    ax[0].set_ylabel("cosine distance of K (same word, cand vs base)")
    ax[0].set_title("A2a: same words diverge on both sides (bidirectional prefix)")
    ax[0].legend(fontsize=7, ncol=2)
    pm, pd = res["a_lang"]["pf_matched_median"], res["a_lang"]["pf_differing_median"]
    ls = sorted(pm, key=int)
    ax[1].plot([int(k) for k in ls], [pm[k] for k in ls], "-o", label="same words (matched)")
    ax[1].plot([int(k) for k in ls], [pd[k] for k in ls], "-s", label="differing words")
    ax[1].axhline(0.25, color="k", ls="--", lw=1, label="random")
    ax[1].set_xlabel("VLM layer output (read by l+1)")
    ax[1].set_ylabel("read-in potent fraction of Δ (median)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("A2a: per-token Kaufman split")
    for (s, c), tr in sorted(traj_by_cond.items()):
        m = np.mean(tr, 0)
        ax[2].plot(m[:, 0], m[:, 1], "-", lw=1.2, label=f"{s}:{c}")
        ax[2].plot(m[0, 0], m[0, 1], "o", ms=3, color="k")
    ax[2].set_xlabel("PC1")
    ax[2].set_ylabel("PC2")
    ax[2].set_title("A2b: mean action-token trajectory 0→49 (L17, step 9, call 0; dot = token 0)")
    ax[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(BO / "a2_tokens.png", dpi=130)
    p.close(fig)


# ============================================================================ A3
def _cov_basis(X: np.ndarray, k: int | None = None, var: float = 0.8, kmax: int = 10):
    Xc = X - X.mean(0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    ev = s**2 / max(len(X) - 1, 1)
    if k is None:
        k = int(min(kmax, np.searchsorted(np.cumsum(ev) / ev.sum(), var) + 1))
    return vt[:k].T, ev, vt, k


def alignment_index(Xp, Xm, n_rand=1000):
    Up, _, _, k = _cov_basis(Xp)
    Um, evm, vtm, _ = _cov_basis(Xm, k=k)
    Xmc = Xm - Xm.mean(0)

    def ai(U):
        proj = Xmc @ U
        return float((proj**2).sum() / max(len(Xm) - 1, 1) / evm[:k].sum())

    obs = ai(Up)
    allX = np.concatenate([Xp - Xp.mean(0), Xm - Xm.mean(0)])
    _, s, vt = np.linalg.svd(allX, full_matrices=False)
    rnd = []
    for _ in range(n_rand):
        r = RNG.standard_normal((len(s), k))
        v = vt.T @ (s[:, None] * r)
        q, _ = np.linalg.qr(v)
        rnd.append(ai(q))
    rnd = np.array(rnd)
    return {"ai": obs, "k": k, "rand_median": float(np.median(rnd)), "rand_q05": float(np.quantile(rnd, 0.05)),
            "p_below": float((np.sum(rnd <= obs) + 1) / (len(rnd) + 1)), "n_prep": len(Xp), "n_move": len(Xm)}


def stage_a3() -> dict:
    T = Tests("A3")
    res = {}
    groups = {"libero3": lambda r: r["set"] == "libero3", "goal_train": lambda r: r["set"] == "goal" and r["condition"] == "train",
              "basket": lambda r: r["set"].startswith("basket")}
    for g0, f0 in list(groups.items()):
        groups[g0 + "_heldout_only"] = (lambda f: (lambda r: f(r) and r["pca_fit"] == 0))(f0)
    for g, f in groups.items():
        ep = [r for r in manifest() if f(r)]
        P = defaultdict(list)
        M = defaultdict(list)
        ridge_X, ridge_Xe, ridge_y, ridge_g = [], [], [], []
        for r in ep:
            anc = r["grasp_call"] if r["grasp_call"] is not None else r["contact_call"]
            if anc is None or anc < 1:
                continue
            z = load(r, ["expmean17", "vseg11"])
            C = len(z["expmean17"])
            rel = r["release_call"] if r["release_call"] is not None else C - 1
            mv = list(range(anc, min(rel, C - 1) + 1))
            if not mv:
                continue
            e = z["expmean17"][:, 2].astype(np.float64)
            va = norm_nomassive(z["vseg11"][:, 0])
            vt = norm_nomassive(z["vseg11"][:, 1])
            P["expert"].append(e[:anc])
            M["expert"].append(e[mv])
            P["vlm_agentview"].append(va[:anc])
            M["vlm_agentview"].append(va[mv])
            P["vlm_task"].append(vt[:anc])
            M["vlm_task"].append(vt[mv])
            ridge_X.append(np.concatenate([va[0], vt[0]]))
            ridge_Xe.append(e[0])
            ridge_y.append(e[mv].mean(0))
            ridge_g.append(r["task"])
        gres = {}
        for sig in P:
            if len(P[sig]) < 3:
                continue
            gres[sig] = alignment_index(np.concatenate(P[sig]), np.concatenate(M[sig]))
            T.add(f"{g}.{sig}.ai_below_random", gres[sig]["p_below"], ai=gres[sig]["ai"])
        if len(ridge_y) >= 10 and len(set(ridge_g)) >= 3:
            gres["ridge_move_from_call0"] = _ridge_cv(np.array(ridge_X), np.array(ridge_Xe), np.array(ridge_y), np.array(ridge_g))
        res[g] = gres
    res.update(T.finish())
    save("a3", res)
    plot_a3(res)
    return res


def _ridge_cv(Xv, Xe, y, g):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    out = {}
    for name, X in (("vlm_call0", Xv), ("expert_call0", Xe)):
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(min(5, len(set(g)))).split(X, y, g):
            px = PCA(min(30, len(tr) - 1)).fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(px.transform(X[tr]), y[tr])
            pred[te] = m.predict(px.transform(X[te]))
        out[name] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
    return out


def plot_a3(res):
    p = plt()
    gs = [g for g in res if isinstance(res[g], dict) and "expert" in res[g]]
    fig, ax = p.subplots(1, max(1, len(gs)), figsize=(5 * max(1, len(gs)), 4), squeeze=False)
    for i, g in enumerate(gs):
        sigs = [s for s in ("expert", "vlm_agentview", "vlm_task") if s in res[g]]
        x = np.arange(len(sigs))
        ax[0, i].bar(x, [res[g][s]["ai"] for s in sigs], color=["#1b3f8b", "#d9822b", "#e8b07a"][: len(sigs)], label="prep→move alignment")
        ax[0, i].errorbar(x, [res[g][s]["rand_median"] for s in sigs], fmt="k_", ms=20, label="random (cov-matched) median")
        ax[0, i].scatter(x, [res[g][s]["rand_q05"] for s in sigs], marker="v", color="gray", label="random 5th pct")
        ax[0, i].set_xticks(x, sigs)
        ax[0, i].set_ylim(0, 1)
        ax[0, i].set_title(f"A3 Elsayed · {g}")
    ax[0, 0].set_ylabel("alignment index")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(BO / "a3_alignment.png", dpi=130)
    p.close(fig)


# ============================================================================ A4
def _skew_fit(X, dX):
    d = X.shape[1]
    idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
    A = np.zeros((X.shape[0] * d, len(idx)))
    for n, (i, j) in enumerate(idx):
        # dX[:, j] += m * X[:, i] ; dX[:, i] -= m * X[:, j]   (dX = X @ M, M[i,j]=m, M[j,i]=-m)
        col = np.zeros((X.shape[0], d))
        col[:, j] += X[:, i]
        col[:, i] -= X[:, j]
        A[:, n] = col.ravel()
    m, *_ = np.linalg.lstsq(A, dX.ravel(), rcond=None)
    M = np.zeros((d, d))
    for n, (i, j) in enumerate(idx):
        M[i, j], M[j, i] = m[n], -m[n]
    return M


def jpca(trajs: list[np.ndarray], groups: list, n_pc: int = 6) -> dict:
    """Churchland jPCA. trajs: equal-length (T, D) trajectories. The cross-condition mean trajectory of the
    training fold is subtracted (standard jPCA), PCA (n_pc) is fit on the training fold, dx = M x is fit
    unconstrained and skew-symmetric, and R² is evaluated on held-out groups."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    ug = sorted(set(groups))
    if len(ug) < 2 or len(trajs) < 4:
        return {}
    Xall = np.stack(trajs)  # n, T, D

    def prep(ids, mt, mu, V):
        Xs, dXs = [], []
        for i in ids:
            x = (Xall[i] - mt - mu) @ V
            Xs.append((x[1:] + x[:-1]) / 2)
            dXs.append(np.diff(x, axis=0))
        return np.concatenate(Xs), np.concatenate(dXs)

    def basis(ids):
        mt = Xall[ids].mean(0)
        S = (Xall[ids] - mt).reshape(-1, Xall.shape[2])
        mu = S.mean(0)
        V = np.linalg.svd(S - mu, full_matrices=False)[2][:n_pc].T
        return mt, mu, V

    r2f, r2s = [], []
    for tr, te in GroupKFold(min(5, len(ug))).split(np.zeros(len(trajs)), groups=groups):
        mt, mu, V = basis(tr)
        Xtr, dtr = prep(tr, mt, mu, V)
        Xte, dte = prep(te, mt, mu, V)
        Mf, *_ = np.linalg.lstsq(Xtr, dtr, rcond=None)
        Ms = _skew_fit(Xtr, dtr)
        sst = ((dte - dte.mean(0)) ** 2).sum()
        r2f.append(1 - ((dte - Xte @ Mf) ** 2).sum() / sst)
        r2s.append(1 - ((dte - Xte @ Ms) ** 2).sum() / sst)
    allids = np.arange(len(trajs))
    mt, mu, V = basis(allids)
    X, dX = prep(allids, mt, mu, V)
    Ms = _skew_fit(X, dX)
    ev = np.linalg.eigvals(Ms)
    f = float(np.abs(ev.imag).max() / (2 * np.pi))
    r2f, r2s = float(np.mean(r2f)), float(np.mean(r2s))
    return {"r2_full_cv": r2f, "r2_skew_cv": r2s, "ratio": float(r2s / r2f) if r2f > 0 else float("nan"),
            "rot_freq_cycles_per_unit": f, "n_traj": len(trajs)}


def stage_a4() -> dict:
    T = Tests("A4")
    res: dict = {"all": {}, "by_condition": {}}
    tok, den, cal = [], [], []
    tg, dg, cg = [], [], []
    tcond, dcond = [], []
    for r in manifest():
        z = load(r, ["exptok", "expL17"])
        e = z["expL17"].astype(np.float64)  # C, 10, 1024
        C = len(e)
        g = r["set"] + "|" + r["task"]
        for c in sorted({0, C // 2, C - 1}):
            den.append(e[c])
            dg.append(g)
            dcond.append((r["set"], r["condition"]))
            if r["pca_fit"] == 0:  # token axis uses T028 PCA coordinates -> held-out episodes only
                tok.append(z["exptok"][c, 1, 2].astype(np.float64))  # 50, 128 (L17 step 9)
                tg.append(g)
                tcond.append((r["set"], r["condition"]))
        if C > 3:
            cal.append(_bin(e[:, 9]))
            cg.append(g)
    res["all"]["tokens"] = jpca(tok, tg)
    res["all"]["denoise"] = jpca(den, dg)
    res["all"]["calls"] = jpca(cal, cg)
    # task-cluster bootstrap of the ratio is expensive; use per-condition fits as the variability estimate
    per = defaultdict(dict)
    for k in sorted(set(dcond)):
        ii = [i for i, c in enumerate(tcond) if c == k]
        jj = [i for i, c in enumerate(dcond) if c == k]
        per[f"{k[0]}:{k[1]}"]["tokens"] = jpca([tok[i] for i in ii], [tg[i] for i in ii])
        per[f"{k[0]}:{k[1]}"]["denoise"] = jpca([den[i] for i in jj], [dg[i] for i in jj])
    res["by_condition"] = per
    rt = [v["tokens"].get("ratio", np.nan) for v in per.values() if v.get("tokens")]
    rd = [v["denoise"].get("ratio", np.nan) for v in per.values() if v.get("denoise")]
    T.add("tokens_ratio_gt_0.5", wil(np.array(rt) - 0.5), median=float(np.nanmedian(rt)) if rt else None)
    T.add("denoise_ratio_lt_0.5", wil(np.array(rd) - 0.5), median=float(np.nanmedian(rd)) if rd else None)
    res.update(T.finish())
    save("a4", res)
    plot_a4(res)
    return res


def plot_a4(res):
    p = plt()
    fig, ax = p.subplots(1, 2, figsize=(13, 4))
    axes = ("tokens", "denoise", "calls")
    a = res["all"]
    x = np.arange(3)
    ax[0].bar(x - 0.2, [a[k].get("r2_full_cv", np.nan) for k in axes], 0.4, label="unconstrained M", color="#aaaaaa")
    ax[0].bar(x + 0.2, [a[k].get("r2_skew_cv", np.nan) for k in axes], 0.4, label="skew-symmetric M (rotation)", color="#1b3f8b")
    ax[0].set_xticks(x, ["action tokens 0→49", "denoise steps 0→9", "calls"])
    ax[0].set_ylabel("held-out-task R² of dx = Mx")
    ax[0].legend(fontsize=8)
    ax[0].set_title("A4 jPCA, expert L17 (all held-out episodes)")
    per = res["by_condition"]
    ks = sorted(per)
    ax[1].scatter([per[k]["tokens"].get("ratio", np.nan) for k in ks], [per[k]["denoise"].get("ratio", np.nan) for k in ks])
    for k in ks:
        ax[1].annotate(k, (per[k]["tokens"].get("ratio", np.nan), per[k]["denoise"].get("ratio", np.nan)), fontsize=6)
    ax[1].axvline(0.5, color="k", ls="--", lw=0.8)
    ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("R²skew / R²full, action-token axis")
    ax[1].set_ylabel("R²skew / R²full, denoise axis")
    ax[1].set_title("per set:condition")
    fig.tight_layout()
    fig.savefig(BO / "a4_jpca.png", dpi=130)
    p.close(fig)


# ============================================================================ A5
def _bin(traj: np.ndarray, nb: int = 20) -> np.ndarray:
    t = np.linspace(0, 1, len(traj))
    tb = np.linspace(0, 1, nb)
    return np.stack([np.interp(tb, t, traj[:, j]) for j in range(traj.shape[1])], 1)


def _angles(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def _cca(X, Y, k):
    from sklearn.cross_decomposition import CCA

    return CCA(n_components=k, max_iter=2000).fit(X, Y)


def stage_a5(dim: int = 10, cdim: int = 4) -> dict:
    T = Tests("A5")
    sem = {r["task"]: r for r in csv.DictReader(open(SEMANTICS))}
    units = defaultdict(list)
    for r in manifest():
        if r["set"] in ("libero90",) or r["condition"] == "cand":
            continue
        units[(r["set"], r["condition"], r["task"])].append(r)
    data = {}
    for u, rr in units.items():
        if len(rr) < 4:
            continue
        rr = sorted(rr, key=lambda r: r["init"])
        trajs = [load(r, ["expmean17"])["expmean17"][:, 2].astype(np.float64) for r in rr]
        data[u] = {"trajs": trajs, "inits": [r["init"] for r in rr]}
    allS = np.concatenate([np.concatenate(v["trajs"]) for v in data.values()]) if data else np.zeros((0, 1024))
    if len(data) < 2:
        res = {"note": "fewer than 2 units with >= 4 held-out episodes"}
        save("a5", res)
        return res
    mu_all = allS.mean(0)
    _, s_all, vt_all = np.linalg.svd(allS - mu_all, full_matrices=False)
    for u, v in data.items():
        S = np.concatenate(v["trajs"])
        v["mu"] = S.mean(0)
        v["U"] = np.linalg.svd(S - v["mu"], full_matrices=False)[2][:dim].T
        half = [np.concatenate(v["trajs"][i::2]) for i in (0, 1)]
        Uh = [np.linalg.svd(h - h.mean(0), full_matrices=False)[2][:dim].T for h in half]
        v["self_angle"] = float(np.mean(_angles(*Uh)))
        v["UA"] = Uh[0]
        v["binsA"] = [_bin(t) for t in v["trajs"][0::2]]
        v["binsB"] = [_bin(t) for t in v["trajs"][1::2]]

    def rand_angle(n=200):
        out = []
        for _ in range(n):
            q = [np.linalg.qr(vt_all.T @ (s_all[:, None] * RNG.standard_normal((len(s_all), dim))))[0] for _ in range(2)]
            out.append(np.mean(_angles(*q)))
        return np.array(out)

    rnd = rand_angle()
    res = {"random_angle_median": float(np.median(rnd)), "random_angle_q05": float(np.quantile(rnd, 0.05)), "pairs": []}
    us = sorted(data)
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            a, b = data[us[i]], data[us[j]]
            ang = float(np.mean(_angles(a["U"], b["U"])))
            n = min(len(a["binsA"]), len(b["binsA"]))
            nB = min(len(a["binsB"]), len(b["binsB"]))
            cc = np.nan
            if n >= 1 and nB >= 1:
                XA = np.concatenate([(x - a["mu"]) @ a["UA"] for x in a["binsA"][:n]])
                YA = np.concatenate([(x - b["mu"]) @ b["UA"] for x in b["binsA"][:n]])
                XB = np.concatenate([(x - a["mu"]) @ a["UA"] for x in a["binsB"][:nB]])
                YB = np.concatenate([(x - b["mu"]) @ b["UA"] for x in b["binsB"][:nB]])
                try:
                    m = _cca(XA, YA, cdim)
                    u_, v_ = m.transform(XB, YB)
                    cc = float(np.mean([np.corrcoef(u_[:, k], v_[:, k])[0, 1] for k in range(cdim)]))
                except Exception:
                    cc = np.nan
            va = sem.get(us[i][2], {}).get("verb", "")
            vb = sem.get(us[j][2], {}).get("verb", "")
            res["pairs"].append({"a": "|".join(us[i]), "b": "|".join(us[j]), "angle": ang, "cca_r": cc,
                                 "self_angle_a": a["self_angle"], "self_angle_b": b["self_angle"],
                                 "both_place": va in ("place", "put") and vb in ("place", "put"),
                                 "same_type": va == vb and va != "", "same_set": us[i][0] == us[j][0]})
    pp = res["pairs"]
    pl = [p["cca_r"] for p in pp if p["both_place"]]
    ot = [p["cca_r"] for p in pp if not p["both_place"] and not p["same_type"]]
    res["cca_place_pairs_median"] = float(np.nanmedian(pl)) if pl else None
    res["cca_mixed_pairs_median"] = float(np.nanmedian(ot)) if ot else None
    T.add("cca_place_gt_mixed", mwu(pl, ot))
    # basket: same task, near vs other condition
    bk = defaultdict(list)
    for p in pp:
        sa, sb = p["a"].split("|"), p["b"].split("|")
        if sa[0].startswith("basket") and sb[0].startswith("basket") and sa[2] == sb[2]:
            conds = {sa[1], sb[1]}
            if "near" in conds or "ctl4090" in conds:
                other = (conds - {"near", "ctl4090"}) or conds
                bk[sorted(other)[0]].append(p)
    res["basket"] = {}
    for c, ps in bk.items():
        da = [p["angle"] - min(p["self_angle_a"], p["self_angle_b"]) for p in ps]
        res["basket"][c] = {"n_tasks": len(ps), "angle_minus_selfangle_median": float(np.median(da)),
                            "cca_r_median": float(np.nanmedian([p["cca_r"] for p in ps]))}
        T.add(f"basket_{c}_angle_gt_self", wil(da))
    res.update(T.finish())
    save("a5", res)
    plot_a5(res)
    return res


def plot_a5(res):
    p = plt()
    pp = res["pairs"]
    fig, ax = p.subplots(1, 2, figsize=(12, 4.5))
    col = ["#1b3f8b" if q["both_place"] else ("#7aa6ef" if q["same_type"] else "#d9822b") for q in pp]
    ax[0].scatter([q["angle"] for q in pp], [q["cca_r"] for q in pp], c=col, s=10, alpha=0.7)
    ax[0].axvline(res["random_angle_median"], color="k", ls="--", lw=1, label="random subspace angle")
    ax[0].set_xlabel("mean principal angle between task manifolds (deg)")
    ax[0].set_ylabel("held-out CCA r of latent dynamics")
    ax[0].set_title("A5 Gallego: blue = both pick-and-place, orange = different types")
    ax[0].legend(fontsize=8)
    b = res.get("basket", {})
    ks = sorted(b)
    ax[1].bar(range(len(ks)), [b[k]["angle_minus_selfangle_median"] for k in ks], color="#d9822b", label="angle − split-half angle")
    ax2 = ax[1].twinx()
    ax2.plot(range(len(ks)), [b[k]["cca_r_median"] for k in ks], "ko-", label="CCA r")
    ax2.set_ylim(0, 1)
    ax[1].set_xticks(range(len(ks)), ks)
    ax[1].set_title("basket: vs near (same object task)")
    ax[1].legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(BO / "a5_manifold.png", dpi=130)
    p.close(fig)


# ============================================================================ A6
def _dip(x):
    import diptest

    x = np.asarray(x, float)
    if len(x) < 10:
        return float("nan"), float("nan")
    d, pv = diptest.diptest(x)
    return float(d), float(pv)


def _bic12(x):
    from sklearn.mixture import GaussianMixture

    x = np.asarray(x, float)[:, None]
    if len(x) < 10:
        return float("nan")
    b1 = GaussianMixture(1, random_state=0).fit(x).bic(x)
    b2 = GaussianMixture(2, random_state=0, n_init=3).fit(x).bic(x)
    return float(b1 - b2)  # > 0 favours 2 components


def a6_family(trials: list[dict], lab_fn, name: str, T: Tests, group_fn=None) -> dict:
    """trials: manifest rows; lab_fn(row) -> 0 (group1), 1 (group2) or None (unlabelled, projected only)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X, L, G, keep = [], [], [], []
    for r in trials:
        z = load(r, ["mn_exp_last"])
        X.append(z["mn_exp_last"].astype(np.float64))  # 4, 16, 1024
        L.append(lab_fn(r))
        G.append(group_fn(r) if group_fn else (r["variant"] if r["set"] == "goal" else r["task"]))
        keep.append(r)
    lab = np.array([np.nan if l is None else l for l in L], float)
    G = np.array(G)
    ok = np.isfinite(lab)
    if ok.sum() < 6 or len(set(lab[ok])) < 2 or len(set(G[ok])) < 2:
        return {"note": "not enough labelled trials", "n": int(ok.sum())}
    proj = np.full((len(X), 4, 16), np.nan)
    idx = np.where(ok)[0]
    for tr, te in GroupKFold(min(5, len(set(G[idx])))).split(idx, groups=G[idx]):
        tri, tei = idx[tr], idx[te]
        if len(set(lab[tri])) < 2:
            continue
        m1 = np.mean([X[i] for i in tri if lab[i] == 0], 0)  # 4, 16, 1024 -> mean over samples below
        m2 = np.mean([X[i] for i in tri if lab[i] == 1], 0)
        for s in range(4):
            cd = m2[s].mean(0) - m1[s].mean(0)
            a0, a1 = m1[s].mean(0) @ cd, m2[s].mean(0) @ cd
            for i in tei:
                proj[i, s] = (X[i][s] @ cd - a0) / (a1 - a0)
    pv = proj[ok]
    flat = pv[np.isfinite(pv)]
    endpoint = float(np.mean((np.abs(flat) < 0.25) | (np.abs(flat - 1) < 0.25))) if len(flat) else np.nan
    middle = float(np.mean((flat > 0.25) & (flat < 0.75))) if len(flat) else np.nan
    within_sd = float(np.nanmedian(np.nanstd(pv, axis=2)))
    d, dp = _dip(flat)
    tm = np.nanmean(pv, axis=(1, 2))
    okk = np.isfinite(tm)
    auc = float(roc_auc_score(lab[ok][okk], tm[okk])) if len(set(lab[ok][okk])) == 2 else np.nan
    # permutation p for AUC (labels shuffled within groups)
    null = []
    for _ in range(2000):
        lp = lab[ok][okk].copy()
        for g in set(G[ok][okk]):
            w = np.where(G[ok][okk] == g)[0]
            lp[w] = RNG.permutation(lp[w])
        if len(set(lp)) == 2:
            null.append(roc_auc_score(lp, tm[okk]))
    p_auc = float((np.sum(np.array(null) >= auc) + 1) / (len(null) + 1)) if null else np.nan
    res = {"n_trials": int(ok.sum()), "n_group2": int(np.nansum(lab)), "dip": d, "dip_p": dp, "bic_1_minus_2": _bic12(flat),
           "share_near_endpoint": endpoint, "share_middle": middle, "within_obs_sd_median": within_sd,
           "auc_trial_mean_projection": auc, "p_auc": p_auc, "hist": np.histogram(flat, bins=np.linspace(-1, 2, 31))[0].tolist()}
    T.add(f"{name}.dip", dp)
    T.add(f"{name}.auc_gt_0.5", p_auc, auc=auc)
    return res


def stage_a6() -> dict:
    T = Tests("A6")
    rows = manifest()  # held-out rule: CD fit and evaluation are both cross-validated by group, all episodes
    res = {}
    cand = [r for r in rows if r["set"] == "goal" and r["condition"] == "cand"]
    res["cand_fallback0_new1"] = a6_family(cand, lambda r: 0 if r["outcome"] in FALLBACK else (1 if r["outcome"] in NEW else None), "cand", T)
    far = [r for r in rows if r["condition"] in ("far20", "far25")]
    res["basketfar_success0_outside1"] = a6_family(far, lambda r: 0 if r["success"] == 1 else (1 if r["outcome"] == "released outside" else None), "basket_far", T)
    ls = [r for r in rows if r["condition"] == "lscene"]
    res["lscene_trained0_nothing1"] = a6_family(ls, lambda r: 0 if r["outcome"] == "trained" else (1 if r["outcome"] == "nothing" else None), "lscene", T)
    # baseline: two trained libero3 tasks per suite (first two task ids present), trained instructions
    base = {}
    for cond in ("spatial", "object", "long"):
        rr = [r for r in rows if r["set"] == "libero3" and r["condition"] == cond]
        ts = sorted({r["task"] for r in rr})[:2]
        if len(ts) == 2:
            sub = [r for r in rr if r["task"] in ts]
            base[cond] = a6_family(sub, lambda r, ts=ts: ts.index(r["task"]), f"baseline_{cond}", T, group_fn=lambda r: r["init"])
    res["baseline_two_trained_tasks"] = base
    res["limitation"] = "only final chunks are stored per seed, so the noise-difference shrinkage along the denoise path is not measurable"
    res.update(T.finish())
    save("a6", res)
    plot_a6(res)
    return res


def plot_a6(res):
    p = plt()
    fams = [k for k in ("cand_fallback0_new1", "basketfar_success0_outside1", "lscene_trained0_nothing1") if "hist" in res.get(k, {})]
    fams += [f"baseline_two_trained_tasks.{k}" for k, v in res.get("baseline_two_trained_tasks", {}).items() if "hist" in v]
    if not fams:
        return
    fig, ax = p.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.6), squeeze=False)
    edges = np.linspace(-1, 2, 31)
    for i, f in enumerate(fams):
        v = res[f] if "." not in f else res["baseline_two_trained_tasks"][f.split(".")[1]]
        ax[0, i].bar(edges[:-1], v["hist"], width=0.1, align="edge", color="#1b3f8b")
        for e in (0, 1):
            ax[0, i].axvline(e, color="k", ls="--", lw=0.8)
        ax[0, i].set_title(f"{f}\ndip p={v['dip_p']:.2g}, AUC={v['auc_trial_mean_projection']:.2f}", fontsize=8)
        ax[0, i].set_xlabel("projection on coding direction (0 = group 1, 1 = group 2)")
    ax[0, 0].set_ylabel("noise samples (16 per probe)")
    fig.tight_layout()
    fig.savefig(BO / "a6_multinoise.png", dpi=130)
    p.close(fig)


# ============================================================================ A7
def stage_a7(nb: int = 20) -> dict:
    from scipy.optimize import nnls

    T = Tests("A7")
    rows = manifest()
    ro = roles()
    train = defaultdict(list)
    for r in rows:
        if r["set"] == "goal" and r["condition"] == "train":
            train[r["task"]].append(r)
    if not train:
        res = {"note": "no goal-train episodes"}
        save("a7", res)
        return res
    cent, cdA = {}, {}
    allst = []
    for t, rr in train.items():
        b = [_bin(load(r, ["expmean17"])["expmean17"][:, 2].astype(np.float64), nb) for r in rr]
        cent[t] = np.mean(b, 0)
        cdA[t] = np.mean(b[0::2], 0)
        allst.append(np.concatenate(b))
    gm = np.concatenate(allst).mean(0)
    tasks = sorted(cent)

    def srcs(rl):
        out = {}
        for col, key in (("base_task", "base"), ("object_source", "object"), ("destination_source", "destination"), ("action_source", "action")):
            for x in (rl.get(col) or "").split(";"):
                if x and f"G{x}" in cent:
                    out.setdefault(key, f"G{x}")
        for col in ("A_tasks", "B_tasks"):
            for x in (rl.get(col) or "").split(";"):
                if x and f"G{x}" in cent:
                    out.setdefault(col, f"G{x}")
        return out

    recs = []
    for r in rows:
        if not (r["set"] == "goal" and r["condition"] == "cand"):
            continue
        rl = ro.get(r["variant"], {})
        s = srcs(rl)
        uniq = sorted(set(s.values()))
        if len(uniq) < 1:
            continue
        e = load(r, ["expmean17"])["expmean17"][:, 2].astype(np.float64)
        C = len(e)
        bins = np.clip(np.round(np.linspace(0, nb - 1, C)).astype(int), 0, nb - 1)
        Wt, r2 = [], []
        for c in range(C):
            A = np.stack([cent[t][bins[c]] - gm for t in uniq], 1)
            y = e[c] - gm
            w, res_ = nnls(A, y)
            Wt.append(w)
            r2.append(1 - res_**2 / (y @ y))
        Wt = np.array(Wt)
        # random non-source baseline
        others = [t for t in tasks if t not in uniq]
        rr2 = []
        for _ in range(100 if len(others) >= len(uniq) else 0):
            pick = list(RNG.choice(others, len(uniq), replace=False))
            q = []
            for c in range(C):
                A = np.stack([cent[t][bins[c]] - gm for t in pick], 1)
                y = e[c] - gm
                q.append(1 - nnls(A, y)[1] ** 2 / (y @ y))
            rr2.append(np.mean(q))
        rec = {"variant": r["variant"], "outcome": r["outcome"], "init": r["init"], "sources": s, "r2_mean": float(np.mean(r2)),
               "r2_random_mean": float(np.mean(rr2)) if rr2 else np.nan, "weights": Wt.tolist(), "uniq": uniq}
        tau = np.linspace(0, 1, C)
        if "object" in s and "destination" in s and s["object"] != s["destination"]:
            wo, wd = Wt[:, uniq.index(s["object"])], Wt[:, uniq.index(s["destination"])]
            rec["slope_dest_minus_obj"] = float(np.polyfit(tau, wd - wo, 1)[0]) if C > 2 else np.nan
        if "base" in s:
            tot = Wt.sum(1) + 1e-12
            rec["base_share"] = float(np.mean(Wt[:, uniq.index(s["base"])] / tot))
        if "A_tasks" in s and "B_tasks" in s and s["A_tasks"] != s["B_tasks"]:
            cd = cdA[s["A_tasks"]] - cdA[s["B_tasks"]]  # nb x D, fit on half of the train inits
            pr = [((e[c] - cdA[s["B_tasks"]][bins[c]]) @ cd[bins[c]]) / (cd[bins[c]] @ cd[bins[c]]) for c in range(C)]
            rec["kim_cd_proj"] = [float(x) for x in pr]
        recs.append(rec)
    res = {"n_trials": len(recs)}
    d = np.array([r["r2_mean"] - r["r2_random_mean"] for r in recs])
    byc = defaultdict(list)
    for r, x in zip(recs, d):
        byc[r["variant"]].append(x)
    cm = np.array([np.nanmean(v) for v in byc.values()])
    res["r2_source_minus_random_median_by_cand"] = float(np.nanmedian(cm)) if len(cm) else None
    res["r2_source_median"] = float(np.median([r["r2_mean"] for r in recs])) if recs else None
    T.add("r2_source_gt_random", wil(cm))
    sel = [r for r in recs if r["outcome"] in FALLBACK + NEW]
    lab = np.array([r["outcome"] in NEW for r in sel])
    for k in ("slope_dest_minus_obj", "base_share"):
        v = np.array([r.get(k, np.nan) for r in sel], float)
        diff, p = strat_perm(v, lab, [r["variant"] for r in sel])
        res[f"{k}_new_minus_fallback"] = diff
        res[f"{k}_new_median"] = float(np.nanmedian(v[lab])) if np.isfinite(v[lab]).any() else None
        res[f"{k}_fallback_median"] = float(np.nanmedian(v[~lab])) if np.isfinite(v[~lab]).any() else None
        T.add(f"{k}_new_vs_fallback", p, diff=diff)
    res["per_trial"] = recs
    res.update(T.finish())
    save("a7", res)
    plot_a7(recs)
    return res


def plot_a7(recs):
    p = plt()
    fig, ax = p.subplots(1, 2, figsize=(12, 4))
    for grp, col in ((NEW, "#1b3f8b"), (FALLBACK, "#c0392b")):
        cur = []
        for r in recs:
            if r["outcome"] in grp and "object" in r["sources"] and "destination" in r["sources"] and r["sources"]["object"] != r["sources"]["destination"]:
                W = np.array(r["weights"])
                wo, wd = W[:, r["uniq"].index(r["sources"]["object"])], W[:, r["uniq"].index(r["sources"]["destination"])]
                tb = np.linspace(0, 1, len(W))
                cur.append(np.interp(np.linspace(0, 1, 20), tb, wd - wo))
        if cur:
            m = np.mean(cur, 0)
            ax[0].plot(np.linspace(0, 1, 20), m, color=col, label=f"{'NEW' if grp == NEW else 'FALLBACK'} (n={len(cur)})")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("normalised episode time")
    ax[0].set_ylabel("w(destination source) − w(object source)")
    ax[0].set_title("A7 NNLS weights on source-task states (expert L17)")
    ax[0].legend(fontsize=8)
    ax[1].hist([r["r2_mean"] - r["r2_random_mean"] for r in recs if np.isfinite(r["r2_random_mean"])], bins=20, color="#1b3f8b")
    ax[1].axvline(0, color="k")
    ax[1].set_xlabel("R² (true sources) − R² (random non-source tasks)")
    ax[1].set_title("source specificity")
    fig.tight_layout()
    fig.savefig(BO / "a7_combinatorial.png", dpi=130)
    p.close(fig)


# ============================================================================ main
STAGES = {"extract": stage_extract, "pairs": stage_pairs, "b1": stage_b1, "a2": stage_a2, "a3": stage_a3,
          "a4": stage_a4, "a5": stage_a5, "a6": stage_a6, "a7": stage_a7}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=list(STAGES) + ["all"])
    ap.add_argument("--out", default="b", help="output subdir of analysis/neuro_dyn (use b_pilot for the 60-ep pilot)")
    a = ap.parse_args()
    global BO
    BO = OUT / a.out
    todo = list(STAGES) if "all" in a.stages else a.stages
    for s in todo:
        print(f"== {s}", flush=True)
        STAGES[s]()


if __name__ == "__main__":
    main()
