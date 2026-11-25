#!/usr/bin/env python
"""T031 stage 2 feature analysis (CPU): T028 1 560 rollout episodes encoded with the stage-2 action-expert SAEs.

  OPENBLAS_CORETYPE=SkylakeX .venv-sae/bin/python ae_analysis.py <stage|all>

stages (PREREG = ae_PREREG.md):
  extract   T028 npz → cache: exp_tok_pc step 9 (6 layers), first executed action per call
  encode    18 mean SAEs on T028 exp_mean (codes + T028 quality); 6 per-token SAEs on the PCA-128 token reconstruction
  select    A2 phase / task selectivity, action correlation (all layers × steps)
  fallback  A3 executed vs instructed task features, timing from call 0
  basket    A4 basket-position features: graded vs on/off
  kauf      A5 output-potent fraction of decoder directions
  dec       A6 decoding: T030 e_s9 vector vs SAE codes (task, phase, fallback executed vs instructed)
  traj      A1 trajectories (figures)
  numbers   numbers.json + summary figures
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ae_common as C  # noqa: E402

S2 = Path(os.environ.get("T031_S2", C.OUT / "stage2"))
AN = Path(os.environ.get("T031_AN", C.OUT / "analysis"))
CACHE = C.DATA / "t028_cache"  # T028 extracts (exp_tok9, act0)
CODES = Path(os.environ.get("T031_CODES", CACHE))  # SAE codes and per-stage pickles
FIG = AN / "figs"
STEPS = (0, 4, 9)
PHASES = ("reach", "grasp", "carry", "release")
ACT_DIMS = ("x", "y", "z", "rx", "ry", "rz", "gripper")


def sae_dir(layer: int, step: int, tok: bool = False) -> Path:
    return S2 / (f"L{layer:02d}_s{step}" + ("_tok" if tok else ""))


def save(name: str, obj) -> None:
    AN.mkdir(parents=True, exist_ok=True)
    (AN / f"{name}.json").write_text(C.dumps(obj))


def load(name: str):
    return json.loads((AN / f"{name}.json").read_text())


# ============================================================================ extract
def _extract_one(m: dict) -> tuple[int, np.ndarray, np.ndarray]:
    z = np.load(C.RICH / m["path"])
    tok = z["exp_tok_pc"][:, :, 2]  # (C, 6, 50, 128) step 9
    act = z["act_pred"][:, 0]  # (C, 7) first executed action of the chunk
    assert tok.shape[0] == m["C"]
    return m["i"], tok.astype(np.float16), act.astype(np.float32)


def stage_extract() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "done.txt").exists():
        return
    M = C.meta()
    tok = np.lib.format.open_memmap(CACHE / "exp_tok9.npy", "w+", np.float16, (6, C.N_TOT, 50, 128))
    act = np.zeros((C.N_TOT, 7), np.float32)
    with ProcessPoolExecutor(C.WORKERS) as ex:
        for n, (i, t, a) in enumerate(ex.map(_extract_one, M, chunksize=8)):
            m = M[i]
            tok[:, m["off"]: m["off"] + m["C"]] = t.transpose(1, 0, 2, 3)
            act[m["off"]: m["off"] + m["C"]] = a
            if n % 200 == 0:
                print(f"extract {n}/{len(M)}", flush=True)
    tok.flush()
    np.save(CACHE / "act0.npy", act)
    (CACHE / "done.txt").write_text("ok\n")


# ============================================================================ encode
def codes(layer: int, step: int, tok: bool = False) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(CODES / f"codes_L{layer:02d}_s{step}{'_tok' if tok else ''}.npz")
    return z["idx"], z["val"]


def dense(layer: int, step: int, tok: bool = False) -> np.ndarray:
    idx, val = codes(layer, step, tok)
    n = json.loads((sae_dir(layer, step, tok) / "config.json").read_text())["n_latents"]
    return C.dense_codes(idx, val, n)


def stage_encode() -> None:
    import torch
    from model import TopKSAE

    torch.set_num_threads(C.WORKERS)
    q = {}
    CODES.mkdir(parents=True, exist_ok=True)
    for layer in C.LAYERS:
        for step in STEPS:
            f = CODES / f"codes_L{layer:02d}_s{step}.npz"
            sae = TopKSAE.load(sae_dir(layer, step), "cpu")
            x = torch.from_numpy(C.exp_mean(layer, step))
            met, fire = C.evaluate(sae, x)
            q[f"L{layer}_s{step}"] = met | {"live_on_t028": int((fire > 0).sum())}
            if not f.exists():
                idx, val = C.encode_np(sae, x.numpy(), "cpu")
                np.savez(f, idx=idx, val=val.astype(np.float16))
            print(f"encode L{layer} s{step}: T028 VE {met['var_explained']:.4f} nmse {met['nmse']:.4f}", flush=True)
        # per-token SAE on the PCA-128 reconstruction (T028 basis exp_tok/L*), step 9
        pb = np.load(C.RICH / "pca_bases.npz")
        comp, mean = pb[f"exp_tok/L{layer}/components"], pb[f"exp_tok/L{layer}/mean"]
        trunc = 1 - float(pb[f"exp_tok/L{layer}/explained_variance"].sum() / pb[f"exp_tok/L{layer}/total_variance"])
        sae = TopKSAE.load(sae_dir(layer, 9, True), "cpu")
        tk = np.load(CACHE / "exp_tok9.npy", mmap_mode="r")[C.LAYERS.index(layer)]  # (N, 50, 128)
        n = sae.cfg.n_latents
        call_mean = np.zeros((C.N_TOT, n), np.float16)  # per call, feature activation averaged over the 50 tokens
        pos_fire = np.zeros((C.N_TOT, 5, n), np.uint8)  # per call, # tokens firing in token bands of 10
        sq = var = 0.0
        acc_sum = np.zeros(1024)
        nn_ = 0
        for i in range(0, C.N_TOT, 1024):
            xr = (tk[i: i + 1024].astype(np.float32) @ comp + mean).reshape(-1, 1024)
            xt = torch.from_numpy(xr)
            with torch.no_grad():
                xh = sae.reconstruct(xt).numpy()
                v, j = sae.encode(xt)
            sq += float(((xr - xh) ** 2).sum())
            acc_sum += xr.sum(0)
            var += float((xr ** 2).sum())
            nn_ += len(xr)
            d = C.dense_codes(j.numpy().astype(np.int16), v.numpy(), n).reshape(-1, 50, n)
            call_mean[i: i + len(d)] = d.mean(1)
            pos_fire[i: i + len(d)] = (d > 0).reshape(len(d), 5, 10, n).sum(2)
        var -= float((acc_sum ** 2).sum() / nn_)
        np.save(CODES / f"tokmean_L{layer:02d}.npy", call_mean)
        np.save(CODES / f"tokpos_L{layer:02d}.npy", pos_fire)
        q[f"L{layer}_s9_tok"] = {"pca128_unexplained_t028_fit": trunc, "sae_ve_on_pca_reconstruction": 1 - sq / var}
        print(f"encode L{layer} tok: PCA trunc {trunc:.4f}, SAE VE on recon {1 - sq / var:.4f}", flush=True)
    save("t028_quality", q)


# ============================================================================ A2 selectivity
def selectivity(fires: np.ndarray, M: list[dict]) -> dict:
    U = C.u40(M)
    ph_rows, task_rows = defaultdict(list), defaultdict(list)
    for m in U:
        for c, p in enumerate(C.phase_labels(m)):
            if p is not None:
                ph_rows[p].append(m["off"] + c)
        task_rows[m["task"]] += list(range(m["off"], m["off"] + m["C"]))
    P = np.stack([fires[ph_rows[p]].mean(0) for p in PHASES])
    tasks = sorted(task_rows, key=lambda t: (t[0], int(t[1:])))
    T = np.stack([fires[task_rows[t]].mean(0) for t in tasks])
    urows = np.concatenate([np.arange(m["off"], m["off"] + m["C"]) for m in U])
    live = fires[urows].any(0)
    pbest, bp = P.max(0), P.argmax(0)
    pother = (P.sum(0) - pbest) / 3
    ph_sel = live & (pbest >= 0.5) & (pbest >= 3 * pother)
    tbest, bt = T.max(0), T.argmax(0)
    tsecond = np.sort(T, 0)[-2]
    t_sel = live & (tbest >= 0.5) & (tsecond <= 0.05)
    return {"P": P, "T": T, "tasks": tasks, "live": live, "ph_sel": ph_sel, "t_sel": t_sel, "bp": bp, "bt": bt,
            "urows": urows}


def act_corr(D: np.ndarray, act: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Pearson r (n_feat, 7) between feature activation and first action dims over rows."""
    X = D[rows].astype(np.float64)
    Y = act[rows].astype(np.float64)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    sx = np.sqrt((X ** 2).sum(0))
    sy = np.sqrt((Y ** 2).sum(0))
    r = (X.T @ Y) / np.maximum(np.outer(sx, sy), 1e-12)
    r[sx < 1e-9] = 0
    return r


def stage_select() -> None:
    M = C.meta()
    act = np.load(CACHE / "act0.npy")
    out, keep = {}, {}
    for layer in C.LAYERS:
        for step in STEPS:
            D = dense(layer, step)
            s = selectivity(D > 0, M)
            r = act_corr(D, act, s["urows"])
            key = f"L{layer}_s{step}"
            nl = int(s["live"].sum())
            out[key] = {
                "live": nl, "phase_selective": int(s["ph_sel"].sum()), "task_selective": int(s["t_sel"].sum()),
                "phase_share": s["ph_sel"].sum() / max(nl, 1), "task_share": s["t_sel"].sum() / max(nl, 1),
                "phase_by_best": dict(Counter(PHASES[i] for i in s["bp"][s["ph_sel"]])),
                "tasks_covered": len({s["tasks"][i] for i in s["bt"][s["t_sel"]]}),
                "max_abs_r": {d: float(np.abs(r[s["live"], j]).max()) for j, d in enumerate(ACT_DIMS)},
                "argmax_r": {d: int(np.flatnonzero(s["live"])[np.abs(r[s["live"], j]).argmax()]) for j, d in enumerate(ACT_DIMS)},
                "n_r_ge_0.5": {d: int((np.abs(r[s["live"], j]) >= 0.5).sum()) for j, d in enumerate(ACT_DIMS)},
            }
            keep[key] = {"live": s["live"], "ph_sel": s["ph_sel"], "t_sel": s["t_sel"], "bt": s["bt"], "bp": s["bp"],
                         "tasks": s["tasks"], "P": s["P"], "T": s["T"], "r": r.astype(np.float32)}
            print(key, {k: out[key][k] for k in ("live", "phase_selective", "task_selective")},
                  "gripper |r|", round(out[key]["max_abs_r"]["gripper"], 3), flush=True)
    # per-token SAE (call mean over the 50 reconstructed tokens): position profile
    for layer in C.LAYERS:
        pf = np.load(CODES / f"tokpos_L{layer:02d}.npy", mmap_mode="r")
        U = C.u40(M)
        urows = np.concatenate([np.arange(m["off"], m["off"] + m["C"]) for m in U])
        band = np.asarray(pf[urows], dtype=np.float32).mean(0) / 10  # (5, n) fire prob per token band
        live = band.max(0) > 0
        early, late = band[0], band[4]
        out[f"L{layer}_s9_tok"] = {
            "live": int(live.sum()),
            "early_2x": int((live & (early >= 2 * late) & (early > 0.01)).sum()),
            "late_2x": int((live & (late >= 2 * early) & (late > 0.01)).sum()),
            "band_fire_mean": band[:, live].mean(1).tolist(),
        }
        D = np.load(CODES / f"tokmean_L{layer:02d}.npy").astype(np.float32)
        s = selectivity(D > 0, M)
        out[f"L{layer}_s9_tok"] |= {"phase_selective_callmean": int(s["ph_sel"].sum()),
                                    "task_selective_callmean": int(s["t_sel"].sum())}
    save("select", out)
    pickle.dump(keep, open(CODES / "select.pkl", "wb"))
    # H2a, H2b
    sh = {L: out[f"L{L}_s9"]["phase_share"] for L in C.LAYERS}
    save("h2", {"phase_share_s9": sh, "H2a_late_gt_early": bool(min(sh[14], sh[17]) > max(sh[2], sh[5])),
                "H2a_mean_late_minus_early": (sh[14] + sh[17] - sh[2] - sh[5]) / 2,
                "H2b_L17_max_abs_r_gripper": out["L17_s9"]["max_abs_r"]["gripper"],
                "H2b": bool(out["L17_s9"]["max_abs_r"]["gripper"] >= 0.7)})


# ============================================================================ A3 fallback
def fallback_pool(M: list[dict]) -> list[dict]:
    """T030 labels() pool definitions."""
    pool = []
    for m in M:
        if m["t027_set"] == "cand" and m.get("exec_other"):
            pool.append({"m": m, "src": "cand", "executed": list(m["exec_other"]), "instructed": m["base"]})
        elif (m["t027_set"] == "l90" and m.get("kind") == "lscene" and m["outcome"] == "trained" and m.get("exec_L")
              and m["nearest_trained"] not in m["exec_L"]):
            pool.append({"m": m, "src": "l90", "executed": list(m["exec_L"]), "instructed": m["nearest_trained"]})
    return pool


def stage_fallback() -> None:
    M = C.meta()
    sel = pickle.load(open(CODES / "select.pkl", "rb"))
    pool = fallback_pool(M)
    out = {"n_pool": len(pool), "by_src": dict(Counter(p["src"] for p in pool))}
    curves = {}
    for layer in C.LAYERS:
        key = f"L{layer}_s9"
        s = sel[key]
        feats_of = defaultdict(list)
        for f in np.flatnonzero(s["t_sel"]):
            feats_of[s["tasks"][s["bt"][f]]].append(int(f))
        F = dense(layer, 9) > 0
        E_c, I_c, rows = [], [], []
        n_used = 0
        for p in pool:
            fe = sorted({f for t in p["executed"] for f in feats_of.get(t, [])})
            fi = feats_of.get(p["instructed"], [])
            if not fe or not fi:
                continue
            n_used += 1
            m = p["m"]
            blk = F[m["off"]: m["off"] + m["C"]]
            e = blk[:, fe].any(1)
            i = blk[:, fi].any(1)
            E_c.append(np.pad(e.astype(float), (0, 60 - m["C"]), constant_values=np.nan))
            I_c.append(np.pad(i.astype(float), (0, 60 - m["C"]), constant_values=np.nan))
            rows.append({"episode": m["episode"], "src": p["src"], "first_exec_call": int(np.argmax(e)) if e.any() else None,
                         "first_instr_call": int(np.argmax(i)) if i.any() else None,
                         "exec_by2": bool(e[:3].any()), "instr_by2": bool(i[:3].any()),
                         "exec_any": bool(e.any()), "instr_any": bool(i.any())})
        E_c, I_c = np.array(E_c), np.array(I_c)
        fr = lambda k: float(np.mean([r[k] for r in rows])) if rows else None  # noqa: E731
        res = {"n_trials_with_features": n_used, "exec_by_call2": fr("exec_by2"), "instr_by_call2": fr("instr_by2"),
               "exec_any_call": fr("exec_any"), "instr_any_call": fr("instr_any"),
               "median_first_exec_call": float(np.median([r["first_exec_call"] for r in rows if r["first_exec_call"] is not None])) if rows else None,
               "by_src": {src: {"n": sum(r["src"] == src for r in rows),
                                "exec_by_call2": float(np.mean([r["exec_by2"] for r in rows if r["src"] == src] or [np.nan])),
                                "instr_by_call2": float(np.mean([r["instr_by2"] for r in rows if r["src"] == src] or [np.nan]))}
                          for src in ("cand", "l90")}}
        res["H3"] = bool(res["exec_by_call2"] is not None and res["exec_by_call2"] >= 0.5 and res["instr_by_call2"] <= 0.1)
        out[key] = res
        curves[key] = {"exec": np.nanmean(E_c, 0)[:30].tolist() if len(E_c) else [],
                       "instr": np.nanmean(I_c, 0)[:30].tolist() if len(I_c) else []}
        print(key, {k: res[k] for k in ("n_trials_with_features", "exec_by_call2", "instr_by_call2", "H3")}, flush=True)
    out["H3_all_layers"] = all(out[f"L{L}_s9"]["H3"] for L in C.LAYERS)
    save("fallback", out | {"curves": curves})


# ============================================================================ A4 basket
DIST = {"ctl": 0.0, "near": 10.0, "far20": 20.6, "far25": 25.7}


def basket_groups(M: list[dict]) -> dict:
    g = defaultdict(list)
    for m in M:
        if m["set"] == "libero3" and m["condition"] == "object":
            g["ctlA"].append(m)
        elif m["set"] == "basket_a100":
            g[m["condition"]].append(m)
        elif m["set"] == "basket_far":
            g["ctlV" if m["condition"] == "ctl4090" else m["condition"]].append(m)
    return g


def ep_mean(D: np.ndarray, m: dict) -> np.ndarray:
    gc = m["gc"] if m["gc"] is not None else m.get("grasp_call")
    a = m["off"] + (gc if gc is not None else 0)
    return D[a: m["off"] + m["C"]].mean(0)


def auroc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """per column P(b > a) (+ ties/2); a (na, n), b (nb, n)."""
    from scipy.stats import rankdata

    X = np.vstack([a, b])
    R = rankdata(X, axis=0)
    rb = R[len(a):].sum(0)
    return (rb - len(b) * (len(b) + 1) / 2) / (len(a) * len(b))


def stage_basket() -> None:
    M = C.meta()
    G = basket_groups(M)
    sel = pickle.load(open(CODES / "select.pkl", "rb"))
    out = {"n": {k: len(v) for k, v in G.items()}}
    keep = {}
    for layer in C.LAYERS:
        key = f"L{layer}_s9"
        D = dense(layer, 9)
        E = {k: np.stack([ep_mean(D, m) for m in v]) for k, v in G.items()}
        live = sel[key]["live"]
        au = {"near": auroc(E["ctlA"], E["near"]), "opp": auroc(E["ctlA"], E["opp"]),
              "far20": auroc(E["ctlV"], E["far20"]), "far25": auroc(E["ctlV"], E["far25"])}
        sens_line = live & np.any([np.abs(au[c] - 0.5) >= 0.3 for c in ("near", "far20", "far25")], axis=0)
        sens_opp = live & (np.abs(au["opp"] - 0.5) >= 0.3)
        ctl = np.vstack([E["ctlA"], E["ctlV"]]).mean(0)
        means = np.stack([ctl, E["near"].mean(0), E["far20"].mean(0), E["far25"].mean(0)])  # by distance
        d = np.diff(means, axis=0)
        mono = (d > 0).all(0) | (d < 0).all(0)
        graded = sens_line & mono
        onoff = sens_line & ~mono
        # strength of the distance relation: Spearman over episodes (ctl both machines, near, far20, far25)
        from scipy.stats import spearmanr

        X = np.vstack([E["ctlA"], E["ctlV"], E["near"], E["far20"], E["far25"]])
        dist = np.r_[np.zeros(len(E["ctlA"]) + len(E["ctlV"])), np.full(len(E["near"]), 10.0),
                     np.full(len(E["far20"]), 20.6), np.full(len(E["far25"]), 25.7)]
        rho = np.array([spearmanr(dist, X[:, f]).statistic if X[:, f].std() > 0 else 0.0 for f in range(X.shape[1])])
        out[key] = {"live": int(live.sum()), "sensitive_distance_line": int(sens_line.sum()), "graded": int(graded.sum()),
                    "onoff": int(onoff.sum()), "sensitive_opp": int(sens_opp.sum()),
                    "opp_only": int((sens_opp & ~sens_line).sum()),
                    "graded_share_of_sensitive": float(graded.sum() / max(sens_line.sum(), 1)),
                    "median_abs_rho_graded": float(np.median(np.abs(rho[graded]))) if graded.any() else None,
                    "median_abs_rho_onoff": float(np.median(np.abs(rho[onoff]))) if onoff.any() else None,
                    "top_graded": [int(f) for f in np.flatnonzero(graded)[np.argsort(-np.abs(rho[graded]))][:5]],
                    "top_onoff": [int(f) for f in np.flatnonzero(onoff)[np.argsort(-np.abs(au['far25'][onoff] - 0.5))][:5]]}
        keep[key] = {"means": means, "opp": E["opp"].mean(0), "ctlA": E["ctlA"].mean(0), "graded": graded, "onoff": onoff,
                     "rho": rho, "au": au}
        print(key, {k: out[key][k] for k in ("sensitive_distance_line", "graded", "onoff", "sensitive_opp")}, flush=True)
    save("basket", out)
    pickle.dump(keep, open(CODES / "basket.pkl", "wb"))


# ============================================================================ A5 Kaufman
def stage_kauf() -> None:
    from scipy.stats import mannwhitneyu

    W = np.load(C.RICH / "weights.npz")["action_out_proj_w"].astype(np.float64)  # (32, 1024)
    Q, _ = np.linalg.qr(W.T)  # (1024, 32) orthonormal basis of the row space
    sel = pickle.load(open(CODES / "select.pkl", "rb"))
    from safetensors.numpy import load_file

    out, keep = {"random_expectation": W.shape[0] / W.shape[1]}, {}
    for layer in C.LAYERS:
        key = f"L{layer}_s9"
        Wd = load_file(str(sae_dir(layer, 9) / "sae.safetensors"))["W_dec"].astype(np.float64)  # (1024, n)
        pot = ((Q.T @ Wd) ** 2).sum(0) / (Wd ** 2).sum(0)
        s = sel[key]
        beh = s["live"] & (s["ph_sel"] | s["t_sel"] | (np.abs(s["r"]) >= 0.5).any(1))
        oth = s["live"] & ~beh
        p = float(mannwhitneyu(pot[beh], pot[oth], alternative="greater").pvalue) if beh.sum() and oth.sum() else None
        sub = {}
        for nm, msk in (("phase", s["ph_sel"]), ("task", s["t_sel"]), ("action_r05", s["live"] & (np.abs(s["r"]) >= 0.5).any(1)),
                        ("gripper_r05", s["live"] & (np.abs(s["r"][:, 6]) >= 0.5))):
            sub[nm] = {"n": int(msk.sum()), "median": float(np.median(pot[msk])) if msk.any() else None}
        out[key] = {"n_behaviour": int(beh.sum()), "n_other": int(oth.sum()),
                    "median_behaviour": float(np.median(pot[beh])) if beh.any() else None,
                    "median_other": float(np.median(pot[oth])) if oth.any() else None,
                    "median_all_live": float(np.median(pot[s["live"]])), "p_one_sided": p, "by_kind": sub}
        keep[key] = {"pot": pot, "beh": beh, "oth": oth}
        print(key, {k: out[key][k] for k in ("n_behaviour", "median_behaviour", "median_other", "p_one_sided")}, flush=True)
    out["H5"] = bool(out["L17_s9"]["p_one_sided"] is not None and out["L17_s9"]["p_one_sided"] < 0.05)
    save("kauf", out)
    pickle.dump(keep, open(CODES / "kauf.pkl", "wb"))


# ============================================================================ A6 decoding
def _std(A):
    mu, sd = A.mean(0), A.std(0)
    return mu, np.where(sd > 1e-6, sd, np.inf)


def fit_predict(Xtr, ytr, Xtes, ncomp=50):
    """T030 fit_predict: standardise → PCA-50 (train fold) → standardise → LogisticRegression(C=1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.extmath import randomized_svd

    mu, sd = _std(Xtr)
    A = (Xtr - mu) / sd
    k = min(ncomp, len(A) - 1, A.shape[1])
    V = randomized_svd(A, k, random_state=0)[2].T
    Z = A @ V
    m2, s2 = _std(Z)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit((Z - m2) / s2, ytr)
    return [clf.predict((((X - mu) / sd) @ V - m2) / s2) for X in Xtes]


def bacc(y, p):
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(list(y), list(p)))


def _rep(layer: int, kind: str) -> np.ndarray:
    if kind == "sae":
        return dense(layer, 9)
    x = C.exp_mean(layer, 9)
    M = C.meta()
    U = C.u40(M)
    ur = np.concatenate([np.arange(m["off"], m["off"] + m["C"]) for m in U])
    mu, sd = x[ur].mean(0), x[ur].std(0)
    z = (x - mu) / np.where(sd > 1e-6, sd, np.inf)
    z[:, C.massive_channels().get(f"e_s9/L{layer}", [])] = 0
    return z.astype(np.float32)


def dec_job(job):
    layer, kind = job
    os.environ["OMP_NUM_THREADS"] = "1"
    M = C.meta()
    U = C.u40(M)
    X = _rep(layer, kind)
    rows, task, init, phase = [], [], [], []
    for m in U:
        for c, p in enumerate(C.phase_labels(m)):
            rows.append(m["off"] + c)
            task.append(m["task"])
            init.append(m["init"])
            phase.append(p)
    rows, task, init, phase = map(np.array, (rows, task, init, phase))
    phase = phase.astype(object)
    res = {}
    pt = np.empty(len(rows), object)
    pp = np.empty(len(rows), object)
    hp = np.array([p is not None for p in phase])
    for g in range(10):
        tr, te = init != g, init == g
        pt[te] = fit_predict(X[rows[tr]], task[tr], [X[rows[te]]])[0]
        trp, tep = tr & hp, te & hp
        pp[tep] = fit_predict(X[rows[trp]], phase[trp].astype(str), [X[rows[tep]]])[0]
    res["task_bacc"] = bacc(task, pt)
    res["phase_bacc"] = bacc(phase[hp].astype(str), pp[hp])
    # fallback: U40-trained task decoder on fallback calls
    pool = fallback_pool(M)
    frows = np.concatenate([np.arange(p["m"]["off"], p["m"]["off"] + p["m"]["C"]) for p in pool])
    pred = fit_predict(X[rows], task, [X[frows]])[0]
    k = 0
    ex = ins = 0
    early_ex = early_ins = n_early = 0
    for p in pool:
        C_ = p["m"]["C"]
        pr = pred[k: k + C_]
        k += C_
        ex += sum(q in p["executed"] for q in pr)
        ins += sum(q == p["instructed"] for q in pr)
        early_ex += sum(q in p["executed"] for q in pr[:3])
        early_ins += sum(q == p["instructed"] for q in pr[:3])
        n_early += min(3, C_)
    res["fallback_executed"] = ex / len(frows)
    res["fallback_instructed"] = ins / len(frows)
    res["fallback_executed_calls0_2"] = early_ex / n_early
    res["fallback_instructed_calls0_2"] = early_ins / n_early
    return (layer, kind), res


def stage_dec() -> None:
    jobs = [(L, k) for L in C.LAYERS for k in ("vec", "sae")]
    out = {}
    with ProcessPoolExecutor(C.WORKERS) as ex:
        for (layer, kind), r in ex.map(dec_job, jobs):
            out.setdefault(f"L{layer}_s9", {})[kind] = r
            print(layer, kind, r, flush=True)
    save("dec", out)


# ============================================================================ A1 trajectories
def cases(M: list[dict]) -> dict:
    """T030 representative cases (layer_pop.cases, same picks)."""
    pick = lambda f: [m["i"] for m in M if f(m)]  # noqa: E731
    out = {
        "basket_O9": {
            "ctl (A100)": pick(lambda m: m["set"] == "libero3" and m["task"] == "O9"),
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
    cnt, single = Counter(), defaultdict(Counter)
    for m in M:
        if m["t027_set"] == "l90" and m.get("kind") == "lscene" and m["outcome"] == "trained" and len(m.get("exec_L") or []) == 1:
            cnt[m["task"]] += 1
            single[m["task"]][m["exec_L"][0]] += 1
    best = sorted(cnt, key=lambda t: (-cnt[t], int(t[1:])))[0]
    lt = single[best].most_common(1)[0][0]
    out[f"l90_{best}"] = {
        f"{best} lscene, did {lt}": pick(lambda m: m["t027_set"] == "l90" and m["task"] == best and m["outcome"] == "trained"
                                         and (m.get("exec_L") or []) == [lt]),
        f"{best} other outcome": pick(lambda m: m["t027_set"] == "l90" and m["task"] == best
                                      and not (m["outcome"] == "trained" and (m.get("exec_L") or []) == [lt])),
        f"{lt} (train)": pick(lambda m: m["t027_set"] == "train" and m["task"] == lt),
    }
    return out


def stage_traj() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = C.meta()
    CS = cases(M)
    FIG.mkdir(parents=True, exist_ok=True)
    out = {}
    for layer in (11, 17):
        D = dense(layer, 9)
        for cname, groups in CS.items():
            groups = {k: v for k, v in groups.items() if v}
            ncall = min(min(M[i]["C"] for i in v) for v in groups.values())
            mean = {g: np.mean([D[M[i]["off"]: M[i]["off"] + ncall] for i in v], 0) for g, v in groups.items()}
            A = np.stack(list(mean.values()))  # (G, T, n)
            score = A.var(0).sum(0)  # variance across conditions, summed over calls
            top = np.argsort(-score)[:12]
            out[f"{cname}/L{layer}"] = {"features": top.tolist(), "n_eps": {g: len(v) for g, v in groups.items()}}
            fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
            for ax, f in zip(axes.flat, top, strict=False):
                for gi, (g, v) in enumerate(groups.items()):
                    col = f"C{gi}"
                    ax.plot(mean[g][:, f], color=col, lw=1.6, label=g)
                    gcs = [M[i]["gc"] for i in v if M[i]["gc"] is not None]
                    rcs = [M[i]["release_call"] for i in v if M[i].get("release_call") is not None]
                    if gcs:
                        ax.axvline(np.median(gcs), color=col, ls="--", lw=0.8)
                    if rcs:
                        ax.axvline(np.median(rcs), color=col, ls=":", lw=0.8)
                ax.set_title(f"f{f}", fontsize=9)
            axes[0, 0].legend(fontsize=7)
            for ax in axes[-1]:
                ax.set_xlabel("call (10 env steps)")
            fig.suptitle(f"{cname}: SAE feature activation per call, expert L{layer} step 9 (mean over episodes; "
                         f"-- median grasp, : median release)", fontsize=10)
            fig.tight_layout()
            fig.savefig(FIG / f"traj_{cname}_L{layer}.png", dpi=110)
            plt.close(fig)
    save("traj", out)


# ============================================================================ numbers + summary figures
def stage_numbers() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    qual = json.loads((S2 / "quality.json").read_text())
    nums = {"stage1": {"selected": json.loads((C.OUT / "stage1" / "selected.json").read_text()),
                       "layers": json.loads((C.OUT / "stage1" / "layers.json").read_text())},
            "stage2_quality": qual}
    for nm in ("t028_quality", "select", "h2", "fallback", "basket", "kauf", "dec", "traj"):
        if (AN / f"{nm}.json").exists():
            nums[nm] = load(nm)
    (C.OUT / "numbers.json").write_text(C.dumps(nums))

    # quality figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for si, step in enumerate(STEPS):
        ve = [next(q["ve"] for q in qual if q["job"] == f"L{L:02d}_s{step}") for L in C.LAYERS]
        ax[0].plot(C.LAYERS, ve, "o-", label=f"mean, step {step}")
        nf = [next(q["never_fire"] for q in qual if q["job"] == f"L{L:02d}_s{step}") for L in C.LAYERS]
        ax[1].plot(C.LAYERS, nf, "o-", label=f"mean, step {step}")
    ve = [next(q["ve"] for q in qual if q["job"] == f"L{L:02d}_s9_tok") for L in C.LAYERS]
    ax[0].plot(C.LAYERS, ve, "s--", color="k", label="per-token, step 9")
    nf = [next(q["never_fire"] for q in qual if q["job"] == f"L{L:02d}_s9_tok") for L in C.LAYERS]
    ax[1].plot(C.LAYERS, nf, "s--", color="k", label="per-token, step 9")
    if "t028_quality" in nums:
        ax[0].plot(C.LAYERS, [nums["t028_quality"][f"L{L}_s9"]["var_explained"] for L in C.LAYERS], "^:", color="C3",
                   label="mean step 9 on T028 rollouts")
    ax[0].set_ylabel("held-out VE (raw space)")
    ax[1].set_ylabel("never-fire fraction")
    for a in ax:
        a.set_xlabel("expert layer")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "quality.png", dpi=120)
    plt.close(fig)

    if "select" in nums:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for step in STEPS:
            ax.plot(C.LAYERS, [nums["select"][f"L{L}_s{step}"]["phase_share"] for L in C.LAYERS], "o-", label=f"phase, step {step}")
        ax.plot(C.LAYERS, [nums["select"][f"L{L}_s9"]["task_share"] for L in C.LAYERS], "s--", color="k", label="task, step 9")
        ax.set_xlabel("expert layer")
        ax.set_ylabel("share of live features")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "selectivity.png", dpi=120)
        plt.close(fig)
    if "fallback" in nums:
        fig, axes = plt.subplots(1, 6, figsize=(17, 3), sharey=True)
        for ax, L in zip(axes, C.LAYERS, strict=True):
            cv = nums["fallback"]["curves"][f"L{L}_s9"]
            ax.plot(cv["exec"], color="C2", label="executed-task features")
            ax.plot(cv["instr"], color="C3", label="instructed-task features")
            ax.set_title(f"L{L}")
            ax.set_xlabel("call")
        axes[0].set_ylabel("share of fallback trials firing")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "fallback_timing.png", dpi=120)
        plt.close(fig)
    if (CODES / "basket.pkl").exists():
        bk = pickle.load(open(CODES / "basket.pkl", "rb"))
        fig, axes = plt.subplots(2, 5, figsize=(15, 5.5))
        for row, key in enumerate(("L11_s9", "L17_s9")):
            b = bk[key]
            fs = (list(np.flatnonzero(b["graded"])[np.argsort(-np.abs(b["rho"][b["graded"]]))][:3])
                  + list(np.flatnonzero(b["onoff"])[np.argsort(-np.abs(b["au"]["far25"][b["onoff"]] - 0.5))][:2]))
            for ax, f in zip(axes[row], fs, strict=False):
                ax.plot(list(DIST.values()), b["means"][:, f], "o-", color="C0")
                ax.plot([26.0], [b["opp"][f]], "x", color="C3", ms=8, label="opp")
                kind = "graded" if b["graded"][f] else "on/off"
                ax.set_title(f"{key} f{f} {kind} ρ={b['rho'][f]:.2f}", fontsize=8)
                ax.set_xlabel("basket shift (cm)")
            axes[row, 0].set_ylabel("mean activation after grasp")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "basket_features.png", dpi=120)
        plt.close(fig)
    if (CODES / "kauf.pkl").exists():
        kp = pickle.load(open(CODES / "kauf.pkl", "rb"))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        data, labels = [], []
        for L in C.LAYERS:
            k = kp[f"L{L}_s9"]
            data += [k["pot"][k["beh"]], k["pot"][k["oth"]]]
            labels += [f"L{L}\nbeh", f"L{L}\nother"]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
        ax.axhline(32 / 1024, color="k", ls="--", lw=0.8, label="random direction")
        ax.set_ylabel("output-potent fraction")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "kaufman_potent.png", dpi=120)
        plt.close(fig)
    if "dec" in nums:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        for ax, (met, title) in zip(axes, (("task_bacc", "task (40), bacc"), ("phase_bacc", "phase (4), bacc"),
                                            ("fallback_executed", "fallback: share decoded as executed")), strict=True):
            for kind, mk in (("vec", "o-"), ("sae", "s--")):
                ax.plot(C.LAYERS, [nums["dec"][f"L{L}_s9"][kind][met] for L in C.LAYERS], mk,
                        label="T030 vector" if kind == "vec" else "SAE codes")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("expert layer")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "decoding_vec_vs_sae.png", dpi=120)
        plt.close(fig)
    print("numbers written", flush=True)


STAGES = ("extract", "encode", "select", "fallback", "basket", "kauf", "dec", "traj", "numbers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=(*STAGES, "all"))
    a = ap.parse_args()
    for s in STAGES:
        if a.stage in (s, "all"):
            print(f"== {s}", flush=True)
            globals()[f"stage_{s}"]()


if __name__ == "__main__":
    main()
