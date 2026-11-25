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

import argparse
import csv
import json
import os
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


# ============================================================================ main
STAGES: dict = {"extract": stage_extract}


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
