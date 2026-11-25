#!/usr/bin/env python
"""T031 stage 1: quick action-expert SAEs on the T028 exp_mean (= aeNN) of 1 560 rollout episodes.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 flock <gpu0 lock> .venv-sae/bin/python ae_stage1.py all

stages: sweep (L11 s9, ER {2,4} × k {32,64}) → select (PREREG rule) → layers (6 layers × steps {0,9}, massive
channels) → examples (phase / task-selective features). Outputs: /outputs/VLA-shortcut-exps/sae_ae/stage1/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ae_common as C  # noqa: E402
import torch  # noqa: E402

S1 = C.OUT / "stage1"
STEPS = 3000
SWEEP = [(2048, 32), (2048, 64), (4096, 32), (4096, 64)]
S1_STEPS = (0, 9)


def data(layer: int, step: int, dev: str = "cuda"):
    M = C.meta()
    hold = C.episode_split(len(M))
    x = C.exp_mean(layer, step)
    tr, ho = C.call_rows(M, ~hold), C.call_rows(M, hold)
    return torch.from_numpy(x[tr]).to(dev), torch.from_numpy(x[ho]).to(dev), x


def train_one(layer: int, step: int, n: int, k: int, out: Path) -> dict:
    if (out / "metrics.json").exists():
        return json.loads((out / "metrics.json").read_text())
    xtr, xho, xall = data(layer, step)
    chans = C.massive_channels().get(f"e_s{step}/L{layer}", [])
    sae, info = C.fit_sae(xtr, n, k, STEPS, tag=f"L{layer} s{step} n{n} k{k}")
    m_tr, f_tr = C.evaluate(sae, xtr, chans)
    m_ho, f_ho = C.evaluate(sae, xho, chans)
    never = float(((f_tr + f_ho) == 0).mean())
    res = {"layer": layer, "step": step, "n_latents": n, "k": k, "train": m_tr, "heldout": m_ho,
           "never_fire_all": never, "massive_channels": chans, **{kk: v for kk, v in info.items() if kk != "log"}}
    out.mkdir(parents=True, exist_ok=True)
    sae.save(out, extra={"layer": layer, "step": step, "source": "T028 exp_mean", "train_steps": STEPS})
    np.save(out / "fire_count.npy", f_tr + f_ho)
    (out / "log.json").write_text(C.dumps(info["log"]))
    (out / "metrics.json").write_text(C.dumps(res))
    print(f"[s1] L{layer} s{step} n{n} k{k}: heldout nmse {m_ho['nmse']:.4f} VE {m_ho['var_explained']:.4f} "
          f"never {never:.3f}", flush=True)
    return res


def stage_sweep() -> list[dict]:
    return [train_one(11, 9, n, k, S1 / "sweep" / f"n{n}_k{k}") for n, k in SWEEP]


def stage_select() -> dict:
    res = stage_sweep()
    ok = [r for r in res if r["never_fire_all"] <= 0.10]
    if ok:
        best = min(r["heldout"]["nmse"] for r in ok)
        cand = [r for r in ok if r["heldout"]["nmse"] <= best + 0.01]
        pick = min(cand, key=lambda r: (r["k"], r["n_latents"]))
        why = f"{len(ok)}/4 with never-fire <= 10%; best nmse {best:.4f}; within 0.01: " + ", ".join(
            f"n{r['n_latents']}k{r['k']}" for r in cand) + "; smallest k, then n"
    else:
        pick = min(res, key=lambda r: r["never_fire_all"])
        why = "no setting with never-fire <= 10%: lowest never-fire"
    sel = {"n_latents": pick["n_latents"], "k": pick["k"], "rule": why,
           "sweep": [{kk: r[kk] for kk in ("n_latents", "k", "never_fire_all")} | {"heldout": r["heldout"]} for r in res]}
    (S1 / "selected.json").write_text(C.dumps(sel))
    print(f"[s1] selected n{sel['n_latents']} k{sel['k']}: {why}", flush=True)
    return sel


def stage_layers() -> list[dict]:
    sel = json.loads((S1 / "selected.json").read_text())
    out = []
    for layer in C.LAYERS:
        for step in S1_STEPS:
            out.append(train_one(layer, step, sel["n_latents"], sel["k"], S1 / f"L{layer:02d}_s{step}"))
    rows = [{"layer": r["layer"], "step": r["step"], "nmse": r["heldout"]["nmse"], "ve": r["heldout"]["var_explained"],
             "never_fire": r["never_fire_all"], "dead_last": r["dead_frac_last_window"],
             "massive_n": r["heldout"].get("massive_n"), "massive_share_raw": r["heldout"].get("massive_share_raw"),
             "massive_share_norm": r["heldout"].get("massive_share_norm"), "ve_massive": r["heldout"].get("ve_massive"),
             "ve_rest": r["heldout"].get("ve_rest"),
             "handled": (r["heldout"].get("massive_share_norm", 0) <= 0.25
                         and r["heldout"].get("ve_rest", 1) >= r["heldout"]["var_explained"] - 0.05)}
            for r in out]
    (S1 / "layers.json").write_text(C.dumps(rows))
    for r in rows:
        print(f"[s1] L{r['layer']:2d} s{r['step']}: nmse {r['nmse']:.4f} VE {r['ve']:.4f} never {r['never_fire']:.3f} "
              f"massive {r['massive_n']} share raw {r['massive_share_raw']:.3f} norm {r['massive_share_norm']:.3f} "
              f"VE massive {r['ve_massive']} rest {r['ve_rest']:.4f} handled {r['handled']}", flush=True)
    return rows


def selectivity(idx: np.ndarray, val: np.ndarray, n: int, M: list[dict]) -> dict:
    """U40 phase / task fire probabilities per feature."""
    U = C.u40(M)
    fires = np.zeros((C.N_TOT, n), bool)
    np.put_along_axis(fires, idx.astype(np.int64), val > 0, axis=1)
    ph_rows, task_rows = defaultdict(list), defaultdict(list)
    for m in U:
        for c, p in enumerate(C.phase_labels(m)):
            if p is not None:
                ph_rows[p].append(m["off"] + c)
        task_rows[m["task"]] += list(range(m["off"], m["off"] + m["C"]))
    phases = ["reach", "grasp", "carry", "release"]
    P = np.stack([fires[ph_rows[p]].mean(0) for p in phases])  # (4, n)
    tasks = sorted(task_rows)
    T = np.stack([fires[task_rows[t]].mean(0) for t in tasks])  # (40, n)
    live = fires.any(0)
    # phase-selective: best >= 0.5 and >= 3 x mean of the others
    bp = P.argmax(0)
    pbest = P.max(0)
    pother = (P.sum(0) - pbest) / 3
    ph_sel = live & (pbest >= 0.5) & (pbest >= 3 * pother)
    bt = T.argmax(0)
    tbest = T.max(0)
    tsecond = np.sort(T, 0)[-2]
    t_sel = live & (tbest >= 0.5) & (tsecond <= 0.05)
    return {"P": P, "T": T, "phases": phases, "tasks": tasks, "live": live, "ph_sel": ph_sel, "t_sel": t_sel,
            "ph_ratio": pbest / np.maximum(pother, 1e-3), "t_ratio": tbest / np.maximum(tsecond, 1e-3), "bp": bp, "bt": bt}


def stage_examples() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from model import TopKSAE

    M = C.meta()
    res = {}
    for layer in (11, 14, 17):
        sae = TopKSAE.load(S1 / f"L{layer:02d}_s9", "cuda")
        idx, val = C.encode_np(sae, C.exp_mean(layer, 9))
        s = selectivity(idx, val, sae.cfg.n_latents, M)
        res[layer] = (s, idx, val)
        print(f"[s1] L{layer} s9: live {int(s['live'].sum())}, phase-sel {int(s['ph_sel'].sum())}, "
              f"task-sel {int(s['t_sel'].sum())}", flush=True)
    layer = max(res, key=lambda L: int(res[L][0]["live"].sum()))
    s, idx, val = res[layer]
    ph = np.flatnonzero(s["ph_sel"])
    ph = ph[np.argsort(-s["ph_ratio"][ph])][:3]
    ts = np.flatnonzero(s["t_sel"])
    ts = ts[np.argsort(-s["t_ratio"][ts])][:3]
    ex = {"layer": layer, "step": 9,
          "counts": {L: {"live": int(r[0]["live"].sum()), "phase_selective": int(r[0]["ph_sel"].sum()),
                         "task_selective": int(r[0]["t_sel"].sum())} for L, r in res.items()},
          "phase_examples": [{"feature": int(f), "best_phase": s["phases"][s["bp"][f]],
                              "p_fire": dict(zip(s["phases"], s["P"][:, f].round(3).tolist(), strict=True))} for f in ph],
          "task_examples": [{"feature": int(f), "task": s["tasks"][s["bt"][f]], "p_best": float(s["T"][s["bt"][f], f]),
                             "p_second": float(np.sort(s["T"][:, f])[-2])} for f in ts]}
    (S1 / "examples.json").write_text(C.dumps(ex))
    # figure: activation over calls for the example features, U40 episodes of the relevant task / all
    feats = [int(f) for f in ph] + [int(f) for f in ts]
    dense = np.zeros((C.N_TOT, len(feats)), np.float32)
    for j, f in enumerate(feats):
        hit = idx == f
        dense[:, j] = (val * hit).sum(1)
    U = C.u40(M)
    fig, axes = plt.subplots(2, max(len(feats), 1), figsize=(3.2 * max(len(feats), 1), 5.5), squeeze=False)
    for j, f in enumerate(feats):
        ax = axes[0, j]
        phase_feat = j < len(ph)
        eps = U if phase_feat else [m for m in U if m["task"] == ex["task_examples"][j - len(ph)]["task"]]
        others = [] if phase_feat else [m for m in U if m["task"] != ex["task_examples"][j - len(ph)]["task"]][::10]
        for m in eps[:: max(1, len(eps) // 40)]:
            y = dense[m["off"]: m["off"] + m["C"], j]
            gc = m["gc"]
            xx = np.arange(m["C"]) - (gc if (phase_feat and gc is not None) else 0)
            ax.plot(xx, y, color="C0", alpha=0.3, lw=0.8)
        for m in others:
            ax.plot(np.arange(m["C"]), dense[m["off"]: m["off"] + m["C"], j], color="0.6", alpha=0.3, lw=0.8)
        ax.set_title(f"f{f} " + ("phase " + ex["phase_examples"][j]["best_phase"] if phase_feat
                                 else "task " + ex["task_examples"][j - len(ph)]["task"]), fontsize=9)
        ax.set_xlabel("call − grasp call" if phase_feat else "call")
        ax2 = axes[1, j]
        if phase_feat:
            ax2.bar(s["phases"], s["P"][:, f], color="C0")
            ax2.set_ylabel("P(fire | phase)")
        else:
            ax2.bar(range(len(s["tasks"])), s["T"][:, f], color="C1")
            ax2.set_xticks(range(0, len(s["tasks"]), 5), [s["tasks"][i] for i in range(0, len(s["tasks"]), 5)], fontsize=7)
            ax2.set_ylabel("P(fire | task)")
        ax2.set_ylim(0, 1)
    fig.suptitle(f"stage 1 examples, expert L{layer} step 9 (U40 rollouts; grey = other tasks)", fontsize=10)
    fig.tight_layout()
    (S1 / "figs").mkdir(exist_ok=True)
    fig.savefig(S1 / "figs" / "examples.png", dpi=130)
    print(C.dumps(ex), flush=True)
    return ex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("sweep", "select", "layers", "examples", "all"))
    a = ap.parse_args()
    S1.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.stage in ("sweep", "all"):
        stage_sweep()
    if a.stage in ("select", "all"):
        stage_select()
    if a.stage in ("layers", "all"):
        stage_layers()
    if a.stage in ("examples", "all"):
        stage_examples()
    with open(S1 / "gpu_time.log", "a") as f:
        f.write(f"{time.strftime('%F %T')} {a.stage} {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
