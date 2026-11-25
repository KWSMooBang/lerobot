#!/usr/bin/env python
"""T031 stage 2 training: action-expert SAEs on the LIBERO training data collected by collect.py.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 flock <gpu0 lock> python ae_stage2_train.py --acts <acts dir>

Jobs: 18 token-mean SAEs (layers {2,5,8,11,14,17} × steps {0,4,9}) + 6 per-token SAEs (step 9, 8 sampled tokens per
timestep). Setting from stage 1 (stage1/selected.json). 90/10 split by LIBERO episode (seed 0), quality on held-out,
steps = max(100 epochs, 3000). T015 stats (stats.run_stats) on all samples. Out: /outputs/.../sae_ae/stage2/<job>/.
Per-token stats: C.1 metrics treat (frame, token slot) as time = frame·8 + slot (codes also keep tok_idx).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ae_common as C  # noqa: E402
import store  # noqa: E402
import torch  # noqa: E402
from stats import run_stats  # noqa: E402

S2 = C.OUT / "stage2"
ARGS = argparse.Namespace(device="cuda", max_steps=0)
MIN_STEPS = 3000
EPOCHS = 100


def load(acts: Path, layer: int, tok: bool) -> tuple[np.ndarray, pd.DataFrame, np.ndarray | None]:
    """(x, meta, tok_idx). Mean: x (T, 3, 1024) for steps (0,4,9). Token: x (T·8, 1024), meta repeated."""
    parts = sorted(store.done_parts(acts), key=lambda p: p["part_id"])
    xs, ms, ti = [], [], []
    for p in parts:
        d = Path(p["dir"])
        m = pd.read_parquet(d / "meta.parquet", columns=["index", "episode_index", "frame_index", "task_index", "suite"])
        if tok:
            a = np.load(d / f"ae{layer:02d}_tok.npy", mmap_mode="r")  # (T, 1, 8, d)
            xs.append(np.asarray(a[:, 0]).reshape(-1, a.shape[-1]))
            idx = np.load(d / "tok_idx.npy")
            ti.append(idx.reshape(-1))
            K = idx.shape[1]
            m = m.loc[m.index.repeat(K)].reset_index(drop=True)
            m["slot"] = np.tile(np.arange(K), len(m) // K)
        else:
            xs.append(np.asarray(np.load(d / f"ae{layer:02d}.npy", mmap_mode="r")))
        ms.append(m)
    meta = pd.concat(ms, ignore_index=True)
    meta["suite"] = meta["suite"].astype("category")
    return np.concatenate(xs), meta, (np.concatenate(ti) if tok else None)


def run_job(acts: Path, layer: int, step: int, tok: bool, x_np: np.ndarray, meta: pd.DataFrame, tok_idx, sel: dict,
            hold_eps: set[int]) -> dict:
    name = f"L{layer:02d}_s{step}" + ("_tok" if tok else "")
    out = S2 / name
    if (out / "done.json").exists():
        print(f"[s2] {name} done, skip", flush=True)
        return json.loads((out / "metrics.json").read_text())
    dev = ARGS.device
    hold = meta["episode_index"].isin(hold_eps).to_numpy()
    x = torch.from_numpy(x_np).to(dev)
    xtr, xho = x[torch.from_numpy(~hold).to(dev)], x[torch.from_numpy(hold).to(dev)]
    n_steps = max(MIN_STEPS, EPOCHS * math.ceil(len(xtr) / 4096))
    if ARGS.max_steps:
        n_steps = min(n_steps, ARGS.max_steps)
    sae, info = C.fit_sae(xtr, sel["n_latents"], sel["k"], n_steps, log_every=1000, tag=name)
    chans = C.massive_channels().get(f"e_s{step}/L{layer}", [])
    m_tr, f_tr = C.evaluate(sae, xtr, chans)
    m_ho, f_ho = C.evaluate(sae, xho, chans)
    del xtr, xho
    res = {"job": name, "layer": layer, "step": step, "per_token": tok, "n_latents": sel["n_latents"], "k": sel["k"],
           "n_train": int((~hold).sum()), "n_heldout": int(hold.sum()), "train": m_tr, "heldout": m_ho,
           "never_fire_all": float(((f_tr + f_ho) == 0).mean()), "massive_channels": chans,
           **{kk: v for kk, v in info.items() if kk != "log"}}
    out.mkdir(parents=True, exist_ok=True)
    sae.save(out, extra={"layer": layer, "step": step, "per_token": tok, "acts_dir": str(acts),
                         "source": "HuggingFaceVLA/libero rev 86958911, collect.py frame_stride 2"})
    (out / "log.json").write_text(C.dumps(info["log"]))
    m = meta.copy()
    if tok:
        m["frame_index"] = m["frame_index"] * 8 + m["slot"]
        np.save(out / "codes_tok_idx.npy", tok_idx.astype(np.int8))
    m["split"] = np.where(hold, "heldout", "train")
    m[["episode_index", "split"]].drop_duplicates().to_parquet(out / "split.parquet", index=False)
    t1 = time.time()
    res["stats"] = run_stats(out, sae, x, m)
    res["stats_seconds"] = round(time.time() - t1, 1)
    (out / "metrics.json").write_text(C.dumps(res))
    (out / "done.json").write_text(C.dumps({"job": name, "time": time.strftime("%F %T")}))
    print(f"[s2] {name}: heldout nmse {m_ho['nmse']:.4f} VE {m_ho['var_explained']:.4f} never "
          f"{res['never_fire_all']:.3f} ({info['train_seconds']:.0f}s train, {n_steps} steps)", flush=True)
    del x
    torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--only", default="all", choices=("all", "mean", "tok"))
    ap.add_argument("--out", default=None, help="default /outputs/VLA-shortcut-exps/sae_ae/stage2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=0, help="testing only")
    a = ap.parse_args()
    global S2
    if a.out:
        S2 = Path(a.out)
    ARGS.device, ARGS.max_steps = a.device, a.max_steps
    acts = Path(a.acts)
    sel = json.loads((C.OUT / "stage1" / "selected.json").read_text())
    S2.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eps = sorted({e for p in store.done_parts(acts) for e in p["episodes"]})
    rng = np.random.default_rng(0)
    hold_eps = {eps[i] for i in rng.permutation(len(eps))[: max(1, int(round(len(eps) * 0.1)))]}
    rows = []
    for layer in C.LAYERS:
        if a.only in ("all", "mean"):
            x3, meta, _ = load(acts, layer, tok=False)
            for si, step in enumerate((0, 4, 9)):
                rows.append(run_job(acts, layer, step, False, np.ascontiguousarray(x3[:, si]), meta, None, sel, hold_eps))
            del x3
        if a.only in ("all", "tok"):
            xt, meta, ti = load(acts, layer, tok=True)
            rows.append(run_job(acts, layer, 9, True, xt, meta, ti, sel, hold_eps))
            del xt
    summ = [{"job": r["job"], "nmse": r["heldout"]["nmse"], "ve": r["heldout"]["var_explained"],
             "never_fire": r["never_fire_all"], "n_train": r["n_train"], "n_heldout": r["n_heldout"],
             "massive_share_norm": r["heldout"].get("massive_share_norm"), "ve_rest": r["heldout"].get("ve_rest"),
             "stats": r.get("stats")} for r in rows]
    (S2 / "quality.json").write_text(C.dumps(summ))
    with open(S2 / "gpu_time.log", "a") as f:
        f.write(f"{time.strftime('%F %T')} train {a.only} {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
