#!/usr/bin/env python
"""T034 crosscoder training + held-out evaluation (one run per call; GPU 0 under the queue lock, see run_gpu.sh).

  python cc_train.py --data expert --n-latents 16384 --k 64 --name expert_n16k_k64
  python cc_train.py --data vlm    --n-latents 8192  --k 64 --name vlm_n8k_k64
  python cc_train.py --data joint  --n-latents 8192  --k 64 --name joint_n8k_k64 --batch 2048

Data: cc_data.py caches (rows = LIBERO timesteps, frame_index % 2 == 0). expert = ae00–ae17 step 9 (18 × 1024);
vlm = pg00–pg17 pooled_all (18 × 2048; --vlm-emb adds emb); joint = vlm blocks then expert blocks, same rows.
Split: T031's held-out episodes (10 %, numpy default_rng(0) permutation of the sorted episode list).
Trainer: model.py / ae_common.fit_sae recipe (Adam 1e-4, grad clip 1, tangent-plane decoder grad + renorm, AuxK
dead window 500 steps, b_pre = per-block geometric median of 10 000 train rows), bf16 autocast for the matmuls,
fp32 master weights. Batches are gathered on the CPU (pinned) by a prefetch thread.

Out (weights, big): /data0/.../sae/crosscoder/<name>/ crosscoder.safetensors, config.json, metrics.json, log.json,
codes_idx.npy (N, k) int16 + codes_val.npy (N, k) fp16 for every row (train + held-out), profile.npy (n, L).
Small: /outputs/VLA-shortcut-exps/sae/crosscoder/runs/<name>.json (metrics without the training log).
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from cc_data import cache_dir  # noqa: E402
from crosscoder import CrossCoder, CrossCoderConfig  # noqa: E402

WEIGHTS = Path("/data0/woosung/datasets/VLA-shortcut-exps/sae/crosscoder")
OUT = Path("/outputs/VLA-shortcut-exps/sae/crosscoder")


# ---------------------------------------------------------------------------- data
def load_data(kind: str, vlm_emb: bool = False, frac: float = 1.0) -> tuple[torch.Tensor, list[str], list[int], pd.DataFrame, dict]:
    """→ X (N, D) fp16 CPU tensor, layer names, dims, meta, massive channels {layer: [ch]}."""
    blocks, names, massive = [], [], {}
    meta = None
    if kind in ("vlm", "joint"):
        d = cache_dir("vlm")
        info = json.loads((d / "info.json").read_text())
        x = np.load(d / "x.npy", mmap_mode="r")
        pick = [i for i, nm in enumerate(info["layers"]) if nm != "emb" or vlm_emb]
        blocks.append((x, pick, 2048))
        names += [info["layers"][i] for i in pick]
        massive.update({info["layers"][i]: info["massive_channels"][info["layers"][i]] for i in pick})
        meta = pd.read_parquet(d / "meta.parquet")
    if kind in ("expert", "joint"):
        d = cache_dir("expert")
        info = json.loads((d / "info.json").read_text())
        x = np.load(d / "x.npy", mmap_mode="r")
        blocks.append((x, list(range(18)), 1024))
        names += info["layers"]
        massive.update(info["massive_channels"])
        me = pd.read_parquet(d / "meta.parquet")
        if meta is not None:
            ok = len(me) == len(meta) and (me[["episode_index", "frame_index"]].to_numpy() == meta[["episode_index", "frame_index"]].to_numpy()).all()
            if not ok:
                raise SystemExit("expert and vlm caches are not row-aligned (run cc_data.py check)")
        meta = me
    assert meta is not None
    N = len(meta)
    rows = np.arange(N)
    if frac < 1.0:  # smoke: a contiguous random block of whole episodes
        eps = np.sort(meta["episode_index"].unique())
        keep = set(np.random.default_rng(1).choice(eps, size=max(2, int(len(eps) * frac)), replace=False).tolist())
        rows = np.where(meta["episode_index"].isin(keep).to_numpy())[0]
        meta = meta.iloc[rows].reset_index(drop=True)
    dims = [dd for (_, pick, dd) in blocks for _ in pick]
    D = sum(dims)
    X = torch.empty(len(rows), D, dtype=torch.float16)
    col = 0
    for x, pick, dd in blocks:
        for i in pick:
            for s in range(0, len(rows), 20000):
                r = rows[s : s + 20000]
                X[s : s + len(r), col : col + dd] = torch.from_numpy(np.ascontiguousarray(x[r[0] : r[-1] + 1, i][r - r[0]]))
            col += dd
    return X, names, dims, meta, massive


def heldout_mask(meta: pd.DataFrame) -> np.ndarray:
    """T031 split: 10 % of the episodes, default_rng(0) permutation of the sorted episode list (all 1 693 episodes)."""
    eps = sorted(meta["episode_index"].unique().tolist())
    rng = np.random.default_rng(0)
    hold = {eps[i] for i in rng.permutation(len(eps))[: max(1, int(round(len(eps) * 0.1)))]}
    return meta["episode_index"].isin(hold).to_numpy()


class Prefetch:
    """Background thread: random batches of X[rows] → pinned → device (non-blocking)."""

    def __init__(self, X: torch.Tensor, rows: np.ndarray, batch: int, steps: int, seed: int, device: str, depth: int = 3):
        self.q: queue.Queue = queue.Queue(depth)
        self.X, self.rows, self.batch, self.steps, self.seed, self.device = X, rows, batch, steps, seed, device
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self) -> None:
        n = len(self.rows)
        spe = math.ceil(n / self.batch)
        perm = None
        for step in range(self.steps):
            epoch, pos = divmod(step, spe)
            if perm is None or pos == 0:
                perm = np.random.default_rng(self.seed * 100_003 + epoch).permutation(n)
            idx = np.sort(self.rows[perm[pos * self.batch : (pos + 1) * self.batch]])
            xb = self.X[torch.from_numpy(idx)].pin_memory()
            self.q.put(xb.to(self.device, non_blocking=True))
        self.q.put(None)

    def __iter__(self):
        while (xb := self.q.get()) is not None:
            yield xb


class AdamLowMem(torch.optim.Optimizer):
    """Adam (no weight decay) with exp_avg stored in bf16 and exp_avg_sq in fp32, updated in row chunks so the
    fp32 temporaries stay small. (bf16 exp_avg_sq would not decay: 0.999·v rounds back to v in bf16.)"""

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 chunk_rows: int = 512):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})
        self.chunk_rows = chunk_rows

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["t"] += 1
                t = st["t"]
                c1, c2 = 1 - b1**t, 1 - b2**t
                P, G, M, V = (x.view(x.shape[0], -1) if x.dim() > 1 else x.view(1, -1) for x in (p, p.grad, st["m"], st["v"]))
                for r in range(0, P.shape[0], self.chunk_rows):
                    sl = slice(r, r + self.chunk_rows)
                    gg = G[sl]
                    m = M[sl].float().mul_(b1).add_(gg, alpha=1 - b1)
                    V[sl].mul_(b2).addcmul_(gg, gg, value=1 - b2)
                    P[sl].addcdiv_(m, (V[sl] / c2).sqrt_().add_(g["eps"]), value=-g["lr"] / c1)
                    M[sl] = m.to(torch.bfloat16)


# ---------------------------------------------------------------------------- train / eval
def fit(X: torch.Tensor, train_rows: np.ndarray, names: list[str], dims: list[int], n_latents: int, k: int,
        steps: int, batch: int, seed: int = 0, lr: float = 1e-4, device: str = "cuda", log_every: int = 250,
        tag: str = "", low_mem_adam: bool = False, aux_coef: float = 1 / 32) -> tuple[CrossCoder, dict]:
    cfg = CrossCoderConfig(layers=names, dims=dims, n_latents=n_latents, k=k, aux_coef=aux_coef)
    cc = CrossCoder(cfg, seed=seed).to(device)
    g = np.random.default_rng(seed)
    pick = np.sort(g.choice(train_rows, size=min(10_000, len(train_rows)), replace=False))
    cc.init_from_data(X[torch.from_numpy(pick)].to(device).float())
    opt = (AdamLowMem(cc.parameters(), lr=lr) if low_mem_adam else
           torch.optim.Adam(cc.parameters(), lr=lr, betas=(0.9, 0.999), fused=device.startswith("cuda")))
    last_fired = torch.zeros(n_latents, dtype=torch.long, device=device)
    log: list[dict] = []
    run_mse = torch.zeros((), device=device)
    run_l = torch.zeros(len(dims), device=device)
    torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
    t0 = time.time()
    for step, xb in enumerate(Prefetch(X, train_rows, batch, steps, seed, device)):
        dead = (step - last_fired) >= cfg.dead_steps
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss, parts, fired = cc.loss(xb, dead)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        cc.project_decoder_grad()
        torch.nn.utils.clip_grad_norm_(cc.parameters(), 1.0)
        opt.step()
        cc.renorm_decoder()
        last_fired[fired] = step + 1
        run_mse += parts["mse"]
        run_l += parts["mse_l"]
        if (step + 1) % log_every == 0 or step + 1 == steps:
            kk = log_every if (step + 1) % log_every == 0 else (step + 1) % log_every
            rec = {"step": step + 1, "nmse_sum": float(run_mse) / kk, "nmse_mean": float(run_l.mean()) / kk,
                   "nmse_l": (run_l / kk).tolist(),
                   "dead_frac": float(((step + 1 - last_fired) >= cfg.dead_steps).float().mean()),
                   "sec": round(time.time() - t0, 1)}
            log.append(rec)
            run_mse.zero_()
            run_l.zero_()
            print(f"[fit {tag}] step {step + 1}/{steps} nmse(mean over layers) {rec['nmse_mean']:.4f} "
                  f"dead {rec['dead_frac']:.3f} {rec['sec']:.0f}s", flush=True)
    sec = time.time() - t0
    info = {"steps": steps, "batch": batch, "lr": lr, "seed": seed, "epochs": steps * batch / len(train_rows),
            "train_seconds": round(sec, 1), "steps_per_s": steps / sec, "samples_per_s": steps * batch / sec,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else None,
            "dead_frac_last_window": float(((steps - last_fired) >= cfg.dead_steps).float().mean()),
            "c_mse": cfg.c_mse, "log": log}
    cc.eval()
    return cc, info


@torch.no_grad()
def evaluate(cc: CrossCoder, X: torch.Tensor, rows: np.ndarray, massive: dict[str, list[int]], device: str,
             batch: int = 4096, codes: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[dict, np.ndarray]:
    """Per-block metrics on X[rows] (ae_common.evaluate per block) + fire counts. codes: (idx, val) arrays (N, k)
    filled at `rows` (all-row code dump)."""
    cfg = cc.cfg
    L, o = len(cfg.dims), cfg.offsets
    n = len(rows)
    mean = torch.zeros(cfg.D, dtype=torch.float64, device=device)
    for s in range(0, n, 16384):
        mean += X[torch.from_numpy(rows[s : s + 16384])].to(device).double().sum(0)
    mean = (mean / n).float()
    sq_n = torch.zeros(L, dtype=torch.float64, device=device)
    err_ch = torch.zeros(cfg.D, dtype=torch.float64, device=device)
    var_ch = torch.zeros(cfg.D, dtype=torch.float64, device=device)
    e_norm = torch.zeros(cfg.D, dtype=torch.float64, device=device)
    fire = torch.zeros(cfg.n_latents, dtype=torch.float64, device=device)
    for s in range(0, n, batch):
        r = rows[s : s + batch]
        xb = X[torch.from_numpy(r)].to(device).float()
        xt, mu, nrm = cc.normalize(xb)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            pre = cc.pre_acts(xt)
        vals, idx = cc.topk(pre)
        xt_hat = cc.decode_sparse(vals, idx).float()
        x_hat = cc.denormalize(xt_hat, mu, nrm)
        sq_n += cc.block_sq(xt - xt_hat).double().sum(0)
        err_ch += (xb - x_hat).double().pow(2).sum(0)
        var_ch += (xb - mean).double().pow(2).sum(0)
        e_norm += xt.double().pow(2).sum(0)
        fire.scatter_add_(0, idx[vals > 0], torch.ones_like(vals[vals > 0], dtype=torch.float64))
        if codes is not None:
            codes[0][r] = idx.to(torch.int16).cpu().numpy()
            codes[1][r] = vals.half().cpu().numpy()
    per = []
    for i, nm in enumerate(cfg.layers):
        sl = slice(o[i], o[i + 1])
        rec = {"layer": nm, "nmse": float(sq_n[i] / n / cfg.c_mse[i]),
               "var_explained": float(1 - err_ch[sl].sum() / var_ch[sl].sum().clamp_min(1e-12))}
        ch = massive.get(nm, [])
        if ch:
            m = torch.zeros(cfg.dims[i], dtype=torch.bool, device=device)
            m[ch] = True
            ec, vc, en = err_ch[sl], var_ch[sl], e_norm[sl]
            rec.update({"massive_n": len(ch), "massive_share_norm": float(en[m].sum() / en.sum()),
                        "massive_share_var": float(vc[m].sum() / vc.sum()),
                        "ve_massive": float(1 - ec[m].sum() / vc[m].sum().clamp_min(1e-12)),
                        "ve_rest": float(1 - ec[~m].sum() / vc[~m].sum().clamp_min(1e-12))})
        else:
            rec.update({"massive_n": 0, "massive_share_norm": 0.0})
        per.append(rec)
    out = {"n": n, "per_layer": per,
           "ve_mean": float(np.mean([p["var_explained"] for p in per])),
           "ve_min": float(np.min([p["var_explained"] for p in per])),
           "nmse_mean": float(np.mean([p["nmse"] for p in per])),
           "never_fire": float((fire == 0).double().mean()), "mean_l0": float(fire.sum() / n)}
    return out, fire.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=("expert", "vlm", "joint"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-latents", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--epochs", type=float, default=100)
    ap.add_argument("--steps", type=int, default=0, help="override epochs")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0, help="smoke: fraction of episodes")
    ap.add_argument("--vlm-emb", action="store_true")
    ap.add_argument("--no-codes", action="store_true")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--out", default=str(OUT / "runs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--low-mem-adam", action="store_true", help="exp_avg in bf16 (joint, Addendum B)")
    ap.add_argument("--aux-coef", type=float, default=1 / 32, help="AuxK weight (Addendum E sensitivity run)")
    a = ap.parse_args()
    wdir = Path(a.weights) / a.name
    if (wdir / "metrics.json").exists():
        raise SystemExit(f"{wdir} exists; not overwriting")
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    X, names, dims, meta, massive = load_data(a.data, a.vlm_emb, a.frac)
    print(f"[cc] {a.name}: data {a.data} X {tuple(X.shape)} ({X.numel() * 2 / 2**30:.1f} GB) load {time.time() - t0:.0f}s", flush=True)
    hold = heldout_mask(meta)
    tr, ho = np.where(~hold)[0], np.where(hold)[0]
    steps = a.steps or int(math.ceil(a.epochs * len(tr) / a.batch))
    cc, info = fit(X, tr, names, dims, a.n_latents, a.k, steps, a.batch, a.seed, a.lr, a.device, tag=a.name,
                   low_mem_adam=a.low_mem_adam, aux_coef=a.aux_coef)
    t1 = time.time()
    codes = None
    if not a.no_codes:
        codes = (np.zeros((len(X), a.k), np.int16), np.zeros((len(X), a.k), np.float16))
    m_ho, f_ho = evaluate(cc, X, ho, massive, a.device, codes=codes)
    m_tr, f_tr = evaluate(cc, X, tr, massive, a.device, codes=codes)
    fire_all = f_ho + f_tr
    prof = cc.layer_profile().cpu().numpy()
    res = {"name": a.name, "data": a.data, "layers": names, "dims": dims, "n_latents": a.n_latents, "k": a.k,
           "n_train": len(tr), "n_heldout": len(ho), "frac": a.frac, "heldout": m_ho, "train": m_tr,
           "never_fire_all": float((fire_all == 0).mean()),
           **{kk: v for kk, v in info.items() if kk != "log"}, "eval_seconds": round(time.time() - t1, 1),
           "weights_dir": str(wdir), "gpu": torch.cuda.get_device_name() if a.device.startswith("cuda") else "cpu"}
    cc.save(wdir, extra={"data": a.data, "name": a.name, "heldout_rule": "T031 10% episodes default_rng(0)",
                         "source_caches": [str(cache_dir(s)) for s in (["vlm"] if a.data != "expert" else []) + (["expert"] if a.data != "vlm" else [])]})
    np.save(wdir / "profile.npy", prof.astype(np.float32))
    np.save(wdir / "fire_count.npy", fire_all.astype(np.int64))
    if codes is not None:
        np.save(wdir / "codes_idx.npy", codes[0])
        np.save(wdir / "codes_val.npy", codes[1])
        np.save(wdir / "heldout.npy", hold)
    (wdir / "log.json").write_text(json.dumps(info["log"]) + "\n")
    (wdir / "metrics.json").write_text(json.dumps(res, indent=1) + "\n")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.name}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"[cc] {a.name}: heldout VE mean {m_ho['ve_mean']:.4f} min {m_ho['ve_min']:.4f} never-fire {res['never_fire_all']:.3f} "
          f"train {info['train_seconds']:.0f}s ({info['steps_per_s']:.2f} it/s, peak {info['peak_mem_gb']} GB) total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
