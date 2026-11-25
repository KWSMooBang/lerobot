"""T031 action-expert SAEs: shared pieces (paths, T028/T030 loaders, held-out trainer, metrics, phase labels).

The trainer is train.py's loop (T015) without the acts-dir bookkeeping: same TopKSAE, init, loss, Adam,
tangent-plane decoder gradient, renorm, AuxK dead mask. Quality is reported on held-out episodes.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from model import SAEConfig, TopKSAE  # noqa: E402

OUT = Path("/outputs/VLA-shortcut-exps/sae_ae")
DATA = Path("/data0/woosung/datasets/VLA-shortcut-exps")
RICH = Path("/outputs/VLA-shortcut-exps/rich_acts")
T030 = Path("/outputs/VLA-shortcut-exps/analysis/layer_pop")
LAYERS = (2, 5, 8, 11, 14, 17)
T028_STEPS = (0, 4, 9)  # exp_mean step axis
N_TOT = 51_100
WORKERS = 8


# ---------------------------------------------------------------------------- T028 / T030 data
def meta() -> list[dict]:
    """T030 episode meta (rich manifest joined with T027 labels); m['off'], m['C'] index the call axis."""
    return pickle.load(open(T030 / "_cache" / "meta.pkl", "rb"))


def exp_mean(layer: int, step: int) -> np.ndarray:
    """T028 exp_mean (= hooks aeNN) for one (layer, denoise step): (51 100, 1024) fp32, rows = calls in meta order."""
    mm = np.load(T030 / "_cache" / "exp_mean.npy", mmap_mode="r")  # (18, N, 3, 1024) fp16
    assert mm.shape == (18, N_TOT, 3, 1024), mm.shape
    return np.asarray(mm[layer, :, T028_STEPS.index(step)], dtype=np.float32)


def episode_split(n_eps: int, frac_hold: float = 0.1, seed: int = 0) -> np.ndarray:
    """bool (n_eps,) held-out mask, random by episode."""
    rng = np.random.default_rng(seed)
    hold = np.zeros(n_eps, bool)
    hold[rng.permutation(n_eps)[: int(round(n_eps * frac_hold))]] = True
    return hold


def call_rows(M: list[dict], mask: np.ndarray) -> np.ndarray:
    return np.concatenate([np.arange(m["off"], m["off"] + m["C"]) for m, h in zip(M, mask, strict=True) if h])


def massive_channels() -> dict[str, list[int]]:
    j = json.load(open(T030 / "res" / "massive_channels.json"))
    return {k: [c["ch"] for c in v] for k, v in j.items() if k.startswith("e_s")}


def phase_labels(m: dict) -> list:
    """T030 / T027 per-call 4-phase label (copied from layer_pop.phase_labels)."""
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


def u40(M: list[dict]) -> list[dict]:
    return [m for m in M if m["t027_set"] == "train"]


# ---------------------------------------------------------------------------- training
def fit_sae(
    x: torch.Tensor,
    n_latents: int,
    k: int,
    steps: int,
    seed: int = 0,
    batch: int = 4096,
    lr: float = 1e-4,
    k_aux: int = 512,
    aux_coef: float = 1 / 32,
    dead_steps: int = 500,
    grad_clip: float = 1.0,
    gm_samples: int = 10_000,
    log_every: int = 500,
    tag: str = "",
) -> tuple[TopKSAE, dict[str, Any]]:
    """x (N, d) on the training device (fp16/fp32). Returns the SAE and training info (log, seconds, dead)."""
    device = x.device
    n, d = x.shape
    sae = TopKSAE(
        SAEConfig(d=d, n_latents=n_latents, k=k, k_aux=k_aux, aux_coef=aux_coef, dead_steps=dead_steps), seed=seed
    ).to(device)
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(n, generator=g)[:gm_samples].to(device)
    sae.init_from_data(x[pick].float())
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    last_fired = torch.zeros(n_latents, dtype=torch.long, device=device)
    bsz = min(batch, n)
    spe = math.ceil(n / bsz)
    perm = None
    log: list[dict[str, float]] = []
    run_mse = 0.0
    t0 = time.time()
    for step in range(steps):
        epoch, pos = divmod(step, spe)
        if perm is None or pos == 0:
            gen = torch.Generator(device=device).manual_seed(seed * 100_003 + epoch)
            perm = torch.randperm(n, generator=gen, device=device)
        xb = x[perm[pos * bsz : (pos + 1) * bsz]].float()
        dead = (step - last_fired) >= dead_steps
        loss, parts, fired = sae.loss(xb, dead)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.project_decoder_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), grad_clip)
        opt.step()
        sae.renorm_decoder()
        last_fired[fired] = step + 1
        run_mse += float(parts["mse"])
        if (step + 1) % log_every == 0 or step + 1 == steps:
            kk = log_every if (step + 1) % log_every == 0 else (step + 1) % log_every
            rec = {
                "step": step + 1,
                "epoch": (step + 1) / spe,
                "nmse": run_mse / kk,
                "dead_frac": float(((step + 1 - last_fired) >= dead_steps).float().mean()),
                "sec": round(time.time() - t0, 1),
            }
            log.append(rec)
            run_mse = 0.0
            print(f"[fit {tag}] step {rec['step']}/{steps} nmse {rec['nmse']:.4f} dead {rec['dead_frac']:.3f}", flush=True)
    info = {
        "steps": steps,
        "epochs": steps / spe,
        "train_seconds": round(time.time() - t0, 1),
        "dead_frac_last_window": float(((steps - last_fired) >= dead_steps).float().mean()),
        "c_mse": sae.cfg.c_mse,
        "log": log,
    }
    sae.eval()
    return sae, info


@torch.no_grad()
def evaluate(
    sae: TopKSAE, x: torch.Tensor, chans: list[int] | None = None, batch: int = 16384
) -> tuple[dict[str, float], np.ndarray]:
    """Metrics on x (N, d) + fire counts (n_latents,). chans: also VE on these channels and on the rest,
    and their share of the raw (mean-over-d removed) and normalised energy."""
    n = x.shape[0]
    mean = sum(x[i : i + 65536].float().sum(0) for i in range(0, n, 65536)) / n
    sq = sq_raw = var_raw = 0.0
    fire = torch.zeros(sae.cfg.n_latents, device=x.device)
    per_ch_err = torch.zeros(x.shape[1], device=x.device)
    per_ch_var = torch.zeros(x.shape[1], device=x.device)
    e_raw = torch.zeros(x.shape[1], device=x.device)
    e_norm = torch.zeros(x.shape[1], device=x.device)
    for i in range(0, n, batch):
        xb = x[i : i + batch].float()
        xt, mu, nrm = sae.normalize(xb)
        vals, idx = sae.topk(sae.pre_acts(xt))
        xt_hat = sae.decode_sparse(vals, idx)
        x_hat = xt_hat * nrm + mu + sae.b_pre
        sq += float((xt - xt_hat).pow(2).sum())
        err = (xb - x_hat).pow(2)
        sq_raw += float(err.sum())
        var_raw += float((xb - mean).pow(2).sum())
        per_ch_err += err.sum(0)
        per_ch_var += (xb - mean).pow(2).sum(0)
        e_raw += (xb - xb.mean(-1, keepdim=True)).pow(2).sum(0)
        e_norm += xt.pow(2).sum(0)
        fire.scatter_add_(0, idx[vals > 0], torch.ones_like(vals[vals > 0]))
    out = {
        "n": n,
        "nmse": sq / n / sae.cfg.c_mse,
        "var_explained": 1 - sq_raw / max(var_raw, 1e-12),
        "never_fire": float((fire == 0).float().mean()),
        "mean_l0": float(fire.sum() / n),
    }
    if chans is not None:
        m = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
        m[chans] = True
        out["massive_n"] = len(chans)
        out["massive_share_raw"] = float(e_raw[m].sum() / e_raw.sum())
        out["massive_share_norm"] = float(e_norm[m].sum() / e_norm.sum())
        out["massive_share_var"] = float(per_ch_var[m].sum() / per_ch_var.sum())
        out["ve_massive"] = float(1 - per_ch_err[m].sum() / per_ch_var[m].sum().clamp_min(1e-12)) if len(chans) else None
        out["ve_rest"] = float(1 - per_ch_err[~m].sum() / per_ch_var[~m].sum().clamp_min(1e-12))
    return out, fire.cpu().numpy()


@torch.no_grad()
def encode_np(sae: TopKSAE, x: np.ndarray, device: str = "cuda", batch: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """x (N, d) numpy → (idx int16 (N, k), val fp32 (N, k))."""
    ii, vv = [], []
    for i in range(0, len(x), batch):
        v, j = sae.encode(torch.from_numpy(np.ascontiguousarray(x[i : i + batch])).to(device).float())
        ii.append(j.to(torch.int16).cpu().numpy())
        vv.append(v.float().cpu().numpy())
    return np.concatenate(ii), np.concatenate(vv)


def dense_codes(idx: np.ndarray, val: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(idx), n), np.float32)
    np.put_along_axis(out, idx.astype(np.int64), val, axis=1)
    return out


def dumps(o: Any) -> str:
    def default(v: Any) -> Any:
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        return str(v)

    return json.dumps(o, indent=1, default=default)
