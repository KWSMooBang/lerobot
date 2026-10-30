"""Feature records of one trained SAE (run at the end of train.py, or alone: `stats.py <job_dir>`).

Writes into the SAE's job dir:
  codes_idx.npy, codes_val.npy   (N, k) int16 / fp16: the k active latents of every sample (values ≥ 0;
                                 the metrics are computed from the fp32 values before this rounding)
  codes_meta.parquet             row-aligned: index, episode_index, frame_index, task_index, suite[, ae_step]
  features.parquet               one row per feature (per denoising step for the action expert):
                                 Swann et al. 2026 App. C.1 metrics + firing frequency
  fire_by_group.parquet          per feature × suite / task: fraction of timesteps firing, episode coverage

C.1 (f = SAE latent activation, per episode e with T_e steps):
  ep_coverage      |E+| / |E|, E+ = episodes where f > 0 at some t
  mean_onsets      mean over E+ of the onsets (0→1) of the hysteresis state s_t = 1 if f > τ_on, 0 if f = 0,
                   else s_{t-1} (s_0 = 0, τ_on = 0.1)
  mean_max_act     mean over E+ of max_t f
  rel_run_length   mean over E+ of (Σ_t s_t / onsets) / T_e   (an active episode that never crosses τ_on
                   has 0 onsets; it counts as run length 0)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import store
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TAU_ON = 0.1
META_COLS = ["index", "episode_index", "frame_index", "task_index", "suite"]


@torch.no_grad()
def encode_all(sae, x: torch.Tensor, batch: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    idx_out, val_out = [], []
    for i in range(0, x.shape[0], batch):
        vals, idx = sae.encode(x[i : i + batch].float())
        idx_out.append(idx.to(torch.int16).cpu())
        val_out.append(vals.float().cpu())  # fp32: the metrics use these; fp16 only on disk
    return torch.cat(idx_out).numpy(), torch.cat(val_out).numpy()


@torch.no_grad()
def episode_metrics(
    idx: np.ndarray,
    val: np.ndarray,
    ep: np.ndarray,
    n: int,
    tau: float = TAU_ON,
    device: str = "cpu",
    chunk: int = 64,
) -> dict[str, np.ndarray]:
    """Rows sorted by (episode, time). Returns per-episode (E, n) arrays and episode ids / lengths."""
    eps, starts, lens = np.unique(ep, return_index=True, return_counts=True)
    order = np.argsort(starts)
    eps, starts, lens = eps[order], starts[order], lens[order]
    n_ep = len(eps)
    res: dict[str, np.ndarray] = {
        k: np.zeros((n_ep, n), np.float32) for k in ("onsets", "active_steps", "max", "active")
    }
    for c0 in range(0, n_ep, chunk):
        c1 = min(c0 + chunk, n_ep)
        tmax = int(lens[c0:c1].max())
        dense = torch.zeros(c1 - c0, tmax, n, device=device)
        for j in range(c0, c1):
            s, ln = starts[j], lens[j]
            ii = torch.from_numpy(idx[s : s + ln].astype(np.int64)).to(device)
            vv = torch.from_numpy(val[s : s + ln].astype(np.float32)).to(device)
            dense[j - c0, :ln].scatter_(1, ii, vv)
        state = torch.zeros(c1 - c0, n, device=device)
        onsets = torch.zeros_like(state)
        act = torch.zeros_like(state)
        for t in range(tmax):
            f = dense[:, t]
            new = torch.where(f > tau, 1.0, torch.where(f == 0, 0.0, state))
            onsets += (new > state).float()
            act += new
            state = new
        res["onsets"][c0:c1] = onsets.cpu().numpy()
        res["active_steps"][c0:c1] = act.cpu().numpy()
        res["max"][c0:c1] = dense.amax(1).cpu().numpy()
        res["active"][c0:c1] = (dense > 0).any(1).float().cpu().numpy()
    res["episodes"] = eps
    res["lengths"] = lens
    return res


def feature_table(em: dict[str, np.ndarray]) -> pd.DataFrame:
    active = em["active"] > 0
    n_act = active.sum(0)
    denom = np.maximum(n_act, 1)
    onsets = em["onsets"]
    run = np.where(onsets > 0, em["active_steps"] / np.maximum(onsets, 1), 0.0)
    rel = run / em["lengths"][:, None]
    nan = np.where(n_act > 0, 1.0, np.nan)
    return pd.DataFrame(
        {
            "feature": np.arange(active.shape[1]),
            "n_active_eps": n_act,
            "ep_coverage": n_act / active.shape[0],
            "mean_onsets": (onsets * active).sum(0) / denom * nan,
            "mean_max_act": (em["max"] * active).sum(0) / denom * nan,
            "rel_run_length": (rel * active).sum(0) / denom * nan,
        }
    )


def fire_counts(idx: np.ndarray, val: np.ndarray, n: int, rows: np.ndarray | None = None) -> np.ndarray:
    ii, vv = (idx, val) if rows is None else (idx[rows], val[rows])
    return np.bincount(ii[vv > 0].astype(np.int64), minlength=n)


def run_stats(
    job_dir: Path, sae, x: torch.Tensor, meta: pd.DataFrame, tau: float = TAU_ON, log=print
) -> dict[str, float]:
    job_dir = Path(job_dir)
    n = sae.cfg.n_latents
    device = str(x.device)
    idx, val = encode_all(sae, x)
    np.save(job_dir / "codes_idx.npy", idx)
    np.save(job_dir / "codes_val.npy", val.astype(np.float16))  # metrics below use the fp32 values
    cols = META_COLS + (["ae_step"] if "ae_step" in meta.columns else [])
    meta = meta[cols].reset_index(drop=True)
    meta.to_parquet(job_dir / "codes_meta.parquet", index=False)

    steps = sorted(meta["ae_step"].unique()) if "ae_step" in meta.columns else [None]
    feats, groups = [], []
    for st in steps:
        rows = np.arange(len(meta)) if st is None else np.flatnonzero(meta["ae_step"].to_numpy() == st)
        m = meta.iloc[rows]
        key = np.lexsort((m["frame_index"].to_numpy(), m["episode_index"].to_numpy()))
        rows = rows[key]
        m = meta.iloc[rows]
        em = episode_metrics(idx[rows], val[rows], m["episode_index"].to_numpy(), n, tau, device)
        ft = feature_table(em)
        ft["fire_freq"] = fire_counts(idx, val, n, rows) / len(rows)
        if st is not None:
            ft.insert(0, "ae_step", int(st))
        feats.append(ft)
        ep_group = m.groupby("episode_index")[["suite", "task_index"]].first().loc[em["episodes"]]
        for gtype, col in (("suite", "suite"), ("task", "task_index")):
            for g, sub in m.groupby(col, observed=True):
                cnt = fire_counts(idx, val, n, rows[np.flatnonzero((m[col] == g).to_numpy())])
                ep_mask = (ep_group[col] == g).to_numpy()
                df = pd.DataFrame(
                    {
                        "feature": np.arange(n),
                        "group_type": gtype,
                        "group": str(g),
                        "n_timesteps": len(sub),
                        "fire_rate": cnt / len(sub),
                        "ep_coverage": em["active"][ep_mask].mean(0),
                    }
                )
                if st is not None:
                    df.insert(0, "ae_step", int(st))
                groups.append(df)
    feats_df = pd.concat(feats, ignore_index=True)
    feats_df.to_parquet(job_dir / "features.parquet", index=False)
    pd.concat(groups, ignore_index=True).to_parquet(job_dir / "fire_by_group.parquet", index=False)
    summary = {
        "n_samples": int(len(meta)),
        "features_never_fire": int((feats_df.groupby("feature")["fire_freq"].max() == 0).sum()),
        "median_ep_coverage": float(feats_df["ep_coverage"].median()),
        "median_mean_onsets": float(feats_df["mean_onsets"].median()),
    }
    log(f"[stats] {job_dir}: {store.dumps(summary)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="recompute the feature records of a trained SAE job dir")
    ap.add_argument("job_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from model import TopKSAE
    from store import load_layer

    job = Path(a.job_dir)
    cfg = json.loads((job / "config.json").read_text())
    sae = TopKSAE.load(job, a.device)
    x, meta = load_layer(Path(cfg["acts_dir"]), cfg["layer"], cfg["input"], cfg.get("ae_steps"))
    run_stats(job, sae, torch.from_numpy(x).to(a.device), meta)


if __name__ == "__main__":
    main()
