"""On-disk layout of collected activations and how to read them back.

<acts_dir> = <out-root>/sae/<model>/acts/
  plan.json                     identity (model, ckpt, adapter, dataset revision, layers, ae_steps, parts spec)
                                + the part list (episode ranges). Same for every pod / shard count.
  parts/<part_id>/              one episode range (part_id = ep<first>-<last>)
    <layer>.npy                 emb, pgNN: (T, 3, 2048) token-group means (cam1, cam2, text)
                                aeNN: (T, S, 1024) action-token means per recorded denoising step
                                fp16, or fp32 for a layer that overflowed fp16 in this part (done.json → dtypes)
    counts.npy                  (T, 3) int32 valid tokens per group (padding / empty camera excluded)
    meta.parquet                one row per timestep: row, index, episode_index, frame_index, task_index,
                                suite, instruction, action (dataset), policy_action0 (first action of the chunk)
    done.json                   written last: identity hash, rows, dtypes, overflow layers, seconds
A part directory without done.json is incomplete and is redone on resume.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hooks import GROUPS, layer_family  # noqa: F401  (re-exported for readers)

PG_INPUTS = ("pooled_all", "img", "text", "cam1", "cam2")
FP16_MAX = 65504.0


def identity_hash(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def part_id(episodes: Sequence[int]) -> str:
    eps = list(episodes)
    pid = f"ep{eps[0]:05d}-{eps[-1]:05d}"
    if eps != list(range(eps[0], eps[-1] + 1)):  # non-contiguous (--episodes): disambiguate
        pid += "_" + hashlib.sha256(",".join(map(str, eps)).encode()).hexdigest()[:6]
    return pid


def split_parts(episodes: Sequence[int], lengths: Sequence[int], n_parts: int) -> list[list[int]]:
    """Contiguous episode ranges with about equal frame counts."""
    n_parts = max(1, min(n_parts, len(episodes)))
    cum = np.cumsum(lengths)
    total = cum[-1]
    parts: list[list[int]] = [[] for _ in range(n_parts)]
    for ep, c, ln in zip(episodes, cum, lengths, strict=True):
        j = min(int((c - ln / 2) / total * n_parts), n_parts - 1)
        parts[j].append(int(ep))
    return [p for p in parts if p]


def shard_parts(n_parts: int, shard: int, n_shards: int) -> list[int]:
    """Part indices of shard i/N: N contiguous blocks."""
    return [j for j in range(n_parts) if j * n_shards // n_parts == shard]


def write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def read_plan(acts_dir: Path) -> dict[str, Any]:
    return json.loads((Path(acts_dir) / "plan.json").read_text())


def done_parts(acts_dir: Path) -> list[dict[str, Any]]:
    """done.json of every finished part in plan order (missing parts are skipped)."""
    acts_dir = Path(acts_dir)
    plan = read_plan(acts_dir)
    want = identity_hash(plan["identity"])
    out = []
    for p in plan["parts"]:
        d = acts_dir / "parts" / p["part_id"] / "done.json"
        if d.exists():
            done = json.loads(d.read_text())
            if done.get("identity") == want:
                out.append({**done, "dir": str(d.parent)})
    return out


def pool_pg(arr: np.ndarray, counts: np.ndarray, which: str) -> np.ndarray:
    """(T, 3, d) group means + (T, 3) counts → (T, d) fp32 mean over the chosen tokens."""
    sel = {
        "pooled_all": [0, 1, 2],
        "img": [0, 1],
        "text": [2],
        "cam1": [0],
        "cam2": [1],
    }[which]
    a: np.ndarray = arr[:, sel].astype(np.float32)
    c: np.ndarray = counts[:, sel].astype(np.float32)
    return (a * c[..., None]).sum(1) / np.maximum(c.sum(1, keepdims=True), 1.0)


def load_layer(
    acts_dir: Path, layer: str, which: str = "pooled_all", ae_steps: Sequence[int] | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """All finished parts of one layer → (X (N, d), meta). PaliGemma/emb: N = timesteps, X fp32 pooled per
    `which`. Action expert: N = timesteps × steps (row-major: all steps of a timestep together), X in the
    stored dtype (fp16 unless a part overflowed), meta gets an `ae_step` column (denoising step index)."""
    parts = done_parts(acts_dir)
    if not parts:
        raise SystemExit(f"{acts_dir}: no finished parts")
    plan = read_plan(acts_dir)
    xs, metas = [], []
    for p in parts:
        d = Path(p["dir"])
        arr = np.load(d / f"{layer}.npy", mmap_mode="r")
        meta = pd.read_parquet(d / "meta.parquet")
        if layer_family(layer) == "pg":
            xs.append(pool_pg(arr, np.load(d / "counts.npy"), which))
            metas.append(meta)
        else:
            steps = plan["identity"]["ae_steps"]
            pick = list(range(len(steps))) if ae_steps is None else [steps.index(s) for s in ae_steps]
            sub = np.asarray(arr[:, pick])
            xs.append(sub.reshape(-1, sub.shape[-1]))
            slim = meta[["index", "episode_index", "frame_index", "task_index", "suite"]]
            m = slim.loc[slim.index.repeat(len(pick))].reset_index(drop=True)
            m["suite"] = m["suite"].astype("category")
            m["ae_step"] = np.tile(np.asarray([steps[i] for i in pick], dtype=np.int16), len(meta))
            metas.append(m)
    dt = np.result_type(*[x.dtype for x in xs])
    x = np.concatenate([x.astype(dt, copy=False) for x in xs])
    return x, pd.concat(metas, ignore_index=True)
