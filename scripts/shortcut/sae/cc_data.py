#!/usr/bin/env python
"""T034 crosscoder training caches (CPU): one (N, L, d) fp16 array per model, rows = LIBERO timesteps.

  python cc_data.py expert   # ae00–ae17, denoising step 9, from the T031 acts dirs (pi05lib_s2 + pi05lib_s2_rest)
  python cc_data.py vlm      # emb + pg00–pg17 pooled_all (and text-only), from the T034 VLM collection
  python cc_data.py check    # (episode, frame) keys of expert and VLM caches match row by row

Out: /data0/woosung/datasets/VLA-shortcut-exps/cache/t034_<name>/{x.npy, meta.parquet, info.json}.
Rows are in plan part order (part_id sorted), inside a part in the order collect.py wrote them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

DATA = Path("/data0/woosung/datasets/VLA-shortcut-exps")
AE_ACTS = [DATA / "sae/action_expert_acts/pi05lib_s2", DATA / "sae/action_expert_acts/pi05lib_s2_rest"]
VLM_ACTS = DATA / "sae/vlm_acts/pi05lib_s2"
CACHE = DATA / "cache"
META_COLS = ["index", "episode_index", "frame_index", "task_index", "suite"]
T030_MASSIVE = Path("/outputs/VLA-shortcut-exps/analysis/layer_pop/res/massive_channels.json")


def cache_dir(name: str) -> Path:
    return CACHE / f"t034_{name}"


def massive_from_rms(x: np.ndarray, ratio: float = 20.0) -> list[int]:
    """T030 rule: channels whose RMS > ratio × median channel RMS (x (N, d))."""
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(0))
    return [int(c) for c in np.where(rms > ratio * np.median(rms))[0]]


def parts_of(acts: Path) -> list[dict]:
    return sorted(store.done_parts(acts), key=lambda p: p["part_id"])


def build_expert(step: int = 9) -> None:
    out = cache_dir("expert")
    if (out / "info.json").exists():
        print(f"{out} exists; skip")
        return
    where: dict[int, Path] = {}
    for d in AE_ACTS:
        ident = store.read_plan(d)["identity"]
        assert ident["ae_steps"] == [0, 4, 9] and ident["frame_stride"] == 2, ident
        for nm in ident["layers"]:
            where.setdefault(int(nm[2:]), d)
    assert sorted(where) == list(range(18)), sorted(where)
    si = [0, 4, 9].index(step)
    p0 = parts_of(AE_ACTS[0])
    N = sum(p["rows"] for p in p0)
    out.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(out / "x.npy.tmp", mode="w+", dtype=np.float16, shape=(N, 18, 1024))
    metas = []
    row = 0
    for p in p0:
        d0 = Path(p["dir"])
        m = pd.read_parquet(d0 / "meta.parquet", columns=META_COLS)
        T = len(m)
        for L in range(18):
            d = where[L] / "parts" / p["part_id"]
            if L == 0 or where[L] != where[0]:
                m2 = pd.read_parquet(d / "meta.parquet", columns=META_COLS)
                assert (m2[["episode_index", "frame_index"]].to_numpy() == m[["episode_index", "frame_index"]].to_numpy()).all()
            a = np.load(d / f"ae{L:02d}.npy", mmap_mode="r")
            assert a.dtype == np.float16 and a.shape == (T, 3, 1024), (d, a.dtype, a.shape)
            x[row : row + T, L] = a[:, si]
        metas.append(m)
        row += T
        print(f"[expert] {p['part_id']} {row}/{N}", flush=True)
    x.flush()
    del x
    (out / "x.npy.tmp").rename(out / "x.npy")
    meta = pd.concat(metas, ignore_index=True)
    meta.to_parquet(out / "meta.parquet", index=False)
    mc = json.loads(T030_MASSIVE.read_text())
    info = {
        "what": f"action expert ae00-ae17 output, mean of 50 action tokens, denoising step {step}",
        "layers": [f"ae{L:02d}" for L in range(18)],
        "shape": [N, 18, 1024],
        "sources": [str(d) for d in AE_ACTS],
        "massive_channels": {f"ae{L:02d}": [c["ch"] for c in mc[f"e_s{step}/L{L}"]] for L in range(18)},
        "massive_rule": "T030 massive_channels.json e_s9 (RMS > 20 × median channel RMS)",
        "time": time.strftime("%F %T"),
    }
    (out / "info.json").write_text(json.dumps(info, indent=1) + "\n")


def build_vlm() -> None:
    """pooled_all (cam1, cam2, text count-weighted) and text-only for emb + pg00–pg17."""
    out = cache_dir("vlm")
    if (out / "info.json").exists():
        print(f"{out} exists; skip")
        return
    ident = store.read_plan(VLM_ACTS)["identity"]
    layers = ["emb"] + [f"pg{i:02d}" for i in range(18)]
    assert ident["layers"] == layers and ident["frame_stride"] == 2, ident
    parts = parts_of(VLM_ACTS)
    N = sum(p["rows"] for p in parts)
    out.mkdir(parents=True, exist_ok=True)
    xa = np.lib.format.open_memmap(out / "x.npy.tmp", mode="w+", dtype=np.float16, shape=(N, 19, 2048))
    xt = np.lib.format.open_memmap(out / "x_text.npy.tmp", mode="w+", dtype=np.float16, shape=(N, 19, 2048))
    metas, row = [], 0
    for p in parts:
        d = Path(p["dir"])
        m = pd.read_parquet(d / "meta.parquet", columns=META_COLS)
        T = len(m)
        cnt = np.load(d / "counts.npy")
        for li, nm in enumerate(layers):
            a = np.load(d / f"{nm}.npy", mmap_mode="r")
            xa[row : row + T, li] = store.pool_pg(a, cnt, "pooled_all")
            xt[row : row + T, li] = store.pool_pg(a, cnt, "text")
        metas.append(m)
        row += T
        print(f"[vlm] {p['part_id']} {row}/{N}", flush=True)
    xa.flush()
    xt.flush()
    del xa, xt
    (out / "x.npy.tmp").rename(out / "x.npy")
    (out / "x_text.npy.tmp").rename(out / "x_text.npy")
    meta = pd.concat(metas, ignore_index=True)
    meta.to_parquet(out / "meta.parquet", index=False)
    x = np.load(out / "x.npy", mmap_mode="r")
    sub = np.sort(np.random.default_rng(0).choice(N, size=min(N, 20000), replace=False))
    xs = np.asarray(x[sub], dtype=np.float32)
    info = {
        "what": "PaliGemma emb + pg00-pg17 output, pooled_all = mean over valid prefix tokens (x.npy); text-only (x_text.npy)",
        "layers": layers,
        "shape": [N, 19, 2048],
        "sources": [str(VLM_ACTS)],
        "massive_channels": {nm: massive_from_rms(xs[:, li]) for li, nm in enumerate(layers)},
        "massive_rule": "T030 rule on 20 000 random pooled_all rows: RMS > 20 × median channel RMS",
        "time": time.strftime("%F %T"),
    }
    (out / "info.json").write_text(json.dumps(info, indent=1) + "\n")


def check() -> dict:
    me = pd.read_parquet(cache_dir("expert") / "meta.parquet")
    mv = pd.read_parquet(cache_dir("vlm") / "meta.parquet")
    same = len(me) == len(mv) and bool(
        (me[["episode_index", "frame_index", "index"]].to_numpy() == mv[["episode_index", "frame_index", "index"]].to_numpy()).all()
    )
    res = {"n_expert": len(me), "n_vlm": len(mv), "rows_match": same,
           "all_even_frames": bool((mv["frame_index"] % 2 == 0).all())}
    print(res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("expert", "vlm", "check"))
    a = ap.parse_args()
    {"expert": build_expert, "vlm": build_vlm, "check": check}[a.what]()
