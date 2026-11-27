"""T035: the VLM's own prefix self-attention (prefix tokens → prefix tokens), per layer and call.

Replays the T028 episodes (rich_acts/manifest.csv: same replay inputs, same batch construction, same sub-batching)
through a **prefix-only** pi05 forward (embed_prefix → PaliGemma language model with eager attention, exactly the
first half of PI05Pytorch.sample_actions; no denoising, so no noise). Forward hooks on every VLM decoder layer read
the post-softmax attention weights that GemmaAttention returns (B, 8 heads, P, P) and reduce them on the GPU:

  ① seg_attn (C, 18, 8, 6, 6) f32  per head; row = query segment, column = key segment; value = mean over the
                                   query segment's tokens of the attention summed over the key segment (rows sum to 1)
  ② i2p      (C, 18, n_i, 512) f16 head-mean attention, instruction (task segment) tokens → 512 image patches
     p2i     (C, 18, 512, n_i) f16 head-mean attention, 512 image patches → instruction tokens
  ③ full     (C, 18, V, V) f16     head-mean attention over all valid prefix tokens (standard episodes only;
                                   written to a separate file)

Segments (T028 vlm_seg / seg_ntok order): 0 agentview (256 patches), 1 wrist (256), 2 BOS, 3 task (instruction),
4 state, 5 template. The empty third camera and the language padding are excluded (they are masked keys: their
attention weight is exactly 0; the check `pad_mass` confirms this).

Checks per episode (in the log and the output meta): the hooks only observe — the VLM layer outputs, reduced like
T028 (segment mean → f16), are compared to T028 `vlm_seg` (bit-equal / Pearson r / max diff); head-mean rows sum
to 1; ① rebuilt from ③ equals the stored ① (full episodes).

Subcommands:
  manifest  rich_acts manifest → <small>/manifest.csv (+ full flag, Goal pool, output paths)
  extract   GPU. --phase smoke|main
  summary   GPU-h, disk, checks

  python vlm_attn_extract.py manifest
  CUDA_VISIBLE_DEVICES=1 python vlm_attn_extract.py extract --phase smoke --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

SHORTCUT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "serve"), str(SHORTCUT / "parallel"), str(SHORTCUT / "analysis")]

from rich_extract import _index_batch, _pearson, load_obs, prompt_layout  # noqa: E402
from serve.activations import IMG_TOKENS, PAD, PREFIX_SEGMENTS  # noqa: E402

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

RICH = Path("/outputs/VLA-shortcut-exps/rich_acts")
SMALL = Path("/outputs/VLA-shortcut-exps/vlm_attn")
BIG = Path("/data0/woosung/datasets/VLA-shortcut-exps/vlm_attn")
N_SEG = len(PREFIX_SEGMENTS)  # 6
N_IMG_PATCH = 2 * IMG_TOKENS  # 512: agentview + wrist
TASK_SEG = 3
# ③ only for the standard training-task episodes (human decision 2026-11-27)
FULL_JOBS = {("goal", "ours-goal-custom-train10"), ("libero3", "ours-obj-custom-train10"),
             ("libero3", "ours-spat-custom-train10")}  # fmt: skip
# Goal pools (T029/T030/T032 definition, layer_pop.py): cand outcome → pool
FALLBACK = ("trained", "fail_trained")
NEW = ("new", "new_byproduct")
F16_MAX = 65000.0


# ----------------------------------------------------------------------------------------------- manifest


def goal_pool(r: dict[str, str]) -> str:
    if r["set"] != "goal":
        return ""
    if r["condition"] == "train":
        return "train"
    return "cand_fallback" if r["outcome"] in FALLBACK else "cand_new" if r["outcome"] in NEW else "cand_other"


def cmd_manifest(a: argparse.Namespace) -> None:
    with open(RICH / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    keep = [c for c in rows[0] if c not in ("pilot", "pca_fit", "probe_anchor", "probe_calls", "path")]
    out = []
    for r in rows:
        full = int((r["set"], r["job"]) in FULL_JOBS)
        out.append({**{c: r[c] for c in keep}, "goal_pool": goal_pool(r), "full": full,
                    "rich_path": r["path"], "path": f"ep/{r['path']}",
                    "full_path": f"full/{r['path']}" if full else ""})  # fmt: skip
    SMALL.mkdir(parents=True, exist_ok=True)
    p = SMALL / "manifest.csv"
    if p.exists() and not a.force:
        raise SystemExit(f"{p} exists (use --force to rebuild the identical manifest)")
    tmp = p.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0]))
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, p)
    n_full = sum(r["full"] for r in out)
    print(f"{len(out)} episodes, full {n_full}; pools", {k: sum(r["goal_pool"] == k for r in out)
          for k in ("train", "cand_new", "cand_fallback", "cand_other")})  # fmt: skip


def read_manifest() -> list[dict[str, str]]:
    with open(SMALL / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------------------------- reduction


def reduce_attn(
    w: torch.Tensor, seg: torch.Tensor, instr_pos: torch.Tensor, vidx: torch.Tensor | None, vvalid: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    """One layer's attention weights → ①②(③).

    w: (B, H, P, P) post-softmax weights (any float dtype); seg: (B, P) segment id per prefix token (0..5, PAD);
    instr_pos: (n_i,) prefix positions of the instruction tokens (same in every call of an episode);
    vidx / vvalid: (B, V) prefix position of each valid token (padded) / its validity, for ③ (None = skip).
    """
    wf = w.float()
    B, H, P, _ = wf.shape  # noqa: N806
    S = torch.nn.functional.one_hot(seg[:, :P].clamp(max=PAD), PAD + 1).float()  # (B, P, 8)  # noqa: N806
    mass = torch.einsum("bhqk,bks->bhqs", wf, S)  # (B, H, P, 8): each query's attention summed per key segment
    cnt = S.sum(1)  # (B, 8) tokens per segment
    seg_sum = torch.einsum("bqr,bhqs->bhrs", S, mass)  # (B, H, 8 query seg, 8 key seg)
    seg_attn = seg_sum / cnt.clamp(min=1)[:, None, :, None]
    q_valid = S[..., :N_SEG].sum(-1)  # (B, P) 1 for valid queries
    pad_mass = (mass[..., N_SEG:].sum(-1) * q_valid[:, None]).amax((1, 2))  # (B,) max weight on excluded keys
    wm = wf.mean(1)  # (B, P, P) head mean
    row_err = ((wm.sum(-1) - 1).abs() * q_valid).amax(1)  # (B,) each valid query row over all keys
    out = {
        "seg_attn": seg_attn[:, :, :N_SEG, :N_SEG],
        "pad_mass": pad_mass,
        "row_err": row_err,
        "i2p": wm[:, instr_pos][:, :, :N_IMG_PATCH],
        "p2i": wm[:, :N_IMG_PATCH][:, :, instr_pos],
    }
    if vidx is not None and vvalid is not None:
        V = vidx.shape[1]  # noqa: N806
        r = torch.gather(wm, 1, vidx[:, :, None].expand(B, V, P))
        full = torch.gather(r, 2, vidx[:, None, :].expand(B, V, V))
        m = (vvalid[:, :, None] & vvalid[:, None, :]).float()
        out["full"] = full * m
    return out


def seg_from_full(full: np.ndarray, fseg: np.ndarray, fvalid: np.ndarray) -> np.ndarray:
    """③ (C, L, V, V) + segment id per valid token → head-mean ① (C, L, 6, 6) (check: equals stored ① head mean)."""
    C, L, V, _ = full.shape  # noqa: N806
    S = np.zeros((C, V, N_SEG), np.float64)  # noqa: N806
    for c in range(C):
        v = fvalid[c]
        S[c, np.nonzero(v)[0], fseg[c, v]] = 1
    St = torch.from_numpy(S)  # noqa: N806
    f = torch.from_numpy(full).double()
    sums = St.transpose(1, 2)[:, None] @ (f @ St[:, None])  # (C, L, 6, 6)
    return (sums / St.sum(1)[:, None, :, None].clamp(min=1)).numpy()


# ----------------------------------------------------------------------------------------------- hooks


class VlmAttnHooks:
    """Forward hooks on the VLM decoder layers: attention weights → reduce_attn; layer output → T028 vlm_seg check."""

    def __init__(self, policy: Any):
        lm = policy.model.paligemma_with_expert.paligemma.model.language_model
        self.layers = lm.layers
        self.active = False
        self.handles = []
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.register_forward_hook(self._attn(i)))
            self.handles.append(layer.register_forward_hook(self._out(i)))

    def start(self, seg: torch.Tensor, instr_pos: torch.Tensor, vidx: torch.Tensor | None, vvalid: Any) -> None:
        self.seg, self.instr_pos, self.vidx, self.vvalid = seg, instr_pos, vidx, vvalid
        self.res: dict[int, dict[str, torch.Tensor]] = {}
        self.vlm_seg: dict[int, torch.Tensor] = {}
        self.active = True

    def _attn(self, i: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            if not self.active:
                return
            w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if w is None:
                raise RuntimeError("VLM attention returned no weights (not eager)")
            r = reduce_attn(w, self.seg, self.instr_pos, self.vidx, self.vvalid)
            if "full" in r:
                r["full"] = r["full"].half().cpu()
            self.res[i] = r

        return hook

    def _seg_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Same as serve.activations.Hooks._seg_mean (3-D case) → T028 vlm_seg."""
        seg = self.seg[:, : x.shape[1]]
        idx = seg[..., None].expand(-1, -1, x.shape[2])
        out = torch.zeros(x.shape[0], PAD + 1, x.shape[2], device=x.device).scatter_add_(1, idx, x.float())
        cnt = torch.zeros(x.shape[0], PAD + 1, device=x.device).scatter_add_(
            1, seg, torch.ones(seg.shape, device=x.device)
        )
        return out / cnt.clamp(min=1)[..., None]

    def _out(self, i: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            if self.active:
                h = out[0] if isinstance(out, tuple) else out
                self.vlm_seg[i] = self._seg_mean(h)[:, :N_SEG]

        return hook

    def stop(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        self.active = False
        n = len(self.layers)
        if sorted(self.res) != list(range(n)) or sorted(self.vlm_seg) != list(range(n)):
            raise RuntimeError(f"hooks saw layers {sorted(self.res)}")
        keys = self.res[0].keys()
        out = {k: torch.stack([self.res[i][k] for i in range(n)], 1) for k in keys}
        return out, torch.stack([self.vlm_seg[i] for i in range(n)], 1)


def prefix_forward(policy: Any, sub: dict[str, Any]) -> None:
    """The prefix half of PI05Pytorch.sample_actions (as called by predict_action_chunk), no denoising."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    model = policy.model
    policy.eval()
    images, img_masks = policy._preprocess_images(sub)
    states, state_masks = policy._prepare_memory_states(sub)
    tokens, masks = sub[OBS_LANGUAGE_TOKENS], sub[OBS_LANGUAGE_ATTENTION_MASK]
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks, states, state_masks
    )
    att2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    mask_dtype = prefix_embs.dtype if getattr(model, "use_typed_attention_masks", False) else None
    m4 = model._prepare_attention_masks_4d(att2d, dtype=mask_dtype)
    lm = model.paligemma_with_expert.paligemma.model.language_model
    if hasattr(lm, "config"):
        lm.config._attn_implementation = "eager"  # noqa: SLF001  (as sample_actions does)
    model.paligemma_with_expert.forward(
        attention_mask=m4, position_ids=pos, past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True
    )


# ----------------------------------------------------------------------------------------------- extraction


def save_npz(path: Path, arrays: dict[str, np.ndarray], level: int) -> None:
    """np.savez_compressed equivalent (lossless deflate, np.load reads it) with a chosen compression level."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level) as z:
            for k, v in arrays.items():
                with z.open(k + ".npy", "w", force_zip64=True) as f:
                    np.lib.format.write_array(f, np.asanyarray(v), allow_pickle=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def tok_strings(tok: Any, ids: list[int]) -> list[str]:
    return [str(s) for s in tok.convert_ids_to_tokens(ids)]


def run_episode(row: dict[str, Any], L: Any, hooks: VlmAttnHooks, tok: Any, dev: str, max_batch: int,  # noqa: N803
                full: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, dict[str, Any]]:  # fmt: skip
    from serve.engine import LIBERO_IMAGE_MAP, Engine

    d = row.get("_obs") or load_obs(row)
    C = len(d["t"])  # noqa: N806
    instruction = str(d["instruction"])
    obs = [{"image": d["image"][k], "image2": d["image2"][k], "state": d["state"][k]} for k in range(C)]
    batch = L.pre(Engine.wire_batch(SimpleNamespace(image_map=LIBERO_IMAGE_MAP), obs, [instruction] * C))
    policy = L.policy
    cfg = policy.config
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
    n_img = len(cfg.image_features)
    seg, ntok, lang_pos, lang_valid, lang_ids, lang_seg = prompt_layout(tok, tokens, masks, n_img, cfg.chunk_size)
    P = n_img * IMG_TOKENS + tokens.shape[1]  # noqa: N806  prefix length
    seg = seg[:, :P]
    # instruction tokens: same prefix positions in every call
    ipos = [tuple(lang_pos[c][torch.from_numpy(lang_seg[c] == TASK_SEG)].tolist()) for c in range(C)]
    if len(set(ipos)) != 1:
        raise RuntimeError(f"instruction token positions differ between calls: {set(ipos)}")
    instr_pos = torch.tensor(ipos[0], dtype=torch.long)
    instr_ids = lang_ids[0][lang_seg[0] == TASK_SEG].tolist()
    # valid prefix tokens per call (prefix order): 512 patches + the non-pad language tokens
    vpos = [torch.cat([torch.arange(N_IMG_PATCH), lang_pos[c][lang_valid[c]]]) for c in range(C)]
    V = max(len(v) for v in vpos)  # noqa: N806
    vidx = torch.zeros(C, V, dtype=torch.long)
    vvalid = torch.zeros(C, V, dtype=torch.bool)
    for c, v in enumerate(vpos):
        vidx[c, : len(v)], vvalid[c, : len(v)] = v, True
    seg_d, instr_d, vidx_d, vvalid_d = seg.to(dev), instr_pos.to(dev), vidx.to(dev), vvalid.to(dev)
    parts: dict[str, list[Any]] = defaultdict(list)
    vseg = []
    n_sub = -(-C // max_batch)
    size = -(-C // n_sub)  # T028 sub-batching (52 calls → 26 + 26): same batch shapes, same numerics
    t0 = time.time()
    for lo in range(0, C, size):
        idx = list(range(lo, min(C, lo + size)))
        sub = _index_batch(batch, idx, C) if len(idx) < C else batch
        hooks.start(seg_d[idx], instr_d, vidx_d[idx] if full else None, vvalid_d[idx] if full else None)
        try:
            with torch.inference_mode():
                prefix_forward(policy, sub)
        except BaseException:
            hooks.active = False
            raise
        r, vs = hooks.stop()
        for k, v in r.items():
            parts[k].append(v if k == "full" else v.cpu())
        vseg.append(vs.cpu())
    torch.cuda.synchronize()
    sec = time.time() - t0
    res = {k: torch.cat(v) for k, v in parts.items()}
    vlm_seg = torch.cat(vseg).numpy()
    vlm_seg16 = vlm_seg.astype(np.float16) if np.abs(vlm_seg).max() <= F16_MAX else vlm_seg
    ep = {
        "seg_attn": res["seg_attn"].numpy().astype(np.float32),
        "i2p": res["i2p"].numpy().astype(np.float16),
        "p2i": res["p2i"].numpy().astype(np.float16),
        "seg_ntok": ntok[:, :N_SEG],
        "instr_pos": instr_pos.numpy().astype(np.int16),
        "instr_ids": np.asarray(instr_ids, np.int32),
        "instr_tok": np.asarray(tok_strings(tok, instr_ids)),
        "lang_ids": lang_ids, "lang_seg": lang_seg, "lang_valid": lang_valid.numpy(),
        "lang_pos": (lang_pos * lang_valid).numpy().astype(np.int16),
        "t": d["t"],
    }  # fmt: skip
    info: dict[str, Any] = {
        "calls": C, "n_instr": len(instr_ids), "P": P, "V_max": V, "sec_gpu": round(sec, 2),
        "pad_mass_max": float(res["pad_mass"].max()), "row_err_max": float(res["row_err"].max()),
        "seg_row_err_max": float((res["seg_attn"].sum(-1) - 1).abs().max()),
    }  # fmt: skip
    # T028 agreement: VLM layer outputs (segment means) vs rich_acts vlm_seg
    rp = RICH / row["rich_path"]
    with np.load(rp) as z:
        old = z["vlm_seg"]
    info["vlm_seg_bit_equal"] = bool(old.dtype == vlm_seg16.dtype and np.array_equal(old, vlm_seg16))
    info["vlm_seg_r"] = _pearson(old.astype(np.float32), vlm_seg)
    info["vlm_seg_max_diff"] = float(np.abs(old.astype(np.float32) - vlm_seg16.astype(np.float32)).max())
    info["vlm_seg_frac_equal"] = float((old.astype(np.float32) == vlm_seg16.astype(np.float32)).mean())
    fz = None
    if full:
        f16 = res["full"].numpy()  # (C, L, V, V) fp16, zeros outside the valid tokens
        fseg = np.full((C, V), -1, np.int8)
        frow = np.full((C, V), -1, np.int8)
        fcol = np.full((C, V), -1, np.int8)
        ftok = np.full((C, V), -1, np.int32)
        fstr = np.full((C, V), "", dtype="<U24")
        for c in range(C):
            n = len(vpos[c])
            fseg[c, :n] = seg[c, vpos[c]].numpy()
            fseg[c, :IMG_TOKENS], fseg[c, IMG_TOKENS:N_IMG_PATCH] = 0, 1
            p = np.arange(N_IMG_PATCH) % IMG_TOKENS
            frow[c, :N_IMG_PATCH], fcol[c, :N_IMG_PATCH] = p // 16, p % 16
            ids = lang_ids[c][lang_valid[c].numpy()]
            ftok[c, N_IMG_PATCH:n] = ids
            fstr[c, N_IMG_PATCH:n] = tok_strings(tok, ids.tolist())
        fvalid = vvalid.numpy()
        rebuilt = seg_from_full(f16, np.where(fvalid, fseg, 0), fvalid)
        info["full_vs_seg_max_diff"] = float(np.abs(rebuilt - ep["seg_attn"].mean(2)).max())
        info["full_row_err_max"] = float(np.abs(f16.astype(np.float32).sum(-1) - 1)[np.broadcast_to(fvalid[:, None], f16.shape[:3])].max())  # noqa: E501
        fz = {"full": f16, "full_valid": fvalid, "full_seg": fseg, "full_pos": vidx.numpy().astype(np.int16),
              "full_row": frow, "full_col": fcol, "full_tok_id": ftok, "full_tok": fstr, "t": d["t"]}  # fmt: skip
    meta = {
        "episode": row["episode"], "set": row["set"], "job": row["job"], "instruction": instruction,
        "segments": PREFIX_SEGMENTS, "n_layers": int(ep["seg_attn"].shape[1]), "n_heads": int(ep["seg_attn"].shape[2]),
        "head_reduction": "mean over the 8 heads (sum = 8 x mean)", "attention": "eager, post-softmax, prefix pass",
        "replay_machine": "VGI-102 RTX 4090 (GPU 1)", "info": info,
    }  # fmt: skip
    ep["meta"] = np.asarray(json.dumps(meta, default=str))
    if fz is not None:
        fz["meta"] = ep["meta"]
    return ep, fz, info


def load_model(dev: str) -> tuple[Any, Any]:
    from rich_extract import load_model as lm

    return lm(dev)


def cmd_extract(a: argparse.Namespace) -> None:
    rows = read_manifest()
    if a.sets:
        rows = [r for r in rows if r["set"] in a.sets.split(",")]
    if a.episodes:
        want = set(a.episodes.split(","))
        rows = [r for r in rows if r["episode"] in want or r["rich_path"].removesuffix(".npz") in want]
    if a.full_only:
        rows = [r for r in rows if r["full"] == "1"]
    if a.limit:
        rows = rows[: a.limit]
    dst = Path(a.dst) if a.dst else BIG
    todo = [r for r in rows if not ((dst / r["path"]).exists() and (r["full"] != "1" or (dst / r["full_path"]).exists()))]  # noqa: E501
    dev = "cuda"
    tic = time.time()
    L, tok = load_model(dev)  # noqa: N806
    hooks = VlmAttnHooks(L.policy)
    print(f"[vlm-attn] {a.phase}: {len(todo)}/{len(rows)} episodes to do; load {time.time() - tic:.0f}s", flush=True)
    log_p = dst / f"extract_{a.phase}.jsonl"
    log_p.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rd = ThreadPoolExecutor(1)
    wr = ThreadPoolExecutor(a.writers)
    nxt = rd.submit(load_obs, todo[0]) if todo else None
    pending: list[Future] = []

    def write(r: dict[str, Any], ep: dict, fz: dict | None, info: dict, t0: float, t1: float, i: int, log: Any) -> None:
        tw = time.time()
        save_npz(dst / r["path"], ep, a.level)
        mb = (dst / r["path"]).stat().st_size / 2**20
        mbf = 0.0
        if fz is not None:
            save_npz(dst / r["full_path"], fz, a.level)
            mbf = (dst / r["full_path"]).stat().st_size / 2**20
        line = {"episode": r["episode"], "set": r["set"], "job": r["job"], "full": int(fz is not None), **info,
                "sec_ep": round(t1 - t0, 2), "sec_write": round(time.time() - tw, 2), "mb": round(mb, 2),
                "mb_full": round(mbf, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
        log.write(json.dumps(line) + "\n")
        log.flush()
        print(f"[vlm-attn] {i}/{len(todo)} {r['set']}/{r['episode']} C={info['calls']} gpu {info['sec_gpu']}s "
              f"ep {line['sec_ep']}s write {line['sec_write']}s {mb:.1f}+{mbf:.0f}MB "
              f"vlm_seg bit={info['vlm_seg_bit_equal']} r={info['vlm_seg_r']:.6f} pad={info['pad_mass_max']:.1e} "
              f"row={info['row_err_max']:.1e} full-seg={info.get('full_vs_seg_max_diff', float('nan')):.1e}",
              flush=True)  # fmt: skip

    with open(log_p, "a") as log:
        for i, r in enumerate(todo, 1):
            if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
                print(f"[vlm-attn] stop: --max-hours {a.max_hours} reached", flush=True)
                break
            t0 = time.time()
            r = {**r, "_obs": nxt.result()}
            nxt = rd.submit(load_obs, todo[i]) if i < len(todo) else None
            ep, fz, info = run_episode(r, L, hooks, tok, dev, a.max_batch, r["full"] == "1")
            t1 = time.time()
            r.pop("_obs")
            # bounded in-flight writes (③ files are ~0.3 GB in memory each)
            while len(pending) >= a.writers:
                pending.pop(0).result()
            pending.append(wr.submit(write, r, ep, fz, info, t0, t1, i, log))
            if i == 1:
                print(f"[vlm-attn] peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        for p in pending:
            p.result()
    rd.shutdown()
    wr.shutdown()
    print(f"[vlm-attn] done {len(todo)} in {(time.time() - t_start) / 3600:.3f} h", flush=True)


# ----------------------------------------------------------------------------------------------- summary


def cmd_summary(a: argparse.Namespace) -> None:
    dst = Path(a.dst) if a.dst else BIG
    lines = []
    for p in sorted(dst.glob("extract_*.jsonl")):
        lines += [json.loads(x) | {"phase": p.stem[8:]} for x in p.read_text().splitlines()]
    if not lines:
        raise SystemExit("no logs")
    rows = {(r["set"], r["job"], r["episode"]): r for r in read_manifest()}
    a100 = [x for x in lines if rows[x["set"], x["job"], x["episode"]]["machine"] == "A100"]
    r4090 = [x for x in lines if rows[x["set"], x["job"], x["episode"]]["machine"] == "4090"]
    du_ep = sum(f.stat().st_size for f in (dst / "ep").rglob("*.npz")) if (dst / "ep").exists() else 0
    du_full = sum(f.stat().st_size for f in (dst / "full").rglob("*.npz")) if (dst / "full").exists() else 0
    full = [x for x in lines if x["full"]]
    s = {
        "episodes": len(lines), "full_episodes": len(full), "calls": sum(x["calls"] for x in lines),
        "wall_h_episodes": round(sum(x["sec_ep"] for x in lines) / 3600, 3),
        "gpu_forward_h": round(sum(x["sec_gpu"] for x in lines) / 3600, 3),
        "disk_gb_ep": round(du_ep / 2**30, 2), "disk_gb_full": round(du_full / 2**30, 2),
        "mb_per_call_ep": round(sum(x["mb"] for x in lines) / sum(x["calls"] for x in lines), 3),
        "mb_per_call_full": round(sum(x["mb_full"] for x in full) / max(1, sum(x["calls"] for x in full)), 2),
        "vlm_seg_bit_equal_4090": f"{sum(x['vlm_seg_bit_equal'] for x in r4090)}/{len(r4090)}",
        "vlm_seg_bit_equal_a100": f"{sum(x['vlm_seg_bit_equal'] for x in a100)}/{len(a100)}",
        "vlm_seg_r_min": min(x["vlm_seg_r"] for x in lines),
        "vlm_seg_r_min_a100": min((x["vlm_seg_r"] for x in a100), default=None),
        "vlm_seg_max_diff_max": max(x["vlm_seg_max_diff"] for x in lines),
        "pad_mass_max": max(x["pad_mass_max"] for x in lines), "row_err_max": max(x["row_err_max"] for x in lines),
        "seg_row_err_max": max(x["seg_row_err_max"] for x in lines),
        "full_vs_seg_max_diff": max((x["full_vs_seg_max_diff"] for x in full), default=None),
        "full_row_err_max": max((x["full_row_err_max"] for x in full), default=None),
    }  # fmt: skip
    print(json.dumps(s, indent=1))
    if a.write:
        (SMALL / "summary.json").write_text(json.dumps(s, indent=1))


def main() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))  # task rule: ≤ 8 CPU workers
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("manifest")
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("extract")
    p.add_argument("--phase", choices=["smoke", "main"], required=True)
    p.add_argument("--dst", default=None, help="write here instead of the big-data root (smoke)")
    p.add_argument("--sets", default=None)
    p.add_argument("--episodes", default=None)
    p.add_argument("--full-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--writers", type=int, default=6, help="compression threads (≤ 8 CPU workers in total)")
    p.add_argument("--level", type=int, default=6, help="deflate level (lossless)")
    p = sp.add_parser("summary")
    p.add_argument("--dst", default=None)
    p.add_argument("--write", action="store_true")
    a = ap.parse_args()
    {"manifest": cmd_manifest, "extract": cmd_extract, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
