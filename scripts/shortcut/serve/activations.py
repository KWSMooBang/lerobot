"""Opt-in activation recording for the policy server (`policy_server.py --record-activations <dir>`, T023).

Forward hooks on a pi0-family policy (PaliGemma + action expert: pi05, pi0) record, at every policy call,
the same per-call features as the replay extraction of T021 (kubectl/libero90/_raw/attn_extract.py), plus
a few hidden-state vectors for population-dynamics analyses. The hooks only read tensors (no in-place ops,
no RNG), so actions are unchanged with recording on.

Per call (C calls in an episode; S = 7 input types, see SEGMENTS; n = denoising steps; Lx/L/Lv = expert /
VLM / SigLIP layers):
  attn_mass  (C, n, Lx, S)   share of the action queries' attention on each input type (mean over heads)
  attn_head  (C, Lx, H, S)   same per head, mean over denoising steps
  attn_img   (C, 3, 512)     f16, attention on the two cameras' 2×256 patches per layer group (0-5/6-11/12-17)
  exp_h, exp_mlp (C, n, Lx)  action expert: mean token L2 norm of the layer output / mean |MLP activation|
  vlm_h, vlm_mlp (C, L, 6)   VLM prefix pass per layer and prefix segment: hidden-state norm / mean |MLP act|
  sig_h      (C, n_img, Lv)  SigLIP mean token L2 norm per camera and layer
  act_pred   (C, n_action_steps, 7)  the executed chunk (post-processed, what the client received)
  exp_vec    (C, n, len(VEC_LAYERS), D_exp) f16  expert hidden state, mean over the action tokens
  vlm_vec    (C, len(VEC_LAYERS), 4, D_vlm) f32  VLM hidden state, mean over agentview / wrist / task / state
  seg_ntok   (C, S)          number of keys per input type
  t          (C,)            env step of each call (= call index × n_action_steps; chunk clients)
  meta       json string     episode id, record key, seed, instruction(s), client meta, policy / layer info

One npz per episode, written at `end` (or when the connection closes) to <dir>/<record_key>.npz, where
record_key is what the client sent in `reset` (default: the episode id), so a benchmark worker can put the
same key into its episodes.jsonl row. Benchmark-agnostic: nothing here knows about LIBERO tasks.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

SEGMENTS = ["img_agentview", "img_wrist", "bos", "task", "state", "template", "action"]
PREFIX_SEGMENTS = SEGMENTS[:6]
PAD = len(SEGMENTS)  # masked language padding: dropped
IMG_TOKENS = 256
LAYER_GROUPS = [(0, 6), (6, 12), (12, 18)]
VEC_LAYERS = (5, 11, 17)  # last layer of each group (expert and VLM both have 18)
VEC_SEGMENTS = (0, 1, 3, 4)  # agentview, wrist, task, state
F16_KEYS = ("attn_img", "exp_vec")  # vlm_vec stays f32: Gemma residual means reach ~2e4 (f16 max 6.5e4)
_SAFE = re.compile(r"[^A-Za-z0-9._=+-]")


def safe_key(key: str) -> str:
    """Client record key → relative path inside the recording dir ('/' kept, no '..', no absolute)."""
    parts = [_SAFE.sub("_", p) for p in str(key).split("/") if p not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"empty record key {key!r}")
    return "/".join(parts)


def lang_segments(tok: Any, ids: list[int]) -> list[int]:
    """Segment id per non-pad language token of pi05's prompt 'Task: <task>, State: <nums>;\\nAction: '."""
    seg = []
    body = ids[1:] if ids and ids[0] == tok.bos_token_id else ids
    text = tok.decode(body)
    a = text.index("Task: ") + len("Task: ")
    b = text.index(", State: ")
    c = b + len(", State: ")
    d = text.index(";\nAction")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != body:
        raise ValueError(f"re-tokenized prompt differs from the batch tokens: {text!r}")
    if len(body) != len(ids):
        seg.append(2)
    for s, e in enc["offset_mapping"]:
        mid = (s + e) / 2
        seg.append(3 if a <= mid < b else 4 if c <= mid < d else 5)
    return seg


def find_tokenizer(pre: Any) -> Any:
    for step in getattr(pre, "steps", ()):
        tok = getattr(step, "input_tokenizer", None)
        if tok is not None:
            return tok
    raise SystemExit("--record-activations: no tokenizer step in the policy preprocessor")


class Hooks:
    """Forward hooks on pi0-family models; buffers (re)allocated per call by start()."""

    def __init__(self, policy: Any):
        try:
            m = policy.model.paligemma_with_expert
            self.vision_layers = m.paligemma.model.vision_tower.vision_model.encoder.layers
            self.vlm_layers = m.paligemma.model.language_model.layers
            self.exp_layers = m.gemma_expert.model.layers
        except AttributeError as e:
            raise SystemExit(
                f"--record-activations supports pi0-family policies (PaliGemma + expert), not {type(policy).__name__}"
            ) from e
        self.n_heads = self.exp_layers[0].self_attn.config.num_attention_heads
        self.active = False
        self.handles = []
        for i, layer in enumerate(self.vision_layers):
            self.handles.append(layer.register_forward_hook(self._vision(i)))
        for i, layer in enumerate(self.vlm_layers):
            self.handles.append(layer.register_forward_hook(self._vlm_h(i)))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._vlm_mlp(i)))
        # denoise_step calls gemma_expert.model's layers directly: the first layer marks each denoising step
        self.handles.append(self.exp_layers[0].register_forward_pre_hook(self._exp_step))
        for i, layer in enumerate(self.exp_layers):
            self.handles.append(layer.self_attn.register_forward_hook(self._attn(i)))
            self.handles.append(layer.register_forward_hook(self._exp_h(i)))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._exp_mlp(i)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def start(self, seg: torch.Tensor, n_img: int, n_steps: int, chunk: int) -> None:
        """seg: (B, K) segment id per key (prefix + suffix)."""
        b = seg.shape[0]
        dev = seg.device
        self.seg, self.n_img, self.step, self.n_steps, self.chunk = seg, n_img, -1, n_steps, chunk
        n_vlm, n_exp, n_vis = len(self.vlm_layers), len(self.exp_layers), len(self.vision_layers)
        z = torch.zeros
        self.buf = {
            "attn_mass": z(b, n_steps, n_exp, PAD, device=dev),
            "attn_head": z(b, n_exp, self.n_heads, PAD, device=dev),
            "attn_img": z(b, len(LAYER_GROUPS), n_img * IMG_TOKENS, device=dev),
            "vlm_h": z(b, n_vlm, len(PREFIX_SEGMENTS), device=dev),
            "vlm_mlp": z(b, n_vlm, len(PREFIX_SEGMENTS), device=dev),
            "exp_h": z(b, n_steps, n_exp, device=dev),
            "exp_mlp": z(b, n_steps, n_exp, device=dev),
            "sig_h": z(b, n_img, n_vis, device=dev),
        }
        self.exp_vec: dict[tuple[int, int], torch.Tensor] = {}
        self.vlm_vec: dict[int, torch.Tensor] = {}
        self.active = True

    def stop(self) -> dict[str, np.ndarray]:
        self.active = False
        if self.step != self.n_steps - 1:
            raise RuntimeError(f"expected {self.n_steps} denoising steps, hooks saw {self.step + 1}")
        out = {k: v.cpu().numpy() for k, v in self.buf.items()}
        out["attn_img"] = out["attn_img"][:, :, : 2 * IMG_TOKENS].astype(np.float16)
        out["exp_vec"] = (
            torch.stack(
                [torch.stack([self.exp_vec[s, i] for i in VEC_LAYERS], 1) for s in range(self.n_steps)], 1
            )
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        out["vlm_vec"] = torch.stack([self.vlm_vec[i] for i in VEC_LAYERS], 1).cpu().numpy()
        return out

    def _seg_mean(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, P) or (B, P, D) per prefix token → (B, PAD + 1[, D]) mean over each segment's tokens."""
        seg = self.seg[:, : x.shape[1]]
        if x.dim() == 3:
            idx = seg[..., None].expand(-1, -1, x.shape[2])
            out = torch.zeros(x.shape[0], PAD + 1, x.shape[2], device=x.device).scatter_add_(
                1, idx, x.float()
            )
            cnt = torch.zeros(x.shape[0], PAD + 1, device=x.device).scatter_add_(
                1, seg, torch.ones(seg.shape, device=x.device)
            )
            return out / cnt.clamp(min=1)[..., None]
        out = torch.zeros(x.shape[0], PAD + 1, device=x.device).scatter_add_(1, seg, x.float())
        cnt = torch.zeros_like(out).scatter_add_(1, seg, torch.ones_like(x, dtype=torch.float32))
        return out / cnt.clamp(min=1)

    def _vision(self, i: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            if not self.active:
                return
            h = out[0] if isinstance(out, tuple) else out
            n = h.float().norm(dim=-1).mean(-1)  # (n_img * B,)
            self.buf["sig_h"][:, :, i] = n.view(self.n_img, -1).T

        return hook

    def _vlm_h(self, i: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            if not self.active:
                return
            h = out[0] if isinstance(out, tuple) else out
            self.buf["vlm_h"][:, i] = self._seg_mean(h.float().norm(dim=-1))[:, : len(PREFIX_SEGMENTS)]
            if i in VEC_LAYERS:
                self.vlm_vec[i] = self._seg_mean(h)[:, list(VEC_SEGMENTS)]

        return hook

    def _vlm_mlp(self, i: int) -> Any:
        def hook(_mod: Any, inp: Any) -> None:
            if self.active:
                self.buf["vlm_mlp"][:, i] = self._seg_mean(inp[0].float().abs().mean(-1))[
                    :, : len(PREFIX_SEGMENTS)
                ]

        return hook

    def _exp_step(self, _mod: Any, _args: Any) -> None:
        if self.active:
            self.step += 1

    def _attn(self, i: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            if not self.active:
                return
            w = out[1]
            if w is None:
                raise RuntimeError("expert attention returned no weights (not eager)")
            b, h, _q, k = w.shape
            wq = w.float().mean(2)  # (b, h, k): mean over the action queries
            seg = self.seg[:, None, :k].expand(b, h, k)
            mass = torch.zeros(b, h, PAD + 1, device=w.device).scatter_add_(2, seg, wq)[..., :PAD]
            self.buf["attn_mass"][:, self.step, i] = mass.mean(1)
            self.buf["attn_head"][:, i] += mass / self.n_steps
            g = next(k for k, (lo, hi) in enumerate(LAYER_GROUPS) if lo <= i < hi)
            lo, hi = LAYER_GROUPS[g]
            n_img_keys = self.n_img * IMG_TOKENS
            self.buf["attn_img"][:, g] += wq[:, :, :n_img_keys].mean(1) / (self.n_steps * (hi - lo))

        return hook

    def _exp_h(self, i: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            if not self.active:
                return
            h = out[0] if isinstance(out, tuple) else out
            self.buf["exp_h"][:, self.step, i] = h.float().norm(dim=-1).mean(-1)
            if i in VEC_LAYERS:
                self.exp_vec[self.step, i] = h[:, -self.chunk :].float().mean(1)

        return hook

    def _exp_mlp(self, i: int) -> Any:
        def hook(_mod: Any, inp: Any) -> None:
            if self.active:
                self.buf["exp_mlp"][:, self.step, i] = inp[0].float().abs().mean(-1).mean(-1)

        return hook


class ActivationRecorder:
    """Engine side: segment ids per call, hook start/stop around predict_action_chunk, per-episode buffers,
    one npz per episode at end."""

    def __init__(self, policy: Any, pre: Any, out_dir: str | Path, n_action_steps: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.cfg = policy.config
        self.tok = find_tokenizer(pre)
        self.hooks = Hooks(policy)
        self.n_action_steps = n_action_steps
        self.n_img = len(self.cfg.image_features)  # incl. empty (masked) cameras; present cameras first
        self.buffers: dict[str, list[dict[str, np.ndarray]]] = {}
        self.calls: dict[str, int] = {}  # policy calls per episode, recorded or not
        self.skipped: dict[str, list[dict[str, Any]]] = {}  # calls not recorded (call index, reason)
        self.saved = 0

    def info(self) -> dict[str, Any]:
        return {
            "dir": str(self.out_dir),
            "segments": SEGMENTS,
            "vec_layers": list(VEC_LAYERS),
            "vec_segments": [SEGMENTS[s] for s in VEC_SEGMENTS],
        }

    def begin(self, batch: dict[str, Any]) -> None:
        """batch: preprocessed policy batch (tokens present)."""
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
        n_rows, n_lang = tokens.shape
        n_img_keys = self.n_img * IMG_TOKENS
        seg = torch.full((n_rows, n_img_keys + n_lang + self.cfg.chunk_size), PAD, dtype=torch.long)
        for c in range(min(self.n_img, 2)):
            seg[:, c * IMG_TOKENS : (c + 1) * IMG_TOKENS] = c
        self._bad: dict[int, str] = {}
        for b in range(n_rows):
            ids = tokens[b][masks[b]].tolist()
            pos = torch.nonzero(masks[b].cpu()).flatten() + n_img_keys
            try:  # e.g. a prompt cut at tokenizer_max_length before ';\nAction': skip this row's record only
                seg[b, pos] = torch.tensor(lang_segments(self.tok, ids))
            except ValueError as e:
                self._bad[b] = f"prompt segments: {e}"[:300]
            seg[b, n_img_keys + n_lang :] = 6
        self._ntok = np.stack([np.bincount(seg[b].numpy(), minlength=PAD + 1)[:PAD] for b in range(n_rows)])
        self.hooks.start(
            seg.to(tokens.device), self.n_img, int(self.cfg.num_inference_steps), self.cfg.chunk_size
        )

    def abort(self) -> None:
        self.hooks.active = False

    def skip(self, keys: list[str | None], reason: str) -> None:
        """The call happened but is not recorded (recorder error): count it and flag the episodes."""
        self.abort()
        for key in keys:
            if key is not None:
                call = self.calls.get(key, 0)
                self.calls[key] = call + 1
                self.skipped.setdefault(key, []).append({"call": call, "reason": reason[:300]})
        print(
            f"[activations] WARNING call not recorded for {[k for k in keys if k]}: {reason[:300]}",
            file=sys.stderr,
            flush=True,
        )

    def collect(self, keys: list[str | None], actions: np.ndarray) -> None:
        """keys: episode buffer key per batch row (None = padding row); actions: (B, n, 7) as served."""
        out = self.hooks.stop()
        for b, key in enumerate(keys):
            if key is None:
                continue
            if b in self._bad:
                self.skip([key], self._bad[b])
                continue
            call = self.calls.get(key, 0)
            self.calls[key] = call + 1
            row = {
                k: v[b].copy() for k, v in out.items()
            }  # copy: a row view would pin the whole batch buffer
            row["seg_ntok"] = self._ntok[b].astype(np.int32)
            row["act_pred"] = np.array(actions[b], np.float32)
            row["call"] = np.asarray(call, np.int32)
            self.buffers.setdefault(key, []).append(row)

    def save(self, key: str, record_key: str, meta: dict[str, Any]) -> str | None:
        """Write the episode's calls to <dir>/<record_key>.npz; returns the path (None if nothing recorded)."""
        rows = self.buffers.pop(key, None)
        n_calls = self.calls.pop(key, 0)
        skipped = self.skipped.pop(key, [])
        if not rows:
            return None
        arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        for k in F16_KEYS:
            arrays[k] = arrays[k].astype(np.float16)
        arrays["t"] = arrays["call"] * self.n_action_steps  # env step of each recorded call (chunk clients)
        meta = {
            **meta,
            "record_key": record_key,
            "n_calls": n_calls,
            "n_recorded": len(rows),
            "skipped_calls": skipped,
            "n_action_steps": self.n_action_steps,
            "policy_type": getattr(self.cfg, "type", None),
            "num_inference_steps": int(self.cfg.num_inference_steps),
            **self.info(),
            "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        arrays["meta"] = np.asarray(json.dumps(meta, default=str))
        path = self.target(record_key)
        if path is None:
            return None
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, path)  # swaps the directory entry: a symlink at `path` is never followed
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.saved += 1
        return str(path)

    def target(self, record_key: str) -> Path | None:
        """<dir>/<record_key>.npz if its directory resolves inside the recording dir (no escape through a
        symlinked subdirectory); else None + warning (inference is not affected)."""
        root = self.out_dir.resolve()
        path = self.out_dir / f"{safe_key(record_key)}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent.resolve()
        if parent != root and root not in parent.parents:
            print(
                f"[activations] WARNING {path} resolves outside {root}: not written",
                file=sys.stderr,
                flush=True,
            )
            return None
        return parent / path.name

    def drop(self, key: str) -> None:
        self.buffers.pop(key, None)
        self.calls.pop(key, None)
        self.skipped.pop(key, None)


# ---------------------------------------------------------------------------------------------------------
# Rich mode (T028): residual / K / V *vectors* instead of norms, plus the token and denoising axes.
# Used by scripts/shortcut/analysis/rich_extract.py (replay extraction); the policy server keeps the light
# recorder above. The raw streams are returned here; PCA projection is the caller's job.
# ---------------------------------------------------------------------------------------------------------

RICH_TOK_LAYERS = (0, 2, 5, 8, 11, 14, 17)  # VLM: every language token's residual / K / V at these layers
RICH_EXP_TOK_LAYERS = (2, 5, 8, 11, 14, 17)  # expert: all action tokens at these layers
RICH_STEPS = (0, 4, 9)  # denoising steps kept for the expert vectors (0 = pure-noise input)
RICH_MLP_LAYERS = tuple(range(18))  # expert MLP middle (down_proj input) token means


class RichHooks(Hooks):
    """Hooks (all T021/T023 features) + vectors. Per batch of B calls (P prefix tokens):

    vlm_seg  (B, L, 6, D_vlm)    VLM decoder-layer output, mean over each prefix segment (all L layers)
    k_seg, v_seg (B, L, 6, D_kv) VLM self_attn k_proj / v_proj output (pre-RoPE), mean per prefix segment
    vlm_tok  (B, len(tok_layers), N, D_vlm)  every language token (N = max non-pad tokens in the batch)
    k_tok, v_tok (B, len(tok_layers), N, D_kv)
    exp_tok  (B, len(exp_tok_layers), len(steps), chunk, D_exp)  expert layer output, all action tokens
    exp_mean (B, len(steps), Lx, D_exp)                         expert layer output, action-token mean
    exp_mlp_mean (B, len(steps), len(mlp_layers), D_mlp)        expert down_proj input, action-token mean
    """

    def __init__(
        self,
        policy: Any,
        tok_layers: tuple[int, ...] = RICH_TOK_LAYERS,
        exp_tok_layers: tuple[int, ...] = RICH_EXP_TOK_LAYERS,
        steps: tuple[int, ...] = RICH_STEPS,
        mlp_layers: tuple[int, ...] = RICH_MLP_LAYERS,
    ):
        super().__init__(policy)
        self.tok_layers = tuple(tok_layers)
        self.exp_tok_layers = tuple(exp_tok_layers)
        self.steps = tuple(steps)
        self.mlp_layers = tuple(mlp_layers)
        for i, layer in enumerate(self.vlm_layers):
            self.handles.append(layer.register_forward_hook(self._r_vlm(i)))
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(self._r_kv(i, "k")))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(self._r_kv(i, "v")))
        for i, layer in enumerate(self.exp_layers):
            self.handles.append(layer.register_forward_hook(self._r_exp(i)))
            if i in self.mlp_layers:
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._r_mlp(i)))

    def start_rich(
        self,
        seg: torch.Tensor,
        n_img: int,
        n_steps: int,
        chunk: int,
        lang_pos: torch.Tensor,
        lang_valid: torch.Tensor,
    ) -> None:
        """lang_pos: (B, N) prefix index of each language token (0 where invalid); lang_valid: (B, N) bool."""
        missing = [s for s in self.steps if s >= n_steps]
        if missing:
            raise ValueError(f"rich steps {missing} >= {n_steps} denoising steps")
        self.start(seg, n_img, n_steps, chunk)
        self.lang_pos, self.lang_valid = lang_pos, lang_valid
        self.rich: dict[str, dict[int, torch.Tensor]] = {
            k: {} for k in ("vlm_seg", "k_seg", "v_seg", "vlm_tok", "k_tok", "v_tok")
        }
        self.rich_exp: dict[tuple[str, int, int], torch.Tensor] = {}

    def _gather_lang(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, P, D) → (B, N, D) at the language positions, zeros where invalid."""
        idx = self.lang_pos[..., None].expand(-1, -1, x.shape[-1])
        return torch.gather(x.float(), 1, idx) * self.lang_valid[..., None]

    def _r_vlm(self, i: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            if not self.active:
                return
            h = out[0] if isinstance(out, tuple) else out
            self.rich["vlm_seg"][i] = self._seg_mean(h)[:, : len(PREFIX_SEGMENTS)]
            if i in self.tok_layers:
                self.rich["vlm_tok"][i] = self._gather_lang(h)

        return hook

    def _r_kv(self, i: int, which: str) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            if not self.active:
                return
            self.rich[f"{which}_seg"][i] = self._seg_mean(out)[:, : len(PREFIX_SEGMENTS)]
            if i in self.tok_layers:
                self.rich[f"{which}_tok"][i] = self._gather_lang(out)

        return hook

    def _r_exp(self, i: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            if not self.active or self.step not in self.steps:
                return
            h = (out[0] if isinstance(out, tuple) else out)[:, -self.chunk :].float()
            self.rich_exp["mean", self.step, i] = h.mean(1)
            if i in self.exp_tok_layers:
                self.rich_exp["tok", self.step, i] = h

        return hook

    def _r_mlp(self, i: int) -> Any:
        def hook(_mod: Any, inp: Any) -> None:
            if self.active and self.step in self.steps:
                self.rich_exp["mlp", self.step, i] = inp[0][:, -self.chunk :].float().mean(1)

        return hook

    def stop_rich(self) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
        """→ (light T021/T023 features as numpy, rich features as float32 tensors on the model device)."""
        light = self.stop()
        n_vlm = len(self.vlm_layers)
        for k, by_layer in self.rich.items():
            want = sorted(self.tok_layers) if k.endswith("_tok") else list(range(n_vlm))
            if sorted(by_layer) != want:
                raise RuntimeError(f"rich {k}: hooks saw layers {sorted(by_layer)}, expected {want}")
        rich = {k: torch.stack([v[i] for i in sorted(v)], 1) for k, v in self.rich.items()}
        e = self.rich_exp
        rich["exp_tok"] = torch.stack(
            [torch.stack([e["tok", s, i] for s in self.steps], 1) for i in self.exp_tok_layers], 1
        )
        rich["exp_mean"] = torch.stack(
            [torch.stack([e["mean", s, i] for i in range(len(self.exp_layers))], 1) for s in self.steps], 1
        )
        rich["exp_mlp_mean"] = torch.stack(
            [torch.stack([e["mlp", s, i] for i in self.mlp_layers], 1) for s in self.steps], 1
        )
        return light, rich
