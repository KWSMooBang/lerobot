"""Residual-stream hooks on lerobot π0.5 (PI05Policy): record and intervene.

Where the residual stream is read (checked by `smoke.py`, see README "Hook location"):
  At inference `PI05Pytorch.sample_actions` runs the prefix once through
  `paligemma.model.language_model` (PiGemmaModel.forward → each `PiGemmaDecoderLayer.forward`) and every
  denoising step through `gemma_expert.model` (same class), so the decoder-layer modules are called and
  `register_forward_hook` on them sees the block output (attention + MLP + both gated residuals), before
  the final norm. The joint layer loop `compute_layer_complete` (both streams in one pass, no module call)
  is only used by the training forward, never by `sample_actions`.

Layer names:
  emb        input of PaliGemma layer 0 (prefix embeddings; baseline)
  pgNN       output of PaliGemma decoder layer NN (0-17, d=2048)
  aeNN       output of action-expert decoder layer NN (0-17, d=1024), once per denoising step

`ActivationRecorder` pools on the fly: PaliGemma layers → per token group (cam1, cam2, text) masked means
+ valid-token counts; action-expert layers → mean over the action tokens, per denoising step.
`Intervention` / `add`, `project_out`, `replace_with` change one layer's output on every token and every
call (prefix once — the KV cache carries it into every denoising step — and the action expert at every
step), as in Swann et al. 2026, App. F.2.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

N_PG = 18
N_AE = 18
GROUPS = ("cam1", "cam2", "text")  # PaliGemma token groups (pooled separately)
N_CAM_GROUPS = 2  # image slots beyond this (LIBERO: empty_camera_0) must be fully padded


# ---------------------------------------------------------------------------- layer names
def all_layers() -> list[str]:
    return ["emb"] + [f"pg{i:02d}" for i in range(N_PG)] + [f"ae{i:02d}" for i in range(N_AE)]


def parse_layers(spec: str | Sequence[str]) -> list[str]:
    """`all`, `emb`, `pg:0-17`, `ae:0,5,11`, `pg5`, `ae11`, comma/space separated → canonical names."""
    items = spec if not isinstance(spec, str) else re.split(r"[\s;]+|,(?=[a-z])", spec.strip())
    out: list[str] = []
    for it in (x.strip() for x in items):
        if not it:
            continue
        if it == "all":
            out += all_layers()
            continue
        if it == "emb":
            out.append("emb")
            continue
        m = re.fullmatch(r"(pg|ae):?([\d,\-]+)", it)
        if not m:
            raise ValueError(f"bad layer spec {it!r} (use all, emb, pg:0-17, ae:0,5,11, pg5)")
        fam, rng = m.groups()
        idx: list[int] = []
        for part in rng.split(","):
            if "-" in part:
                a, b = part.split("-")
                idx += list(range(int(a), int(b) + 1))
            elif part:
                idx.append(int(part))
        n = N_PG if fam == "pg" else N_AE
        for i in idx:
            if not 0 <= i < n:
                raise ValueError(f"{fam} layer {i} out of range 0-{n - 1}")
            out.append(f"{fam}{i:02d}")
    return list(dict.fromkeys(out))


def layer_family(name: str) -> str:
    return "ae" if name.startswith("ae") else "pg"  # emb pools like a PaliGemma layer


def layer_dim(name: str) -> int:
    return 1024 if name.startswith("ae") else 2048


def _pi05_model(policy: Any) -> Any:
    return policy.model if hasattr(policy, "model") else policy


def pg_layers(policy: Any) -> nn.ModuleList:
    return _pi05_model(policy).paligemma_with_expert.paligemma.model.language_model.layers


def ae_layers(policy: Any) -> nn.ModuleList:
    return _pi05_model(policy).paligemma_with_expert.gemma_expert.model.layers


def layer_module(policy: Any, name: str) -> nn.Module:
    if name == "emb":
        return pg_layers(policy)[0]
    fam, i = name[:2], int(name[2:])
    return (pg_layers(policy) if fam == "pg" else ae_layers(policy))[i]


def _hidden_from_args(args: tuple, kwargs: dict) -> Tensor:
    return args[0] if args else kwargs["hidden_states"]


# ---------------------------------------------------------------------------- recorder
@dataclass
class _PrefixLayout:
    group_masks: Tensor  # (B, G, L) float, 1 on valid tokens of the group
    counts: Tensor  # (B, G) valid tokens per group


class ActivationRecorder:
    """Pools residual-stream activations of the selected layers during `predict_action_chunk`.

    After one forward: `pg[name]` (B, G, d) fp32 group means, `counts` (B, G) int, `ae[name]` (B, S, d)
    fp32 action-token means for the recorded denoising steps `ae_steps` (None = all), in step order.
    Hooks return None (observe only), so the policy output is unchanged.
    """

    def __init__(self, policy: Any, layers: Iterable[str], ae_steps: Sequence[int] | None = None):
        self.policy = policy
        self.model = _pi05_model(policy)
        self.layers = list(layers)
        self.ae_steps = None if ae_steps is None else sorted(set(ae_steps))
        self.handles: list[Any] = []
        self._layout: _PrefixLayout | None = None
        self._img_lens: list[int] | None = None
        self._step = -1
        self.pg: dict[str, Tensor] = {}
        self.counts: Tensor | None = None
        self._ae: dict[str, list[Tensor]] = {}
        self._orig: dict[str, Any] = {}

    # -- prefix layout: wrap the instance's embed_prefix/_embed_images (no change to lerobot code)
    def _wrap_prefix(self) -> None:
        model = self.model
        orig_embed_images = model._embed_images
        orig_embed_prefix = model.embed_prefix

        def embed_images(images, img_masks):
            embs = orig_embed_images(images, img_masks)
            self._img_lens = [int(e.shape[1]) for e in embs]
            return embs

        def embed_prefix(images, img_masks, tokens, masks, states=None, state_masks=None):
            embs, pad_masks, att_masks = orig_embed_prefix(
                images, img_masks, tokens, masks, states, state_masks
            )
            self._layout = self._build_layout(pad_masks, int(tokens.shape[1]))
            self.counts = self._layout.counts
            self._step = -1
            self.pg, self._ae = {}, {}
            return embs, pad_masks, att_masks

        # instance attributes that were there before (another wrapper), restored on detach
        self._orig = {k: model.__dict__.get(k) for k in ("_embed_images", "embed_prefix")}
        model._embed_images = embed_images
        model.embed_prefix = embed_prefix

    def _build_layout(self, pad_masks: Tensor, n_lang: int) -> _PrefixLayout:
        if self._img_lens is None:
            raise RuntimeError("image token lengths were not recorded")
        bsz, total = pad_masks.shape
        lens = self._img_lens
        if sum(lens) + n_lang != total:
            raise RuntimeError(f"prefix layout: images {lens} + text {n_lang} != {total} tokens")
        gm = torch.zeros(bsz, len(GROUPS), total, device=pad_masks.device)
        pos = 0
        for k, n in enumerate(lens):
            seg = pad_masks[:, pos : pos + n]
            if k < N_CAM_GROUPS:
                gm[:, k, pos : pos + n] = seg.float()
            elif bool(seg.any()):
                raise RuntimeError(f"image slot {k} has valid tokens; only {N_CAM_GROUPS} camera groups")
            pos += n
        gm[:, GROUPS.index("text"), pos:] = pad_masks[:, pos:].float()
        return _PrefixLayout(group_masks=gm, counts=gm.sum(-1))

    def _pool_prefix(self, name: str, h: Tensor) -> None:
        lay = self._layout
        if lay is None:
            raise RuntimeError("PaliGemma hook fired before embed_prefix")
        sums = torch.einsum("bgl,bld->bgd", lay.group_masks, h.float())
        self.pg[name] = sums / lay.counts.clamp_min(1).unsqueeze(-1)

    def attach(self) -> ActivationRecorder:
        self._wrap_prefix()
        for name in self.layers:
            mod = layer_module(self.policy, name)
            if name == "emb":

                def pre(_m, args, kwargs, name=name):
                    self._pool_prefix(name, _hidden_from_args(args, kwargs))

                self.handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
            elif name.startswith("pg"):

                def post(_m, _a, out, name=name):
                    self._pool_prefix(name, out)

                self.handles.append(mod.register_forward_hook(post))
            else:

                def post_ae(_m, _a, out, name=name):
                    if self.ae_steps is None or self._step in self.ae_steps:
                        self._ae.setdefault(name, []).append(out.float().mean(dim=1))

                self.handles.append(mod.register_forward_hook(post_ae))

        def count_step(_m, _a, _k):
            self._step += 1

        # one denoising step = one pass through the expert's layers. lerobot calls
        # `gemma_expert.model.forward(...)` directly (module hooks on the model would not fire), so count on layer 0
        self.handles.append(ae_layers(self.policy)[0].register_forward_pre_hook(count_step, with_kwargs=True))
        return self

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        for k, prev in self._orig.items():
            if prev is None:
                self.model.__dict__.pop(k, None)  # back to the class method
            else:
                self.model.__dict__[k] = prev
        self._orig = {}

    def __enter__(self) -> ActivationRecorder:
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()

    @property
    def n_steps(self) -> int:
        return self._step + 1

    @property
    def ae(self) -> dict[str, Tensor]:
        return {k: torch.stack(v, dim=1) for k, v in self._ae.items()}


# ---------------------------------------------------------------------------- interventions
@dataclass
class Intervention:
    """One edit of one layer's residual stream, on every token and every call.

    kind: "add" (y + alpha·v), "project_out" (y − (y·v̂)v̂), "replace" (SAE reconstruction of y).
    For `replace`, mode "token" reconstructs every token vector (as asked for the API); mode "mean"
    replaces only the token-mean (the vector the pooled SAEs were trained on) and keeps each token's
    deviation from it. `stats` collects per-call reconstruction metrics for `replace`.
    """

    layer: str
    kind: str
    vec: Tensor | None = None
    alpha: float = 1.0
    sae: Any = None
    mode: str = "token"
    stats: list[dict[str, float]] = field(default_factory=list)
    calls: int = 0

    def apply(self, y: Tensor) -> Tensor:
        self.calls += 1
        yf = y.float()
        if self.kind == "add":
            assert self.vec is not None
            out = yf + self.alpha * self.vec.to(yf.device, torch.float32)
        elif self.kind == "project_out":
            assert self.vec is not None
            v = self.vec.to(yf.device, torch.float32)
            v = v / v.norm()
            out = yf - (yf @ v).unsqueeze(-1) * v
        elif self.kind == "replace":
            sae = self.sae
            if self.mode == "token":
                x = yf
            elif self.mode == "mean":
                x = yf.mean(dim=-2, keepdim=True)
            else:
                raise ValueError(f"replace mode {self.mode!r}")
            with torch.no_grad():
                x_hat, metrics = sae.reconstruct(x, return_metrics=True)
            self.stats.append(metrics)
            out = x_hat if self.mode == "token" else yf - x + x_hat
        else:
            raise ValueError(f"unknown intervention {self.kind!r}")
        return out.to(y.dtype)


class Steering:
    """Context manager holding a set of interventions on a policy (or any module tree with the π0.5 layout)."""

    def __init__(self, policy: Any, interventions: Sequence[Intervention]):
        self.policy = policy
        self.interventions = list(interventions)
        self.handles: list[Any] = []

    def attach(self) -> Steering:
        for iv in self.interventions:
            mod = layer_module(self.policy, iv.layer)
            if iv.layer == "emb":

                def pre(_m, args, kwargs, iv=iv):
                    if args:
                        return (iv.apply(args[0]), *args[1:]), kwargs
                    return args, {**kwargs, "hidden_states": iv.apply(kwargs["hidden_states"])}

                self.handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
            else:

                def post(_m, _a, out, iv=iv):
                    return iv.apply(out)

                self.handles.append(mod.register_forward_hook(post))
        return self

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self) -> Steering:
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()


def add(policy: Any, layer: str, v: Tensor, alpha: float) -> Steering:
    """y' = y + alpha·v on `layer` (paper eq. 1). Use as `with add(policy, "pg05", v, 50): ...`."""
    return Steering(policy, [Intervention(layer, "add", vec=v, alpha=alpha)])


def project_out(policy: Any, layer: str, v: Tensor) -> Steering:
    """y' = y − (y·v̂)v̂ on `layer` (paper eq. 2, v normalized)."""
    return Steering(policy, [Intervention(layer, "project_out", vec=v)])


def replace_with(policy: Any, layer: str, sae: Any, mode: str = "token") -> Steering:
    """y' = SAE(y) on `layer` (mode "token": every token; "mean": the token-mean only)."""
    return Steering(policy, [Intervention(layer, "replace", sae=sae, mode=mode)])
