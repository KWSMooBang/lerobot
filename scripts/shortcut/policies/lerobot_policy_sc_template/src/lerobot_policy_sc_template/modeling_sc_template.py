"""sc_template policy: per-camera conv encoder (optionally FiLM-conditioned on the instruction) + MLP
action-chunk head, trained with masked L1 on normalized actions. ~4 M parameters, trains on one GPU.

Places to put an idea (see README.md):
  - `InstructionEncoder` / `hash_tokens`: how the instruction is read.
  - `ConvEncoder` + `lang_cond`: where language enters the visual pathway.
  - `ScTemplatePolicy.aux_losses`: extra training losses (returned in the loss dict, logged by lerobot-train).
"""

import re
import zlib
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_sc_template import ScTemplateConfig

_WORD = re.compile(r"[a-z0-9]+")


def hash_tokens(text: str, vocab_size: int) -> list[int]:
    """Words and word bigrams of a lowercased instruction, hashed into [1, vocab_size). Empty text -> []."""
    words = _WORD.findall(text.lower())
    grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
    return [1 + zlib.crc32(g.encode()) % (vocab_size - 1) for g in grams]


class InstructionEncoder(nn.Module):
    """Mean of hashed word/bigram embeddings. An empty instruction gives the zero vector."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.bag = nn.EmbeddingBag(vocab_size, dim, mode="mean", padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, texts: list[str], device: torch.device) -> Tensor:
        ids: list[int] = []
        offsets: list[int] = []
        for t in texts:
            offsets.append(len(ids))
            ids.extend(hash_tokens(t, self.vocab_size))
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        off_t = torch.tensor(offsets, dtype=torch.long, device=device)
        return self.proj(self.bag(ids_t, off_t))


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, cond_dim: int | None):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, c_out)
        self.film = nn.Linear(cond_dim, 2 * c_out) if cond_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, cond: Tensor | None) -> Tensor:
        x = self.norm(self.conv(x))
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond).chunk(2, dim=-1)
            x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return F.gelu(x)


class ConvEncoder(nn.Module):
    """4 stride-2 conv blocks -> 1x1 conv -> flatten (keeps the spatial layout) -> linear."""

    def __init__(self, config: ScTemplateConfig, cond_dim: int | None):
        super().__init__()
        chans = (3, *config.vision_channels)
        self.blocks = nn.ModuleList(ConvBlock(a, b, cond_dim) for a, b in zip(chans, chans[1:], strict=False))
        self.squeeze = nn.Conv2d(chans[-1], 32, 1)
        side = config.image_size // 16
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(32 * side * side, config.vision_out_dim), nn.GELU())
        self.image_size = config.image_size

    def forward(self, img: Tensor, cond: Tensor | None) -> Tensor:
        x = F.interpolate(img, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        for block in self.blocks:
            x = block(x, cond)
        return self.out(self.squeeze(x))


class ScTemplatePolicy(PreTrainedPolicy):
    config_class = ScTemplateConfig
    name = "sc_template"

    def __init__(self, config: ScTemplateConfig, dataset_stats: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.image_keys = list(config.image_features)
        assert config.action_feature is not None  # validate_features checked it
        action_dim = config.action_feature.shape[0]
        self.action_dim = action_dim
        use_lang = config.lang_cond != "none"

        self.lang = InstructionEncoder(config.lang_vocab_size, config.lang_dim) if use_lang else None
        film_dim = config.lang_dim if config.lang_cond == "film" else None
        # one encoder per camera (agentview and wrist look very different)
        self.encoders = nn.ModuleList(ConvEncoder(config, film_dim) for _ in self.image_keys)
        state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else 0
        self.state_proj = nn.Linear(state_dim, config.state_dim_embed) if state_dim else None

        d_in = config.vision_out_dim * len(self.image_keys)
        d_in += config.state_dim_embed if state_dim else 0
        d_in += config.lang_dim if use_lang else 0
        layers: list[nn.Module] = []
        for i in range(config.n_head_layers):
            layers += [
                nn.Linear(d_in if i == 0 else config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ]
        layers.append(nn.Linear(config.hidden_dim, config.chunk_size * action_dim))
        self.head = nn.Sequential(*layers)
        self.reset()

    # ------------------------------------------------------------------ lerobot API
    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @staticmethod
    def _texts(batch: dict[str, Any], n: int) -> list[str]:
        task = batch.get("task")
        if task is None:
            return [""] * n
        if isinstance(task, str):
            return [task] * n
        return [str(t) for t in task]

    def _features(self, batch: dict[str, Any]) -> Tensor:
        ref = batch[self.image_keys[0]]
        b, device = ref.shape[0], ref.device
        lang = self.lang(self._texts(batch, b), device) if self.lang is not None else None
        film = lang if self.config.lang_cond == "film" else None
        feats = [enc(batch[k], film) for enc, k in zip(self.encoders, self.image_keys, strict=True)]
        if self.state_proj is not None:
            feats.append(self.state_proj(batch[OBS_STATE]))
        if lang is not None:
            feats.append(lang)
        return torch.cat(feats, dim=-1)

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        out = self.head(self._features(batch))
        return out.view(-1, self.config.chunk_size, self.action_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Any], reduction: str = "mean") -> tuple[Tensor, dict]:
        pred = self.predict_action_chunk(batch)
        target = batch[ACTION]
        err = F.l1_loss(pred, target, reduction="none")  # (B, chunk, action_dim)
        if "action_is_pad" in batch:
            err = err * (~batch["action_is_pad"]).unsqueeze(-1)
        per_sample = err.mean(dim=(1, 2))
        info = {"l1_loss": per_sample.mean().item()}
        aux = self.aux_losses(batch, pred)
        for k, v in aux.items():
            info[k] = v.item()
            per_sample = per_sample + v
        return (per_sample if reduction == "none" else per_sample.mean()), info

    # ------------------------------------------------------------------ idea slot
    def aux_losses(self, batch: dict[str, Any], pred: Tensor) -> dict[str, Tensor]:
        """Extra weighted losses, e.g. an instruction-contrastive term or a counterfactual-instruction
        penalty. Each value is added to the training loss and logged under its key. None by default."""
        return {}
