"""Config of sc_template. Every field is a CLI flag: `--policy.<field>=...` (sc-train passes extra args through)."""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("sc_template")
@dataclass
class ScTemplateConfig(PreTrainedConfig):
    """Small image encoder + instruction embedding + action-chunk head.

    Args:
        chunk_size: actions predicted per forward pass (dataset returns this many future actions).
        n_action_steps: actions executed from each chunk before predicting again.
        image_size: camera images are resized to this square size before the encoder.
        vision_channels: channels of the 4 conv blocks (stride 2 each).
        lang_cond: how the instruction conditions the policy.
            "film": FiLM (per-channel scale/shift) on every conv block + concat at the head;
            "concat": instruction embedding only concatenated at the head; "none": ignored (vision-only).
        lang_vocab_size: hashed word/bigram buckets of the instruction embedding (no tokenizer download).
    """

    n_obs_steps: int = 1
    chunk_size: int = 20
    n_action_steps: int = 10

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    image_size: int = 128
    vision_channels: tuple[int, ...] = (32, 64, 128, 128)
    vision_out_dim: int = 256
    lang_cond: str = "film"
    lang_vocab_size: int = 8192
    lang_dim: int = 128
    state_dim_embed: int = 64
    hidden_dim: int = 512
    n_head_layers: int = 3
    dropout: float = 0.1

    optimizer_lr: float = 3e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 500
    # CosineDecayWithWarmup rescales warmup/decay to --steps when steps < decay steps.
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 3e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(f"n_action_steps ({self.n_action_steps}) > chunk_size ({self.chunk_size})")
        if self.lang_cond not in ("film", "concat", "none"):
            raise ValueError(f"lang_cond must be film|concat|none, got {self.lang_cond!r}")
        if len(self.vision_channels) != 4:
            raise ValueError("vision_channels needs 4 entries (one per conv block)")

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("sc_template needs at least one image feature")
        if self.action_feature is None:
            raise ValueError("sc_template needs 'action' in output_features")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
