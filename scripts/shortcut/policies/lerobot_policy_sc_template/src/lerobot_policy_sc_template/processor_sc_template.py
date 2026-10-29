"""Pre/post processors of sc_template: lerobot's default Rename -> Batch -> Device -> Normalize pipeline.

The instruction string stays in the batch as `task` (complementary data) and is embedded inside the
model (`modeling_sc_template.hash_tokens`), so loading a checkpoint needs no extra processor step.
Put a tokenizer step here instead if the language encoder needs one (e.g. a pretrained text model).
"""

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_sc_template import ScTemplateConfig


def make_sc_template_pre_post_processors(
    config: ScTemplateConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
