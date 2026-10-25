"""Policy + processors for batched inference, with per-episode flow-matching noise.

The single-env path seeds the global torch RNG with the episode seed and pi05 draws each chunk's noise
with `sample_noise` (torch.normal, float32, on the policy device). Here every episode owns a
torch.Generator seeded with the same episode seed and draws the same shape in the same order, then the
rows are concatenated and passed as `noise=` (pi05 `predict_action_chunk` takes it). So an episode's noise
does not depend on which other episodes share the batch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [p for p in (str(HERE), str(HERE.parent)) if p not in sys.path]

from common import load_policy  # noqa: E402

from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


def stack_obs(obs_list: list[Any]) -> Any:
    """[obs] → one obs dict with a leading batch axis (np.stack; equals _batchify for one obs)."""
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: stack_obs([o[k] for o in obs_list]) for k in first}
    return np.stack([np.asarray(o) for o in obs_list])


class PolicyBundle:
    def __init__(
        self,
        policy: Any,
        pre: Any,
        post: Any,
        env_pre: Any,
        env_post: Any,
        inf_meta: dict[str, Any],
        device: str,
        n_action_steps: int,
    ):
        cfg = policy.config
        if getattr(cfg, "use_visual_memory", None) or getattr(cfg, "use_proprioceptive_memory", None):
            raise ValueError("memory policies keep per-episode state; the batched path does not support them")
        if any(
            type(s).__name__ == "RelativeActionsProcessorStep" and s.enabled
            for s in getattr(pre, "steps", ())
        ):
            raise ValueError("relative-action preprocessors hold a per-episode anchor; not supported batched")
        self.policy, self.pre, self.post, self.env_pre, self.env_post = policy, pre, post, env_pre, env_post
        self.inf_meta = inf_meta
        self.device = device
        self.n_action_steps = n_action_steps
        self.noise_shape = (1, int(cfg.chunk_size), int(cfg.max_action_dim))
        policy.reset()

    def generator(self, ep_seed: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(ep_seed)

    def draw_noise(self, gen: torch.Generator) -> torch.Tensor:
        """Next chunk's noise of one episode; same call as flow_matching.sample_noise but with its generator."""
        return torch.normal(
            mean=0.0, std=1.0, size=self.noise_shape, generator=gen, dtype=torch.float32, device=self.device
        )

    def infer(self, obs_list: list[dict], instructions: list[str], noise: torch.Tensor) -> np.ndarray:
        """One chunk for every row: (B, n_action_steps, action_dim) float32 on CPU, post-processed per step
        exactly like select_action's queue (post / env_post on each (B, action_dim) slice)."""
        batch = preprocess_observation(stack_obs(obs_list))
        batch["task"] = list(instructions)
        batch = self.pre(self.env_pre(batch))
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch, noise=noise)[:, : self.n_action_steps]
        steps = [self.env_post({ACTION: self.post(chunk[:, k])})[ACTION] for k in range(chunk.shape[1])]
        return torch.stack([a.to("cpu") for a in steps], dim=1).numpy()


def load_bundle(group: dict[str, Any], device: str, compile_model: bool) -> PolicyBundle:
    """group: {backend, policy, ckpt, adapter, suite, n_action_steps} (see run_multi.group_of)."""
    n = int(group["n_action_steps"])
    if group["backend"] == "mock":
        from mock_backend import load_mock_policy

        policy, pre, post, env_pre, env_post, _, meta = load_mock_policy(group["ckpt"], n)
        return PolicyBundle(policy, pre, post, env_pre, env_post, meta, "cpu", n)
    policy, pre, post, env_pre, env_post, _, meta = load_policy(
        group["policy"],
        group["ckpt"],
        group["suite"],
        n,
        device,
        libero_adapter=group["adapter"],
        compile_model=compile_model,
    )
    meta["policy_built_with_suite"] = group["suite"]  # LIBERO features/processors are the same per suite
    return PolicyBundle(policy, pre, post, env_pre, env_post, meta, device, n)
