"""Inference engine of the policy server: policy loading, per-episode noise, action queues, batching.

exec modes (recorded by clients in run identities; T004 rule: never pair across them):
  single     one row per policy call. Before each call the episode's own global-RNG state (torch.manual_seed
             (episode_seed) at reset) is restored, so a chunk equals what run_eval.py's in-process loop draws
             for the same episode, bit for bit, whatever other episodes the server interleaves.
  batched-N  requests of several episodes (clients) are stacked into one call padded to N rows (row 0
             copies), noise from a per-episode torch.Generator (parallel/bundle.py, same as T004 batched-N).
  step       (stateful policies: observation history, temporal ensembling, memory, relative-action anchor)
             policy.select_action per env step, one live episode at a time; clients must send `step: true`.
"""

from __future__ import annotations

import inspect
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

SHORTCUT = Path(__file__).resolve().parents[1]
sys.path[:0] = [p for p in (str(SHORTCUT), str(SHORTCUT / "parallel")) if p not in sys.path]

from bench.core.protocol import (  # noqa: E402
    ACTION_DIM,
    IMAGE_SLOTS,
    ORIENT_UPRIGHT,
    PROTOCOL_VERSION,
    STATE_DIM,
    ProtocolError,
    validate_obs,
)

from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

LIBERO_IMAGE_MAP = {"image": "observation.images.image", "image2": "observation.images.image2"}
PI_TOKENIZER_TYPES = ("pi0", "pi05", "pi0_fast")  # gated paligemma tokenizer → common.load_policy fallback


@dataclass
class Episode:
    seed: int
    instruction: str
    queue: deque = field(default_factory=deque)
    rng: tuple[torch.Tensor, torch.Tensor | None] | None = None  # single: (cpu, cuda) global RNG state
    gen: torch.Generator | None = None  # batched: per-episode generator
    calls: int = 0


@dataclass
class Loaded:
    policy: Any
    pre: Any
    post: Any
    env_post: Any
    inf_meta: dict[str, Any]
    policy_type: str


def stateful_reason(policy: Any, pre: Any) -> str | None:
    cfg = policy.config
    if int(getattr(cfg, "n_obs_steps", 1) or 1) > 1:
        return f"n_obs_steps={cfg.n_obs_steps}"
    if getattr(cfg, "temporal_ensemble_coeff", None) is not None:
        return "temporal ensembling"
    if getattr(cfg, "use_visual_memory", None) or getattr(cfg, "use_proprioceptive_memory", None):
        return "memory"
    if any(
        type(s).__name__ == "RelativeActionsProcessorStep" and getattr(s, "enabled", False)
        for s in getattr(pre, "steps", ())
    ):
        return "relative-action anchor"
    return None


def load_any(ckpt: str, adapter: str | None, suite: str, n_action_steps: int, device: str) -> Loaded:
    """Any lerobot checkpoint (hub id or local pretrained_model dir; out-of-tree plugins via
    register_third_party_plugins). pi0-family goes through common.load_policy (LIBERO adapter, local
    PaliGemma tokenizer); others: make_policy + their saved pre/post processors."""
    from common import load_policy

    import lerobot.policies  # noqa: F401  (registers config types)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    ptype = PreTrainedConfig.from_pretrained(adapter or ckpt).type
    if ptype in PI_TOKENIZER_TYPES or adapter:
        policy, pre, post, _env_pre, env_post, _tok, meta = load_policy(
            ptype, ckpt, suite, n_action_steps, device, libero_adapter=adapter
        )
        return Loaded(policy, pre, post, env_post, meta, ptype)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = Path(ckpt)
    cfg.device = device
    if hasattr(cfg, "n_action_steps"):
        cfg.n_action_steps = n_action_steps
    env_cfg = LiberoEnvConfig(task=suite)
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, pretrained_path=ckpt)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=ckpt, preprocessor_overrides={"device_processor": {"device": device}}
    )
    _env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    meta = {
        "policy_type": cfg.type,
        "chunk_size": getattr(cfg, "chunk_size", None),
        "n_action_steps_stored": getattr(cfg, "n_action_steps", None),
        "n_action_steps": n_action_steps,
        "dtype": getattr(cfg, "dtype", None),
        "temporal_ensembling": getattr(cfg, "temporal_ensemble_coeff", None) is not None,
        "env_observation_hw": [env_cfg.observation_height, env_cfg.observation_width],
        "control_mode": env_cfg.control_mode,
        "libero_adapter": None,
    }
    return Loaded(policy, pre, post, env_post, meta, cfg.type)


def load_mock(ckpt: str, n_action_steps: int) -> Loaded:
    """CPU fake (parallel/mock_backend.py) that reads the wire state instead of robot_state."""
    from mock_backend import load_mock_policy

    policy, pre, post, _ep, env_post, _tok, meta = load_mock_policy(ckpt, n_action_steps)
    inner = policy.predict_action_chunk

    def predict(batch: dict[str, Any], *, noise: torch.Tensor | None = None) -> torch.Tensor:
        st = batch[OBS_STATE].to(torch.float64)
        return inner({**batch, "observation.robot_state": {"eef": {"pos": st[:, :3]}}}, noise=noise)

    policy.predict_action_chunk = predict
    return Loaded(policy, pre, post, env_post, meta, "mock")


class Engine:
    def __init__(
        self,
        loaded: Loaded,
        device: str,
        n_action_steps: int,
        batch: int,
        info: dict[str, Any],
        image_map: dict[str, str] | None = None,
    ):
        self.l = loaded
        self.device = device
        self.n = n_action_steps
        self.batch = max(1, batch)
        self.image_map = image_map or dict(LIBERO_IMAGE_MAP)
        self.episodes: dict[str, Episode] = {}
        policy, pre = loaded.policy, loaded.pre
        cfg = policy.config
        self.stateful = stateful_reason(policy, pre)
        self.takes_noise = "noise" in inspect.signature(policy.predict_action_chunk).parameters
        self.cuda = device.startswith("cuda") and torch.cuda.is_available()
        if self.stateful:
            if self.batch > 1:
                raise SystemExit(f"--batch {self.batch}: policy is stateful ({self.stateful}); use --batch 1")
            self.exec_mode = "step"
        elif self.batch > 1:
            if not self.takes_noise:
                raise SystemExit("--batch > 1 needs predict_action_chunk(noise=...) for per-episode noise")
            self.exec_mode = f"batched-{self.batch}"
        else:
            self.exec_mode = "single"
        if self.batch > 1:
            from bundle import PolicyBundle

            self.bundle = PolicyBundle(
                policy, pre, loaded.post, None, loaded.env_post, loaded.inf_meta, device, n_action_steps
            )
        feats = list((getattr(cfg, "image_features", None) or {}).keys())
        missing = [k for k in self.image_map.values() if feats and k not in feats]
        if missing:
            raise SystemExit(f"policy image features {feats} lack {missing}; pass --image-map")
        self.hello = {
            "protocol_version": PROTOCOL_VERSION,
            "policy_type": loaded.policy_type,
            "exec_mode": self.exec_mode,
            "requires_step": bool(self.stateful),
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "chunk_size": int(getattr(cfg, "chunk_size", 0) or 0) or None,
            "n_action_steps": self.n,
            "image_keys": {slot: self.image_map[slot] for slot in IMAGE_SLOTS},
            "policy_image_features": feats,
            "image_orientation": ORIENT_UPRIGHT,
            "image_hw": list(loaded.inf_meta.get("env_observation_hw") or [256, 256]),
            "control_mode": loaded.inf_meta.get("control_mode", "relative"),
            "inference": loaded.inf_meta,
            **info,
        }

    # ------------------------------------------------------------ episodes

    def reset(self, eid: str, seed: int, instruction: str) -> None:
        ep = Episode(int(seed), instruction)
        if self.exec_mode == "step":
            from lerobot.processor.relative_action_processor import bind_relative_anchor

            self.episodes.clear()  # one live episode
            self.l.policy.reset()
            self.l.pre.reset()
            self.l.post.reset()
            bind_relative_anchor(self.l.policy, self.l.pre)
            torch.manual_seed(ep.seed)
        elif self.exec_mode == "single":
            torch.manual_seed(ep.seed)
            ep.rng = (torch.get_rng_state(), torch.cuda.get_rng_state() if self.cuda else None)
        else:
            ep.gen = self.bundle.generator(ep.seed)
        self.episodes[eid] = ep

    def end(self, eid: str) -> None:
        self.episodes.pop(eid, None)

    def episode(self, eid: str) -> Episode:
        ep = self.episodes.get(eid)
        if ep is None:
            raise ProtocolError(f"unknown episode {eid!r} (reset first)")
        return ep

    # ------------------------------------------------------------ inference

    def wire_batch(self, obs: list[dict[str, np.ndarray]], instructions: list[str]) -> dict[str, Any]:
        """Wire obs → the batch run_eval.py hands to the policy preprocessor (after preprocess_observation
        + LiberoProcessorStep): float32 [0, 1] CHW images, float32 state, task strings."""
        for o in obs:
            validate_obs(o)
        batch: dict[str, Any] = {}
        for slot, key in self.image_map.items():
            img = torch.from_numpy(np.stack([o[slot] for o in obs]))
            img = img.permute(0, 3, 1, 2).contiguous().type(torch.float32)
            img /= 255
            batch[key] = img
        batch[OBS_STATE] = torch.from_numpy(np.stack([o["state"] for o in obs]))
        batch["task"] = list(instructions)
        return batch

    def _post(self, chunk: torch.Tensor) -> np.ndarray:
        """(B, n, D) policy output → (B, n, 7) env actions, post-processed per step like select_action."""
        steps = [self.l.env_post({ACTION: self.l.post(chunk[:, k])})[ACTION] for k in range(chunk.shape[1])]
        return torch.stack([a.to("cpu") for a in steps], dim=1).numpy().astype(np.float32)

    def need_policy(self, eid: str, step: bool) -> bool:
        """False when a step request can be served from the episode's queue."""
        return not (step and self.episodes.get(eid) is not None and self.episode(eid).queue)

    def infer_one(self, eid: str, obs: dict[str, np.ndarray], instruction: str, step: bool) -> np.ndarray:
        """single / step modes."""
        ep = self.episode(eid)
        if self.exec_mode == "step":
            if not step:
                raise ProtocolError(f"policy is stateful ({self.stateful}): send infer with step=true")
            batch = self.l.pre(self.wire_batch([obs], [instruction]))
            with torch.inference_mode():
                a = self.l.policy.select_action(batch)
            out = self._post(a[:, None])[0]
            ep.calls += 1
            return out
        if step and ep.queue:
            return np.asarray([ep.queue.popleft()])
        assert ep.rng is not None
        torch.set_rng_state(ep.rng[0])
        if ep.rng[1] is not None:
            torch.cuda.set_rng_state(ep.rng[1])
        batch = self.l.pre(self.wire_batch([obs], [instruction]))
        with torch.inference_mode():
            chunk = self.l.policy.predict_action_chunk(batch)[:, : self.n]
        ep.rng = (torch.get_rng_state(), torch.cuda.get_rng_state() if self.cuda else None)
        ep.calls += 1
        return self._serve(ep, self._post(chunk)[0], step)

    def infer_batch(self, rows: list[tuple[str, dict[str, np.ndarray], str, bool]]) -> list[np.ndarray]:
        """batched mode: one padded call for rows from different episodes."""
        eps = [self.episode(eid) for eid, *_ in rows]
        obs = [o for _, o, _, _ in rows]
        instr = [i for _, _, i, _ in rows]
        noise = torch.cat([self.bundle.draw_noise(ep.gen) for ep in eps])
        k = self.batch - len(rows)
        if k > 0:
            obs += [obs[0]] * k
            instr += [instr[0]] * k
            noise = torch.cat([noise, noise[:1].expand(k, *noise.shape[1:])])
        batch = self.l.pre(self.wire_batch(obs, instr))
        with torch.inference_mode():
            chunk = self.l.policy.predict_action_chunk(batch, noise=noise)[:, : self.n]
        acts = self._post(chunk)
        out = []
        for i, (ep, (_, _, _, step)) in enumerate(zip(eps, rows, strict=True)):
            ep.calls += 1
            out.append(self._serve(ep, acts[i], step))
        return out

    def serve_from_queue(self, eid: str) -> np.ndarray:
        ep = self.episode(eid)
        return np.asarray([ep.queue.popleft()])

    @staticmethod
    def _serve(ep: Episode, chunk: np.ndarray, step: bool) -> np.ndarray:
        if not step:
            ep.queue.clear()
            return chunk
        ep.queue = deque(chunk[1:])
        return chunk[:1]


def timed(fn: Any, *a: Any) -> tuple[Any, float]:
    tic = time.perf_counter()
    out = fn(*a)
    return out, 1000 * (time.perf_counter() - tic)
