"""CPU-only stand-ins for LIBERO and π0.5 (`--backend mock`), for the queue / batching / merge tests.

Deterministic like the real stack: the env draws its initial offset from the global numpy RNG (seeded per
episode by EpisodeStepper), the policy draws flow-matching noise with `sample_noise` (global torch RNG)
unless noise is passed in, and every op is row-wise, so batched and single runs give identical bits.
"""

from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from lerobot.policies.common.flow_matching import sample_noise

MOCK_MAX_STEPS = 40
MOCK_HW = (32, 32)
MOCK_CHUNK = 16
MOCK_ACTION_DIM_PADDED = 8
_KNOWN = {1: "put the bowl on the stove", 8: "put the bowl on the plate"}  # tests/fixtures tasks


def text_pos(s: str) -> np.ndarray:
    """Deterministic 3-d point for a string (instruction → where the mock policy heads)."""
    h = hashlib.sha256(s.encode()).digest()
    return (np.frombuffer(h[:3], np.uint8).astype(np.float64) / 255.0 - 0.5) * 0.4


def mock_suite(suite: str, n_tasks: int = 10) -> SimpleNamespace:
    tasks = [
        {"task_id": i, "original": _KNOWN.get(i, f"mock {suite} task {i}"), "bddl": f"mock/{suite}/task{i}"}
        for i in range(n_tasks)
    ]
    return SimpleNamespace(name=suite, task_info=tasks)


class MockEnv:
    """Same surface as lerobot LiberoEnv for EpisodeStepper: init_state_id, _init_states, reset, step."""

    def __init__(self, task: dict[str, Any], max_steps: int, hw: list[int] | tuple[int, int] = MOCK_HW):
        self.task = task
        self.goal = text_pos(task["original"])
        self.hw = (int(hw[0]), int(hw[1]))
        self._init_states = list(range(50))
        self.init_state_id = 0
        self.max_steps = max_steps
        self.pos = np.zeros(3)
        self.grip = 0.0

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        offset = np.random.uniform(-0.25, 0.25, 3)  # global RNG, seeded per episode like LIBERO
        self.pos = self.goal + offset + 0.001 * self.init_state_id
        self.grip = -1.0
        return self._obs(), {"is_success": False}

    def step(self, a: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        a = np.asarray(a, dtype=np.float64)
        assert a.ndim == 1, a.shape
        self.pos = self.pos + 0.02 * np.clip(a[:3], -1, 1)
        self.grip = float(a[6])
        ok = bool(np.linalg.norm(self.pos - self.goal) < 0.03)
        return self._obs(), 0.0, ok, False, {"is_success": ok}

    def _obs(self) -> dict[str, Any]:
        h, w = self.hw
        img: np.ndarray = np.empty((h, w, 3), np.uint8)
        img[:] = np.clip((self.pos + 0.5) * 255, 0, 255).astype(np.uint8)
        width = 0.04 if self.grip <= 0 else 0.01
        return {
            "pixels": {"image": img, "image2": img[::-1].copy()},
            "robot_state": {
                "eef": {"pos": self.pos.copy(), "quat": np.array([1.0, 0.0, 0.0, 0.0])},
                "gripper": {"qpos": np.array([width, -width])},
            },
        }

    def close(self) -> None:
        pass


class MockProbe:
    def __init__(self, env: MockEnv):
        self.env = env

    def __call__(self) -> tuple[list[str], list[str]]:
        e = self.env
        if np.linalg.norm(e.pos - e.goal) < 0.06:
            return (["bowl"], ["bowl"]) if e.grip > 0 else (["bowl"], [])
        return [], []


class MockGoal:
    def __init__(self, bddl_path: str):
        self.path = bddl_path
        self.target = text_pos(Path(bddl_path).name)

    def missing_objects(self, env: MockEnv) -> list[str]:
        return []

    def __call__(self, env: MockEnv) -> bool:
        return bool(np.linalg.norm(env.pos - self.target) < 0.05)


class MockPolicy:
    """Heads to text_pos(instruction) + noise. Same select_action / predict_action_chunk contract as pi05."""

    def __init__(self, ckpt: str, n_action_steps: int):
        self.config = SimpleNamespace(
            n_action_steps=n_action_steps,
            chunk_size=MOCK_CHUNK,
            max_action_dim=MOCK_ACTION_DIM_PADDED,
            use_visual_memory=None,
            use_proprioceptive_memory=None,
        )
        self.gain = 4.0 + int(hashlib.sha256(ckpt.encode()).hexdigest(), 16) % 3
        self._queue: deque = deque(maxlen=n_action_steps)

    def reset(self) -> None:
        self._queue.clear()

    def count_queued_actions(self) -> int:
        return len(self._queue)

    def select_action(self, batch: dict[str, Any], *, noise: torch.Tensor | None = None) -> torch.Tensor:
        if len(self._queue) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)[:, : self.config.n_action_steps]
            self._queue.extend(actions.transpose(0, 1))
        return self._queue.popleft()

    def predict_action_chunk(
        self, batch: dict[str, Any], *, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        pos = batch["observation.robot_state"]["eef"]["pos"].to(torch.float64)
        b = pos.shape[0]
        if noise is None:
            noise = sample_noise((b, self.config.chunk_size, self.config.max_action_dim), "cpu")
        noise = noise.to(torch.float64)
        target = torch.from_numpy(np.stack([text_pos(t) for t in batch["task"]]))
        delta = target - pos
        move = torch.clamp(delta * self.gain * 10, -1, 1)[:, None, :] + 0.5 * noise[:, :, :3]
        rot = 0.1 * noise[:, :, 3:6]
        close = (delta.norm(dim=-1) < 0.05).to(torch.float64) * 2 - 1
        grip = close[:, None, None].expand(b, self.config.chunk_size, 1)
        return torch.cat([move, rot, grip], dim=-1).to(torch.float32)


class Identity:
    def __call__(self, x: Any) -> Any:
        return x

    def reset(self) -> None:
        pass


class MockTokenizer:
    def encode(self, s: str, add_special_tokens: bool = False) -> list[str]:
        return s.split()

    def get_vocab(self) -> dict[str, int]:
        return {f"▁{w}": i for i, w in enumerate(("apple", "river", "stone", "cloud", "lamp", "fork"))}


def load_mock_policy(ckpt: str, n_action_steps: int) -> tuple:
    """Same 7-tuple as common.load_policy."""
    meta = {
        "policy_type": "mock",
        "chunk_size": MOCK_CHUNK,
        "n_action_steps": n_action_steps,
        "dtype": "float32",
        "compile_model": False,
        "env_observation_hw": list(MOCK_HW),
        "control_mode": "relative",
    }
    ident = Identity()
    return MockPolicy(ckpt, n_action_steps), ident, ident, ident, ident, MockTokenizer(), meta
