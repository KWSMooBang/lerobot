"""Policy-server client (websockets + msgpack + numpy only; usable from any benchmark venv).

    import sys; sys.path.insert(0, "<repo>/scripts/shortcut")
    from bench.core.client import PolicyClient, robosuite_obs_to_wire

    pc = PolicyClient("ws://127.0.0.1:8765")          # hello + checks on connect
    pc.reset("ep-0", episode_seed=7_000_000, instruction="put the bowl on the plate")
    actions = pc.infer("ep-0", robosuite_obs_to_wire(raw_obs), "put the bowl on the plate")   # [n, 7]
    pc.end("ep-0")

The client is responsible for the observation convention (bench.core.protocol docstring): images upright as
in HuggingFaceVLA/libero, state = eef pos + axis-angle + gripper qpos (8, float32). `*_to_wire` helpers
below build it from the usual LIBERO observation layouts; `PolicyClient(orientation=...)` states which
orientation the caller sends and is compared with the server's hello (mismatch → ProtocolError).
"""

from __future__ import annotations

import contextlib
import itertools
import os
import time
import uuid
from typing import Any

import numpy as np
from websockets.sync.client import ClientConnection, connect

from .protocol import (
    ACTION_DIM,
    ORIENT_UPRIGHT,
    PROTOCOL_VERSION,
    STATE_DIM,
    ProtocolError,
    check_response,
    packb,
    unpackb,
    validate_obs,
)

# ---------------------------------------------------------------- observation helpers


def quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion → axis-angle, float32; same formula and precision as lerobot's
    LiberoProcessorStep._quat2axisangle (and robosuite's quat2axisangle). numpy vs torch CPU float32 math
    differ in the last bit for about half of all quaternions (≤ 1 ulp); bench/ours uses the torch step
    itself to stay bit-identical with run_eval.py."""
    q = np.asarray(quat_xyzw, dtype=np.float32)
    w = np.clip(q[3], np.float32(-1.0), np.float32(1.0))
    den = np.sqrt(np.maximum(np.float32(1.0) - w * w, np.float32(0.0)))
    if den <= 1e-10:
        return np.zeros(3, np.float32)
    angle = np.float32(2.0) * np.arccos(w)
    return ((q[:3] / den) * angle).astype(np.float32)


def libero_state(eef_pos: np.ndarray, eef_quat_xyzw: np.ndarray, gripper_qpos: np.ndarray) -> np.ndarray:
    """8-d state of HuggingFaceVLA/libero: eef pos (3) + axis-angle (3) + gripper finger qpos (2)."""
    pos = np.asarray(eef_pos, dtype=np.float64)
    grip = np.asarray(gripper_qpos, dtype=np.float64)
    aa: np.ndarray = quat2axisangle(eef_quat_xyzw).astype(np.float64)
    return np.concatenate([pos, aa, grip]).astype(np.float32)


def upright(img: np.ndarray) -> np.ndarray:
    """Raw LIBERO render → training-data orientation (180° rotation)."""
    return np.ascontiguousarray(np.asarray(img)[::-1, ::-1])


def lerobot_libero_obs_to_wire(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """lerobot `LiberoEnv(obs_type="pixels_agent_pos")` observation → wire obs."""
    rs = obs["robot_state"]
    return {
        "image": upright(obs["pixels"]["image"]),
        "image2": upright(obs["pixels"]["image2"]),
        "state": libero_state(rs["eef"]["pos"], rs["eef"]["quat"], rs["gripper"]["qpos"]),
    }


def robosuite_obs_to_wire(
    obs: dict[str, Any], agent_key: str = "agentview_image", wrist_key: str = "robot0_eye_in_hand_image"
) -> dict[str, np.ndarray]:
    """Raw LIBERO `OffScreenRenderEnv` observation (LIBERO, LIBERO-Plus, LIBERO-PRO) → wire obs."""
    return {
        "image": upright(obs[agent_key]),
        "image2": upright(obs[wrist_key]),
        "state": libero_state(obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]),
    }


# ---------------------------------------------------------------- client


class PolicyClient:
    """Synchronous client. One connection; episodes are namespaced per client so several clients can share
    a server. `orientation` is what the caller's images look like (after any *_to_wire helper)."""

    def __init__(
        self,
        url: str,
        orientation: str = ORIENT_UPRIGHT,
        connect_timeout_s: float = 600.0,
        check_image_hw: bool = False,
        name: str | None = None,
    ):
        self.url = url
        self.orientation = orientation
        self.check_image_hw = check_image_hw
        self.name = name or f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self._stack = contextlib.ExitStack()
        self._ws = self._stack.enter_context(self._connect(connect_timeout_s))
        self._ids = itertools.count()
        self.server_ms = 0.0
        self.calls = 0
        self.info = self.hello()

    def _connect(self, timeout_s: float) -> ClientConnection:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                return connect(self.url, max_size=None, compression=None, open_timeout=30)
            except (OSError, TimeoutError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)

    def _call(self, msg: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(packb(msg))
        raw = self._ws.recv()
        if isinstance(raw, str):
            raise ProtocolError(f"{msg['type']}: text frame from server: {raw[:200]}")
        return check_response(unpackb(raw), msg["type"])

    def hello(self) -> dict[str, Any]:
        info = self._call({"type": "hello", "client": self.name, "protocol_version": PROTOCOL_VERSION})
        problems = []
        if info.get("protocol_version") != PROTOCOL_VERSION:
            problems.append(f"protocol {info.get('protocol_version')} != client {PROTOCOL_VERSION}")
        if info.get("action_dim") != ACTION_DIM:
            problems.append(f"action_dim {info.get('action_dim')} != {ACTION_DIM}")
        if info.get("state_dim") != STATE_DIM:
            problems.append(f"state_dim {info.get('state_dim')} != {STATE_DIM}")
        if info.get("image_orientation") != self.orientation:
            problems.append(
                f"server policy expects images {info.get('image_orientation')!r}, client sends {self.orientation!r}"
            )
        if problems:
            raise ProtocolError(f"{self.url}: " + "; ".join(problems))
        return info

    def _eid(self, episode_id: str) -> str:
        return f"{self.name}/{episode_id}"

    def reset(self, episode_id: str, episode_seed: int, instruction: str) -> None:
        self._call(
            {
                "type": "reset",
                "episode_id": self._eid(episode_id),
                "episode_seed": int(episode_seed),
                "instruction": instruction,
            }
        )

    def infer(
        self, episode_id: str, obs: dict[str, np.ndarray], instruction: str, step: bool = False
    ) -> np.ndarray:
        """Next chunk [n_action_steps, 7] (or one action [1, 7] with step=True), float32."""
        validate_obs(obs, tuple(self.info["image_hw"]) if self.check_image_hw else None)
        resp = self._call(
            {
                "type": "infer",
                "episode_id": self._eid(episode_id),
                "obs": obs,
                "instruction": instruction,
                "step": bool(step),
            }
        )
        actions = np.array(resp["actions"], dtype=np.float32)  # copy: frame buffers are read-only
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ProtocolError(f"infer: actions shape {actions.shape}")
        self.server_ms += float(resp.get("server_ms", 0.0))
        self.calls += 1
        return actions

    def end(self, episode_id: str) -> None:
        self._call({"type": "end", "episode_id": self._eid(episode_id)})

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> PolicyClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
