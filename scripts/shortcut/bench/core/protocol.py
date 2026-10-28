"""Policy-server wire protocol (TESTBED_DESIGN.md §1). numpy + msgpack only.

Transport: one websocket per client, one binary msgpack frame per message, strict request → response.
numpy arrays travel as openpi's msgpack_numpy does (`{"__ndarray__": True, "data", "dtype", "shape"}`), so
openpi-style clients and servers can talk to this one.

Requests (`type` field) and responses (`ok: true` + fields, or `ok: false, error: str`):
  hello                                       → {policy_type, ckpt, ckpt_id, exec_mode, action_dim=7, chunk_size,
                                                 n_action_steps, image_keys, state_dim=8, image_orientation,
                                                 protocol_version, exec, inference, server}
  reset {episode_id, episode_seed, instruction}   → {}            new noise stream + empty action queue
  infer {episode_id, obs:{image, image2, state}, instruction[, step]}
                                              → {actions: float32[n, 7], server_ms}
        without `step` (default) the server returns the next n_action_steps actions (one chunk); with
        `step: true` it returns one action [1, 7] from its per-episode queue (new chunk when empty).
  end {episode_id}                            → {}            optional; frees the episode (batching waits less)

Observation convention (the client's job, checked against hello): images are uint8 [H, W, 3] in the
orientation of the training data HuggingFaceVLA/libero, i.e. upright (raw LIBERO/robosuite renders rotated
180°: `img[::-1, ::-1]`); `image` = agentview, `image2` = wrist. state = float32 [8] = eef pos (3, world m)
+ eef axis-angle (3) + gripper finger qpos (2).
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

PROTOCOL_VERSION = "sc-policy/1"
ACTION_DIM = 7
STATE_DIM = 8
# image_orientation values. The server states what its policy was trained on; the client states what it sends.
ORIENT_UPRIGHT = "hf_libero_upright"  # HuggingFaceVLA/libero = raw LIBERO render rotated 180°
ORIENT_RAW = "libero_raw"  # straight from robosuite / LiberoEnv (upside down)
IMAGE_SLOTS = ("image", "image2")  # agentview, wrist
REQUEST_TYPES = ("hello", "reset", "infer", "end")


class ProtocolError(RuntimeError):
    """Malformed message, version mismatch, or an error reported by the other side."""


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in "VOc":
        raise TypeError(f"unsupported dtype {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _decode(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    """Arrays come back read-only views of the frame; copy before writing into them."""
    return msgpack.unpackb(data, object_hook=_decode, raw=False)


def error(msg: str) -> dict[str, Any]:
    return {"ok": False, "error": msg}


def check_response(resp: Any, what: str) -> dict[str, Any]:
    if not isinstance(resp, dict) or "ok" not in resp:
        raise ProtocolError(f"{what}: malformed response {type(resp).__name__}")
    if not resp["ok"]:
        raise ProtocolError(f"{what}: server error: {resp.get('error')}")
    return resp


def validate_obs(obs: dict[str, Any], image_hw: tuple[int, int] | None = None) -> None:
    """Shape / dtype check of one wire observation (both sides call it)."""
    for k in IMAGE_SLOTS:
        if k not in obs:
            raise ProtocolError(f"obs is missing {k!r} (have {sorted(obs)})")
        img = obs[k]
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            desc = f"{img.dtype} {img.shape}" if isinstance(img, np.ndarray) else type(img).__name__
            raise ProtocolError(f"obs[{k!r}] must be uint8 [H, W, 3], got {desc}")
        if image_hw is not None and tuple(img.shape[:2]) != tuple(image_hw):
            raise ProtocolError(f"obs[{k!r}] is {img.shape[:2]}, server expects {tuple(image_hw)}")
    st = obs.get("state")
    if not isinstance(st, np.ndarray) or st.dtype != np.float32 or st.shape != (STATE_DIM,):
        desc = f"{st.dtype} {st.shape}" if isinstance(st, np.ndarray) else type(st).__name__
        raise ProtocolError(f"obs['state'] must be float32 [{STATE_DIM}], got {desc}")
