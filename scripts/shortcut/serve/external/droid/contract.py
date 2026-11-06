"""openpi π0.5-DROID input/output contract on our wire (T017). numpy + PIL only (no torch / lerobot / LIBERO).

openpi (Physical-Intelligence/openpi): policies/droid_policy.py (DroidInputs / DroidOutputs, model_type PI05),
transforms.py (Normalize / Unnormalize with quantiles, PadStatesAndActions), training/config.py (pi05_droid:
Pi0Config(action_horizon=15, pi05=True) → discrete_state_input=True, max_token_len=200, quantile norm, asset
"droid"), models/tokenizer.py (state discretized into 256 bins in the prompt), openpi_client/image_tools.py
(resize_with_pad), examples/droid/main.py (client: resize_with_pad 224, clip, gripper binarization, 8 open-loop
steps). DROID: droid/franka/robot.py (gripper_position = 1 − width / max_width).

  openpi DROID                                     here (wire → policy, policy → wire)
  observation/exterior_image_1_left → base_0_rgb   wire `image` (agentview, upright)        → base_0_rgb
  observation/wrist_image_left → left_wrist_0_rgb  wire `image2` (eye-in-hand, upright)     → left_wrist_0_rgb
  right_wrist_0_rgb = zeros, image_mask False      key absent → lerobot pi05 pads −1 (= zeros·2−1), mask False
  client resize_with_pad(img, 224, 224) PIL        resize_with_pad (same code) on the 256² render
  state = joint_position (7) + gripper_position    wire joint_pos (7) + gripper_position(finger qpos)
  Normalize quantile, 8-D stats, ε 1e-6            lerobot QUANTILES with q99 + 1e-6 (same arithmetic)
  prompt "Task: …, State: <8 bins>;\\nAction: "     lerobot Pi05PrepareStateTokenizerProcessorStep on 8-D state
  Unnormalize quantile, actions[..., :8]           lerobot QUANTILES on 8-D actions (output feature (8,))
  client: clip [-1, 1], gripper > 0.5 → 1          to_env_actions: clip, gripper > 0.5 → +1 (close) else −1
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
NORM_STATS = HERE / "norm_stats.json"
# gs://openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json (4,460 B; vendored here with a trailing
# newline, the sha is over the file without trailing newlines = the downloaded bytes)
NORM_STATS_URL = (
    "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json"
)
NORM_STATS_SHA256 = "403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b"
CKPT_REPO = "lerobot/pi05_droid"
CKPT_SHA = "72824c0a93f00ce5bb8bedb7feb58953ba1da364"
DROID_DIM = 8  # joint (7) + gripper (1), state and action
OPENPI_EPS = 1e-6  # transforms._normalize_quantile / _unnormalize_quantile
MODEL_IMAGE_HW = (224, 224)
CAMERA_MAP = {
    "image": "observation.images.base_0_rgb",  # exterior_image_1_left
    "image2": "observation.images.left_wrist_0_rgb",  # wrist_image_left
}
EMPTY_CAMERA = "observation.images.right_wrist_0_rgb"
GRIPPER_THRESHOLD = 0.5
PANDA_MAX_GRIPPER_WIDTH = 0.08  # = joint_control.PANDA_MAX_GRIPPER_WIDTH


# ---------------------------------------------------------------- stats


def stats_sha256(path: Path = NORM_STATS) -> str:
    return hashlib.sha256(path.read_bytes().rstrip(b"\n")).hexdigest()


def load_openpi_stats(path: Path = NORM_STATS, check_sha: bool = True) -> dict[str, dict[str, np.ndarray]]:
    """openpi norm_stats.json → {"state": {mean, std, q01, q99}, "actions": {...}} (float64, 32-long as stored)."""
    if check_sha and stats_sha256(path) != NORM_STATS_SHA256:
        raise ValueError(f"{path}: sha256 {stats_sha256(path)} != pinned {NORM_STATS_SHA256}")
    raw = json.loads(path.read_text())["norm_stats"]
    return {k: {s: np.asarray(v, dtype=np.float64) for s, v in raw[k].items()} for k in ("state", "actions")}


def lerobot_stats(openpi: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """Stats for lerobot's QUANTILES normalizer on 8-D state / action giving openpi's arithmetic:
    lerobot  y = 2 (x − q01) / (q99' − q01) − 1,   x = (y + 1) (q99' − q01) / 2 + q01
    openpi   y = (x − q01) / (q99 − q01 + 1e-6) · 2 − 1,   x = (y + 1) / 2 · (q99 − q01 + 1e-6) + q01
    → q99' = q99 + 1e-6 (8 dims: openpi slices the stats to the 8-D state; the policy's 32-D output is
    cut to 8 before unnormalization, the dims openpi's DroidOutputs keeps)."""
    out = {}
    for ours, theirs in (("observation.state", "state"), ("action", "actions")):
        s = openpi[theirs]
        q01 = s["q01"][:DROID_DIM]
        q99 = s["q99"][:DROID_DIM] + OPENPI_EPS
        out[ours] = {
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
            "mean": s["mean"][:DROID_DIM].astype(np.float32),
            "std": s["std"][:DROID_DIM].astype(np.float32),
        }
    return out


def openpi_normalize(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """transforms.Normalize._normalize_quantile (reference for tests)."""
    q01, q99 = stats["q01"][..., : x.shape[-1]], stats["q99"][..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + OPENPI_EPS) * 2.0 - 1.0


def openpi_unnormalize(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """transforms.Unnormalize._unnormalize_quantile (reference for tests)."""
    q01, q99 = stats["q01"], stats["q99"]
    dim = q01.shape[-1]
    if dim < x.shape[-1]:
        return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + OPENPI_EPS) + q01, x[..., dim:]], -1)
    return (x + 1.0) / 2.0 * (q99 - q01 + OPENPI_EPS) + q01


def discretize_state(normalized: np.ndarray) -> np.ndarray:
    """models/tokenizer.PaligemmaTokenizer.tokenize: 256 bins over [-1, 1]."""
    return np.asarray(np.digitize(normalized, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1)


# ---------------------------------------------------------------- observation


def resize_with_pad(image: np.ndarray, height: int, width: int, method: int = Image.BILINEAR) -> np.ndarray:
    """openpi_client.image_tools.resize_with_pad for one uint8 HWC image (same arithmetic)."""
    if image.shape[-3:-1] == (height, width):
        return image
    im = Image.fromarray(image)
    cur_w, cur_h = im.size
    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    resized = im.resize((new_w, new_h), resample=method)
    canvas = Image.new(resized.mode, (width, height), 0)
    canvas.paste(resized, (max(0, int((width - new_w) / 2)), max(0, int((height - new_h) / 2))))
    return np.asarray(canvas)


def gripper_position(finger_qpos: np.ndarray) -> np.float32:
    """Panda finger qpos (2: +q, −q, each 0..0.04 m) → DROID gripper_position (0 open .. 1 closed).

    DROID's robot state is 1 − width / max_width of its Robotiq 2F-85 (franka/robot.py); here width = q0 − q1
    and max_width = 0.08 m (the Panda's stroke), clipped to [0, 1]. LIBERO open ≈ 0.026, closed empty ≈ 0.98."""
    q = np.asarray(finger_qpos, dtype=np.float64)
    width = q[0] - q[1]
    return np.float32(np.clip(1.0 - width / PANDA_MAX_GRIPPER_WIDTH, 0.0, 1.0))


def droid_state(wire_obs: dict[str, Any]) -> np.ndarray:
    """DroidInputs state: joint_position (7) + gripper_position (1), float32 [8]."""
    jp = np.asarray(wire_obs["joint_pos"], dtype=np.float32)
    return np.concatenate([jp, [gripper_position(wire_obs["state"][6:8])]]).astype(np.float32)


def droid_images(wire_obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Wire images (upright uint8) → {policy key: uint8 224×224×3} (right wrist slot left out: padded, masked)."""
    return {
        key: resize_with_pad(np.asarray(wire_obs[slot]), *MODEL_IMAGE_HW) for slot, key in CAMERA_MAP.items()
    }


# ---------------------------------------------------------------- action


def to_env_actions(chunk: np.ndarray) -> np.ndarray:
    """Unnormalized policy actions (T, 8) → joint_velocity env actions (T, 8) float32, as openpi's DROID client
    builds them: gripper > 0.5 → 1 (closed) else 0, then clip to [-1, 1]; gripper mapped to LIBERO's −1 open /
    +1 close. Joint velocities stay normalized (the env scales them by 0.2 rad per step)."""
    c = np.asarray(chunk, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] != DROID_DIM:
        raise ValueError(f"DROID chunk must be (T, {DROID_DIM}), got {c.shape}")
    out = np.clip(c, -1.0, 1.0).astype(np.float32)
    out[:, 7] = np.where(c[:, 7] > GRIPPER_THRESHOLD, np.float32(1.0), np.float32(-1.0))
    return out
