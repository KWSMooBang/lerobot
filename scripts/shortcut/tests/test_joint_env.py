"""T017 joint_velocity LIBERO env (joint_env.JointVelocityLiberoEnv) on CPU, real LIBERO + robosuite:

- the controller is robosuite JOINT_POSITION with DROID's scale (±0.2 rad per step at |v| = 1), 15 Hz;
- a known joint velocity moves that joint in its direction by ≈ 0.2·v per step (feed-forward tracking), other
  joints stay ≈ still; several joints at once move in the commanded directions;
- gripper +1 closes, −1 opens (finger qpos);
- the default env (lerobot LiberoEnv) is still OSC_POSE, 7-D, 20 Hz.

Runs in a subprocess (MuJoCo GL backend: osmesa, or $SC_TEST_MUJOCO_GL; no GPU). Skipped without LIBERO.

    uv run pytest scripts/shortcut/tests/test_joint_env.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SHORTCUT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("libero") is None or importlib.util.find_spec("robosuite") is None,
    reason="needs hf-libero + robosuite (uv sync --extra libero)",
)

SCRIPT = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/parallel"]
import numpy as np
import common
common.ensure_libero_config()
import joint_control as jc
from envproc import _make_env

out = {}
spec = {"suite": "libero_goal", "task_id": 1, "max_steps": 300, "env_hw": [64, 64], "control_mode": "relative"}
env = _make_env("libero", {**spec, "action_space": "joint_velocity"}, {})
env.init_state_id = 0
obs, _ = env.reset(seed=0)
inner = env._env.env
ctrl = inner.robots[0].controller
out["controller"] = type(ctrl).__name__
out["output_max"] = np.asarray(ctrl.output_max).tolist()
out["control_freq"] = float(inner.control_freq)
out["action_dim"] = int(inner.action_dim)

def run(v, n=6, grip=-1.0):
    o, _ = env.reset(seed=0)
    qs = [o["robot_state"]["joints"]["pos"].copy()]
    for _ in range(n):
        o, *_ = env.step(np.r_[v, grip])
        qs.append(o["robot_state"]["joints"]["pos"].copy())
    return np.diff(np.array(qs), axis=0), o

single = {}
for j in (0, 3, 6):
    for sign in (1, -1):
        v = np.zeros(7); v[j] = 0.25 * sign
        d, _ = run(v)
        ratio = d[:, j] / (jc.MAX_JOINT_DELTA * v[j])
        leak = np.abs(np.delete(d, j, axis=1)).max() / (jc.MAX_JOINT_DELTA * 0.25)
        single[f"{j}{'+' if sign > 0 else '-'}"] = {"ratio": ratio.tolist(), "leak": float(leak)}
out["single"] = single
v = np.array([0.2, -0.15, 0.1, 0.2, -0.1, 0.0, -0.2])
d, o = run(v)
out["multi"] = {"v": v.tolist(), "dq": d.sum(0).tolist(), "steps": len(d)}
g0 = o["robot_state"]["gripper"]["qpos"].copy()
for _ in range(15):
    o, *_ = env.step(np.r_[np.zeros(7), 1.0])
g1 = o["robot_state"]["gripper"]["qpos"].copy()
for _ in range(15):
    o, *_ = env.step(np.r_[np.zeros(7), -1.0])
g2 = o["robot_state"]["gripper"]["qpos"].copy()
out["gripper_width"] = [float(g[0] - g[1]) for g in (g0, g1, g2)]
try:
    env.step(np.zeros(7))
    out["rejects_7d"] = False
except ValueError:
    out["rejects_7d"] = True
env.close()

osc = _make_env("libero", spec, {})
osc.init_state_id = 0
osc.reset(seed=0)
oi = osc._env.env
out["osc"] = {"controller": type(oi.robots[0].controller).__name__, "control_freq": float(oi.control_freq),
              "action_dim": int(oi.action_dim), "class": type(osc).__name__}
osc.close()
print("@@" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def result() -> dict:
    # CPU rendering; not os.environ's MUJOCO_GL (importing common sets egl in this process). SC_TEST_MUJOCO_GL overrides.
    gl = os.environ.get("SC_TEST_MUJOCO_GL", "osmesa")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": gl, "PYOPENGL_PLATFORM": gl}
    r = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(SHORTCUT)], capture_output=True, text=True, env=env, timeout=1800
    )
    line = next((x for x in r.stdout.splitlines() if x.startswith("@@")), None)
    assert line is not None, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(line[2:])


def test_controller_is_droid_scaled_joint_position_at_15hz(result):
    assert result["controller"] == "JointPositionController"
    assert result["output_max"] == pytest.approx([0.2] * 7)
    assert result["control_freq"] == 15 and result["action_dim"] == 8
    assert result["rejects_7d"]


def test_single_joint_velocity_moves_that_joint_by_delta(result):
    for name, r in result["single"].items():
        ratio = r["ratio"]
        assert ratio[0] > 0.3, (name, ratio)  # right direction from rest
        assert all(0.85 <= x <= 1.1 for x in ratio[1:]), (name, ratio)  # ≈ 0.2·v rad per step after the first
        assert r["leak"] < 0.2, (name, r["leak"])


def test_multi_joint_command_moves_in_commanded_directions(result):
    m = result["multi"]
    for v, dq in zip(m["v"], m["dq"], strict=True):
        if abs(v) >= 0.15:  # the large joints; joint 6 (12 Nm) lags when the whole arm accelerates
            assert dq * v > 0 and abs(dq) > 0.4 * 0.2 * abs(v) * m["steps"], (v, dq)


def test_gripper_closes_and_opens(result):
    w_open, w_closed, w_reopened = result["gripper_width"]
    assert w_open > 0.07 and w_closed < 0.01 and w_reopened > 0.07


def test_default_env_is_unchanged_osc(result):
    assert result["osc"] == {
        "controller": "OperationalSpaceController",
        "control_freq": 20.0,
        "action_dim": 7,
        "class": "LiberoEnv",
    }
