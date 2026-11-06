"""DROID control contract constants (T017), importable without LIBERO / robosuite (coordinator, policy server).

Sources: droid-dataset/droid robot_ik/robot_ik_solver.py (RobotIKSolver: relative_max_joint_delta = 0.2 for all
7 joints, max_joint_delta = 0.2, control_hz = 15; joint_velocity_to_delta(v) = v · max_joint_delta after scaling v
so that max |v| ≤ 1), franka/robot.py create_action_dict (joint target = measured joint position + joint_delta;
gripper_position = clip(action[-1], 0, 1); get_gripper_position = 1 − width / max_width), openpi
examples/droid/main.py (DROID_CONTROL_FREQUENCY = 15, gripper binarized at 0.5, action clipped to [-1, 1],
open_loop_horizon = 8). The env side (joint_env.JointVelocityLiberoEnv) implements them in robosuite.
"""

from __future__ import annotations

from typing import Any

ACTION_SPACE = "joint_velocity"  # = bench.core.protocol.ACTION_SPACE_JOINT_VELOCITY
CONTROL_FREQ = 15
MAX_JOINT_DELTA = 0.2  # rad per control step at |v| = 1
GRIPPER_THRESHOLD = 0.5  # openpi: action[-1] > 0.5 → 1 (closed)
OPEN_LOOP_HORIZON = 8
# Panda (robosuite PandaGripper): finger_joint1 range [0, 0.04], finger_joint2 [-0.04, 0] → max width 0.08 m.
PANDA_MAX_GRIPPER_WIDTH = 0.08
# env-side tracking (joint_env module doc; tests/test_joint_env.py)
JOINT_KP = 400.0
JOINT_DAMPING_RATIO = 1.0
FEEDFORWARD = True


def controller_overrides() -> dict[str, Any]:
    """Overrides of robosuite/controllers/config/joint_position.json."""
    return {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": MAX_JOINT_DELTA,
        "output_min": -MAX_JOINT_DELTA,
        "kp": JOINT_KP,
        "damping_ratio": JOINT_DAMPING_RATIO,
        "impedance_mode": "fixed",
        "kp_limits": [0, 5000],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
    }


def policy_contract() -> dict[str, Any]:
    """What a DROID-contract policy server declares (hello identity) and the client checks."""
    return {"action_space": ACTION_SPACE, "control_freq": CONTROL_FREQ, "max_joint_delta": MAX_JOINT_DELTA}


def env_identity() -> dict[str, Any]:
    """The env side of the contract, recorded in run identities next to the server's."""
    return {
        "controller": "robosuite JOINT_POSITION, target = q + 0.2*v"
        + (", velocity feed-forward tracking" if FEEDFORWARD else ""),
        "joint_kp": JOINT_KP,
        "joint_damping_ratio": JOINT_DAMPING_RATIO,
        "env_fresh_obs": True,
    }
