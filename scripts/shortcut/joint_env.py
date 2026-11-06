"""LIBERO under DROID's control contract (T017): joint-velocity arm actions at 15 Hz, binary gripper.

The default env of the harness (lerobot `LiberoEnv`: robosuite OSC_POSE, delta-EEF, 20 Hz) is not touched; this
subclass is used only when a policy server declares `action_space: joint_velocity` (bench.core.protocol).

DROID (droid-dataset/droid: robot_env.RobotEnv(action_space="joint_velocity", gripper_action_space="position"),
franka/robot.create_action_dict, robot_ik/robot_ik_solver.RobotIKSolver):
    joint_delta    = joint_velocity_to_delta(v) = v * max_joint_delta        (max_joint_delta = 0.2 rad, |v| <= 1)
    joint target   = measured joint position + joint_delta                   (joint impedance, every 1/15 s)
    gripper        = position target in [0, 1] (1 = closed); openpi binarizes at 0.5 and clips to [-1, 1]

Here (robosuite 1.4 JOINT_POSITION controller, "target = measured q + delta", plus velocity feed-forward):
    input [-1, 1] → delta = 0.2·v rad (output range ±max_joint_delta = joint_velocity_to_delta for clipped v);
    env action = [v (7), gripper (1)] with LIBERO's gripper convention -1 open / +1 close.
    control_freq = 15 (DROID_CONTROL_FREQUENCY): one policy action spans 1/15 s like on the robot. Re-scaling
    the deltas to 20 Hz instead would time-warp the chunk (8 actions = 0.4 s instead of 0.533 s).
    Tracking (FEEDFORWARD): robosuite's stock law tau = M·(kp·(q_goal − q) − kd·q̇) damps towards zero velocity
    while the goal is re-anchored every step, so a constant command moves the arm only ~0.18 (kp 50) … ~0.6
    (kp ≥ 900, torque-saturated) of the commanded delta per step. The run_controller of the instance is replaced
    by trajectory tracking of the same step target: reference q_ref(t) = q_start + (δ/T)·t over the step
    (q_start = measured q at the step, T = 1/15 s), tau = M·(kp·(q_ref − q) + kd·(δ/T − q̇)) + gravity comp.,
    clipped by robosuite to the Panda torque limits. A constant command is then tracked at ~1 per step (the
    real Franka's 1 kHz joint impedance follows these ≤ 3 rad/s targets closely); tests/test_joint_env.py
    measures direction and magnitude.
    FRESH_OBS: robosuite steps int((1/15) / 0.002) = 33 substeps (66 ms) per action, while observables sample
    on a 1/15 s timer, so the observation a step returns would come from a varying substep (or the previous
    step). The step's observation is re-sampled at its end (`_get_observations(force_update=True)`, one more
    render). At 20 Hz (25 substeps) the OSC env does not have this problem and is left as it is.
"""

from __future__ import annotations

import types
from typing import Any

import joint_control as jc
import numpy as np
from gymnasium import spaces
from libero.libero.envs import OffScreenRenderEnv

from lerobot.envs.libero import LiberoEnv

ACTION_DIM = 8  # 7 joint velocities + gripper
DUMMY_ACTION = np.array([0.0] * 7 + [-1.0], dtype=np.float32)  # hold joints, gripper open


# ---------------------------------------------------------------- feed-forward tracking (see FEEDFORWARD)


def _ff_set_goal(self: Any, action: Any, set_qpos: Any = None) -> None:
    type(self).set_goal(self, action, set_qpos)  # updates state, goal_qpos = q + scale(action)
    self._ff_start = np.array(self.joint_pos)
    self._ff_vel = (np.array(self.goal_qpos) - self._ff_start) * float(self.control_freq)
    self._ff_k = 0


def _ff_run_controller(self: Any) -> np.ndarray:
    if self.goal_qpos is None:
        self.set_goal(np.zeros(self.control_dim))
    self.update()
    self._ff_k += 1
    t = min(self._ff_k * self.model_timestep, 1.0 / float(self.control_freq))
    q_ref = self._ff_start + self._ff_vel * t
    desired = self.kp * (q_ref - self.joint_pos) + self.kd * (self._ff_vel - self.joint_vel)
    self.torques = np.dot(self.mass_matrix, desired) + self.torque_compensation
    self.new_update = True  # Controller.run_controller
    return self.torques


def install_feedforward(ctrl: Any) -> None:
    ctrl._ff_start = np.array(ctrl.joint_pos)
    ctrl._ff_vel = np.zeros_like(ctrl._ff_start)
    ctrl._ff_k = 0
    ctrl.set_goal = types.MethodType(_ff_set_goal, ctrl)
    ctrl.run_controller = types.MethodType(_ff_run_controller, ctrl)


class JointVelocityLiberoEnv(LiberoEnv):
    """lerobot LiberoEnv with the JOINT_POSITION controller (see module doc). Same observations; actions 8-D."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["control_freq"] = jc.CONTROL_FREQ
        kwargs["control_mode"] = "relative"  # unused by the joint controller; LiberoEnv validates it
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        env = OffScreenRenderEnv(
            bddl_file_name=self._task_bddl_file,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            control_freq=self.control_freq,
            controller="JOINT_POSITION",
            hard_reset=self.hard_reset,
        )
        inner = env.env
        over = jc.controller_overrides()
        # robots are rebuilt from robot_configs on hard resets and reload their controller on every reset
        for cfg in inner.robot_configs:
            cfg["controller_config"].update(over)
        for robot in inner.robots:
            robot.controller_config.update(over)
        self._env = env
        self._reset_inner()

    def _reset_inner(self) -> dict:
        """inner env reset (robots + controllers are rebuilt) → controller checks + feed-forward patch."""
        assert self._env is not None
        raw_obs = self._env.reset()
        inner = self._env.env
        for robot in inner.robots:
            ctrl = robot.controller
            assert type(ctrl).__name__ == "JointPositionController", type(ctrl)
            assert np.allclose(ctrl.output_max, jc.MAX_JOINT_DELTA), ctrl.output_max
            assert np.allclose(ctrl.kp, jc.JOINT_KP), ctrl.kp
            if jc.FEEDFORWARD:
                install_feedforward(ctrl)
        assert int(inner.control_freq) == jc.CONTROL_FREQ
        return raw_obs

    def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[dict, dict]:
        """LiberoEnv.reset with an 8-D settle action (LiberoEnv's dummy is the 7-D OSC no-op)."""
        self._ensure_env()
        assert self._env is not None
        super(LiberoEnv, self).reset(seed=seed)
        self._env.seed(seed)
        raw_obs = self._reset_inner()
        if self.init_states and self._init_states is not None:
            raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
            self.init_state_id += self._reset_stride
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(DUMMY_ACTION)
        if self.num_steps_wait:
            raw_obs = self._env.env._get_observations(force_update=True)
        return self._format_raw_obs(raw_obs), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        """LiberoEnv.step, but the observation is re-sampled at the end of the step (see FRESH_OBS)."""
        a = np.asarray(action)
        if a.shape != (ACTION_DIM,):
            raise ValueError(f"{jc.ACTION_SPACE} env expects actions ({ACTION_DIM},), got {a.shape}")
        self._ensure_env()
        assert self._env is not None
        _raw, reward, done, info = self._env.step(a)
        raw_obs = self._env.env._get_observations(force_update=True)
        is_success = self._env.check_success()
        info.update({"task": self.task, "task_id": self.task_id, "done": done, "is_success": is_success})
        return self._format_raw_obs(raw_obs), reward, done or is_success, False, info
