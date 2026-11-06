# openpi π0.5-DROID zero-shot on the shortcut testbed (T017)

`lerobot/pi05_droid` (openpi's `pi05_droid`, trained on DROID only) evaluated on LIBERO **without LIBERO
training**: the policy sees what openpi's DROID pipeline feeds it, and its joint-velocity actions run in a LIBERO
env under DROID's control contract. `sc-bench --ckpt pi05droid` launches the server (lerobot venv, no extra setup)
and runs the `ours` battery unchanged; the joint env is chosen from the server's `hello`.

| file                               | what                                                                                           |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| `contract.py`                      | openpi DROID input/output contract (numpy + PIL): stats, resize_with_pad, state, actions       |
| `norm_stats.json`                  | openpi `assets/droid/norm_stats.json` (vendored, sha pinned)                                   |
| `server.py`                        | sc-policy/1 server (engine for `serve/policy_server.Server`), exec_mode single                 |
| `serve.sh`                         | launcher sc-bench uses for registry rows with `server=droid`                                   |
| `cpu_check.py`                     | real checkpoint on CPU, one real LIBERO observation → actions (finite, in range), joint motion |
| `smoke.sh`                         | GPU smoke: `droid_smoke` battery (36 episodes) + summary + main-battery GPU-h estimate         |
| `tests/`                           | CPU unit tests (mock model): normalization vs openpi, contract, protocol, sc-bench mock flow   |
| `../../../joint_control.py`        | DROID control constants (no LIBERO import)                                                     |
| `../../../joint_env.py`            | `JointVelocityLiberoEnv` (robosuite JOINT_POSITION + feed-forward tracking, 15 Hz)             |
| `../../../tests/test_joint_env.py` | CPU LIBERO checks of the joint env                                                             |

## Pins

| what                                                                   | value                                                                                                          |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| checkpoint `lerobot/pi05_droid`                                        | `72824c0a93f00ce5bb8bedb7feb58953ba1da364` (type pi05, chunk 15, 32-D padded features, fp32, **no stats**)     |
| norm stats (openpi `pi05_droid/assets/droid/norm_stats.json`, 4,460 B) | sha256 `403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b`                                      |
| tokenizer                                                              | PaliGemma sentencepiece, local copy (`SC_TOKENIZER_DIR` rule of `common.load_tokenizer`)                       |
| registry row                                                           | `pi05droid`: `server=droid;n_action_steps=8;zero_shot=1;ckpt_sha=…;stats_sha256=…;action_space=joint_velocity` |

The vendored stats file ends with a newline (pre-commit's end-of-file-fixer); its sha is computed over the bytes
without trailing newlines, which are the downloaded bytes. `server.py` refuses other checkpoints / revisions
(`--allow-unpinned`; also needed for any `--revision` other than the pinned sha, e.g. `main`). The identity
records the commit sha of the snapshot the hub resolved (`.../snapshots/<sha>`), never the requested name.

## Contract: openpi DROID vs ours

openpi side: `src/openpi/policies/droid_policy.py` (DroidInputs / DroidOutputs, `ModelType.PI05`),
`transforms.py`, `training/config.py` (`pi05_droid` = `Pi0Config(action_horizon=15, pi05=True)` →
`discrete_state_input=True`, `max_token_len=200`, quantile norm, asset `droid`, no delta-action transform),
`models/tokenizer.py`, `packages/openpi-client/.../image_tools.py`, `examples/droid/main.py`. DROID side:
`droid/robot_ik/robot_ik_solver.py`, `droid/franka/robot.py`.

| item               | openpi DROID                                                                  | ours                                                                                                    | same?                           |
| ------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------- |
| exterior camera    | `exterior_image_1_left` → `base_0_rgb`                                        | wire `image` (agentview, upright = raw `[::-1, ::-1]`) → `base_0_rgb`                                   | slot yes; view differs (scene)  |
| wrist camera       | `wrist_image_left` → `left_wrist_0_rgb`                                       | wire `image2` (eye-in-hand, upright) → `left_wrist_0_rgb`                                               | slot yes                        |
| 3rd image          | `right_wrist_0_rgb` = zeros, mask False                                       | key absent → lerobot pi05 pads −1 (= zeros·2−1) with mask False                                         | yes (same values)               |
| resize             | client `resize_with_pad(img, 224, 224)` (PIL bilinear, black pad)             | same code (`contract.resize_with_pad`) on the 256² render (square → plain bilinear resize)              | yes                             |
| state              | `joint_position` (7) + `gripper_position` (1)                                 | wire `joint_pos` (robot0_joint_pos) + `clip(1 − (q0 − q1)/0.08, 0, 1)` from the finger qpos             | yes (gripper by analogy, below) |
| normalization      | quantile, stats sliced to the 8-D state, `(x − q01)/(q99 − q01 + 1e-6)·2 − 1` | lerobot QUANTILES on 8-D state/action with `q99' = q99 + 1e-6` → same formula (tested ≤ 2e-6)           | yes                             |
| state in prompt    | `Task: …, State: <8 bins>;\nAction: ` (256 bins, before padding to 32)        | lerobot `Pi05PrepareStateTokenizerProcessorStep` on the 8-D normalized state (state feature shape (8,)) | yes                             |
| tokens             | PaliGemma, max 200                                                            | same tokenizer, `tokenizer_max_length` 200, right padding                                               | yes                             |
| actions            | 32-D output, unnormalize (q01/q99 + 1e-6), `actions[..., :8]`                 | output feature (8,) → model output cut to 8, unnormalized with the same stats                           | yes                             |
| client post        | gripper `> 0.5` → 1 else 0, `np.clip(action, −1, 1)`                          | `to_env_actions`: clip, gripper `> 0.5` → +1 (close) else −1 (LIBERO's convention)                      | yes (sign mapping only)         |
| horizon            | chunk 15, 8 executed open-loop (`open_loop_horizon`)                          | chunk 15, `n_action_steps` 8 (hello, registry)                                                          | yes                             |
| control            | `RobotEnv(action_space="joint_velocity", gripper="position")`, 15 Hz          | `JointVelocityLiberoEnv` at 15 Hz (below)                                                               | contract yes, dynamics sim      |
| flow steps / noise | 10 / random                                                                   | 10 (checkpoint config) / `torch.manual_seed(episode_seed)`, per-episode RNG (exec_mode single)          | deterministic per episode       |
| dtype              | (JAX bf16)                                                                    | bf16 on CUDA (as pi05lib), fp32 on CPU; recorded in the identity                                        | —                               |

**Gripper state.** DROID's `gripper_position` is `1 − width / max_width` of its Robotiq 2F-85
(`franka/robot.py`; 0 open … 1 closed, stats q01 0 / q99 0.991). The Panda's width is `q0 − q1` of the two finger
joints (range 0..0.04 m each, max width 0.08 m), so the same normalized closure is `1 − (q0 − q1)/0.08`. LIBERO
open ≈ 0.026, closed on nothing ≈ 0.98. On an object both report the object's width relative to their own stroke.

**Camera orientation.** The wire images are upright (HuggingFaceVLA/libero convention). In the wrist view the
gripper fingers are at the bottom edge, which is where DROID's wrist frames have them; the agentview is upright
(gravity down) like DROID's exterior ZED views. This was checked by looking at the 224² model inputs that
`cpu_check.py` writes; no DROID frame was compared pixel-wise. The scene itself is out of distribution (LIBERO's
agentview looks at the robot from the front; DROID's exterior cameras are to the side) — that is the zero-shot
setting, not a contract error.

## Control: DROID joint velocity in LIBERO (`joint_control.py`, `joint_env.py`)

DROID: `joint_delta = joint_velocity_to_delta(v) = v · 0.2 rad` (`RobotIKSolver.max_joint_delta`, all joints
0.2; the rescale by max |v| never triggers after openpi's clip to [−1, 1]); joint target = **measured** joint
position + delta, sent to the Franka's joint impedance controller every 1/15 s.

Ours: robosuite 1.4 `JOINT_POSITION` controller (also "goal = current q + scaled delta", delta mode) with input
[−1, 1] → output ±0.2 rad, so the env action is `[v (7), gripper ±1]` and the scale is DROID's.

- **15 Hz vs 20 Hz: the env runs at `control_freq=15`.** One policy action then spans 1/15 s as on the robot and
  the 8 open-loop actions cover 0.533 s. Rescaling the deltas to 20 Hz (×0.75) would keep joint speed but
  time-warp the chunk (8 actions = 0.4 s, 75 % of the planned path before re-planning). The physics timestep is
  unchanged (2 ms); 15 Hz gives `int((1/15)/0.002) = 33` substeps = 66 ms per action (1 % short of 1/15 s).
  `max_steps` stays the suite value (spatial 220, object 280, goal 300, 10: 520), so an episode lasts 4/3 of the
  simulated time an OSC episode does. Settle steps: 10 × no-op at 15 Hz (0.67 s instead of 0.5 s).
- **Fresh observations.** robosuite's observables sample on a 1/15 s timer, but a step is 66 ms, so the returned
  observation would come from a varying substep (or the previous step: the first step returned the reset state).
  The joint env re-samples the observation at the end of every step (`_get_observations(force_update=True)`, one
  extra render). The OSC env (25 substeps = 50 ms exactly) is not changed.
- **Tracking (feed-forward).** robosuite's stock law `τ = M (kp (q_goal − q) − kd q̇)` damps towards zero
  velocity while the goal is re-anchored every step; measured: a constant command moves a joint only 0.18 (kp 50)
  … ≈ 0.6 (kp ≥ 900, torque-saturated) of the commanded delta per step. The controller instance's `run_controller`
  is replaced by tracking of the same step target: `q_ref(t) = q_start + (δ/T)·t` over the step,
  `τ = M (kp (q_ref − q) + kd (δ/T − q̇)) + gravity comp.`, clipped to the Panda torque limits (80 / 12 Nm);
  kp 400, critically damped. The real Franka's 1 kHz joint impedance follows these ≤ 3 rad/s targets closely, so
  ≈ 1 per step is the reference.
- **Gripper:** `+1` close / `−1` open (LIBERO's PandaGripper, unchanged).

Measured on CPU (`tests/test_joint_env.py`, libero_goal task 1, init 0; ratio = joint motion per step /
(0.2·v)):

| command                                                      | step 1 | steps 2–6                                   | max motion of the other joints |
| ------------------------------------------------------------ | ------ | ------------------------------------------- | ------------------------------ |
| joint 0, v = ±0.25                                           | 0.46   | 0.98–1.00                                   | 0.14 of the commanded delta    |
| joint 3, v = ±0.25                                           | 0.72   | 0.98–1.00                                   | 0.06                           |
| joint 6, v = ±0.25                                           | 0.54   | 0.92–0.93                                   | 0.02                           |
| all 7 at once (0.2, −0.15, 0.1, 0.2, −0.1, 0, −0.2), 6 steps |        | total 0.89, 0.92, 0.91, 0.93, 0.82, –, 0.52 |                                |

All directions are right. Joint 6 (wrist, 12 Nm) lags when the whole arm accelerates (torque limit). Gripper
width 0.079 m open → 0.002 m after 15 close steps → 0.078 m after 15 open steps. The default env is still
`OperationalSpaceController`, 7-D, 20 Hz.

## Identity and mixing

- `hello`: `action_space: joint_velocity`, `action_dim: 8`; the client lists the action spaces its env executes
  (`PolicyClient(action_spaces=...)`, default LIBERO OSC only) and refuses others at hello; it requires
  `joint_pos` in every obs for this action space. Servers before T017 send no `action_space` (= `osc_pose_delta`)
  and ignore the new optional `joint_pos` field: old clients and servers keep working.
- `ours` (only adapter that executes joint_velocity): batteries can pin `action_space` (droid batteries do;
  mismatch → fail before any episode); the server's contract (`action_space`, `control_freq`, `max_joint_delta`)
  must equal `joint_control.policy_contract()`.
- run_meta identity (all jobs): `policy=pi05`, `checkpoint`, `ckpt_sha`, `stats_sha256`, `action_space`,
  `control_freq`, `max_joint_delta`, `controller`, `joint_kp`, `joint_damping_ratio`, `camera`, `state`, `dtype`,
  `contract=openpi-droid`, exec fields. Episodes' `bench_version` ends in `/joint_velocity`, so sc-report marks
  every row that puts pi05droid next to pi05lib (OSC) as `mixed: bench_version` (tested).
- `gripper_cmd` in the step logs is the last action entry (7 here; 6 for OSC as before).

## CPU checks (this container, no GPU)

- `tests/` (15) and `scripts/shortcut/tests/test_joint_env.py` (5) pass; the existing suites pass unchanged
  (`scripts/shortcut/tests`: 108 passed incl. the joint-env tests; `serve/external`: StarVLA tests pass, 2 skipped as
  before — StarVLA-venv-only reference and a real second GPU), so the OSC path is unchanged.
- `cpu_check.py` (lerobot/pi05_droid fp32 on CPU, libero_goal task 1 "put the bowl on the stove", init 0):
  load 84 s, one chunk 3.6 s; prompt `Task: put the bowl on the stove, State: 121 78 138 34 132 117 170 6;\nAction: `;
  raw chunk 15 × 8 finite; the 8 served actions finite, in [−1, 1] (joint velocities |v| ≤ 0.19), gripper −1;
  executing them moves every commanded joint in the commanded direction.

## GPU smoke and main evaluation

```bash
scripts/shortcut/serve/external/droid/smoke.sh --out-root /outputs/VLA-shortcut-exps --gpu 0
# CPU plumbing check of the same script: smoke.sh --mock --out-root /tmp/x
```

36 episodes (4 suites × C0 × tasks 0–2 × 3 inits), one client (exec_mode single). Planner: 12,780 env steps
(zero-shot: full-length episodes) → ≈ 12 min + ≈ 1.5 min model load on an RTX 4090, ≈ 0.25 GPU-h (≤ 1 approved).
Output `<out-root>/smoke/T017/summary.txt`: PASS/FAIL (36 episodes, all executed actions finite, in range,
gripper ±1), C0 SR per suite × task, s/episode, ms per env step, video paths, and the main-battery estimate.

Main battery `droid_main` (P1: Spatial·Goal × C0/C1/C4/C5, 10 tasks × 10 inits; C4/C5 per feasible variant):
1,000 episodes, planner 298,200 env steps (all full length). **GPU-h ≈ env steps × (measured s per env step) /
3600**; with the planner's 4090 model (26 ms env step + 174 ms / 8 per step) ≈ 4.0 GPU-h, one client per GPU (the
server does not batch). Run only after the human decision on the smoke's C0 (C0 ≈ 0 → shortcut conditions are not
readable; the LIBERO fine-tune, ≈ 48 GPU-h, is a separate decision).

```bash
scripts/shortcut/bench/sc-bench --ckpt pi05droid --battery droid_main --gpus 0,1,2,3 --clients-per-gpu 1
```
