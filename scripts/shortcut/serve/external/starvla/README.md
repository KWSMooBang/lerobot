# StarVLA/Qwen3-VL-PI-LIBERO-4in1 on the shortcut testbed (T016)

Public proxy for Qwen-RobotManip (weights not released): StarVLA `QwenPI` = Qwen3-VL-4B-Instruct + layer-wise
cross-DiT flow-matching head, trained on the IPEC-COMMUNITY LIBERO 4-in-1 mixture (100k steps). It runs behind
an sc-policy/1 server in **its own venv**; `sc-bench --ckpt starvla-qwen3pi-libero` launches it and runs the
`ours` battery (C0–C6) unchanged. Background: `/outputs/VLA-shortcut-exps/tasks/refs/T016-starvla-notes.md`.

| file                       | what                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| `setup_starvla.sh`         | venv (uv, Python 3.10, torch 2.6.0) + pinned StarVLA source + pinned checkpoint / base VLM |
| `requirements-starvla.txt` | inference subset of StarVLA's requirements                                                 |
| `server.py`                | sc-policy/1 server (engine for `serve/policy_server.Server`), StarVLA venv                 |
| `serve.sh`                 | launcher sc-bench uses for registry rows with `server=starvla`                             |
| `compare.py`               | action equality: our server vs StarVLA's own server + LIBERO client, same model and noise  |
| `dump_frames.py`           | LIBERO frames for `compare.py` (lerobot venv): real env render or HuggingFaceVLA/libero    |
| `smoke.sh`                 | GPU smoke: (a) `compare.py` on real renders, (b) sc-bench goal C0 × 2 tasks × 2 inits      |
| `tests/`                   | CPU unit tests (mock model)                                                                |

## Pins

| what                                         | revision                                                                                 |
| -------------------------------------------- | ---------------------------------------------------------------------------------------- |
| StarVLA `starVLA/starVLA`                    | `13d8ad43eb4128554c57227507055ecf9c3a62a2` (branch starVLA_dev, 2026-10-31)              |
| checkpoint `StarVLA/Qwen3-VL-PI-LIBERO-4in1` | `a461ce262ae7c8890d948e122c92fd06cd4803c7` (`checkpoints/steps_100000_pytorch_model.pt`) |
| base VLM `Qwen/Qwen3-VL-4B-Instruct`         | `ebb281ec70b05090aa6165b016eac8ec08e71b17`                                               |
| torch / transformers (StarVLA venv)          | 2.6.0 (cu124) / 4.57.0 (StarVLA's pins)                                                  |

The StarVLA sha has the card's compatibility override (`use_canonical_forward` alias, added in `631aae0`
"support legacy forward for released Qwen3-PI eval", kept through `7276c5e` and HEAD) and the current LIBERO
eval path (server-side un-normalization, `policy_wrapper.py`). The server loads with the card's overrides
`framework.qwenvl.base_vlm=<pinned local snapshot>` and
`framework.action_model.diffusion_model_cfg.use_canonical_forward=false`, `--use_bf16` as on the card.
`server.py` refuses other revisions (`--allow-unpinned` to override; then the identity records what it got).
flash-attn is not installed: StarVLA's `QWen3.py` forces `attn_implementation="sdpa"` at this sha.

## Contract: StarVLA LIBERO eval vs our wire

StarVLA side = `examples/simBenchmarks/LIBERO/eval_files/{eval_libero.py, model2libero_interface.py}` +
`deployment/model_server/{server_policy.py, policy_wrapper.py, policy_norm_processor.py}` at the pinned sha.

| item               | StarVLA LIBERO eval                                                                                                                                                                   | our wire / server                                                                                      | same?                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------- |
| cameras            | `[agentview, wrist]` (example_dict `image` list, that order)                                                                                                                          | `image` → `example.image[0]`, `image2` → `[1]`                                                         | yes                       |
| rotation           | `obs[...][::-1, ::-1]`                                                                                                                                                                | wire is upright = raw `[::-1, ::-1]` (HuggingFaceVLA convention); no further flip                      | yes (identical pixels)    |
| render size        | 256 × 256 (`LIBERO_ENV_RESOLUTION`)                                                                                                                                                   | hello `image_hw = [256, 256]` → ours env renders 256                                                   | yes                       |
| resize             | client: `PIL.Image.resize((224, 224), BILINEAR)` if not 224                                                                                                                           | server: same call                                                                                      | yes                       |
| state              | computed, **not sent** (example = `{image, lang}`)                                                                                                                                    | wire state (8) ignored                                                                                 | yes                       |
| instruction        | `task.language` verbatim; model wraps it in `CoT_prompt` (config.yaml)                                                                                                                | instruction as given (C0 = suite language), same wrapping in the model                                 | yes                       |
| call               | `predict_action(examples=[ex], unnorm_key=None, do_sample=False, use_ddim=True, num_ddim_steps=10)`                                                                                   | identical call on the same `PolicyServerWrapper`                                                       | yes                       |
| un-normalization   | server: `PolicyNormProcessor` = training `Libero4in1DataConfig.transform()`, stats `dataset_statistics.json["franka"]` (only key): min/max for x y z roll pitch yaw, gripper identity | same object                                                                                            | yes                       |
| gripper            | `1 - 2·(open > 0.5)` (`_binarize_gripper_open`; 1 = open → −1)                                                                                                                        | `to_env_actions`: same formula, float32                                                                | yes                       |
| horizon / executed | chunk 8 (`action_horizon`); all 8 executed (`step % 8 == 0` → new chunk)                                                                                                              | `n_action_steps = action_chunk_size = 8` (hello); client queue of 8                                    | yes                       |
| action ensembling  | `AdaptiveEnsembler` constructed but not applied in `step()`                                                                                                                           | none                                                                                                   | yes                       |
| flow steps         | 4 (`num_inference_timesteps`)                                                                                                                                                         | same model                                                                                             | yes                       |
| noise              | global `torch.randn` on the device, server seed unset                                                                                                                                 | `torch.manual_seed(episode_seed)` at reset, per-episode RNG state restored per call (exec_mode single) | deterministic per episode |

Card vs code, camera: the card says "One `image_0` RGB view" (it reads `datasets.vla_data.obs: [image_0]` in
config.yaml), but that YAML field is not what the dataloader uses. Training (`Libero4in1DataConfig.video_keys =
[video.primary_image, video.wrist_image]`, starVLA `f8ba178ae91669b53dfd1174e7984b6a4c2b081c` of 2025-12-09 — the run id is `1208_libero_all_QwenPI_qwen3` — and at the pinned sha; the
mixture dataset appends one PIL 224² image per video key) and the eval both feed **two** views, agentview then
wrist. The server therefore defaults to `--cameras agentview+wrist` and records `camera: agentview+wrist` in the
identity; `--cameras agentview` (identity `agentview-only`) exists for an ablation only. Training resized with
`Image.resize((224, 224))` (Pillow default BICUBIC) while the eval resizes BILINEAR; we follow the eval.

Output to the env: 7-D float32 `[dx dy dz droll dpitch dyaw gripper]`, gripper ∈ {−1 open, +1 close}, relative
(delta) OSC control — what `LiberoEnv.step` in the `ours` harness takes.

### Harness vs StarVLA eval (not changed; recorded)

| item             | StarVLA eval                                  | `ours` harness (T009 fix env)                                 | effect on C0                                  |
| ---------------- | --------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| max policy steps | spatial **220**, object 280, goal 300, 10 520 | lerobot `TASK_SUITE_MAX_STEPS`: spatial **280**, others equal | spatial gets 60 more steps: can only raise SR |
| settle steps     | 10 × `[0]*6 + [-1]`                           | 10 × lerobot dummy action (same)                              | none                                          |
| init states      | `task_suite.get_task_init_states`, init i     | same files, init i                                            | none                                          |
| env seed         | `env.seed(7)` once per task                   | `env.reset(seed=episode_seed)` per episode                    | object jitter differs; SR within noise        |
| simulator        | upstream LIBERO                               | hf-libero + T009 init patch (common.py)                       | T009 fix applies to all models alike          |
| success          | `done` from `env.step`                        | `check_success()` every step, early stop for C0–C3            | none                                          |

None of these lowers C0 relative to StarVLA's protocol, so the lead was not asked before the gate.

## CPU check (done here, no GPU)

`compare.py --device cpu --no-bf16 --fp32` with the real pinned checkpoint (CPU: StarVLA's bf16 path relies on
CUDA autocast to return float32, so the shared model is cast to fp32 — the same object serves both paths), two
HuggingFaceVLA/libero frame sets (episode 0: libero_10 "put the white mug on the left plate …"; episode 900:
"pick up the cream cheese and place it in the basket") rotated back to raw renders, 2 chunks each:
**PASS, 32/32 actions bit-identical (max |Δ| = 0)** between our server (sc-policy/1, wire obs) and StarVLA's own
WebsocketPolicyServer + LIBERO ModelClient with the same seed. Agentview-only input instead moves xyz/rpy by up
to 0.43 and flips 13/32 gripper commands — the camera contract matters. Load 23 s, ~37 s per 2-chunk comparison
(224 CPU cores). Output: `/outputs/VLA-shortcut-exps/smoke/T016/cpu/compare.{json,log}`. The GPU smoke repeats this
on real LIBERO renders from the `ours` env, in bf16 as on the card.

## Commands (GPU server)

```bash
# install (≈ 35 GB: venv 9 + checkpoint 16.4 + Qwen3-VL-4B 8.9), once; lerobot side = setup_server.sh
scripts/shortcut/serve/external/starvla/setup_starvla.sh --root $SC_OUT_ROOT/starvla

# smoke: (a) action equality on real renders (GPU), (b) goal C0 × tasks 1,8 × 2 inits; PASS/FAIL + s/episode
scripts/shortcut/serve/external/starvla/smoke.sh --out-root $SC_OUT_ROOT --gpu 0

# C0 gate: 4 suites × 10 tasks × 5 inits (200 ep); exec_mode single only (one client per GPU)
scripts/shortcut/bench/sc-bench --ckpt starvla-qwen3pi-libero --battery starvla_gate --gpus 0,1,2 --dry-run
scripts/shortcut/bench/sc-bench --ckpt starvla-qwen3pi-libero --battery starvla_gate --gpus 0,1,2
scripts/shortcut/bench/sc-report --runs bench-starvla-qwen3pi-libero-starvla_gate
```

Gate: per-suite C0 ≥ card − 6 pp (goal 90.2 / object 92.6 / spatial 93.2 / libero_10 90.0) **and** mean ≥ 92 %.
Fail → contract mismatch: report, do not run the main battery.

```bash
# main (P1 convention, pairs with pi05lib): spatial·goal × {C0,C1,C4,C5}, object·10 × {C0,C1,C4}, 10 tasks × 10 inits
scripts/shortcut/bench/sc-bench --ckpt starvla-qwen3pi-libero --battery starvla_main --gpus 0,1,2
scripts/shortcut/bench/sc-bench --ckpt pi05lib --battery starvla_main --gpus 0,1,2 --run-id bench-pi05lib-starvla_main  # pair, if not already in P1 runs
scripts/shortcut/bench/sc-report --runs bench-starvla-qwen3pi-libero-starvla_main,bench-pi05lib-starvla_main
```

Time (dry-run, exec_mode single, env step 26 ms, A100 batch-1 call assumed 150 ms — replace with smoke's
measured s/episode; `--est-infer1-ms` sets it): gate 33k env steps ≈ 0.45 GPU-h; main 1,770 episodes, 497k env
steps ≈ 6.2 GPU-h (5.4–7.9 for 100–250 ms calls). With `--gpus 0,1,2` the wall time is a third of that.
`--clients-per-gpu > 1` is refused by the server (batching would need per-row noise inside StarVLA).

## Tests

```bash
uv run --extra shortcut-bench pytest scripts/shortcut/serve/external/starvla/tests -q              # lerobot venv
T=$PWD/scripts/shortcut/serve/external/starvla/tests   # StarVLA venv: + StarVLA's own server/client path (mock)
(cd $STARVLA_SRC && PYTHONPATH=$STARVLA_SRC $STARVLA_PYTHON -m pytest -p no:cacheprovider --noconftest -c /dev/null --rootdir $T $T -q)
```
