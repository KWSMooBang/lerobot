# LIBERO shortcut-observation harness (T001)

Design: "LIBERO Shortcut Learning 관측 실험 설계" §4–6 (summary in `RESEARCH.md`). The env, BDDL and init
states stay original. Only the instruction string given to π0.5 changes.

| file                    | what                                                                                                                                                                                |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_eval.py`           | one (checkpoint, suite, condition, repeat) run → `run_meta.json`, `episodes.jsonl`, `steps/**.npz`, `videos/**.mp4`                                                                 |
| `metrics.py`            | episodes.jsonl → `by_condition.csv`, `by_task.csv`, `by_variant.csv` (per-variant McNemar), `repeats.csv` (C0 repeat agreement), `summary.md`                                       |
| `make_compare_video.py` | same (task, init), several conditions side by side → mp4 + GIF ≤ 3 MB                                                                                                               |
| `submit.sh`             | one run per (ckpt, suite, cond, repeat). Default `--direct` runs sequentially on GPU 0; `--launch` (`/harness/launch.sh`, any GPU) is refused unless `--allow-any-gpu`; `--dry-run` |
| `submit_multi.sh`       | multi-GPU: all runs of a call share one queue over G GPUs × P workers × N batched envs (`parallel/README.md`); same run_ids/layout                                                  |
| `setup_server.sh`       | new server: venv, LIBERO assets, checkpoints, tokenizer, EGL check, per-GPU self-test (no sudo/apt)                                                                                 |
| `parallel/`             | coordinator, work queue, batched runner, env subprocesses, mock backend (`parallel/README.md`)                                                                                      |
| `common.py`             | LIBERO config, NVIDIA EGL, policy loading (+ LIBERO adapter for base ckpt), instruction variants                                                                                    |
| `bench/`, `serve/`      | testbed (T010): policy server + `sc-bench` batteries + `sc-report`, any checkpoint, benchmark adapters (`bench/README.md`)                                                          |
| `tests/fixtures/`       | 2-task libero_goal instruction set (tasks 1, 8) + swap goal BDDLs, used by the smoke test                                                                                           |

## Checkpoints (`--policy pi05 --checkpoint ...`)

- `pi05lib` = `lerobot/pi05_libero_finetuned`. Trained on `HuggingFaceVLA/libero` (all 4 suites mixed),
  6k steps from `lerobot/pi05_libero`.
- `pi05base` = `lerobot/pi05_base`. Not trained on LIBERO. Its config has `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb`
  and a 32-d state, with no normalization stats. `--libero-adapter auto` takes config, features and pre/post processors
  (LIBERO MEAN_STD stats) from `pi05lib` and only the weights from base. The details are in `run_meta.json → inference.libero_adapter`.
  Videos default to `all` for base.

Fixed inference: `n_action_steps=10`, chunk 50, 10 flow steps, 224 px policy input, 360 px env render,
relative control, no temporal ensembling, max steps = lerobot `TASK_SUITE_MAX_STEPS` (spatial/object 280,
goal 300, libero_10 520).

## Conditions and outputs

- C0 original, C1 `""`, C2 random whole-word vocab pieces with the same token count (seeded), C3 paraphrases,
  C4/C5 object/target swaps (`swaps[].type`, `feasible=false` skipped unless `--include-infeasible`),
  C6 other task's original sentence (`cross_task`). C3–C6 run every variant × every init.
- C0–C3 stop at the original goal. C4–C6 run to max steps and check the original goal (`_check_success`)
  and the new goal (`goal_bddl` `:goal`, via the scene's `_eval_predicate`) every step → `t_orig`, `t_new`.
- Seed per episode = `seed·1e6 + repeat·1e4 + task·100 + init`. It is the same in every condition, so episodes pair across conditions.
- `run_eval.py --num-envs N` (N > 1) runs the same run batched (`exec_mode batched-N`); see `parallel/README.md` before mixing with single-env results.
- Resume: an `--out` dir holds one configuration. `run_meta.json → identity` (checkpoint, adapter, suite,
  condition, repeat, seed, instruction-set sha256, n_action_steps, max_steps, exec_mode, GPU model, torch, CUDA) must match, and so must every
  existing episode's checkpoint/condition/seed; otherwise run_eval.py stops before loading the policy.
- metrics: episodes pair on (model, ckpt, suite, task, init, repeat, seed) with C0 repeat 0 as the reference.
  Condition-level McNemar aggregates C3–C6 variants per (task, init) first (majority; `any` also reported);
  per-variant McNemar is in `by_variant.csv`. Duplicate episodes across inputs are an error.
- `first_grasp_object` = first step where the gripper is commanded closed and both fingers touch the same
  object/fixture. `first_descent_xy` = eef xy where z has dropped 5 cm below its start and stops descending (or the gripper closes).
- `scene_k` = tasks in the suite sharing the scene (same objects, fixtures, regions, init): goal 10,
  spatial/object 1, libero_10 tasks 0–1 share a scene (k=2). chance = 1/k.
- Video file: `videos/<suite>/taskNN/<cond>_<variant>[_rK]_initNN.mp4` (agentview | wrist, 10 fps, 512×256).

## Environment notes

- Python 3.12 venv (`uv sync --locked --python 3.12 --extra libero --extra pi`). With 3.14, draccus fails to parse configs.
- This container lists only mesa in glvnd, so MuJoCo EGL fell back to llvmpipe (~100 ms/frame).
  `common.py` points `__EGL_VENDOR_LIBRARY_FILENAMES` at a user-level NVIDIA vendor JSON. Env steps go from 175 ms to 10 ms.
- Tokenizer: `google/paligemma-3b-pt-224` is gated. We fall back to `/cache/pid_demo_dl/paligemma_tok_hf` (`SC_TOKENIZER_DIR`), as `exp/pid-demo` does.
- LIBERO env fix (T009, on by default; `SC_LIBERO_RAW=1` turns it off and gives the env the P1 pilot used).
  The LIBERO files match the original repo, but MuJoCo 3.8 (lerobot's pin) renders two things unlike the MuJoCo
  (<= 3.3.2) that rendered `HuggingFaceVLA/libero`: it honours the PNG sRGB chunk (>= 3.3.3: dark brown floor in the
  Object suite, near-black salad dressing), and its box-box contacts throw milk / orange juice / ketchup off the
  table in LIBERO-10 t0, t1, t7 (>= 3.4). `ensure_libero_config()` builds a config with an asset mirror (sRGB chunk
  removed for object textures and the floor scenes' textures) and an init-state mirror (those objects lifted out of
  the table, `libero_init_patch.json` from `libero_init_patch.py`). Spatial and Goal renders are byte-identical to
  the unfixed env. Check: `analysis/env_fix_render.sh` + `analysis/env_fix_compare.py`.
  Each config dir carries `libero_env.json` (env tag + source key: hf-libero version, init_files content, assets
  revision); an inherited `LIBERO_CONFIG_PATH` of another mode or source is ignored. Runs record the env as
  `libero_env` = `raw` | `fix-<LIBERO_FIX>-<patch hash>` in `run_meta.json` (also in the identity when not raw) and
  at the end of the ours `bench_version`. Runs without the field (P1) are raw.
