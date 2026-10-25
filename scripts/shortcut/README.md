# LIBERO shortcut-observation harness (T001)

Design: "LIBERO Shortcut Learning 관측 실험 설계" §4–6 (summary in `RESEARCH.md`). The env, BDDL and init
states stay original. Only the instruction string given to π0.5 changes.

| file                    | what                                                                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `run_eval.py`           | one (checkpoint, suite, condition, repeat) run → `run_meta.json`, `episodes.jsonl`, `steps/**.npz`, `videos/**.mp4`                       |
| `metrics.py`            | episodes.jsonl → `by_condition.csv`, `by_task.csv`, `summary.md`                                                                          |
| `make_compare_video.py` | same (task, init), several conditions side by side → mp4 + GIF ≤ 3 MB                                                                     |
| `submit.sh`             | one run per (ckpt, suite, cond, repeat). Default `--direct` runs sequentially on GPU 0; `--launch` uses `/harness/launch.sh`; `--dry-run` |
| `common.py`             | LIBERO config, NVIDIA EGL, policy loading (+ LIBERO adapter for base ckpt), instruction variants                                          |
| `tests/fixtures/`       | 2-task libero_goal instruction set (tasks 1, 8) + swap goal BDDLs, used by the smoke test                                                 |

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
