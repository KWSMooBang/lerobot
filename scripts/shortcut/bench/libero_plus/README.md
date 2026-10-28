# LIBERO-Plus on the shortcut testbed (T012)

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) (pinned `4976dc3`) perturbs the 40 original LIBERO tasks along
7 axes, ~10k registered variants. It installs as the `libero` package (replaces vanilla LIBERO), so it runs in its own
venv as a policy-server client; the model stays in the main venv (`sc-bench` starts both).

```bash
scripts/shortcut/bench/libero_plus/setup_libero_plus.sh        # once per machine (venv, fork, 6.4 GB assets, self-test)
scripts/shortcut/bench/sc-bench --ckpt pi05lib --battery quick --only libero_plus --dry-run
scripts/shortcut/bench/sc-bench --ckpt pi05lib --battery quick  # ours + Plus (+ PRO) together
```

## Setup (`setup_libero_plus.sh`)

| piece                                                           | where                                                       | size                |
| --------------------------------------------------------------- | ----------------------------------------------------------- | ------------------- |
| venv (Python 3.12, CPU torch, mujoco 3.8.1, robosuite 1.4.0, …) | `<repo>/.venvs/libero_plus` (`--venv`, `$LIBERO_PLUS_VENV`) | 2.2 GB              |
| fork source                                                     | `<venv>/src/LIBERO-plus`                                    | 155 MB              |
| assets (`Sylvest/LIBERO-plus` `assets.zip`, 6.0 GiB download)   | `<venv>/src/LIBERO-plus/libero/libero/assets`               | 9.5 GB              |
| LIBERO config                                                   | `<venv>/libero_config/config.yaml` (`LIBERO_CONFIG_PATH`)   | ~/.libero untouched |

- Simulator versions = the main venv's (hf-libero), so physics and rendering match our C0 runs; lerobot's Dockerfile pins
  robosuite 1.4.1 (same `single_arm_env` API).
- `--torch-backend cpu` overrides a global `UV_TORCH_BACKEND=cu1xx` (CUDA torch would add ~5 GB; torch only loads init states).
- **System package**: `env_wrapper.py` imports Wand (ImageMagick) at module level; only Sensor Noise 1–10 (motion
  blur) calls it. Without `libMagickWand` (apt `libmagickwand-6.q16-6`) the worker stubs Wand and the adapter leaves the
  motion-blur variants out of the selection (`run_meta.identity.motion_blur = false`, so a later run with ImageMagick
  is a different configuration). Re-run the setup script after installing it.

## Records

`jobs/plus-<suite>-<axis>/episodes.jsonl`, standard fields (`bench/core/records.py`) plus T001-style outcome fields:

| field                                       | value                                                                                                                                                                                                                                    |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `perturbation`                              | `{dimension: language/layout/camera/robot/lighting/texture/noise/none, level: difficulty 1–5 (null for some Goal lights), detail: sub-type}`                                                                                             |
| `detail`                                    | language `rewrite`; layout `add` (confounding objects) / `moved` (positions); camera `view`; robot `initstate`; light `light`; texture `table`/`tb`; noise `motion_blur`/`gaussian_blur`/`zoom_blur`/`fog`/`glass_blur`; none `original` |
| `task_id`, `variant_id`                     | original task (vanilla order, = ours) and `p<registry index>` (`orig` for none); `plus_name`, `plus_index`, `plus_params`                                                                                                                |
| `instruction_original`, `instruction_given` | original LIBERO instruction; what the policy got                                                                                                                                                                                         |
| `plus_instruction_registry`                 | the fork's own `task.language`                                                                                                                                                                                                           |
| `success`                                   | LIBERO-Plus' check (`env.check_success()`, the variant's BDDL goal), early stop at success                                                                                                                                               |
| `success_orig`                              | none / language / layout: = `success`. Every registered variant keeps the original goal predicates (checked against the BDDL files for all 10,030), so this is the original goal judged in the perturbed scene. Other axes: null         |
| `seed`                                      | ours' `episode_seed(seed, 0, task, init)`: `none` pairs with ours C0 episode by episode                                                                                                                                                  |

**Instruction mode.** For every non-language variant the fork's `task.language` is the instruction plus the
file-name suffix (`"put the bowl on the plate view 0 0 100 0 0 initstate 19"`, `"... table 1"`, `"... add 3"`), and that
is what upstream evaluation and lerobot's `libero_plus` env feed the policy. Default `instruction: clean` gives the
original instruction instead, so each axis is measured alone; `instruction: registry` in a job reproduces upstream.

## Batteries

| battery | Plus jobs                                                                                            | episodes | est. env steps |
| ------- | ---------------------------------------------------------------------------------------------------- | -------- | -------------- |
| quick   | 4 suites × {none 10 tasks × 5 inits; language, layout 10 tasks × 10 variants (2 per level) × init 0} | 1,000    | 215 k          |
| full    | 4 suites × {none × 10 inits; 7 axes × 20 variants per task × init 0}                                 | 5,596    | 1.41 M         |

Job keys: `adapter.py` docstring (`tasks`, `variants` N|all, `levels`, `details`, `episodes`, `instruction`, `video`, `tag`).

## Example gallery

`gallery.py` (Plus venv) renders the first frame of the original and of each axis at its lowest / highest difficulty for
fixed example tasks → `analysis/examples/libero_plus/` + `manifest.json` (id, title, caption, path, instructions, cue).

## Tests

```bash
uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_libero_plus.py -q   # CPU, fake fork + mock env
.venvs/libero_plus/bin/python scripts/shortcut/bench/libero_plus/lp_env.py selftest # real env, one variant per axis
```
