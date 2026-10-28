# LIBERO-PRO adapter (T013)

[LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) (code MIT, dataset CC-BY-4.0) through the sc-bench policy
server: its own venv, five perturbation axes, standard episode records. Shared design:
`/outputs/VLA-shortcut-exps/tasks/TESTBED_DESIGN.md`; adapter interface: `../README.md`.

```bash
scripts/shortcut/bench/libero_pro/setup_venv.sh                       # once per checkout (~10 min, no GPU, no sudo)
scripts/shortcut/bench/sc-bench --ckpt pi05lib --battery quick --only libero_pro --dry-run
scripts/shortcut/bench/sc-bench --ckpt pi05lib --battery quick        # ours + LIBERO-PRO (+ Plus) jobs
```

## Install (`setup_venv.sh`)

| path (repo root)        | what                                                                                                                                                                   | size   |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `.venvs/libero_pro`     | Python 3.12, robosuite 1.4.0 + mujoco 3.8.1 (= main venv's hf-libero stack), numpy, msgpack, websockets, cv2, imageio; `libero_config/config.yaml` (never `~/.libero`) | 1.5 GB |
| `.venvs/libero_pro_src` | LIBERO-PRO @ `eafdb809` (its own `libero` package + assets incl. the extra objects) + the dataset's bddl / init files + generated environment axis; `SC_VERSION`       | 0.9 GB |
| `.venvs/libero_pro_hf`  | `zhouxueyang/LIBERO-Pro` @ `c86fc3b8`                                                                                                                                  | 15 MB  |

`bench_version` = `libero-pro-<sha>/hf-<rev>/env-<hash of generated env-axis files>`. No system packages are
needed beyond what the main harness uses (EGL). `check_install.py` lists every (suite, axis, level) with its
task and init counts and checks that the task axis' original goals are evaluable.

## Axes → standard records

| axis (battery `perturbations`) | PRO files                                                           | levels                                   | `perturbation.dimension` | tasks                    |
| ------------------------------ | ------------------------------------------------------------------- | ---------------------------------------- | ------------------------ | ------------------------ |
| `none` (baseline)              | `<suite>`                                                           | `original`                               | none                     | 4×10                     |
| `object`                       | `<suite>_object` (dataset)                                          | `object`                                 | object                   | 4×10                     |
| `position`                     | `<suite>_swap` (dataset); `libero_object_temp_{x,y}0.{1..5}` (repo) | `swap`; libero_object also `x0.1`…`y0.5` | position                 | 4×10 (+10 per x/y level) |
| `semantic`                     | `<suite>_lan` (dataset)                                             | `paraphrase`                             | language                 | 4×10                     |
| `task`                         | `<suite>_task` (dataset)                                            | `task`                                   | task                     | 4×10                     |
| `environment`                  | `<suite>_env` (generated, `gen_env_axis.py`)                        | `table`                                  | environment              | 4×10                     |

`perturbation.detail` = `pro:<axis>`, `variant_id` = level. 50 init states per (axis, level, task), except
`libero_object_temp_y0.5` cream cheese (task 1): PRO ships an empty file, the planner skips it.

- **success** = the PRO BDDL's own goal (`env.check_success()`), PRO's definition; early stop on it.
- **task axis**: PRO replaces `(:goal …)` and `(:language …)` in the same scene and judges only the new goal. The
  worker also evaluates the _original_ suite BDDL's goal every step and runs to max_steps (T001 dual-goal rule):
  `success_new` = PRO goal, `success_orig` = original goal → `sc-report` Shortcut Rate as for ours C4–C6.
  All 40 task-axis tasks keep the original goal's objects and regions (checked with the live env).
  Other axes: the goal is the original one (renamed objects on the object axis), `success_orig = success`.
- **instructions** (`instruction_source`): PRO's BDDL `:language` differs from the LIBERO training sentence even
  on the clean suite in 25 / 40 tasks ("Pick the akita black bowl …" vs "pick up the black bowl …"; goal "layer of
  the drawer"; table: `instructions.csv` from `instruction_table.py`). Default `training`: the training sentence
  (= ours C0) on none / object / position / environment, PRO's sentence on semantic / task (there the sentence is
  the perturbation), so visual axes carry no language shift. `pro_bddl` = the axis BDDL's `:language` everywhere;
  `filename` = PRO's own `Benchmark.get_task(i).language`, derived from the file name, which PRO keeps unchanged
  (= the training sentence on every axis, also semantic / task: a harness using it, like PRO's OpenVLA example,
  never shows the policy the changed sentence). Records: `instruction_original` = training sentence,
  `pro_instruction_original` = PRO's clean-suite sentence, `instruction_given`, `instruction_bddl`.
- **init states** (`init_source`): PRO ships its own re-sampled init file for every axis, so at the same init index
  objects sit 1–8 cm away from the clean suite's placement even on semantic / task (`compare_inits.py`,
  `init_compare_*.csv`). Default: task and semantic start from the **clean** suite's init file (same sim-state layout,
  the only change is the sentence / goal, paired with ours C0 at the same init); the other axes use PRO's file
  (position: the move is the perturbation; object / environment: different object sizes / table height).
  `init_source: pro` / `clean` overrides it (job id gets `-initpro` / `-initclean` when it differs from the default).
- **environment axis**: PRO's `EnvironmentReplacePerturbator` rewrite (table name, problem class, fixture type)
  with target `living_room_table` for main-table scenes (spatial, goal: PRO's hard-coded choice) and
  `main_table` otherwise (PRO hard-codes living_room_table, a no-op for those scenes; its README recommends
  main_table because other tables can let objects drift; the gallery measures drift ≤ 0.1 cm over 50 no-op steps).
  Init states: 50 fresh placements per task as PRO's `generate_init_states.py`, without a renderer, seeded; a
  `.meta.json` sidecar per file (seed, number of inits, PRO commit, script version) — a rerun refuses to fill gaps
  next to files made with other settings (`--force` regenerates all).
- Episode = lerobot `LiberoEnv` + T001: `env.seed(seed)` → reset → `set_init_state` → 10 no-op steps, delta
  control, 360 px, max steps 280 / 280 / 300 / 520 (lerobot, = ours; PRO's OpenVLA script uses 220 for spatial),
  seed = `episode_seed(seed, repeat, task, init)` as in ours (pairs with ours C0 at the same task / init).
- Observation convention: `robosuite_obs_to_wire`; for the clean suite the wire obs (both images, state) is
  bit-identical to the main harness' lerobot `LiberoEnv` path at the same init (`compare_obs.py`).

## Batteries

- `quick`: none (5 inits), task (10), position swap (10) × 4 suites, libero_object x/y 0.1 / 0.3 / 0.5 (5) =
  1,245 episodes, 371k env steps ≈ 16 min on 4×A100 (K=8), 4.5 h on GPU 0 single.
- `full`: none + all five axes × 4 suites × 50 inits (all levels) = 16,950 episodes ≈ 3 h on 4×A100 (K=8).

## Files

| file                   | venv | what                                                                                         |
| ---------------------- | ---- | -------------------------------------------------------------------------------------------- |
| `catalog.py`           | both | axes, levels, dirs, task entries (stdlib; numpy for init files)                              |
| `adapter.py`           | main | `LiberoProAdapter`: jobs, units = (level, task), run_meta identity, worker command           |
| `worker.py`            | PRO  | units through the policy server; records, steps npz, 512×256 videos                          |
| `sim.py`               | PRO  | `ProEnv` (reset / step / original goal / contact probe), frames, `MockProEnv` (tests)        |
| `gen_env_axis.py`      | PRO  | environment-axis bddl + init files                                                           |
| `check_install.py`     | PRO  | install check, task-axis goal table (`--json`)                                               |
| `gallery.py`           | PRO  | example gallery (Spatial t0, Goal t8 × axes; Object t0 x/y extremes) + manifest.json         |
| `compare_obs.py`       | both | observation-convention check against the main harness                                        |
| `compare_inits.py`     | PRO  | per axis: placement vs the clean init at the same index, clean-init feasibility, goal at t=0 |
| `instruction_table.py` | any  | PRO `:language` vs training sentence for 40 tasks (csv)                                      |

Tests: `uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_libero_pro.py -q` (CPU, fake PRO tree,
mock env + mock policy server).
