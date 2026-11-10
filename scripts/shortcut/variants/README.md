# User-designed LIBERO variants (T020)

Write a LIBERO variant as YAML → generate its BDDL and init states → check that it can be done → look at it →
evaluate a checkpoint on it with `sc-bench`. Each episode also records which **training** goals were reached, so
a failure can be told apart: the policy did nothing, or it did a trained task instead (shortcut / fallback).
Everything here runs on the CPU. Policy evaluation runs on the GPU server.

## Make a variant in 3 steps

**1. Write the YAML** at `scripts/shortcut/variants/<suite>/<id>.yaml`. A directory of your own also works: pass
`--variants-dir DIR`, which holds `DIR/<suite>/<id>.yaml`.

```yaml
suite: libero_goal
base_task: 8 # the training task the scene, init states and seeds come from
id: my_cheese_plate # = file name; [A-Za-z0-9_.-]
instruction: put the cream cheese on the plate
goal: [On cream_cheese_1 plate_1] # conjunction of predicates
```

**2. Build, check and preview it**

```bash
scripts/shortcut/variants/sc-variants all --suite libero_goal --ids my_cheese_plate
#   build   → variants/<suite>/generated/<id>/{<id>.bddl, init_states.npy, manifest.json}
#   check   → manifest.feasibility (T002 scripted policy on init 0–2)
#   preview → /outputs/VLA-shortcut-exps/analysis/variants/<suite>/<id>.png + gallery.{html,md}
scripts/shortcut/variants/sc-variants list --suite libero_goal          # catalog: built? feasible? n inits
```

**3. Put it in a battery**

```yaml
jobs:
  - {
      benchmark: ours,
      suite: libero_goal,
      condition: custom,
      variants: [my_cheese_plate],
      episodes: 20,
    }
  # optional: variants_dir: /path/to/DIR, name: <job id suffix>; no `variants:` = every variant of the suite
```

```bash
scripts/shortcut/bench/sc-bench --ckpt pi05lib --battery my.yaml --dry-run    # then without --dry-run (GPU)
scripts/shortcut/bench/sc-report --runs bench-pi05lib-my                       # + "Custom variants: fallback"
```

You can also run it without the server: `run_eval.py --condition custom --variants my_cheese_plate [--variants-dir DIR]`.
This runs a single env. `--instr-set` defaults to `instructions/<suite>.json`.

## Spec keys

| key                   | meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `suite`               | `libero_spatial`, `libero_object`, `libero_goal` or `libero_10` (must match the directory)                                                                                                                                                                                                                                                                                                                                                              |
| `base_task`           | task id (0–9), task name or sentence. The variant uses its scene, init states and episode seeds, so its episodes pair with the base task's C0 at the same init index                                                                                                                                                                                                                                                                                    |
| `id`                  | unique in the suite, file-name safe, = file name                                                                                                                                                                                                                                                                                                                                                                                                        |
| `instruction`         | free sentence given to the policy                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `goal`                | list of `Pred arg [arg]`, combined with AND. Predicates: `On a b`, `In a region`, `Up a` (z ≥ 1.0 m), `Open x`, `Close x`, `TurnOn x`, `TurnOff x`. Names are full scene names: objects (`plate_1`), fixtures (`flat_stove_1`), regions as `<target>_<region>` (`flat_stove_1_cook_region`, `wooden_cabinet_1_top_region`, `main_table_stove_front_region`, `basket_1_contain_region`). `[]` = no achievable goal (probe variants such as Object C6/C8) |
| `track_goals`         | `default` = the goal of every training task in the suite (`t0`…`t9`). Or a list of task ids and/or `{name, goal: [...]}` / `{name, task: k}`. `none` = do not track. Goals that name an object missing from the variant scene are listed as absent and not tracked                                                                                                                                                                                      |
| `scene.move`          | `[{object, to: <region>}]` (a table/floor region from the BDDL) or `[{object, xy: [x, y]}]` (region frame = the BDDL ranges' frame). The object keeps its own height and orientation                                                                                                                                                                                                                                                                    |
| `scene.swap`          | `[[a, b]]`: a and b trade places (xy) in every init state, and their `:init` placements are exchanged                                                                                                                                                                                                                                                                                                                                                   |
| `scene.remove`        | `[object]`: the object, its regions (e.g. a basket's `contain_region`) and its `:init` entries are dropped                                                                                                                                                                                                                                                                                                                                              |
| `scene.add`           | `[{object: ketchup_1, category: ketchup, to: <region> \| xy: [x, y]}]`: the category must be used in the same suite                                                                                                                                                                                                                                                                                                                                     |
| `scene.fixture_state` | `{wooden_cabinet_1_top_region: open\|close, flat_stove_1: on\|off}`. The joint is set to the middle of LIBERO's default range for that state                                                                                                                                                                                                                                                                                                            |
| `init`                | `keep`: the base init states unchanged (only language and goal differ; not allowed with scene edits). `patch`: every base init state with the edits applied, then 20 no-op settle steps. `{resample: N}`: N new states from LIBERO's placement sampler (+ fixture states, settle)                                                                                                                                                                       |
| `group`, `notes`      | free labels (`group` shows in the gallery and the fallback table)                                                                                                                                                                                                                                                                                                                                                                                       |

YAML 1.1 reads bare `on`/`off` as booleans. They are accepted for `fixture_state` (true = on, false = off).

## What `build` checks

- **Names.** Every edited object, region and goal name exists in the (edited) scene. Added categories exist in the suite.
- **The `:init` can be sampled.** The eval env resets with LIBERO's placement sampler before it loads a state. A variant whose
  `:init` puts two objects on one spot fails here with a clear error, not at eval time.
- **Every stored state, loaded the way the eval loads it** (reset, `set_init_state`, 10 no-op steps):
  - moved, swapped and added objects are within 2.5 cm of their target xy and did not fall;
  - untouched objects are within 2.5 cm of where they rest in the base scene;
  - the fixture states hold;
  - for `resample`, the `:init` placements hold.

  States that fail are dropped and listed in `manifest.init_states.check`. `init_states.source[i]` is the base
  init index of stored state i.

- **T009 env rule.** `ensure_libero_config()` runs first, so base states come from the fixed init mirror (L10 lifts),
  previews use the fixed asset mirror, and the manifest records `libero_env`. `run_eval` refuses a variant that was built
  for another env tag, a YAML changed after its build, and a run dir whose episodes came from other generated files.

## Feasibility (`check`)

Uses T002's scripted policy (`feasibility.py`: pick-and-place, push, drawer open/close, turn on) plus `Up` (pick and
hold) and `TurnOff` (push the knob back), on init 0–2.

| status       | meaning                                                                                            |
| ------------ | -------------------------------------------------------------------------------------------------- |
| `feasible`   | the scripted policy reached the goal at least once                                                 |
| `infeasible` | no success with T002-validated primitives (a scripted-policy result, like T002's `feasible=false`) |
| `unknown`    | no success, and the plan needs Up / TurnOff (new primitives) → teleop                              |
| `n/a`        | `goal: []`                                                                                         |

The manifest also records goals that already hold at step 0 (`goal_at_init`, `track_at_init`).

**TurnOff.** LIBERO's `TurnOff` needs knob qpos < 0 inside a joint range of [−0.005, 2.1], so a knob turned back to
rest (0) never counts. Variant goals therefore evaluate `TurnOff x` as `not TurnOn x` (qpos < 0.5). The written BDDL
keeps `TurnOff`, and the manifest states this (`goal_semantics`).

## Evaluation records (`condition: custom`)

- One unit = one variant. `task_id` = base task, and the seed rule is unchanged.
- Episodes run to `max_steps`.
- `success_new` / `t_new`: the variant goal. `success_orig` / `t_orig`: the base task's goal (False if its objects
  were removed).
- New fields:
  - `goals_reached {name: first step | null}`, `goals_final {name: bool}`, `goals_initial [names true at reset]`;
  - in `steps/*.npz`: `goal_track` [T × goals] and `goal_track_names`.
- `variant_meta` holds the goals, the edits, `variant_key` (generated-file hashes) and feasibility.
- `perturbation = {dimension: language|layout|object, level: custom, detail: custom:<id>}`.
- C0–C6 records are unchanged. The mock run of C0/C4/C6 is byte-identical to the previous harness.

**sc-report** writes `fallback.csv` and adds a section to `report.md`. Per variant it reports:

- new-goal success;
- `trained`: a training goal was reached, the new goal was not;
- `nothing`: neither was reached;
- the rate for each tracked goal.

Goals that were true at reset, and a tracked goal identical to the new goal, are left out.

`--num-envs > 1` (batched in-process path) does not support `custom`. Use `sc-bench`, which supports
`--clients-per-gpu K`.

## Shipped examples (`make_examples.py` wrote them)

### goal15: human-written LIBERO-Goal variants (battery `bench/batteries/goal15.yaml`)

The 15 variants (g01–g05, g07–g11, g13–g17; g06 and g12 equal training tasks and were dropped, ids kept) use `init: keep`: the training scene and init states, with only the instruction and the goal changed.
Each variant comes in two wordings:

- `gNN_human`: the human sentence, in BDDL `:language` style;
- `gNN_train`: the same goal in the style of the training-data task strings (`instructions/vocab_libero_goal.json`; table set by the lead, BDDL "on the bowl" ↔ training "in the bowl").

Both wordings share `pair: gNN`, and the fallback table lists them next to each other. "and put … inside" goals
are judged with `In` only, as in training task 3. Variants marked † may be physically impossible. They stay in
the battery because fallback can still be measured; feasibility is recorded in each manifest.

| #    | human (`gNN_human`)                                                 | training style (`gNN_train`)                           | goal                                                 | base |
| ---- | ------------------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------- | ---- |
| g01  | Open the bottom layer of the drawer                                 | open the bottom drawer of the cabinet                  | Open wooden_cabinet_1_bottom_region                  | t0   |
| g02  | Open the top layer of the drawer                                    | open the top drawer of the cabinet                     | Open wooden_cabinet_1_top_region                     | t0   |
| g03† | Put the wine bottle on the bowl                                     | put the wine bottle in the bowl                        | On wine_bottle_1 akita_black_bowl_1                  | t2   |
| g04  | Put the wine bottle on the plate                                    | put the wine bottle on the plate                       | On wine_bottle_1 plate_1                             | t2   |
| g05  | Put the wine bottle on the stove                                    | put the wine bottle on the stove                       | On wine_bottle_1 flat_stove_1_cook_region            | t2   |
| g07† | Open the top layer of the drawer and put the wine bottle inside     | open the top drawer and put the wine bottle inside     | In wine_bottle_1 wooden_cabinet_1_top_region         | t3   |
| g08  | Open the top layer of the drawer and put the cream cheese inside    | open the top drawer and put the cream cheese inside    | In cream_cheese_1 wooden_cabinet_1_top_region        | t3   |
| g09  | Open the middle layer of the drawer and put the bowl inside         | open the middle drawer and put the bowl inside         | In akita_black_bowl_1 wooden_cabinet_1_middle_region | t3   |
| g10† | Open the middle layer of the drawer and put the wine bottle inside  | open the middle drawer and put the wine bottle inside  | In wine_bottle_1 wooden_cabinet_1_middle_region      | t3   |
| g11  | Open the middle layer of the drawer and put the cream cheese inside | open the middle drawer and put the cream cheese inside | In cream_cheese_1 wooden_cabinet_1_middle_region     | t3   |
| g13† | Put the bowl on the rack                                            | put the bowl on the rack                               | On akita_black_bowl_1 wine_rack_1_top_region         | t9   |
| g14  | Put the cream cheese on the plate                                   | put the cream cheese on the plate                      | On cream_cheese_1 plate_1                            | t8   |
| g15  | Put the cream cheese on the stove                                   | put the cream cheese on the stove                      | On cream_cheese_1 flat_stove_1_cook_region           | t1   |
| g16  | Put the cream cheese on the rack                                    | put the cream cheese on the rack                       | On cream_cheese_1 wine_rack_1_top_region             | t9   |
| g17  | Put the cream cheese on top of the drawer                           | put the cream cheese on top of the cabinet             | On cream_cheese_1 wooden_cabinet_1_top_side          | t4   |

Base task = the nearest training task. Its C0 episodes pair with the variant's episodes, and its goal is
`success_orig`. All 10 training goals are tracked whatever the base task is.

**Wording control** (`ctl_tNN_train`, `ctl_tNN_bddl`, pair `ctl_tNN`): each training task with its training sentence
(= C0, but run to `max_steps` with tracking) and with its BDDL `:language` sentence. For example, t0 has
"open the middle drawer of the cabinet" vs "Open the middle layer of the drawer", and t6 has "put the cream cheese in
the bowl" vs "Put the cream cheese on the bowl". The control measures the wording effect alone. It is the second job of `goal15.yaml`; delete that job to skip it.

### Other examples

- `libero_goal/n01–n23`: the new-instruction candidates N1–N4 of `tasks/refs/goal-novel-candidates.md`.
  - `#18 close the top drawer` starts with the top drawer open; `#19 turn off the stove` starts with the stove lit
    (`fixture_state` + `patch`).
  - The rest keep the training scene and init states.
- `libero_object/c6_tNN`: an instruction for a trained object that is not in the scene and was trained at the other
  target spot. No achievable goal; scene unchanged.
- `libero_object/c7_tNN`: the target and the object on the other target spot trade places. Original instruction and goal.
- `libero_object/c8_tNN`: the target removed. Original instruction; no achievable goal.
- Battery: `bench/batteries/variants_examples.yaml`.

## sc-bench catalog (T019)

`variants.list_variants(suite)` returns T019's `CatalogEntry` format: benchmark `ours`, key_field `condition`, key
`custom`, `details` = variant ids, `options.phrasing`. Once T019 is merged, `OursAdapter.catalog()` appends these
entries and `job_keys` gains `variants_dir`, `name` and `phrasing`; then `sc-bench --list` and the battery validation
see the user variants. Until then, custom jobs run without validation.

## Files

| file                    | what                                                                                                                                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spec.py`               | YAML schema, validation, paths; `list_variants(suite)` = bench/core CatalogEntry (one `condition: custom` entry per suite, `details` = ids), `variant_rows` = per-variant rows (`sc-variants list`); PyYAML only |
| `bddl_io.py`            | BDDL s-expression read/write + edits (round-trips all 130 LIBERO BDDLs through LIBERO's parser)                                                                                                                  |
| `generate.py`           | spec → BDDL + init states + manifest, state checks                                                                                                                                                               |
| `check.py`              | feasibility with the scripted policy                                                                                                                                                                             |
| `preview.py`            | CPU first-frame render, per-variant PNG, gallery                                                                                                                                                                 |
| `goals.py`              | predicate-list goal checker (+ mock)                                                                                                                                                                             |
| `runtime.py`            | eval side: plan, env, goals for `run_eval.py` / `bench/ours`                                                                                                                                                     |
| `fallback.py`           | fallback table for `sc-report`                                                                                                                                                                                   |
| `cli.py`, `sc-variants` | command line                                                                                                                                                                                                     |
