# Shortcut testbed: policy server + benchmark batteries (T010)

Shared design: `/outputs/VLA-shortcut-exps/tasks/TESTBED_DESIGN.md`. Any lerobot checkpoint (registry id, local
`pretrained_model` dir, or hub id) is served by one process per GPU; benchmarks connect as clients from
their own venvs; results land in the standard run-dir layout and are compared with `sc-report`.

## Evaluate a new model (3 lines)

```bash
scripts/shortcut/bench/sc-bench --ckpt tb-smolvla-a --battery quick --dry-run      # plan + time estimate
scripts/shortcut/bench/sc-bench --ckpt tb-smolvla-a --battery quick [--gpus 0,1,2,3] [--clients-per-gpu 8]
scripts/shortcut/bench/sc-report --runs tb-smolvla-a,pi05lib                        # tables, videos, manifest
```

- `--ckpt`: an id in `<out-root>/ckpt_registry.tsv` (sc-train / T011 adds rows; the file is created with
  `pi05lib` and `pi05base` if missing), a local checkpoint dir, or a hub id. `notes: adapter=<ckpt>` in the
  registry = LIBERO adapter (config + processors from that finetune, weights from the checkpoint; pi05base).
- `--out-root` (default `$SC_OUT_ROOT` or `/outputs/VLA-shortcut-exps`): `runs/<run-id>/`, `results.tsv`,
  `ckpt_registry.tsv`, `reports/`. Run id default `bench-<ckpt_id>-<battery>`. Same command again = resume.
- Same commands on the 4×A100 server and here (GPU 0: `--gpus 0`, the default from `CUDA_VISIBLE_DEVICES`).
  `/queue/locks/<uuid>.lock` is flock'ed per GPU when it exists (`--no-gpu-lock` to skip).
- venv: the main one plus `--extra shortcut-bench` (websockets, msgpack; the wrappers pass it).
  `setup_server.sh` already syncs this extra; elsewhere add `--extra shortcut-bench` to your `uv sync`.
- `--tasks 1,8 --episodes 2 --suites libero_goal --only C0,C4` shrink a battery (smoke runs; use a new `--run-id`).

### Exec modes (T004 rule: never pair across them)

| `--clients-per-gpu K` | server      | what it means                                                                                     |
| --------------------- | ----------- | ------------------------------------------------------------------------------------------------- |
| 1 (default)           | `single`    | batch-1 calls, episode's own RNG state restored per call: bit-identical to `run_eval.py`          |
| K > 1                 | `batched-K` | K clients' requests in one call padded to K rows, per-episode noise generators (T004 `batched-N`) |
| (stateful policy)     | `step`      | n_obs_steps > 1 / temporal ensembling / memory: `select_action` per env step, one live episode    |

Measured on GPU 0 (RTX 4090, pi05lib, libero_goal tasks 1/8 × init 0–3, C0 + C4: 24 episodes; outputs in
`/outputs/VLA-shortcut-exps/smoke/T010/`): server single vs `run_eval.py` → success_orig / success_new / t_orig /
t_new / first grasp / n_steps agree 24/24, actions bit-identical in 21/24. The other 3 (long C4 episodes)
first differ at step 130–290; `run_eval.py` against itself differs the same way (8/12 bit-identical, the same
episodes, first difference step 130–290), so this is run-to-run nondeterminism of the existing pipeline, not
the server. Throughput: single 23–26 env-steps/s (≈ in-process), batched-2 38 env-steps/s (221 ms/call).

Each job's `run_meta.json → identity` holds `exec_mode`, GPU model, torch, CUDA and `via: policy_server`;
`metrics.py` and `sc-report` warn when results of different exec modes / GPU models / bench versions are compared.

## Layout

| path                        | what                                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `../serve/policy_server.py` | websocket server: `--ckpt`, `--batch`, `--ready-file`; `engine.py` = loading, noise, queues, batching |
| `core/protocol.py`          | wire protocol `sc-policy/1` (hello / reset / infer / end, msgpack + numpy), obs convention            |
| `core/client.py`            | `PolicyClient` + `robosuite_obs_to_wire` / `lerobot_libero_obs_to_wire` (numpy, msgpack, websockets)  |
| `core/registry.py`          | `ckpt_registry.tsv` read / create / resolve                                                           |
| `core/records.py`           | standard episode record (design §2), perturbation dimensions, flock'ed jsonl appends                  |
| `core/battery.py`           | battery YAML expansion, dry-run time model                                                            |
| `core/adapters.py`          | benchmark adapter interface + registry (`ADAPTERS`)                                                   |
| `core/worker.py`            | generic worker loop (stdin units → `@@result` lines)                                                  |
| `core/runner.py`            | `sc-bench` coordinator: servers, worker slots, resume, bookkeeping, `results.tsv`                     |
| `core/report.py`            | `sc-report`                                                                                           |
| `ours/`                     | our C0–C6 harness as a client (T001 `EpisodeStepper`, dual goals, videos)                             |
| `batteries/`                | `quick.yaml`, `full.yaml`, `smoke.yaml`                                                               |

Run dir: `runs/<run-id>/{bench_meta.json, jobs/<job_id>/, servers/gpu<g>.log, workers/*.log, jobs_summary.json,
git_sha, git_dirty, patch.diff, cmd.txt, meta.txt, gpus, STATUS, started_at, ended_at}`. An `ours` job dir is a
T001 run dir (`run_meta.json`, `episodes.jsonl`, `steps/`, `videos/`), so `metrics.py runs/<run-id>/jobs/*`,
`make_compare_video.py` and the P1 analysis scripts work on it unchanged.

## Protocol (`core/protocol.py`, version `sc-policy/1`)

- `hello` → `{policy_type, ckpt, ckpt_id, exec_mode, exec{exec_mode,gpu,torch,cuda}, action_dim: 7, chunk_size,
n_action_steps, image_keys, state_dim: 8, image_orientation: "hf_libero_upright", image_hw, control_mode,
requires_step, protocol_version, inference{...}}`
- `reset {episode_id, episode_seed, instruction}` → new noise stream + empty action queue
- `infer {episode_id, obs{image, image2: uint8 HxWx3, state: float32[8]}, instruction[, step]}` →
  `{actions: float32[n_action_steps, 7], server_ms}`; `step: true` → one action from the server-side queue
- `end {episode_id}` (optional) frees the episode; a closed connection frees all of its episodes.
- Obs convention = training data (HuggingFaceVLA/libero): images upright (raw LIBERO render rotated 180°),
  `image` agentview, `image2` wrist; state = eef pos + axis-angle + gripper qpos. `robosuite_obs_to_wire` does
  this from a raw LIBERO `OffScreenRenderEnv` obs; its numpy axis-angle can differ from lerobot's torch one
  in the last float32 bit (the ours client uses lerobot's step itself). `image_hw` is the render size the
  policy was evaluated with (360 for LIBERO); other sizes are accepted (the policy resizes).

## Adding a benchmark adapter (T012 LIBERO-Plus, T013 LIBERO-PRO)

1. Separate venv in the worktree: `uv venv .venvs/<name> --python 3.x`, install the benchmark + `numpy msgpack
websockets`; report its size. Nothing from lerobot is needed on the client side.
2. `bench/<name>/adapter.py`: subclass `core.adapters.BenchmarkAdapter`, `name = "<benchmark>"`, and implement
   - `job_id(job)`: unique and file-name safe, e.g. `plus-goal-language`;
   - `plan(job, ctx)`: units (a few minutes of work each, e.g. one task × perturbation), with `unit_id`,
     `job_id`, `job_dir`, `benchmark`, `n_episodes`, `n_pending` (episodes not yet in `job_dir/episodes.jsonl`)
     and `est_steps` (env steps, for `--dry-run`); runs in the main venv — shell out to the benchmark venv
     if task lists need it;
   - `prepare(job, plan, ctx, hello)`: write `job_dir/run_meta.json` with an `identity` that includes
     `hello["exec"]` fields + `bench_version`, refuse an existing dir with another identity, and put the
     per-unit constants into `plan.info["unit_common"]`;
   - `worker_cmd(ctx)`: `[<venv python>, <path>/bench/<name>/worker.py]`.
3. `bench/<name>/worker.py`: `sys.path.insert(0, <repo>/scripts/shortcut)`, then
   `from bench.core.worker import run_worker` and `run_worker(run_unit)`; `run_unit(unit, client)` runs the
   unit's episodes (`client.reset` / `client.infer(eid, robosuite_obs_to_wire(obs), instruction)` / `client.end`)
   and appends one record per episode with `core.records.append_jsonl` — T001 fields where they apply +
   `core.records.standard_fields(...)` (`benchmark`, `bench_version` = package version or git sha, `suite`,
   `task_id`, `variant_id`, `perturbation{dimension, level, detail}` with `dimension="none"` for the clean
   baseline, `instruction_original`, `instruction_given`, `success`, `success_orig`/`success_new` if measurable,
   `video` relative to `job_dir`, `seed`, `init_idx`, `model`, `ckpt`). Skip episodes already in the file.
   Imports from `core` must stay numpy / msgpack / websockets only.
4. Register it in `core/adapters.py → ADAPTERS` and add `benchmark: <name>` jobs to `batteries/quick.yaml`
   (shortcut-relevant axes; check `--dry-run` keeps quick ≈ 1 h on 4×A100) and `full.yaml`.
5. Videos: 10 fps, agentview | wrist side by side (512×256 like ours), so `sc-report` can stack clips.
6. Tests: a mock-env unit test (see `tests/test_bench.py`) and a short real run with `--tasks/--episodes`
   under `/outputs/VLA-shortcut-exps/smoke/<task>/` (not runs/ or results.tsv).

## Tests

```bash
uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_bench.py -q    # CPU, mock env + policy
```
