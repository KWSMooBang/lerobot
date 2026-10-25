# Multi-GPU shortcut eval (T004)

Same episodes as `run_eval.py` (T001), spread over G GPUs × P workers per GPU × N envs per worker.
Run dirs look like T001 run dirs: `run_meta.json`, `episodes.jsonl`, `steps/`, `videos/` (+ the
`submit.sh` bookkeeping files `git_sha`, `cmd.txt`, `meta.txt`, `STATUS`, `ended_at`, `stdout.log`).

```
coordinator (run_multi.py)            plans every run like run_eval.py, writes the queue, starts workers,
│                                     re-queues the claims of a dead worker, merges finished runs
├── worker w0-0  (GPU 0)              CUDA_VISIBLE_DEVICES=0, MUJOCO_EGL_DEVICE_ID=0, one policy in bf16
│   ├── env subprocess × N            LIBERO env + EpisodeStepper (same code as the single path)
│   └── writer threads                npz + mp4 encoding + part-file line, off the GPU loop
├── worker w0-1  (GPU 0) ...
└── worker w3-1  (GPU 3)
```

| file              | what                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| `run_multi.py`    | CLI, planning, coordinator, worker entry (`--worker`), merge                                     |
| `workqueue.py`    | flock'ed shared queue (`units.jsonl` + append-only `log`), work stealing, release of dead claims |
| `batched.py`      | continuous batching over N env slots, chunk-aligned starts, batch padding                        |
| `envproc.py`      | env subprocess server (start / step / result)                                                    |
| `bundle.py`       | policy + processors, batched `predict_action_chunk`, per-episode noise generators                |
| `resources.py`    | `--procs-per-gpu auto` / `--num-envs auto`, dry-run throughput model                             |
| `mock_backend.py` | CPU fake env + policy (`--backend mock`) for `tests/test_parallel.py`                            |
| `selftest.py`     | per-GPU check used by `../setup_server.sh`                                                       |

## 4 × A100 80 GB

```bash
scripts/shortcut/setup_server.sh --gpus 0,1,2,3          # venv, LIBERO assets, ckpts, EGL, self-test
scripts/shortcut/submit_multi.sh --dry-run --ckpt pi05lib,pi05base --suites spat,obj,goal,l10 \
    --conds C0,C1,C2,C3,C4,C5,C6 --repeats 0,1,2 --episodes 50 --gpus 0,1,2,3
tmux new -s p2 'scripts/shortcut/submit_multi.sh --ckpt pi05lib,pi05base --suites spat,obj,goal,l10 \
    --conds C0,C1,C2,C3,C4,C5,C6 --repeats 0,1,2 --episodes 50 --gpus 0,1,2,3 \
    --out-root ./outputs/VLA-shortcut-exps'
uv run python scripts/shortcut/metrics.py ./outputs/VLA-shortcut-exps/runs/sc-* --out results/shortcut_p2
```

- One call = one queue. All runs of the call share every GPU; a worker keeps one checkpoint loaded,
  prefers units of its env's current task (env reuse), and loads the other checkpoint only when its own has
  nothing pending (work stealing). Long C4–C6 episodes go first within a task.
- Interrupt (Ctrl-C / kill) and rerun the same command: finished episodes (in `episodes.jsonl` or
  `parts/*.jsonl`) are skipped, the rest is planned again. A worker that dies mid-session has its claims
  put back and is restarted (a unit that kills a worker 3 times is marked failed).
- `--num-envs auto` = min(CPU cores per worker − 1, GPU memory per worker / 0.7 GB, 16);
  `--procs-per-gpu auto` = 2 on ≥ 40 GB GPUs, else 1. The choice and its arithmetic go to stdout,
  `run_meta.json → exec.auto` and the dry-run. Set them by hand once measured (see below).
- `--compile` turns on `torch.compile` (off by default until results are checked against eager).
- Render size, policy input, `n_action_steps`, video (10 fps, 512×256, `--video` rule) are T001's.
- Without `/harness` or `/queue`: nothing changes except no GPU locks. With `/queue/locks` (lab
  servers) the coordinator holds `flock -n` on each GPU's lock for the session (`--no-gpu-lock` to skip).
- EGL device ids are taken as nvidia-smi (PCI) ids. If a server enumerates EGL devices in another
  order, the self-test still passes but envs render on another GPU; check with `nvidia-smi` (type `G`
  processes) during the self-test.

## Do not mix with single-path results

bf16 matmuls at batch N are not bit-identical to batch 1, so a batched episode is a different (equally
valid) sample from the same seed. Therefore:

- `run_meta.json → identity` holds `exec_mode` (`single` | `batched-N`[`+compile`]), the GPU model, torch
  and CUDA versions. A run dir refuses to resume under another exec mode or GPU model (old T001 dirs
  count as `single` on their recorded GPU).
- `metrics.py` pairs (McNemar, DTW, Δ vs C0, C0 repeats) only within one (exec_mode, GPU model) and warns
  when inputs mix; `--allow-mixed-exec` pairs across. SR rows still pool every input, so keep C0 and the
  conditions it is compared with in the same exec mode: run a whole comparison set with one command.
- `--num-envs 1` gives `exec_mode=single`: batch-1 calls with the same noise stream as `run_eval.py`.
- Within a batched run the result of an episode does not depend on its batch mates: noise comes from a
  per-episode `torch.Generator` seeded with the episode seed (same stream as `torch.manual_seed` in the
  single path; pi05 `predict_action_chunk(noise=...)`, no policy change), and every call is padded to N
  rows so kernel shapes never change. Env seeds are unchanged (`seed·1e6 + repeat·1e4 + task·100 + init`).

## Throughput and P2 estimate

P2 as planned by `--dry-run` with the T002 instruction sets ({pi05lib, pi05base} × 4 suites × C0–C6,
50 init, C0 × 3 repeats, infeasible swaps skipped): **46,100 episodes, ≈ 13.0 M env steps** (C4–C6 and
pi05base run to max steps; pi05lib C0–C3 assumed to stop at 0.45 × max steps). At the T001 single-path
rate (45 ms/step, RTX 4090) that is ≈ 162 GPU-h.

Dry-run model (`resources.ThroughputModel`, flags `--est-*`): one worker alternates a lockstep env tick
(`env_ms·(1 + 0.1 ln N)`) and its share of a chunk inference (`infer_ms(N)/n_action_steps`,
`infer_ms(N) = infer1·(1 + β(N−1))`); P workers per GPU overlap, capped by the GPU being busy with inference
all the time. Defaults from the 4090 smoke: env 10 ms, infer1 300 ms; β (cost of each extra batch row)
is not measured yet:

| β                      | P=1, N=8 | P=2, N=8 | P=2, N=16 |
| ---------------------- | -------- | -------- | --------- |
| 0.05                   | 6.2 h    | 4.8 h    | 3.1 h     |
| 0.1 (default)          | 7.4 h    | 6.0 h    | 4.4 h     |
| 0.3                    | 12.3 h   | 10.9 h   | 9.6 h     |
| 1.0 (no batching gain) | 29 h     | 28 h     | 28 h      |

(wall-clock on 4 GPUs, incl. one model load per worker per checkpoint). Workers log measured
`ms/call` and `ms/env-step` (`<out-root>/multi/<session>/w*.log`); put them into `--est-infer1-ms`,
`--est-beta`, `--est-env-ms` for the next dry-run.

## Tests

```bash
uv run pytest scripts/shortcut/tests -q     # CPU only; fake GPU ids, mock env + policy
```

`test_parallel.py` runs the single-env loop and the parallel path on the mock backend and requires
identical episode records (except timing), identical step logs, the same files, (task, init, variant)
order, resume after `SIGKILL` of the whole process group, recovery from a worker crash, and work
stealing between two checkpoints.
