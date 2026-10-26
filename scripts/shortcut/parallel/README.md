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
pi05base run to max steps; pi05lib C0–C3 assumed to stop at 0.45 × max steps). At the measured
single-path rate (48 ms/step, RTX 4090, T006) that is ≈ 173 GPU-h.

Dry-run model (`resources.ThroughputModel`, flags `--est-*`): one worker alternates a lockstep env tick
(`env_ms·(1 + straggler·ln N)`) and its share of a chunk inference (`infer_ms(N)/n_action_steps`,
`infer_ms(N) = infer1·(1 + β(N−1))`); P workers per GPU overlap, capped by the GPU being busy with inference
all the time.

**Measured on one RTX 4090 (T006, pi05lib bf16, eager; `/outputs/VLA-shortcut-exps/smoke/T006-gpucheck/`):**

| quantity                                  | value                                                            |
| ----------------------------------------- | ---------------------------------------------------------------- |
| chunk inference, batch N (incl. pre/post) | N=1 174 ms · 2 216 · 4 302 · 8 463 · 16 903 ms                   |
| β (least squares over N=2–16)             | **0.27** (per N: 0.24, 0.25, 0.24, 0.28); peak mem 9.5 → 11.9 GB |
| env step (360 px EGL render, in-process)  | 13 ms                                                            |
| lockstep tick, N=4 env subprocesses       | 26 ms (6.5 ms/env-step) → straggler ≈ 0.72 at N=4 (IPC + render) |
| single path (`run_eval.py`)               | 20.8 env-steps/s (48 ms/step: policy 22, env 14, probe/video 12) |
| batched N=4, P=1 (`--num-envs 4`)         | ≈ 70 env-steps/s (model with the values above: 69) → **3.4×**    |

P2 on 4 GPUs with these values (`--est-env-ms 13 --est-infer1-ms 174 --est-beta 0.25 --est-straggler 0.72`,
wall-clock incl. one model load per worker per checkpoint; the `--est-*` defaults are still the T001
guesses env 10 ms, infer1 300 ms, β 0.1, straggler 0.1); P=2 rows are inference-bound, so the
straggler value does not matter there:

| β                 | P=1, N=4 | P=1, N=8 | P=2, N=8 | P=2, N=16 |
| ----------------- | -------- | -------- | -------- | --------- |
| 0.15              | 12.0 h   | 8.0 h    | 4.2 h    | 3.3 h     |
| 0.25 (4090 meas.) | 13.2 h   | 9.4 h    | 5.6 h    | 4.9 h     |
| 0.35              | 14.4 h   | 10.8 h   | 7.0 h    | 6.4 h     |

→ 4×A100 P2 expectation **≈ 5–6 h at P=2, N=8–16 (range 3–7 h); 9–13 h at P=1**. These are 4090
numbers transferred to the A100: its batch-1 latency and β (more SMs/HBM bandwidth, usually lower β), the
EGL render cost and the CPU-side straggler at N=8–16 are not measured. **Re-measure on the A100** with
`setup_server.sh` + a short run of `submit_multi.sh` (P=2, N=8 and N=16); workers log `ms/call` and
`ms/env-step` (`<out-root>/multi/<session>/w*.log`); put them into `--est-infer1-ms`, `--est-beta`,
`--est-env-ms`, `--est-straggler` for the next dry-run. P=2 overlap on one GPU is also unmeasured.

**Single vs batched (same 4090, pi05lib libero_goal tasks 1, 8 × init 0–3, C0 + C4-a/b = 24 paired
episodes):** success_orig 24/24 agree, first_grasp_object 24/24, success_new (C4) 13/16 (SR 9 vs 8 of 16).
Actions differ already at step 0 (max |Δa| ≈ 1e-3, bf16 batch-N kernels), eef DTW median 3 mm (max 14 cm).
C0 trajectories stay within 1 cm (one exception: 1.1 cm at step 78); 15/16 C4 episodes drift apart
past 1 cm (steps 52–222) and 8/16 past 5 cm (steps 104–214), which is where the 3 success_new flips
come from. Expected (see above): treat batched episodes as a separate sample, never pair them with
single-path ones.

## Tests

```bash
uv run pytest scripts/shortcut/tests -q     # CPU only; fake GPU ids, mock env + policy
```

`test_parallel.py` runs the single-env loop and the parallel path on the mock backend and requires
identical episode records (except timing), identical step logs, the same files, (task, init, variant)
order, resume after `SIGKILL` of the whole process group, recovery from a worker crash, and work
stealing between two checkpoints.
