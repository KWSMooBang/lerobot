# Open-loop language probe (T018)

How much, and in which direction, does a policy's action chunk change when **only the instruction** changes — same
observation, same flow-matching noise? No success rate is needed, so models with C0 = 0 (pi05droid zero-shot, pi05base)
sit on the same axis as pi05lib and StarVLA.

| file           | where it runs               | what                                                                                         |
| -------------- | --------------------------- | -------------------------------------------------------------------------------------------- |
| `make_obs.py`  | dev container, CPU (osmesa) | fixed observation sets A, B → `set_{A,B}.npz` + `.meta.json` + `manifest.json` (sha256)      |
| `probe.py`     | GPU server                  | one sc-policy/1 server at a time (sc-bench's `ServerProc`), one chunk per (obs, instruction) |
| `measure.py`   | anywhere (CPU physics)      | open-loop execution of every set-A chunk from the stored sim state + metrics + summary       |
| `run_probe.sh` | GPU server                  | probe → measure, bookkeeping, `--dry-run` GPU-h estimate                                     |
| `lp_common.py` | —                           | query plan, seeds, file loading                                                              |

Tests: `uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_langprobe.py -q` (CPU; mock servers + the
real LIBERO OSC and joint-velocity envs, ~2 min).

## Observation sets (`/outputs/VLA-shortcut-exps/analysis/langprobe/obs/`)

- **A — common, primary** (200 = 4 suites × 10 tasks × init_idx 0–4): the harness's first observation. The `ours`
  env (lerobot `LiberoEnv`, OSC 20 Hz, T009 fix mode) reset exactly as `EpisodeStepper.reset` with
  `seed = episode_seed(7, 0, task, init)` (sc-bench `--seed 7`, repeat 0), incl. the 10 settle steps. Stored:
  agentview + wrist upright at **256 and 360 px** (two envs reset identically; sim states checked equal — lerobot
  servers render 360, StarVLA/DROID 256; probe sends the size closest to the server's `image_hw`), wire state (8,
  lerobot's `LiberoProcessorStep`), `joint_pos` (7), MuJoCo state + `qacc_warmstart` + the Panda gripper controller's
  accumulated command (for an exact restore), EEF pose, and the world position of every LIBERO object/region.
  DROID gets the same images/state (its own harness settles with joint-space no-ops instead; the scene is the same).
- **B — auxiliary, no DROID** (240 = 40 tasks × 2 episodes × progress 25/50/75 %): HuggingFaceVLA/libero @ 86958911
  frames (local copy `<out-root>/data/hf_libero`), the two lowest `episode_index` per task, frame
  `round(p·(len−1))`, 256 px as stored, + the dataset's next 10 actions (`gt_action`). No joint state → pi05droid skips it.
- Rendering is CPU (osmesa). GPU EGL renders can differ slightly in pixels; that is why the files are made once
  and copied to the server (`probe.py` checks the sha256 against `manifest.json`). The npz writer uses fixed zip
  timestamps, so rerunning `make_obs.py` with the same code reproduces the same sha256.

```bash
CUDA_VISIBLE_DEVICES= uv run --extra shortcut-bench python scripts/shortcut/analysis/langprobe/make_obs.py \
    --out /outputs/VLA-shortcut-exps/analysis/langprobe/obs --workers 40          # ~2 min with 40 processes
```

## Instructions per observation

`orig_s0`, `orig_s1`, `orig_s2` (original, seeds = episode*seed repeat 0/1/2: noise floor), `blank` (`""`, seed 0) and
every swap of the task in `instructions/<suite>.json` (C4-* object swaps and C5-\_ target swaps, 102 over 40 tasks;
seed 0). Within an observation everything with seed 0 shares the noise: only the text differs. Per observation
3 + 1 + n_swaps queries; set A = 5 × (40·4 + 102) = 1310, set B = 6 × 262 = 1572 (+ `--repeat-check` 4 repeats of
`orig_s0` per set: same seed + same input must give the same chunk).

## Metrics (`measure.py`, details in its docstring)

1. **Displacement (set A)** — each chunk is executed open-loop in the model's own env (OSC 20 Hz or DROID joint
   velocity 15 Hz) from the observation's restored sim state; Δ = EEF displacement after **0.4 s** (8 OSC / 6 joint
   steps; StarVLA's 8-step chunk is the shortest). effect = ‖Δ(alt) − Δ(orig_s0)‖, floor = ‖Δ(orig_si) − Δ(orig_sj)‖;
   ratio = mean effect / mean floor, frac_above, absolute mm, |Δ(orig)| for scale; gripper command and EEF rotation
   the same way. Per model, suite, task, and alt kind (blank / swap / swap_object / swap_target).
2. **Goal alignment (set A swaps with named targets)** — tilt = how much more Δ(alt) points to the new target than
   Δ(orig_s0) does (cos differences), cos_shift = cos(Δalt − Δorig, p_new − p_orig) in 3-D and xy; null = the same with
   another seed instead of the swap.
3. **Action space (sets A, B; within one action space)** — first 8 steps: arm L2 of env-scale actions, gripper
   sign-flip rate, with the seed floor; set B also orig_s0 vs the dataset's actions (`gt_l2`).

Restore check: `--self-check` replays the first observation of each task from a fresh env reset; the restore path
agrees exactly (0.0 m). Poses are read after `sim.forward()` with observables forced (robosuite samples them on a
timer, so `LiberoEnv.step`'s returned EEF can lag one 2 ms substep).

## Server run (needs approval; ≈ 1 GPU-h)

```bash
# 1. copy the obs dir (made here) to the server's out-root (sha256 checked by probe.py)
rsync -a /outputs/VLA-shortcut-exps/analysis/langprobe/obs/ <server>:/outputs/VLA-shortcut-exps/analysis/langprobe/obs/
# 2. estimate, then run (one model after another on one GPU; resumable)
scripts/shortcut/analysis/langprobe/run_probe.sh --models pi05droid,pi05lib,pi05base,starvla-qwen3pi-libero \
    --out-root /outputs/VLA-shortcut-exps --gpu 0 --dry-run
scripts/shortcut/analysis/langprobe/run_probe.sh --models pi05droid,pi05lib,pi05base,starvla-qwen3pi-libero \
    --out-root /outputs/VLA-shortcut-exps --gpu 0
```

StarVLA needs its venv (`serve/external/starvla/setup_starvla.sh`, as for sc-bench). Output:
`<out-root>/analysis/langprobe/probe/<model>/{hello.json, probe_{A,B}.{jsonl,parquet}, probe_meta.json, servers/}` and
`.../probe/measure/{summary.md, metrics_*.csv, *.parquet, arrows.png}`. `probe_meta.json → server_ms_median` is the
measured ms per forward: feed it back to `--dry-run --est-ms model=ms` (the defaults are placeholders).
Smoke first: `--run-id probe-smoke -- --tasks 1 --max-obs-per-task 1`.

Resume / cache identity: each probe row records `instruction`, `seed` and `obs_npz_sha256`. On resume, a done row
whose text, seed or observation file differs from the current plan stops the run (rows from before this check have
no sha: text and seed only). `exec_A.parquet` is reused only if `exec_A.identity.json` (probe parquet sha256, set-A
npz sha256, backend, LIBERO env tag, control rates, execution version) is unchanged; probe rows recorded with another
observation file are refused. Set B refuses episodes split over several parquet files (none in @86958911).

## Current files and CPU checks (2026-11-06, code 33dfe1ad)

| set | obs | npz sha256                                                         | meta sha256                                                        |
| --- | --- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| A   | 200 | `91057f486b4b9790080246ff269eeab8ab543a6386cad87e67ee394db753c8d5` | `94f6ff0504df1c3734e7b2a7f39757a22b7fb334032d1afe2141e7c8eff7efa8` |
| B   | 240 | `86cd6417b285d70ca5edeb5327dd5cd14212121c245129fa6c4546405bb1e19b` | `dc8d91183cea4244c639dc90f3730fa15c2a06c4036d5fb0a419890cb4cd9f2e` |

Two independent `make_obs.py` runs gave identical sha256 for both files of both sets.

pi05lib on CPU (`policy_server.py --device cpu`, `probe.py --url`): libero_goal task 1, init 0–2, orig_s0 / orig_s1
/ blank / C4-a ("put the cream cheese on the stove"), outputs in `/outputs/VLA-shortcut-exps/smoke/T018/pi05lib_cpu/`:
repeat of orig_s0 bit-identical; 2.8 s per forward on CPU; |Δ(orig)| 36 mm in 0.4 s; effect blank 9.3 mm, swap 14.5 mm
vs floor 4.2 mm (one seed pair); the swap tilts toward the cream cheese in 3/3 (cos_shift 0.68–0.97, null −0.03 … −0.52);
restore vs replay 0.0 m.
