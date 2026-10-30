# π0.5 residual-stream SAEs on LIBERO (T015)

Reproduces the SAE setup of Swann et al. 2026, "Sparse Autoencoders Reveal Interpretable and Steerable
Features in VLA Models" (arXiv 2603.19183): App. A.3 (activation collection), B.1 (TopK + AuxK SAE),
C.1 (feature metrics), F.2 (steering). This version covers **every layer** (emb + PaliGemma 0–17 + action
expert 0–17 = 37 SAEs) and keeps the token groups apart.

| file         | what                                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------------- |
| `hooks.py`   | layer names, `ActivationRecorder` (pooled recording), interventions `add`, `project_out`, `replace_with`      |
| `collect.py` | dataset frames → policy preprocessor → `predict_action_chunk` with the recorder → fp16 memmap parts           |
| `store.py`   | on-disk layout, part splitting, `load_layer()` reader (pooled inputs)                                         |
| `model.py`   | `TopKSAE` (TopK + AuxK, pre-bias, per-sample normalization, unit-norm decoder, tangent-plane grads)           |
| `train.py`   | one SAE per (layer, seed), GPUs / pods split the jobs, resume, W&B offline + `log.jsonl`, then `stats.py`     |
| `stats.py`   | sparse codes of every timestep, C.1 metrics per feature, firing rate per suite / task                         |
| `smoke.sh`   | server GPU smoke in one command: hook checks, 3-episode collection, 2 SAEs, timing → PASS/FAIL + estimates    |
| `smoke.py`   | hook checks on the real model (`hook_check.json`), SAE step timing, `--summarize`                             |
| `tests/`     | CPU tests (`test_sae.py`: SAE, stats, interventions; `test_collect_mock.py`: mock π0.5 on real LIBERO frames) |

## Hook location

lerobot's `PI05Pytorch.sample_actions` (inference) runs the prefix once through
`paligemma.model.language_model` and each denoising step through `gemma_expert.model`; both are
`PiGemmaModel.forward`, which calls every `PiGemmaDecoderLayer` module. So `register_forward_hook` on
`…language_model.layers[L]` / `…gemma_expert.model.layers[L]` gets the block output (after attention, MLP and
both gated residuals, before the final norm). The joint loop `compute_layer_complete` (no module call, hooks
would not fire) only runs in the training forward. `smoke.py` checks this on pi05lib: layer L's hooked output
is bit-identical to layer L+1's input and layer 17's output to the final norm's input, for PaliGemma (1 call
per timestep) and the action expert (10 calls); `emb` (pre-hook on PaliGemma layer 0) is the `embed_prefix`
output in bf16; the action with every hook attached is bit-identical to the one without.

| name   | where                                                       | d    | stored per timestep                       |
| ------ | ----------------------------------------------------------- | ---- | ----------------------------------------- |
| `emb`  | input of PaliGemma layer 0 (image + text embeddings)        | 2048 | 3 group means (cam1, cam2, text) + counts |
| `pgNN` | output of PaliGemma layer NN, 0–17                          | 2048 | same                                      |
| `aeNN` | output of action-expert layer NN, 0–17, each denoising step | 1024 | mean of the 50 action tokens, per step    |

Token groups (prefix = 3 × 256 image tokens + 200 text tokens): cam1 = `observation.images.image`
(agentview), cam2 = `observation.images.image2` (wrist), text = instruction + discretized state + `Action:`
(π0.5 puts the state in the prompt). The third image slot (`empty_camera_0`, all padding) and the text
padding are excluded via the prefix pad mask; the counts of valid tokens are stored, so the paper's
mean over all tokens is the count-weighted mean of the three groups (`store.pool_pg`, `--input pooled_all`).

## Server procedure

```bash
# 1. environment (venv, LIBERO, checkpoints, tokenizer, HuggingFaceVLA/libero ~35 GB); no extra deps for SAEs
SC_OUT_ROOT=/data/sc scripts/shortcut/setup_server.sh --gpus 0,1,2,3
export SC_OUT_ROOT=/data/sc          # outputs go to $SC_OUT_ROOT/sae/<model>/

# 2. smoke (one GPU, ~10 min): new run dir <out-root>/run-<time>-<host>-<rand>/ per call; every step's exit
#    code (steps.tsv) and every sub-check → PASS/FAIL (exit 0 only on PASS), measured timesteps/s → hours and GB
scripts/shortcut/sae/smoke.sh --out-root $SC_OUT_ROOT/smoke/T015 --gpu 0

# 2b. plan: timesteps, storage, time (time uses the THROUGHPUT table in collect.py; update it from step 2)
uv run python scripts/shortcut/sae/collect.py --model pi05lib --dry-run

# 3. collect: one pod per shard (N=8 here), all GPUs of the pod; re-run the same command to resume
uv run python scripts/shortcut/sae/collect.py --model pi05lib --shard $I/8 --gpus 0,1,2,3

# 4. train 37 layer SAEs (+ stats); split the jobs over pods with --shard, over GPUs with --gpus
uv run python scripts/shortcut/sae/train.py --model pi05lib --layers all --shard $I/4 --gpus 0,1,2,3
```

The shared dev container does not run GPU work for T015 (smoke.sh refuses where `/queue/locks` exists). If the
scripts are run there anyway, they flock the GPU and accept only GPU 0 unless `--allow-any-gpu`; `--gpus cpu`
runs collection / training on the CPU (tests). Without `/queue` nothing is locked.

Collection options: `--layers` (default all 37), `--ae-steps all|first,last|0,5,9` (default all 10),
`--episodes 0-9` (smoke), `--parts 64` (resume unit; same for every pod count), `--batch 32`,
`--workers 6` (PNG decoding), `--model pi05lib|pi05base|<ckpt_registry id>|<path>` (pi05base gets the LIBERO
adapter as in sc-bench). Noise for the denoising loop is seeded by (episode, frame), so a timestep gives the
same activations in any shard / batch layout (up to bf16 batch-size effects). Each worker opens the dataset
filtered to its episodes; `datasets` writes an Arrow copy of those rows into `$HF_DATASETS_CACHE` (in total
about the size of the parquet data, ~35 GB, once per node).

Training options: `--layers all|emb|pg:0-17|ae:0-17`, `--input pooled_all|img|text|cam1|cam2`
(PaliGemma/emb), `--ae-steps all|0,9` (action expert: which recorded steps are samples; all together by
default, the step stays in the records), `--seeds 0,1,2` (stability), `--expansion 1`, `--k-pg 100`,
`--k-ae 64`, `--batch 4096`, `--epochs 100`, `--steps N`, `--lr 1e-4`, `--force`, `--dry-run`,
`--wandb offline|off`. Jobs = layers × seeds; `--shard i/N` takes every N-th job, `--gpus` runs the shard's
jobs biggest-first on the least loaded GPU. W&B logs offline into each job dir (`wandb sync <dir>/wandb/offline-run-*`
where a key works).

### Running many pods on one shared acts dir

Assumption: all pods share **one PVC** (same filesystem, with hard links — NFS / CephFS have them; without them
the scripts stop with an error). Mount paths may differ between pods. Pods may overlap (a replacement pod
starting before the old one is gone, or two shard specs covering the same parts), and every pod is often PID 1.
Collection is safe under that:

Leases are best-effort (POSIX filesystems have no compare-and-swap): a race between writers only causes duplicate
work. Correctness rests on the fixed identity and on the atomic rename that publishes a part (first one wins).

- `plan.json` is created exclusively (a complete temp file hard-linked into place); a pod that finds it compares
  the identity and stops if it differs. Multi-GPU worker processes read the verified `plan.json`; they never
  recompute it.
- the identity is content only: checkpoint and adapter revisions (`hub:<commit sha>`, or `sha256:` of a local
  checkpoint's `*.safetensors` + `*.json`), inference dtype, dataset revision, layers, steps, episodes, parts,
  noise seed. Locations (hub id, mount path of a local checkpoint, dataset root) are kept under `plan.json →
source` for reference and are not compared. Each pod resolves `--model` to its own location and loads
  **exactly the recorded revision** (Hub: that commit's snapshot; local: the dir, whose hash must match), and
  re-checks the revision after loading, so a moved Hub `main` or edited local checkpoint cannot mix weights.
- a part is claimed with a lease `parts/<part>.lease` holding an owner token (host, pid, token, start). A
  heartbeat thread touches it during the whole part (also through data loading and flushes) while the main loop
  checks it every batch and before publishing. A fresh lease of another writer → the part is skipped (re-run
  later to pick it up); a lease not touched for `SC_SAE_LEASE_TTL` s (default 3600) is stale and taken over by
  replacing it with a new token. A writer that finds another token in its lease (it stalled past the TTL and
  the part was taken over) drops its work and never touches the new owner's lease.
- each writer works in its own `parts/.<part>.tmp-<host>.<pid>.<rand>/` and publishes with one rename; if the
  final part already exists (another writer was first), its own copy is dropped. Its tmp dir and lease are
  removed when it finishes or fails; a killed pod's tmp dir stays behind (safe to delete once its lease is stale).
  A final part dir without `done.json` is never made by this code; if one exists, collection stops with its
  path (delete it by hand).
- NaN/inf in any activation fails the part (no done.json, the run exits non-zero); only finite values beyond the
  fp16 range switch a layer to fp32.

Training: a job is skipped only if its `done.json` was written for the same config — SAE hyperparameters, input,
ae steps, the collection identity and the exact list of collected parts (not the acts dir path). A finished job
of another config stops the run (`--force` retrains it, or use another `--sae-dir`); an unfinished job deletes
any old `done.json` before training, so a crash never leaves a stale "done".

Pod split examples: collection N=8 pods × 4 GPUs = 32 workers over 64 parts (2 parts each). Training: 37 jobs; the
18 action-expert jobs are 10× bigger (2.7 M samples) than the PaliGemma/emb ones, so `--shard i/2 --gpus 0,1,2,3`
(2 pods × 4 GPUs) or one 8-GPU pod with `--gpus 0,...,7` balances by size.

## Outputs

```
$SC_OUT_ROOT/sae/<model>/
  acts/plan.json                    identity + part list (episode ranges)
  acts/parts/ep00000-00026/         emb.npy, pg00..17.npy (T,3,2048) · ae00..17.npy (T,10,1024) fp16
                                    counts.npy (T,3) · meta.parquet · done.json
  sae/<input>/<layer>/seed<s>/      <input> = pooled_all|img|… for emb/pg, ae-steps-all for ae
     sae.safetensors, config.json   TopKSAE.load(dir)
     metrics.json                   nmse, var_explained, dead_frac_*, train_seconds, stats summary
     fire_freq.npy, log.jsonl, wandb/, ckpt.pt
     codes_idx.npy, codes_val.npy   (N, k) top-k latent indices / values for every sample
     codes_meta.parquet             index, episode_index, frame_index, task_index, suite[, ae_step]
     features.parquet               feature[, ae_step], ep_coverage, mean_onsets, mean_max_act,
                                    rel_run_length (C.1, τ_on = 0.1), n_active_eps, fire_freq
     fire_by_group.parquet          feature[, ae_step], group_type (suite|task), group, fire_rate, ep_coverage
```

`meta.parquet` per timestep: row, index, episode_index, frame_index, task_index, suite, instruction, action
(dataset), policy_action0 (first action of the policy chunk, unnormalized: sanity check vs the dataset).
A layer whose values exceed fp16 range in a part is stored fp32 for that part (`done.json → overflow_fp32`,
warning printed); `store.load_layer` handles mixed dtypes.

## SAE details (paper B.1)

TopK + AuxK (Gao et al. 2024): x̃ = ((x − b_pre) − μ)/‖·‖ (μ = scalar mean over d), z = ReLU(TopK(W_enc x̃)),
x̂̃ = W_dec z, no encoder/decoder bias, unit-norm decoder columns with tangent-plane gradient projection,
W_enc = W_decᵀ·√(k/n) at init, b_pre = geometric median of 10k samples (Weiszfeld). Loss = ‖x̃ − x̂̃‖²/C +
(1/32)‖ẽ − ê_aux‖²/C with k_aux = 512 dead latents (no firing for 500 steps) and C = variance of the
centered normalized activations at init. Both terms are in the normalized space (the paper's eq. 5 is ambiguous
about where the first term lives; this makes `nmse` = fraction of variance unexplained of x̃). `metrics.json`
also gives the raw-space variance explained. Adam 1e-4 (0.9, 0.999), batch 4096, 100 epochs, clip 1.0.

## Interventions (paper 3.4, F.2)

```python
from hooks import add, project_out, replace_with
from model import TopKSAE

sae = TopKSAE.load(".../sae/pooled_all/pg05/seed0", "cuda")
v = sae.W_dec[:, 158].detach()
with add(policy, "pg05", v, alpha=100):  # y + α·v
    chunk = policy.predict_action_chunk(batch)
with project_out(policy, "pg05", v):
    ...  # y − (y·v̂)v̂
with replace_with(policy, "pg05", sae):
    ...  # y → SAE(y), every token; mode="mean" replaces the token-mean only
```

Applied to every token of every call: PaliGemma layers run once per query and reach every denoising step through
the KV cache; action-expert layers are hit at each of the 10 steps. Rollouts with steering are not part of T015.

## Estimates (computed, not measured: confirm with `smoke.sh` on the server)

Full LIBERO (1,693 episodes, 273,465 timesteps), all 37 layers, all 10 denoising steps. The dev container ran no
GPU work for T015, so the times below come from FLOP counts; `smoke.sh` prints the measured values
(`summary.json → estimates`), and the `THROUGHPUT` table in `collect.py` (used by `--dry-run`) should be updated from them.

- storage: 588 KB/timestep fp16 → **~165 GB** (PaliGemma+emb 19 × 3 × 2048 × 2 B = 233 KB, action expert
  18 × 10 × 1024 × 2 B = 369 KB). `--ae-steps first,last` → ~74 GB. Per SAE, codes + stats: PaliGemma ~0.1 GB,
  action expert (2.7 M samples) ~0.7 GB.
- collection: ~4.8 TFLOP per timestep (Gemma-2B prefix of 968 tokens ≈ 3.9, SigLIP on 3 images ≈ 0.6, 10 expert
  steps ≈ 0.3). At ~40 % of the bf16 peak: **A100 ~25 timesteps/s → ~3 h on one GPU**, 8 pods × 4 A100 → ~6 min
  of compute per pod plus model load. RTX 4090 ~7/s → ~11 h on one GPU.
- SAE training (100 epochs, batch 4096, fp32): PaliGemma/emb ≈ 6.7 k steps of ~0.3 TFLOP → ~5 min per layer on an
  A100 (with stats); action expert ≈ 67 k steps of ~0.08 TFLOP → ~15 min per layer. All 37 layers ≈ 6 h on one
  A100, ~1.5 h on 4. Training needs the layer in GPU memory: 2.2 GB (PaliGemma, fp32) or 5.6 GB (action expert, fp16).

## Checks done for T015 (no GPU in the dev container)

- `smoke.py --device cpu --frames 2` on the real pi05lib (`/outputs/VLA-shortcut-exps/smoke/T015/cpu/hook_check.json`):
  actions bit-identical with and without the hooks; PaliGemma layer L output = layer L+1 input (bitwise) and layer 17
  = final-norm input, same for the action expert over the 10 denoising steps; emb = embed_prefix output; group counts
  256 / 256 / 46 (text valid tokens = attention mask), empty camera masked, group-mean error 6e-8, weighted
  group mean vs all-token mean rel. error 1e-9; policy first action vs dataset action L1 0.029.
- This run found that lerobot calls `gemma_expert.model.forward()` directly, so a hook on the expert model never
  fires; denoising steps are counted on action-expert layer 0 instead (the mock test now calls `.forward()` too).
- CPU tests (`uv run pytest scripts/shortcut/sae/tests -q`, 38): layer specs, part / shard split, fp16 overflow →
  fp32, SAE init / constraints / loss decrease / AuxK / save-load, C.1 hysteresis, interventions on a toy layout
  (`project_out` leaves no v component, `add`, `replace_with` matches the SAE metrics), and a mock π0.5 on the
  real 3-episode LIBERO data through collect (shard 0/2 + 1/2, resume, config clash), reader, train (stop + resume),
  stats, `--ae-steps` subset and the `smoke.py --summarize` PASS/FAIL logic. `test_review.py` reproduces the
  review cases: concurrent plan creation with PID 1 everywhere, checkpoint fingerprint + dtype in the identity,
  lease (fresh skip / stale takeover) and first-writer-wins publish (also a killed pod's tmp + stale lease in the
  mock run), NaN/inf failing a part with nothing left behind, train skip only on an identical config (another
  config is refused), `--force` crash leaving no done.json, smoke FAIL on any step exit code / hook sub-check /
  missing bench, stats from fp32 codes near τ_on.
