# T034 sparse crosscoder — PREREG (2026-11-26, before any training result)

Copy of this file: `/outputs/VLA-shortcut-exps/sae/crosscoder/PREREG.md`. Code: branch `exp/sc-crosscoder`,
`scripts/shortcut/sae/{crosscoder,cc_data,cc_train,cc_smoke,cc_analysis}.py`.

## 1. Model (acausal crosscoder, Lindsey et al. 2024, with the TopK + AuxK recipe of model.py)

- Blocks l = 0 … L−1 (layers). Per block, exactly the T015 / T031 SAE normalisation:
  x̃_l = ((x_l − b_pre_l) − μ_l) / ‖(x_l − b_pre_l) − μ_l‖₂, b_pre_l = geometric median of block l on 10 000 train rows,
  μ_l = scalar mean over the d_l dims. Every block is a unit vector → every layer counts equally.
- Encoder reads all blocks: z = ReLU(TopK(Σ_l W_enc^l x̃_l)) (no encoder bias). Decoder per block: x̂̃_l = W_dec^l z (no bias).
- Loss = Σ_l ‖x̃_l − x̂̃_l‖² / C_l + (1/32) Σ_l ‖ẽ_l − ê_aux,l‖² / C_l, C_l = variance of the centred x̃_l at init;
  AuxK on latents dead for 500 steps, k_aux = 512 (model.py).
- Decoder constraint: the concatenated decoder vector of feature i, [d_i^0; …; d_i^{L−1}], has unit norm
  (tangent-plane gradient + renorm every step). **Layer profile** p_il = ‖d_i^l‖² (Σ_l p_il = 1).
- Init as model.py (W_dec random unit rows, W_enc = W_decᵀ·√(k/D)). Adam lr 1e-4, betas (0.9, 0.999), grad clip 1.0,
  batch 4096 (joint: 2048 if 4096 does not fit in 18 GB), bf16 autocast for the matmuls, fp32 master weights.
- With L = 1 the crosscoder is model.TopKSAE (CPU test `tests/test_crosscoder.py`: same codes, loss, gradients).

## 2. Data (all on the same LIBERO rows)

HuggingFaceVLA/libero rev 86958911, every timestep with frame_index % 2 == 0: 137 154 rows, 1 693 episodes.
- **expert**: ae00–ae17 output, mean of the 50 action tokens, denoising step 9 (T031 acts, noise seeded by (episode, frame)). 18 × 1024.
- **vlm**: pg00–pg17 output, pooled_all (count-weighted mean of cam1, cam2, text group means = T015 input). 18 × 2048.
  New collection (collect.py prefix-only, layers emb + pg00–pg17, frame stride 2, 64 parts, noise seed 0).
  emb is collected and stored but not in the crosscoder (kept optional).
- **joint**: vlm blocks then expert blocks, rows paired by (episode, frame) (checked row by row). 36 blocks.
- Split: T031 held-out episodes (10 % = 169 episodes, numpy default_rng(0) permutation of the sorted episode list);
  train 123 996 rows, held-out 13 158 (expert). Length: 100 epochs over the train rows (≈ 3 030 steps at batch 4096).

## 3. Size sweep and selection rule

- **expert**: n_latents ∈ {8 192, 16 384} × k ∈ {64, 128}, 4 runs, seed 0.
  If one full-length run takes > 20 min (smoke measurement), the sweep runs use 30 epochs and the selected config
  is retrained at 100 epochs.
- Rule: candidates = runs with never-fire ≤ 10 % (never fires on any of the 137 154 rows). Score = mean held-out VE
  over the 18 layers. Start from the smallest (n 8 192, k 64); move to k 128 only if it raises the mean VE by ≥ 0.01;
  move to n 16 384 only if it raises the mean VE by ≥ 0.005 at the chosen k. (Smaller L0 / dictionary preferred
  for interpretation unless the gain is clear.)
- **vlm** (reduced sweep): k ∈ {chosen expert k, other k} at n_latents = the largest n in {16 384, 8 192} whose run
  fits in 18 GB (smoke probe); same rule for k.
- **joint**: n_latents = the vlm choice if it fits, else 8 192; k = vlm choice. No sweep.

## 4. Metrics per crosscoder

- Per layer, held-out: VE in the raw space (1 − Σ‖x_l − x̂_l‖² / Σ‖x_l − mean_l‖², same as T031 `evaluate`), nmse_l.
  Next to the per-layer SAE: T031 `weights/L??_s9` held-out VE (expert); T015 `pg??` VE (vlm; T015 reports
  training-data VE, no held-out split — stated in the table).
- Never-fire fraction (over all 137 154 rows); mean L0.
- Massive-channel share per layer (T031 definition): share of ‖x̃_l‖² in massive channels; VE on massive / other
  channels. Expert channels: T030 `massive_channels.json` e_s9/L*. VLM: same rule (RMS > 20 × median channel RMS)
  on 20 000 random pooled_all rows of the new collection.

## 5. Collection checks (run 3)

- (episode, frame, dataset index) keys equal the expert cache row by row.
- T015 per-layer SAEs (pg00–pg17, `TopKSAE.load`) on a 20 000-row random sample of the new pooled_all: VE within
  0.03 of their T015 training VE (pg17 reported, not required). Fail → stop and report before VLM training.
- Disk: stop if the collection exceeds 60 GB.

## 6. Analysis (CPU, after training)

Features with zero fires are excluded from all counts.

**Layer-profile categories** (thirds: S = layers 0–5, M = 6–11, D = 12–17; m_S, m_M, m_D = profile mass), first match:
1. shallow-only: m_S ≥ 0.7
2. deep-only: m_D ≥ 0.7
3. middle-only: m_M ≥ 0.7
4. carried through all layers: min(m_S, m_M, m_D) ≥ 0.15
5. born in the middle: m_S < 0.15 and m_M ≥ 0.3 (starts in the middle, continues deep)
6. shallow-to-middle (fades before deep): m_D < 0.15
7. other.
Also the depth centroid c_i = Σ_l l · p_il / (L−1) and a heat map of profiles sorted by centroid.
Joint crosscoder: VLM share v_i = Σ_{VLM blocks} p_il. VLM-only v ≥ 0.9, expert-only v ≤ 0.1, **shared** otherwise;
categories above applied within each model's part (renormalised).

**Labels** (training-data codes, T032 rules A2 unchanged): rare / always-on / task-specific (sub-tag
instruction (same-scene)) / scene / suite / phase / mixed, with task_index → our task codes and phase from the
gripper command (dataset action[:, 6]) as in T032. Fire = code > 0 on a row. Expert: T031 selectivity is
reported as well where computable.
- Question: where in depth do **instruction** features sit compared with scene and task features?
  Compare the depth centroid distributions (median, Mann–Whitney U, two-sided) of instruction vs scene vs
  task-mixed features in the VLM and joint crosscoders.

**T028 episodes** (1 560 episodes, /outputs/VLA-shortcut-exps/rich_acts): VLM pooled_all rebuilt from vlm_seg + seg_ntok
as T032; expert exp_mean step 9. Encode every call.
- Depth trajectory: per-layer contribution of active feature i = z_i · ‖d_i^l‖.
- Fallback trials (T032 pool: cand31 + L90 lscene, executed vs instructed task): for task-feature sets F_E (executed)
  and F_I (instructed) from the training labels, fraction active per call and their per-layer contribution
  profile (where they appear and vanish). Compare with U40 training trials of the same tasks (T032 IDX_I / IDX_E
  normalisation, per call).
- Joint: among shared features, the instructed vs executed task features active in fallback trials.

## 7. Hypotheses (stated before results; descriptive, no correction)

- **H1 quality**: crosscoder held-out VE is lower than the per-layer SAEs (one k for 18 layers), mean gap ≤ 0.10
  for expert and vlm; per-layer gap largest in the deep expert layers (14–17) where per-layer SAEs are already ≈ 0.90.
- **H2 profiles**: in both models, "carried through all layers" is the largest single category among live
  features (residual stream persistence); expert layers 1–4 (massive channels) do not dominate the profiles
  because of the per-layer normalisation.
- **H3 joint**: shared features are a minority (5–30 % of live features); VLM-only features are the majority.
- **H4 instruction depth**: instruction (same-scene) features have a deeper depth centroid than scene features
  in the VLM crosscoder (T030/T032: instruction information grows with depth).
- **H5 fallback**: in fallback trials executed-task features are active across depth from call 0; instructed-task
  features are weaker and, where present, sit in the deep VLM layers and do not reach the expert (joint shared
  features carry the executed task).

## 8. Budget and stopping

GPU 0 only, one job at a time under the queue lock; wall clock logged per job (`run_gpu.sh` → gpu_time.tsv).
Stop and report at 8 GPU-h total, OOM, smoke throughput < 50 % of the T031 measurement, or collection > 60 GB.

## Addendum A (2026-11-26 15:42, after run expert_n8192_k64 only; the other 3 sweep runs not finished)

- Observation: expert_n8192_k64 has never-fire 0.57 (dead in the last 500-step window 0.47), held-out VE mean 0.961.
  The ≤ 10 % never-fire filter in §3 will likely exclude every config.
- Rule change, fixed now: if no run passes the never-fire filter, the filter is dropped and the §3 VE rule is applied
  to all runs (start n 8 192 k 64; k 128 if mean VE +≥ 0.01; n 16 384 if +≥ 0.005). Never-fire and the number of
  live features are reported next to VE. Dead (never-fire) features are excluded from every analysis count (§6).
- Throughput deviation: the smoke showed 0.32 × the T031 SAE throughput (shared GPU); lead informed, continued.
- Memory deviations (lead approved): VLM n_latents capped at 8 192 (16 384 OOM); joint tries n 8 192 with bf16 Adam
  states, else n 4 096 (to be fixed in Addendum B before the joint run).
