# T031 action-expert SAE: pre-registration

Written 2026-11-26, before any SAE was trained or any SAE number was computed.
- Checks done beforehand: array shapes / keys of the T028 npz, the T030 cache (`analysis/layer_pop/_cache/exp_mean.npy`
  (18, 51 100, 3, 1024) fp16 + `meta.pkl`), T030 `res/massive_channels.json` (channel lists only), weights.npz keys.
- No SAE, reconstruction metric, firing statistic or decoding score was looked at.

## Common
- **SAE** = T015 `model.TopKSAE` unchanged (TopK + AuxK, arXiv 2603.19183 B.1): per-sample normalisation
  x̃ = ((x − b_pre) − μ)/‖·‖, b_pre = geometric median of 10k training samples, unit-norm decoder, k_aux 512,
  aux 1/32, dead = no firing for 500 steps, Adam 1e-4, batch 4096, grad clip 1.0, seed 0.
- **Split:** 90 / 10 by episode, random with seed 0 (held-out = 10 % of episodes). Quality is reported on the held-out
  episodes; never-fire is counted on train + held-out.
- **Metrics:** NMSE (normalised space, T015 definition), VE (raw space, `1 − Σ‖x − x̂‖² / Σ‖x − x̄‖²`), never-fire
  fraction, dead fraction in the last 500 steps.
- **Steps:** stage 1 fixed 3 000 optimizer steps (≈ 260 epochs of 46 k samples). Stage 2 `max(100 epochs, 3 000)`.

## Stage 1 (T028 data, quick)
- Input: T028 `exp_mean` (= hooks.py `aeNN`, action-token mean), layers {2, 5, 8, 11, 14, 17} × denoise steps {0, 9}
  → 12 SAEs, 51 100 calls each (1 560 episodes).
- **Sweep** at L11 step 9: ER ∈ {2, 4} (n = 2048 / 4096) × k ∈ {32, 64}.
  **Selection rule:** among settings with never-fire ≤ 10 %, take those with held-out NMSE within 0.01 of the best;
  of these the smallest k, then the smallest n. If no setting has never-fire ≤ 10 %, the lowest never-fire.
  The selected setting is used for all 12 stage-1 SAEs and for stage 2.
- **Massive channels** (T030 list, RMS > 20 × median channel RMS, from `massive_channels.json` keys `e_s0/L*`,
  `e_s9/L*`): per SAE report
  (a) share of ‖x − mean_d(x)‖² in massive channels (raw, before b_pre),
  (b) share of ‖x̃‖² in massive channels (after b_pre and per-sample normalisation, what the SAE fits),
  (c) held-out VE on massive channels only and on the rest.
  "Handled" if (b) ≤ 0.25 and VE(rest) ≥ VE(all) − 0.05. Otherwise reported as not handled (no change to the method in
  this task; the alternative would be per-channel standardisation).
- **Quick feature examples** (U40 training episodes, step 9, layer with the most live features among 11, 14, 17):
  phase-selective = P(fire | best phase) ≥ 0.5 and ≥ 3 × mean P(fire | other phases) (phases = T030 `phase_labels`:
  reach / grasp / carry / release); task-selective = P(fire | best task) ≥ 0.5 and ≤ 0.05 in every other task.
  Show the top 2–3 of each by the ratio.

## Stage 2 (LIBERO training data, T015 method)
- **Data:** HuggingFaceVLA/libero rev 86958911 (same revision as T008 / T015), copied from the T008 local copy into
  `/data0/woosung/datasets/VLA-shortcut-exps/hf_libero` (no network download needed; same files).
  `HF_DATASETS_CACHE` = `/data0/woosung/datasets/VLA-shortcut-exps/hf_cache`.
- **Collection** (`collect.py`, full forward, pi05lib, noise seeded by (episode, frame)):
  (a) token mean: ae{2, 5, 8, 11, 14, 17} × steps {0, 4, 9};
  (b) per token: step 9 only, 8 of the 50 action tokens per timestep, drawn without replacement with a seed from
      (noise seed, episode, frame); token index saved.
  **Timestep subsampling:** frames with `frame_index % s == 0`. s = the smallest of {2, 3, 4, 5, 6} for which the
  estimate (smoke-measured ts/s) of collection + training ≤ 3.2 GPU-h (smoke + slack inside the 4 GPU-h cap).
  Per-token samples come from the same timesteps.
- **Training:** 18 mean SAEs (layer × step) and 6 per-token SAEs (step 9), stage-1 setting.
  T015 stats (`stats.run_stats`: codes, features.parquet C.1 metrics, fire rate per suite / task) on all samples.

## Stage 2 feature analysis (CPU; T028 1 560 episodes encoded with the stage-2 SAEs)
- `exp_mean` step 0 / 4 / 9 → the mean SAEs of the same (layer, step) directly. Per-token: x ≈ pc @ components + mean
  (T028 PCA-128); report the PCA truncation (1 − Σ explained_variance / total_variance) and the SAE VE on these
  reconstructed tokens. "Fires" = activation > 0. Primary step 9, primary layers all six (reported per layer).
- **A1 trajectories** (descriptive): feature activation per call with grasp / release markers for basket O9
  ctl / opp / far20 / far25, Goal g10 vs G7 / G2, the T030 L90 lscene fallback case. Features shown = the top 12 by
  variance of the per-call condition mean.
- **A2 selectivity** (U40): phase-selective / task-selective counts as in stage 1, per layer; action correlation =
  Pearson r between feature activation and the executed first action (gripper = act dim 6, z = dim 2) over U40 calls.
  H2a: share of phase-selective live features is higher at L14/L17 than at L2/L5.
  H2b: at least one live L17 step-9 feature has |r| ≥ 0.7 with the gripper command.
- **A3 fallback** (T030 pool: cand with `exec_other`, L90 lscene fallbacks): task-selective features (A2) of the
  executed task vs of the instructed task; per call c from 0: fraction of trials where ≥ 1 executed-task feature
  fires and where ≥ 1 instructed-task feature fires. Features that separate executed from instructed = executed-task
  features firing in fallback trials ∧ instructed-task features not firing.
  H3 (from T030 H3): executed-task features fire in ≥ 50 % of fallback trials by call 2; instructed-task features
  in ≤ 10 % (all layers).
- **A4 basket** (Object tasks with the basket moved; pairs within the original machine: near / opp vs libero3 O ctl
  on A100; far20 / far25 vs ctl4090): per episode, mean activation over calls ≥ grasp_call (all calls if no grasp).
  Basket-sensitive = |AUROC(ctl vs moved) − 0.5| ≥ 0.3 for any moved condition. Among those: **graded** if the
  condition means are monotone in distance (ctl 0 < near ≈ 10 < far20 20.6 < far25 25.7 cm), else **on/off**.
  opp is reported separately (other side, not on the distance line). Count graded vs on/off per layer.
- **A5 Kaufman output-potent:** potent fraction of a decoder direction d = ‖P d‖² / ‖d‖², P = projector onto the
  row space of `action_out_proj` (32 × 1024; random direction 32/1024 = 0.031). Compared: behaviour features (phase-,
  task-selective or |r| ≥ 0.5 with any action dim) vs other live features (Mann–Whitney), per layer.
  H5: behaviour features have a higher potent fraction than other live features at L17 (one-sided p < 0.05).
  Caveat registered: the final norm before `action_out_proj` is not included; for layers < 17 the projection is of
  a residual-stream direction that later layers can still rotate.
- **A6 comparison with T030 vectors:** same decoding as T030 (`fit_predict`: standardise → PCA-50 → logistic, folds
  grouped by init for task / phase on U40; fallback executed-vs-instructed with the U40-trained task decoder) on
  (i) T030 `e_s9` z-scored vector (massive channels zeroed) and (ii) the SAE code vector (dense n). Report balanced
  accuracies side by side. Expectation (not a test): SAE codes within 0.05 of the vector.

## Budget / stop rules
- GPU: VGI-102 GPU 0 only, flock on its queue lock. Stop and report before exceeding 4 GPU-h (smoke + collection +
  training). Disk: activations ≤ 40 GB under `/data0/woosung/datasets/VLA-shortcut-exps/acts/`.
- CPU workers ≤ 8. No existing output is overwritten.

## Addendum 2026-11-26 (scope extension from the human via lead, before any stage-2 SAE existed)
- **All 18 expert layers:** token-mean SAEs for layers 0–17 × steps {0, 4, 9} = 54 (stage-1 setting, ER 2 / k 64);
  per-token SAEs stay at {2, 5, 8, 11, 14, 17} step 9. Layers {0,1,3,4,6,7,9,10,12,13,15,16} come from a second
  collection (same timesteps: frame stride 2, same noise seeds, token mean only). GPU cap 6 GPU-h; acts + caches ≤ 80 GB.
- Layers 0–2: massive channels (T030) are flagged in the interpretation; the method is unchanged (stage 1: L2 fails
  the "handled" criterion).
- A2–A6 run on all 18 layers (step 9; A2 on all 3 steps). H2a / H2b / H3 / H5 keep their pre-registered layers.
- **A1b layer × call maps (new, descriptive):** per (layer, step), calls 0–27, references = U40 per-task mean SAE
  code at the same call. (i) fallback pool: share of trials whose code is closer (cosine) to the executed than to the
  instructed task, and the mean cosine difference; (ii) basket O9: cosine distance of the condition-mean code from its
  machine's ctl, minus the ctl split-half distance; (iii) Goal g10: cos(G7) − cos(G2); (iv) T030 L90 lscene case:
  cos(executed) − cos(instructed ≈ nearest_trained). Trajectory small multiples for all 18 layers (step 9).
- **A3 exploratory (not a test):** the PREREG task-selective definition leaves almost no fallback trial with features
  for both its executed and instructed task, so a pair-contrast version is added: per trial, U40 features with
  P(fire | executed) ≥ 0.3 ∧ P(fire | instructed) ≤ 0.05 and the reverse; per call the share of each set that fires.
- Disclosure: the analysis code was tested end to end with the stage-1 SAEs (T028-trained, layers 2–17 of 6) in a
  temp folder; those test numbers were seen and motivated the exploratory A3 variant. They are not reported as results.
