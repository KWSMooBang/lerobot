# T029 neuro_dyn: pre-registration

Written 2026-11-25, before any A1 number was computed. Data checks done beforehand covered only shapes, keys, condition lists and init_file equality.
Stage-B (T028) details get an addendum, committed before any B analysis.

## Common
- Model: pi05lib (lerobot/pi05_libero_finetuned), weights read on CPU from the HF cache safetensors.
- Stage-A data: runs/bench-pi05lib-pro_all (3,390 ep, all on one 4090, GPU 1). The same-machine rule (T022) therefore holds trivially. Any comparison with A100 runs is paired only within the same machine.
- Hook semantics (scripts/shortcut/serve/activations.py): `vlm_vec[l]` and `exp_vec[l]` are the **output** of decoder layer l (forward hook), averaged over tokens of a segment / over the 50 action tokens.
- Statistical unit: the (suite, task) cluster (40 clusters). Each test is a two-sided Wilcoxon signed-rank test over cluster means. Benjamini–Hochberg FDR (q = 0.05) runs over every test listed in this file and its addendum, per analysis family. The number of tests is reported.
- CV: by init (leave-one-init-out) or by task, as stated per analysis. PCA, LDA and CCA are fit on the training fold only.

## A1 Kaufman output-null / output-potent (stage A, PRO)
### VLM → expert read-in
- The expert reads VLM layer j only through K_j, V_j = RMSNorm_j(x_in_j) · W_{K,V}^j. The input to layer j is the output of layer j−1.
- Read-in matrix for `vlm_vec[l]` (output of l): W̃ = [W_K^{l+1}; W_V^{l+1}] · diag(1 + w_ln^{l+1}) (512 × 2048).
  - l = 5 → layer 6, l = 11 → layer 12.
  - l = 17 is the last layer: its output is never read by the expert, so it is structurally null (potent fraction 0).
  - For layer 17 I also report a proxy using layer-17 weights, labelled as a proxy.
- Robustness: also use the layer-l weights (the "K = RMSNorm(x)·W_K at layer l" framing from the task spec).
- Potent subspace P = row space of W̃ (rank r ≤ 512). Secondary: P90 = the top singular directions holding 90 % of ||W̃||_F².
- Metrics for a difference vector d:
  - PF(d) = ||P d||² / ||d||². Chance = r / 2048.
  - GR(d) = (||W̃ d||² / ||d||²) / (||W̃||_F² / 2048), the gain relative to an isotropic random direction. Chance = 1.
- Direction vs scale: RMSNorm divides each token by its rms.
  - Primary: d is computed on rms-normalised segment means, x / rms(x).
  - Secondary: raw d.
  - Separately, Δ log rms is reported as the scale channel.
  - Limitation: the rms is that of the segment mean, not per token. B data fixes this.
- Contrasts at call 0, where the same init gives an identical observation. Validity check: sig_h must be identical between none and semantic/task for the same init.
  - INSTR-para: semantic − none (same suite, task, init).
  - INSTR-goal: task − none (new goal, same init file).
  - INSTR-between: none task i − none task j, same suite, same init index, all pairs. In libero_goal the scene layout is shared; in other suites the scene differs too, so this contrast is labelled "mixed".
  - SCENE: environment − none, position(swap) − none, object − none (same task, same init index).
  - ACT: total variance of the call-0 vectors across all none episodes, trace(P C) / trace(C). This is the anisotropy baseline.
  - Random: 1,000 random r-dimensional subspaces (Haar), giving the PF null distribution.
- Segments: agentview, wrist, task, state. Layers 5, 11, 17.
- **H1 (primary): in the task segment at layers 5 and 11, PF and GR are lower for INSTR-para and INSTR-goal than for SCENE (pooled environment + position + object) and lower than ACT.**
  - Tests: per (layer, segment, contrast) vs SCENE and vs ACT.
  - The "below random chance" test is reported but counts as secondary, because isotropic chance is a weak baseline in an anisotropic residual stream.
- Over time: the same PF for INSTR-goal at calls 0 … 10 (aligned call index, episodes still running). This is descriptive.

### Expert → action output
- Readout for `exp_vec[17]` at denoise step k (t_k = 1 − 0.1 k): W_o = action_out_proj.W[:7] · diag(1 + scale(t_k)), where scale comes from the expert final AdaRMS norm (time_mlp → norm.dense). Potent dim = 7, chance = 7 / 1024.
- Layers 5 and 11 use the same W_o as a logit-lens-style projection, labelled approximate.
- Contrasts: INSTR-goal, INSTR-para, INSTR-between, SCENE (as above, call 0, every k). MOVE = within-episode variance across calls (centred per episode).
- **H2 (Kaufman preparation analogue): at call 0, PF and GR of INSTR-goal and INSTR-between in exp_vec[17] are lower than for MOVE.** Task information is present in the null space before it drives output.
- Information, not only variance: decode task identity (10 tasks per suite, none condition, call 0, leave-one-init-out, shrinkage LDA).
  - Inputs: (a) the 7-d potent projection; (b) 7-d random subspaces of the null space, 100 draws.
  - The same test on VLM task and agentview segments uses the r-d potent projection vs an r-d random null subspace, each reduced to 20 PCs fit on the training fold.
  - Prediction: task identity is decodable at least as well from the null space as from the potent space.

## A2 token-axis trajectories (stage B, design)
- (a) Language tokens: for instruction pairs in the same scene, take the per-token cosine distance and the per-token Kaufman PF (A1 metric) by layer.
  - Prediction: divergence is concentrated at the differing word and spreads to later tokens with depth. The extra divergence lies mostly in the read-in null.
- (b) 50 action tokens: PCA (fit on training-fold tasks) of the token 0 → 49 trajectory, per layer and denoise step.
  - Prediction: the trajectory length grows over denoising and correlates with the executed eef path length over the next 50 steps (Spearman, CV by task).

## A3 Elsayed preparation vs movement subspaces
- Prep = calls before the first grasp (or call 0 alone). Move = calls from grasp to release (carry).
- Top-k PCs per epoch (k chosen for 80 % variance, capped at 10).
- Alignment index = tr(U_prep^T C_move U_prep) / (sum of the top-k eigenvalues of C_move).
- Baseline: Elsayed's covariance-matched random subspaces (1,000 draws).
- Prediction: expert alignment is below the random median (orthogonal reorganisation), and the VLM alignment is higher than the expert's.
- VLM(prep) vs expert(execution) uses CCA-free principal angles after projecting through the read-in map.

## A4 Churchland jPCA
- Axes: action tokens, denoise steps, calls.
- PCA to 6 dims (training fold), then fit dx = M x by least squares, unconstrained vs skew-symmetric. R² is reported on held-out tasks.
- Prediction: R²_skew / R²_full > 0.5 on the action-token axis, < 0.5 on the denoise axis.

## A5 Gallego manifolds
- Per task/condition manifold: top-10 PCs. Principal angles vs a random baseline (covariance-matched).
- CCA of latent dynamics, aligned to time-normalised calls. CCA is fit on half of the inits and evaluated on the other half.
- Predictions:
  - Pick-and-place tasks share latent dynamics: CCA r is higher within pick-and-place than across types.
  - Basket opp/far gives high CCA r with large principal angles: same dynamics, different readout.

## A6 Inagaki multi-noise (stage B)
- 16 chunks per observation.
- Coding direction = the difference of the means of the two trained behaviours (fit on training inits).
- Bimodality: Hartigan dip test plus GMM BIC (1 vs 2 components).
- Prediction: under new or fallback instructions, the chunks are bimodal (discrete), with mode identity predicting the executed behaviour. Noise-induced spread shrinks along the denoising path.

## A7 combinatorial code (Kim et al. 2024; NNLS is my extension, the paper uses CDs)
- Fit cand31 trial states as NNLS combinations of source-task mean states (roles.csv) plus the base task.
- Baseline: the same fit with shuffled source labels (task-label permutation).
- Prediction: successful compositions show weight moving from the object source to the destination source over the episode. Failures stay on one source.
- Also project onto Kim-style source CDs.

## A8 Ames: proposal only, no data.
