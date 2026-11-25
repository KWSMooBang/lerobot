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

---
# Addendum B (2026-11-25, written before any stage-B number; data seen: manifest columns, keys and shapes, and one design check of call-0 obs equality for cand vs base)

## Data and common rules
- T028 rich_acts: `/outputs/VLA-shortcut-exps/rich_acts/<set>/<job>/<episode>.npz` plus manifest.csv, pca_bases.npz, weights.npz.
  - Code is written on the pilot (60 ep) and run on all 1,560 ep after T028 [DONE].
  - The pilot has raw `vlm_tok` / `exp_tok`. These are projected with pca_bases to match the final `*_tok_pc` keys.
- **Held-out rule for the T028 PCA bases**: episodes with pca_fit = 1 are used only as the fitting fold. Confirmatory statistics use pca_fit = 0 episodes. Analysis-specific PCA, CCA and LDA are fit on the training fold (by init or by task).
- **Machine**: every activation was replayed on the 4090 (replay_machine). Every comparison is between episodes with the same replay_machine, so this holds trivially.
  - Outcome labels come from the original runs. These are A100, except basket_far: same machine, bit-identical.
  - Replays are teacher-forced on the recorded observations.
- **Massive activations** (VLM residual dims 1577 and 1959, up to 2.9e5):
  - (i) Every residual-space distance, PCA or CCA uses per-token RMS-normalised vectors (x / rms(x), which is what K/V see), with dims 1577 and 1959 removed. Their share of the difference energy is reported separately.
  - (ii) Kaufman read-in PF and GR use the full normalised vector, because the weights decide what is read. A variant with the two dims zeroed is reported as robustness.
  - (iii) K/V-level metrics need no treatment (post-norm, absmax < 62).
- Families for BH-FDR: B1, A2, A3, A4, A5, A6, A7. The number of tests per family is reported.
- Cluster units: candidate (cand), task (train sets), condition × task (basket).

## B1 (confirmatory retest of the stage-A exploratory finding)
- **Design**: each goal-cand trial is paired with the base-task (roles.csv base_task) train trial at the same init.
  - A pair is valid only if call-0 obs are identical within render jitter: image max |Δ| ≤ 2, state max |Δ| < 1e-3. Candidates with a modified scene fail this.
- **Groups**:
  - FALLBACK = outcome ∈ {trained, fail_trained}
  - NEW = {new, new_byproduct}
  - partial, fail_new and fail_other are reported, not tested.
- **Metrics at call 0, task segment**:
  - (a) Read-in PF of Δ (exact per-token rms, then segment mean) at tok layers 0, 2, 5, 8, 11, 14, each read by layer l+1. From vlm_seg, all 18 layers (approximate) are also shown.
  - (b) K/V level: ||Δk_seg|| / ||k_seg|| and ||Δv_seg|| / ||v_seg||, task segment, all 18 layers. Expert attention mass on task tokens (attn_mass), step 0.
  - (c) Expert: ||Δ exp_mean[L17, step 0]|| and its PF on the action readout (7 rows × AdaRMS scale at t = 1).
- **H_B1**:
  - (a) and (b) do not differ between FALLBACK and NEW: two one-sided tests (TOST) with margin = 0.25 pooled SD, plus the usual two-sided test.
  - (c) is smaller in FALLBACK (both norm and PF).
- Test: Mann–Whitney U, with p from a permutation that keeps candidate strata (labels shuffled within candidate, 10,000 permutations).
- Secondary (unpaired; LIBERO-90 lscene, which has no identical-obs partner): attention-weighted value read from task tokens, trained vs nothing.

## A2 token axis
- (a) **Language**: valid cand–base pairs (identical obs).
  - Tokens are aligned with difflib on lang_ids. Matched tokens are the same word in both instructions; unmatched tokens are the differing words.
  - For matched tokens: cosine distance (normalised, massive dims removed) per layer, as a function of signed distance to the nearest differing token, and per-token read-in PF.
  - Predictions:
    - Because the prefix attention is bidirectional, matched tokens on both sides of the differing word diverge, and divergence grows with depth.
    - The matched-token Δ has PF ≤ the PF of the differing-token Δ.
  - L90 lscene vs part (same scene, different init) is reported as confounded and descriptive only.
- (b) **Action tokens**: exp_tok PCs (L17, steps 0, 4, 9).
  - Per call: path length over tokens 0 → 49 in the top-10 PCs.
  - Prediction: path length grows from step 0 to step 9. At step 9, the Spearman ρ with the predicted chunk's path length (den_x[-1][:, :7] cumulative |Δ|) is > 0.3, and with the executed eef path length over the next 10 steps it is > 0.
  - Between-condition distance of mean token trajectories: descriptive.

## A3 Elsayed
- Prep = calls < grasp_call (or contact_call). Move = grasp_call ≤ call ≤ release_call (or the end).
- Sets: libero3, goal-train, basket (all conditions).
- Signals: expert exp_mean L17 step 9, VLM vlm_seg L11 (agentview and task, normalised, massive dims removed).
- Epoch-wise covariance (pooled over calls of pca_fit = 0 episodes, mean-centred per epoch). k = dims for 80 % of variance, capped at 10.
- AI = tr(U_p^T C_m U_p) / Σ top-k λ(C_m). Baseline: 1,000 covariance-matched random subspaces (Elsayed 2016).
- Prediction: expert AI < the 5th percentile of the random distribution, and VLM AI > expert AI.
- VLM(prep) → expert(move): ridge prediction (CV by task) of the move-epoch mean expert state from the call-0 VLM state vs from the call-0 expert state. Descriptive.

## A4 jPCA
- Axes:
  - tokens: exp_tok PCs, L17, step 9
  - denoise: exp_vec L17 over 10 steps
  - calls: exp_vec L17 step 9
- 6 PCs (training fold). Fit dx = M x unconstrained vs skew-symmetric. R² on held-out tasks.
- Rotation frequency = the largest |Im λ(M_skew)| / 2π per unit.
- Prediction: R²_skew / R²_full > 0.5 on tokens, < 0.5 on the denoise axis. The call axis is not predicted.
- Condition differences: descriptive (per set / condition).

## A5 Gallego
- Per-task / per-condition manifold: top-10 PCs of exp_mean L17 step 9 (calls pooled).
- Principal angles between manifolds vs (i) covariance-matched random subspaces and (ii) the split-half same-task angle.
- CCA of latent dynamics:
  - Time-normalise to 20 bins; condition-mean on inits of split A; project onto each manifold; CCA fit on A, evaluated on split B.
- Predictions:
  - Pick-and-place pairs (semantics.csv verb = place/put) have a higher mean CCA r than pairs of different types.
  - Basket opp/far vs near: CCA r is within 0.1 of the near-split-half r, while the principal angles exceed the split-half angles. That is the same dynamics with a different readout.

## A6 Inagaki (multi-noise)
- 16 noise samples at 4 probe calls (mn_exp_last, mn_act).
- Coding direction (CD) per family, fit on a training fold of inits:
  - cand: FALLBACK minus NEW trial mean, at the same probe slot
  - basket far: success vs released-outside
  - lscene: trained vs nothing
- Projection is normalised so that 0 = mean of group 1 and 1 = mean of group 2 (test fold).
- Metrics:
  - pooled Hartigan dip test
  - GMM BIC (1 vs 2 components)
  - per-observation share of samples within 0.25 of an endpoint vs in the middle
  - within-observation spread vs between-endpoint distance
- Baseline: train-instruction trials on a CD between two trained tasks.
- Prediction: new-instruction trials are discrete (bimodal, mostly at the endpoints), and the per-trial share on the NEW side predicts the outcome (AUC > 0.6, CV by candidate).
- Convergence along the denoise path cannot be measured across seeds, because only final chunks are stored per seed. This is reported as a limitation, plus the single-seed den_x path.

## A7 combinatorial (Kim 2024 CD idea + NNLS extension)
- cand states (exp_mean L17 step 9, and VLM L11 task / agentview, normalised) per call.
- Sources = goal-train centroids, time-normalised to 20 bins, for base_task, A_tasks and B_tasks (roles.csv).
- NNLS weights per call. Baseline: the same fit with 3 random non-source train tasks (100 draws).
- Kim-style: projection on the source CD (A − B), fit on half of the train inits.
- Predictions:
  - Source weights > random-source weights.
  - In successful compositions (NEW), the weight moves from the object source to the destination source over the episode; in FALLBACK the base task dominates throughout.
