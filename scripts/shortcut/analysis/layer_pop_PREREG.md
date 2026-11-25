# T030 layer_pop: pre-registration

Written 2026-11-25, before any analysis number was computed.
- Checks done beforehand: array shapes, keys, label joins (rich manifest × T027 episode cache) and the extraction cache.
- No activation statistic, decoding score or distance was looked at.

## Data and scope
- **Source:** T028 rich re-extraction, 1,560 episodes / 51,100 calls. All episodes were replayed on one 4090 (`replay_machine`).
  - Comparisons that need the same original machine pair episodes within a machine:
    - basket A100: near / opp vs the A100 libero3 O-suite ctl
    - basket 4090: far20 / far25 vs ctl4090
  - CPU only. No re-extraction.
- **Labels:** joined from the T027 episode cache (`rsa_decoding/_cache/eps_*.pkl`, key = task or variant + init):
  - events: `gc`, `rcs`, `bound_call` (first grasp, else first contact)
  - motion vectors `motion_act`, `motion`
  - first-frame pixels `pix`
  - executed task: `exec`, `exec_other`, `exec_L`, `nearest_trained`, `base`
  - semantics (verb / object / destination): `semantics.csv`
  - roles: `roles.csv`
  - identical-observation cand↔base pairs: T029 `pairs.json` (valid ∧ rich-available, 150 pairs)
- **U40** = the 400 training-task episodes: libero3 (S/O/L) + goal train (G), 40 tasks × 10 inits. Every U40 episode has ≥ 28 calls.

## Representations (per layer; per call)
| id | source | dim | layers |
|---|---|---|---|
| `v_task`, `v_agent`, `v_wrist`, `v_state` | `vlm_seg` segment mean (task / agentview / wrist / state) | 2048 | 0–17 |
| `v_all6` | `vlm_seg`, all 6 segments concatenated | 12288 | 0–17 |
| `e_s0`, `e_s4`, `e_s9` | `exp_mean`, action-token mean at denoise step 0 / 4 / 9 | 1024 | 0–17 |
| `m_s0`, `m_s4`, `m_s9` | `exp_mlp_mean`, MLP units (down_proj input) | 4096 | 2,5,8,11,14,17 |
| `et_s0`, `et_s9` | `exp_tok_pc`: 50 action tokens × top-32 T028 PCs | 1600 | 2,5,8,11,14,17 |
| `vt` | `vlm_tok_pc`: first 6 + last 6 task-segment tokens × top-32 T028 PCs | 384 | 0,2,5,8,11,14,17 |

- Primary expert step is **9** (last denoise step, closest to the output). Steps 0 and 4 are secondary.
- **Magnitude-only counterparts (T021 definitions), same layer:**
  - VLM segment s at layer l: [‖h‖_{l,s}, mean|MLP|_{l,s}] (`vlm_h`, `vlm_mlp`), 2-d. For `v_all6`: all 6 segments, 12-d.
  - Expert layer l, step k: [`exp_h`[k,l], `exp_mlp`[k,l]], 2-d. Used for `e_sk`, `m_sk` and `et_sk`.
  - `vt`: the task-segment magnitude at that layer.
- **All-layer magnitude (the T021/T027 feature itself):**
  - VLM segment: 36-d. `v_all6`: 216-d.
  - Expert step k: 36-d.
  - This is the strongest magnitude-only baseline.

## Normalisation (applies to every analysis unless stated)
- **Main:** per-channel z-score, (x − μ_j) / σ_j.
  - μ, σ are fitted on the reference episodes = U40, all calls. U40 is the training distribution shared by every analysis set; all sets were replayed on the same 4090.
  - Decoding refits the z-score inside each training fold.
  - Channels with σ < 1e-6 are set to 0.
- **Massive channels, rule fixed now:**
  - For each (representation, layer), compute each channel's raw RMS over U40 calls, √mean(x_j²).
  - A channel is massive if its RMS > 20 × the median channel RMS.
  - Massive channels are set to 0 (excluded) in the main analysis. They are listed with RMS, max|x| and the ratio in `numbers.json`.
  - Expected: VLM dims 1577 and 1959 in late layers / template segment. The rule is applied blindly to every representation.
- **Robustness — "what the next layer sees":**
  - x / rms(x) per segment-mean vector, massive channels kept.
  - For VLM layers 0–16, multiplied by (1 + w_ln^{l+1}) (`vlm/L{l+1}/input_layernorm`). For VLM 17 and the expert, plain x / rms(x); the expert's AdaRMS affine depends on the time embedding and is not stored.
  - Then the same analyses (dim, decoding at calls 0 / 10 / 20, divergence) are repeated without z-scoring.
  - Limitation: RMSNorm acts per token, but these are segment / token means.
- **Representations already in T028 PCA coordinates** (`et_*`, `vt`): z-scored the same way. The rule is not applied; PCs have no massive channel.

## Analyses
Statistical units and corrections are stated per analysis. BH-FDR (q = 0.05) runs per family, and the number of tests is reported.

### 1. Per-layer trajectories
- **PCA:** per (representation, layer), fit on U40 z-scored calls only. Plot PC1–PC2.
- **Cases** (each shown as condition means of time-normalised trajectories, 20 bins, plus thin single trials):
  - (a) basket **O9**: A100 ctl (libero3 O9), near, opp; 4090 ctl4090, far20, far25
  - (b) goal cand **g10** with training **G7** and **G2**
  - (c) L90 lscene fallback: the lscene task with the most `outcome = trained` trials whose `exec_L` is a single L task (tie → lowest task id), vs the training trials of that L task
  - (d) Spatial **S1** and Long **L1**
- **Markers:** time gradient (light → dark); ○ start, × first grasp (`gc`, mean bin), ■ end.
- **Depth panels:**
  - VLM (`v_task`, `v_agent`, `v_all6`) and expert (`e_s9`): all 18 layers
  - MLP (`m_s9`): 6 layers
- **Shape metrics per layer** (U40, z-space, massive excluded):
  - (i) time-variance fraction = mean within-episode variance across calls ÷ total variance
  - (ii) path length per call ÷ √(total variance / D), i.e. normalised speed
  - (iii) straightness = end-to-end distance ÷ path length
  - (iv) variance in PC1–2
- Descriptive. No test.

### 2. Effective dimensionality
- **Measures** per (representation, layer), U40 calls:
  - participation ratio PR = (Σλ)² / Σλ²
  - variance captured by the top 1 / 5 / 10 / 50 PCs
  - number of PCs for 90 %
- **Views:**
  - main: z-scored, massive excluded
  - also: raw, and the RMSNorm view
- PR is also computed separately for between-episode (episode means) and within-episode (episode-centred) covariance.
- Descriptive.

### 3. Decoding maps
- **Classifier:** StandardScaler → PCA(min(50, n_train − 1), fit on the training fold) → StandardScaler → LogisticRegression(C = 1). Same as T027.
- **Calls:** c ∈ {0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 24, 27} (absolute call index; all U40 episodes have ≥ 28 calls).
- **Targets on U40:**
  - `task40`: 40-way, leave-one-init-out
  - `within`: 10-way per suite, leave-one-init-out, mean over the 4 suites
  - `suite`: 4-way, leave-one-init-out
  - `verb`, `object`, `destination`: primary label from semantics.csv; labels shared by ≥ 2 tasks; CV = GroupKFold(10) by **task**, i.e. generalisation to unseen tasks
  - `phase`: reach / grasp / carry / release, T027 per-call definition; all calls pooled; leave-one-init-out. Reported as a layer × phase recall map plus balanced accuracy.
- **Cross-generalisation:** train on all 400 U40 episodes at call c, apply to fallback trials at the same call.
  - **cand FALLBACK:** cand31 trials with a non-empty `exec_other` (T027 definition).
    - executed = prediction ∈ `exec_other`
    - instructed = prediction == `base` (the instruction's base task)
  - **L90 lscene trained:** trials with `outcome = trained`, `exec_L` non-empty, `nearest_trained` ∉ `exec_L` (T027 definition).
    - executed = prediction ∈ `exec_L`
    - instructed = prediction == `nearest_trained`
  - **Object / destination decoders:** applied to the same pools, restricted to trials where the instructed label (semantics of the trial's own instruction) and the executed task's label are both in the decoder's label set and differ.
    - Share = prediction equals the instructed vs the executed label.
- **Output:** layer × call maps per representation. Magnitude counterparts are computed identically.

### 4. Semantics / motion / scene shares (T027 pair regression per layer × phase)
- **Sets** (T027 `bset`, same blocks):
  - `L90_lscene`: lscene trials + L training trials of the same scenes
  - `cand31`: cand + G training
  - `basket_4090`
  - `basket_A100`
- **Episode vector:** 5 bins before + 5 after `bound_call`, for `pre` / `post` / `traj`, in z-space.
- **Regression:** feature distance ~ sem + motion + scene. Standardised β; pairs only within a block.
- **Motion:** `motion_act` (primary here, because T027 showed the eef-position motion RDM is dominated by scene layout).
- **Scene:** `pix` distance. Sem: T027 `sem`.
- **p-values:** Mantel permutation within blocks, 199 permutations.
- **Representations:** `v_task`, `v_agent`, `v_all6`, `e_s9` (all 18 layers), `m_s9` (6 layers), plus the same-layer magnitude counterparts.
- **Family:** all β_sem and β_motion p-values.

### 5. Divergence maps
- **basket, same machine, same task·init:**
  - A100: near / opp vs ctl (libero3 O)
  - 4090: far20 / far25 vs ctl4090
- **cand vs base:** 150 identical-observation pairs (T029 `pairs.json`), split by outcome FALLBACK (`trained`, `fail_trained`) vs NEW (`new`, `new_byproduct`).
- **Ratio:** at call c, d(pair) ÷ baseline.
  - Baseline = mean distance of the reference trial (ctl / base) to the other-init trials of the same task at call c.
  - Euclidean distance in z-space, massive excluded.
  - Per-layer × call map = median over pairs. Calls 0 … 27.
- **Divergence onset per layer:** the first call where the median ratio ≥ 0.5. "Diverges first" = the earliest onset; ties are broken by the larger call-0 ratio.
- **Magnitude counterparts:** z-scored [‖h‖, |MLP|] of that layer, same ratio.
- **Tests:**
  - cand: Wilcoxon over cand variants (variant means) of the call-0 ratio, FALLBACK vs NEW, per layer (`v_task`, `e_s9`).
  - Mann–Whitney between FALLBACK and NEW variant means. Family BH.

### 6. Kaufman read-in, all VLM layers
- **Read-in of layer l's output (l = 0 … 16):**
  - W̃_{l+1} = [W_K; W_V]^{l+1} · diag(1 + w_ln^{l+1}) (512 × 2048).
  - Potent space = its row space. PF chance = r / 2048 (0.25).
  - **Layer 17's output is never read by the expert: PF ≡ 0 by structure.** A proxy with layer-17 weights is reported and labelled as a proxy.
- **Difference vectors at call 0,** RMS-normalised segment means, massive channels kept (the weights decide what is read). Segments: task, agentview, wrist, state.
  - INSTR: cand − base, 150 identical-observation pairs
  - SCENE: basket moved − ctl, same machine, same task·init (near, opp, far20, far25)
  - ACT baseline: trace(P C) / trace(C) of U40 call-0 vectors
- **Tests per (layer, segment):**
  - INSTR PF vs SCENE PF: Mann–Whitney on cluster means; clusters = cand variant / basket task × condition
  - INSTR PF − ACT: Wilcoxon over cand variants
  - Family BH.

### 7. Gallego and Kim per layer
- **Basket manifolds** per (set, condition, task):
  - Manifold = top-10 PCs of the z-scored calls of that unit.
  - Mean principal angle to the reference, minus the mean split-half (even vs odd inits) self angle.
  - CCA r: 4 components, fit on even inits, evaluated on odd inits; time-normalised 20 bins.
  - Reference = **near** (A100) for far20, far25, opp and ctl4090 (T029 convention). Secondary same-machine reference: ctl4090 for far20 / far25.
  - Representations: `e_s9`, `v_all6`, `v_task` (18 layers), `m_s9` (6).
- **"Same dynamics, different readout" holds at layer l** if both:
  - CCA_r(far) ≥ CCA_r(ctl4090) − 0.1
  - median over tasks of (angle excess far − angle excess ctl4090) > 0, Wilcoxon over the 10 tasks, q < 0.05
  - Reported: the first layer where it holds for both far20 and far25, and whether it holds for opp.
- **Goal cand (Kim / NNLS, T029 A7 per layer)** on `e_s9` (18 layers), `v_all6`, `v_task`, `m_s9`:
  - R² of the source-task combination minus a random 3-task baseline. Wilcoxon over cand variants.
  - Object → destination weight slope, NEW vs FALLBACK: permutation within variant.

### 8. Magnitude-only vs vector
At the same layer, vector − magnitude for:
- (a) decoding accuracy, mean over calls, per target
- (b) β_sem (`v_task`) and β_motion (`e_s9`)
- (c) divergence: call-0 ratio and onset

Also vector at its best layer vs the all-layer magnitude feature.

## Hypotheses (confirmatory)
- **H1 (instruction in the VLM task segment peaks in middle layers; executed share rises later)**
  - **H1a:** the instruction-only divergence ratio (cand vs base at call 0, `v_task`) is larger in the middle layers (6–11, mean) than in early layers (0–5) and than in late layers (12–17).
    - Two Wilcoxon tests over cand variants (variant-mean ratios).
    - Descriptive companion: the layer of peak `within`-Goal decoding at call 0 from `v_task`.
  - **H1b:** for fallback trials (cand FALLBACK + L90 lscene trained, pooled), the cross-generalised task decoder from `v_task` has a larger (executed − instructed) share in late layers (12–17) than in middle layers (6–11).
    - Averaged over calls ≥ 4 in the grid.
    - Wilcoxon over trials of the per-trial difference.
- **H2 (vector beats magnitude at every layer):** for every (representation, layer, target) of analysis 3 on U40, the call-mean accuracy of the vector exceeds that of the same-layer magnitude feature.
  - Per (representation family, target): Wilcoxon over layers.
  - The share of cells where vector > magnitude is reported. "Every layer" holds only if the share is 100 %.
- **H3 (executed behaviour is readable in the expert from the first calls, at every layer):** for each expert layer 0–17 (`e_s9`), over calls {0, 1, 2}, the cross-generalised task decoder's executed share > instructed share on the pooled fallback trials.
  - Wilcoxon over trials of per-trial (executed hits − instructed hits).
  - 18 tests, BH. H3 holds if all 18 are q < 0.05 in the predicted direction.
- Analyses 1, 2, 5 (onset), 6, 7 also answer the spec's questions. Their tests are listed above; anything added after results is labelled exploratory.

## Statistics notes
- Cluster / unit choices are as stated per analysis.
- Same-machine pairing for basket. cand / base pairs share one machine (A100) and one replay machine.
- CPU workers ≤ 8, BLAS single-threaded per worker.
