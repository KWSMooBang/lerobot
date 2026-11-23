# T026 pre-registration (fixed before any candidate result was computed) — 2026-11-23

Space: per feature (expert, gemma, siglip, attention, patch_attention, action), T025 fixed PCA-50
(z-score + PCA fitted on the 3,000 train10 calls only; cache `8_phase_traj/_cache/space_G_<feat>.npz`, key P50).
All distances are Euclidean in these 50 dims. Training trials used for source segments: successful ones only.

Role segments of a training trial (calls, from `8_phase_traj/events/events_Goal.csv`):
- P (pick / carry): first grasp call → first release call (inclusive).
- L (place): release − 2 → release.
- H (handle / contact, tasks without a grasp or the drawer part of G4): c − 1 → c + 3, c = first contact call
  with the goal object (G1 cabinet, G6 plate, G8 stove) or, for G4's drawer, the first cabinet-contact call.
  Clipped to [0, 29].

Normaliser s_k (within-task spread of training task k): median over calls 0–29 and over pairs of successful trials of
task k of the same-call distance. For a pair of sources (a, b): s = (s_a + s_b) / 2.

**A1 (primary, fixed here):** for a candidate with component sources A (role segment rA) and B (role segment rB)
from `roles.csv`: for every pair (trial i of A, trial j of B), m_ij = min over calls p ∈ rA(i), q ∈ rB(j) of
‖x_i,p − x_j,q‖. A1 = median_ij(m_ij) / s. If A or B lists several training tasks, A1 is the minimum over the
(a, b) combinations (the closest available pair); that combination is also used for A1c and A2.
Reported with the median argmin calls (p, q).
A1c (crossing length): per pair, number of calls p ∈ rA(i) with min_q ‖x_i,p − x_j,q‖ / s < 1; median over pairs.
Prediction tested: smaller A1 (and larger A1c) → higher new-task success. Main feature: patch_attention.

A2 start overlap: same pairs, same-call distance at call 0 (A2_0), and mean same-call distance over calls
0..min(bound_i, bound_j) (A2_pre; bound = first grasp call, or contact call if no grasp), both / s.
A3 destination exclusivity: for each point of B's role segment, nearest-neighbour distance to any call of any trial
of another training task divided by NN distance to any call of another trial of task B; A3 = median (>1 = exclusive).
A4 dominant pull of the base task(s): s_base / median_k s_k (tightness), and, with task mean trajectories μ_k,
d(k,l) = mean_c ‖μ_k,c − μ_l,c‖ / ((s_k + s_l)/2); A4_min = min_l d(base, l), A4_mean = mean_l d(base, l).
Several base tasks → average.
A5 patch-location overlap (feature-independent): attn_img, layer group 0 (early layers), agentview patches 0–255,
mean map over successful trials and role-segment calls of A and of B, minus the train10 mean map over all calls;
A5 = cosine(A map, B map). Also both cameras (A5_both).

Candidates with no second training source (g01, g06, g22, g30: novel drawer/action) have A1/A2/A5 = NA.
Label: success = new + new_byproduct trials (task_judgment.csv) ≥ 3/10; continuous rate for Spearman.
Main set: 29 candidates (g18, g28 excluded, reported separately). g01, g22 (and g13 per cand31 table) flagged
infeasible, kept. Stats: Spearman with 10,000-permutation two-sided p; success − failure group median difference
with 10,000 bootstrap CI. One LOO single-threshold check, on A1 patch_attention only.
