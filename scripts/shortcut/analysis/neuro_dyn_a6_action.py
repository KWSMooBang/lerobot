"""A6 robustness: the same within-observation vs between-group spread in action space (mn_act, 16 seeds)."""
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import neuro_dyn_b as nb

rows = nb.manifest()
out = {}
fams = {
    "cand": ([r for r in rows if r["set"] == "goal" and r["condition"] == "cand"], lambda r: 0 if r["outcome"] in nb.FALLBACK else (1 if r["outcome"] in nb.NEW else None)),
    "basket_far": ([r for r in rows if r["condition"] in ("far20", "far25")], lambda r: 0 if r["success"] == 1 else (1 if r["outcome"] == "released outside" else None)),
    "lscene": ([r for r in rows if r["condition"] == "lscene"], lambda r: 0 if r["outcome"] == "trained" else (1 if r["outcome"] == "nothing" else None)),
}
for name, (tr, lf) in fams.items():
    X, lab = [], []
    for r in tr:
        a = nb.load(r, ["mn_act"])["mn_act"].astype(np.float64)  # 4, 16, 50, 7
        X.append(a[..., :3].sum(2))  # 4, 16, 3: summed translation over the chunk (planned displacement)
        lab.append(lf(r))
    X = np.array(X)
    within = np.linalg.norm(X - X.mean(2, keepdims=True), axis=-1).mean(2)  # trial, slot
    lab = np.array([np.nan if l is None else l for l in lab])
    ok = np.isfinite(lab)
    m0 = X[ok & (lab == 0)].mean((0, 2))
    m1 = X[ok & (lab == 1)].mean((0, 2))
    between = np.linalg.norm(m1 - m0, axis=-1)  # slot
    across_trials = np.linalg.norm(X.mean(2) - X.mean(2).mean(0), axis=-1).mean(0)  # slot
    out[name] = {"within_obs_mean_dist": float(np.median(within)), "between_group_mean_dist_by_slot": between.tolist(),
                 "across_trial_spread_by_slot": across_trials.tolist(),
                 "within_over_across": float(np.median(within) / np.median(across_trials))}
print(json.dumps(out, indent=1))
json.dump(out, open(nb.OUT / "b" / "a6_action_space_robustness.json", "w"), indent=1)
