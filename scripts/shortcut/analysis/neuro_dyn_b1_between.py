"""B1 exploratory: between-candidate vs within-candidate structure of the K/V difference."""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

B = "/outputs/VLA-shortcut-exps/analysis/neuro_dyn/b/"
r = json.load(open(B + "b1.json"))
pp = r["per_pair"]
FB, NW = ("trained", "fail_trained"), ("new", "new_byproduct")
out = {}
for key in ("v_rel", "k_rel", "resid_rel"):
    v = {"FALLBACK": [np.mean(p[key][1:]) for p in pp if p["outcome"] in FB], "NEW": [np.mean(p[key][1:]) for p in pp if p["outcome"] in NW]}
    out[key + "_unstratified_mwu_p"] = float(mannwhitneyu(v["FALLBACK"], v["NEW"]).pvalue)
    # candidate level: mean difference size vs share of NEW outcomes (all 10 inits)
    byc = defaultdict(list)
    for p in pp:
        byc[p["variant"]].append(p)
    xs = [np.mean([np.mean(p[key][1:]) for p in q]) for q in byc.values()]
    ys = [np.mean([p["outcome"] in NW for p in q]) for q in byc.values()]
    rho = spearmanr(xs, ys)
    out[key + "_candidate_level_spearman"] = {"rho": float(rho.correlation), "p": float(rho.pvalue), "n_cand": len(xs)}
    # within-candidate: candidates that have both outcomes
    w = []
    for q in byc.values():
        a = [np.mean(p[key][1:]) for p in q if p["outcome"] in NW]
        b = [np.mean(p[key][1:]) for p in q if p["outcome"] in FB]
        if a and b:
            w.append(np.mean(a) - np.mean(b))
    out[key + "_within_cand_new_minus_fallback"] = {"n_cand_with_both": len(w), "median": float(np.median(w)) if w else None}
print(json.dumps(out, indent=1))
json.dump(out, open(B + "b1_exploratory_between_within.json", "w"), indent=1)
