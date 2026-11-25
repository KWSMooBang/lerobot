import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import neuro_dyn as n

W = dict(np.load(n.CACHE / "weights.npz"))
meta = json.load(open(n.CACHE / "meta.json"))
vlm = np.load(n.CACHE / "vlm.npy", mmap_mode="r")
exp = np.load(n.CACHE / "exp.npy", mmap_mode="r")
pr = np.array(n.build_pairs(meta)["instr_goal"])
so = np.array([bool(meta[a]["success_orig"]) for a, b in pr])
sp = np.array([bool(meta[a]["success"]) for a, b in pr])
print("n", len(pr), "fallback(success_orig)", so.sum(), "success_new", sp.sum())
Bv, _ = n.row_space(W["readin_12"])
Wo = W["action_out_w"][:7].astype(np.float64) * (1 + W["exp_final_scale"][0])[None]
Bo, _ = n.row_space(Wo)
d_v = n.rmsnorm(np.asarray(vlm[pr[:, 0], 0, 1, 2], np.float64)) - n.rmsnorm(np.asarray(vlm[pr[:, 1], 0, 1, 2], np.float64))
d_e = np.asarray(exp[pr[:, 0], 0, 0, 2], np.float64) - np.asarray(exp[pr[:, 1], 0, 0, 2], np.float64)
out = {}
for name, m in (("fallback", so), ("success_new", sp), ("other", ~so & ~sp)):
    out[name] = {
        "n": int(m.sum()),
        "vlm_L11_task_pf": float(np.median(n.pf(d_v[m], Bv))),
        "vlm_L11_task_relnorm": float(np.median(np.linalg.norm(d_v[m], axis=1) / np.sqrt(2048))),
        "exp_L17_pf": float(np.median(n.pf(d_e[m], Bo))),
        "exp_L17_norm": float(np.median(np.linalg.norm(d_e[m], axis=1))),
    }
print(json.dumps(out, indent=1))
json.dump(out, open(n.OUT / "a1" / "a1_exploratory_fallback_split.json", "w"), indent=1)
