#!/usr/bin/env python
"""T034 run-3 checks (CPU): (a) VLM cache rows == expert cache rows (episode, frame, index); (b) T015 per-layer SAEs
reach their T015 VE on 20 000 random rows of the new pooled_all (PREREG §5: within 0.03; pg17 reported only).
Exit 1 on failure. Out: /outputs/VLA-shortcut-exps/sae/crosscoder/collection_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from cc_data import cache_dir, check  # noqa: E402
from model import TopKSAE  # noqa: E402

T015 = Path("/data0/woosung/datasets/VLA-shortcut-exps/sae/vlm")
OUT = Path("/outputs/VLA-shortcut-exps/sae/crosscoder/collection_check.json")


def main() -> None:
    torch.set_num_threads(32)
    res: dict = {"keys": check()}
    d = cache_dir("vlm")
    info = json.loads((d / "info.json").read_text())
    x = np.load(d / "x.npy", mmap_mode="r")
    rows = np.sort(np.random.default_rng(0).choice(x.shape[0], 20000, replace=False))
    per = {}
    ok = res["keys"]["rows_match"] and res["keys"]["all_even_frames"]
    for li, nm in enumerate(info["layers"]):
        sae = TopKSAE.load(T015 / nm)
        xs = torch.from_numpy(np.asarray(x[rows, li], np.float32))
        with torch.no_grad():
            m = sae.metrics(xs)
        ref = json.loads((T015 / nm / "metrics.json").read_text())["var_explained"]
        passed = m["var_explained"] >= ref - 0.03
        per[nm] = {"ve_new": m["var_explained"], "ve_t015": ref, "pass": bool(passed)}
        if nm != "pg17":
            ok &= passed
        print(nm, per[nm], flush=True)
    res["t015_sae_ve"] = per
    res["pass"] = bool(ok)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
