"""T035 check: run one episode's prefix pass N times; are vlm_seg (segment means, T028 reduction) identical run to run?

If repeats differ from each other by 1 f16 ulp, the residual non-bit-equality with T028 comes from nondeterministic
CUDA scatter_add (atomic float adds in the segment mean), not from the attention hooks.

  CUDA_VISIBLE_DEVICES=1 python vlm_attn_repeat_check.py basket_far ours-obj-custom-train10 ctl_t09_train_i05 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlm_attn_extract as VA  # noqa: E402, N812


def main() -> None:
    set_, job, ep, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    row = next(r for r in VA.read_manifest() if (r["set"], r["job"], r["episode"]) == (set_, job, ep))
    L, tok = VA.load_model("cuda")  # noqa: N806
    hooks = VA.VlmAttnHooks(L.policy)
    seg_runs, i2p_runs, infos = [], [], []
    orig = hooks.stop

    def stop():  # keep vlm_seg of each sub-batch
        out, vs = orig()
        segs.append(vs.cpu().numpy())
        return out, vs

    hooks.stop = stop
    for _ in range(n):
        segs: list[np.ndarray] = []
        ep_arr, _fz, info = VA.run_episode(row, L, hooks, tok, "cuda", 32, False)
        seg_runs.append(np.concatenate(segs))
        i2p_runs.append(ep_arr["i2p"])
        infos.append({k: info[k] for k in ("vlm_seg_bit_equal", "vlm_seg_max_diff", "vlm_seg_frac_equal")})
    res = {
        "episode": f"{set_}/{job}/{ep}",
        "vs_T028": infos,
        "vlm_seg_f32_repeat_equal": [bool(np.array_equal(seg_runs[0], s)) for s in seg_runs[1:]],
        "vlm_seg_f32_repeat_max_diff": [float(np.abs(seg_runs[0] - s).max()) for s in seg_runs[1:]],
        "vlm_seg_f16_repeat_equal": [bool(np.array_equal(seg_runs[0].astype(np.float16), s.astype(np.float16)))
                                     for s in seg_runs[1:]],  # fmt: skip
        "i2p_repeat_equal": [bool(np.array_equal(i2p_runs[0], s)) for s in i2p_runs[1:]],
    }
    print(json.dumps(res, indent=1))
    (VA.SMALL / "logs" / "repeat_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
