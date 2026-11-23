"""Check --record-activations against the earlier replay extraction (T023 step 1), GPU.

For episodes whose per-call policy inputs were already rebuilt by the replay pipeline (attn_obs.py) and run
through attn_extract.py (libero90 / cand31), feed the same observations call by call through the server's
Engine (exec single: episode seed at reset, batch-1 calls) twice — recording off and on — and compare:
  1. actions: recording on vs off, bit for bit;
  2. recorded features vs the attn_extract.py npz of the same episode (that ran all calls in one batch, so
     batch numerics differ): per key max |diff|, relative max |diff| and Pearson r over all elements.

    CUDA_VISIBLE_DEVICES=1 uv run --extra shortcut-bench python scripts/shortcut/serve/check_recording.py \
        --obs <attn>/obs/<job> --ref <attn>/out/<job> --out <dir> [--limit 4]
Output: <out>/check.json (+ the recorded npz under <out>/rec/).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "serve"), str(SHORTCUT / "parallel")]

from engine import Engine, load_any  # noqa: E402

KEYS = ("attn_mass", "attn_head", "attn_img", "exp_h", "exp_mlp", "vlm_h", "vlm_mlp", "sig_h", "act_pred")


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = np.abs(a - b)
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {
        "max_abs": float(d.max()),
        "rel_max": float(d.max() / max(np.abs(b).max(), 1e-12)),
        "rel_mean": float(d.mean() / max(np.abs(b).mean(), 1e-12)),
        "pearson_r": r,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="lerobot/pi05_libero_finetuned")
    ap.add_argument("--limit", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    loaded = load_any(a.ckpt, None, "libero_goal", 10, "cuda")
    plain = Engine(loaded, "cuda", 10, 1, {})
    files = sorted(Path(a.obs).glob("*.npz"))[: a.limit]
    res: dict = {"episodes": [], "ckpt": a.ckpt}
    # actions without hooks first (the recorder registers hooks at construction)
    ref_actions = {}
    for f in files:
        d = np.load(f)
        plain.reset(f.stem, int(d["seed"]), str(d["instruction"]))
        ref_actions[f.stem] = [
            plain.infer_one(
                f.stem,
                {"image": d["image"][k], "image2": d["image2"][k], "state": d["state"][k]},
                str(d["instruction"]),
                False,
            )
            for k in range(len(d["t"]))
        ]
        plain.end(f.stem)
    rec = Engine(loaded, "cuda", 10, 1, {}, record_dir=str(out / "rec"))
    for f in files:
        d = np.load(f)
        tic = time.perf_counter()
        rec.reset(f.stem, int(d["seed"]), str(d["instruction"]), record_key=f.stem)
        acts = [
            rec.infer_one(
                f.stem,
                {"image": d["image"][k], "image2": d["image2"][k], "state": d["state"][k]},
                str(d["instruction"]),
                False,
            )
            for k in range(len(d["t"]))
        ]
        sec = time.perf_counter() - tic
        path = rec.end(f.stem)
        same = all(np.array_equal(x, y) for x, y in zip(acts, ref_actions[f.stem], strict=True))
        got, ref = np.load(path), np.load(Path(a.ref) / f.name)
        ep = {
            "name": f.stem,
            "calls": len(acts),
            "actions_bit_identical_on_vs_off": same,
            "sec_with_recording": round(sec, 2),
            "npz_mb": round(Path(path).stat().st_size / 1e6, 3),
            "vs_replay": {
                k: compare(got[k], ref[k][..., : got[k].shape[-1]] if k == "attn_img" else ref[k])
                for k in KEYS
                if k in ref.files
            },
            "ranges": {k: [float(np.abs(got[k].astype(np.float32)).max())] for k in ("exp_vec", "vlm_vec")},
        }
        res["episodes"].append(ep)
        print(
            json.dumps(
                {k: ep[k] for k in ("name", "calls", "actions_bit_identical_on_vs_off", "sec_with_recording")}
            ),
            flush=True,
        )
    res["all_bit_identical"] = all(e["actions_bit_identical_on_vs_off"] for e in res["episodes"])
    (out / "check.json").write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {
                k: {
                    kk: round(np.median([e["vs_replay"][k][kk] for e in res["episodes"]]), 5)
                    for kk in ("rel_max", "rel_mean", "pearson_r")
                }
                for k in res["episodes"][0]["vs_replay"]
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
