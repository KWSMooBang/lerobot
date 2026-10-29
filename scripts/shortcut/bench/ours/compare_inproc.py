"""Check the policy-server path against run_eval.py's in-process loop on the same episodes.

    uv run python scripts/shortcut/bench/ours/compare_inproc.py --inproc RUN_DIR [RUN_DIR ...] \
        --server BENCH_RUN_DIR [--out result.json]

Pairs episodes on (suite, task, condition, variant, init, repeat, seed) and compares success_orig,
success_new, t_orig, t_new, first_grasp_object, n_steps and the per-step logs (actions, eef positions).
With a single-mode server every action should be bit-identical; batched runs are a different sample (T004).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from bench.core.records import read_jsonl  # noqa: E402

FIELDS = ("success_orig", "success_new", "t_orig", "t_new", "first_grasp_object", "n_steps")


def load(dirs: list[Path]) -> dict[tuple, tuple[dict, Path]]:
    out = {}
    for d in dirs:
        for f in [d / "episodes.jsonl", *sorted(d.glob("jobs/*/episodes.jsonl"))]:
            for r in read_jsonl(f):
                k = (
                    r["suite"],
                    r["task_id"],
                    r["condition"],
                    r["variant"],
                    r["init_idx"],
                    r["repeat_idx"],
                    r["seed"],
                )
                out[k] = (r, f.parent)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inproc", nargs="+", required=True)
    ap.add_argument("--server", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    a, b = load([Path(x) for x in args.inproc]), load([Path(x) for x in args.server])
    keys = sorted(set(a) & set(b), key=str)
    rows, field_agree = [], dict.fromkeys(FIELDS, 0)
    bit_identical = 0
    for k in keys:
        (ra, da), (rb, db) = a[k], b[k]
        same = {f: ra.get(f) == rb.get(f) for f in FIELDS}
        for f, ok in same.items():
            field_agree[f] += ok
        sa, sb = np.load(da / ra["steps_file"]), np.load(db / rb["steps_file"])
        n = min(len(sa["action"]), len(sb["action"]))
        act_equal = sa["action"].shape == sb["action"].shape and np.array_equal(sa["action"], sb["action"])
        bit_identical += act_equal
        diff = np.abs(sa["action"][:n] - sb["action"][:n])
        first = int(np.argmax(diff.max(axis=1) > 0)) if n and diff.max() > 0 else None
        rows.append(
            {
                "key": list(k),
                **{f"same_{f}": v for f, v in same.items()},
                "actions_bit_identical": bool(act_equal),
                "max_abs_action_diff": float(diff.max()) if n else 0.0,
                "first_diff_step": first,
                "eef_max_diff_m": float(np.abs(sa["eef_pos"][:n] - sb["eef_pos"][:n]).max()) if n else 0.0,
            }
        )
    summary = {
        "paired_episodes": len(keys),
        "only_inproc": len(set(a) - set(b)),
        "only_server": len(set(b) - set(a)),
        "fields_agree": field_agree,
        "actions_bit_identical": bit_identical,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2))


if __name__ == "__main__":
    main()
