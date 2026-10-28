"""Check a LIBERO-PRO install (PRO venv): every (suite, axis, level, task) has its bddl + init files, the
task axis' original goal is evaluable in the task scene (its objects exist there), one env builds.

    .venvs/libero_pro/bin/python scripts/shortcut/bench/libero_pro/check_install.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from bench.libero_pro import catalog  # noqa: E402
from bench.libero_pro.sim import ensure_libero_config, load_init_states  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    ensure_libero_config()
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from libero.libero.envs.env_wrapper import ControlEnv

    root = catalog.pro_root()
    out: dict = {"bench_version": catalog.bench_version(root), "axes": {}, "task_axis": {}}
    bad = 0
    for suite in catalog.SUITES:
        for axis in catalog.AXES:
            for lv in catalog.levels(suite, axis):
                ok, n_init = 0, set()
                for t in range(10):
                    try:
                        e = catalog.task_entry(root, suite, axis, lv, t)
                    except FileNotFoundError as err:
                        print(f"MISSING {err}")
                        bad += 1
                        continue
                    ok += 1
                    n_init.add(len(load_init_states(e["init"])))
                    if axis == "task":  # objects + their regions / sites, as the running env sees them
                        new = robosuite_parse_problem(e["bddl"])
                        orig = robosuite_parse_problem(e["orig_bddl"])
                        env = ControlEnv(
                            bddl_file_name=e["bddl"], has_offscreen_renderer=False, use_camera_obs=False
                        )
                        names = set(env.env.object_states_dict)
                        env.close()
                        missing = sorted({n for s in orig["goal_state"] for n in s[1:]} - names)
                        out["task_axis"][f"{suite}/t{t}"] = {
                            "original": e["instruction_original"],
                            "task": e["instruction_given"],
                            "goal_original": orig["goal_state"],
                            "goal_task": new["goal_state"],
                            "orig_goal_missing_objects": missing,
                        }
                out["axes"][f"{suite}/{axis}/{lv}"] = {"tasks": ok, "inits": sorted(n_init)}
    for k, v in out["axes"].items():
        print(f"{k:<40} tasks {v['tasks']:>2}  inits {v['inits']}")
    ta = out["task_axis"]
    evaluable = sum(not v["orig_goal_missing_objects"] for v in ta.values())
    same = sum(v["goal_original"] == v["goal_task"] for v in ta.values())
    print(f"task axis: {len(ta)} tasks, original goal evaluable in {evaluable}, goal unchanged in {same}")
    e = catalog.task_entry(root, "libero_goal", "task", "task", 8)
    env = ControlEnv(bddl_file_name=e["bddl"], has_offscreen_renderer=False, use_camera_obs=False)
    env.reset()
    print(f"env ok: {e['dir']}/{e['task_name']} → {env.language_instruction!r}")
    env.close()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
