"""Does each PRO axis change only what it means to? (PRO venv, CPU, no renderer)

For every (suite, task, axis/level) and init index: object / fixture positions after set_init_state with the
axis' own init file vs the clean suite's init file at the same index (objects matched by name, else by order
in the BDDL — the object axis renames objects). Also: can the axis BDDL be run from the *clean* init states
(same sim-state layout), and is its goal already satisfied at t=0 (after the 10 settle steps) with either init.

    .venvs/libero_pro/bin/python scripts/shortcut/bench/libero_pro/compare_inits.py --out DIR [--inits 10]

DIR/init_compare_objects.csv (one row per object × init), DIR/init_compare_summary.csv (per suite × axis × task).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from bench.libero_pro import catalog  # noqa: E402
from bench.libero_pro.sim import DUMMY_ACTION, NUM_STEPS_WAIT, ensure_libero_config  # noqa: E402

MOVED_M = 0.01  # an object counts as moved when its xy differs by more than 1 cm


def build(bddl: str) -> Any:
    from libero.libero.envs.env_wrapper import ControlEnv

    env = ControlEnv(bddl_file_name=bddl, has_offscreen_renderer=False, use_camera_obs=False)
    env.reset()
    return env


def positions(env: Any, state: np.ndarray) -> dict[str, np.ndarray]:
    env.set_init_state(state)
    e = env.env
    out = {}
    for n, o in {**e.objects_dict, **e.fixtures_dict}.items():
        out[n] = e.sim.data.body_xpos[e.sim.model.body_name2id(o.root_body)].copy()
    return out


def goal_at_start(env: Any, state: np.ndarray) -> bool:
    env.set_init_state(state)
    for _ in range(NUM_STEPS_WAIT):
        env.step(DUMMY_ACTION)
    return bool(env.check_success())


def match(clean: list[str], axis: list[str]) -> dict[str, str]:
    """axis object → clean object: same name, else the same position in the BDDL order."""
    m = {n: n for n in axis if n in clean}
    rest_c = [n for n in clean if n not in m.values()]
    rest_a = [n for n in axis if n not in m]
    m.update(zip(rest_a, rest_c, strict=False))
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inits", type=int, default=10)
    ap.add_argument("--suites", default=",".join(catalog.SUITES))
    args = ap.parse_args(argv)
    ensure_libero_config()
    root = catalog.pro_root()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    obj_rows, sum_rows = [], []
    for suite in args.suites.split(","):
        for tid in range(10):
            ce = catalog.task_entry(root, suite, "none", "original", tid)
            cenv = build(ce["bddl"])
            cinit = catalog.load_init_states(ce["init"])
            cpos = [positions(cenv, cinit[i]) for i in range(args.inits)]
            cenv.close()
            for axis in ("object", "position", "semantic", "task", "environment"):
                for lv in catalog.levels(suite, axis):
                    e = catalog.task_entry(root, suite, axis, lv, tid)
                    ainit = catalog.load_init_states(e["init"])
                    if len(ainit) == 0:
                        continue
                    env = build(e["bddl"])
                    names = list(positions(env, ainit[0]))
                    m = match(list(cpos[0]), names)
                    worst, moved_inits = 0.0, 0
                    for i in range(min(args.inits, len(ainit))):
                        apos = positions(env, ainit[i])
                        moved = 0
                        for n, p in apos.items():
                            c = cpos[i].get(m.get(n, ""))
                            if c is None:
                                continue
                            dxy = float(np.linalg.norm(p[:2] - c[:2]))
                            worst = max(worst, dxy)
                            moved += dxy > MOVED_M
                            obj_rows.append(
                                {"suite": suite, "task_id": tid, "axis": axis, "level": lv, "init": i, "object": n,
                                 "clean_object": m[n], "dx": round(float(p[0] - c[0]), 4),
                                 "dy": round(float(p[1] - c[1]), 4), "dxy": round(dxy, 4)}
                            )  # fmt: skip
                        moved_inits += moved > 0
                    same_layout = len(env.get_sim_state()) == len(cinit[0])
                    clean_ok = trivial_clean = None
                    if same_layout:  # can this axis' BDDL start from the clean suite's init states?
                        try:
                            trivial_clean = goal_at_start(env, cinit[0])
                            clean_ok = True
                        except Exception:  # noqa: BLE001
                            clean_ok = False
                    trivial_pro = goal_at_start(env, ainit[0])
                    env.close()
                    sum_rows.append(
                        {"suite": suite, "task_id": tid, "axis": axis, "level": lv,
                         "inits_with_moved_object": f"{moved_inits}/{min(args.inits, len(ainit))}",
                         "max_dxy_m": round(worst, 4), "same_state_layout": same_layout,
                         "runs_from_clean_inits": clean_ok, "goal_true_at_start_pro_init": trivial_pro,
                         "goal_true_at_start_clean_init": trivial_clean,
                         "objects_renamed": sorted(n for n in m if m[n] != n)}
                    )  # fmt: skip
                    print(
                        f"{suite} t{tid} {axis}/{lv}: moved in {sum_rows[-1]['inits_with_moved_object']} inits, "
                        f"max {worst * 100:.1f} cm, clean-init ok {clean_ok}, goal@0 pro {trivial_pro} clean {trivial_clean}",
                        flush=True,
                    )
    for name, rows in (("init_compare_objects.csv", obj_rows), ("init_compare_summary.csv", sum_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
