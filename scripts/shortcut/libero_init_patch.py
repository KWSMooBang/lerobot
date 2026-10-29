"""Build libero_init_patch.json: lift objects that start inside a fixture in the LIBERO init states.

    PYTHONPATH=src:scripts/shortcut python scripts/shortcut/libero_init_patch.py [--min-pen 0.02]

MuJoCo >= 3.4 (lerobot pins 3.8) reports box-box penetrations about twice as deep as MuJoCo <= 3.3, which
rendered HuggingFaceVLA/libero. A box that starts 2.6 cm inside the table box (milk, orange juice, ketchup in
LIVING_ROOM_SCENE1/2) is now pushed out hard enough to fly off the table during the settle steps; before, it
settled on the table. For every (suite, task, init state) of the four suites this reads the contacts right
after set_init_state (unpatched files, SC_LIBERO_RAW=1) and, for each movable object whose deepest box-box
contact with a non-movable geom (table, fixture) is below -min_pen, raises the object's free joint z by that
depth. common.ensure_libero_config applies the result to a mirror of init_files.

Only box-box contacts count: box-plane contacts did not change (Object-suite objects start up to 3.8 cm into
the floor plane and settle the same under MuJoCo 3.3.2 and 3.8.1). Settled positions under the two versions
(Object + LIBERO-10, 10 init states per task) differ by > 1 cm only for the objects this patch lifts and for
objects they knock over; their box-box penetrations are 2.65-2.9 cm, all others are at most 1.5 cm deep.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco

os.environ["SC_LIBERO_RAW"] = "1"
from common import INIT_PATCH, ensure_libero_config  # noqa: E402

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def object_penetration(inner) -> dict[str, tuple[float, int]]:
    """Deepest box-box contact of each movable object with a non-movable geom → (dist, z state index)."""
    sim = inner.sim
    m, d = sim.model, sim.data
    root_of = {m.body_name2id(obj.root_body): name for name, obj in inner.objects_dict.items()}
    rootid = m._model.body_rootid
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    pen = dict.fromkeys(inner.objects_dict, 0.0)
    for k in range(d.ncon):
        c = d.contact[k]
        if m.geom_type[c.geom1] != box or m.geom_type[c.geom2] != box:
            continue
        r1, r2 = (int(rootid[m.geom_bodyid[g]]) for g in (c.geom1, c.geom2))
        for a, b in ((r1, r2), (r2, r1)):
            if a in root_of and b not in root_of:
                pen[root_of[a]] = min(pen[root_of[a]], float(c.dist))
    out = {}
    for name, obj in inner.objects_dict.items():
        qadr = m.get_joint_qpos_addr(obj.joints[0])[0]
        out[name] = (pen[name], 1 + int(qadr) + 2)  # flattened state = [time, qpos, qvel]; free joint z
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pen", type=float, default=0.02, help="lift objects deeper than this (m)")
    ap.add_argument("--out", type=Path, default=INIT_PATCH)
    ap.add_argument("--report", type=Path, help="also write every object's penetration (tsv)")
    args = ap.parse_args()
    ensure_libero_config()
    from lerobot.envs.libero import LiberoEnv, _get_suite

    files: dict[str, list[dict]] = {}
    rows = []
    for s in SUITES:
        suite = _get_suite(s)
        for t in range(10):
            env = LiberoEnv(
                suite, t, s, obs_type="pixels_agent_pos", observation_height=64, observation_width=64
            )
            env._ensure_env()
            raw_env = env._env
            assert raw_env is not None and env._init_states is not None
            raw_env.reset()
            task = suite.tasks[t]
            rel = f"{task.problem_folder}/{Path(task.init_states_file).name}"
            for i, st in enumerate(env._init_states):
                raw_env.set_init_state(st)
                for name, (p, j) in sorted(object_penetration(raw_env.env).items()):
                    rows.append(f"{s}\t{t}\t{i}\t{name}\t{p:.5f}")
                    if p < -args.min_pen:
                        files.setdefault(rel, []).append(
                            {
                                "init": i,
                                "object": name,
                                "state_index": int(j),
                                "z0": float(st[j]),
                                "dz": round(-p, 6),
                            }
                        )
            env.close()
            n = len(files.get(rel, []))
            print(f"{s} t{t}: {n} lifts", flush=True)
    if args.report:
        args.report.write_text("suite\ttask\tinit\tobject\tpen\n" + "\n".join(rows) + "\n")
    patch = {
        "doc": "objects lifted out of fixtures in the LIBERO init states; see scripts/shortcut/libero_init_patch.py",
        "min_pen": args.min_pen,
        "files": files,
    }
    args.out.write_text(json.dumps(patch, indent=1) + "\n")
    print("wrote", args.out, sum(len(v) for v in files.values()), "lifts in", len(files), "files")


if __name__ == "__main__":
    main()
