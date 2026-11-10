"""Variant spec → BDDL + init states + manifest (LIBERO, CPU only: no renderer is created).

Steps for one spec (`build(spec)`):
1. Base task: BDDL and init states as the eval env loads them. `ensure_libero_config()` runs first, so the
   T009 env fix applies: init states come from the fixed init mirror (L10 lifts) and the manifest records
   `libero_env` (run_eval refuses a variant built for another env tag).
2. BDDL: base BDDL with :language, :goal, :obj_of_interest replaced and the scene edits applied to
   :objects / :regions / :init (move → On obj <region>, swap → placements exchanged, remove → object and
   its regions dropped, add → new object + region, fixture_state → (Open|Close|TurnOn|TurnOff x)).
3. Init states (`init:`):
   keep      base init states unchanged (only the language / goal differ; scene must be identical);
   patch     every base init state is loaded into the base scene, every joint that also exists in the
             variant scene is copied over by name (robot, objects, drawers, knobs), then the edits are
             applied (object xy, fixture joints) and the scene settles for SETTLE_STEPS no-op steps;
   resample  LIBERO's own placement sampler (env.reset with seed i) + fixture joints + settle, N times.
4. Check every state the way the eval loads it (fresh reset, set_init_state, 10 no-op steps as lerobot's
   LiberoEnv): moved objects within POS_TOL of their target xy, untouched objects within POS_TOL of
   where they rest in the base scene, nothing fell off the support surface, fixture predicates hold,
   :init placements hold (resample). States that fail are dropped and listed in the manifest
   (`init_states.source` maps each kept state to the base init it came from).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess  # nosec B404
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .bddl_io import Problem, predicate_str
from .goals import TURNOFF_SEMANTICS, eval_predicate
from .spec import VariantSpec, generated_dir, load_spec

GENERATOR_VERSION = "variants-v1"  # bump when generated files change meaning
SETTLE_STEPS = 20  # no-op steps after editing a state, before it is stored
EVAL_SETTLE_STEPS = 10  # lerobot LiberoEnv.num_steps_wait
POS_TOL = 0.025  # m, xy distance for "is where the edit put it" / "untouched object did not move"
FALL_TOL = 0.05  # m below its resting height = fell off / sank
DUMMY = [0.0] * 6 + [-1.0]
FIXTURE_PRED = {"open": "open", "close": "close", "on": "turnon", "off": "turnoff"}
FIXTURE_RANGE = {
    "open": "default_open_ranges",
    "close": "default_close_ranges",
    "on": "default_turnon_ranges",
    "off": "default_turnoff_ranges",
}
REPO = Path(__file__).resolve().parents[3]


class BuildError(RuntimeError):
    pass


# ---------------------------------------------------------------- LIBERO access


@cache
def lib() -> SimpleNamespace:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import ensure_libero_config, libero_env_tag

    ensure_libero_config()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    return SimpleNamespace(
        benchmark=benchmark,
        get_libero_path=get_libero_path,
        ControlEnv=ControlEnv,
        env_tag=libero_env_tag(),
    )


@cache
def suite_tasks(suite: str) -> list[dict[str, Any]]:
    """[{task_id, name, language, bddl, init_file}] in the suite's task order (lerobot's order)."""
    lb = lib()
    s = lb.benchmark.get_benchmark_dict()[suite]()
    root = Path(lb.get_libero_path("bddl_files"))
    init_root = Path(lb.get_libero_path("init_states"))
    return [
        {
            "task_id": i,
            "name": t.name,
            "language": t.language,
            "bddl": str(root / t.problem_folder / t.bddl_file),
            "init_file": str(init_root / t.problem_folder / t.init_states_file),
        }
        for i, t in enumerate(s.tasks)
    ]


def base_task_id(spec: VariantSpec) -> int:
    tasks = suite_tasks(spec.suite)
    if isinstance(spec.base_task, int) or str(spec.base_task).isdigit():
        i = int(spec.base_task)
        if not 0 <= i < len(tasks):
            raise BuildError(f"{spec.id}: base_task {i} not in 0..{len(tasks) - 1}")
        return i
    for t in tasks:
        if spec.base_task in (t["name"], t["language"]):
            return t["task_id"]
    raise BuildError(
        f"{spec.id}: base_task {spec.base_task!r} is not a task id, name or sentence of {spec.suite}"
    )


def load_init_states(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    import torch

    return np.asarray(torch.load(p, weights_only=False))  # nosec B614 (LIBERO's own numpy pickles)


def control_env(bddl: str | Path) -> Any:
    """LIBERO env without renderer or camera observations (fast, CPU)."""
    return lib().ControlEnv(
        bddl_file_name=str(bddl),
        use_camera_obs=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        hard_reset=False,
    )


def joint_qpos(e: Any) -> dict[str, np.ndarray]:
    m, d = e.sim.model, e.sim.data
    return {
        m.joint_id2name(j): np.array(d.get_joint_qpos(m.joint_id2name(j)), dtype=np.float64).reshape(-1)
        for j in range(m.njnt)
    }


def obj_pos(e: Any, name: str) -> np.ndarray:
    return np.array(e.sim.data.body_xpos[e.obj_body_id[name]], dtype=np.float64)


def eval_settle(env: Any, state: np.ndarray, seed: int) -> Any:
    """Load a state like the eval does: reset (seeded) → set_init_state → 10 no-op steps."""
    env.seed(seed)
    env.reset()
    env.set_init_state(state)
    for _ in range(EVAL_SETTLE_STEPS):
        env.step(DUMMY)
    return env.env


def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()  # nosec B607
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- BDDL


def suite_categories(suite: str) -> dict[str, str]:
    """{category: a task whose BDDL uses it} for every movable object category of the suite."""
    out: dict[str, str] = {}
    for t in suite_tasks(suite):
        for cat in Problem.read(t["bddl"]).objects.values():
            out.setdefault(cat, t["name"])
    return out


def support_regions(p: Problem) -> dict[str, dict[str, Any]]:
    """Regions on the table / floor (where objects can be moved or added)."""
    return {k: r for k, r in p.regions.items() if r["target"] in p.fixtures and r["ranges"]}


def region_center(r: dict[str, Any]) -> list[float]:
    x0, y0, x1, y1 = r["ranges"][0]
    return [(x0 + x1) / 2, (y0 + y1) / 2]


def surface_fixture(p: Problem) -> str:
    for name, cat in p.fixtures.items():
        if cat in ("table", "floor") or name in ("main_table", "floor"):
            return name
    raise BuildError("scene has no table / floor fixture")


def make_bddl(spec: VariantSpec, base: Problem, base_task: dict[str, Any]) -> tuple[Problem, dict[str, Any]]:
    """Edited BDDL + the resolved edits ({object: target xy in region frame}, fixture joints, ...)."""
    p = base.copy()
    p.set_language(spec.instruction)
    names = base.names()
    sup = support_regions(base)
    surface = surface_fixture(base)
    resolved: dict[str, Any] = {"targets": {}, "removed": [], "added": {}, "fixtures": {}, "swapped": []}

    def where(entry: dict[str, Any], what: str) -> tuple[str, list[float]]:
        if "to" in entry:
            reg = entry["to"]
            if reg not in sup:
                by_short = {r["short"]: k for k, r in sup.items()}
                if reg in by_short:
                    reg = by_short[reg]
                else:
                    raise BuildError(
                        f"{spec.id}: {what} {entry['object']} to {entry['to']!r}: not a table/floor region "
                        f"(known: {sorted(sup)})"
                    )
            return reg, region_center(sup[reg])
        x, y = entry["xy"]
        short = f"{entry['object']}_{what}_region"
        reg = p.add_region(short, surface, [[x - 0.005, y - 0.005, x + 0.005, y + 0.005]])
        return reg, [x, y]

    for name in spec.scene.remove:
        if name not in base.objects:
            raise BuildError(
                f"{spec.id}: remove {name}: not a movable object (objects: {sorted(base.objects)})"
            )
        resolved["removed"].append({"object": name, "regions": p.remove_object(name)})
    for m in spec.scene.move:
        obj = m["object"]
        if obj not in base.objects:
            raise BuildError(f"{spec.id}: move {obj}: not a movable object (objects: {sorted(base.objects)})")
        reg, xy = where(m, "moved")
        p.set_placement(obj, reg)
        resolved["targets"][obj] = {"xy": xy, "region": reg}
    for a, b in spec.scene.swap:
        for o in (a, b):
            if o not in base.objects:
                raise BuildError(
                    f"{spec.id}: swap {o}: not a movable object (objects: {sorted(base.objects)})"
                )
        pa, pb = base.placement(a), base.placement(b)
        if pa is None or pb is None:
            raise BuildError(f"{spec.id}: swap {a} {b}: both need an On/In placement in :init")
        p.set_placement(a, pb[2], pb[0])
        p.set_placement(b, pa[2], pa[0])
        resolved["swapped"].append([a, b])  # xy taken from the base state per init (patch)
    cats = suite_categories(spec.suite) if spec.scene.add else {}
    for add in spec.scene.add:
        obj, cat = add["object"], add["category"]
        if cat not in cats:
            raise BuildError(
                f"{spec.id}: add {obj}: category {cat!r} is not used in {spec.suite} (known: {sorted(cats)})"
            )
        if not obj.startswith(cat):
            raise BuildError(f"{spec.id}: add {obj}: LIBERO object names are <category>_<k>, e.g. {cat}_1")
        p.add_object(obj, cat)
        reg, xy = where(add, "added")
        p.set_placement(obj, reg)
        resolved["added"][obj] = {"category": cat, "xy": xy, "region": reg}
    for name, state in spec.scene.fixture_state.items():
        if name not in names:
            raise BuildError(f"{spec.id}: fixture_state {name}: not in the scene")
        p.set_fixture_state(name, state)
        resolved["fixtures"][name] = state

    scene_names = p.names()
    for g in spec.goal:
        missing = [n for n in g[1:] if n not in scene_names]
        if missing:
            raise BuildError(f"{spec.id}: goal {predicate_str(g)} names {missing}, not in the variant scene")
    p.set_goal(spec.goal)
    interest = [n for g in spec.goal for n in g[1:] if n in p.objects] or [
        x for x in base.obj_of_interest if x in p.objects
    ]
    p.set_obj_of_interest(interest)
    resolved["base_task"] = base_task["task_id"]
    return p, resolved


# ---------------------------------------------------------------- track goals


def resolve_track_goals(spec: VariantSpec, variant: Problem) -> dict[str, dict[str, Any]]:
    """{name: {goal, task, instruction, absent}}; `absent` = names the variant scene lacks (not tracked)."""
    tasks = suite_tasks(spec.suite)
    items: list[tuple[str, int | None, list[list[str]]]] = []
    if spec.track_goals == "default":
        items = [(f"t{t['task_id']}", t["task_id"], Problem.read(t["bddl"]).goal) for t in tasks]
    else:
        assert isinstance(spec.track_goals, list)
        for tg in spec.track_goals:
            if tg.task is not None:
                if not 0 <= tg.task < len(tasks):
                    raise BuildError(f"{spec.id}: track_goals task {tg.task} not in the suite")
                items.append((tg.name, tg.task, Problem.read(tasks[tg.task]["bddl"]).goal))
            else:
                items.append((tg.name, None, tg.goal or []))
    names = variant.names()
    out = {}
    for name, task, goal in items:
        goal = [[g[0], *g[1:]] for g in goal]
        out[name] = {
            "goal": goal,
            "task": task,
            "instruction": tasks[task]["language"] if task is not None else None,
            "absent": sorted({n for g in goal for n in g[1:] if n not in names}),
        }
    return out


# ---------------------------------------------------------------- init states


def fixture_value(e: Any, name: str, state: str) -> float:
    """Middle of the object's default range for `state` (LIBERO articulation properties)."""
    st = e.object_states_dict[name]
    parent = getattr(st, "parent_name", None) or name
    props = e.get_object(parent).object_properties["articulation"]
    lo, hi = props[FIXTURE_RANGE[state]]
    return float((lo + hi) / 2)


def apply_fixture_states(e: Any, fixtures: dict[str, str]) -> None:
    for name, state in fixtures.items():
        e.object_states_dict[name].set_joint(fixture_value(e, name, state))


def set_xy(e: Any, name: str, xy_world: np.ndarray) -> None:
    jn = e.objects_dict[name].joints[-1]
    q = np.array(e.sim.data.get_joint_qpos(jn), dtype=np.float64)
    q[:2] = xy_world
    e.sim.data.set_joint_qpos(jn, q)


def settle(env: Any, n: int = SETTLE_STEPS) -> None:
    for _ in range(n):
        env.step(DUMMY)


def flat_state(e: Any) -> np.ndarray:
    return np.array(e.sim.get_state().flatten(), dtype=np.float64)


def patch_states(
    spec: VariantSpec, resolved: dict[str, Any], base_env: Any, var_env: Any, base_states: np.ndarray
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """One edited state per base init state + the intended pose of every edited object."""
    out: list[np.ndarray] = []
    intents: list[dict[str, Any]] = []
    for i, s in enumerate(base_states):
        base_env.set_init_state(s)
        be = base_env.env
        qpos = joint_qpos(be)
        ref = {n: obj_pos(be, n) for n in be.objects_dict}
        var_env.seed(i)
        var_env.reset()
        ve = var_env.env
        d = ve.sim.data
        m = ve.sim.model
        for j in range(m.njnt):
            jn = m.joint_id2name(j)
            if jn in qpos and np.size(d.get_joint_qpos(jn)) == qpos[jn].size:
                d.set_joint_qpos(jn, qpos[jn] if qpos[jn].size > 1 else float(qpos[jn][0]))
        d.qvel[:] = 0
        off = np.asarray(ve.workspace_offset, dtype=np.float64)[:2]
        intent: dict[str, Any] = {}
        for obj, t in resolved["targets"].items():
            xy = off + np.asarray(t["xy"])
            set_xy(ve, obj, xy)
            intent[obj] = {"xy": xy.tolist(), "z": float(ref[obj][2])}
        for a, b in resolved["swapped"]:
            set_xy(ve, a, ref[b][:2])
            set_xy(ve, b, ref[a][:2])
            intent[a] = {"xy": ref[b][:2].tolist(), "z": float(ref[a][2])}
            intent[b] = {"xy": ref[a][:2].tolist(), "z": float(ref[b][2])}
        for obj, t in resolved["added"].items():
            intent[obj] = {"xy": (off + np.asarray(t["xy"])).tolist(), "z": None, "tol": 0.02}
        apply_fixture_states(ve, resolved["fixtures"])
        ve.sim.forward()
        settle(var_env)
        out.append(flat_state(ve))
        intents.append(intent)
    return out, intents


def resample_states(
    spec: VariantSpec, resolved: dict[str, Any], var_env: Any, n: int, seed0: int = 10_000
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    out: list[np.ndarray] = []
    intents: list[dict[str, Any]] = []
    for i in range(n):
        var_env.seed(seed0 + i)
        var_env.reset()
        ve = var_env.env
        apply_fixture_states(ve, resolved["fixtures"])
        ve.sim.forward()
        settle(var_env)
        out.append(flat_state(ve))
        intents.append({})
    return out, intents


def check_state(
    var_env: Any,
    state: np.ndarray,
    seed: int,
    intent: dict[str, Any],
    resolved: dict[str, Any],
    base_rest: dict[str, np.ndarray] | None,
    init_preds: list[list[str]] | None,
) -> list[str]:
    """Problems of one stored state, loaded like the eval loads it (empty = ok)."""
    e = eval_settle(var_env, state, seed)
    errs = []
    for obj, want in intent.items():
        pos = obj_pos(e, obj)
        dxy = float(np.linalg.norm(pos[:2] - np.asarray(want["xy"])))
        tol = want.get("tol", POS_TOL)
        if obj in resolved["added"]:
            tol = max(tol, 0.03)
        if dxy > tol:
            errs.append(f"{obj}: {dxy * 100:.1f} cm from its target xy")
        if want.get("z") is not None and pos[2] < want["z"] - FALL_TOL:
            errs.append(f"{obj}: fell ({pos[2]:.3f} m, rests at {want['z']:.3f} m)")
    if base_rest is not None:
        for obj, p0 in base_rest.items():
            if obj in intent or obj not in e.objects_dict:
                continue
            pos = obj_pos(e, obj)
            d = float(np.linalg.norm(pos - p0))
            if d > POS_TOL:
                errs.append(f"{obj} (untouched) moved {d * 100:.1f} cm vs the base scene")
    for name, fstate in resolved["fixtures"].items():
        if not eval_predicate(e, [FIXTURE_PRED[fstate], name]):
            errs.append(f"{name}: not {fstate} after loading")
    for p in init_preds or []:
        if p[0].lower() in ("on", "in") and not eval_predicate(e, p):
            errs.append(f"init {predicate_str(p)} does not hold")
    return errs


# ---------------------------------------------------------------- build


def build(spec: VariantSpec, variants_root: str | Path | None = None, force: bool = False) -> dict[str, Any]:
    """Write <generated>/<id>.bddl, init_states.npy, manifest.json; returns the manifest."""
    lb = lib()
    out = generated_dir(spec.suite, spec.id, variants_root)
    mpath = out / "manifest.json"
    if mpath.exists() and not force:
        old = json.loads(mpath.read_text())
        if (
            old.get("spec_sha256") == spec.sha256
            and old.get("libero_env") == lb.env_tag
            and old.get("generator") == GENERATOR_VERSION
        ):
            return old
    tid = base_task_id(spec)
    bt = suite_tasks(spec.suite)[tid]
    base = Problem.read(bt["bddl"])
    variant, resolved = make_bddl(spec, base, bt)
    out.mkdir(parents=True, exist_ok=True)
    bddl_path = out / f"{spec.id}.bddl"
    bddl_path.write_text(variant.text())
    base_states = load_init_states(bt["init_file"])
    same_scene = variant.scene_key() == base.scene_key()

    log: dict[str, Any] = {"mode": spec.init_mode}
    if spec.init_mode == "keep":
        if not same_scene:
            raise BuildError(f"{spec.id}: init keep but the BDDL scene differs from the base task")
        states = [np.asarray(s, dtype=np.float64) for s in base_states]
        source: list[int | None] = list(range(len(states)))
        failures: list[dict[str, Any]] = []
    else:
        try:
            var_env = control_env(bddl_path)
            for k in range(3):  # the eval resets (= samples :init) before loading a state: must work
                var_env.seed(k)
                var_env.reset()
        except Exception as e:
            if type(e).__name__ != "RandomizationError":
                raise
            raise BuildError(
                f"{spec.id}: LIBERO's placement sampler cannot place the variant :init ({e}); the eval env "
                "samples it on every reset. Two objects in one region / overlapping spots? Move to a free spot."
            ) from e
        base_env = control_env(bt["bddl"])
        base_env.seed(0)
        base_env.reset()
        try:
            if spec.init_mode == "patch":
                cand, intents = patch_states(spec, resolved, base_env, var_env, base_states)
                src: list[int | None] = list(range(len(cand)))
            else:
                assert spec.init_n is not None
                cand, intents = resample_states(spec, resolved, var_env, spec.init_n)
                src = [None] * len(cand)
            states, source, failures = [], [], []
            init_preds = variant.init if spec.init_mode == "resample" else None
            for k, (s, intent) in enumerate(zip(cand, intents, strict=True)):
                base_rest = None
                if spec.init_mode == "patch":
                    be = eval_settle(base_env, base_states[k], 1000 + k)
                    base_rest = {n: obj_pos(be, n) for n in be.objects_dict}
                errs = check_state(var_env, s, 1000 + k, intent, resolved, base_rest, init_preds)
                if errs:
                    failures.append({"candidate": k, "source_init": src[k], "errors": errs})
                else:
                    states.append(s)
                    source.append(src[k])
        finally:
            var_env.close()
            base_env.close()
        log.update(n_candidates=len(cand), n_failed=len(failures))
        if not states:
            raise BuildError(f"{spec.id}: no init state passed the check; first failure: {failures[:1]}")

    arr = np.stack(states).astype(np.float64)
    init_path = out / "init_states.npy"
    with open(init_path, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    track = resolve_track_goals(spec, variant)
    base_goal = base.goal
    manifest = {
        "id": spec.id,
        "suite": spec.suite,
        "group": spec.group,
        "generator": GENERATOR_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "libero_env": lb.env_tag,
        "spec_path": os.path.relpath(spec.path, out) if spec.path else None,
        "spec_sha256": spec.sha256,
        "spec": spec.to_json(),
        "instruction": spec.instruction,
        "goal": spec.goal,
        "goal_achievable": bool(spec.goal),
        "goal_semantics": TURNOFF_SEMANTICS,
        "base_task": {
            "task_id": tid,
            "name": bt["name"],
            "language": bt["language"],
            "bddl": bt["bddl"],
            "init_file_sha256": sha256_file(bt["init_file"]),
            "goal": base_goal,
            "goal_absent": sorted({n for g in base_goal for n in g[1:] if n not in variant.names()}),
        },
        "track_goals": track,
        "edits": {
            "kinds": spec.scene.kinds(),
            "resolved": resolved,
            "dimension": spec.perturbation_dimension(),
        },
        "scene_changed": not same_scene,
        "files": {
            "bddl": {"path": bddl_path.name, "sha256": sha256_file(bddl_path)},
            "init_states": {"path": init_path.name, "sha256": sha256_file(init_path)},
        },
        "init_states": {
            "mode": spec.init_mode,
            "n": int(arr.shape[0]),
            "state_dim": int(arr.shape[1]),
            "source": source,
            "check": {"failures": failures, **log},
        },
    }
    old = json.loads(mpath.read_text()) if mpath.exists() else {}
    if old.get("feasibility") and old["feasibility"].get("for") == _feas_key(manifest):
        manifest["feasibility"] = old["feasibility"]  # same files → keep the earlier check
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    manifest["_dir"] = str(out)
    return manifest


def _feas_key(m: dict[str, Any]) -> str:
    """Feasibility results stay valid while the BDDL and init states are byte-identical."""
    return m["files"]["bddl"]["sha256"][:16] + "/" + m["files"]["init_states"]["sha256"][:16]


def build_path(
    path: str | Path, variants_root: str | Path | None = None, force: bool = False
) -> dict[str, Any]:
    return build(load_spec(path), variants_root, force)
