"""Feasibility of a built variant with T002's scripted (privileged-state) policy, on its first init states.

Plan = T002 `feasibility.make_plan` (open / turn on first, pick-and-place or push in goal order, close
last) + two primitives added here: `Up x` (pick and hold x up) and `TurnOff x` (push the knob back).
Success = the variant goal (variants.goals, TurnOff = not TurnOn) holds at any step, as LIBERO counts it.

status:
  feasible     the scripted policy reached the goal from at least one init state
  infeasible   it never did, with primitives T002 validated (pick-place, push, drawer, turn on); like
               T002's feasible=false this is a scripted-policy result, not a proof
  unknown      it never did and the plan needs a primitive T002 did not validate (Up, TurnOff), or a
               predicate the scripted policy has no primitive for → needs teleop
  n/a          the variant has no achievable goal (goal: [])
Also recorded per init: whether the new goal / each tracked goal already holds after the eval settle
(a goal true at step 0 makes success meaningless; the fallback table leaves those goals out).
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[1]
VALIDATED = {"on", "in", "open", "close", "turnon"}  # T002 primitives
NEW_PRIMITIVES = {"up", "turnoff"}
N_INITS = 3


def _feas() -> Any:
    if str(SHORTCUT) not in sys.path:
        sys.path.insert(0, str(SHORTCUT))
    import feasibility

    return feasibility


def _scripted_cls() -> type:
    fm = _feas()
    from .goals import PredicateGoal

    class VariantScripted(fm.Scripted):  # type: ignore[name-defined]
        """T002 scripted policy; success is the variant goal (not the BDDL check), plus Up / TurnOff."""

        def __init__(self, env: Any, goal: list[list[str]]):
            super().__init__(env)
            self.goal_fn = PredicateGoal(goal)
            self.goal_hit = False

        def step(self, *a: Any, **k: Any) -> None:
            super().step(*a, **k)
            if self.goal_fn(self.env):
                self.goal_hit = True

        def up(self, name: str) -> bool:
            for attempt in range(3):
                if self.pick(name, self.table_z + 0.3, skip=attempt):
                    self.hold(30)
                    return True
                self.gripper(False, 5)
                self.home()
            return False

        def turn_off(self, stove: str) -> bool:
            """turn_on with the push direction reversed (knob qpos down to its rest position)."""
            j = self._hinge_of(stove)
            body = self.mm.jnt_bodyid[j]
            zax = self.d.xmat[body].reshape(3, 3) @ self.mm.jnt_axis[j]
            self.gripper(True, 10)
            for it in range(8):
                if self._jq(j) < 0.3:
                    break
                anchor = self.d.xanchor[j].copy()
                handle = [
                    g for g in self.geoms_of_bodies([body]) if self.d.geom_xpos[g][2] > anchor[2] + 0.015
                ]
                pts = self.geom_corners(handle)
                xy = pts[:, :2] - anchor[:2]
                _w, v = np.linalg.eigh(np.cov(xy.T))
                major = v[:, 1]
                proj = xy @ major
                zmid = float((pts[:, 2].max() + pts[:, 2].min()) / 2 + 0.005)
                ends = [anchor[:2] + major * (proj.max() - 0.012), anchor[:2] + major * (proj.min() + 0.012)]
                end = ends[it % 2]
                rv = np.array([end[0] - anchor[0], end[1] - anchor[1], 0.0])
                tang = -np.cross(zax, rv)
                tang /= np.linalg.norm(tang) + 1e-9
                th = self.nearest_theta(float(np.arctan2(tang[1], tang[0])) + np.pi / 2)
                rot = fm.down_rot(th)
                pre = end - tang[:2] * 0.03
                self.move([pre[0], pre[1], zmid + 0.08], rot, max_steps=150)
                self.move([pre[0], pre[1], zmid], rot, max_steps=80, speed=0.4)
                for _k in range(4):
                    cur = self.eef
                    rv = np.array([cur[0] - anchor[0], cur[1] - anchor[1], 0.0])
                    tg = -np.cross(zax, rv)
                    tg /= np.linalg.norm(tg) + 1e-9
                    self.move(cur + tg * 0.025, rot, max_steps=30, speed=0.4)
                self.lift_to(zmid + 0.08, rot)
            return self._jq(j) < 0.5

    return VariantScripted


def plan_for(goal: list[list[str]], instruction: str) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """(before, T002 plan, after): turn-offs first, then T002's plan, Up last (lift and hold)."""
    fm = _feas()
    std = [[g[0].lower(), *g[1:]] for g in goal if g[0].lower() not in NEW_PRIMITIVES]
    verb = instruction.split()[0] if instruction else ""
    before = [("turnoff", g[1]) for g in goal if g[0].lower() == "turnoff"]
    after = [("up", g[1]) for g in goal if g[0].lower() == "up"]
    return before, fm.make_plan(std, verb), after


def run_trial(job: dict[str, Any]) -> dict[str, Any]:
    """One scripted rollout: (variant BDDL, init state) → goal reached?"""
    from .generate import lib, load_init_states

    lib()  # LIBERO config (T009 env) before the env is built
    fm = _feas()
    from .goals import PredicateGoal

    t0 = time.time()
    out = {"init": job["init"]}
    try:
        state = load_init_states(job["init_states"])[job["init"]]
        env = fm._make_env(job["bddl"])
        env.set_init_state(state)
        for _ in range(10):
            env.step([0, 0, 0, 0, 0, 0, -1.0])
        goal_fn = PredicateGoal(job["goal"])
        out["goal_at_init"] = bool(goal_fn(env))
        out["track_at_init"] = sorted(
            n for n, g in job["track"].items() if not g["absent"] and PredicateGoal(g["goal"])(env)
        )
        pol = _scripted_cls()(env, job["goal"])
        before, plan, after = plan_for(job["goal"], job["instruction"])
        out["plan"] = [list(p) for p in before + plan + after]
        with contextlib.suppress(fm.EpisodeDoneError):
            for st in before:
                pol.turn_off(st[1])
                pol.home()
            fm.run_plan(pol, plan)
            for st in after:
                pol.up(st[1])
        out["success_at_end"] = bool(goal_fn(env))
        out["success"] = bool(pol.goal_hit or out["success_at_end"])
        out["steps"] = pol.t
        env.close()
    except Exception as e:  # noqa: BLE001
        out.update(
            success=False,
            success_at_end=False,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}",
        )
    out["sec"] = round(time.time() - t0, 1)
    return out


def status_of(goal: list[list[str]], trials: list[dict[str, Any]]) -> str:
    if not goal:
        return "n/a"
    if any(t["success"] for t in trials):
        return "feasible"
    kinds = {g[0].lower() for g in goal}
    if kinds - VALIDATED:
        return "unknown"
    return "infeasible"


def jobs_for(m: dict[str, Any], n_inits: int = N_INITS) -> list[dict[str, Any]]:
    from .spec import manifest_file

    n = min(n_inits, m["init_states"]["n"])
    return [
        {
            "id": m["id"],
            "suite": m["suite"],
            "init": i,
            "bddl": str(manifest_file(m, "bddl")),
            "init_states": str(manifest_file(m, "init_states")),
            "goal": m["goal"],
            "instruction": m["instruction"],
            "track": m["track_goals"],
        }
        for i in range(n)
    ]


def summarize(m: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    from .generate import _feas_key

    trials = sorted(trials, key=lambda t: t["init"])
    track_init: dict[str, int] = {}
    for t in trials:
        for n in t.get("track_at_init", []):
            track_init[n] = track_init.get(n, 0) + 1
    return {
        "status": status_of(m["goal"], trials),
        "trials": f"{sum(t['success'] for t in trials)}/{len(trials)}" if m["goal"] else "0/0",
        "method": "T002 scripted policy (feasibility.py) + Up/TurnOff primitives (variants/check.py)",
        "goal_at_init": [bool(t.get("goal_at_init")) for t in trials],
        "track_at_init": track_init,
        "rollouts": trials,
        "for": _feas_key(m),
        "checked_at": datetime.now(UTC).isoformat(),
    }


def check(
    manifests: list[dict[str, Any]], workers: int = 8, n_inits: int = N_INITS, log: Any = print
) -> None:
    """Run the trials of every manifest (parallel), write `feasibility` into each manifest.json."""
    from .generate import _feas_key

    todo = []
    for m in manifests:
        if m.get("feasibility", {}).get("for") == _feas_key(m):
            log(f"[check] {m['suite']}/{m['id']}: {m['feasibility']['status']} (cached)")
            continue
        if not m["goal"]:
            # still load the scene once: goals true at init are worth knowing for probes too
            todo += jobs_for(m, 1)
        else:
            todo += jobs_for(m, n_inits)
    results: dict[str, list[dict[str, Any]]] = {}
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(max(1, min(workers, len(todo))), maxtasksperchild=4) as pool:
            for job, r in zip(todo, pool.imap(run_trial, todo), strict=True):
                results.setdefault(f"{job['suite']}/{job['id']}", []).append(r)
                tag = "OK " if r["success"] else "-- "
                err = r.get("error", "").splitlines()[0] if r.get("error") else ""
                log(f"[check] {tag}{job['suite']}/{job['id']} init{job['init']} {r['sec']}s {err}")
    for m in manifests:
        key = f"{m['suite']}/{m['id']}"
        if key not in results:
            continue
        m["feasibility"] = summarize(m, results[key])
        path = Path(m["_dir"]) / "manifest.json"
        disk = json.loads(path.read_text())
        disk["feasibility"] = m["feasibility"]
        path.write_text(json.dumps(disk, indent=1) + "\n")
        log(f"[check] {key}: {m['feasibility']['status']} {m['feasibility']['trials']}")
