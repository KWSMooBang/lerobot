"""Eval side of user variants (condition `custom`): plan, env, goals. Used by run_eval.py and bench/ours.

One custom Variant per built manifest: the episode runs in the variant's own BDDL with its init states
(lerobot LiberoEnv of the base task with `_task_bddl_file` / `_init_states` replaced), runs to max_steps
like C4–C6, and tracks
  new goal   = the variant goal                → success_new / t_new
  orig goal  = the base task's goal             → success_orig / t_orig (False when its objects are gone)
  track goals (default: every training goal)   → goals_reached {name: first step}, goals_final, goals_initial
task_id = base task (seed and pairing with the base task's C0 episodes of the same init index).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .goals import goal_cls
from .spec import find_spec, load_manifest, load_spec, manifest_file, spec_paths

CUSTOM = "custom"


def _variant_key(m: dict[str, Any]) -> str:
    """What an episode's outcome depends on (resume refuses a mix): the generated files (BDDL, init states) and
    the scoring (instruction, variant goal, base task and its goal, tracked goals incl. absent ones, goal
    semantics). Editing only notes / group / comments of the YAML keeps the key."""
    scoring = {
        "instruction": m["instruction"],
        "goal": m["goal"],
        "base_task": m["base_task"]["task_id"],
        "orig_goal": m["base_task"]["goal"],
        "orig_goal_absent": m["base_task"].get("goal_absent"),
        "track_goals": {n: [t["goal"], bool(t["absent"])] for n, t in sorted(m["track_goals"].items())},
        "goal_semantics": m.get("goal_semantics"),
    }
    h = hashlib.sha256(json.dumps(scoring, sort_keys=True).encode()).hexdigest()[:12]
    return f"{m['files']['bddl']['sha256'][:16]}/{m['files']['init_states']['sha256'][:16]}/goals-{h}"


def plan_custom(
    suite: str,
    ids: list[str] | None,
    variants_dir: str | None,
    backend: str,
    env_tag: str | None,
    episodes_path: Path | None = None,
    phrasings: list[str] | None = None,
) -> list[tuple[int, list[Any]]]:
    """[(base task id, [Variant])] — one entry per variant, in the given order. `phrasings` keeps only variants
    whose spec `phrasing` is in the list (e.g. ["train"])."""
    from common import Variant

    if not ids:
        ids = [p.stem for p in spec_paths(suite, variants_dir)]
        if not ids:
            raise SystemExit(f"no variants for {suite} in {variants_dir or 'scripts/shortcut/variants'}")
    prev: dict[str, str | None] = {}
    if episodes_path is not None and episodes_path.exists():
        for line in episodes_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("condition") == CUSTOM:
                prev.setdefault(r["variant"], (r.get("variant_meta") or {}).get("variant_key"))
    out = []
    for vid in ids:
        if phrasings is not None and load_spec(find_spec(suite, vid, variants_dir)).phrasing not in phrasings:
            continue
        m = load_manifest(suite, vid, variants_dir)
        spec_file = find_spec(suite, vid, variants_dir)
        if load_spec(spec_file).sha256 != m["spec_sha256"]:
            raise SystemExit(
                f"{spec_file} changed since it was built: run sc-variants build --suite {suite} --ids {vid}"
            )
        if backend != "mock" and env_tag is not None and m["libero_env"] != env_tag:
            raise SystemExit(
                f"variant {suite}/{vid} was built for libero_env {m['libero_env']}, this run uses {env_tag}: rebuild it"
            )
        key = _variant_key(m)
        if vid in prev and prev[vid] != key:
            raise SystemExit(
                f"{episodes_path}: variant {vid} has episodes from other generated files ({prev[vid]} vs {key}); use a new --out"
            )
        track = {n: t["goal"] for n, t in m["track_goals"].items() if not t["absent"]}
        meta = {
            "variant_key": key,
            "spec_sha256": m["spec_sha256"],
            "group": m.get("group"),
            "phrasing": (m.get("spec") or {}).get("phrasing"),
            "pair": (m.get("spec") or {}).get("pair"),
            "base_task": m["base_task"]["task_id"],
            "bddl": str(manifest_file(m, "bddl")),
            "init_states": str(manifest_file(m, "init_states")),
            "n_init": m["init_states"]["n"],
            "init_mode": m["init_states"]["mode"],
            "goal": m["goal"],
            "orig_goal": m["base_task"]["goal"],
            "orig_goal_absent": bool(m["base_task"]["goal_absent"]),
            "track_goals": track,
            "track_instructions": {
                n: t["instruction"] for n, t in m["track_goals"].items() if not t["absent"]
            },
            "track_absent": sorted(n for n, t in m["track_goals"].items() if t["absent"]),
            "dimension": m["edits"]["dimension"],
            "edits": m["edits"]["kinds"],
            "feasibility": (m.get("feasibility") or {}).get("status"),
            "libero_env": m["libero_env"],
        }
        v = Variant(CUSTOM, vid, m["instruction"], goal_bddl=meta["bddl"], meta=meta)
        out.append((int(m["base_task"]["task_id"]), [v]))
    return out


def goals_for(v: Any, backend: str) -> tuple[Any, Any, dict[str, Any]]:
    """(new goal, original goal, {track name: goal}) checkers for a custom Variant."""
    goal_type = goal_cls(backend)
    m = v.meta
    new = goal_type(m["goal"], label=f"{v.variant}:new")
    orig = goal_type([] if m["orig_goal_absent"] else m["orig_goal"], label=f"t{m['base_task']}")
    track = {n: goal_type(g, label=n) for n, g in m["track_goals"].items()}
    return new, orig, track


def make_env(
    v: Any, suite_obj: Any, suite: str, backend: str, max_steps: int, hw: list[int], control_mode: str
) -> Any:
    """Eval env of a custom variant (lerobot LiberoEnv on the variant BDDL + init states)."""
    m = v.meta
    if backend == "mock":
        from mock_backend import MockEnv

        return MockEnv(suite_obj.task_info[m["base_task"]], max_steps, hw)
    from run_eval import libero

    from .generate import load_init_states

    env = libero().LiberoEnv(
        suite_obj, m["base_task"], suite, obs_type="pixels_agent_pos",
        observation_height=hw[0], observation_width=hw[1], control_mode=control_mode, episode_length=max_steps,
    )  # fmt: skip
    env._task_bddl_file = m["bddl"]
    env._init_states = load_init_states(m["init_states"])
    env.task_description = v.instruction
    return env
