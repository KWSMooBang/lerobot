"""Goal checks for user variants: a conjunction of BDDL predicates evaluated in the running scene.

Same interface as run_eval.GoalChecker (`missing_objects(env)`, `__call__(env) -> bool`), but built from
predicate lists (the variant goal, the base task goal, each tracked training goal) instead of a BDDL file.

One deviation from LIBERO: `TurnOff x` is evaluated as `not TurnOn x`. LIBERO's flat_stove TurnOff
needs knob qpos < 0.0 while the joint range is [-0.005, 2.1], so a knob turned back to its rest position
(0.0) never counts as off. TURNOFF_SEMANTICS is written into every manifest.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

TURNOFF_SEMANTICS = (
    "TurnOff x := not TurnOn x (LIBERO's TurnOff needs knob qpos < 0 inside a [-0.005, 2.1] range)"
)


def _inner(env: Any) -> Any:
    """LIBERO problem env from a lerobot LiberoEnv (reset) or a ControlEnv / OffScreenRenderEnv."""
    if hasattr(env, "_env"):
        assert env._env is not None, "reset() the env first"
        return env._env.env
    return env.env if hasattr(env, "env") else env


def eval_predicate(e: Any, pred: list[str]) -> bool:
    name = pred[0].lower()
    if name == "turnoff":
        return not bool(e._eval_predicate(["turnon", *pred[1:]]))
    return bool(e._eval_predicate([name, *pred[1:]]))


class PredicateGoal:
    """Conjunction of predicates; an empty list is never satisfied (variants without an achievable goal)."""

    def __init__(self, preds: list[list[str]], label: str = ""):
        self.preds = [list(p) for p in preds]
        self.label = label

    def names(self) -> set[str]:
        return {n for p in self.preds for n in p[1:]}

    def missing_objects(self, env: Any) -> list[str]:
        states = _inner(env).object_states_dict
        return sorted(n for n in self.names() if n not in states)

    def __call__(self, env: Any) -> bool:
        if not self.preds:
            return False
        e = _inner(env)
        return all(eval_predicate(e, p) for p in self.preds)


class MockPredicateGoal:
    """CPU stand-in (`--backend mock`): satisfied near a point derived from the goal text, like MockGoal."""

    def __init__(self, preds: list[list[str]], label: str = ""):
        self.preds = [list(p) for p in preds]
        self.label = label
        key = label or repr(self.preds)
        h = hashlib.sha256(key.encode()).digest()
        self.target = (np.frombuffer(h[:3], np.uint8).astype(np.float64) / 255.0 - 0.5) * 0.4

    def missing_objects(self, env: Any) -> list[str]:
        return []

    def __call__(self, env: Any) -> bool:
        if not self.preds:
            return False
        return bool(np.linalg.norm(env.pos - self.target) < 0.05)


def goal_cls(backend: str) -> type:
    return MockPredicateGoal if backend == "mock" else PredicateGoal
