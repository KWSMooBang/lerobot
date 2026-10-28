"""LIBERO-PRO task catalog (stdlib only: read by the adapter in the main venv and by the worker in the PRO venv).

Five PRO axes (+ the clean suite as baseline) → perturbed BDDL / init dirs under the PRO install
(`<root>` = `.venvs/libero_pro_src/libero/libero`, or $LIBERO_PRO_ROOT), and the standard record mapping:

| axis        | dirs (bddl_files/, init_files/)        | levels                                | dimension   |
| ----------- | -------------------------------------- | ------------------------------------- | ----------- |
| none        | <suite>                                | original                              | none        |
| object      | <suite>_object                         | object (appearance / colour / size)   | object      |
| position    | <suite>_swap; libero_object_temp_<lvl> | swap; x0.1–x0.5, y0.1–y0.5 (object)   | position    |
| semantic    | <suite>_lan                            | paraphrase                            | language    |
| task        | <suite>_task                           | task (new goal + instruction)         | task        |
| environment | <suite>_env (bench/libero_pro/gen_env_axis.py) | <from>-><to> table               | environment |

Task ids are the clean suite's order (= lerobot / hf-libero), matched by file name: PRO keeps each task's file
name in every axis dir. The instruction given to the policy is the perturbed BDDL's `(:language ...)`
(`instruction_source: bddl`); PRO's own `Benchmark.get_task(i).language` is derived from the *file name*,
i.e. the original instruction even on the semantic and task axes (`instruction_source: filename` reproduces that).
"""

from __future__ import annotations

import ast
import os
import pickle  # nosec B403: LIBERO init-state files are pickled numpy arrays
import re
import zipfile
from functools import cache
from pathlib import Path
from typing import Any

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
AXES = ("none", "object", "position", "semantic", "task", "environment")
DIMENSION = {
    "none": "none",
    "object": "object",
    "position": "position",
    "semantic": "language",
    "task": "task",
    "environment": "environment",
}
XY_LEVELS = tuple(f"{a}{v / 10:.1f}" for a in "xy" for v in range(1, 6))  # x0.1 … y0.5 (libero_object only)
# PRO's EnvironmentReplacePerturbator hard-codes living_room_table; a scene already on it would be unchanged,
# and the PRO README recommends main_table (other tables can let objects drift). gen_env_axis.py: main_table
# for every scene not on it, living_room_table for main_table scenes (spatial, goal).
ENV_TARGET = {"main_table": "living_room_table"}
ENV_TARGET_DEFAULT = "main_table"
# lerobot TASK_SUITE_MAX_STEPS (= ours); PRO's OpenVLA script uses 220 / 280 / 300 / 520 (spatial shorter)
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SUITE_SHORT = {"libero_spatial": "spat", "libero_object": "obj", "libero_goal": "goal", "libero_10": "l10"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def pro_root() -> Path:
    """`libero/libero` dir of the PRO install (bddl_files/, init_files/, assets/, benchmark/)."""
    env = os.environ.get("LIBERO_PRO_ROOT")
    return Path(env) if env else repo_root() / ".venvs" / "libero_pro_src" / "libero" / "libero"


def pro_python() -> Path:
    env = os.environ.get("LIBERO_PRO_PYTHON")
    return Path(env) if env else repo_root() / ".venvs" / "libero_pro" / "bin" / "python"


def load_init_states(path: str | Path) -> Any:
    """LIBERO `.pruned_init` (torch.save of a numpy array: zip with <x>/data.pkl, no tensors) without torch."""
    import numpy as np

    with zipfile.ZipFile(path) as z:
        pk = next(n for n in z.namelist() if n.endswith("data.pkl"))
        arr = pickle.loads(z.read(pk))  # nosec B301: pinned PRO dataset / our gen_env_axis.py output
    return np.asarray(arr)


@cache
def n_init_states(path: str) -> int:
    return len(load_init_states(path))


def bench_version(root: Path | None = None) -> str:
    """`libero-pro-<git sha>/hf-<dataset revision>/env-<hash of generated env-axis files>` (setup_venv.sh)."""
    stamp = (root or pro_root()).parents[1] / "SC_VERSION"
    return stamp.read_text().strip() if stamp.exists() else "libero-pro-unknown"


def levels(suite: str, axis: str) -> tuple[str, ...]:
    if axis == "position":
        return ("swap", *XY_LEVELS) if suite == "libero_object" else ("swap",)
    return {"none": ("original",), "semantic": ("paraphrase",), "environment": ("table",)}.get(axis, (axis,))


def axis_dir(suite: str, axis: str, level: str) -> str:
    if axis == "none":
        return suite
    if axis == "position":
        if level == "swap":
            return f"{suite}_swap"
        if suite != "libero_object" or level not in XY_LEVELS:
            raise ValueError(f"position level {level!r}: x/y shifts exist for libero_object only (PRO repo)")
        return f"libero_object_temp_{level}"
    suffix = {"object": "object", "semantic": "lan", "task": "task", "environment": "env"}[axis]
    return f"{suite}_{suffix}"


@cache
def suite_task_names(root: Path, suite: str) -> tuple[str, ...]:
    """Clean-suite task order from PRO's libero_suite_task_map.py (a dict literal; not imported: no torch)."""
    tree = ast.parse((root / "benchmark" / "libero_suite_task_map.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "libero_task_map":
            return tuple(ast.literal_eval(node.value)[suite])
    raise ValueError(f"{root}: libero_task_map not found")


def _block(text: str, head: str) -> str | None:
    """The balanced `(<head> ...)` block of a BDDL text."""
    start = text.find(f"({head}")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return text[start : i + 1]
    return None


def bddl_language(path: Path) -> str:
    m = re.search(r"\(:language\s*(.*?)\)", path.read_text(), flags=re.S)
    return " ".join(m.group(1).split()) if m else ""


def bddl_goal(path: Path) -> str:
    g = _block(path.read_text(), ":goal")
    return " ".join(g.split()) if g else ""


def filename_language(task_name: str) -> str:
    """PRO / LIBERO `grab_language_from_filename` (what Benchmark.get_task(i).language returns)."""
    x = task_name
    if x[0].isupper():  # LIBERO-10 / 90 scene prefix
        x = x[x.find("SCENE") + (8 if "SCENE10" in x else 7) :]
    return " ".join(x.split("_"))


def task_entry(
    root: Path, suite: str, axis: str, level: str, task_id: int, instruction_source: str = "bddl"
) -> dict[str, Any]:
    name = suite_task_names(root, suite)[task_id]
    d = axis_dir(suite, axis, level)
    bddl = root / "bddl_files" / d / f"{name}.bddl"
    init = root / "init_files" / d / f"{name}.pruned_init"
    orig = root / "bddl_files" / suite / f"{name}.bddl"
    for p in (bddl, init, orig):
        if not p.exists():
            raise FileNotFoundError(
                f"LIBERO-PRO file missing: {p} (bench/libero_pro/setup_venv.sh installs them)"
            )
    given = bddl_language(bddl) if instruction_source == "bddl" else filename_language(name)
    return {
        "suite": suite,
        "task_id": task_id,
        "task_name": name,
        "axis": axis,
        "level": level,
        "dir": d,
        "bddl": str(bddl),
        "init": str(init),
        "orig_bddl": str(orig),
        "instruction_original": bddl_language(orig),
        "instruction_given": given,
        "instruction_bddl": bddl_language(bddl),
        # task axis: new goal → both goals tracked every step, no early stop (T001 dual-goal rule)
        "dual": axis == "task",
        "goal_changed": bddl_goal(bddl) != bddl_goal(orig),
    }
