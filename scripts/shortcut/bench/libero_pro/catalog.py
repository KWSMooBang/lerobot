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
name in every axis dir. Instructions (`instruction_source`): `training` (default) gives the LIBERO training
sentence (= ours C0) on the non-language axes and PRO's sentence on semantic / task; PRO's BDDL `:language`
differs from the training sentences even on the clean suite ("Pick the akita black bowl ..."), so `pro_bddl`
(PRO's text everywhere) mixes a language shift into the visual axes. `filename` = PRO's own
`Benchmark.get_task(i).language` (file name: the original sentence even on semantic / task).
`instruction_original` is always the training sentence; PRO's clean-suite sentence is `pro_instruction_original`.
Init states (`init_source`): see `DEFAULT_INIT_SOURCE`.
"""

from __future__ import annotations

import ast
import json
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


INSTRUCTIONS_DIR = Path(__file__).resolve().parents[2] / "instructions"  # ours: <suite>.json "original"
LANGUAGE_AXES = ("semantic", "task")  # the sentence is (part of) the perturbation
INSTRUCTION_SOURCES = ("training", "pro_bddl", "filename")
INIT_SOURCES = ("default", "pro", "clean")
# init_source "default": which init file each axis starts from (compare_inits.py; README "Init states")
# PRO re-samples placements for every axis (1–8 cm from the clean init at the same index); where only the sentence /
# goal should change, start from the clean suite's init states instead.
DEFAULT_INIT_SOURCE = {"task": "clean", "semantic": "clean"}


def resolve_init_source(axis: str, init_source: str) -> str:
    """pro = the axis' own PRO init file; clean = the clean suite's init file at the same index (same sim-state
    layout required; pairs with ours C0 at the same init)."""
    if init_source not in INIT_SOURCES:
        raise ValueError(f"init_source {init_source!r} not in {INIT_SOURCES}")
    if axis == "none":
        return "pro"  # the clean suite's file either way
    return DEFAULT_INIT_SOURCE.get(axis, "pro") if init_source == "default" else init_source


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


@cache
def training_instructions(suite: str) -> dict[str, str]:
    """BDDL file stem → the LIBERO training instruction (HuggingFaceVLA/libero task string = ours C0)."""
    data = json.loads((INSTRUCTIONS_DIR / f"{suite}.json").read_text())
    return {Path(t["bddl"]).stem: t["original"] for t in data["tasks"]}


def given_instruction(axis: str, source: str, training: str, pro_bddl: str, filename: str) -> str:
    """training (default): the training sentence on the non-language axes, PRO's sentence where the sentence
    *is* the perturbation (semantic, task); pro_bddl: the axis BDDL's :language everywhere (PRO's text);
    filename: PRO Benchmark.get_task().language (file name, unchanged on every axis)."""
    if source == "training":
        return pro_bddl if axis in LANGUAGE_AXES else training
    if source == "pro_bddl":
        return pro_bddl
    if source == "filename":
        return filename
    raise ValueError(f"instruction_source {source!r} not in {INSTRUCTION_SOURCES}")


def task_entry(
    root: Path,
    suite: str,
    axis: str,
    level: str,
    task_id: int,
    instruction_source: str = "training",
    init_source: str = "default",
) -> dict[str, Any]:
    name = suite_task_names(root, suite)[task_id]
    d = axis_dir(suite, axis, level)
    bddl = root / "bddl_files" / d / f"{name}.bddl"
    init = root / "init_files" / d / f"{name}.pruned_init"
    orig = root / "bddl_files" / suite / f"{name}.bddl"
    clean_init = root / "init_files" / suite / f"{name}.pruned_init"
    for p in (bddl, init, orig, clean_init):
        if not p.exists():
            raise FileNotFoundError(
                f"LIBERO-PRO file missing: {p} (bench/libero_pro/setup_venv.sh installs them)"
            )
    src = resolve_init_source(axis, init_source)
    training = training_instructions(suite)[name]
    pro_bddl = bddl_language(bddl)
    return {
        "suite": suite,
        "task_id": task_id,
        "task_name": name,
        "axis": axis,
        "level": level,
        "dir": d,
        "bddl": str(bddl),
        "init": str(clean_init if src == "clean" else init),
        "init_source": src,
        "orig_bddl": str(orig),
        "instruction_original": training,
        "pro_instruction_original": bddl_language(orig),
        "instruction_given": given_instruction(
            axis, instruction_source, training, pro_bddl, filename_language(name)
        ),
        "instruction_bddl": pro_bddl,
        # task axis: new goal → both goals tracked every step, no early stop (T001 dual-goal rule)
        "dual": axis == "task",
        "goal_changed": bddl_goal(bddl) != bddl_goal(orig),
    }
