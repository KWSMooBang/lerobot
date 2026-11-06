"""Shared constants and helpers of the open-loop language probe (T018). numpy / json / pandas only.

Query plan per observation (`queries_for`): the task's original instruction with three noise seeds
(orig_s0 / orig_s1 / orig_s2: the noise floor), the blank instruction "" (seed 0) and every swap of the task in
instructions/<suite>.json (C4-* object swaps and C5-* target swaps, feasible or not; seed 0). Seed 0 is the
same for every instruction of an observation, so instruction is the only thing that differs between orig_s0,
blank and the swaps.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parents[1]
INSTR_DIR = SHORTCUT / "instructions"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
N_TASKS = 10
N_SEEDS = 3  # orig_s0 .. orig_s2
SEED_BASE = {
    "A": 7,
    "B": 8,
}  # set A: = sc-bench's default --seed 7, repeat r → common.episode_seed(7, r, task, init)
T_EVAL_S = (
    0.4  # common horizon: 8 OSC steps (20 Hz) = 6 joint-velocity steps (15 Hz); StarVLA returns 8 steps
)
CONTROL_HZ = {"osc_pose_delta": 20, "joint_velocity": 15}
SET_A_INITS = 5
SET_B_EPISODES = 2
SET_B_PROGRESS = (0.25, 0.5, 0.75)
DEFAULT_ROOT = "/outputs/VLA-shortcut-exps"


def episode_seed(base: int, repeat_idx: int, task_id: int, init_idx: int) -> int:
    """= common.episode_seed (kept here so the probe client needs no lerobot import)."""
    return base * 1_000_000 + repeat_idx * 10_000 + task_id * 100 + init_idx


def load_instruction_sets(instr_dir: Path = INSTR_DIR) -> dict[str, dict[str, Any]]:
    return {s: json.loads((instr_dir / f"{s}.json").read_text()) for s in SUITES}


def task_entry(sets: dict[str, dict[str, Any]], suite: str, task_id: int) -> dict[str, Any]:
    for t in sets[suite]["tasks"]:
        if int(t["task_id"]) == int(task_id):
            return t
    raise KeyError(f"{suite} task {task_id} not in the instruction set")


def queries_for(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Instructions asked for one observation of this task (see module doc)."""
    orig = entry["original"]
    out: list[dict[str, Any]] = [
        {"instr_id": f"orig_s{r}", "kind": "orig", "instruction": orig, "rep": r} for r in range(N_SEEDS)
    ]
    out.append({"instr_id": "blank", "kind": "blank", "instruction": "", "rep": 0})
    for sw in entry.get("swaps", []):
        out.append(
            {
                "instr_id": sw["id"],
                "kind": "swap",
                "instruction": sw["instruction"],
                "rep": 0,
                "swap_type": sw.get("type"),
                "feasible": sw.get("feasible"),
                "orig_target": sw.get("orig_target"),
                "new_target": sw.get("new_target"),
            }
        )
    return out


def count_queries(sets: dict[str, dict[str, Any]], tasks: list[tuple[str, int]]) -> dict[str, int]:
    n = {"orig": 0, "blank": 0, "swap": 0}
    for suite, tid in tasks:
        for q in queries_for(task_entry(sets, suite, tid)):
            n[q["kind"]] += 1
    return n


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def set_files(obs_dir: Path, name: str) -> tuple[Path, Path]:
    return obs_dir / f"set_{name}.npz", obs_dir / f"set_{name}.meta.json"


def load_set(obs_dir: str | Path, name: str, verify: bool = True) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Arrays + per-observation rows of set A / B; with verify, the sha256 in manifest.json is checked."""
    obs_dir = Path(obs_dir)
    npz, meta = set_files(obs_dir, name)
    if verify:
        man = json.loads((obs_dir / "manifest.json").read_text())["sets"][name]
        for p, key in ((npz, "npz_sha256"), (meta, "meta_sha256")):
            got = sha256_file(p)
            if got != man[key]:
                raise SystemExit(f"{p}: sha256 {got} != manifest {man[key]} (regenerate or copy again)")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    rows = json.loads(meta.read_text())["rows"]
    if len(rows) != len(arrays["state"]):
        raise SystemExit(f"{meta}: {len(rows)} rows, {npz}: {len(arrays['state'])} observations")
    return arrays, rows


def set_sha(obs_dir: str | Path, name: str) -> str:
    """sha256 of the set's npz as it is on disk (the identity probe rows and measure caches record)."""
    return sha256_file(set_files(Path(obs_dir), name)[0])


def image_sizes(arrays: dict[str, np.ndarray]) -> list[int]:
    return sorted(int(k.split("_")[1]) for k in arrays if k.startswith("image_"))


def pick_size(arrays: dict[str, np.ndarray], want_hw: Any) -> int:
    """Rendered size closest to the server's hello image_hw (its eval render size)."""
    sizes = image_sizes(arrays)
    want = int(want_hw[0]) if want_hw else sizes[0]
    return min(sizes, key=lambda s: (abs(s - want), s))


def wire_obs(arrays: dict[str, np.ndarray], i: int, size: int) -> dict[str, np.ndarray]:
    out = {
        "image": np.ascontiguousarray(arrays[f"image_{size}"][i]),
        "image2": np.ascontiguousarray(arrays[f"image2_{size}"][i]),
        "state": np.asarray(arrays["state"][i], dtype=np.float32),
    }
    if "joint_pos" in arrays:
        out["joint_pos"] = np.asarray(arrays["joint_pos"][i], dtype=np.float32)
    return out


def obs_seed(set_name: str, row: dict[str, Any], rep: int) -> int:
    """Set A: sc-bench's episode seed of (task, init, repeat) with --seed 7. Set B: base 8, slot = episode
    slot × 10 + progress index."""
    if set_name == "A":
        return episode_seed(SEED_BASE["A"], rep, int(row["task_id"]), int(row["init_idx"]))
    return episode_seed(SEED_BASE["B"], rep, int(row["task_id"]), int(row["slot"]))
