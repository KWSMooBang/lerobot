"""Standard episode record (TESTBED_DESIGN.md §2), shared by every benchmark client.

A benchmark's episodes.jsonl line = T001 fields where they apply + `standard_fields(...)`:
  benchmark (ours|libero_plus|libero_pro), bench_version, suite, task_id, variant_id,
  perturbation {dimension, level, detail}, instruction_original, instruction_given,
  success (the benchmark's own definition), success_orig / success_new (when measurable), video, seed,
  init_idx, model, ckpt.

Clean / unperturbed episodes use perturbation.dimension = "none"; report.py takes them as each
(model, benchmark, suite)'s baseline. Line appends are flock'ed, so several workers can share one file.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any

BENCHMARKS = ("ours", "libero_plus", "libero_pro")
DIMENSIONS = (
    "none",
    "language",
    "layout",
    "object",
    "camera",
    "lighting",
    "texture",
    "robot",
    "noise",
    "position",
    "task",  # LIBERO-PRO task axis: new goal + instruction in the same scene (T013)
    "environment",  # LIBERO-PRO environment axis: work table replaced (T013)
)
REQUIRED = (
    "benchmark",
    "bench_version",
    "model",
    "ckpt",
    "suite",
    "task_id",
    "variant_id",
    "init_idx",
    "perturbation",
    "instruction_original",
    "instruction_given",
    "success",
    "success_orig",
    "success_new",
    "video",
    "seed",
)
# ours: condition → (dimension, level). detail = condition id (+ variant id in variant_id).
OURS_PERTURBATION = {
    "C0": ("none", "original"),
    "C1": ("language", "blank"),
    "C2": ("language", "nonsense"),
    "C3": ("language", "paraphrase"),
    "C4": ("language", "object_swap"),
    "C5": ("language", "target_swap"),
    "C6": ("language", "cross_task"),
    "C7": ("layout", "position_swap"),
    "C8": ("object", "object_swap_visual"),
}


def perturbation(dimension: str, level: Any, detail: Any) -> dict[str, Any]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"perturbation dimension {dimension!r} not in {DIMENSIONS}")
    return {"dimension": dimension, "level": level, "detail": detail}


def standard_fields(
    *,
    benchmark: str,
    bench_version: str,
    model: str,
    ckpt: str,
    suite: str,
    task_id: int,
    variant_id: str,
    init_idx: int,
    pert: dict[str, Any],
    instruction_original: str,
    instruction_given: str,
    success: bool,
    success_orig: bool | None,
    success_new: bool | None,
    video: str | None,
    seed: int,
) -> dict[str, Any]:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"benchmark {benchmark!r} not in {BENCHMARKS}")
    return {
        "benchmark": benchmark,
        "bench_version": bench_version,
        "model": model,
        "ckpt": ckpt,
        "suite": suite,
        "task_id": int(task_id),
        "variant_id": str(variant_id),
        "init_idx": int(init_idx),
        "perturbation": pert,
        "instruction_original": instruction_original,
        "instruction_given": instruction_given,
        "success": bool(success),
        "success_orig": None if success_orig is None else bool(success_orig),
        "success_new": None if success_new is None else bool(success_new),
        "video": video,
        "seed": int(seed),
    }


def missing_fields(rec: dict[str, Any]) -> list[str]:
    miss = [k for k in REQUIRED if k not in rec]
    p = rec.get("perturbation")
    if not isinstance(p, dict) or not {"dimension", "level", "detail"} <= set(p):
        miss.append("perturbation.{dimension,level,detail}")
    return miss


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, default=str) + "\n"
    with open(path, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.seek(0, 2) > 0:
            f.seek(-1, 2)
            if f.read(1) != b"\n":  # fragment of a killed writer: start on a fresh line
                line = "\n" + line
        f.write(line.encode())
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Complete lines only (a writer killed mid-line leaves a fragment that is skipped)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
