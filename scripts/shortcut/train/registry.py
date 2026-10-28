"""Checkpoint registry (TESTBED_DESIGN.md §3): `<out-root>/ckpt_registry.tsv`, read by `sc-bench --ckpt <id>`.

Columns: ckpt_id path_or_hub policy_type train_data train_variant steps run_id git_sha created_at notes.
One line per ckpt_id; appending an existing id is refused unless `replace=True`. Writes are flock'ed.
"""

from __future__ import annotations

import csv
import fcntl
import io
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

COLUMNS = (
    "ckpt_id",
    "path_or_hub",
    "policy_type",
    "train_data",
    "train_variant",
    "steps",
    "run_id",
    "git_sha",
    "created_at",
    "notes",
)
REGISTRY_NAME = "ckpt_registry.tsv"

# Checkpoints that exist before any sc-train run (same as common.POLICY_CKPT / BASE_CKPT).
SEED_ROWS = (
    {
        "ckpt_id": "pi05lib",
        "path_or_hub": "lerobot/pi05_libero_finetuned",
        "policy_type": "pi05",
        "train_data": "HuggingFaceVLA/libero",
        "train_variant": "none",
        "steps": "6000",
        "run_id": "",
        "git_sha": "",
        "notes": "lerobot release: 6k steps from lerobot/pi05_libero, all 4 suites mixed, bs 256",
    },
    {
        "ckpt_id": "pi05base",
        "path_or_hub": "lerobot/pi05_base",
        "policy_type": "pi05",
        "train_data": "none",
        "train_variant": "none",
        "steps": "",
        "run_id": "",
        "git_sha": "",
        "notes": "not trained on LIBERO; needs the LIBERO adapter (config, processors, stats from pi05lib)",
    },
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clean(v: object) -> str:
    return str(v if v is not None else "").replace("\t", " ").replace("\n", " ")


def read_registry(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows


def lookup(path: Path, ckpt_id: str) -> dict[str, str] | None:
    return next((r for r in read_registry(path) if r["ckpt_id"] == ckpt_id), None)


def add_entries(path: Path, entries: Sequence[Mapping[str, object]], replace: bool = False) -> list[str]:
    """Append rows (missing columns -> ""; created_at defaults to now). Returns the ids written.

    Raises ValueError if an id already exists and `replace` is False.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        text = f.read()
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t")) if text else []
        if text and tuple(text.splitlines()[0].split("\t")) != COLUMNS:
            raise ValueError(f"{path}: header differs from {COLUMNS}")
        ids = {r["ckpt_id"] for r in rows}
        new = []
        for e in entries:
            row = {c: _clean(e.get(c)) for c in COLUMNS}
            row["created_at"] = row["created_at"] or now_iso()
            if not row["ckpt_id"] or not row["path_or_hub"]:
                raise ValueError(f"ckpt_id and path_or_hub are required: {row}")
            if row["ckpt_id"] in ids and not replace:
                raise ValueError(f"ckpt_id {row['ckpt_id']!r} already in {path} (use --replace)")
            new.append(row)
        if replace:
            drop = {r["ckpt_id"] for r in new}
            rows = [r for r in rows if r["ckpt_id"] not in drop]
        rows += new
        f.seek(0)
        f.truncate()
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        return [r["ckpt_id"] for r in new]


def ensure_seed_rows(path: Path) -> list[str]:
    """Add pi05lib / pi05base if absent (T010 may have created the file first)."""
    have = {r["ckpt_id"] for r in read_registry(path)}
    missing = [dict(r) for r in SEED_ROWS if r["ckpt_id"] not in have]
    return add_entries(path, missing) if missing else []
