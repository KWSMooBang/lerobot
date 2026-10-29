"""Checkpoint registry (TESTBED_DESIGN.md §3): `<out_root>/ckpt_registry.tsv`, written by sc-train (T011).

    ckpt_id  path_or_hub  policy_type  train_data  train_variant  steps  run_id  git_sha  created_at  notes

`notes` may hold `key=value` items separated by `;`. Read here: `adapter=<ckpt>` (config + pre/post
processors from that LIBERO-finetuned checkpoint, weights from path_or_hub; common.load_policy).
If the file does not exist, `ensure_registry` creates it with the two pre-existing checkpoints.
"""

from __future__ import annotations

import csv
import fcntl
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_OUT_ROOT = "/outputs/VLA-shortcut-exps"
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
INITIAL_ROWS = (
    {
        "ckpt_id": "pi05lib",
        "path_or_hub": "lerobot/pi05_libero_finetuned",
        "policy_type": "pi05",
        "train_data": "HuggingFaceVLA/libero",
        "train_variant": "all4-mixed",
        "steps": "6000",
        "run_id": "-",
        "git_sha": "-",
        "notes": "lerobot hub finetune of lerobot/pi05_libero on all 4 suites mixed",
    },
    {
        "ckpt_id": "pi05base",
        "path_or_hub": "lerobot/pi05_base",
        "policy_type": "pi05",
        "train_data": "-",
        "train_variant": "-",
        "steps": "-",
        "run_id": "-",
        "git_sha": "-",
        "notes": "adapter=lerobot/pi05_libero_finetuned; not trained on LIBERO",
    },
)


def out_root(arg: str | None = None) -> Path:
    return Path(arg or os.environ.get("SC_OUT_ROOT") or DEFAULT_OUT_ROOT)


def registry_path(root: str | Path | None = None) -> Path:
    if os.environ.get("SC_CKPT_REGISTRY") and root is None:
        return Path(os.environ["SC_CKPT_REGISTRY"])
    return out_root(str(root) if root else None) / "ckpt_registry.tsv"


def read_registry(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {r["ckpt_id"]: r for r in rows if r.get("ckpt_id")}


def ensure_registry(path: Path) -> bool:
    """Create the file with the initial rows if it does not exist. True if this call created it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(f".{path.name}.lock"), "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if path.exists() and path.stat().st_size > 0:
            return False
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
        with open(tmp, "w") as f:
            f.write("\t".join(COLUMNS) + "\n")
            for r in INITIAL_ROWS:
                f.write("\t".join({**r, "created_at": now}[c] for c in COLUMNS) + "\n")
        tmp.replace(path)
        return True


def parse_notes(notes: str | None) -> dict[str, str]:
    out = {}
    for part in (notes or "").split(";"):
        k, sep, v = part.strip().partition("=")
        if sep and re.fullmatch(r"[A-Za-z_][\w-]*", k.strip()):
            out[k.strip()] = v.strip()
    return out


def safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "ckpt"


def resolve(ckpt: str, path: Path) -> dict[str, Any]:
    """`--ckpt` (registry id, local checkpoint dir, or hub id) → registry-like row + `adapter`.

    Unregistered local dirs / hub ids get ckpt_id = sanitized last path component (pretrained_model dirs
    use the run dir name) and `registered: False`."""
    reg = read_registry(path)
    row: dict[str, Any]
    by_path = {r["path_or_hub"]: k for k, r in reg.items()}
    if ckpt in reg or ckpt in by_path:
        row = dict(reg[ckpt if ckpt in reg else by_path[ckpt]])
        row["registered"] = True
    else:
        p = Path(ckpt)
        if p.exists():
            parts = [x for x in p.resolve().parts if x not in ("pretrained_model", "checkpoints", "last")]
            cid = safe_id(parts[-1])
        elif re.fullmatch(r"[\w.-]+/[\w.-]+", ckpt):
            cid = safe_id(ckpt.split("/")[-1])
        else:
            raise SystemExit(
                f"--ckpt {ckpt!r}: not in {path} ({', '.join(sorted(reg)) or 'empty'}), not a path, not a hub id"
            )
        row = {**dict.fromkeys(COLUMNS, "-"), "ckpt_id": cid, "path_or_hub": ckpt, "registered": False}
    row["adapter"] = parse_notes(row.get("notes")).get("adapter")
    return row
