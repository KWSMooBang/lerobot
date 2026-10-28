"""Make the local policy plugins (scripts/shortcut/policies/lerobot_policy_*) importable and registered.

lerobot only auto-discovers installed `lerobot_policy_*` distributions. This puts each plugin's `src/`
on sys.path and imports it, so `--policy.type=<name>` works without `pip install -e`. Other tools
(sc-bench server, lerobot-eval wrappers) can call `register_local_plugins()` the same way.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

POLICIES_DIR = Path(__file__).resolve().parents[1] / "policies"


def local_plugin_dirs() -> list[Path]:
    return sorted(p for p in POLICIES_DIR.glob("lerobot_policy_*") if (p / "src" / p.name).is_dir())


def register_local_plugins() -> list[str]:
    names = []
    for p in local_plugin_dirs():
        src = str(p / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        importlib.import_module(p.name)
        names.append(p.name)
    return names
