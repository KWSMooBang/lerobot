"""Battery files (`bench/batteries/<name>.yaml`, TESTBED_DESIGN.md §4) and the dry-run time model.

    name: quick
    description: ...
    defaults: {episodes: 20, video: "first:3", seed: 7}
    jobs:
      - {benchmark: ours, suite: libero_goal, conditions: [C0, C1], tasks: 0-9, episodes: 20}
      - {benchmark: libero_plus, suite: libero_goal, perturbations: [language, layout], ...}   # T012

List-valued keys named in `EXPAND` (conditions → condition, perturbations → perturbation, suites → suite,
repeats → repeat) expand into one job per value; everything else is passed to the benchmark adapter as is.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BATTERY_DIR = Path(__file__).resolve().parents[1] / "batteries"
EXPAND = {"suites": "suite", "conditions": "condition", "perturbations": "perturbation", "repeats": "repeat"}


def battery_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    q = BATTERY_DIR / f"{name_or_path}.yaml"
    if q.exists():
        return q
    known = sorted(x.stem for x in BATTERY_DIR.glob("*.yaml"))
    raise SystemExit(f"battery {name_or_path!r}: no such file (known: {known})")


def load_battery(name_or_path: str) -> dict[str, Any]:
    path = battery_path(name_or_path)
    raw = path.read_bytes()
    data = yaml.safe_load(raw) or {}
    if not isinstance(data.get("jobs"), list) or not data["jobs"]:
        raise SystemExit(f"{path}: needs a non-empty `jobs:` list")
    data.setdefault("name", path.stem)
    data["_path"] = str(path)
    data["_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def expand_jobs(battery: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = battery.get("defaults") or {}
    out = []
    for i, job in enumerate(battery["jobs"]):
        if "benchmark" not in job:
            raise SystemExit(f"{battery['_path']}: job {i} has no `benchmark`")
        base = {**defaults, **job}
        keys = [k for k in EXPAND if k in base]
        values = [base.pop(k) if isinstance(base[k], list) else [base.pop(k)] for k in keys]
        for combo in itertools.product(*values):
            out.append({**base, **{EXPAND[k]: v for k, v in zip(keys, combo, strict=True)}})
    return out


# ---------------------------------------------------------------- dry-run time model


@dataclass
class TimeModel:
    """Clients step their env in parallel; a GPU's server answers K clients with one call of batch K.
    cycle = n_action_steps · step_ms + infer_ms(K);  infer_ms(K) = infer1 · (1 + β (K − 1)) + rpc_ms.
    Defaults: RTX 4090 pi05 bf16 (T006): infer1 174 ms (incl. pre/post), β 0.25, env step incl. contact
    probe + video 26 ms. RPC (msgpack, 2×360² images) ~3 ms."""

    step_ms: float = 26.0
    infer1_ms: float = 174.0
    beta: float = 0.25
    rpc_ms: float = 3.0
    n_action_steps: int = 10
    load_s: float = 90.0  # server start + model load per GPU

    def steps_per_s(self, clients: int) -> float:
        k = max(1, clients)
        infer = self.infer1_ms * (1 + self.beta * (k - 1)) + self.rpc_ms
        cycle = self.n_action_steps * self.step_ms + infer
        return 1000 * k * self.n_action_steps / cycle

    def hours(self, env_steps: float, gpus: int, clients_per_gpu: int) -> float:
        return (env_steps / (self.steps_per_s(clients_per_gpu) * gpus) + self.load_s) / 3600


def fmt_hours(h: float) -> str:
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def estimate_table(env_steps: float, tm: TimeModel, setups: list[tuple[str, int, int]]) -> list[str]:
    lines = []
    for label, gpus, k in setups:
        h = tm.hours(env_steps, gpus, k)
        mode = "single" if k == 1 else f"batched-{k}"
        lines.append(
            f"  {label:<28} {gpus} GPU × {k} clients ({mode}): {tm.steps_per_s(k) * gpus:6.0f} env-steps/s → {fmt_hours(h)}"
        )
    return lines
