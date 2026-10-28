"""Benchmark adapter interface. One adapter per benchmark (`ours` here; `libero_plus` T012, `libero_pro` T013).

The coordinator (bench/core/runner.py, main venv) calls, per expanded battery job:
  job_id(job)                       → unique, file-name-safe id inside a bench run (dir runs/<run>/jobs/<job_id>/)
  plan(job, ctx)                    → JobPlan: units (the work items, e.g. one (task) of a job), episode counts
  prepare(job, plan, ctx, hello)    → write the job dir's run_meta.json and refuse a dir written with another
                                      configuration (resume guard). `hello` = the policy server's hello.
  worker_cmd(ctx)                   → argv of a long-lived worker process, run in the benchmark's own venv
                                      with CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID of its GPU and
                                      SC_POLICY_URL = that GPU's server.
  summarize(job_dir)                → short metrics string for results.tsv / STATUS lines

Worker protocol (bench/core/worker.py implements the loop; an adapter only supplies `run_unit`):
stdin gets one JSON line per unit, stdout answers one line `@@result {"unit_id", "status": "done"|"failed",
"episodes", "error"}` per unit; everything else a worker prints goes to its log. Episodes are appended to
`<job_dir>/episodes.jsonl` (records.append_jsonl) with the standard fields, and a worker skips episodes
already there, so a killed run resumes where it stopped.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ADAPTERS = {
    "ours": "bench.ours.adapter:OursAdapter",
    "libero_plus": "bench.libero_plus.adapter:LiberoPlusAdapter",  # T012
    # "libero_pro": "bench.libero_pro.adapter:LiberoProAdapter",      (T013)
}


@dataclass
class Context:
    """What adapters get from the coordinator."""

    run_dir: Path
    ckpt: dict[str, Any]  # registry row (+ adapter)
    backend: str  # libero | mock
    repo: Path
    overrides: dict[str, Any] = field(default_factory=dict)  # --tasks / --episodes from the CLI

    def job_dir(self, job_id: str) -> Path:
        return self.run_dir / "jobs" / job_id


@dataclass
class JobPlan:
    job_id: str
    # each unit: unit_id, job_id, job_dir, benchmark, est_steps, n_episodes, n_pending, ...
    units: list[dict[str, Any]]
    n_episodes: int
    n_pending: int
    est_steps: float  # pending env steps
    info: dict[str, Any] = field(default_factory=dict)  # adapter-specific (e.g. ours: RunSpec args)


class BenchmarkAdapter:
    name = "base"

    def job_id(self, job: dict[str, Any]) -> str:
        raise NotImplementedError

    def plan(self, job: dict[str, Any], ctx: Context) -> JobPlan:
        raise NotImplementedError

    def prepare(self, job: dict[str, Any], plan: JobPlan, ctx: Context, hello: dict[str, Any]) -> None:
        raise NotImplementedError

    def worker_cmd(self, ctx: Context) -> list[str]:
        raise NotImplementedError

    def summarize(self, job_dir: Path) -> str:
        from .records import read_jsonl

        eps = read_jsonl(job_dir / "episodes.jsonl")
        if not eps:
            return "n=0"
        return f"n={len(eps)};sr={sum(bool(e.get('success')) for e in eps) / len(eps):.3f}"


def get_adapter(benchmark: str) -> BenchmarkAdapter:
    if benchmark not in ADAPTERS:
        raise SystemExit(f"benchmark {benchmark!r} has no adapter (known: {sorted(ADAPTERS)})")
    mod, _, cls = ADAPTERS[benchmark].partition(":")
    return getattr(importlib.import_module(mod), cls)()
