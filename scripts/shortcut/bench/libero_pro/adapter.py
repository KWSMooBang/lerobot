"""`libero_pro` benchmark adapter (T013): LIBERO-PRO's five axes through the policy server.

Battery job keys: suite(s), perturbation(s) = axis (none | object | position | semantic | task | environment),
levels (default: all of the axis, see catalog.levels; e.g. position on libero_object: swap, x0.1 … y0.5),
tasks (default 0-9), episodes (inits per task × level, ≤ 50), seed, video (all | none | first:N), max_steps,
instruction_source (bddl = the perturbed BDDL's :language, default | filename = PRO Benchmark's task.language),
repeat.

One job = (suite, axis[, instruction_source, repeat]) → `jobs/pro-<suite>-<axis>/` (run_meta.json,
episodes.jsonl, steps/, videos/). One unit = one (level, task) × its inits. The worker runs in the PRO venv
(`.venvs/libero_pro`, setup_venv.sh); the mock backend runs it in this interpreter with a numpy env.
Records: perturbation = {dimension: catalog.DIMENSION[axis], level, detail: "pro:<axis>"}, variant_id = level,
success = the PRO BDDL's goal; task axis also success_orig (original suite goal, every step) / success_new.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from bench.core.adapters import BenchmarkAdapter, Context, JobPlan
from bench.core.records import read_jsonl

from . import catalog

SHORTCUT = Path(__file__).resolve().parents[2]
MOCK_MAX_STEPS = 40
# Expected fraction of max_steps per episode (early stop at the PRO goal); task axis runs to max_steps.
EARLY_STOP_FRAC = {"none": 0.5, "semantic": 0.55, "object": 0.75, "environment": 0.85, "position": 0.9}
_warned: set[str] = set()


def _parse_range(spec: Any) -> list[int]:
    if isinstance(spec, int):
        return [spec]
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _video_cap(spec: Any) -> int | None:
    s = str(spec or "first:3")
    if s == "all":
        return None
    if s == "none":
        return 0
    return int(s.split(":", 1)[1])


class LiberoProAdapter(BenchmarkAdapter):
    name = "libero_pro"

    def _job(self, job: dict[str, Any], ctx: Context) -> dict[str, Any]:
        job = {**job, **{k: v for k, v in ctx.overrides.items() if v is not None}}
        axis = job.get("perturbation", "none")
        if axis not in catalog.AXES:
            raise SystemExit(f"libero_pro: perturbation {axis!r} not in {catalog.AXES}")
        if job["suite"] not in catalog.SUITES:
            raise SystemExit(f"libero_pro: suite {job['suite']!r} not in {catalog.SUITES}")
        lv = job.get("levels") or catalog.levels(job["suite"], axis)
        lv = [lv] if isinstance(lv, str) else [str(x) for x in lv]
        bad = [x for x in lv if x not in catalog.levels(job["suite"], axis)]
        if bad:
            raise SystemExit(
                f"libero_pro {job['suite']} {axis}: levels {bad} not in {catalog.levels(job['suite'], axis)}"
            )
        src = job.get("instruction_source", "bddl")
        if src not in ("bddl", "filename"):
            raise SystemExit(f"libero_pro: instruction_source {src!r} not in (bddl, filename)")
        max_steps = int(
            job.get("max_steps")
            or (MOCK_MAX_STEPS if ctx.backend == "mock" else catalog.MAX_STEPS[job["suite"]])
        )
        return {
            **job,
            "axis": axis,
            "levels": lv,
            "task_ids": _parse_range(job.get("tasks", "0-9")),
            "episodes": int(job.get("episodes", 10)),
            "instruction_source": src,
            "max_steps": max_steps,
            "repeat": int(job.get("repeat", 0) or 0),
            "seed": int(job.get("seed", 7)),
        }

    def job_id(self, job: dict[str, Any]) -> str:
        jid = f"pro-{catalog.SUITE_SHORT[job['suite']]}-{job.get('perturbation', 'none')}"
        if job.get("instruction_source", "bddl") != "bddl":
            jid += f"-{job['instruction_source']}"
        rep = int(job.get("repeat", 0) or 0)
        return jid + (f"-r{rep}" if rep else "")

    def plan(self, job: dict[str, Any], ctx: Context) -> JobPlan:
        j = self._job(job, ctx)
        jid = self.job_id(job)
        jdir = ctx.job_dir(jid)
        root = catalog.pro_root()
        n_eps = {(lv, t): j["episodes"] for lv in j["levels"] for t in j["task_ids"]}
        if root.exists():  # every unit's files must exist (catches a missing env-axis generation early)
            for lv, t in n_eps:
                e = catalog.task_entry(root, j["suite"], j["axis"], lv, t, j["instruction_source"])
                n_init = catalog.n_init_states(e["init"])
                if n_init < j["episodes"]:  # PRO ships one empty file: libero_object_temp_y0.5 cream cheese
                    n_eps[lv, t] = n_init
                    if e["init"] not in _warned:
                        _warned.add(e["init"])
                        print(
                            f"[libero_pro] {e['dir']}/{e['task_name']}: {n_init} init states, using {n_init}"
                        )
        elif "root" not in _warned:
            _warned.add("root")
            print(f"[libero_pro] {root} missing (setup_venv.sh): planning without file checks", flush=True)
        done = {
            (r["task_id"], r["level"], r["init_idx"])
            for r in read_jsonl(jdir / "episodes.jsonl")
            if r.get("repeat_idx", 0) == j["repeat"]
        }
        frac = 1.0 if j["axis"] == "task" else EARLY_STOP_FRAC[j["axis"]]
        per_ep = frac * j["max_steps"] + 10
        units = []
        for (lv, t), n in n_eps.items():
            if n == 0:
                continue
            pending = sum((t, lv, i) not in done for i in range(n))
            units.append(
                {
                    "unit_id": f"{jid}/{lv}/t{t:02d}",
                    "job_id": jid,
                    "job_dir": str(jdir),
                    "benchmark": self.name,
                    "suite": j["suite"],
                    "axis": j["axis"],
                    "level": lv,
                    "task_id": t,
                    "episodes": n,
                    "n_episodes": n,
                    "n_pending": pending,
                    "est_steps": pending * per_ep,
                }
            )
        return JobPlan(
            job_id=jid,
            units=units,
            n_episodes=sum(u["n_episodes"] for u in units),
            n_pending=sum(u["n_pending"] for u in units),
            est_steps=sum(u["est_steps"] for u in units),
            info={"job": j},
        )

    def bench_version(self, ctx: Context) -> str:
        return "libero-pro-mock" if ctx.backend == "mock" else catalog.bench_version()

    def prepare(self, job: dict[str, Any], plan: JobPlan, ctx: Context, hello: dict[str, Any]) -> None:
        j = plan.info["job"]
        jdir = ctx.job_dir(plan.job_id)
        jdir.mkdir(parents=True, exist_ok=True)
        ex = hello["exec"]
        bv = self.bench_version(ctx)
        identity = {
            "benchmark": self.name,
            "bench_version": bv,
            "ckpt_id": ctx.ckpt["ckpt_id"],
            "checkpoint": ctx.ckpt["path_or_hub"],
            "suite": j["suite"],
            "axis": j["axis"],
            "instruction_source": j["instruction_source"],
            "seed": j["seed"],
            "repeat_idx": j["repeat"],
            "max_steps": j["max_steps"],
            "env_hw": list(hello["image_hw"]),
            "n_action_steps": int(hello["n_action_steps"]),
            "backend": ctx.backend,
            **{k: ex[k] for k in ("exec_mode", "gpu", "torch", "cuda")},
            "via": "policy_server",
        }
        meta_path = jdir / "run_meta.json"
        prev = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if prev is not None:
            diff = {
                k: (prev["identity"].get(k), v) for k, v in identity.items() if prev["identity"].get(k) != v
            }
            if diff:
                raise SystemExit(f"{jdir} was written with another configuration (old, new): {diff}")
        elif (jdir / "episodes.jsonl").exists():
            raise SystemExit(f"{jdir}/episodes.jsonl exists without run_meta.json")
        meta = {
            "identity": identity,
            "benchmark": self.name,
            "bench_version": bv,
            "bench_run": ctx.run_dir.name,
            "model": ctx.ckpt["ckpt_id"],
            "levels": j["levels"],
            "tasks": j["task_ids"],
            "episodes": j["episodes"],
            "video": j.get("video", "first:3"),
            "pro_root": str(catalog.pro_root()),
            "policy_server": {k: v for k, v in hello.items() if k not in ("inference", "ok")},
            "created": (prev or {}).get("created") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        plan.info["unit_common"] = {
            "model": ctx.ckpt["ckpt_id"],
            "ckpt": ctx.ckpt["path_or_hub"],
            "seed": j["seed"],
            "repeat_idx": j["repeat"],
            "max_steps": j["max_steps"],
            "video_cap": _video_cap(j.get("video")),
            "env_hw": int(hello["image_hw"][0]),
            "backend": ctx.backend,
            "bench_version": bv,
            "instruction_source": j["instruction_source"],
            "pro_root": str(catalog.pro_root()),
            "requires_step": bool(hello.get("requires_step")),
        }

    def worker_cmd(self, ctx: Context) -> list[str]:
        worker = str(Path(__file__).with_name("worker.py"))
        if ctx.backend == "mock":
            return [sys.executable, worker]
        py = catalog.pro_python()
        if not py.exists():
            raise SystemExit(f"{py} missing: run scripts/shortcut/bench/libero_pro/setup_venv.sh")
        return [str(py), worker]

    def summarize(self, job_dir: Path) -> str:
        eps = read_jsonl(job_dir / "episodes.jsonl")
        if not eps:
            return "n=0"
        s = f"n={len(eps)};sr={sum(bool(e['success']) for e in eps) / len(eps):.3f}"
        dual = [e for e in eps if e.get("success_new") is not None and e.get("success_orig") is not None]
        if dual:
            s += f";sr_orig={sum(bool(e['success_orig']) for e in dual) / len(dual):.3f}"
        return s
