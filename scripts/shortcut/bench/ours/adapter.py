"""`ours` benchmark adapter: our instruction perturbations C0–C6 (T001/T002) through the policy server.

One battery job = one (suite, condition, repeat) = one T001 run dir (`jobs/ours-<suite>-<cond>[-r<k>]/`:
run_meta.json with the T001 identity + the server's exec fields, episodes.jsonl, steps/, videos/), so
metrics.py, make_compare_video.py and the P1 analysis scripts read it unchanged. One unit = one task of
a job (all its variants × inits). Job keys: suite, condition, tasks (default 0-9), episodes, repeat, seed,
video (all|none|first:N), instr_set (default instructions/<suite>.json), variants, include_infeasible,
max_steps.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from functools import cache
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from bench.core.adapters import BenchmarkAdapter, Context, JobPlan
from bench.core.records import read_jsonl

SHORTCUT = Path(__file__).resolve().parents[2]
BENCH_VERSION = "ours-v1"  # bump when the harness changes what an episode means
# Expected fraction of max_steps a non-dual episode runs (early stop at the goal), LIBERO-finetuned policy.
EARLY_STOP_FRAC = {"C0": 0.45, "C3": 0.5, "C1": 0.9, "C2": 0.9}


def _paths() -> None:
    for p in (str(SHORTCUT), str(SHORTCUT / "parallel")):
        if p not in sys.path:
            sys.path.insert(0, p)


@cache
def _tokenizer(backend: str) -> Any:
    _paths()
    if backend == "mock":
        from mock_backend import MockTokenizer

        return MockTokenizer()
    from common import load_tokenizer

    return load_tokenizer()


def bench_version(instr_set: str, backend: str) -> str:
    sha = hashlib.sha256(Path(instr_set).read_bytes()).hexdigest()[:8]
    if backend == "mock":
        return f"{BENCH_VERSION}/instr-{sha}/mock"
    try:
        lib = metadata.version("hf-libero")
    except metadata.PackageNotFoundError:
        lib = "unknown"
    _paths()
    from common import libero_env_tag

    # The T009 env fix is appended (/fix-...); raw (SC_LIBERO_RAW=1, the P1 env) keeps the string it had
    # before T009, so old and new raw results are one version and report.py flags only raw vs fix mixes.
    bv = f"{BENCH_VERSION}/instr-{sha}/hf-libero-{lib}"
    tag = libero_env_tag()
    return bv if tag == "raw" else f"{bv}/{tag}"


def suite_short(suite: str) -> str:
    _paths()
    from common import SUITE_SHORT

    return SUITE_SHORT[suite]


class OursAdapter(BenchmarkAdapter):
    name = "ours"

    def job_id(self, job: dict[str, Any]) -> str:
        rep = int(job.get("repeat", 0) or 0)
        return f"ours-{suite_short(job['suite'])}-{job['condition']}" + (f"-r{rep}" if rep else "")

    def run_args(self, job: dict[str, Any], ctx: Context) -> Any:
        """run_eval.py arguments for this job (the T001 CLI is the source of defaults)."""
        _paths()
        import run_eval

        job = {**job, **{k: v for k, v in ctx.overrides.items() if v is not None}}
        instr = job.get("instr_set") or str(SHORTCUT / "instructions" / f"{job['suite']}.json")
        argv = [
            "--policy", "pi05", "--suite", job["suite"], "--condition", job["condition"],
            "--instr-set", str(instr), "--tasks", str(job.get("tasks", "0-9")),
            "--episodes", str(job.get("episodes", 50)), "--repeat-idx", str(job.get("repeat", 0) or 0),
            "--seed", str(job.get("seed", 7)), "--out", str(ctx.job_dir(self.job_id(job))),
            "--video", str(job.get("video", "first:3")), "--backend", ctx.backend,
            "--model-tag", ctx.ckpt["ckpt_id"], "--checkpoint", ctx.ckpt["path_or_hub"],
            "--libero-adapter", ctx.ckpt.get("adapter") or "none",
        ]  # fmt: skip
        if job.get("variants"):
            argv += ["--variants", ",".join(job["variants"])]
        if job.get("include_infeasible"):
            argv.append("--include-infeasible")
        if job.get("max_steps"):
            argv += ["--max-steps", str(job["max_steps"])]
        return run_eval.build_parser().parse_args(argv)

    def plan(self, job: dict[str, Any], ctx: Context) -> JobPlan:
        _paths()
        import run_eval
        from common import load_instruction_set

        args = self.run_args(job, ctx)
        jid = self.job_id(job)
        jdir = ctx.job_dir(jid)
        _suite, tasks, k_by_task = run_eval.suite_tasks(args)
        max_steps = run_eval.default_max_steps(args)
        instr = load_instruction_set(args.instr_set)
        instr["_suite_tasks"] = tasks
        spec_like = cast(Any, SimpleNamespace(args=args, instr_set=instr))  # plan_variants reads only these
        planned = run_eval.plan_variants(spec_like, _tokenizer(ctx.backend))
        done = {
            (r["task_id"], r["condition"], r["variant"], r["init_idx"], r["repeat_idx"])
            for r in read_jsonl(jdir / "episodes.jsonl")
        }
        dual = args.condition in ("C4", "C5", "C6")
        full = dual or bool(ctx.ckpt.get("adapter"))
        per_ep = (max_steps if full else EARLY_STOP_FRAC.get(args.condition, 0.6) * max_steps) + 10
        units = []
        for task_id, variants in planned:
            n = len(variants) * args.episodes
            pending = sum(
                (task_id, args.condition, v.variant, i, args.repeat_idx) not in done
                for v in variants
                for i in range(args.episodes)
            )
            units.append(
                {
                    "unit_id": f"{jid}/t{task_id:02d}",
                    "job_id": jid,
                    "job_dir": str(jdir),
                    "benchmark": "ours",
                    "suite": args.suite,
                    "condition": args.condition,
                    "task_id": task_id,
                    "variants": [asdict(v) for v in variants],
                    "episodes": args.episodes,
                    "n_episodes": n,
                    "n_pending": pending,
                    "est_steps": pending * per_ep,
                    "dual": dual,
                }
            )
        return JobPlan(
            job_id=jid,
            units=units,
            n_episodes=sum(u["n_episodes"] for u in units),
            n_pending=sum(u["n_pending"] for u in units),
            est_steps=sum(u["est_steps"] for u in units),
            info={"args": args, "k_by_task": k_by_task, "max_steps": max_steps},
        )

    def prepare(self, job: dict[str, Any], plan: JobPlan, ctx: Context, hello: dict[str, Any]) -> None:
        """T001 run dir for this job: identity (+ server exec fields) checked against an existing dir,
        run_meta.json written, and the per-unit constants every worker needs."""
        _paths()
        import run_eval

        args = plan.info["args"]
        args.policy = hello["policy_type"]
        args.n_action_steps = int(hello["n_action_steps"])
        ex = hello["exec"]
        extra = {**{k: ex[k] for k in ("exec_mode", "gpu", "torch", "cuda")}, "via": "policy_server"}
        spec = run_eval.prepare_run(args, ex["exec_mode"], ex["gpu"], identity_extra=extra)
        spec.out.mkdir(parents=True, exist_ok=True)
        meta_path = spec.out / "run_meta.json"
        prev = json.loads(meta_path.read_text()) if meta_path.exists() else None
        bv = bench_version(args.instr_set, ctx.backend)
        meta = run_eval.run_meta(
            spec,
            hello.get("inference"),
            prev,
            benchmark="ours",
            bench_version=bv,
            bench_run=ctx.run_dir.name,
            policy_server={k: v for k, v in hello.items() if k not in ("inference", "ok")},
        )
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        plan.info["unit_common"] = {
            "run": spec.info(),
            "seed": args.seed,
            "repeat_idx": args.repeat_idx,
            "max_steps": spec.max_steps,
            "video_cap": spec.video_cap,
            "env_hw": hello["image_hw"],
            "control_mode": hello.get("control_mode", "relative"),
            "backend": ctx.backend,
            "bench_version": bv,
            "requires_step": bool(hello.get("requires_step")),
        }

    def worker_cmd(self, ctx: Context) -> list[str]:
        return [sys.executable, str(SHORTCUT / "bench" / "ours" / "worker.py")]

    def summarize(self, job_dir: Path) -> str:
        eps = read_jsonl(job_dir / "episodes.jsonl")
        if not eps:
            return "n=0"
        s = f"n={len(eps)};sr_orig={sum(bool(e['success_orig']) for e in eps) / len(eps):.3f}"
        dual = [e for e in eps if e.get("success_new") is not None]
        if dual:
            s += f";sr_new={sum(bool(e['success_new']) for e in dual) / len(dual):.3f}"
        return s
