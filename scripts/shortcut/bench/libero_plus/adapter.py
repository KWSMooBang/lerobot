"""`libero_plus` benchmark adapter (T012): LIBERO-Plus perturbation axes through the policy server.

One battery job = one (suite, perturbation axis) = `jobs/plus-<suite>-<axis>[-<tag>]/` (run_meta.json,
episodes.jsonl, videos/). One unit = one original task of the job: its selected registered variants ×
init states. The env runs in the LIBERO-Plus venv (`setup_libero_plus.sh`, default <repo>/.venvs/libero_plus,
or $LIBERO_PLUS_VENV); this module only reads the fork's task_classification.json (path from the venv's
sc_env.json), so planning / --dry-run work in the main venv.

Job keys (battery YAML):
  suite, perturbation   axis: none | language | layout | camera | robot | lighting | texture | noise
                        (`perturbations: [...]` expands); none = the original task in the Plus env
  tasks                 original task ids (vanilla LIBERO order, = ours task_id), default 0-9
  variants              registered variants per task, stratified over difficulty levels (round-robin
                        from level 1, seeded shuffle inside a level), or "all"; default 5 (none: 1)
  levels, details       only these difficulty levels / sub-types (registry.py docstring), e.g. [1, 5]
  episodes              init states per variant (init 0..n−1; upstream protocol = 1), default 1
                        (none: default 10). `_add_` / `_level` layouts have one init state: init i → 0
  instruction           clean (default): non-language axes get the original instruction, so each axis is
                        isolated; registry: the fork's own task.language, which for every non-language
                        variant is the instruction + the filename suffix ("... view 0 0 100 0 0
                        initstate 19", "... table 1") — what upstream / lerobot's libero_plus env feed
  seed (7), video (first:N per unit | all | none), max_steps (lerobot TASK_SUITE_MAX_STEPS), tag
Episode seed = ours' episode_seed(seed, 0, task_id, init_idx), so `none` pairs with ours C0 episodes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from bench.core.adapters import BenchmarkAdapter, Context, JobPlan
from bench.core.records import read_jsonl
from bench.libero_plus import registry as reg

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parents[1]
REPO = SHORTCUT.parents[1]
SUITE_SHORT = {"libero_spatial": "spat", "libero_object": "obj", "libero_goal": "goal", "libero_10": "l10"}
# Expected fraction of max_steps an episode runs (early stop at success), LIBERO-finetuned policy.
EARLY_STOP_FRAC = {"none": 0.45, "language": 0.6, "layout": 0.7}
DEFAULT_FRAC = 0.75
SETTLE_STEPS = 10
INSTRUCTION_MODES = ("clean", "registry")
DUAL_GOAL_AXES = ("none", "language", "layout")  # success_orig recorded (goal = original's, checked)


def venv_dir() -> Path:
    return Path(os.environ.get("LIBERO_PLUS_VENV") or REPO / ".venvs" / "libero_plus")


def sc_env() -> dict[str, Any]:
    p = venv_dir() / "sc_env.json"
    if not p.exists():
        raise SystemExit(
            f"LIBERO-Plus venv not set up ({p} missing): scripts/shortcut/bench/libero_plus/setup_libero_plus.sh"
        )
    return json.loads(p.read_text())


def parse_tasks(spec: Any) -> list[int]:
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return [int(x) for x in spec]
    out: list[int] = []
    for part in str(spec).split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def video_cap(spec: Any) -> int | None:
    """first:N → N recorded episodes per unit; all → None (every episode); none → 0."""
    s = str(spec)
    if s == "all":
        return None
    if s == "none":
        return 0
    if s.startswith("first:"):
        return int(s.split(":", 1)[1])
    raise ValueError(f"video: {spec!r} (all | none | first:N)")


def check_selection(jdir: Path, sel_sha: str, selection: dict[str, Any]) -> None:
    """Resume guard for the variant selection, at plan time (before servers start; also in --dry-run): a job dir
    holds episodes of exactly one selection, else its success rate would depend on the run history."""
    meta_path = jdir / "run_meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    prev = (meta.get("identity") or {}).get("selection_sha")
    if prev != sel_sha:
        old = meta.get("selection") or {}
        diff: dict[str, Any] = {
            k: (old.get(k), v) for k, v in selection.items() if k != "picked" and old.get(k) != v
        }
        if not diff:
            diff = {"picked": "registered variants differ (registry / fork revision changed?)"}
        raise SystemExit(
            f"{jdir}: holds episodes of another variant selection {diff}; run this job under another "
            "`tag:` (job name) or --run-id"
        )


class LiberoPlusAdapter(BenchmarkAdapter):
    name = "libero_plus"

    def job_id(self, job: dict[str, Any]) -> str:
        axis = job.get("perturbation", "none")
        tag = f"-{job['tag']}" if job.get("tag") else ""
        return f"plus-{SUITE_SHORT[job['suite']]}-{axis}{tag}"

    def settings(self, job: dict[str, Any], ctx: Context) -> dict[str, Any]:
        job = {**job, **{k: v for k, v in ctx.overrides.items() if v is not None}}
        axis = job.get("perturbation", "none")
        if axis not in reg.AXES:
            raise SystemExit(f"libero_plus job: perturbation {axis!r} not in {reg.AXES}")
        mode = job.get("instruction", "clean")
        if mode not in INSTRUCTION_MODES:
            raise SystemExit(f"libero_plus job: instruction {mode!r} not in {INSTRUCTION_MODES}")
        suite = job["suite"]
        return {
            "suite": suite,
            "axis": axis,
            "tasks": parse_tasks(job.get("tasks", "0-9")),
            "variants": job.get("variants", 1 if axis == "none" else 5),
            "levels": job.get("levels"),
            "details": job.get("details"),
            "episodes": int(job.get("episodes", 10 if axis == "none" else 1)),
            "instruction": mode,
            "seed": int(job.get("seed", 7)),
            "video": str(job.get("video", "first:3")),
            "max_steps": int(job.get("max_steps") or reg.MAX_STEPS[suite]),
        }

    def plan(self, job: dict[str, Any], ctx: Context) -> JobPlan:
        st = self.settings(job, ctx)
        env = sc_env()
        src = Path(env["src"])
        jid = self.job_id(job)
        jdir = ctx.job_dir(jid)
        motion_blur = bool(env.get("magick", False))  # no libMagickWand: motion-blur noise cannot run
        if st["axis"] == "none":
            picked = {t: [reg.original_variant(st["suite"], t)] for t in st["tasks"]}
        else:
            pool = reg.load_variants(src, st["suite"])
            if not motion_blur:
                n_mb = sum(reg.motion_blur(v) and v.task_id in st["tasks"] for v in pool)
                pool = [v for v in pool if not reg.motion_blur(v)]
                if n_mb and st["axis"] == "noise":
                    print(
                        f"[libero_plus] {jid}: {n_mb} motion-blur variants left out of the selection "
                        "(no libMagickWand; apt libmagickwand-6.q16-6)",
                        file=sys.stderr,
                    )
            picked = reg.select(
                pool,
                dimension=st["axis"],
                tasks=st["tasks"],
                n=st["variants"],
                levels=st["levels"],
                details=st["details"],
                seed=st["seed"],
            )
        done = {(r["task_id"], r["variant_id"], r["init_idx"]) for r in read_jsonl(jdir / "episodes.jsonl")}
        frac = EARLY_STOP_FRAC.get(st["axis"], DEFAULT_FRAC)
        per_ep = frac * st["max_steps"] + SETTLE_STEPS
        units = []
        for t, vs in picked.items():
            if not vs:
                continue
            eps = [(v, i) for v in vs for i in range(st["episodes"])]
            pending = sum((t, v.variant_id, i) not in done for v, i in eps)
            units.append(
                {
                    "unit_id": f"{jid}/t{t:02d}",
                    "job_id": jid,
                    "job_dir": str(jdir),
                    "benchmark": "libero_plus",
                    "suite": st["suite"],
                    "task_id": t,
                    "axis": st["axis"],
                    "variants": [v.to_dict() for v in vs],
                    "episodes": st["episodes"],
                    "n_episodes": len(eps),
                    "n_pending": pending,
                    "est_steps": pending * per_ep,
                }
            )
        selection = {
            "axis": st["axis"],
            "tasks": st["tasks"],
            "variants": st["variants"],
            "levels": st["levels"],
            "details": st["details"],
            "episodes": st["episodes"],
            "instruction": st["instruction"],
            "motion_blur": motion_blur,
            "picked": {str(t): [v.name for v in vs] for t, vs in picked.items() if vs},
        }
        sel_sha = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()[:16]
        check_selection(jdir, sel_sha, selection)
        return JobPlan(
            job_id=jid,
            units=units,
            n_episodes=sum(u["n_episodes"] for u in units),
            n_pending=sum(u["n_pending"] for u in units),
            est_steps=sum(u["est_steps"] for u in units),
            info={
                "settings": st,
                "src": str(src),
                "sha": env.get("sha", "unknown"),
                "motion_blur": motion_blur,
                "selection": selection,
                "selection_sha": sel_sha,
            },
        )

    def bench_version(self, plan: JobPlan, ctx: Context) -> str:
        v = f"libero-plus-{str(plan.info['sha'])[:7]}"
        return f"{v}/mock" if ctx.backend == "mock" else v

    def prepare(self, job: dict[str, Any], plan: JobPlan, ctx: Context, hello: dict[str, Any]) -> None:
        st = plan.info["settings"]
        jdir = ctx.job_dir(plan.job_id)
        jdir.mkdir(parents=True, exist_ok=True)
        ex = hello["exec"]
        bv = self.bench_version(plan, ctx)
        identity = {
            "benchmark": "libero_plus",
            "bench_version": bv,
            "suite": st["suite"],
            "axis": st["axis"],
            "instruction": st["instruction"],
            "motion_blur": plan.info["motion_blur"],  # selection differs without libMagickWand
            "selection_sha": plan.info["selection_sha"],  # axis, levels, details, tasks, variant names, inits
            "seed": st["seed"],
            "max_steps": st["max_steps"],
            "ckpt": ctx.ckpt["path_or_hub"],
            "model": ctx.ckpt["ckpt_id"],
            "policy": hello["policy_type"],
            "n_action_steps": int(hello["n_action_steps"]),
            "env_hw": list(hello["image_hw"]),
            "control_mode": hello.get("control_mode", "relative"),
            "backend": ctx.backend,
            **{k: ex[k] for k in ("exec_mode", "gpu", "torch", "cuda")},
            "via": "policy_server",
        }
        meta_path = jdir / "run_meta.json"
        prev = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if prev is not None and prev.get("identity") != identity:
            diff = {
                k: (prev["identity"].get(k), identity.get(k))
                for k in set(prev["identity"]) | set(identity)
                if prev["identity"].get(k) != identity.get(k)
            }
            raise SystemExit(f"{jdir}: written with another configuration {diff}; use a new --run-id")
        meta = {
            "identity": identity,
            "benchmark": "libero_plus",
            "bench_version": bv,
            "bench_run": ctx.run_dir.name,
            "settings": st,
            "selection": plan.info["selection"],
            "libero_plus_src": plan.info["src"],
            "policy_server": {k: v for k, v in hello.items() if k not in ("inference", "ok")},
            "inference": hello.get("inference"),
            "created": (prev or {}).get("created") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        plan.info["unit_common"] = {
            "model": ctx.ckpt["ckpt_id"],
            "ckpt": ctx.ckpt["path_or_hub"],
            "policy": hello["policy_type"],
            "seed": st["seed"],
            "max_steps": st["max_steps"],
            "video_cap": video_cap(st["video"]),
            "env_hw": int(hello["image_hw"][0]),
            "control_mode": hello.get("control_mode", "relative"),
            "requires_step": bool(hello.get("requires_step")),
            "instruction_mode": st["instruction"],
            "dual_goal": st["axis"] in DUAL_GOAL_AXES,
            "backend": ctx.backend,
            "bench_version": bv,
        }

    def worker_cmd(self, ctx: Context) -> list[str]:
        worker = str(HERE / "worker.py")
        if ctx.backend == "mock":  # mock env: no LIBERO needed, the main venv will do
            return [sys.executable, worker]
        py = venv_dir() / "bin" / "python"
        if not py.exists():
            raise SystemExit(f"{py} missing: run scripts/shortcut/bench/libero_plus/setup_libero_plus.sh")
        return [str(py), worker]

    def summarize(self, job_dir: Path) -> str:
        eps = read_jsonl(job_dir / "episodes.jsonl")
        if not eps:
            return "n=0"
        return f"n={len(eps)};sr={sum(bool(e['success']) for e in eps) / len(eps):.3f}"
