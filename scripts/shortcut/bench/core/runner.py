"""sc-bench: evaluate one checkpoint on a battery of benchmark jobs.

    scripts/shortcut/bench/sc-bench --ckpt pi05lib --battery quick [--gpus 0,1,2,3] [--out-root DIR] [--dry-run]

- ckpt: ckpt_registry.tsv id (<out-root>/ckpt_registry.tsv; created with pi05lib/pi05base if missing),
  local pretrained_model dir, or hub id.
- One policy server per GPU (serve/policy_server.py, CUDA_VISIBLE_DEVICES=<g>), `--clients-per-gpu K`
  workers per GPU (each one env, MUJOCO_EGL_DEVICE_ID=<g>). K = 1 → server exec_mode single (bit-identical
  to run_eval.py); K > 1 → batched-K (one padded call for the K clients; a different sample, never paired
  with single, T004).
- Run dir <out-root>/runs/<run-id>/ (default run id: bench-<ckpt_id>-<battery>): bench_meta.json, jobs/<job_id>/
  (one dir per expanded job; ours = T001 run dir), servers/, workers/, + submit.sh bookkeeping (git_sha,
  git_dirty, patch.diff, cmd.txt, meta.txt, STATUS, started_at, ended_at), one results.tsv line at the end.
- Interrupt and rerun the same command: finished episodes are skipped (per job episodes.jsonl). The run dir
  refuses another checkpoint / battery file / exec mode.
- GPU locks: with /queue/locks (lab servers) each GPU's /queue/locks/<uuid>.lock is flock'ed while its
  server runs (waits if busy; --no-gpu-lock to skip). Without /harness or /queue nothing else changes.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import json
import os
import shlex
import signal
import subprocess  # nosec B404
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Any

SHORTCUT = Path(__file__).resolve().parents[2]
REPO = SHORTCUT.parents[1]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from bench.core import registry  # noqa: E402
from bench.core.adapters import Context, JobPlan, get_adapter  # noqa: E402
from bench.core.battery import TimeModel, estimate_table, expand_jobs, fmt_hours, load_battery  # noqa: E402
from bench.core.worker import RESULT_PREFIX  # noqa: E402

MAX_ATTEMPTS = 3  # a unit that fails (or kills its worker) this many times is given up
SERVER_TIMEOUT_S = 1800


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git(*a: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *a], text=True)  # nosec B603 B607
    except Exception:
        return ""


def gpu_uuids() -> dict[str, str]:
    try:
        out = subprocess.check_output(  # nosec B603 B607
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        )
    except Exception:
        return {}
    return dict(line.split(", ", 1) for line in out.strip().splitlines() if ", " in line)


# ---------------------------------------------------------------- planning


def plan_all(
    args: argparse.Namespace, ctx: Context, battery: dict[str, Any]
) -> list[tuple[dict, Any, JobPlan]]:
    jobs = expand_jobs(battery)
    if args.only:
        keep = set(args.only.split(","))
        jobs = [
            j
            for j in jobs
            if keep & {j.get("suite"), j.get("condition"), j.get("perturbation"), j["benchmark"]}
        ]
    if args.suites:
        jobs = [j for j in jobs if j.get("suite") in set(args.suites.split(","))]
    out, seen = [], set()
    for job in jobs:
        ad = get_adapter(job["benchmark"])
        plan = ad.plan(job, ctx)
        if plan.job_id in seen:
            raise SystemExit(f"duplicate job id {plan.job_id} in battery {battery['name']}")
        seen.add(plan.job_id)
        out.append((job, ad, plan))
    return out


def print_plan(plans: list[tuple[dict, Any, JobPlan]], args: argparse.Namespace, tm: TimeModel) -> None:
    tot_eps = sum(p.n_episodes for _, _, p in plans)
    pend = sum(p.n_pending for _, _, p in plans)
    steps = sum(p.est_steps for _, _, p in plans)
    print(f"{'job':<28} {'units':>5} {'episodes':>8} {'pending':>8} {'est env-steps':>14}")
    for _, _, p in plans:
        print(f"{p.job_id:<28} {len(p.units):>5} {p.n_episodes:>8} {p.n_pending:>8} {p.est_steps:>14,.0f}")
    print(f"{'total':<28} {sum(len(p.units) for _, _, p in plans):>5} {tot_eps:>8} {pend:>8} {steps:>14,.0f}")
    print(
        f"time model: env step {tm.step_ms} ms (incl. probe/video), infer batch-1 {tm.infer1_ms} ms, β {tm.beta}, "
        f"rpc {tm.rpc_ms} ms, n_action_steps {tm.n_action_steps}, model load {tm.load_s:.0f} s per GPU"
    )
    ngpu = len(args.gpus.split(","))
    setups = [
        ("this call", ngpu, args.clients_per_gpu),
        ("GPU 0 (4090), single", 1, 1),
        ("GPU 0 (4090), K=4", 1, 4),
        ("4×A100 (4090 numbers), K=8", 4, 8),
        ("4×A100 (4090 numbers), K=16", 4, 16),
    ]
    print(f"estimated wall time for {steps:,.0f} pending env steps:")
    print("\n".join(estimate_table(steps, tm, setups)))


# ---------------------------------------------------------------- servers and workers


class ServerProc:
    def __init__(self, gpu: str, args: argparse.Namespace, ctx: Context):
        self.gpu = gpu
        sdir = ctx.run_dir / "servers"
        sdir.mkdir(parents=True, exist_ok=True)
        self.ready = sdir / f"gpu{gpu}.ready.json"
        self.ready.unlink(missing_ok=True)
        env = dict(os.environ)
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM="false")
        if ctx.backend == "mock":
            env["CUDA_VISIBLE_DEVICES"] = ""
        batch = args.server_batch if args.server_batch is not None else args.clients_per_gpu
        cmd = [
            sys.executable, str(SHORTCUT / "serve" / "policy_server.py"), "--ckpt", ctx.ckpt["path_or_hub"],
            "--port", "0", "--ready-file", str(self.ready), "--batch", str(batch),
            "--n-action-steps", str(args.n_action_steps), "--backend", ctx.backend,
            "--libero-adapter", ctx.ckpt.get("adapter") or "none", "--registry", str(args.registry_path),
        ]  # fmt: skip
        self.log = open(sdir / f"gpu{gpu}.log", "a")  # noqa: SIM115
        self.log.write(f"\n[{now()}] {shlex.join(cmd)}\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT)  # nosec B603

    def wait_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + SERVER_TIMEOUT_S
        while not self.ready.exists():
            if self.proc.poll() is not None:
                raise SystemExit(
                    f"policy server on GPU {self.gpu} exited ({self.proc.returncode}); see {self.log.name}"
                )
            if time.monotonic() > deadline:
                raise SystemExit(f"policy server on GPU {self.gpu} not ready after {SERVER_TIMEOUT_S} s")
            time.sleep(0.5)
        return json.loads(self.ready.read_text())

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


class Worker:
    def __init__(self, wid: str, gpu: str, url: str, cmd: list[str], ctx: Context, backend: str):
        env = dict(os.environ)
        env.update(
            SC_POLICY_URL=url,
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_VISIBLE_DEVICES=gpu,
            MUJOCO_GL="egl",
            MUJOCO_EGL_DEVICE_ID=gpu,
            OMP_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
        )
        if backend == "mock":
            env["CUDA_VISIBLE_DEVICES"] = ""
            env.pop("MUJOCO_EGL_DEVICE_ID")
        wdir = ctx.run_dir / "workers"
        wdir.mkdir(parents=True, exist_ok=True)
        self.log = open(wdir / f"{wid}.log", "a")  # noqa: SIM115
        self.proc = subprocess.Popen(  # nosec B603
            cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1
        )

    def run(self, unit: dict[str, Any]) -> dict[str, Any]:
        """Send one unit, wait for its result line. A dead worker → status 'crashed'."""
        assert self.proc.stdin is not None and self.proc.stdout is not None
        try:
            self.proc.stdin.write(json.dumps(unit, default=str) + "\n")
            self.proc.stdin.flush()
            for line in self.proc.stdout:
                if line.startswith(RESULT_PREFIX):
                    return json.loads(line[len(RESULT_PREFIX) :])
                self.log.write(line)
        except (BrokenPipeError, OSError):
            pass
        return {"unit_id": unit["unit_id"], "status": "crashed", "error": f"worker exit {self.proc.poll()}"}

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write('{"quit": true}\n')
                self.proc.stdin.close()
                self.proc.wait(30)
            except Exception:
                self.proc.kill()
        self.log.close()


class Dispatcher:
    """Shared unit pool; one thread per (GPU, slot). A slot keeps its worker per benchmark and prefers
    units of the (suite, task) its env already holds; long units first."""

    def __init__(
        self, units: list[dict[str, Any]], ctx: Context, adapters: dict[str, Any], urls: dict[str, str]
    ):
        self.pool = collections.deque(sorted(units, key=lambda u: (u["benchmark"], -u["est_steps"])))
        self.lock = threading.Lock()
        self.ctx, self.adapters, self.urls = ctx, adapters, urls
        self.attempts: collections.Counter = collections.Counter()
        self.failed: dict[str, str] = {}
        self.done = 0
        self.total = len(units)
        self.workers: list[Worker] = []
        self.stopping = False

    def take(self, loc: tuple | None, bench: str | None) -> dict[str, Any] | None:
        with self.lock:
            if self.stopping or not self.pool:
                return None
            for want in (
                lambda u: (u["benchmark"], u.get("suite"), u.get("task_id")) == loc,
                lambda u: u["benchmark"] == bench,
            ):
                for i, u in enumerate(self.pool):
                    if want(u):
                        del self.pool[i]
                        return u
            return self.pool.popleft()

    def slot(self, gpu: str, k: int) -> None:
        workers: dict[str, Worker] = {}
        loc: tuple | None = None
        bench: str | None = None
        try:
            while True:
                unit = self.take(loc, bench)
                if unit is None:
                    return
                b = unit["benchmark"]
                w = workers.get(b)
                if w is None or not w.alive():
                    w = Worker(
                        f"gpu{gpu}-{k}-{b}",
                        gpu,
                        self.urls[gpu],
                        self.adapters[b].worker_cmd(self.ctx),
                        self.ctx,
                        self.ctx.backend,
                    )
                    workers[b] = w
                    with self.lock:
                        self.workers.append(w)
                tic = time.time()
                res = w.run(unit)
                loc, bench = (b, unit.get("suite"), unit.get("task_id")), b
                with self.lock:
                    if res["status"] == "done":
                        self.done += 1
                        print(
                            f"[sc-bench] {now()} gpu{gpu}/{k} {unit['unit_id']}: {res.get('episodes', 0)} episodes in "
                            f"{time.time() - tic:.0f} s ({self.done}/{self.total} units)",
                            flush=True,
                        )
                        continue
                    self.attempts[unit["unit_id"]] += 1
                    n = self.attempts[unit["unit_id"]]
                    print(
                        f"[sc-bench] gpu{gpu}/{k} {unit['unit_id']} {res['status']} ({n}/{MAX_ATTEMPTS}): {res.get('error')}",
                        flush=True,
                    )
                    if n < MAX_ATTEMPTS and not self.stopping:
                        self.pool.append(unit)
                    else:
                        self.failed[unit["unit_id"]] = str(res.get("error"))
                if res["status"] == "crashed":
                    loc = None
        finally:
            for w in workers.values():
                w.stop()

    def stop(self) -> None:
        with self.lock:
            self.stopping = True
            ws = list(self.workers)
        for w in ws:
            if w.alive():
                w.proc.terminate()


# ---------------------------------------------------------------- run dir


def bench_identity(
    args: argparse.Namespace, ctx: Context, battery: dict[str, Any], hello: dict[str, Any] | None
) -> dict[str, Any]:
    ident = {
        "ckpt_id": ctx.ckpt["ckpt_id"],
        "ckpt": ctx.ckpt["path_or_hub"],
        "libero_adapter": ctx.ckpt.get("adapter"),
        "battery": battery["name"],
        "battery_sha256": battery["_sha256"],
        "overrides": {k: v for k, v in ctx.overrides.items() if v is not None},
        "n_action_steps": args.n_action_steps,
        "backend": ctx.backend,
    }
    if hello is not None:
        ident.update({k: hello["exec"][k] for k in ("exec_mode", "gpu", "torch", "cuda")})
    return ident


def check_bench_identity(run_dir: Path, ident: dict[str, Any]) -> None:
    p = run_dir / "bench_meta.json"
    if not p.exists():
        return
    prev = json.loads(p.read_text()).get("identity", {})
    diff = {k: (prev.get(k), v) for k, v in ident.items() if k in prev and prev.get(k) != v}
    if diff:
        raise SystemExit(
            f"{run_dir} was written with another configuration (old, new): {diff}; use another --run-id"
        )


def write_bookkeeping(run_dir: Path, args: argparse.Namespace, gpus: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "git_sha").write_text(git("rev-parse", "HEAD"))
    (run_dir / "git_dirty").write_text(git("status", "--porcelain"))
    (run_dir / "patch.diff").write_text(git("diff", "HEAD", "--binary"))
    (run_dir / "cmd.txt").write_text(shlex.join(sys.argv) + "\n")
    (run_dir / "meta.txt").write_text(
        f"env={env_label(args, gpus)}\ncwd={os.getcwd()}\ngpus={','.join(gpus)}\nmode=sc-bench\n"
        f"battery={args.battery}\nckpt={args.ckpt}\n"
    )
    (run_dir / "gpus").write_text(",".join(gpus) + "\n")
    (run_dir / "started_at").write_text(now() + "\n")
    (run_dir / "STATUS").write_text("running\n")


def env_label(args: argparse.Namespace, gpus: list[str]) -> str:
    return f"sc-bench-g{len(gpus)}xK{args.clients_per_gpu}"


def record_result(
    args: argparse.Namespace,
    run_dir: Path,
    run_id: str,
    status: str,
    gpus: list[str],
    wall_s: float,
    metrics: str,
) -> None:
    tsv = args.results_tsv
    tsv.parent.mkdir(parents=True, exist_ok=True)
    ended = now()
    (run_dir / "ended_at").write_text(ended + "\n")
    sha = (run_dir / "git_sha").read_text().strip()[:12] if (run_dir / "git_sha").exists() else "-"
    with open(tsv.parent / ".results.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        new = not tsv.exists() or tsv.stat().st_size == 0
        with open(tsv, "a") as f:
            if new:
                f.write("run_id\tstatus\tgit_sha\tenv\tgpu_h\tended_at\tmetrics\n")
            gpu_h = wall_s * len(gpus) / 3600
            f.write(f"{run_id}\t{status}\t{sha}\t{env_label(args, gpus)}\t{gpu_h:.2f}\t{ended}\t{metrics}\n")


def take_gpu_lock(gpu: str, uuids: dict[str, str], enabled: bool) -> IO[str] | None:
    lock_dir = Path("/queue/locks")
    if not enabled or not lock_dir.is_dir() or gpu not in uuids:
        return None
    f = open(lock_dir / f"{uuids[gpu]}.lock", "a")  # noqa: SIM115
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[sc-bench] {now()} GPU {gpu}: waiting for {f.name}", flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
    return f


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="sc-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--battery", default="quick", help="quick | full | <path.yaml>")
    ap.add_argument("--gpus", default=None, help="physical GPU ids (default: $CUDA_VISIBLE_DEVICES or 0)")
    ap.add_argument(
        "--clients-per-gpu", type=int, default=1, help="workers (envs) per GPU; >1 = batched server"
    )
    ap.add_argument("--server-batch", type=int, default=None, help="default = --clients-per-gpu")
    ap.add_argument(
        "--out-root",
        default=None,
        help=f"runs/, results.tsv here (default $SC_OUT_ROOT or {registry.DEFAULT_OUT_ROOT})",
    )
    ap.add_argument("--registry", default=None, help="default <out-root>/ckpt_registry.tsv")
    ap.add_argument("--run-id", default=None, help="default bench-<ckpt_id>-<battery>")
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--tasks", default=None, help="override every job's tasks (e.g. 1,8)")
    ap.add_argument("--episodes", type=int, default=None, help="override every job's episodes")
    ap.add_argument("--suites", default=None, help="keep jobs of these suites only (libero_goal,...)")
    ap.add_argument(
        "--only", default=None, help="keep jobs matching any of these benchmark/suite/condition values"
    )
    ap.add_argument("--dry-run", action="store_true", help="plan + time estimate only (no GPU)")
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument(
        "--backend", choices=("libero", "mock"), default="libero", help="mock = CPU fake env/policy (tests)"
    )
    for k, v in vars(TimeModel()).items():
        ap.add_argument(f"--est-{k.replace('_', '-')}", type=type(v), default=v)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = registry.out_root(args.out_root)
    args.registry_path = Path(args.registry) if args.registry else registry.registry_path(root)
    args.results_tsv = root / "results.tsv"
    gpus = (args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")
    args.gpus = ",".join(gpus)
    if args.backend == "mock":
        ckpt = {
            "ckpt_id": registry.safe_id(args.ckpt.split("/")[-1]),
            "path_or_hub": args.ckpt,
            "adapter": None,
        }
    else:
        if registry.ensure_registry(args.registry_path):
            print(f"[sc-bench] created {args.registry_path} (pi05lib, pi05base)")
        ckpt = registry.resolve(args.ckpt, args.registry_path)
    battery = load_battery(args.battery)
    run_id = args.run_id or f"bench-{ckpt['ckpt_id']}-{battery['name']}"
    run_dir = root / "runs" / run_id
    ctx = Context(run_dir, ckpt, args.backend, REPO, {"tasks": args.tasks, "episodes": args.episodes})
    tm = TimeModel(**{k: getattr(args, f"est_{k}") for k in vars(TimeModel())})
    tm.n_action_steps = args.n_action_steps

    print(
        f"[sc-bench] ckpt {ckpt['ckpt_id']} = {ckpt['path_or_hub']} (adapter {ckpt.get('adapter')}), battery {battery['name']}, run {run_dir}"
    )
    check_bench_identity(run_dir, bench_identity(args, ctx, battery, None))
    plans = plan_all(args, ctx, battery)
    print_plan(plans, args, tm)
    if args.dry_run:
        return 0

    write_bookkeeping(run_dir, args, gpus)
    t_start = time.time()
    uuids = gpu_uuids()
    locks, servers, status = [], [], "failed"
    dispatcher: Dispatcher | None = None

    def on_signal(sig: int, _frame: Any) -> None:
        print(f"[sc-bench] signal {sig}: stopping", flush=True)
        if dispatcher is not None:
            dispatcher.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)
    try:
        for g in gpus:
            if args.backend != "mock":
                locks.append(take_gpu_lock(g, uuids, not args.no_gpu_lock))
            servers.append(ServerProc(g, args, ctx))
        readies = {s.gpu: s.wait_ready() for s in servers}
        hellos = {g: r["hello"] for g, r in readies.items()}
        h0 = hellos[gpus[0]]
        sigs = {
            g: tuple(h["exec"][k] for k in ("exec_mode", "gpu", "torch", "cuda")) for g, h in hellos.items()
        }
        if len(set(sigs.values())) > 1:
            raise SystemExit(
                f"servers differ in exec identity {sigs}; one run dir = one (exec_mode, GPU model)"
            )
        ident = bench_identity(args, ctx, battery, h0)
        check_bench_identity(run_dir, ident)
        prev = (
            json.loads((run_dir / "bench_meta.json").read_text())
            if (run_dir / "bench_meta.json").exists()
            else {}
        )
        (run_dir / "bench_meta.json").write_text(
            json.dumps(
                {
                    "identity": ident,
                    "created": prev.get("created", now()),
                    "resumed": [*prev.get("resumed", []), now()] if prev else [],
                    "battery_file": battery["_path"],
                    "battery": {k: v for k, v in battery.items() if not k.startswith("_")},
                    "ckpt_row": ckpt,
                    "gpus": gpus,
                    "clients_per_gpu": args.clients_per_gpu,
                    "hello": h0,
                    "jobs": [p.job_id for _, _, p in plans],
                },
                indent=2,
                default=str,
            )
        )
        units: list[dict[str, Any]] = []
        adapters = {}
        for job, ad, plan in plans:
            ad.prepare(job, plan, ctx, h0)
            adapters[ad.name] = ad
            common = plan.info.get("unit_common", {})
            units += [{**common, **u} for u in plan.units if u["n_pending"] > 0]
        urls = {g: r["url"] for g, r in readies.items()}
        dispatcher = Dispatcher(units, ctx, adapters, urls)
        print(
            f"[sc-bench] {now()} {len(units)} units pending on {len(gpus)} GPU × {args.clients_per_gpu} clients",
            flush=True,
        )
        threads = [
            threading.Thread(target=dispatcher.slot, args=(g, k), daemon=True)
            for g in gpus
            for k in range(args.clients_per_gpu)
        ]
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(1.0)
        status = "done" if not dispatcher.failed else "partial"
        if dispatcher.failed:
            (run_dir / "failed_units.json").write_text(json.dumps(dispatcher.failed, indent=2))
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        if dispatcher is not None:
            dispatcher.stop()
        for s in servers:
            s.stop()
        for lk in locks:
            if lk is not None:
                lk.close()
        summaries = {p.job_id: ad.summarize(ctx.job_dir(p.job_id)) for _, ad, p in plans}
        (run_dir / "jobs_summary.json").write_text(json.dumps(summaries, indent=2))
        (run_dir / "STATUS").write_text(status + "\n")
        wall = time.time() - t_start
        metrics = f"jobs={len(plans)};" + ";".join(f"{j}:{s}" for j, s in summaries.items())
        record_result(args, run_dir, run_id, status, gpus, wall, metrics[:4000])
        print(f"[sc-bench] {now()} {run_id}: {status} after {fmt_hours(wall / 3600)}", flush=True)
    return 0 if status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
