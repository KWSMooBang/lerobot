"""Multi-GPU shortcut eval: G GPUs × P workers × N envs, one shared work queue, T001 run dirs.

    uv run python scripts/shortcut/parallel/run_multi.py --ckpt pi05lib,pi05base --suites goal,spat \
        --conds C0,C1,C4 --repeats 0,1,2 --episodes 50 --gpus 0,1,2,3 [--procs-per-gpu auto] \
        [--num-envs auto] [--out-root ./outputs/VLA-shortcut-exps] [--dry-run]

Work unit = (ckpt, suite, condition, variant, task, init) (+ repeat). The coordinator plans every run
like run_eval.py (same identity guard, same seeds), writes the pending units to a queue and starts G×P
workers, each pinned to one GPU (CUDA_VISIBLE_DEVICES, MUJOCO_EGL_DEVICE_ID). A worker runs N env
subprocesses and batches the policy (parallel/batched.py), writes steps/ and videos/ into the run dir and
one line per episode to <run>/parts/<worker>-<pid>.jsonl. A run whose units are all done is merged into
episodes.jsonl in (task, init, variant) order and its parts/ is removed, so the run dir looks like a
T001 run. Restarting the same command skips finished units (the records in episodes.jsonl and parts/).

exec_mode = "single" when N=1 (batch-1 calls, same noise stream as run_eval.py), else "batched-N"
("+compile" with --compile). It is part of the run identity; metrics.py does not pair across exec modes.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import multiprocessing as mp
import os
import shlex
import shutil
import signal
import subprocess  # nosec B404
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parent
REPO = SHORTCUT.parents[1]
sys.path[:0] = [p for p in (str(HERE), str(SHORTCUT)) if p not in sys.path]

import run_eval  # noqa: E402
from common import DUAL_GOAL_CONDITIONS, SUITE_SHORT, Variant, episode_seed, video_relpath  # noqa: E402
from resources import ThroughputModel, choose, cpu_count, gpu_info  # noqa: E402
from workqueue import WorkQueue, write_queue  # noqa: E402

CKPT_TAGS = {"pi05lib": "lerobot/pi05_libero_finetuned", "pi05base": "lerobot/pi05_base"}
SUITE_LONG = {v: k for k, v in SUITE_SHORT.items()}
FATAL_EXIT = 3  # worker refuses (wrong GPU model, ...): do not respawn
MAX_ATTEMPTS = 3
EARLY_STOP_FRAC = 0.45  # pi05lib C0–C3 end at ~0.4 x max_steps (T001 smoke: goal C0 5.5 s of 13.5 s)


# ---------------------------------------------------------------- planning


@dataclass
class PlannedRun:
    run_id: str
    spec: run_eval.RunSpec
    units: list[dict[str, Any]]  # every unit of the run, done or not, in merge order
    done: set[tuple]  # (task, cond, variant, init, repeat) with a record already
    group: str = ""
    order: dict[tuple, tuple] = field(default_factory=dict)


def exec_mode_of(num_envs: int, compile_model: bool) -> str:
    mode = "single" if num_envs == 1 else f"batched-{num_envs}"
    return mode + ("+compile" if compile_model else "")


def record_key(r: dict[str, Any]) -> tuple:
    return (r["task_id"], r["condition"], r["variant"], r["init_idx"], r["repeat_idx"])


def read_records(path: Path) -> list[dict[str, Any]]:
    """Episode lines of one file; a torn last line (worker killed mid-write) is skipped."""
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[warn] {path}: skipping a partial line", flush=True)
    return out


def part_files(out: Path) -> list[Path]:
    return sorted((out / "parts").glob("*.jsonl"), key=lambda p: (p.stat().st_mtime, p.name))


def plan_run(
    run_id: str, run_args: argparse.Namespace, exec_mode: str, gpu: str, tokenizer: Any
) -> PlannedRun:
    spec = run_eval.prepare_run(run_args, exec_mode, gpu)
    done = spec.done_keys()
    for p in part_files(spec.out):
        done |= {record_key(r) for r in read_records(p)}
    a = run_args
    if a.backend == "mock":
        n_init = dict.fromkeys(spec.instr_set["_suite_tasks"], 50)
    else:
        from lerobot.envs.libero import get_task_init_states

        n_init = {t: len(get_task_init_states(spec.suite, t)) for t in spec.instr_set["_suite_tasks"]}
    units = []
    order = {}
    for task_id, variants in run_eval.plan_variants(spec, tokenizer):
        if a.episodes > n_init[task_id]:
            raise ValueError(f"{run_id}: task {task_id} has only {n_init[task_id]} init states")
        for init_idx in range(a.episodes):
            for vpos, v in enumerate(variants):
                key = (task_id, v.condition, v.variant, init_idx, a.repeat_idx)
                order[key] = (task_id, init_idx, vpos)
                units.append(
                    {
                        "uid": f"{run_id}/t{task_id:02d}/{v.variant}/i{init_idx:02d}",
                        "run_id": run_id,
                        "suite": a.suite,
                        "task_id": task_id,
                        "init_idx": init_idx,
                        "vpos": vpos,
                        "condition": v.condition,
                        "repeat_idx": a.repeat_idx,
                        "variant": asdict(v),
                        "ep_seed": episode_seed(a.seed, a.repeat_idx, task_id, init_idx),
                        "record_video": spec.video_cap is None or init_idx < spec.video_cap,
                        "max_steps": spec.max_steps,
                        "tag": spec.tag,
                        "rel": video_relpath(
                            a.suite, task_id, v.condition, v.variant, init_idx, a.repeat_idx
                        ),
                        "dual": v.condition in DUAL_GOAL_CONDITIONS,
                        "done": key in done,
                    }
                )
    return PlannedRun(run_id, spec, units, done, order=order)


def group_of(p: PlannedRun) -> dict[str, Any]:
    a = p.spec.args
    return {
        "backend": a.backend,
        "policy": a.policy,
        "ckpt": p.spec.ckpt,
        "adapter": p.spec.adapter,
        "n_action_steps": a.n_action_steps,
        "suite": a.suite,  # suite used to build the policy (LIBERO features do not depend on it)
    }


def assign_groups(plans: list[PlannedRun]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    keys: dict[str, str] = {}
    for p in plans:
        g = group_of(p)
        k = json.dumps({x: g[x] for x in g if x != "suite"}, sort_keys=True)
        if k not in keys:
            keys[k] = f"g{len(keys)}"
            groups[keys[k]] = g
        p.group = keys[k]
        for u in p.units:
            u["group"] = p.group
    return groups


def queue_order(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group, then (suite, task) for env reuse, long (no early stop) episodes first within a task."""
    return sorted(
        units,
        key=lambda u: (
            u["group"],
            u["suite"],
            u["task_id"],
            not u["dual"],
            u["run_id"],
            u["init_idx"],
            u["vpos"],
        ),
    )


def est_steps(u: dict[str, Any], adapter: bool) -> float:
    """Expected env steps: max_steps for C4–C6 and for the base checkpoint (rarely succeeds), else an
    early stop at EARLY_STOP_FRAC (pi05lib C0–C3). +10 settle steps per reset."""
    full = u["dual"] or adapter
    return (u["max_steps"] if full else EARLY_STOP_FRAC * u["max_steps"]) + 10


# ---------------------------------------------------------------- merge


def merge_run(p: PlannedRun, failed: set[str]) -> tuple[int, bool]:
    """parts/*.jsonl + episodes.jsonl → episodes.jsonl in (task, init, variant) order. Returns
    (records, complete). A complete run loses its parts/ dir (T001 layout)."""
    out = p.spec.out
    recs: dict[tuple, dict[str, Any]] = {}
    dups = 0
    for src in [out / "episodes.jsonl", *part_files(out)]:
        for r in read_records(src):
            k = record_key(r)
            dups += k in recs
            recs[k] = r  # later files win: their steps/ and videos/ are the ones on disk
    if dups:
        print(f"[warn] {p.run_id}: {dups} duplicate episode records (a unit ran twice); kept the latest")
    far = (1 << 30, 1 << 30, 1 << 30)
    ordered = sorted(recs.items(), key=lambda kv: (p.order.get(kv[0], far), kv[0][2]))
    tmp = out / "episodes.jsonl.tmp"
    with open(tmp, "w") as f:
        for _, r in ordered:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out / "episodes.jsonl")
    want = set(p.order)
    complete = want <= set(recs) and not any(u["uid"] in failed for u in p.units)
    if complete and (out / "parts").exists():
        shutil.rmtree(out / "parts")
    return len(recs), complete


# ---------------------------------------------------------------- worker


class EpisodeWriter:
    """Writes steps/, videos/ and the part-file line off the GPU loop, then marks the unit done."""

    def __init__(self, session: dict[str, Any], queue: WorkQueue, wid: str, max_pending: int):
        self.session, self.queue, self.wid = session, queue, wid
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="writer")
        self.sem = threading.BoundedSemaphore(max_pending)
        self.lock = threading.Lock()
        self.files: dict[str, IO[str]] = {}

    def _part(self, run_id: str) -> IO[str]:
        if run_id not in self.files:
            parts = Path(self.session["runs"][run_id]["out"]) / "parts"
            parts.mkdir(parents=True, exist_ok=True)
            self.files[run_id] = open(parts / f"{self.wid}-{os.getpid()}.jsonl", "a")  # noqa: SIM115
        return self.files[run_id]

    def submit(self, unit: dict[str, Any], res: dict[str, Any], wall: float) -> None:
        self.sem.acquire()
        fut = self.pool.submit(self._write, unit, res, wall)
        fut.add_done_callback(lambda _: self.sem.release())

    def _write(self, unit: dict[str, Any], res: dict[str, Any], wall: float) -> None:
        try:
            run = self.session["runs"][unit["run_id"]]
            out = Path(run["out"])
            steps_file, video_file = run_eval.save_episode(out, unit["rel"], res, unit["record_video"])
            rec = run_eval.episode_record(
                run,
                unit["task_id"],
                Variant(**unit["variant"]),
                unit["init_idx"],
                unit["ep_seed"],
                res,
                steps_file,
                video_file,
                wall,
            )
            line = run_eval.episode_line(rec)
            with self.lock:
                f = self._part(unit["run_id"])
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
                with open(out / "stdout.log", "a") as log:
                    log.write(f"[{self.wid}] {line}\n")
            print(f"[{self.wid}] {unit['run_id']} {line}", flush=True)
            self.queue.mark(unit["uid"], "done")
        except Exception:
            print(f"[{self.wid}] write failed for {unit['uid']}:\n{traceback.format_exc()}", flush=True)
            self.queue.mark(unit["uid"], "fail")

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for f in self.files.values():
            f.close()


def worker_main(args: argparse.Namespace) -> int:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    import torch
    from batched import BatchedRunner, Slot, Stats
    from bundle import load_bundle
    from envproc import EnvClient

    qdir = Path(args.queue)
    session = json.loads((qdir / "session.json").read_text())
    backend = session["backend"]
    device = "cpu" if backend == "mock" else "cuda"
    if backend != "mock":
        name = torch.cuda.get_device_name(0)
        if name != session["gpu"]:
            print(f"[{args.wid}] GPU is {name!r}, run identity says {session['gpu']!r}; refusing", flush=True)
            return FATAL_EXIT
    torch.set_num_threads(2)
    q = WorkQueue(qdir, args.wid)
    n = int(session["num_envs"])
    ctx = mp.get_context("spawn")
    slots = [Slot(i, EnvClient(ctx, backend, f"{args.wid}-env{i}")) for i in range(n)]

    def restart(i: int) -> EnvClient:
        print(f"[{args.wid}] env {i} died; restarting it", flush=True)
        slots[i].client.close()
        return EnvClient(ctx, backend, f"{args.wid}-env{i}")

    def on_fail(unit: dict[str, Any], err: str) -> None:
        print(f"[{args.wid}] unit {unit['uid']} failed:\n{err}", flush=True)
        out = Path(session["runs"][unit["run_id"]]["out"])
        with open(out / "stdout.log", "a") as log:
            log.write(f"[{args.wid}] [fail] {unit['uid']}: {err.strip().splitlines()[-1]}\n")
        q.mark(unit["uid"], "fail")

    writer = EpisodeWriter(session, q, args.wid, max_pending=max(2, n))
    crash_uid = os.environ.get("SC_TEST_CRASH_UID") if backend == "mock" else None
    stats = Stats()
    bundle: Any = None
    current: str | None = None
    groups = q.pending_groups()
    prefer = groups[args.index % len(groups)] if groups else None
    try:
        while True:
            groups = q.pending_groups()
            if not groups:
                break
            g: str = current if current in groups else (prefer if prefer in groups else groups[0])
            if g != current:
                if bundle is not None:
                    print(f"[{args.wid}] group {current} drained; stealing {g}", flush=True)
                bundle = None
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                t0 = time.time()
                bundle = load_bundle(session["groups"][g], device, session["compile"])
                current = g
                print(
                    f"[{args.wid}] loaded {g} ({session['groups'][g]['ckpt']}) in {time.time() - t0:.0f}s",
                    flush=True,
                )
                inf = qdir / "inference" / f"{g}.json"
                inf.parent.mkdir(exist_ok=True)
                if not inf.exists():
                    inf.write_text(json.dumps(bundle.inf_meta, indent=2, default=str))

            def claim(loc: tuple | None, group: str = g) -> dict[str, Any] | None:
                unit = q.claim(group, loc)
                if unit is not None and unit["uid"] == crash_uid and q.attempts[unit["uid"]] == 1:
                    os._exit(9)  # test hook: die holding a claim
                return unit

            runner = BatchedRunner(
                bundle,
                slots,
                claim=claim,
                on_done=writer.submit,
                on_fail=on_fail,
                restart=restart,
                pad=session["pad"],
                stats=stats,
            )
            runner.run()
            print(f"[{args.wid}] {stats.line()}", flush=True)
    finally:
        writer.close()
        for s in slots:
            s.client.close()
    if device == "cuda":
        print(
            f"[{args.wid}] max GPU memory allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
            flush=True,
        )
    stats_path = qdir / "stats" / f"{args.wid}-{os.getpid()}.json"
    stats_path.parent.mkdir(exist_ok=True)
    stats_path.write_text(json.dumps(asdict(stats)))
    return 0


# ---------------------------------------------------------------- coordinator


@dataclass
class Opts:
    gpus: list[str]
    procs_per_gpu: int
    num_envs: int
    compile: bool
    backend: str
    session_dir: Path
    pad: bool = True
    bookkeeping: bool = False  # git_sha / cmd.txt / STATUS / results.tsv per run (submit_multi.sh)
    results_tsv: Path | None = None
    lock_gpus: bool = True
    max_restarts: int = 3
    cmd: str = ""
    reasons: list[str] = field(default_factory=list)


def git(*a: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *a], text=True)  # nosec B603 B607
    except Exception:
        return ""


def write_bookkeeping(p: PlannedRun, opts: Opts) -> None:
    out = p.spec.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "git_sha").write_text(git("rev-parse", "HEAD"))
    (out / "git_dirty").write_text(git("status", "--porcelain"))
    (out / "patch.diff").write_text(git("diff", "HEAD", "--binary"))
    (out / "cmd.txt").write_text(opts.cmd + "\n")
    env = f"multi-g{len(opts.gpus)}xP{opts.procs_per_gpu}xN{opts.num_envs}"
    (out / "meta.txt").write_text(
        f"env={env}\ncwd={os.getcwd()}\ngpus={','.join(opts.gpus)}\nmode=multi\n"
        f"exec_mode={p.spec.identity['exec_mode']}\n"
    )
    (out / "STATUS").write_text("running\n")


def finalize_run(p: PlannedRun, opts: Opts, status: str, session: dict[str, Any]) -> None:
    out = p.spec.out
    meta_path = out / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    inf = opts.session_dir / "queue" / "inference" / f"{p.group}.json"
    if inf.exists() and meta.get("inference") is None:
        meta["inference"] = json.loads(inf.read_text())
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
    if not opts.bookkeeping:
        return
    (out / "STATUS").write_text(status + "\n")
    ended = datetime.now().astimezone().isoformat(timespec="seconds")
    (out / "ended_at").write_text(ended + "\n")
    eps = read_records(out / "episodes.jsonl")
    n = len(eps)
    metrics = "n=0"
    if n:
        metrics = f"n={n};sr_orig={sum(bool(e['success_orig']) for e in eps) / n:.3f}"
        dual = [e for e in eps if e.get("success_new") is not None]
        if dual:
            metrics += f";sr_new={sum(bool(e['success_new']) for e in dual) / len(dual):.3f}"
    # GPU share (approximate): a GPU serves P x N slots at once; wall_time_s is slot occupancy
    slot_s = sum(e.get("wall_time_s", 0.0) for e in eps)
    gpu_h = slot_s / (opts.procs_per_gpu * opts.num_envs) / 3600
    if opts.results_tsv is not None:
        tsv = opts.results_tsv
        tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(tsv.parent / ".results.lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            new = not tsv.exists() or tsv.stat().st_size == 0
            with open(tsv, "a") as f:
                if new:
                    f.write("run_id\tstatus\tgit_sha\tenv\tgpu_h\tended_at\tmetrics\n")
                env = f"multi-g{len(opts.gpus)}xP{opts.procs_per_gpu}xN{opts.num_envs}"
                sha = (out / "git_sha").read_text().strip()[:12] if (out / "git_sha").exists() else "-"
                f.write(f"{p.run_id}\t{status}\t{sha}\t{env}\t{gpu_h:.2f}\t{ended}\t{metrics}\n")


def take_gpu_locks(opts: Opts, info: dict[str, dict[str, Any]]) -> list[IO[str]]:
    """Hold /queue/locks/<uuid>.lock (lab servers) for the session; absent elsewhere."""
    lock_dir = Path("/queue/locks")
    if opts.backend == "mock" or not opts.lock_gpus or not lock_dir.is_dir():
        return []
    held: list[IO[str]] = []
    for g in opts.gpus:
        f = open(lock_dir / f"{info[g]['uuid']}.lock", "a")  # noqa: SIM115
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"GPU {g} ({info[g]['uuid']}) is locked by another job; not starting") from None
        held.append(f)
    return held


def spawn_worker(wid: str, index: int, gpu: str, opts: Opts) -> tuple[subprocess.Popen, IO[str]]:
    env = dict(os.environ)
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=gpu,
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID=gpu,
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
    )
    if opts.backend == "mock":
        env["CUDA_VISIBLE_DEVICES"] = ""  # fake ids: CPU only
        env.pop("MUJOCO_EGL_DEVICE_ID")
    log = open(opts.session_dir / f"{wid}.log", "a")  # noqa: SIM115
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--wid", wid, "--index", str(index)]
    cmd += ["--queue", str(opts.session_dir / "queue")]
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)  # nosec B603
    return proc, log


def execute(plans: list[PlannedRun], opts: Opts) -> int:
    """Run all pending units of `plans` and merge. Returns 0 when every run is complete."""
    groups = assign_groups(plans)
    pending = queue_order([u for p in plans for u in p.units if not u["done"]])
    gpu = plans[0].spec.identity["gpu"]
    info = {} if opts.backend == "mock" else gpu_info(opts.gpus)
    locks = take_gpu_locks(opts, info)
    opts.session_dir.mkdir(parents=True, exist_ok=True)
    qdir = opts.session_dir / "queue"
    if qdir.exists():
        shutil.rmtree(qdir)
    write_queue(qdir, pending)
    session = {
        "backend": opts.backend,
        "gpu": gpu,
        "num_envs": opts.num_envs,
        "pad": opts.pad,
        "compile": opts.compile,
        "groups": groups,
        "runs": {p.run_id: p.spec.info() for p in plans},
    }
    (qdir / "session.json").write_text(json.dumps(session, indent=1, default=str))
    exec_info = {
        "gpus": opts.gpus,
        "procs_per_gpu": opts.procs_per_gpu,
        "num_envs": opts.num_envs,
        "pad_batch": opts.pad,
        "compile": opts.compile,
        "cpu_cores": cpu_count(),
        "auto": opts.reasons,
        "session_dir": str(opts.session_dir),
    }
    for p in plans:
        meta_path = p.spec.out / "run_meta.json"
        prev = json.loads(meta_path.read_text()) if meta_path.exists() else None
        p.spec.out.mkdir(parents=True, exist_ok=True)
        meta = run_eval.run_meta(p.spec, None, prev, exec=exec_info)
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        if opts.bookkeeping:
            write_bookkeeping(p, opts)
    print(
        f"[multi] {len(pending)} pending units in {len(plans)} runs, {len(groups)} policy groups", flush=True
    )
    for line in opts.reasons:
        print(f"[multi] {line}", flush=True)

    q = WorkQueue(qdir, "coord")
    workers: dict[str, dict[str, Any]] = {}
    index = 0
    for g in opts.gpus:
        for k in range(opts.procs_per_gpu):
            wid = f"w{g}-{k}"
            proc, log = spawn_worker(wid, index, g, opts)
            workers[wid] = {
                "proc": proc,
                "log": log,
                "gpu": g,
                "index": index,
                "restarts": 0,
                "state": "running",
            }
            index += 1
    stop: dict[str, int | None] = {"sig": None}

    def on_signal(sig: int, _frame: Any) -> None:
        stop["sig"] = sig

    old = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    finalized: set[str] = set()
    t_start = time.time()
    last_report = 0.0
    n0 = len(pending)
    try:
        while True:
            if stop["sig"] is not None:
                print(f"[multi] signal {stop['sig']}: stopping workers", flush=True)
                for w in workers.values():
                    if w["state"] == "running":
                        w["proc"].terminate()
                for w in workers.values():
                    w["proc"].wait(60)
                break
            for wid, w in workers.items():
                if w["state"] != "running":
                    continue
                rc = w["proc"].poll()
                if rc is None:
                    continue
                if rc == 0:
                    w["state"] = "finished"
                    continue
                released, failed = q.release_worker(wid, MAX_ATTEMPTS)
                print(
                    f"[multi] {wid} exited {rc}: released {len(released)}, failed {len(failed)} units "
                    f"(log {opts.session_dir / (wid + '.log')})",
                    flush=True,
                )
                if rc == FATAL_EXIT or w["restarts"] >= opts.max_restarts:
                    w["state"] = "dead"
                    continue
                w["restarts"] += 1
                w["log"].close()
                w["proc"], w["log"] = spawn_worker(wid, w["index"], w["gpu"], opts)
            snap = q.snapshot()
            open_runs = {q.units[u]["run_id"] for u in snap["pending"] + snap["claim"]}
            failed_uids = set(snap["fail"])
            for p in plans:
                if p.run_id in finalized or p.run_id in open_runs:
                    continue
                n, complete = merge_run(p, failed_uids)
                finalize_run(p, opts, "done" if complete else "failed", session)
                finalized.add(p.run_id)
                print(f"[multi] {p.run_id}: {'done' if complete else 'FAILED'} ({n} episodes)", flush=True)
            now = time.time()
            if now - last_report > 60:
                k = len(snap["done"])
                rate = k / max(now - t_start, 1e-6)
                eta = (n0 - k - len(snap["fail"])) / rate / 3600 if rate > 0 else float("nan")
                print(
                    f"[multi] {k}/{n0} done, {len(snap['claim'])} running, {len(snap['fail'])} failed, "
                    f"{60 * rate:.1f} ep/min, eta {eta:.2f} h",
                    flush=True,
                )
                last_report = now
            if all(w["state"] != "running" for w in workers.values()):
                break
            time.sleep(0.5 if opts.backend == "mock" else 5)
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        for w in workers.values():
            if w["proc"].poll() is None:
                w["proc"].kill()
            w["log"].close()
        snap = q.snapshot()
        for p in plans:
            if p.run_id in finalized:
                continue
            n, complete = merge_run(p, set(snap["fail"]))
            status = "done" if complete else ("stopped" if stop["sig"] is not None else "failed")
            finalize_run(p, opts, status, session)
            print(f"[multi] {p.run_id}: {status} ({n} episodes)", flush=True)
        for f in locks:
            f.close()
    snap = q.snapshot()
    left = len(snap["pending"]) + len(snap["claim"]) + len(snap["fail"])
    print(f"[multi] finished in {(time.time() - t_start) / 3600:.2f} h; {left} units not done", flush=True)
    return 0 if left == 0 and stop["sig"] is None else 1


# ---------------------------------------------------------------- CLI


def run_args_for(
    a: argparse.Namespace, tag: str, suite: str, cond: str, repeat: int, out: Path
) -> argparse.Namespace:
    ckpt = f"mock/{tag}" if a.backend == "mock" else CKPT_TAGS[tag]
    instr = a.instr_set or str(Path(a.instr_dir) / f"{suite}.json")
    argv = ["--policy", "pi05", "--checkpoint", ckpt, "--model-tag", tag, "--suite", suite]
    argv += ["--condition", cond, "--instr-set", instr, "--tasks", a.tasks, "--episodes", str(a.episodes)]
    argv += ["--repeat-idx", str(repeat), "--seed", str(a.seed), "--out", str(out), "--backend", a.backend]
    argv += ["--n-action-steps", str(a.n_action_steps)]
    if a.video:
        argv += ["--video", a.video]
    if a.max_steps:
        argv += ["--max-steps", str(a.max_steps)]
    if a.variants:
        argv += ["--variants", a.variants]
    if a.include_infeasible:
        argv.append("--include-infeasible")
    return run_eval.build_parser().parse_args(argv)


def expand_runs(a: argparse.Namespace) -> list[tuple[str, argparse.Namespace]]:
    runs = []
    for tag in a.ckpt.split(","):
        if a.backend != "mock" and tag not in CKPT_TAGS:
            raise SystemExit(f"unknown ckpt tag {tag} ({'|'.join(CKPT_TAGS)})")
        for s in a.suites.split(","):
            if s not in SUITE_LONG:
                raise SystemExit(f"unknown suite {s} ({'|'.join(SUITE_LONG)})")
            for cond in a.conds.split(","):
                reps = [int(r) for r in a.repeats.split(",")] if cond == "C0" else [0]
                for k in reps:
                    rid = f"{a.id_prefix}-{tag}-{s}-{cond}" + (f"-r{k}" if k > 0 else "")
                    runs.append(
                        (rid, run_args_for(a, tag, SUITE_LONG[s], cond, k, Path(a.out_root) / "runs" / rid))
                    )
    return runs


def load_tokenizer_for(backend: str) -> Any:
    if backend == "mock":
        from mock_backend import MockTokenizer

        return MockTokenizer()
    from common import load_tokenizer

    return load_tokenizer()


def resolve_exec(a: argparse.Namespace) -> tuple[int, int, list[str], str]:
    gpus = a.gpus.split(",")
    info = {} if a.backend == "mock" else gpu_info(gpus)
    if a.backend != "mock":
        missing = [g for g in gpus if g not in info]
        if missing:
            raise SystemExit(f"--gpus {missing} not found by nvidia-smi")
        names = {info[g]["name"] for g in gpus}
        if len(names) > 1:
            raise SystemExit(f"--gpus mixes GPU models {sorted(names)}; one run identity needs one model")
        gpu = names.pop()
        mem = min(info[g]["mem_gb"] for g in gpus)
    else:
        gpu, mem = "cpu", None
    p, n, why = choose(gpus, a.procs_per_gpu, a.num_envs, mem)
    return p, n, why, gpu


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="pi05lib", help="comma list: pi05lib,pi05base")
    ap.add_argument("--suites", default="goal", help="comma list: spat,obj,goal,l10")
    ap.add_argument("--conds", default="C0")
    ap.add_argument("--repeats", default="0", help="C0 repeats (other conditions: repeat 0)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--video", default=None, help="all|none|first:N (default as run_eval.py)")
    ap.add_argument("--id-prefix", default="sc")
    ap.add_argument("--out-root", default="./outputs/VLA-shortcut-exps")
    ap.add_argument("--instr-dir", default=str(SHORTCUT / "instructions"))
    ap.add_argument("--instr-set", default=None, help="one instruction file for every suite (tests)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--include-infeasible", action="store_true")
    ap.add_argument("--gpus", default="0", help="comma list of GPU ids (nvidia-smi / PCI order)")
    ap.add_argument("--procs-per-gpu", default="auto")
    ap.add_argument("--num-envs", default="auto")
    ap.add_argument("--compile", action="store_true", help="torch.compile the policy (off until checked)")
    ap.add_argument("--no-pad", action="store_true", help="do not pad policy batches to --num-envs rows")
    ap.add_argument("--no-gpu-lock", action="store_true", help="do not take /queue/locks/<uuid>.lock")
    ap.add_argument("--backend", choices=run_eval.BACKENDS, default="libero")
    ap.add_argument("--results-tsv", default=None, help="default <out-root>/results.tsv")
    ap.add_argument("--no-bookkeeping", action="store_true", help="no git_sha/cmd.txt/STATUS/results.tsv")
    ap.add_argument("--dry-run", action="store_true", help="plan, count units, estimate time; run nothing")
    for k, v in asdict(ThroughputModel()).items():
        ap.add_argument(f"--est-{k.replace('_', '-')}", type=float, default=v, help=argparse.SUPPRESS)
    # worker mode (spawned by the coordinator)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--wid", help=argparse.SUPPRESS)
    ap.add_argument("--index", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--queue", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.worker:
        return worker_main(a)
    p, n, why, gpu = resolve_exec(a)
    exec_mode = exec_mode_of(n, a.compile)
    tokenizer = load_tokenizer_for(a.backend)
    plans = [plan_run(rid, ra, exec_mode, gpu, tokenizer) for rid, ra in expand_runs(a)]
    model = ThroughputModel(**{k: getattr(a, f"est_{k}") for k in asdict(ThroughputModel())})
    if a.dry_run:
        total = steps = 0.0
        print(f"exec_mode={exec_mode} gpu={gpu} gpus={a.gpus}")
        for line in why:
            print(f"  {line}")
        print(f"{'run_id':32s} {'units':>6s} {'done':>6s} {'todo':>6s} {'est steps':>10s}")
        for pl in plans:
            todo = [u for u in pl.units if not u["done"]]
            st = sum(est_steps(u, pl.spec.adapter is not None) for u in todo)
            total += len(todo)
            steps += st
            print(
                f"{pl.run_id:32s} {len(pl.units):6d} {len(pl.units) - len(todo):6d} {len(todo):6d} {st:10.0f}"
            )
        n_groups = len({json.dumps(group_of(pl)["ckpt"]) for pl in plans})
        print(f"total: {int(total)} units to run, {steps / 1e6:.2f} M env-steps (expected)")
        for line in model.report(steps, len(a.gpus.split(",")), p, n, a.n_action_steps, n_groups):
            print(f"  {line}")
        return 0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    opts = Opts(
        gpus=a.gpus.split(","),
        procs_per_gpu=p,
        num_envs=n,
        compile=a.compile,
        backend=a.backend,
        session_dir=Path(a.out_root) / "multi" / f"{stamp}-{os.getpid()}",
        pad=not a.no_pad,
        bookkeeping=not a.no_bookkeeping,
        results_tsv=None if a.no_bookkeeping else Path(a.results_tsv or Path(a.out_root) / "results.tsv"),
        lock_gpus=not a.no_gpu_lock,
        cmd=shlex.join([sys.executable, *sys.argv]),
        reasons=why,
    )
    return execute(plans, opts)


def run_single_dir(args: argparse.Namespace) -> None:
    """run_eval.py --num-envs N: this one run on the visible GPU (one worker), same --out layout."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    gpus = [visible]
    gpu = "cpu" if args.backend == "mock" else run_eval.gpu_name("cuda")
    if args.backend != "mock":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # the worker gets the physical id
    exec_mode = exec_mode_of(args.num_envs, args.compile)
    plan = plan_run(Path(args.out).name, args, exec_mode, gpu, load_tokenizer_for(args.backend))
    opts = Opts(
        gpus=gpus,
        procs_per_gpu=1,
        num_envs=args.num_envs,
        compile=args.compile,
        backend=args.backend,
        session_dir=Path(args.out) / f".session-{os.getpid()}",
        reasons=[f"run_eval.py --num-envs {args.num_envs} on GPU {visible}"],
    )
    rc = execute([plan], opts)
    if rc:
        raise SystemExit(f"{rc}: not every episode finished; logs in {opts.session_dir}")
    shutil.rmtree(opts.session_dir)  # same layout as a single-env run dir


if __name__ == "__main__":
    sys.exit(main())
