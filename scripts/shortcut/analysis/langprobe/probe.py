"""T018 step 2: ask each policy server for one action chunk per (observation, instruction). GPU side.

    uv run --extra shortcut-bench python scripts/shortcut/analysis/langprobe/probe.py \
        --models pi05droid,pi05lib,pi05base,starvla-qwen3pi-libero --obs-dir <obs> --out <run dir> --gpu 0
    ... --dry-run [--est-ms pi05lib=180,...]     forwards per model × ms → GPU-h (no server, no GPU)
    ... --url ws://127.0.0.1:8765 --models pi05lib   query a server that is already running (e.g. a CPU one)

Servers: the sc-bench way (bench/core/runner.py ServerProc): registry id → lerobot serve/policy_server.py, or
serve/external/<server>/serve.sh for rows with `server=` (starvla, droid); exec_mode single (--batch 1), one
model at a time (each server is stopped before the next one loads: the --stagger-start reason, StarVLA peaks at
~40 GB CPU RAM while loading). n_action_steps: the sc-bench default (10 lerobot, registry notes for external).
The GPU's /queue/locks/<uuid>.lock is held while servers run (--no-gpu-lock to skip).

Per (observation, instruction) query: reset(episode_seed) → infer once → end. Within an observation every
instruction except orig_s1/orig_s2 uses the same seed, so the noise is identical and only the text differs
(lp_common.queries_for). The server returns the post-processed (unnormalized, env-scale) actions of the first
n_action_steps of the chunk in its action space (hello: osc_pose_delta [n, 7] or joint_velocity [n, 8]).
Images: the set's render size closest to hello image_hw (set A has 256 and 360; set B 256). A set without
joint_pos (B) is skipped for servers that need it (joint_velocity: pi05droid).
--repeat-check K: orig_s0 is asked twice for the first K observations (kind "repeat"): same seed + same input
must give the same chunk (max |Δ| in probe_meta.json; GPU kernels may differ in the last bits).

Output <out>/<model>/: hello.json, probe_<set>.jsonl (one row per query; resumable: rerun = skip done rows),
probe_<set>.parquet (written at the end of each set), probe_meta.json (counts, server_ms median, wall time),
servers/gpu<g>.log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parents[1]
REPO = SHORTCUT.parents[1]
sys.path[:0] = [p for p in (str(HERE), str(SHORTCUT)) if p not in sys.path]

import lp_common as lp  # noqa: E402
from bench.core import registry  # noqa: E402
from bench.core.client import PolicyClient  # noqa: E402
from bench.core.protocol import (  # noqa: E402
    ACTION_SPACE_JOINT_VELOCITY,
    ACTION_SPACE_OSC,
    hello_action_space,
)
from bench.core.records import append_jsonl, read_jsonl  # noqa: E402

# Placeholder ms per forward (server_ms, single, A100) for --dry-run until measured; probe_meta.json records the
# measured median (server_ms_median) of every run: pass it back with --est-ms model=ms.
EST_MS = {"pi05lib": 250.0, "pi05base": 250.0, "pi05droid": 250.0, "starvla-qwen3pi-libero": 350.0}
EST_MS_DEFAULT = 300.0
EST_LOAD_S = {"starvla-qwen3pi-libero": 300.0}  # server start → ready (model download not included)
EST_LOAD_S_DEFAULT = 180.0
IDENTITY_KEYS = ("ckpt", "ckpt_id", "exec_mode", "n_action_steps", "action_space", "protocol_version")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- planning


def select_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    suites = set(args.suites.split(",")) if args.suites else None
    tasks = {int(t) for t in args.tasks.split(",")} if args.tasks else None
    out = [
        r
        for r in rows
        if (suites is None or r["suite"] in suites) and (tasks is None or int(r["task_id"]) in tasks)
    ]
    if args.max_obs_per_task:
        seen: dict[tuple, int] = {}
        keep = []
        for r in out:
            k = (r["suite"], r["task_id"])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= args.max_obs_per_task:
                keep.append(r)
        out = keep
    return out


def plan_queries(
    set_name: str, rows: list[dict], sets: dict[str, Any], repeat_k: int, instr_ids: set[str] | None = None
) -> list[dict]:
    """Every (observation, instruction) query of a set, in order."""
    out = []
    for n, r in enumerate(rows):
        qs = lp.queries_for(lp.task_entry(sets, r["suite"], r["task_id"]))
        if instr_ids:
            qs = [q for q in qs if q["instr_id"] in instr_ids or q["instr_id"] == "orig_s0"]
        if n < repeat_k:
            q0 = next(q for q in qs if q["instr_id"] == "orig_s0")
            qs = [*qs, {**q0, "instr_id": "orig_s0_repeat", "kind": "repeat"}]
        for q in qs:
            out.append({**q, "set": set_name, "obs_idx": int(r["obs_idx"]), "obs_id": r["obs_id"],
                        "suite": r["suite"], "task_id": int(r["task_id"]),
                        "seed": lp.obs_seed(set_name, r, int(q["rep"]))})  # fmt: skip
    return out


def instr_filter(args: argparse.Namespace) -> set[str] | None:
    return set(args.instr_ids.split(",")) if args.instr_ids else None


def ckpt_row(model: str, args: argparse.Namespace) -> dict[str, Any]:
    """As sc-bench's main(): registry id / path / hub id → row with `server` and LIBERO `adapter`."""
    builtin = {r["ckpt_id"]: r for r in registry.BUILTIN_ROWS}
    if args.backend == "mock":
        notes = registry.parse_notes(builtin.get(model, {}).get("notes"))
        return {"ckpt_id": registry.safe_id(model.split("/")[-1]), "path_or_hub": model, "adapter": None,
                "server": notes.get("server"), "notes": builtin.get(model, {}).get("notes", "")}  # fmt: skip
    if registry.ensure_registry(args.registry_path):
        print(f"[probe] created {args.registry_path} (pi05lib, pi05base)")
    row = registry.resolve(model, args.registry_path)
    if row.get("server"):
        registry.external_server_launcher(row["server"])
    elif not row.get("adapter"):
        sys.path.insert(0, str(SHORTCUT / "serve"))
        from engine import auto_libero_adapter

        row["adapter"] = auto_libero_adapter(row["path_or_hub"])
    return row


def default_n_action_steps(row: dict[str, Any]) -> int:
    default = registry.parse_notes(row.get("notes")).get("n_action_steps") if row.get("server") else None
    return int(default or (8 if row.get("server") else 10))


def model_action_space(model: str) -> str:
    notes = registry.parse_notes({r["ckpt_id"]: r for r in registry.BUILTIN_ROWS}.get(model, {}).get("notes"))
    return notes.get("action_space", ACTION_SPACE_OSC)


# ---------------------------------------------------------------- servers (sc-bench's ServerProc)


class Served:
    """Start one policy server like sc-bench does, wait for its ready file, stop it on exit."""

    def __init__(self, model: str, model_dir: Path, args: argparse.Namespace):
        from bench.core import runner
        from bench.core.adapters import Context

        row = ckpt_row(model, args)
        n = args.n_action_steps or default_n_action_steps(row)
        sargs = argparse.Namespace(
            server_batch=1,
            clients_per_gpu=1,
            n_action_steps=n,
            n_action_steps_given=args.n_action_steps is not None,
            registry_path=args.registry_path,
        )
        ctx = Context(model_dir, row, args.backend, REPO)
        self.proc = runner.ServerProc(args.gpu, sargs, ctx)
        self.row = row

    def __enter__(self) -> dict[str, Any]:
        try:
            return self.proc.wait_ready()
        except BaseException:
            self.proc.stop()
            raise

    def __exit__(self, *exc: object) -> None:
        self.proc.stop()


# ---------------------------------------------------------------- probing


def hello_identity(hello: dict[str, Any]) -> dict[str, Any]:
    h = {k: hello.get(k) for k in IDENTITY_KEYS}
    h["action_space"] = hello_action_space(hello)
    return h


def check_identity(model_dir: Path, hello: dict[str, Any]) -> None:
    path = model_dir / "hello.json"
    ident = hello_identity(hello)
    if path.exists():
        prev = hello_identity(json.loads(path.read_text()))
        diff = {k: (prev[k], ident[k]) for k in ident if prev.get(k) != ident[k]}
        if diff:
            raise SystemExit(
                f"{model_dir}: server identity changed since the earlier run {diff}; use a new --out"
            )
    model_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hello, indent=1, default=str) + "\n")


def probe_set(
    client: PolicyClient,
    model: str,
    model_dir: Path,
    set_name: str,
    arrays: dict[str, np.ndarray],
    queries: list[dict],
) -> dict[str, Any]:
    space = client.action_space
    path = model_dir / f"probe_{set_name}.jsonl"
    done = {(r["obs_idx"], r["instr_id"]) for r in read_jsonl(path)}
    size = lp.pick_size(arrays, client.info.get("image_hw"))
    t0, n_new = time.time(), 0
    for q in queries:
        if (q["obs_idx"], q["instr_id"]) in done:
            continue
        eid = f"{set_name}/{q['obs_idx']}/{q['instr_id']}"
        obs = lp.wire_obs(arrays, q["obs_idx"], size)
        client.server_ms = 0.0
        client.reset(eid, q["seed"], q["instruction"])
        acts = client.infer(eid, obs, q["instruction"])
        client.end(eid)
        rec = {
            **q,
            "model": model,
            "action_space": space,
            "image_size": size,
            "n_actions": int(acts.shape[0]),
            "action_dim": int(acts.shape[1]),
            "actions": acts.astype(np.float32).tolist(),
            "server_ms": round(client.server_ms, 2),
        }
        append_jsonl(path, rec)
        n_new += 1
        if n_new % 200 == 0:
            rate = n_new / (time.time() - t0)
            print(f"[probe] {now()} {model} set {set_name}: {n_new} new, {rate:.2f} q/s", flush=True)
    rows = read_jsonl(path)
    write_parquet(rows, model_dir / f"probe_{set_name}.parquet")
    ms = [r["server_ms"] for r in rows if r.get("server_ms")]
    return {
        "n_rows": len(rows),
        "n_new": n_new,
        "wall_s": round(time.time() - t0, 1),
        "image_size": size,
        "server_ms_median": float(np.median(ms)) if ms else None,
        "repeat_max_abs_diff": repeat_diff(rows),
    }


def repeat_diff(rows: list[dict]) -> float | None:
    by = {(r["obs_idx"], r["instr_id"]): np.asarray(r["actions"]) for r in rows}
    d = [float(np.max(np.abs(a - by[(o, "orig_s0")]))) for (o, i), a in by.items() if i == "orig_s0_repeat"]
    return max(d) if d else None


def write_parquet(rows: list[dict], path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if "actions" in df:
        df["actions"] = df["actions"].map(lambda a: np.asarray(a, dtype=np.float32).ravel().tolist())
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def run_model(
    model: str, url: str, args: argparse.Namespace, loaded: dict[str, tuple], sets: dict[str, Any]
) -> dict[str, Any]:
    model_dir = Path(args.out) / registry.safe_id(model)
    client = PolicyClient(
        url, action_spaces=(ACTION_SPACE_OSC, ACTION_SPACE_JOINT_VELOCITY), name="langprobe"
    )
    meta: dict[str, Any] = {"model": model, "started_at": now(), "sets": {}}
    try:
        check_identity(model_dir, client.info)
        meta["hello"] = hello_identity(client.info)
        for set_name, (arrays, rows) in loaded.items():
            if client.action_space == ACTION_SPACE_JOINT_VELOCITY and "joint_pos" not in arrays:
                print(f"[probe] {model}: set {set_name} skipped (no joint_pos, server needs it)", flush=True)
                meta["sets"][set_name] = {"skipped": "no joint_pos"}
                continue
            queries = plan_queries(set_name, rows, sets, args.repeat_check, instr_filter(args))
            print(f"[probe] {now()} {model} set {set_name}: {len(queries)} queries", flush=True)
            meta["sets"][set_name] = probe_set(client, model, model_dir, set_name, arrays, queries)
            print(f"[probe] {now()} {model} set {set_name}: {meta['sets'][set_name]}", flush=True)
    finally:
        client.close()
    meta["ended_at"] = now()
    (model_dir / "probe_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return meta


# ---------------------------------------------------------------- dry run


def dry_run(
    args: argparse.Namespace, loaded: dict[str, tuple], sets: dict[str, Any], models: list[str]
) -> float:
    est_ms = dict(EST_MS)
    for kv in (args.est_ms or "").split(","):
        if kv:
            k, v = kv.split("=")
            est_ms[k] = float(v)
    print("model                     set  obs  forwards   ms/fwd   GPU-h  (forwards × ms + server load)")
    total = 0.0
    for m in models:
        joint = model_action_space(m) == ACTION_SPACE_JOINT_VELOCITY
        fw = 0
        for set_name, (arrays, rows) in loaded.items():
            if joint and "joint_pos" not in arrays:
                print(f"{m:25s} {set_name:>3s}    -         -        -       -  (skipped: no joint_pos)")
                continue
            q = plan_queries(set_name, rows, sets, args.repeat_check, instr_filter(args))
            fw += len(q)
            ms = est_ms.get(m, EST_MS_DEFAULT)
            print(f"{m:25s} {set_name:>3s} {len(rows):4d} {len(q):9d} {ms:8.0f} {len(q) * ms / 3.6e6:7.3f}")
        load = EST_LOAD_S.get(m, EST_LOAD_S_DEFAULT)
        h = fw * est_ms.get(m, EST_MS_DEFAULT) / 3.6e6 + load / 3600
        total += h
        print(f"{m:25s} all      {fw:9d}          {h:7.3f}  (incl. load {load:.0f} s)")
    print(f"total ≈ {total:.2f} GPU-h on one GPU (models run one after another); ms/fwd are placeholders "
          "until measured (probe_meta.json server_ms_median)")  # fmt: skip
    return total


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="registry ids (or paths / hub ids), comma-separated")
    ap.add_argument("--obs-dir", default=f"{lp.DEFAULT_ROOT}/analysis/langprobe/obs")
    ap.add_argument("--sets", default="A,B")
    ap.add_argument("--out", default=None, help="default <out-root>/analysis/langprobe/<run-id>")
    ap.add_argument("--run-id", default="probe")
    ap.add_argument("--out-root", default=None, help=f"default $SC_OUT_ROOT or {registry.DEFAULT_OUT_ROOT}")
    ap.add_argument("--registry", default=None, help="default <out-root>/ckpt_registry.tsv")
    ap.add_argument("--gpu", default=None, help="physical GPU id (default: $CUDA_VISIBLE_DEVICES or 0)")
    ap.add_argument("--url", default=None, help="use this running server (one model) instead of starting one")
    ap.add_argument(
        "--n-action-steps", type=int, default=None, help="default: sc-bench's (10 / registry notes)"
    )
    ap.add_argument("--suites", default=None)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--max-obs-per-task", type=int, default=None, help="smoke runs")
    ap.add_argument(
        "--instr-ids",
        default=None,
        help="smoke runs: only these instr_ids (+ orig_s0), e.g. orig_s1,blank,C4-a",
    )
    ap.add_argument("--repeat-check", type=int, default=4)
    ap.add_argument("--no-verify", action="store_true", help="skip the obs-file sha256 check")
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--est-ms", default=None, help="model=ms,... for --dry-run (default: placeholders)")
    ap.add_argument("--backend", choices=("libero", "mock"), default="libero", help="mock = CPU fake servers")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = registry.out_root(args.out_root)
    args.registry_path = Path(args.registry) if args.registry else registry.registry_path(root)
    args.out = args.out or str(root / "analysis" / "langprobe" / args.run_id)
    args.gpu = args.gpu or (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    models = [m for m in args.models.split(",") if m]
    if args.url and len(models) != 1:
        raise SystemExit("--url serves one model: pass exactly one --models id (used for the output dir)")
    sets = lp.load_instruction_sets()
    loaded = {}
    for s in args.sets.split(","):
        arrays, rows = lp.load_set(args.obs_dir, s, verify=not args.no_verify)
        loaded[s] = (arrays, select_rows(rows, args))
    if args.dry_run:
        dry_run(args, loaded, sets, models)
        return 0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "cmd.txt").open("a").write(f"{now()} {' '.join(sys.argv)}\n")
    if args.url:
        run_model(models[0], args.url, args, loaded, sets)
        return 0
    lock = None
    if args.backend != "mock":
        from bench.core import runner

        lock = runner.take_gpu_lock(args.gpu, runner.gpu_uuids(), not args.no_gpu_lock)
    try:
        for m in models:  # one server at a time
            model_dir = Path(args.out) / registry.safe_id(m)
            print(f"[probe] {now()} starting server for {m} on GPU {args.gpu}", flush=True)
            t0 = time.time()
            with Served(m, model_dir, args) as ready:
                print(f"[probe] {now()} {m} ready in {time.time() - t0:.0f} s at {ready['url']}", flush=True)
                meta = run_model(m, ready["url"], args, loaded, sets)
            meta["server_start_s"] = round(time.time() - t0, 1)
            (model_dir / "probe_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    finally:
        if lock is not None:
            lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
