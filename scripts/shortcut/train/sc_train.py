"""sc-train: LIBERO training harness around `lerobot-train` (TESTBED_DESIGN.md, T011).

  sc-train --preset sc_template_libero_all --run-id tb-tmpl-a [--gpus 0] [--steps N] [--batch B]
           [--grad-accum K] [--data-variant instr=blank] [--out-root DIR] [--resume] [--dry-run]
           [-- <extra lerobot-train args>]
  sc-train register --run-id tb-tmpl-a [--ckpt-id X]          # (re-)register a finished run
  sc-train register --ckpt-id X --path P --policy-type T ...  # register an external checkpoint
  sc-train presets                                             # list presets with batch/steps/time

Run dir `<out-root>/runs/<run_id>/`: run_meta.json, git_sha, cmd.sh, train.log, loss.tsv, wandb_url.txt,
train/ (lerobot output_dir: checkpoints/<step>/pretrained_model + train_config.json), checkpoints -> train/checkpoints.
At the end: one line in `<out-root>/results.tsv` and one in `<out-root>/ckpt_registry.tsv`.

GPUs: CUDA_VISIBLE_DEVICES=--gpus; >1 GPU -> `accelerate launch --num_processes N` (plain DDP).
If /queue/locks exists (this container) each GPU's `<uuid>.lock` is flock'ed for the whole run and only
GPU 0 is allowed unless --allow-any-gpu. No /harness queue is used.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import socket
import subprocess  # nosec B404
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from data_variants import DataVariant, parse_variant  # noqa: E402
from registry import REGISTRY_NAME, add_entries, ensure_seed_rows, now_iso  # noqa: E402

PRESET_DIR = HERE / "presets"
ENTRY = HERE / "lerobot_train_entry.py"
DATASET_REPO = "HuggingFaceVLA/libero"
DATASET_REV = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"  # the revision T008 downloaded
DEFAULT_OUT_ROOT = os.environ.get("SC_OUT_ROOT", "/outputs/VLA-shortcut-exps")
LOCK_DIR = Path(os.environ.get("SC_GPU_LOCK_DIR", "/queue/locks"))
WANDB_PROJECT = os.environ.get("SC_WANDB_PROJECT", "VLA-shortcut-exps")
RESULTS_HEADER = "run_id\tstatus\tgit_sha\tenv\tgpu_h\tended_at\tmetrics\n"


# ---------------------------------------------------------------------------- presets
def load_preset(name: str) -> dict[str, Any]:
    path = PRESET_DIR / f"{name}.yaml"
    if not path.exists():
        names = sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))
        raise SystemExit(f"unknown preset {name!r}; available: {', '.join(names)}")
    preset = yaml.safe_load(path.read_text())
    preset["name"] = name
    return preset


def scale_for(preset: dict[str, Any], n_gpus: int) -> dict[str, Any]:
    """Per-GPU batch / loop steps / grad accum for the largest preset GPU count <= n_gpus."""
    table = {int(k): v for k, v in preset["scale"].items()}
    keys = [k for k in sorted(table) if k <= n_gpus] or [min(table)]
    return {"grad_accum": 1, **table[keys[-1]], "_from_gpus": keys[-1]}


def cli_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    return str(v)


# ---------------------------------------------------------------------------- environment
def git_sha() -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()  # nosec
    # untracked files count only under the code that runs (new policies, presets)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "scripts/shortcut"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.strip()  # nosec
    return sha + ("-dirty" if dirty else "")


def gpu_inventory() -> dict[int, dict[str, Any]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout  # nosec
    except (OSError, subprocess.CalledProcessError):
        return {}
    inv = {}
    for line in out.strip().splitlines():
        idx, uuid, name, mem = (x.strip() for x in line.split(","))
        inv[int(idx)] = {"uuid": uuid, "name": name, "mem_gb": round(int(mem) / 1024)}
    return inv


def local_dataset_root(out_root: Path) -> Path | None:
    cands = [
        os.environ.get("SC_LIBERO_DATA_ROOT"),
        out_root / "data" / "hf_libero",
        DEFAULT_OUT_ROOT + "/data/hf_libero",
    ]
    for c in cands:
        if c and (Path(c) / "meta" / "info.json").exists():
            return Path(c)
    return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wandb_login_ok() -> bool:
    """True if the W&B key works and has a default entity (or SC_WANDB_ENTITY/WANDB_ENTITY is set)."""
    netrc = Path.home() / ".netrc"
    if not os.environ.get("WANDB_API_KEY") and not (netrc.exists() and "api.wandb.ai" in netrc.read_text()):
        return False
    probe = "import wandb; v = wandb.Api(timeout=15).viewer; print(getattr(v, 'entity', '') or '')"
    try:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)  # nosec
    except subprocess.TimeoutExpired:
        return False
    entity = os.environ.get("SC_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY") or r.stdout.strip()
    return r.returncode == 0 and bool(entity)


def wandb_mode(choice: str) -> str:
    """auto -> online if the login works, else offline (sync later with `wandb sync`)."""
    if choice != "auto":
        return choice
    return "online" if wandb_login_ok() else "offline"


# ---------------------------------------------------------------------------- plan
def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    run_dir = out_root / "runs" / args.run_id
    train_dir = run_dir / "train"
    gpus = [int(g) for g in str(args.gpus).split(",") if g.strip() != ""]
    n = len(gpus)

    if args.resume:
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            raise SystemExit(f"--resume: {meta_path} not found")
        prev = json.loads(meta_path.read_text())
        preset = load_preset(prev["preset"])
        variant = DataVariant.from_json(json.dumps(prev["data_variant"]))
        if args.data_variant and parse_variant(args.data_variant) != variant:
            raise SystemExit(f"--resume: data variant differs from the run's ({variant.id})")
        last = train_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if not last.exists():
            raise SystemExit(f"--resume: no checkpoint at {last}")
        lerobot_args = [f"--config_path={last}", "--resume=true", f"--output_dir={train_dir}"]
        if args.steps:
            lerobot_args.append(f"--steps={args.steps}")
        sc = {**prev["scale"], **({"steps": args.steps} if args.steps else {})}
    else:
        preset = load_preset(args.preset)
        variant = parse_variant(args.data_variant)
        sc = scale_for(preset, n)
        if args.batch:
            sc["batch"] = args.batch
        if args.steps:
            sc["steps"] = args.steps
        if args.grad_accum:
            sc["grad_accum"] = args.grad_accum
        lerobot: dict[str, Any] = dict(preset["lerobot"])
        lerobot["dataset.repo_id"] = DATASET_REPO
        root = local_dataset_root(out_root)
        if root is not None:
            lerobot["dataset.root"] = str(root)
        else:
            lerobot["dataset.revision"] = DATASET_REV
        lerobot["batch_size"] = sc["batch"]
        lerobot["steps"] = sc["steps"]
        if sc["grad_accum"] > 1:
            lerobot["accelerator.gradient_accumulation.steps"] = sc["grad_accum"]
        sched = preset.get("schedule")
        if sched:
            lerobot["policy.scheduler_decay_steps"] = sc["steps"]
            lerobot["policy.scheduler_warmup_steps"] = max(1, round(sc["steps"] * sched["warmup_frac"]))
        lerobot["output_dir"] = str(train_dir)
        lerobot["job_name"] = args.run_id
        lerobot["policy.push_to_hub"] = False
        mode = wandb_mode(args.wandb) if not args.dry_run else args.wandb
        lerobot["wandb.enable"] = mode != "off"
        if mode != "off":
            lerobot["wandb.project"] = WANDB_PROJECT
            lerobot["wandb.disable_artifact"] = True
            if mode in ("online", "offline"):
                lerobot["wandb.mode"] = mode
            entity = os.environ.get("SC_WANDB_ENTITY")
            if entity:
                lerobot["wandb.entity"] = entity
        lerobot_args = [f"--{k}={cli_value(v)}" for k, v in lerobot.items()]

    lerobot_args += args.extra
    launcher = args.launcher if args.launcher != "auto" else ("accelerate" if n > 1 else "python")
    if launcher == "accelerate":
        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            f"--num_processes={n}",
            f"--main_process_port={free_port()}",
            str(ENTRY),
        ]
    else:
        cmd = [sys.executable, str(ENTRY)]
    cmd += lerobot_args

    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
        "SC_DATA_VARIANT": variant.to_json(),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }
    eff_batch = sc["batch"] * n * sc.get("grad_accum", 1)
    return {
        "run_id": args.run_id,
        "preset": preset["name"],
        "policy_type": preset["policy_type"],
        "train_data": preset.get("train_data", DATASET_REPO),
        "out_root": str(out_root),
        "run_dir": str(run_dir),
        "train_dir": str(train_dir),
        "gpus": gpus,
        "launcher": launcher,
        "scale": {k: v for k, v in sc.items() if not k.startswith("_")},
        "scale_from_gpus": sc.get("_from_gpus"),
        "effective_batch": eff_batch,
        "optimizer_updates": sc["steps"] // max(1, sc.get("grad_accum", 1)),
        "data_variant": json.loads(variant.to_json()),
        "data_variant_id": variant.id,
        "dataset": {
            "repo_id": DATASET_REPO,
            "revision": DATASET_REV,
            "root": str(local_dataset_root(out_root) or ""),
        },
        "estimate": preset.get("estimate", {}),
        "resume": bool(args.resume),
        "env": env,
        "cmd": cmd,
    }


def print_plan(plan: dict[str, Any], inv: dict[int, dict[str, Any]]) -> None:
    sc = plan["scale"]
    est = plan["estimate"] or {}
    print(f"[sc-train] run {plan['run_id']}  preset {plan['preset']}  policy {plan['policy_type']}")
    print(
        f"[sc-train] data {plan['dataset']['repo_id']} @ {plan['dataset']['revision'][:8]}  "
        f"root={plan['dataset']['root'] or '(Hub)'}  variant={plan['data_variant_id']}"
    )
    print(
        f"[sc-train] GPUs {plan['gpus']} ({plan['launcher']}; preset row for {plan['scale_from_gpus']} GPU)  "
        f"batch/GPU {sc['batch']} x {len(plan['gpus'])} GPU x accum {sc.get('grad_accum', 1)} = "
        f"effective {plan['effective_batch']}; loop steps {sc['steps']} = {plan['optimizer_updates']} optimizer updates"
    )
    samples = plan["effective_batch"] * plan["optimizer_updates"]
    print(f"[sc-train] samples seen {samples:,} (~{samples / 273465:.1f} epochs of 273,465 frames)")
    if est:
        print(
            f"[sc-train] estimate: memory/GPU ~{est.get('mem_gb_per_gpu')} GB, "
            f"4xA100 ~{est.get('hours_a100x4')} h, GPU0 ~{est.get('hours_gpu0')} h"
            + (f", params ~{est['params_b']} B" if est.get("params_b") else "")
        )
        need = est.get("mem_gb_per_gpu")
        for g in plan["gpus"]:
            have = inv.get(g, {}).get("mem_gb")
            if isinstance(need, (int, float)) and have and have < need:
                print(f"[sc-train] WARNING: GPU {g} ({inv[g]['name']}, {have} GB) < estimated {need} GB/GPU")
    print(
        "[sc-train] env "
        + " ".join(f"{k}={shlex.quote(v)}" for k, v in plan["env"].items() if k != "SC_DATA_VARIANT")
    )
    print("[sc-train] cmd " + shlex.join(plan["cmd"]), flush=True)


# ---------------------------------------------------------------------------- run
def acquire_gpu_locks(stack: ExitStack, gpus: list[int], inv: dict[int, dict[str, Any]]) -> None:
    if not LOCK_DIR.is_dir():
        return
    for g in gpus:
        uuid = inv.get(g, {}).get("uuid")
        if not uuid:
            raise SystemExit(f"GPU {g}: uuid unknown, cannot take {LOCK_DIR}/<uuid>.lock")
        f = stack.enter_context(open(LOCK_DIR / f"{uuid}.lock", "a"))  # noqa: SIM115 (held until the run ends)
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[sc-train] GPU {g} lock busy ({LOCK_DIR}/{uuid}.lock), waiting ...", flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        print(f"[sc-train] GPU {g} lock held", flush=True)


LOSS_RE = re.compile(r"step:(\S+) .*?\bloss:([0-9.eE+-]+)")
WANDB_RE = re.compile(r"(https://wandb\.ai/\S+/runs/\S+)")


def stream(
    cmd: list[str], env: dict[str, str], log_path: Path, run_dir: Path
) -> tuple[int, list[tuple[str, float]]]:
    losses: list[tuple[str, float]] = []
    wandb_written = False
    full_env = {**os.environ, **env}
    with open(log_path, "a") as log, open(run_dir / "loss.tsv", "a") as lt:
        if lt.tell() == 0:
            lt.write("log_idx\tstep\tloss\n")
        proc = subprocess.Popen(
            cmd,
            env=full_env,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )  # nosec
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            m = LOSS_RE.search(line)
            if m:
                losses.append((m.group(1), float(m.group(2))))
                lt.write(f"{len(losses)}\t{m.group(1)}\t{m.group(2)}\n")
                lt.flush()
            if not wandb_written:
                w = WANDB_RE.search(line)
                if w:
                    (run_dir / "wandb_url.txt").write_text(w.group(1).rstrip(".,") + "\n")
                    wandb_written = True
        return proc.wait(), losses


def checkpoint_info(train_dir: Path) -> tuple[Path | None, int | None]:
    last = train_dir / "checkpoints" / "last"
    if not last.exists():
        return None, None
    real = last.resolve()
    step = None
    ts = real / "training_state" / "training_step.json"
    if ts.exists():
        step = json.loads(ts.read_text()).get("step")
    if step is None and real.name.isdigit():
        step = int(real.name)
    return real / "pretrained_model", step


def append_result(out_root: Path, row: list[str]) -> None:
    path = out_root / "results.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            f.write(RESULTS_HEADER)
        f.write("\t".join(row) + "\n")


def register_run(run_dir: Path, ckpt_id: str | None, replace: bool, notes: str = "") -> str:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    ckpt, step = checkpoint_info(Path(meta["train_dir"]))
    if ckpt is None:
        raise SystemExit(f"{run_dir}: no checkpoint to register")
    reg = Path(meta["out_root"]) / REGISTRY_NAME
    ensure_seed_rows(reg)
    cid = ckpt_id or meta["run_id"]
    add_entries(
        reg,
        [
            {
                "ckpt_id": cid,
                "path_or_hub": str(ckpt),
                "policy_type": meta["policy_type"],
                "train_data": f"{meta['dataset']['repo_id']}@{meta['dataset']['revision'][:8]}",
                "train_variant": meta["data_variant_id"],
                "steps": step,
                "run_id": meta["run_id"],
                "git_sha": meta["git_sha"],
                "notes": notes or meta.get("notes", "") or f"preset {meta['preset']}",
            }
        ],
        replace=replace,
    )
    print(f"[sc-train] registered {cid} -> {ckpt} in {reg}")
    return cid


def cmd_run(args: argparse.Namespace) -> int:
    if not args.run_id or (not args.preset and not args.resume):
        raise SystemExit("--run-id and --preset (or --resume) are required")
    inv = gpu_inventory()
    plan = build_plan(args)
    if LOCK_DIR.is_dir() and plan["gpus"] != [0] and not args.allow_any_gpu:
        raise SystemExit(
            f"this container uses GPU 0 only (found {LOCK_DIR}); pass --allow-any-gpu to override"
        )
    print_plan(plan, inv)
    if args.dry_run:
        # parse + validate the same lerobot-train args in one CPU process (no GPU, no weights, no data)
        launch_args = plan["cmd"][plan["cmd"].index(str(ENTRY)) + 1 :]
        if args.resume:
            print("[sc-train] dry-run: resume config comes from the checkpoint; not re-validated")
            return 0
        env = {**os.environ, **plan["env"], "SC_CHECK_CONFIG": "1", "CUDA_VISIBLE_DEVICES": ""}
        launch_args = [a for a in launch_args if not a.startswith("--output_dir=")]
        return subprocess.run([sys.executable, str(ENTRY), *launch_args], env=env, cwd=REPO).returncode  # nosec

    run_dir, out_root = Path(plan["run_dir"]), Path(plan["out_root"])
    if not args.resume and Path(plan["train_dir"]).exists():
        raise SystemExit(f"{plan['train_dir']} exists; use --resume or another --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    meta = {
        **{k: v for k, v in plan.items() if k not in ("cmd", "env")},
        "git_sha": sha,
        "host": socket.gethostname(),
        "gpu_models": [inv.get(g, {}).get("name") for g in plan["gpus"]],
        "started_at": now_iso(),
        "notes": args.notes,
        "command": shlex.join(plan["cmd"]),
        "env": plan["env"],
    }
    if args.resume:
        prev = json.loads((run_dir / "run_meta.json").read_text())
        meta["resumed_from"] = {k: prev.get(k) for k in ("started_at", "git_sha", "scale")}
        meta["preset"] = prev["preset"]
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "git_sha").write_text(sha + "\n")
    (run_dir / "cmd.sh").write_text(
        "#!/usr/bin/env bash\ncd "
        + shlex.quote(str(REPO))
        + "\n"
        + " ".join(f"{k}={shlex.quote(v)}" for k, v in plan["env"].items())
        + " \\\n  "
        + shlex.join(plan["cmd"])
        + "\n"
    )
    link = run_dir / "checkpoints"
    if not link.exists() and not link.is_symlink():
        link.symlink_to("train/checkpoints")
    wandb_note = {
        "--wandb.enable=false": "disabled (--wandb off)",
        "--wandb.mode=offline": "offline (W&B login unavailable); upload later: wandb sync <run_dir>/train/wandb/offline-run-*",
    }
    for flag, note in wandb_note.items():
        if flag in plan["cmd"] and not (run_dir / "wandb_url.txt").exists():
            (run_dir / "wandb_url.txt").write_text(note + "\n")

    t0 = time.time()
    with ExitStack() as stack:
        acquire_gpu_locks(stack, plan["gpus"], inv)
        rc, losses = stream(plan["cmd"], plan["env"], run_dir / "train.log", run_dir)
    hours = (time.time() - t0) / 3600
    offline = sorted((Path(plan["train_dir"]) / "wandb").glob("offline-run-*"))
    if offline:
        (run_dir / "wandb_url.txt").write_text(f"offline; upload with: wandb sync {offline[-1]}\n")
    gpu_h = hours * len(plan["gpus"])
    ckpt, step = checkpoint_info(Path(plan["train_dir"]))
    status = "done" if rc == 0 and ckpt is not None else "failed"
    metrics = [f"preset={plan['preset']}", f"steps={step}", f"variant={plan['data_variant_id']}"]
    if losses:
        metrics += [f"loss_first={losses[0][1]:.4f}", f"loss_last={losses[-1][1]:.4f}"]
    env_tag = (
        f"{'accel' if plan['launcher'] == 'accelerate' else 'direct'}-gpu{','.join(map(str, plan['gpus']))}"
    )

    cid = ""
    if status == "done" and not args.no_register:
        cid = register_run(run_dir, args.ckpt_id, replace=args.resume, notes=args.notes)
        metrics.append(f"ckpt_id={cid}")
    append_result(
        out_root, [plan["run_id"], status, sha[:12], env_tag, f"{gpu_h:.2f}", now_iso(), ";".join(metrics)]
    )
    meta.update(
        {
            "ended_at": now_iso(),
            "status": status,
            "returncode": rc,
            "wall_h": round(hours, 3),
            "gpu_h": round(gpu_h, 3),
            "final_step": step,
            "checkpoint": str(ckpt or ""),
            "ckpt_id": cid,
            "loss_first": losses[0][1] if losses else None,
            "loss_last": losses[-1][1] if losses else None,
        }
    )
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[sc-train] {status}: {plan['run_id']} step {step}, {hours:.2f} h, ckpt {ckpt}")
    return 0 if status == "done" else 1


def cmd_register(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    if args.run_id:
        register_run(out_root / "runs" / args.run_id, args.ckpt_id, args.replace, args.notes)
        return 0
    if not (args.ckpt_id and args.path and args.policy_type):
        raise SystemExit("register needs --run-id, or --ckpt-id + --path + --policy-type")
    reg = out_root / REGISTRY_NAME
    ensure_seed_rows(reg)
    add_entries(
        reg,
        [
            {
                "ckpt_id": args.ckpt_id,
                "path_or_hub": args.path,
                "policy_type": args.policy_type,
                "train_data": args.train_data,
                "train_variant": args.train_variant,
                "steps": args.steps or "",
                "run_id": "",
                "git_sha": "",
                "notes": args.notes,
            }
        ],
        replace=args.replace,
    )
    print(f"[sc-train] registered {args.ckpt_id} in {reg}")
    return 0


def cmd_presets(_: argparse.Namespace) -> int:
    for p in sorted(PRESET_DIR.glob("*.yaml")):
        pr = load_preset(p.stem)
        est = pr.get("estimate", {})
        rows = ", ".join(
            f"{g} GPU: bs {v['batch']}x{v.get('grad_accum', 1)} {v['steps']} steps"
            for g, v in sorted(pr["scale"].items())
        )
        print(
            f"{p.stem:24s} {pr['policy_type']:12s} {rows} | ~{est.get('hours_a100x4')} h 4xA100, "
            f"~{est.get('hours_gpu0')} h GPU0 | {pr.get('description', '')}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    if not argv or argv[0] not in ("run", "register", "presets"):
        argv = ["run", *argv]

    ap = argparse.ArgumentParser(
        prog="sc-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train (default subcommand)")
    r.add_argument("--preset")
    r.add_argument("--run-id")
    r.add_argument("--gpus", default=os.environ.get("SC_GPUS", "0"))
    r.add_argument("--steps", type=int)
    r.add_argument("--batch", type=int, help="per-GPU micro-batch")
    r.add_argument("--grad-accum", type=int)
    r.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    r.add_argument("--data-variant", action="append", help="e.g. suites=goal  instr=blank  instr=paraphrase")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument(
        "--wandb",
        choices=("auto", "online", "offline", "off"),
        default="auto",
        help="auto: online if the W&B login works, else offline",
    )
    r.add_argument("--launcher", choices=("auto", "python", "accelerate"), default="auto")
    r.add_argument("--ckpt-id", help="registry id (default: run id)")
    r.add_argument("--no-register", action="store_true")
    r.add_argument("--allow-any-gpu", action="store_true")
    r.add_argument("--notes", default="")
    g = sub.add_parser("register", help="add a checkpoint to ckpt_registry.tsv")
    g.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    g.add_argument("--run-id")
    g.add_argument("--ckpt-id")
    g.add_argument("--path")
    g.add_argument("--policy-type")
    g.add_argument("--train-data", default=DATASET_REPO)
    g.add_argument("--train-variant", default="none")
    g.add_argument("--steps", type=int)
    g.add_argument("--notes", default="")
    g.add_argument("--replace", action="store_true")
    sub.add_parser("presets", help="list presets")
    args = ap.parse_args(argv)
    args.extra = extra
    return {"run": cmd_run, "register": cmd_register, "presets": cmd_presets}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
