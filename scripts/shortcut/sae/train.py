"""Train one TopK+AuxK SAE per layer on collected activations (Swann et al. 2026 App. B.1, `model.py`).

    uv run python scripts/shortcut/sae/train.py --model pi05lib --layers all --gpus 0,1,2,3
    uv run python scripts/shortcut/sae/train.py --model pi05lib --layers pg:5 --seeds 0,1,2,3,4,5

Defaults = paper Table 6: ER 1, k 100 (PaliGemma, emb) / 64 (action expert), k_aux 512, aux 1/32, dead after
500 steps, Adam 1e-4 (0.9, 0.999), batch 4096, 100 epochs, grad clip 1.0, b_pre = geometric median of 10k
samples. A job = (layer, seed); its data is held on the GPU for the whole run.

Inputs: PaliGemma/emb `--input pooled_all|img|text|cam1|cam2` (count-weighted means of the stored token
groups; pooled_all = the paper's mean over all valid tokens). Action expert: every recorded denoising step
of every timestep is a sample (`--ae-steps` to restrict); the step index stays in the records.

Splitting: `--shard i/N` takes jobs i, i+N, ... of the sorted job list (one pod = one shard); `--gpus` runs that
shard's jobs on local GPUs, one process per GPU, biggest first. Resume: a job dir with done.json is skipped,
one with ckpt.pt continues from it (same config), anything else starts fresh.

Output: <out-root>/sae/<model>/sae/<input-tag>/<layer>/seed<s>/
  sae.safetensors, config.json      the SAE (model.TopKSAE.load(dir)); config has layer, input, acts_dir, k, ...
  metrics.json                      nmse (normalized space), var_explained (raw), dead fraction, timing
  fire_freq.npy                     per-feature fraction of samples with the latent active
  log.jsonl, wandb/ (offline)       training curves
  ckpt.pt                           resume state (kept; small)
  features.parquet, fire_by_group.parquet, codes_*.npy, codes_meta.parquet   stats.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess  # nosec B404
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # --gpus = nvidia-smi indices
for p in (str(HERE), str(HERE.parent), str(HERE.parent / "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import store  # noqa: E402
from hooks import layer_family, parse_layers  # noqa: E402

DEFAULT_OUT_ROOT = os.environ.get("SC_OUT_ROOT", "/outputs/VLA-shortcut-exps")
WANDB_PROJECT = os.environ.get("SC_WANDB_PROJECT", "VLA-shortcut-exps") + "-sae"


def parse_shard(s: str) -> tuple[int, int]:
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"--shard {s}: need 0 <= i < N")
    return i, n


def input_tag(layer: str, args: argparse.Namespace) -> str:
    if layer_family(layer) == "pg":
        return args.input
    return "ae-steps-" + ("all" if args.ae_steps == "all" else args.ae_steps.replace(",", "-"))


def ae_steps_list(args: argparse.Namespace) -> list[int] | None:
    return None if args.ae_steps == "all" else [int(x) for x in args.ae_steps.split(",")]


def job_dir(args: argparse.Namespace, layer: str, seed: int) -> Path:
    return Path(args.sae_dir) / input_tag(layer, args) / layer / f"seed{seed}"


def job_config(args: argparse.Namespace, layer: str, seed: int, n_samples: int, d: int) -> dict[str, Any]:
    k = args.k_ae if layer_family(layer) == "ae" else args.k_pg
    return {
        "layer": layer,
        "input": args.input if layer_family(layer) == "pg" else "action_tokens_mean",
        "ae_steps": ae_steps_list(args) if layer_family(layer) == "ae" else None,
        "seed": seed,
        "acts_dir": str(args.acts_dir),
        "acts_identity": store.identity_hash(store.read_plan(Path(args.acts_dir))["identity"]),
        "n_samples": n_samples,
        "d": d,
        "n_latents": int(round(d * args.expansion)),
        "k": k,
        "k_aux": args.k_aux,
        "aux_coef": args.aux_coef,
        "dead_steps": args.dead_steps,
        "lr": args.lr,
        "batch": args.batch,
        "epochs": args.epochs,
        "steps": args.steps,
        "grad_clip": args.grad_clip,
        "gm_samples": args.gm_samples,
    }


# ---------------------------------------------------------------------------- one job
class Logger:
    def __init__(self, jdir: Path, cfg: dict[str, Any], mode: str, name: str):
        self.f = open(jdir / "log.jsonl", "a")  # noqa: SIM115
        self.run = None
        if mode != "off":
            try:
                import wandb

                os.environ.setdefault("WANDB_SILENT", "true")
                self.run = wandb.init(
                    project=WANDB_PROJECT,
                    name=name,
                    id=name.replace("/", "-"),
                    resume="allow",
                    mode="offline",
                    dir=str(jdir),
                    config=cfg,
                    reinit="create_new",
                )
            except Exception as e:  # W&B must never stop training (keys are refused in some containers)
                print(f"[train] wandb disabled: {e!r}", flush=True)

    def log(self, rec: dict[str, Any]) -> None:
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        if self.run is not None:
            self.run.log(rec, step=rec["step"])

    def close(self) -> None:
        self.f.close()
        if self.run is not None:
            self.run.finish()


def save_ckpt(path: Path, state: dict[str, Any]) -> None:
    import torch

    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train_job(args: argparse.Namespace, layer: str, seed: int, x: Any, meta: Any) -> dict[str, Any]:
    import torch
    from model import SAEConfig, TopKSAE
    from stats import run_stats

    jdir = job_dir(args, layer, seed)
    jdir.mkdir(parents=True, exist_ok=True)
    n_samples, d = x.shape
    cfg = job_config(args, layer, seed, n_samples, d)
    if (jdir / "done.json").exists():
        done = json.loads((jdir / "done.json").read_text())
        if done.get("config") == cfg and not args.force:
            print(f"[train] {layer} seed{seed}: done, skip", flush=True)
            return done["metrics"]
    device = x.device
    bsz = min(args.batch, n_samples)
    spe = math.ceil(n_samples / bsz)
    total = args.steps or args.epochs * spe

    sae = TopKSAE(
        SAEConfig(
            d=d,
            n_latents=cfg["n_latents"],
            k=cfg["k"],
            k_aux=args.k_aux,
            aux_coef=args.aux_coef,
            dead_steps=args.dead_steps,
        ),
        seed=seed,
    ).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    last_fired = torch.zeros(cfg["n_latents"], dtype=torch.long, device=device)
    step = 0
    ckpt = jdir / "ckpt.pt"
    if ckpt.exists() and not args.force:
        st = torch.load(ckpt, map_location=device, weights_only=True)
        if st["config"] != cfg:
            raise SystemExit(f"{ckpt} was made with another config; --force to start over")
        sae.load_state_dict(st["sae"])
        sae.cfg.c_mse = st["c_mse"]
        opt.load_state_dict(st["opt"])
        last_fired, step = st["last_fired"].to(device), st["step"]
        print(f"[train] {layer} seed{seed}: resume at step {step}/{total}", flush=True)
    else:
        g = torch.Generator().manual_seed(seed)
        pick = torch.randperm(n_samples, generator=g)[: args.gm_samples].to(device)
        sae.init_from_data(x[pick].float())
    name = f"{Path(args.sae_dir).parent.name}/{input_tag(layer, args)}/{layer}/seed{seed}"
    logger = Logger(jdir, cfg, args.wandb, name)
    print(
        f"[train] {layer} seed{seed}: {n_samples} samples × {d}, n={cfg['n_latents']} k={cfg['k']}, "
        f"batch {bsz}, {spe} steps/epoch, {total} steps, C_MSE {sae.cfg.c_mse:.4f}",
        flush=True,
    )
    t0 = time.time()
    run_mse = run_aux = 0.0
    perm = None
    while step < total:
        epoch, pos = divmod(step, spe)
        if perm is None or pos == 0:
            gen = torch.Generator(device=device).manual_seed(seed * 100_003 + epoch)
            perm = torch.randperm(n_samples, generator=gen, device=device)
        rows = perm[pos * bsz : (pos + 1) * bsz]
        xb = x[rows].float()
        dead = (step - last_fired) >= args.dead_steps
        loss, parts, fired = sae.loss(xb, dead)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.project_decoder_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        opt.step()
        sae.renorm_decoder()
        step += 1
        last_fired[fired] = step
        run_mse += float(parts["mse"])
        run_aux += float(parts["aux"])
        if step % args.log_every == 0 or step == total:
            k = args.log_every if step % args.log_every == 0 else step % args.log_every
            rec = {
                "step": step,
                "epoch": step / spe,
                "nmse": run_mse / k,
                "aux": run_aux / k,
                "dead_frac": float(((step - last_fired) >= args.dead_steps).float().mean()),
                "sec": round(time.time() - t0, 1),
            }
            logger.log(rec)
            run_mse = run_aux = 0.0
            if step % (args.log_every * 20) == 0 or step == total:
                print(
                    f"[train] {layer} seed{seed} step {step}/{total} nmse {rec['nmse']:.4f} "
                    f"dead {rec['dead_frac']:.3f}",
                    flush=True,
                )
        if step % args.ckpt_every == 0 or step == total:
            save_ckpt(
                ckpt,
                {
                    "config": cfg,
                    "sae": sae.state_dict(),
                    "c_mse": sae.cfg.c_mse,
                    "opt": opt.state_dict(),
                    "last_fired": last_fired,
                    "step": step,
                },
            )
        if args.stop_after and step >= args.stop_after:  # testing resume
            logger.close()
            print(f"[train] {layer} seed{seed}: stopped at step {step} (--stop-after)", flush=True)
            return {}
    train_sec = time.time() - t0
    logger.close()

    # final metrics on all samples
    sae.eval()
    with torch.no_grad():
        sq = sq_raw = 0.0
        fire = torch.zeros(cfg["n_latents"], device=device)
        mean = sum(x[i : i + 65536].float().sum(0) for i in range(0, n_samples, 65536)) / n_samples
        var_raw = 0.0
        for i in range(0, n_samples, 8192):
            xb = x[i : i + 8192].float()
            xt, mu, nrm = sae.normalize(xb)
            vals, idx = sae.topk(sae.pre_acts(xt))
            xt_hat = sae.decode_sparse(vals, idx)
            x_hat = xt_hat * nrm + mu + sae.b_pre
            sq += float((xt - xt_hat).pow(2).sum())
            sq_raw += float((xb - x_hat).pow(2).sum())
            var_raw += float((xb - mean).pow(2).sum())
            fire.scatter_add_(0, idx[vals > 0], torch.ones_like(vals[vals > 0]))
    fire_freq = (fire / n_samples).cpu().numpy()
    metrics = {
        "nmse": sq / n_samples / sae.cfg.c_mse,
        "var_explained": 1 - sq_raw / max(var_raw, 1e-12),
        "dead_frac_never_fired": float((fire_freq == 0).mean()),
        "dead_frac_last_window": float(((step - last_fired) >= args.dead_steps).float().mean()),
        "c_mse": sae.cfg.c_mse,
        "steps": step,
        "train_seconds": round(train_sec, 1),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    np.save(jdir / "fire_freq.npy", fire_freq)
    sae.save(jdir, extra={k: v for k, v in cfg.items() if k not in ("d", "n_latents", "k", "k_aux")})
    if not args.no_stats:
        t1 = time.time()
        metrics["stats"] = run_stats(jdir, sae, x, meta)
        metrics["stats_seconds"] = round(time.time() - t1, 1)
    (jdir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    store.write_json_atomic(jdir / "done.json", {"config": cfg, "metrics": metrics})
    print(
        f"[train] {layer} seed{seed}: nmse {metrics['nmse']:.4f} var_expl {metrics['var_explained']:.4f} "
        f"dead {metrics['dead_frac_never_fired']:.3f} in {train_sec:.0f}s",
        flush=True,
    )
    return metrics


def run_jobs(args: argparse.Namespace, jobs: list[tuple[str, int]]) -> None:
    import torch

    device = torch.device("cuda" if args.gpus != "cpu" and torch.cuda.is_available() else "cpu")
    by_layer: dict[str, list[int]] = {}
    for layer, seed in jobs:
        by_layer.setdefault(layer, []).append(seed)
    for layer, seeds in by_layer.items():
        if all((job_dir(args, layer, s) / "done.json").exists() for s in seeds) and not args.force:
            print(f"[train] {layer}: all seeds done, skip", flush=True)
            continue
        t0 = time.time()
        x_np, meta = store.load_layer(Path(args.acts_dir), layer, args.input, ae_steps_list(args))
        x = torch.from_numpy(x_np).to(device)
        del x_np
        print(f"[train] {layer}: loaded {tuple(x.shape)} {x.dtype} in {time.time() - t0:.0f}s", flush=True)
        for s in seeds:
            train_job(args, layer, s, x, meta)
        del x
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="pi05lib", help="names <out-root>/sae/<model>/ (collect.py --model)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--acts-dir", default=None, help="default <out-root>/sae/<model>/acts")
    ap.add_argument("--sae-dir", default=None, help="default <out-root>/sae/<model>/sae")
    ap.add_argument("--layers", default="all", help="all | emb | pg:0-17 | ae:0-17 (comma separated)")
    ap.add_argument("--input", default="pooled_all", choices=store.PG_INPUTS, help="PaliGemma/emb input")
    ap.add_argument("--ae-steps", default="all", help="action-expert steps used as samples: all | 0,9 | ...")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--expansion", type=float, default=1.0)
    ap.add_argument("--k-pg", type=int, default=100)
    ap.add_argument("--k-ae", type=int, default=64)
    ap.add_argument("--k-aux", type=int, default=512)
    ap.add_argument("--aux-coef", type=float, default=1 / 32)
    ap.add_argument("--dead-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=0, help="total optimizer steps (overrides --epochs)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--gm-samples", type=int, default=10_000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--wandb", default="offline", choices=("offline", "off"))
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain finished / checkpointed jobs from scratch")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1))
    ap.add_argument("--gpus", default="0", help="local GPUs (one process each), or cpu")
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--worker-jobs", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    base = Path(args.out_root) / "sae" / args.model
    args.acts_dir = str(Path(args.acts_dir) if args.acts_dir else base / "acts")
    args.sae_dir = str(Path(args.sae_dir) if args.sae_dir else base / "sae")

    if args.worker_jobs is not None:
        jobs = [(j.split(":")[0], int(j.split(":")[1])) for j in args.worker_jobs.split(",")]
        run_jobs(args, jobs)
        return

    layers = parse_layers(args.layers)
    seeds = [int(s) for s in args.seeds.split(",")]
    all_jobs = [(lay, s) for lay in layers for s in seeds]
    mine = all_jobs[args.shard[0] :: args.shard[1]]
    parts = store.done_parts(Path(args.acts_dir))
    plan = store.read_plan(Path(args.acts_dir))
    n_t = sum(p["rows"] for p in parts)
    n_steps = len(ae_steps_list(args) or plan["identity"]["ae_steps"])

    def cost(job: tuple[str, int]) -> int:
        return n_t * (n_steps if layer_family(job[0]) == "ae" else 1)

    print(
        f"[train] acts {args.acts_dir}: {len(parts)}/{len(plan['parts'])} parts, {n_t} timesteps", flush=True
    )
    if len(parts) < len(plan["parts"]):
        print("[train] WARNING: collection is incomplete; training on the finished parts only", flush=True)
    gpus = [] if args.gpus == "cpu" else [int(g) for g in args.gpus.split(",")]
    print(
        f"[train] shard {args.shard[0]}/{args.shard[1]}: {len(mine)} jobs of {len(all_jobs)} on GPUs {gpus}"
    )
    for lay, s in mine:
        c = cost((lay, s))
        print(
            f"[train]   {lay} seed{s}: {c} samples, {args.steps or args.epochs * math.ceil(c / args.batch)} steps"
        )
    if args.dry_run or not mine:
        return
    from collect import gpu_lock

    if not gpus:
        run_jobs(args, mine)
        return
    with ExitStack() as stack:
        gpu_lock(stack, gpus, args.allow_any_gpu)
        if len(gpus) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
            run_jobs(args, mine)
            return
        load = dict.fromkeys(gpus, 0)
        assign: dict[int, list[tuple[str, int]]] = {g: [] for g in gpus}
        for job in sorted(mine, key=lambda j: -cost(j)):
            g = min(load, key=load.__getitem__)
            assign[g].append(job)
            load[g] += cost(job)
        base_argv = list(argv if argv is not None else sys.argv[1:])
        procs = []
        for g, js in assign.items():
            if not js:
                continue
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
            spec = ",".join(f"{lay}:{s}" for lay, s in js)
            cmd = [sys.executable, str(Path(__file__).resolve()), *base_argv, "--worker-jobs", spec]
            procs.append((g, subprocess.Popen(cmd, env=env)))  # nosec B603
        bad = [g for g, pr in procs if pr.wait() != 0]
        if bad:
            raise SystemExit(f"[train] workers on GPUs {bad} failed; re-run the same command to resume")


if __name__ == "__main__":
    main()
