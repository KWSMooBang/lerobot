"""Collect π0.5 residual-stream activations on the LIBERO training data (Swann et al. 2026, App. A.3).

    uv run python scripts/shortcut/sae/collect.py --model pi05lib --dry-run
    uv run python scripts/shortcut/sae/collect.py --model pi05lib --shard 3/8 --gpus 0,1,2,3

Every dataset frame goes through the policy preprocessor (the one lerobot-train uses: normalization, state
discretized into the prompt, tokenizer) and `PI05Policy.predict_action_chunk` (prefix once + 10 denoising
steps, the inference path) with noise seeded by (episode, frame). `hooks.ActivationRecorder` pools the
residual stream: emb + PaliGemma layers 0-17 per token group (cam1, cam2, text; padding and the empty
camera excluded; valid-token counts saved), action-expert layers 0-17 per denoising step (action-token
mean). Layout and reader: `store.py`.

Prefix-only (default when `--layers` has no action-expert layer; `--full-forward` turns it off): the same
`sample_actions` path is cut right after the deepest requested PaliGemma layer is recorded (`hooks.PrefixDone`),
so the 10 denoising steps and the later PaliGemma layers are skipped and the recorded values are bit-identical.
The first batch of each part still runs in full for the `policy_action0` sanity column (NaN on the other rows).
How a part ran is in done.json (`forward`, `full_forward_batches`), not in the identity.

Splitting: the selected episodes are cut into `--parts` contiguous ranges (default 64, independent of how
the work is distributed). `--shard i/N` takes the i-th of N contiguous blocks of parts (one pod = one shard),
`--gpus` spreads that shard's parts over local GPUs (one process each). A part is the resume unit: finished
parts (done.json, same identity) are skipped, unfinished ones are redone.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import shutil
import subprocess  # nosec B404
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # --gpus = nvidia-smi indices
SHORTCUT = HERE.parent
REPO = SHORTCUT.parents[1]
for p in (str(HERE), str(SHORTCUT), str(SHORTCUT / "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import store  # noqa: E402
from hooks import GROUPS, PrefixDone, layer_dim, layer_family, parse_layers, prefix_stop_layer  # noqa: E402

# --out-root default; the training data is looked up by sc-train's rule (dataset_source): $SC_LIBERO_DATA_ROOT,
# then <out-root>/data/hf_libero, then $SC_OUT_ROOT/data/hf_libero, else the Hub
DEFAULT_OUT_ROOT = os.environ.get("SC_OUT_ROOT", "/outputs/VLA-shortcut-exps")
N_DENOISE = 10  # pi05 num_inference_steps (checked against the loaded config)
# timesteps/s per GPU, ESTIMATES to be replaced by the server smoke (smoke.sh prints the measured value):
# ~4.8 TFLOP per timestep (Gemma-2B prefix 968 tok ≈ 3.9, SigLIP 3 × 256 tok ≈ 0.6, 10 expert steps ≈ 0.3)
# at ~40 % of the dense bf16 peak with fp32 accumulation (RTX 4090 ≈ 83, A100 ≈ 312 TFLOPS).
# A100: measured 17 ts/s by the server smoke (5df0e208, full forward, all 37 layers); RTX 4090: FLOP estimate.
THROUGHPUT = {"RTX 4090": 7.0, "A100": 17.0}
# Share of a full forward's wall time (CPU, real pi05lib, 2 frames; smoke.py --prefix-check): vision + embeddings,
# the 18 PaliGemma layers, the 10 denoising steps. A prefix-only forward that stops after PaliGemma layer L costs
# about VISION + PG_LAYERS·(L+1)/18. Measured full/prefix-only: emb,pg:0-5 3.2×, pg:6-11 2.3×, pg:12-17 1.8×.
TIME_SHARE = {"vision": 0.23, "pg_layers": 0.50, "denoise": 0.27}


def forward_time_fraction(layers: list[str], full_forward: bool) -> float:
    """Estimated wall time per timestep relative to a full forward (1.0)."""
    stop = prefix_stop_layer(layers)
    if full_forward or stop is None:
        return 1.0
    n = 0 if stop == "emb" else int(stop[2:]) + 1
    return TIME_SHARE["vision"] + TIME_SHARE["pg_layers"] * n / 18


def parse_shard(s: str) -> tuple[int, int]:
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"--shard {s}: need 0 <= i < N")
    return i, n


def parse_ae_steps(s: str) -> list[int]:
    if s == "all":
        return list(range(N_DENOISE))
    if s == "none":
        return []
    named = {"first": 0, "last": N_DENOISE - 1}
    out = [named[x.strip()] if x.strip() in named else int(x) for x in s.split(",")]
    if any(not 0 <= x < N_DENOISE for x in out):
        raise argparse.ArgumentTypeError(f"--ae-steps {s}: steps are 0-{N_DENOISE - 1}, first, last")
    return sorted(set(out))


def parse_episodes(s: str | None) -> list[int] | None:
    if not s:
        return None
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return sorted(set(out))


# ---------------------------------------------------------------------------- model / data
def resolve_model(name: str, out_root: Path) -> dict[str, Any]:
    """pi05lib | pi05base | registry id | path | hub id → ckpt + LIBERO adapter (as sc-bench does)."""
    from bench.core import registry

    reg = registry.registry_path(out_root)
    initial = {r["ckpt_id"]: r for r in registry.INITIAL_ROWS}
    if name in initial and name not in registry.read_registry(reg):
        row: dict[str, Any] = {
            **initial[name],
            "adapter": registry.parse_notes(initial[name]["notes"]).get("adapter"),
        }
    else:
        row = registry.resolve(name, reg)
    adapter = row.get("adapter")
    if not adapter:
        from common import POLICY_CKPT, needs_libero_adapter

        adapter = POLICY_CKPT["pi05"] if needs_libero_adapter(row["path_or_hub"]) else None
    return {"model": row["ckpt_id"], "ckpt": row["path_or_hub"], "adapter": adapter}


def _hash_local(p: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p.glob("*.safetensors")) + sorted(p.glob("*.json")) if p.is_dir() else [p]
    for f in files:
        h.update(f.name.encode())
        with open(f, "rb") as fh:
            while block := fh.read(1 << 24):
                h.update(block)
    return "sha256:" + h.hexdigest()[:20]


@functools.cache
def checkpoint_fingerprint(ckpt: str) -> str:
    """Content id of the weights now: `sha256:<hash>` of a local checkpoint's *.safetensors + *.json, or
    `hub:<commit sha>` of the Hub repo's current main. Recorded in the plan; every loader then loads exactly
    that revision (`pinned_path`) and re-checks it after loading (`verify_pinned`)."""
    p = Path(ckpt)
    if p.exists():
        return _hash_local(p)
    if hub_offline():
        return "hub:" + cached_commit(ckpt)
    from huggingface_hub import HfApi

    sha = HfApi().model_info(ckpt).sha
    if not sha:
        raise SystemExit(f"{ckpt}: no commit sha from the Hub")
    return "hub:" + sha


def hub_offline() -> bool:
    from huggingface_hub import constants

    return bool(constants.HF_HUB_OFFLINE)


def cached_commit(repo: str, revision: str = "main") -> str:
    """Commit sha that `revision` points to in the local HF cache (`<cache>/models--org--name/refs/<revision>`),
    for HF_HUB_OFFLINE=1 where the Hub cannot be asked."""
    from huggingface_hub import constants

    ref = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / revision
    sha = ref.read_text().strip() if ref.exists() else ""
    if len(sha) != 40:
        raise SystemExit(
            f"HF_HUB_OFFLINE=1 and no cached revision for {repo} ({ref} missing or invalid): download it once "
            f"online (scripts/shortcut/setup_server.sh) or unset HF_HUB_OFFLINE"
        )
    return sha


def pinned_path(locator: str, revision: str, config_only: bool = False) -> str:
    """Local dir holding exactly `revision` of the checkpoint (`locator` = this pod's hub id or path).
    Hub: the snapshot of that commit (config.json only with `config_only`). Local: the dir itself."""
    if revision.startswith("hub:"):
        from huggingface_hub import hf_hub_download, snapshot_download

        sha = revision[4:]
        if config_only:
            return str(Path(hf_hub_download(locator, "config.json", revision=sha)).parent)
        return snapshot_download(locator, revision=sha)
    if not Path(locator).exists():
        raise SystemExit(f"{locator}: local checkpoint not found (plan revision {revision})")
    return locator


def verify_pinned(path: str, revision: str) -> None:
    """The files at `path` are `revision` (hub: snapshot dir named by the commit; local: content hash)."""
    got = "hub:" + Path(path).name if revision.startswith("hub:") else _hash_local(Path(path))
    if got != revision:
        raise SystemExit(f"checkpoint at {path} is {got}, the plan says {revision}: not mixing weights")


def policy_dtype(ckpt_dir: str, adapter_dir: str | None) -> str:
    """Inference dtype: the config the loader uses (the adapter's when there is one)."""
    import lerobot.policies  # noqa: F401  (registers config types)
    from lerobot.configs.policies import PreTrainedConfig

    return str(getattr(PreTrainedConfig.from_pretrained(adapter_dir or ckpt_dir), "dtype", None))


def dataset_source(out_root: Path) -> tuple[str, Path | None, str]:
    from sc_train import DATASET_REPO, DATASET_REV, local_dataset_root

    return DATASET_REPO, local_dataset_root(out_root), DATASET_REV


def local_data_root(out_root: Path | None = None) -> Path | None:
    """Local HuggingFaceVLA/libero copy by the same rule as collection (None → only the Hub)."""
    return dataset_source(Path(out_root or DEFAULT_OUT_ROOT))[1]


def episode_table(out_root: Path) -> pd.DataFrame:
    from data_variants import suite_of

    from lerobot.datasets import LeRobotDatasetMetadata

    repo, root, rev = dataset_source(out_root)
    meta = LeRobotDatasetMetadata(repo, root=root, revision=rev)
    eps = meta.episodes.to_pandas()
    tasks = meta.tasks.reset_index()
    df = pd.DataFrame(
        {
            "episode_index": eps["episode_index"].astype(int),
            "length": eps["length"].astype(int),
            "task": eps["tasks"].map(lambda t: t[0]),
        }
    ).merge(tasks[["task", "task_index"]], on="task", how="left")
    df["suite"] = df["task"].map(suite_of)
    return df.sort_values("episode_index").reset_index(drop=True)


# ---------------------------------------------------------------------------- plan
def bytes_per_timestep(layers: list[str], n_steps: int, fp_bytes: int = 2) -> int:
    b = 0
    for name in layers:
        b += (len(GROUPS) if layer_family(name) == "pg" else n_steps) * layer_dim(name) * fp_bytes
    return b + len(GROUPS) * 4 + 300  # counts + meta row (approx.)


def make_plan(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    eps = episode_table(out_root)
    want = parse_episodes(args.episodes)
    if want is not None:
        missing = sorted(set(want) - set(eps["episode_index"]))
        if missing:
            raise SystemExit(f"--episodes: unknown episodes {missing[:10]}")
        eps = eps[eps["episode_index"].isin(want)]
    m = resolve_model(args.model, out_root)
    layers = parse_layers(args.layers)
    ae_steps = parse_ae_steps(args.ae_steps)
    repo, root, rev = dataset_source(out_root)
    ck_rev = checkpoint_fingerprint(m["ckpt"])
    ad_rev = checkpoint_fingerprint(m["adapter"]) if m["adapter"] else None
    dtype = policy_dtype(
        pinned_path(m["ckpt"], ck_rev, config_only=True),
        pinned_path(m["adapter"], ad_rev, config_only=True) if m["adapter"] and ad_rev else None,
    )
    # compared between pods: content only (revisions, dtype, data, settings). Locations (hub id or mount path
    # of the checkpoint, dataset root) differ between pods and are kept under "source" for reference.
    identity = {
        "model": m["model"],
        "ckpt_revision": ck_rev,
        "adapter_revision": ad_rev,
        "dtype": dtype,
        "dataset": repo,
        "dataset_revision": rev,
        "layers": layers,
        "ae_steps": ae_steps,
        "episodes": eps["episode_index"].tolist() if want is not None else "all",
        "n_parts": args.parts,
        "noise_seed": args.noise_seed,
        "pooling": "group-mean cam1,cam2,text (pad + empty camera excluded); ae: action-token mean per step",
    }
    parts = store.split_parts(eps["episode_index"].tolist(), eps["length"].tolist(), args.parts)
    lengths = dict(zip(eps["episode_index"], eps["length"], strict=True))
    plan = {
        "identity": identity,
        "parts": [
            {"part_id": store.part_id(p), "episodes": p, "n_frames": int(sum(lengths[e] for e in p))}
            for p in parts
        ],
        "source": {"ckpt": m["ckpt"], "adapter": m["adapter"], "dataset_root": str(root) if root else None},
        "groups": list(GROUPS),
    }
    return plan


def print_dry_run(plan: dict[str, Any], mine: list[int], args: argparse.Namespace, gpus: list[int]) -> None:
    ident = plan["identity"]
    parts = plan["parts"]
    n_all = sum(p["n_frames"] for p in parts)
    n_mine = sum(parts[j]["n_frames"] for j in mine)
    bpt = bytes_per_timestep(ident["layers"], len(ident["ae_steps"]))
    n_pg = sum(layer_family(x) == "pg" for x in ident["layers"])
    src = plan.get("source", {})
    print(
        f"[collect] model {ident['model']} = {src.get('ckpt')} @ {ident['ckpt_revision']} "
        f"(adapter {src.get('adapter')} @ {ident['adapter_revision']}), dtype {ident['dtype']}"
    )
    print(
        f"[collect] layers {len(ident['layers'])}: {n_pg} PaliGemma/emb × {len(GROUPS)} groups × 2048, "
        f"{len(ident['layers']) - n_pg} action expert × {len(ident['ae_steps'])} steps × 1024"
    )
    print(
        f"[collect] episodes {sum(len(p['episodes']) for p in parts)}, timesteps {n_all}, parts {len(parts)}"
    )
    print(f"[collect] shard {args.shard[0]}/{args.shard[1]}: parts {len(mine)}, timesteps {n_mine}")
    print(
        f"[collect] storage fp16: {bpt / 1024:.0f} KB/timestep → shard {n_mine * bpt / 1e9:.1f} GB, "
        f"all {n_all * bpt / 1e9:.1f} GB"
    )
    frac = forward_time_fraction(ident["layers"], getattr(args, "full_forward", False))
    stop = prefix_stop_layer(ident["layers"])
    mode = (
        f"prefix-only, stops after {stop}: ~{frac:.2f}× the full-forward time per timestep"
        if frac < 1
        else "full forward (prefix + 10 denoising steps)"
    )
    print(f"[collect] forward: {mode}")
    for gpu, full_tps in THROUGHPUT.items():
        tps = full_tps / frac
        n_g = max(len(gpus), 1)
        print(
            f"[collect] time on {n_g}× {gpu} (~{tps:.0f} ts/s each): shard {n_mine / tps / n_g / 3600:.2f} h, "
            f"all on 1 GPU {n_all / tps / 3600:.1f} h"
        )


# ---------------------------------------------------------------------------- worker
def noise_for(episodes: np.ndarray, frames: np.ndarray, shape: tuple[int, int], seed: int) -> Any:
    import torch

    out = []
    for e, f in zip(episodes.tolist(), frames.tolist(), strict=True):
        h = int.from_bytes(hashlib.sha256(f"{seed}:{e}:{f}".encode()).digest()[:8], "little") & (2**63 - 1)
        out.append(torch.randn(*shape, generator=torch.Generator().manual_seed(h)))
    return torch.stack(out)


class NonFiniteActivationError(RuntimeError):
    pass


class PartWriter:
    """Memmaps of one part in <part>.tmp/; fp16 unless a layer overflows (then that layer turns fp32)."""

    def __init__(self, tmp: Path, layers: list[str], n_rows: int, n_steps: int):
        self.tmp = tmp
        self.n = n_rows
        self.n_steps = n_steps
        self.arrs: dict[str, Any] = {}  # np.memmap
        self.overflow: list[str] = []
        for name in layers:
            self.arrs[name] = self._open(name, np.float16)
        self.counts: Any = np.lib.format.open_memmap(
            tmp / "counts.npy", "w+", np.int32, (n_rows, len(GROUPS))
        )

    def _shape(self, name: str) -> tuple[int, ...]:
        mid = len(GROUPS) if layer_family(name) == "pg" else self.n_steps
        return (self.n, mid, layer_dim(name))

    def _open(self, name: str, dt: Any) -> Any:
        return np.lib.format.open_memmap(self.tmp / f"{name}.npy", "w+", dt, self._shape(name))

    def write(self, name: str, rows: slice, x: np.ndarray) -> None:
        arr = self.arrs[name]
        if not np.isfinite(x).all():
            bad = int((~np.isfinite(x)).sum())
            raise NonFiniteActivationError(
                f"{name}: {bad} NaN/inf values from the model; this part fails (no done.json)"
            )
        if arr.dtype == np.float16 and np.abs(x).max() > store.FP16_MAX:
            print(
                f"[collect] WARNING {name}: |x| > fp16 max ({np.abs(x).max():.3g}); this layer is stored fp32",
                flush=True,
            )
            old = np.asarray(arr[: rows.start])
            arr.flush()
            del self.arrs[name], arr
            (self.tmp / f"{name}.npy").unlink()
            arr = self._open(name, np.float32)
            arr[: rows.start] = old
            self.arrs[name] = arr
            self.overflow.append(name)
        arr[rows] = x

    def close(self) -> dict[str, str]:
        for a in self.arrs.values():
            a.flush()
        self.counts.flush()
        return {k: str(v.dtype) for k, v in self.arrs.items()}


def load_for_collect(
    ident: dict[str, Any], device: str, src: dict[str, Any]
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """(policy, preprocessor, postprocessor, inference meta), loaded like lerobot-eval / sc-bench, from the
    exact checkpoint revisions in the plan (`src` = this pod's locations of them), re-verified after loading."""
    from common import load_policy

    ck = pinned_path(src["ckpt"], ident["ckpt_revision"])
    ad = pinned_path(src["adapter"], ident["adapter_revision"]) if ident["adapter_revision"] else None
    if bool(src["adapter"]) != bool(ad):
        raise SystemExit(f"adapter {src['adapter']} vs plan adapter_revision {ident['adapter_revision']}")
    policy, pre, post, *_, inf_meta = load_policy(
        "pi05", ck, "libero_spatial", n_action_steps=10, device=device, libero_adapter=ad
    )
    verify_pinned(ck, ident["ckpt_revision"])
    if ad:
        verify_pinned(ad, ident["adapter_revision"])
    return policy, pre, post, inf_meta


def run_parts(args: argparse.Namespace, plan: dict[str, Any], acts_dir: Path, part_idx: list[int]) -> None:
    import torch
    from hooks import ActivationRecorder
    from torch.utils.data import DataLoader, Subset

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ident = plan["identity"]
    want = store.identity_hash(ident)
    parts_dir = acts_dir / "parts"

    def is_done(p: dict[str, Any]) -> bool:
        done = parts_dir / p["part_id"] / "done.json"
        if not done.exists():
            return False
        if json.loads(done.read_text()).get("identity") != want:
            raise SystemExit(f"{done.parent} was written with another identity than {acts_dir}/plan.json")
        return True

    todo = []
    for j in part_idx:
        p = plan["parts"][j]
        if is_done(p):
            print(f"[collect] {p['part_id']} done, skip", flush=True)
        else:
            todo.append(p)
    if not todo:
        return
    device = "cpu" if args.gpus == "cpu" else "cuda"
    src = resolve_model(args.model, Path(args.out_root))  # this pod's locations of the plan's checkpoints
    policy, pre, post, inf_meta = load_for_collect(ident, device, src)
    cfg = policy.config
    if cfg.num_inference_steps != N_DENOISE:
        raise SystemExit(f"num_inference_steps {cfg.num_inference_steps} != {N_DENOISE}")
    if str(getattr(cfg, "dtype", None)) != ident["dtype"]:
        raise SystemExit(f"loaded policy dtype {getattr(cfg, 'dtype', None)} != plan dtype {ident['dtype']}")
    repo, root, rev = dataset_source(Path(args.out_root))
    my_eps = sorted(e for p in todo for e in p["episodes"])
    ds = LeRobotDataset(repo, root=root, revision=rev, episodes=my_eps)
    cols = ds.hf_dataset.select_columns(["index", "episode_index", "frame_index", "task_index"]).to_pandas()
    cols["rel"] = np.arange(len(cols))
    from data_variants import suite_of

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    stop_layer = prefix_stop_layer(ident["layers"])
    prefix_only = not args.full_forward and stop_layer is not None
    # not part of the identity: prefix-only records the same values; only done.json says how it ran
    forward_desc = (
        f"prefix-only (stopped after {stop_layer}; full forward on the first batch of the part for policy_action0)"
        if prefix_only
        else "full (prefix + 10 denoising steps)"
    )
    print(f"[collect] forward: {forward_desc}", flush=True)
    elsewhere = []
    for p in todo:
        pid = p["part_id"]
        if is_done(p):
            print(f"[collect] {pid} finished by another writer, skip", flush=True)
            continue
        lease = store.Lease(parts_dir, pid)
        if not lease.acquire():
            print(f"[collect] {pid} is being written by another writer (fresh lease), skip", flush=True)
            elsewhere.append(pid)
            continue
        lease.start_heartbeat()  # keeps the lease fresh through data loading / flushes too
        tmp = parts_dir / f".{pid}.tmp-{store.unique_tag()}"
        try:
            if is_done(p):  # finished between our check and the lease
                continue
            tmp.mkdir(parents=True)
            t0 = time.time()
            rows = cols[cols["episode_index"].isin(p["episodes"])].sort_values(
                ["episode_index", "frame_index"]
            )
            if len(rows) != p["n_frames"]:
                raise SystemExit(f"{pid}: {len(rows)} frames in the data, plan says {p['n_frames']}")
            writer = PartWriter(tmp, ident["layers"], len(rows), len(ident["ae_steps"]))
            loader = DataLoader(
                Subset(ds, rows["rel"].tolist()),
                batch_size=args.batch,
                num_workers=args.workers,
                shuffle=False,
            )
            meta_rows: list[dict[str, Any]] = []
            pos = 0
            n_full = 0
            rec = ActivationRecorder(policy, ident["layers"], ident["ae_steps"])
            for batch in loader:
                lease.check()  # raises LeaseLostError if the part was taken over
                bsz = len(batch["task"])
                ep = batch["episode_index"].numpy()
                fr = batch["frame_index"].numpy()
                exp = rows.iloc[pos : pos + bsz]
                if not (np.array_equal(ep, exp["episode_index"]) and np.array_equal(fr, exp["frame_index"])):
                    raise SystemExit(f"{pid}: loader order != plan order at row {pos}")
                raw_action = batch["action"].numpy().copy()
                obs = pre(batch)
                noise = noise_for(ep, fr, (cfg.chunk_size, cfg.max_action_dim), ident["noise_seed"])
                # prefix-only: same sample_actions path, cut after the deepest requested PaliGemma layer;
                # the first batch of a part still runs in full for the policy_action0 sanity check
                full = not prefix_only or pos == 0
                rec.stop_after_prefix = not full
                try:
                    with torch.inference_mode(), rec:
                        chunk = policy.predict_action_chunk(obs, noise=noise.to(device))
                except PrefixDone:
                    chunk = None
                if full:
                    if chunk is None or rec.n_steps != N_DENOISE:
                        raise SystemExit(f"recorder saw {rec.n_steps} denoising steps, expected {N_DENOISE}")
                    n_full += 1
                    a0 = post(chunk[:, 0]).float().cpu().numpy()
                else:
                    if chunk is not None:
                        raise SystemExit(f"{pid}: prefix-only forward was not cut at {stop_layer}")
                    a0 = np.full(raw_action.shape, np.nan, np.float32)
                sl = slice(pos, pos + bsz)
                for name, x in rec.pg.items():
                    writer.write(name, sl, x.cpu().numpy())
                for name, x in rec.ae.items():
                    writer.write(name, sl, x.cpu().numpy())
                assert rec.counts is not None
                writer.counts[sl] = rec.counts.cpu().numpy().astype(np.int32)
                for i in range(bsz):
                    task = str(batch["task"][i])
                    meta_rows.append(
                        {
                            "row": pos + i,
                            "index": int(batch["index"][i]),
                            "episode_index": int(ep[i]),
                            "frame_index": int(fr[i]),
                            "task_index": int(batch["task_index"][i]),
                            "suite": suite_of(task),
                            "instruction": task,
                            "action": raw_action[i].tolist(),
                            "policy_action0": a0[i].tolist(),
                        }
                    )
                pos += bsz
                if args.max_batches and pos >= args.max_batches * args.batch:
                    break
            if pos != len(rows):
                raise SystemExit(f"{pid}: wrote {pos} rows of {len(rows)} (--max-batches is for debugging)")
            dtypes = writer.close()
            pd.DataFrame(meta_rows).to_parquet(tmp / "meta.parquet", index=False)
            secs = time.time() - t0
            store.write_json_atomic(
                tmp / "done.json",
                {
                    "identity": want,
                    "part_id": pid,
                    "episodes": p["episodes"],
                    "rows": pos,
                    "forward": forward_desc,
                    "full_forward_batches": n_full,
                    "dtypes": dtypes,
                    "overflow_fp32": writer.overflow,
                    "seconds": round(secs, 1),
                    "timesteps_per_s": round(pos / secs, 2),
                    "gpu": gpu_name,
                    "writer": store.unique_tag(),
                    "batch": args.batch,
                    "tasks_seen": sorted({int(r["task_index"]) for r in meta_rows}),
                    # inference meta from common.load_policy: dtype is a torch.dtype on GPU paths
                    "inference": {
                        k: inf_meta[k] if isinstance(inf_meta[k], int | float | None) else str(inf_meta[k])
                        for k in ("num_inference_steps", "chunk_size", "dtype")
                        if k in inf_meta
                    },
                },
            )
            lease.check()
            if store.publish_part(tmp, parts_dir / pid):
                print(
                    f"[collect] {pid}: {pos} timesteps in {secs:.0f}s ({pos / secs:.1f}/s) on {gpu_name}",
                    flush=True,
                )
            else:
                print(f"[collect] {pid}: another writer published it first; dropped this copy", flush=True)
        except store.LeaseLostError as e:
            print(f"[collect] {pid}: {e}; dropping this copy, the new owner writes the part", flush=True)
            elsewhere.append(pid)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            lease.release()  # only removes the lease file while it still holds our token
    if elsewhere:
        print(
            f"[collect] {len(elsewhere)} parts left to other writers: {elsewhere}; re-run later to pick up "
            f"any whose writer died (lease older than {store.LEASE_TTL_S:.0f}s)",
            flush=True,
        )


# ---------------------------------------------------------------------------- main
def gpu_lock(stack: ExitStack, gpus: list[int], allow_any: bool) -> None:
    from sc_train import LOCK_DIR, acquire_gpu_locks, gpu_inventory

    if LOCK_DIR.is_dir() and not allow_any and any(g != 0 for g in gpus):
        raise SystemExit(f"{LOCK_DIR} exists (shared container): only GPU 0 unless --allow-any-gpu")
    acquire_gpu_locks(stack, gpus, gpu_inventory())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--model", default="pi05lib", help="pi05lib | pi05base | ckpt_registry id | path | hub id"
    )
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--acts-dir", default=None, help="default <out-root>/sae/<model>/acts")
    ap.add_argument("--layers", default="all", help="all | emb,pg:0-17,ae:0-17 (default all 37)")
    ap.add_argument("--ae-steps", default="all", help="all | first,last | 0,5,9")
    ap.add_argument("--episodes", default=None, help="subset, e.g. 0,500,1000 or 0-9 (smoke)")
    ap.add_argument("--parts", type=int, default=64, help="episode ranges the data is cut into (resume unit)")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), help="i/N: this pod's block of parts")
    ap.add_argument("--gpus", default="0", help="local GPUs, one worker process each; cpu = no GPU (tests)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6, help="dataloader workers per GPU (PNG decoding)")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument(
        "--full-forward",
        action="store_true",
        help="with only emb/PaliGemma layers, still run the 10 denoising steps on every batch (default: stop "
        "after the deepest requested PaliGemma layer; same activations, not part of the identity)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--max-batches", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--worker-parts", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    gpus = [] if args.gpus == "cpu" else [int(g) for g in str(args.gpus).split(",") if g != ""]

    if args.worker_parts is not None:  # child: uses the plan.json its parent created / verified
        if not args.acts_dir:
            raise SystemExit("--worker-parts needs --acts-dir")
        plan = store.read_plan(Path(args.acts_dir))
        run_parts(args, plan, Path(args.acts_dir), [int(x) for x in args.worker_parts.split(",")])
        return
    plan = make_plan(args, out_root)
    acts_dir = Path(args.acts_dir) if args.acts_dir else out_root / "sae" / plan["identity"]["model"] / "acts"
    mine = store.shard_parts(len(plan["parts"]), *args.shard)
    print_dry_run(plan, mine, args, gpus)
    print(f"[collect] out {acts_dir}", flush=True)
    if args.dry_run:
        return
    store.create_or_check_plan(acts_dir / "plan.json", plan)
    plan = store.read_plan(acts_dir)  # the verified plan on disk is what every writer (and child) uses
    if not gpus:
        run_parts(args, plan, acts_dir, mine)
        print(f"[collect] parts done: {len(store.done_parts(acts_dir))}/{len(plan['parts'])} in {acts_dir}")
        return
    with ExitStack() as stack:
        gpu_lock(stack, gpus, args.allow_any_gpu)
        if len(gpus) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
            run_parts(args, plan, acts_dir, mine)
        else:
            # greedy: biggest part to the least-loaded GPU
            load = dict.fromkeys(gpus, 0)
            assign: dict[int, list[int]] = {g: [] for g in gpus}
            for j in sorted(mine, key=lambda j: -plan["parts"][j]["n_frames"]):
                g = min(load, key=load.__getitem__)
                assign[g].append(j)
                load[g] += plan["parts"][j]["n_frames"]
            procs = []
            base = list(argv if argv is not None else sys.argv[1:])
            for g, js in assign.items():
                if not js:
                    continue
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *base,
                    "--acts-dir",
                    str(acts_dir),
                    "--worker-parts",
                    ",".join(map(str, sorted(js))),
                ]
                procs.append((g, subprocess.Popen(cmd, env=env)))  # nosec B603
            bad = [g for g, pr in procs if pr.wait() != 0]
            if bad:
                raise SystemExit(f"[collect] workers on GPUs {bad} failed; re-run the same command to resume")
    n_done = len(store.done_parts(acts_dir))
    print(f"[collect] parts done: {n_done}/{len(plan['parts'])} in {acts_dir}", flush=True)


if __name__ == "__main__":
    main()
