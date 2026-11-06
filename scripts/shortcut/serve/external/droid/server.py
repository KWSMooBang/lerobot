"""sc-policy/1 server for openpi π0.5-DROID (lerobot/pi05_droid) on LIBERO, zero-shot (T017). lerobot venv.

    CUDA_VISIBLE_DEVICES=0 uv run --extra shortcut-bench python scripts/shortcut/serve/external/droid/server.py \
        --ckpt lerobot/pi05_droid [--port 8765] [--ready-file F] [--device cuda] [--n-action-steps 8]

Transport and request handling are serve/policy_server.py's `Server`; this file supplies an engine that feeds the
policy what openpi's DROID pipeline feeds it (contract.py) and returns `joint_velocity` actions (8-D: normalized
joint velocities clipped to [-1, 1] + gripper −1 open / +1 close) that the `ours` env executes under DROID's
control contract (joint_control.py, joint_env.py: 15 Hz, 0.2 rad per step at |v| = 1).

Model: lerobot PI05Policy with the checkpoint's weights; its config is changed only in the feature shapes
(observation.state (8,), action (8,): openpi normalizes and tokenizes the 8-D DROID state and keeps actions[:8])
and dtype (--dtype; default bf16 on CUDA as pi05lib runs, fp32 on CPU). The checkpoint ships no normalization
stats: openpi's assets/droid/norm_stats.json (vendored, sha pinned) as lerobot QUANTILES stats.

Noise: exec_mode single (like serve/engine.py): reset seeds torch with episode_seed, and the episode's (CPU, CUDA)
RNG state is restored before / saved after each call. `--backend mock`: a CPU fake model (tests).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess  # nosec B404
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parents[2]
sys.path[:0] = [p for p in (str(HERE), str(SHORTCUT), str(SHORTCUT / "serve")) if p not in sys.path]

import contract  # noqa: E402
import joint_control  # noqa: E402
import numpy as np  # noqa: E402
from bench.core.protocol import (  # noqa: E402
    ACTION_DIMS,
    ACTION_SPACE_JOINT_VELOCITY,
    ORIENT_UPRIGHT,
    PROTOCOL_VERSION,
    STATE_DIM,
    ProtocolError,
    validate_obs,
)

RENDER_HW = (256, 256)  # our LIBERO render; resize_with_pad → 224 like openpi's DROID client
ACTION_DIM = ACTION_DIMS[ACTION_SPACE_JOINT_VELOCITY]


# ---------------------------------------------------------------- models


class LerobotDroidModel:
    """lerobot PI05Policy + DROID pre/post processing. chunk(obs, instr) → unnormalized (chunk_size, 8)."""

    def __init__(self, ckpt_dir: str, device: str, dtype: str):
        import torch
        from common import load_tokenizer

        import lerobot.policies  # noqa: F401  (registers config types)
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
        from lerobot.processor import (
            AddBatchDimensionProcessorStep,
            DeviceProcessorStep,
            NormalizerProcessorStep,
            RenameObservationsProcessorStep,
            TokenizerProcessorStep,
            UnnormalizerProcessorStep,
        )
        from lerobot.processor.factory import make_policy_processor_pipelines
        from lerobot.utils.constants import ACTION, OBS_STATE

        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
        if cfg.type != "pi05":
            raise SystemExit(f"{ckpt_dir}: type {cfg.type}, expected pi05")
        self.stored = {
            "input_features": {k: list(v.shape) for k, v in (cfg.input_features or {}).items()},
            "output_features": {k: list(v.shape) for k, v in (cfg.output_features or {}).items()},
            "chunk_size": cfg.chunk_size,
            "n_action_steps": cfg.n_action_steps,
            "dtype": str(cfg.dtype),
            "normalization_mapping": {str(k): str(v) for k, v in cfg.normalization_mapping.items()},
        }
        img = PolicyFeature(type=FeatureType.VISUAL, shape=(3, *contract.MODEL_IMAGE_HW))
        cfg.input_features = {
            contract.CAMERA_MAP["image"]: img,
            contract.CAMERA_MAP["image2"]: img,
            contract.EMPTY_CAMERA: img,  # not sent: padded with -1 and masked, as openpi's zeros + mask False
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(contract.DROID_DIM,)),
        }
        cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(contract.DROID_DIM,))}
        cfg.device = device
        cfg.dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        cfg.pretrained_path = Path(ckpt_dir)
        self.cfg = cfg
        self.policy = PI05Policy.from_pretrained(ckpt_dir, config=cfg)
        self.policy.eval()
        stats = contract.lerobot_stats(contract.load_openpi_stats())
        feats = {OBS_STATE: cfg.input_features[OBS_STATE], ACTION: cfg.output_features[ACTION]}
        norm_map = cfg.normalization_mapping
        self.pre, self.post = make_policy_processor_pipelines(
            input_steps=[
                RenameObservationsProcessorStep(rename_map={}),
                AddBatchDimensionProcessorStep(),
                NormalizerProcessorStep(features=feats, norm_map=norm_map, stats=stats),
                Pi05PrepareStateTokenizerProcessorStep(max_state_dim=cfg.max_state_dim),
                TokenizerProcessorStep(
                    tokenizer=load_tokenizer(),
                    max_length=cfg.tokenizer_max_length,
                    padding_side="right",
                    padding="max_length",
                ),
                DeviceProcessorStep(device=device),
            ],
            output_steps=[
                UnnormalizerProcessorStep(features={ACTION: feats[ACTION]}, norm_map=norm_map, stats=stats),
                DeviceProcessorStep(device="cpu"),
            ],
        )
        self.chunk_size = int(cfg.chunk_size)
        self.OBS_STATE = OBS_STATE

    def batch(self, obs: dict[str, Any], instruction: str) -> dict[str, Any]:
        """Wire obs → the policy batch before the preprocessor (float [0, 1] CHW 224² images, 8-D state)."""
        t = self.torch
        out: dict[str, Any] = {}
        for key, im in contract.droid_images(obs).items():
            out[key] = t.from_numpy(np.array(im)).permute(2, 0, 1).contiguous().float() / 255
        out[self.OBS_STATE] = t.from_numpy(contract.droid_state(obs))
        out["task"] = str(instruction)
        return out

    def chunk(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        batch = self.pre(self.batch(obs, instruction))
        with self.torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)  # (1, chunk, 8) normalized
        return self.post(actions)[0].float().numpy()


class MockDroidModel:
    """CPU stand-in: unnormalized (chunk, 8) from the images, the DROID state, the instruction length and
    torch.randn from the global RNG (seed determinism is testable). Gripper channel in [0, 1]."""

    chunk_size = 15

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def chunk(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        import torch

        imgs = contract.droid_images(obs)
        state = contract.droid_state(obs)
        self.seen.append({"images": imgs, "state": state, "instruction": instruction})
        feat = sum(float(np.asarray(v, np.float32).mean()) / 255 for v in imgs.values())
        noise = torch.randn(self.chunk_size, contract.DROID_DIM).numpy()
        a = 0.3 * noise + 0.01 * (feat + len(instruction)) - 0.05 * np.r_[state[:7], 0.0]
        a[:, 7] = (noise[:, 7] > 0).astype(np.float32)
        return a.astype(np.float32)


# ---------------------------------------------------------------- engine


@dataclass
class Episode:
    seed: int
    instruction: str
    queue: deque = field(default_factory=deque)
    rng: tuple[Any, Any] | None = None  # (cpu, cuda) global RNG state
    calls: int = 0


class DroidEngine:
    """The engine interface serve/policy_server.Server drives (exec_mode single only)."""

    batch = 1
    exec_mode = "single"

    def __init__(self, model: Any, device: str, n_action_steps: int | None, info: dict[str, Any]):
        import torch

        self.torch = torch
        self.model = model
        self.device = device
        self.cuda = device.startswith("cuda") and torch.cuda.is_available()
        if self.cuda:
            d = torch.device(device)
            self.rng_device = d if d.index is not None else torch.device("cuda", torch.cuda.current_device())
        self.chunk = int(model.chunk_size)
        self.n = int(n_action_steps or joint_control.OPEN_LOOP_HORIZON)
        if not 1 <= self.n <= self.chunk:
            raise SystemExit(f"--n-action-steps {self.n} outside 1..{self.chunk} (chunk_size)")
        self.episodes: dict[str, Episode] = {}
        self.hello = {
            "protocol_version": PROTOCOL_VERSION,
            "policy_type": "pi05",
            "exec_mode": self.exec_mode,
            "requires_step": False,
            "action_space": ACTION_SPACE_JOINT_VELOCITY,
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "requires_joint_pos": True,
            "chunk_size": self.chunk,
            "n_action_steps": self.n,
            "image_keys": dict(contract.CAMERA_MAP),
            "policy_image_features": [
                *contract.CAMERA_MAP.values(),
                f"{contract.EMPTY_CAMERA} (empty, masked)",
            ],
            "image_orientation": ORIENT_UPRIGHT,
            "image_hw": list(info.pop("render_hw", RENDER_HW)),
            "control_mode": "relative",
            **info,
        }

    def _rng_get(self) -> tuple[Any, Any]:
        t = self.torch
        return (t.get_rng_state(), t.cuda.get_rng_state(self.rng_device) if self.cuda else None)

    def _rng_set(self, rng: tuple[Any, Any]) -> None:
        t = self.torch
        t.set_rng_state(rng[0])
        if rng[1] is not None:
            t.cuda.set_rng_state(rng[1], self.rng_device)

    def reset(self, eid: str, seed: int, instruction: str) -> None:
        ep = Episode(int(seed), instruction)
        self.torch.manual_seed(ep.seed)
        ep.rng = self._rng_get()
        self.episodes[eid] = ep

    def end(self, eid: str) -> None:
        self.episodes.pop(eid, None)

    def episode(self, eid: str) -> Episode:
        ep = self.episodes.get(eid)
        if ep is None:
            raise ProtocolError(f"unknown episode {eid!r} (reset first)")
        return ep

    def need_policy(self, eid: str, step: bool) -> bool:
        return not (step and self.episodes.get(eid) is not None and self.episode(eid).queue)

    def serve_from_queue(self, eid: str) -> np.ndarray:
        return np.asarray([self.episode(eid).queue.popleft()])

    def predict_chunk(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        """One policy call (current global RNG) → joint_velocity env actions (chunk, 8)."""
        validate_obs(obs, require_joint_pos=True)
        raw = np.asarray(self.model.chunk(obs, instruction))
        if not np.all(np.isfinite(raw)):
            raise ProtocolError(f"policy returned non-finite actions: {raw}")
        return contract.to_env_actions(raw)

    def infer_one(self, eid: str, obs: dict[str, Any], instruction: str, step: bool) -> np.ndarray:
        ep = self.episode(eid)
        if step and ep.queue:
            return np.asarray([ep.queue.popleft()])
        assert ep.rng is not None
        self._rng_set(ep.rng)
        acts = self.predict_chunk(obs, instruction)[: self.n]
        ep.rng = self._rng_get()
        ep.calls += 1
        if not step:
            ep.queue.clear()
            return acts
        ep.queue = deque(acts[1:])
        return acts[:1]

    def infer_batch(self, rows: list) -> list[np.ndarray]:
        raise ProtocolError("DROID server runs exec_mode single only (--batch 1)")


# ---------------------------------------------------------------- loading


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(  # nosec B603 B607
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def resolve_ckpt(args: argparse.Namespace) -> tuple[str, str]:
    """--ckpt (the pinned hub repo or a local dir) → (local dir, revision)."""
    if Path(args.ckpt).is_dir():
        if not args.allow_unpinned:
            raise SystemExit(
                f"--ckpt {args.ckpt}: local dir is not the pinned {contract.CKPT_REPO}; --allow-unpinned"
            )
        return args.ckpt, f"local:{Path(args.ckpt).resolve()}"
    if args.ckpt != contract.CKPT_REPO and not args.allow_unpinned:
        raise SystemExit(f"--ckpt {args.ckpt!r}: not the pinned {contract.CKPT_REPO} (--allow-unpinned)")
    rev = args.revision or contract.CKPT_SHA
    if rev != contract.CKPT_SHA and not args.allow_unpinned:
        kind = "a commit sha" if SHA_RE.fullmatch(rev) else "a movable name (branch / tag)"
        raise SystemExit(
            f"--revision {rev!r} is {kind}, not the pinned {contract.CKPT_SHA} (--allow-unpinned to serve it)"
        )
    from huggingface_hub import snapshot_download

    path = snapshot_download(args.ckpt, revision=rev)
    return path, snapshot_commit(path)


SHA_RE = re.compile(r"[0-9a-f]{40}")


def snapshot_commit(path: str) -> str:
    """The commit sha a hub snapshot dir resolves to (HF cache layout: .../snapshots/<commit sha>), whatever
    revision name was asked for, so the identity never records a movable name like `main`."""
    sha = Path(path).name
    if not SHA_RE.fullmatch(sha):
        raise SystemExit(
            f"snapshot {path}: cannot read the commit sha from the path (not .../snapshots/<sha>)"
        )
    return sha


def build_engine(args: argparse.Namespace) -> DroidEngine:
    import torch

    if args.backend == "mock":
        model: Any = MockDroidModel()
        device, dtype, ckpt_sha, stored = "cpu", "fp32", "mock", {}
        render_hw = list(args.render_hw or (32, 32))
    else:
        device = args.device
        dtype = args.dtype or ("bf16" if device.startswith("cuda") else "fp32")
        ckpt_dir, ckpt_sha = resolve_ckpt(args)
        model = LerobotDroidModel(ckpt_dir, device, dtype)
        stored = model.stored
        render_hw = list(args.render_hw or RENDER_HW)
    gpu = (
        torch.cuda.get_device_name(torch.device(device))
        if device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    camera = "agentview->base_0_rgb,wrist->left_wrist_0_rgb,right_wrist_0_rgb=empty(masked)"
    identity = {
        "policy_type": "pi05",
        "contract": "openpi-droid",
        "ckpt_repo": contract.CKPT_REPO,
        "ckpt_sha": ckpt_sha,
        "stats_sha256": contract.stats_sha256(),
        "camera": camera,
        "state": "joint_pos(7)+gripper_position(1-width/0.08)",
        "dtype": dtype,
        **joint_control.policy_contract(),
    }
    info = {
        "ckpt": args.ckpt,
        "ckpt_id": args.ckpt_id or args.ckpt.rstrip("/").split("/")[-1],
        "libero_adapter": None,
        "backend": args.backend,
        "render_hw": render_hw,
        "identity": identity,
        "inference": {
            "policy_type": "pi05",
            "contract": "openpi pi05_droid (DroidInputs/DroidOutputs, quantile norm, discrete 8-D state)",
            "chunk_size": model.chunk_size,
            "n_action_steps": int(args.n_action_steps or joint_control.OPEN_LOOP_HORIZON),
            "env_observation_hw": render_hw,
            "model_image_hw": list(contract.MODEL_IMAGE_HW),
            "resize": "openpi_client.image_tools.resize_with_pad (PIL BILINEAR, zero pad)",
            "cameras": camera,
            "state": "wire joint_pos (7) + gripper_position = clip(1 - (q0 - q1) / 0.08, 0, 1)",
            "norm_stats": {"url": contract.NORM_STATS_URL, "sha256": contract.stats_sha256()},
            "normalization": "QUANTILES on 8-D state/action, q99 + 1e-6 (openpi epsilon)",
            "actions": "unnormalized[:8]; clip [-1, 1]; gripper > 0.5 → +1 (close) else -1",
            "dtype": dtype,
            "checkpoint_stored_config": stored,
            "noise": "torch.manual_seed(episode_seed) at reset; per-episode global RNG state restored per call",
            "control": joint_control.policy_contract(),
        },
        "server": {"pid": os.getpid(), "device": device, "git_sha": git_head(SHORTCUT), "kind": "droid"},
    }
    eng = DroidEngine(model, device, args.n_action_steps, info)
    eng.hello["exec"] = {
        "exec_mode": eng.exec_mode,
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    eng.hello = json.loads(json.dumps(eng.hello, default=str))  # msgpack-safe
    return eng


# ---------------------------------------------------------------- main


async def amain(args: argparse.Namespace) -> None:
    from policy_server import Server
    from websockets.asyncio.server import serve

    engine = build_engine(args)
    srv = Server(engine, batch_wait_ms=0.0)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    async with serve(
        srv.handler, args.host, args.port, max_size=None, compression=None, ping_interval=None
    ) as ws:
        port = next(iter(ws.sockets)).getsockname()[1]
        url = f"ws://{args.host}:{port}"
        h = engine.hello
        print(
            f"[droid-server] {url} ckpt={h['ckpt']}@{str(h['identity']['ckpt_sha'])[:8]} "
            f"stats={h['identity']['stats_sha256'][:8]} dtype={h['identity']['dtype']} "
            f"n_action_steps={h['n_action_steps']} gpu={h['exec']['gpu']}",
            flush=True,
        )
        if args.ready_file:
            p = Path(args.ready_file)
            tmp = p.with_name(p.name + f".tmp{os.getpid()}")
            tmp.write_text(
                json.dumps({"url": url, "port": port, "pid": os.getpid(), "hello": h}, default=str)
            )
            tmp.replace(p)
        await stop.wait()
    print("[droid-server] stopped", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=contract.CKPT_REPO, help=f"{contract.CKPT_REPO} (pinned) | local dir")
    ap.add_argument(
        "--revision",
        default=None,
        help=f"hub revision (default and only without --allow-unpinned: {contract.CKPT_SHA[:12]}); recorded as the resolved commit sha",
    )
    ap.add_argument("--ckpt-id", default=None)
    ap.add_argument("--allow-unpinned", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765, help="0 = any free port (see --ready-file)")
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", choices=("bf16", "fp32"), default=None, help="default bf16 on cuda, fp32 on cpu"
    )
    ap.add_argument(
        "--n-action-steps", type=int, default=None, help="default 8 (openpi DROID open_loop_horizon)"
    )
    ap.add_argument("--render-hw", type=int, nargs=2, default=None, help="hello image_hw (default 256 256)")
    ap.add_argument("--batch", type=int, default=1, help="only 1 (exec_mode single)")
    ap.add_argument("--backend", choices=("libero", "mock"), default="libero")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch != 1:
        raise SystemExit("DROID server: --batch must be 1 (exec_mode single; per-episode noise needs it)")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
