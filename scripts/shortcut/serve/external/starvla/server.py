"""sc-policy/1 server for StarVLA checkpoints (T016), run in the StarVLA venv (setup_starvla.sh), not lerobot's.

    source $STARVLA_ROOT/env.sh
    CUDA_VISIBLE_DEVICES=0 $STARVLA_PYTHON scripts/shortcut/serve/external/starvla/server.py \
        --ckpt StarVLA/Qwen3-VL-PI-LIBERO-4in1 [--port 8765] [--ready-file F] [--device cuda]

Transport and request handling are serve/policy_server.py's `Server` (same websocket + msgpack frames,
bench.core.protocol unchanged); this file supplies an engine that turns a wire observation into exactly what
StarVLA's own LIBERO evaluation sends to its policy server, and the policy output into the env action that
evaluation executes (starVLA @ STARVLA_SHA, examples/simBenchmarks/LIBERO/eval_files/{eval_libero.py,
model2libero_interface.py}, deployment/model_server/{server_policy.py,policy_wrapper.py}):

  wire (ours)                                  StarVLA LIBERO eval
  image  = agentview, upright (raw[::-1,::-1])  obs["agentview_image"][::-1, ::-1]              → example.image[0]
  image2 = wrist, upright                       obs["robot0_eye_in_hand_image"][::-1, ::-1]     → example.image[1]
  render 256×256 (hello image_hw)               LIBERO_ENV_RESOLUTION = 256
  resize here: PIL BILINEAR to 224×224          ModelClient: PIL Image.resize((224, 224), BILINEAR) if not 224
  state (8) ignored                             state computed but never sent (example = {image, lang})
  instruction as given                          task.language as is (the model wraps it in CoT_prompt)
  policy: PolicyServerWrapper.predict_action(examples=[ex], unnorm_key=None, do_sample=False, use_ddim=True,
          num_ddim_steps=10) → un-normalized (8, 7), dataset_statistics.json["franka"] min/max for xyz/rpy
          (Libero4in1DataConfig.transform), gripper channel passed through
  action: [:6] as is, gripper 1 - 2·(g > 0.5)   _binarize_gripper_open: 1 = open → -1, else +1
  n_action_steps 8 = action_chunk_size          chunk refreshed when step % action_chunk_size == 0

Noise (flow matching draws torch.randn on the model device from the global RNG): exec_mode single, like
serve/engine.py — reset seeds torch with episode_seed and the episode's (CPU, CUDA) RNG state is restored before
and saved after each call, so an episode's chunks do not depend on other episodes the server interleaves.
`--backend mock`: a CPU fake model with StarVLA's wrapper interface (tests; runs in any venv with torch).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess  # nosec B404
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parents[2]
sys.path[:0] = [p for p in (str(SHORTCUT), str(SHORTCUT / "serve")) if p not in sys.path]

import numpy as np  # noqa: E402
from bench.core.protocol import (  # noqa: E402
    ACTION_DIM,
    ORIENT_UPRIGHT,
    PROTOCOL_VERSION,
    STATE_DIM,
    ProtocolError,
    validate_obs,
)
from PIL import Image  # noqa: E402

# ---------------------------------------------------------------- pins (setup_starvla.sh writes the same values)
STARVLA_SHA = "13d8ad43eb4128554c57227507055ecf9c3a62a2"
CKPT_REPO = "StarVLA/Qwen3-VL-PI-LIBERO-4in1"
CKPT_SHA = "a461ce262ae7c8890d948e122c92fd06cd4803c7"
CKPT_FILE = "checkpoints/steps_100000_pytorch_model.pt"
BASE_VLM_REPO = "Qwen/Qwen3-VL-4B-Instruct"
BASE_VLM_SHA = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
# model card: the released weights need the historical (non-canonical) LayerwiseFM forward
LEGACY_OVERRIDES = ("framework.action_model.diffusion_model_cfg.use_canonical_forward=false",)

# ---------------------------------------------------------------- StarVLA LIBERO eval contract
RENDER_HW = (256, 256)  # eval_libero.LIBERO_ENV_RESOLUTION
MODEL_IMAGE_HW = (224, 224)  # ModelClient(image_size=(224, 224))
GRIPPER_OPEN_THRESHOLD = 0.5  # eval_libero._binarize_gripper_open
PREDICT_KWARGS = {"do_sample": False, "use_ddim": True, "num_ddim_steps": 10}  # ModelClient.step vla_input
CAMERAS = {
    "agentview+wrist": ("image", "image2"),
    "agentview": ("image",),
}  # wire slots → example.image order
DEFAULT_CAMERAS = (
    "agentview+wrist"  # eval_libero.py example_dict; = Libero4in1DataConfig.video_keys (training)
)


def resize_like_starvla(img: np.ndarray, hw: tuple[int, int] = MODEL_IMAGE_HW) -> np.ndarray:
    """ModelClient.step: PIL bilinear resize to (H, W) when the shape differs; else unchanged."""
    arr = np.asarray(img)
    if arr.shape[:2] != tuple(hw):
        arr = np.asarray(Image.fromarray(arr).resize((hw[1], hw[0]), Image.BILINEAR))
    return arr


def wire_to_example(obs: dict[str, Any], instruction: str, cameras: str = DEFAULT_CAMERAS) -> dict[str, Any]:
    """Wire obs (upright images) → StarVLA example {image: [uint8 224×224×3, ...], lang}. No state (the
    StarVLA LIBERO eval does not send it)."""
    validate_obs(obs)
    return {"image": [resize_like_starvla(obs[s]) for s in CAMERAS[cameras]], "lang": str(instruction)}


def to_env_actions(chunk: np.ndarray) -> np.ndarray:
    """Un-normalized StarVLA chunk (T, 7) → LIBERO env actions (T, 7) float32, as eval_libero.py builds them
    (float32 world_vector / rotation_delta, gripper 1 - 2·(open > 0.5))."""
    c = np.asarray(chunk, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] != ACTION_DIM:
        raise ValueError(f"StarVLA chunk must be (T, {ACTION_DIM}), got {c.shape}")
    out = c.copy()
    out[:, 6] = np.float32(1.0) - np.float32(2.0) * (c[:, 6] > GRIPPER_OPEN_THRESHOLD).astype(np.float32)
    return out


# ---------------------------------------------------------------- engine


@dataclass
class Episode:
    seed: int
    instruction: str
    queue: deque = field(default_factory=deque)
    rng: tuple[Any, Any] | None = None  # (cpu, cuda) global RNG state
    calls: int = 0


class StarVLAEngine:
    """The engine interface serve/policy_server.Server drives (single exec mode only)."""

    batch = 1
    exec_mode = "single"

    def __init__(
        self, wrapper: Any, device: str, n_action_steps: int | None, cameras: str, info: dict[str, Any]
    ):
        import torch

        self.torch = torch
        self.w = wrapper
        self.device = device
        self.cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.cameras = cameras
        self.chunk = int(wrapper.metadata["action_chunk_size"])
        self.n = int(n_action_steps or self.chunk)
        if not 1 <= self.n <= self.chunk:
            raise SystemExit(f"--n-action-steps {self.n} outside 1..{self.chunk} (action_chunk_size)")
        self.episodes: dict[str, Episode] = {}
        slots = CAMERAS[cameras]
        self.hello = {
            "protocol_version": PROTOCOL_VERSION,
            "policy_type": "starvla",
            "exec_mode": self.exec_mode,
            "requires_step": False,
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "chunk_size": self.chunk,
            "n_action_steps": self.n,
            "image_keys": {
                "image": "example.image[0] (agentview)",
                "image2": "example.image[1] (wrist)" if "image2" in slots else "unused",
            },
            "policy_image_features": [f"example.image[{i}]<-{s}" for i, s in enumerate(slots)],
            "image_orientation": ORIENT_UPRIGHT,
            "image_hw": list(info.pop("render_hw", RENDER_HW)),
            "control_mode": "relative",
            **info,
        }

    # ------------------------------------------------------------ episodes

    def reset(self, eid: str, seed: int, instruction: str) -> None:
        t = self.torch
        ep = Episode(int(seed), instruction)
        t.manual_seed(ep.seed)
        ep.rng = (t.get_rng_state(), t.cuda.get_rng_state() if self.cuda else None)
        self.episodes[eid] = ep

    def end(self, eid: str) -> None:
        self.episodes.pop(eid, None)

    def episode(self, eid: str) -> Episode:
        ep = self.episodes.get(eid)
        if ep is None:
            raise ProtocolError(f"unknown episode {eid!r} (reset first)")
        return ep

    # ------------------------------------------------------------ inference

    def need_policy(self, eid: str, step: bool) -> bool:
        return not (step and self.episodes.get(eid) is not None and self.episode(eid).queue)

    def serve_from_queue(self, eid: str) -> np.ndarray:
        return np.asarray([self.episode(eid).queue.popleft()])

    def predict_chunk(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        """One StarVLA call (current global RNG) → env actions (chunk, 7)."""
        ex = wire_to_example(obs, instruction, self.cameras)
        out = self.w.predict_action(examples=[ex], unnorm_key=None, **PREDICT_KWARGS)
        return to_env_actions(np.asarray(out["actions"])[0])

    def infer_one(self, eid: str, obs: dict[str, Any], instruction: str, step: bool) -> np.ndarray:
        t = self.torch
        ep = self.episode(eid)
        if step and ep.queue:
            return np.asarray([ep.queue.popleft()])
        assert ep.rng is not None
        t.set_rng_state(ep.rng[0])
        if ep.rng[1] is not None:
            t.cuda.set_rng_state(ep.rng[1])
        acts = self.predict_chunk(obs, instruction)[: self.n]
        ep.rng = (t.get_rng_state(), t.cuda.get_rng_state() if self.cuda else None)
        ep.calls += 1
        if not step:
            ep.queue.clear()
            return acts
        ep.queue = deque(acts[1:])
        return acts[:1]

    def infer_batch(self, rows: list) -> list[np.ndarray]:
        raise ProtocolError("StarVLA server runs exec_mode single only (--batch 1)")


# ---------------------------------------------------------------- loading


class MockWrapper:
    """CPU stand-in with PolicyServerWrapper's surface. The chunk depends on the images (mean of each view),
    the instruction length and torch.randn from the global RNG (seed determinism is testable)."""

    def __init__(self, chunk: int = 8):
        self.metadata = {"action_chunk_size": chunk, "available_unnorm_keys": ["franka"], "mock": True}
        self.seen: list[dict[str, Any]] = []

    def predict_action(self, examples: list[dict], unnorm_key: str | None = None, **kw: Any) -> dict:
        import torch

        self.seen.append({"examples": examples, "unnorm_key": unnorm_key, **kw})
        n = self.metadata["action_chunk_size"]
        out = []
        for ex in examples:
            feat = [float(np.asarray(im, np.float32).mean()) / 255 for im in ex["image"]]
            noise = torch.randn(n, ACTION_DIM).numpy()
            a = 0.01 * noise + 0.001 * (sum(feat) + len(ex["lang"]))
            a[:, 6] = (noise[:, 6] > 0).astype(np.float32)  # open probability-like channel in {0, 1}
            out.append(a.astype(np.float32))
        return {"actions": np.stack(out)}


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(  # nosec B603 B607
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def read_revision(d: Path) -> str | None:
    f = d / ".sc_revision"
    return f.read_text().strip() if f.exists() else None


def resolve_paths(args: argparse.Namespace) -> dict[str, Any]:
    """--ckpt (hub id of the pinned repo, a local *.pt, or a local snapshot dir) → local files + revisions.
    A pinned hub id resolves to $STARVLA_CKPT / $STARVLA_ROOT/models (setup_starvla.sh)."""
    root = Path(os.environ.get("STARVLA_ROOT", ""))
    ck = Path(args.ckpt)
    if ck.suffix == ".pt" and ck.is_file():
        pt = ck
    elif ck.is_dir():
        pt = ck / CKPT_FILE
    elif args.ckpt == CKPT_REPO:
        pt = Path(os.environ.get("STARVLA_CKPT") or root / "models" / "Qwen3-VL-PI-LIBERO-4in1" / CKPT_FILE)
    else:
        raise SystemExit(f"--ckpt {args.ckpt!r}: not a local checkpoint and not the pinned repo {CKPT_REPO}")
    if not pt.is_file():
        raise SystemExit(f"checkpoint {pt} missing; run setup_starvla.sh (or source its env.sh)")
    vlm = Path(
        args.base_vlm or os.environ.get("STARVLA_BASE_VLM") or root / "models" / "Qwen3-VL-4B-Instruct"
    )
    if not (vlm / "config.json").is_file():
        raise SystemExit(f"base VLM {vlm} missing; run setup_starvla.sh or pass --base-vlm")
    src = Path(args.starvla_src or os.environ.get("STARVLA_SRC") or root / "src" / "starVLA")
    revs = {
        "starvla_sha": git_head(src),
        "ckpt_sha": read_revision(pt.parents[1]),
        "base_vlm_sha": read_revision(vlm),
    }
    want = {"starvla_sha": STARVLA_SHA, "ckpt_sha": CKPT_SHA, "base_vlm_sha": BASE_VLM_SHA}
    bad = {k: (revs[k], v) for k, v in want.items() if revs[k] != v}
    if bad and not args.allow_unpinned:
        raise SystemExit(f"not the pinned revisions (have, want): {bad}; --allow-unpinned to serve anyway")
    return {"pt": pt, "vlm": vlm, "src": src, **revs}


def build_engine(args: argparse.Namespace) -> StarVLAEngine:
    import torch

    cameras = args.cameras
    if args.backend == "mock":
        wrapper: Any = MockWrapper()
        device = "cpu"
        paths: dict[str, Any] = {"starvla_sha": "mock", "ckpt_sha": "mock", "base_vlm_sha": "mock"}
        overrides: list[str] = []
        render_hw = list(args.render_hw or (32, 32))  # mock env renders small frames
    else:
        paths = resolve_paths(args)
        sys.path.insert(0, str(paths["src"]))
        from deployment.model_server.policy_wrapper import PolicyServerWrapper

        overrides = [f"framework.qwenvl.base_vlm={paths['vlm']}", *LEGACY_OVERRIDES]
        device = args.device
        wrapper = PolicyServerWrapper(
            ckpt_path=str(paths["pt"]), device=device, use_bf16=args.bf16, config_overrides=overrides
        )
        render_hw = list(args.render_hw or RENDER_HW)
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    meta = dict(wrapper.metadata)
    n_steps = int(args.n_action_steps or meta["action_chunk_size"])
    identity = {
        "policy_type": "starvla",
        "starvla_sha": paths["starvla_sha"],
        "ckpt_repo": CKPT_REPO,
        "ckpt_sha": paths["ckpt_sha"],
        "base_vlm_sha": paths["base_vlm_sha"],
        "camera": "agentview-only" if cameras == "agentview" else cameras,
        "dtype": "bf16" if args.bf16 else "fp32",
        "contract": "starvla-libero-eval",
    }
    info = {
        "ckpt": args.ckpt,
        "ckpt_id": args.ckpt_id or args.ckpt.rstrip("/").split("/")[-1],
        "libero_adapter": None,
        "backend": args.backend,
        "render_hw": render_hw,
        "identity": identity,
        "inference": {
            "policy_type": "starvla",
            "framework": "QwenPI",
            "chunk_size": meta["action_chunk_size"],
            "n_action_steps": n_steps,
            "env_observation_hw": render_hw,
            "control_mode": "relative",
            "model_image_hw": list(MODEL_IMAGE_HW),
            "resize": "PIL BILINEAR (model2libero_interface.ModelClient)",
            "cameras": list(CAMERAS[cameras]),
            "state": "not sent (eval_libero example_dict = {image, lang})",
            "instruction": "as given; StarVLA wraps it in datasets.vla_data.CoT_prompt",
            "unnorm_key": meta.get("default_unnorm_key"),
            "action_unnorm": "PolicyNormProcessor (Libero4in1DataConfig: min_max xyz/rpy, gripper identity)",
            "gripper": f"1 - 2*(open > {GRIPPER_OPEN_THRESHOLD})",
            "predict_kwargs": PREDICT_KWARGS,
            "config_overrides": overrides,
            "use_bf16": args.bf16,
            "attn_implementation": "sdpa (forced by starVLA QWen3.py)",
            "noise": "torch.manual_seed(episode_seed) at reset; per-episode global RNG state restored per call",
            "paths": {k: str(v) for k, v in paths.items() if k in ("pt", "vlm", "src")},
            "starvla_metadata": meta,
        },
        "server": {"pid": os.getpid(), "device": device, "git_sha": git_head(SHORTCUT), "kind": "starvla"},
    }
    eng = StarVLAEngine(wrapper, device, args.n_action_steps, cameras, info)
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
            f"[starvla-server] {url} ckpt={h['ckpt']} starvla={h['identity']['starvla_sha'][:8]} "
            f"ckpt_sha={str(h['identity']['ckpt_sha'])[:8]} cameras={args.cameras} n_action_steps={h['n_action_steps']} "
            f"gpu={h['exec']['gpu']}",
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
    print("[starvla-server] stopped", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ckpt", default=CKPT_REPO, help=f"{CKPT_REPO} (pinned, via env.sh) | local .pt | snapshot dir"
    )
    ap.add_argument("--ckpt-id", default=None)
    ap.add_argument("--base-vlm", default=None, help="default $STARVLA_BASE_VLM")
    ap.add_argument("--starvla-src", default=None, help="default $STARVLA_SRC")
    ap.add_argument("--allow-unpinned", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765, help="0 = any free port (see --ready-file)")
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="card: --use_bf16")
    ap.add_argument("--cameras", choices=sorted(CAMERAS), default=DEFAULT_CAMERAS)
    ap.add_argument(
        "--n-action-steps", type=int, default=None, help="default = action_chunk_size (StarVLA eval)"
    )
    ap.add_argument("--render-hw", type=int, nargs=2, default=None, help="hello image_hw (default 256 256)")
    ap.add_argument("--batch", type=int, default=1, help="only 1 (exec_mode single)")
    ap.add_argument("--backend", choices=("libero", "mock"), default="libero")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch != 1:
        raise SystemExit("StarVLA server: --batch must be 1 (exec_mode single; per-episode noise needs it)")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
