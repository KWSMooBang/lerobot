"""Action equality: our sc-policy/1 StarVLA server vs StarVLA's own LIBERO evaluation path, same model, same noise.

    source $STARVLA_ROOT/env.sh
    $STARVLA_PYTHON scripts/shortcut/serve/external/starvla/compare.py --frames frames.npz [--device cuda] \
        [--chunks 2] [--out compare.json]

One model is loaded (PolicyServerWrapper, the pinned checkpoint + card overrides) and served twice in this process:
  reference  StarVLA's WebsocketPolicyServer + examples/simBenchmarks/LIBERO/eval_files ModelClient, driven the
             way eval_libero.py drives them: raw LIBERO frames rotated [::-1, ::-1], [agentview, wrist],
             client.step(example, step) per env step, action = world_vector + rotation_delta +
             eval_libero._binarize_gripper_open (the function's own source, compiled from eval_libero.py).
  ours       server.py's StarVLAEngine behind serve/policy_server.Server, driven by bench.core.client.PolicyClient
             with wire obs (upright = raw[::-1, ::-1]) — what the `ours` harness sends.
Before each episode the reference runs torch.manual_seed(episode_seed) (our engine does the same at reset), so
both draw the same flow noise. Frames (npz from dump_frames.py, lerobot venv): agentview / wrist uint8
[N, 256, 256, 3] in raw LIBERO orientation, `instruction`, `seed`; frame k is the observation at the start of
chunk k. Exit 0 iff every action matches exactly (max |Δ| = 0). Also reports how far agentview-only inputs are.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parents[2]
sys.path[:0] = [p for p in (str(HERE), str(SHORTCUT), str(SHORTCUT / "serve")) if p not in sys.path]

import server as sv  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def starvla_binarize(src: Path) -> Any:
    """eval_libero._binarize_gripper_open, compiled from StarVLA's file (eval_libero imports LIBERO itself)."""
    tree = ast.parse((src / "examples/simBenchmarks/LIBERO/eval_files/eval_libero.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_binarize_gripper_open")
    ns: dict[str, Any] = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "eval_libero.py", "exec"), ns)  # nosec B102
    return ns["_binarize_gripper_open"]


def start_thread_loop(coro_fn: Any) -> None:
    def run() -> None:
        asyncio.run(coro_fn())

    threading.Thread(target=run, daemon=True).start()


def wait_port(port: int, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(port)


def reference_actions(wrapper: Any, src: Path, frames: dict[str, np.ndarray], n_chunks: int) -> np.ndarray:
    """StarVLA's own server + LIBERO ModelClient, eval_libero.py's loop body (no env: frame k per chunk)."""
    import torch
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient

    port = free_port()
    srv = WebsocketPolicyServer(policy=wrapper, host="127.0.0.1", port=port, metadata=dict(wrapper.metadata))
    start_thread_loop(srv.run)
    wait_port(port)
    binarize = starvla_binarize(src)
    client = ModelClient(host="127.0.0.1", port=port, unnorm_key=None)
    chunk = client.action_chunk_size
    task = str(frames["instruction"])
    torch.manual_seed(int(frames["seed"]))
    client.reset(task_description=task)
    out = []
    for step in range(n_chunks * chunk):
        k = step // chunk
        img = np.ascontiguousarray(frames["agentview"][k][::-1, ::-1])
        wrist = np.ascontiguousarray(frames["wrist"][k][::-1, ::-1])
        example = {"image": [img, wrist], "lang": task}
        raw = client.step(example=example, step=step)["raw_action"]
        wv = np.asarray(raw.get("world_vector"), dtype=np.float32).reshape(-1)
        rot = np.asarray(raw.get("rotation_delta"), dtype=np.float32).reshape(-1)
        grip = binarize(np.asarray(raw.get("open_gripper"), dtype=np.float32).reshape(-1))
        out.append(np.concatenate([wv, rot, grip], axis=0))
    return np.stack(out)


def ours_actions(
    wrapper: Any, frames: dict[str, np.ndarray], n_chunks: int, cameras: str = sv.DEFAULT_CAMERAS
) -> tuple[np.ndarray, dict]:
    """Our engine behind policy_server.Server, PolicyClient with wire obs (what bench/ours sends)."""
    from bench.core.client import PolicyClient, upright
    from policy_server import Server
    from websockets.asyncio.server import serve

    info = {"ckpt": "compare", "ckpt_id": "compare", "identity": {"camera": cameras}, "inference": {}}
    eng = sv.StarVLAEngine(wrapper, "cuda" if wrapper_on_cuda(wrapper) else "cpu", None, cameras, info)
    import torch

    eng.hello["exec"] = {
        "exec_mode": "single",
        "gpu": "-",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    eng.hello = json.loads(json.dumps(eng.hello, default=str))
    port = free_port()
    s = Server(eng, batch_wait_ms=0.0)

    async def run() -> None:
        async with serve(s.handler, "127.0.0.1", port, max_size=None, compression=None, ping_interval=None):
            await asyncio.Event().wait()

    start_thread_loop(run)
    wait_port(port)
    pc = PolicyClient(f"ws://127.0.0.1:{port}")
    task = str(frames["instruction"])
    pc.reset("cmp", int(frames["seed"]), task)
    out = []
    for k in range(n_chunks):
        obs = {
            "image": upright(frames["agentview"][k]),
            "image2": upright(frames["wrist"][k]),
            "state": np.asarray(frames.get("state", np.zeros((n_chunks, 8)))[k], dtype=np.float32),
        }
        out.append(pc.infer("cmp", obs, task))
    pc.close()
    return np.concatenate(out), pc.info


def wrapper_on_cuda(wrapper: Any) -> bool:
    fw = getattr(wrapper, "_framework", None)
    if fw is None:
        return False
    return next(fw.parameters()).is_cuda


def compare(wrapper: Any, src: Path, frames: dict[str, np.ndarray], n_chunks: int) -> dict[str, Any]:
    ref = reference_actions(wrapper, src, frames, n_chunks)
    ours, hello = ours_actions(wrapper, frames, n_chunks)
    agent_only, _ = ours_actions(wrapper, frames, n_chunks, cameras="agentview")
    d = np.abs(ref.astype(np.float64) - ours.astype(np.float64))
    da = np.abs(ref.astype(np.float64) - agent_only.astype(np.float64))
    return {
        "pass": bool(ref.shape == ours.shape and d.max() == 0.0),
        "n_actions": int(ref.shape[0]),
        "max_abs_diff": float(d.max()),
        "n_equal_actions": int((d.max(axis=1) == 0).sum()),
        "agentview_only_max_abs_diff_xyzrpy": float(da[:, :6].max()),
        "agentview_only_gripper_flips": int((da[:, 6] > 0).sum()),
        "reference": ref.tolist(),
        "ours": ours.tolist(),
        "hello_n_action_steps": hello["n_action_steps"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--frames", required=True, help="npz from dump_frames.py (several files: comma-separated)"
    )
    ap.add_argument("--ckpt", default=sv.CKPT_REPO)
    ap.add_argument("--base-vlm", default=None)
    ap.add_argument("--starvla-src", default=None)
    ap.add_argument("--allow-unpinned", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    paths = sv.resolve_paths(args)
    sys.path.insert(0, str(paths["src"]))
    os.chdir(paths["src"])  # ModelClient's module imports `examples.*` / `deployment.*` relative to the repo
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    overrides = [f"framework.qwenvl.base_vlm={paths['vlm']}", *sv.LEGACY_OVERRIDES]
    tic = time.time()
    wrapper = PolicyServerWrapper(
        str(paths["pt"]), device=args.device, use_bf16=args.bf16, config_overrides=overrides
    )
    load_s = time.time() - tic
    results = []
    for f in args.frames.split(","):
        z = np.load(f, allow_pickle=False)
        frames = {k: z[k] for k in z.files}
        tic = time.time()
        r = compare(wrapper, paths["src"], frames, args.chunks)
        r.update(frames=f, instruction=str(frames["instruction"]), seconds=round(time.time() - tic, 1))
        results.append(r)
        print(
            f"[compare] {Path(f).name}: {'PASS' if r['pass'] else 'FAIL'} {r['n_equal_actions']}/{r['n_actions']} "
            f"actions identical, max|Δ|={r['max_abs_diff']:.3g}; agentview-only: max|Δxyzrpy|="
            f"{r['agentview_only_max_abs_diff_xyzrpy']:.3g}, gripper flips {r['agentview_only_gripper_flips']} "
            f"({r['seconds']} s)",
            flush=True,
        )
    summary = {
        "pass": all(r["pass"] for r in results),
        "device": args.device,
        "bf16": args.bf16,
        "load_s": round(load_s, 1),
        "revisions": {k: paths[k] for k in ("starvla_sha", "ckpt_sha", "base_vlm_sha")},
        "results": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[compare] {'PASS' if summary['pass'] else 'FAIL'}", flush=True)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
