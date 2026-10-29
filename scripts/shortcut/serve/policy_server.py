"""Policy server: one lerobot policy on one GPU, served over websocket + msgpack (bench.core.protocol).

    CUDA_VISIBLE_DEVICES=0 uv run --extra shortcut-bench python scripts/shortcut/serve/policy_server.py \
        --ckpt pi05lib [--port 8765] [--batch 1] [--ready-file /tmp/srv.json]

--ckpt: a ckpt_registry.tsv id (`<out-root>/ckpt_registry.tsv`, created with pi05lib/pi05base if missing),
a local pretrained_model dir, or a hub id. LIBERO adapter (`notes: adapter=...` in the registry, or
--libero-adapter): config + processors from a LIBERO finetune, weights from --ckpt (common.load_policy).

--batch 1 (default): exec_mode single — bit-identical to run_eval.py's in-process loop per episode.
--batch N: requests from up to N concurrent episodes (clients) go into one padded call (exec_mode batched-N;
waits until every live episode has a request or --batch-wait-ms passed). Stateful policies (observation
history, temporal ensembling) run per step (exec_mode step, one live episode).
--ready-file: written (JSON: url, port, pid, hello) once the model is loaded and the socket listens.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import os
import signal
import subprocess  # nosec B404
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SHORTCUT = Path(__file__).resolve().parents[1]
sys.path[:0] = [p for p in (str(SHORTCUT), str(Path(__file__).resolve().parent)) if p not in sys.path]

import numpy as np  # noqa: E402
from bench.core import registry  # noqa: E402
from bench.core.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    REQUEST_TYPES,
    ProtocolError,
    error,
    packb,
    unpackb,
)
from websockets.asyncio.server import ServerConnection, serve  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(  # nosec B603 B607
            ["git", "-C", str(SHORTCUT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


class Server:
    def __init__(self, engine: Any, batch_wait_ms: float, log_every_s: float = 60.0):
        self.engine = engine
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="infer")  # GPU + global RNG: one thread
        self.batch_wait = batch_wait_ms / 1000
        self.pending: list[tuple[tuple, asyncio.Future, float]] = []  # (row, future, enqueue time)
        self.wake = asyncio.Event()
        self.ids = itertools.count()
        self.stats = {"calls": 0, "rows": 0, "ms": 0.0, "clients": 0}
        self.log_every = log_every_s
        self._last_log = time.monotonic()

    async def run_engine(self, fn: Any, *a: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self.pool, fn, *a)

    # ------------------------------------------------------------ batching

    async def batch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self.wake.wait()
            self.wake.clear()
            if not self.pending:
                continue
            deadline = self.pending[0][2] + self.batch_wait
            while (
                len(self.pending) < self.engine.batch
                and len(self.pending) < len(self.engine.episodes)
                and loop.time() < deadline
            ):
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.wake.wait(), deadline - loop.time())
                self.wake.clear()
            take, self.pending = self.pending[: self.engine.batch], self.pending[self.engine.batch :]
            rows = [r for r, _, _ in take]
            tic = time.perf_counter()
            try:
                outs = await self.run_engine(self.engine.infer_batch, rows)
            except Exception as e:
                for _, fut, _ in take:
                    if not fut.done():
                        fut.set_exception(e)
            else:
                ms = 1000 * (time.perf_counter() - tic)
                self._count(len(rows), ms)
                for (_, fut, _), out in zip(take, outs, strict=True):
                    if not fut.done():
                        fut.set_result((out, ms))
            if self.pending:
                self.wake.set()

    def _count(self, rows: int, ms: float) -> None:
        s = self.stats
        s["calls"] += 1
        s["rows"] += rows
        s["ms"] += ms
        now = time.monotonic()
        if now - self._last_log > self.log_every:
            self._last_log = now
            print(
                f"[server] calls={s['calls']} rows={s['rows']} {s['ms'] / max(1, s['calls']):.0f} ms/call "
                f"episodes={len(self.engine.episodes)} clients={s['clients']}",
                flush=True,
            )

    # ------------------------------------------------------------ requests

    async def infer(self, eid: str, msg: dict[str, Any]) -> dict[str, Any]:
        obs, instr, step = msg["obs"], msg.get("instruction", ""), bool(msg.get("step", False))
        if not self.engine.need_policy(eid, step):
            return {"ok": True, "actions": self.engine.serve_from_queue(eid), "server_ms": 0.0}
        self.engine.episode(eid)  # unknown episode → error before queuing
        if self.engine.exec_mode.startswith("batched"):
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self.pending.append(((eid, obs, instr, step), fut, loop.time()))
            self.wake.set()
            out, ms = await fut
        else:
            tic = time.perf_counter()
            out = await self.run_engine(self.engine.infer_one, eid, obs, instr, step)
            ms = 1000 * (time.perf_counter() - tic)
            self._count(1, ms)
        return {"ok": True, "actions": np.asarray(out, dtype=np.float32), "server_ms": round(ms, 2)}

    async def dispatch(self, msg: Any, cid: int, owned: set[str]) -> dict[str, Any]:
        if not isinstance(msg, dict) or msg.get("type") not in REQUEST_TYPES:
            raise ProtocolError(f"bad request (type must be one of {REQUEST_TYPES})")
        kind = msg["type"]
        if kind == "hello":
            v = msg.get("protocol_version")
            if v not in (None, PROTOCOL_VERSION):
                raise ProtocolError(f"client protocol {v} != server {PROTOCOL_VERSION}")
            return {"ok": True, **self.engine.hello}
        eid = f"c{cid}:{msg['episode_id']}"
        if kind == "reset":
            await self.run_engine(
                self.engine.reset, eid, int(msg["episode_seed"]), msg.get("instruction", "")
            )
            owned.add(eid)
            if self.engine.exec_mode == "step":
                owned.intersection_update(self.engine.episodes)
            return {"ok": True}
        if kind == "end":
            await self.run_engine(self.engine.end, eid)
            owned.discard(eid)
            self.wake.set()  # a batch may be waiting for this episode
            return {"ok": True}
        return await self.infer(eid, msg)

    async def handler(self, ws: ServerConnection) -> None:
        cid = next(self.ids)
        owned: set[str] = set()
        self.stats["clients"] += 1
        try:
            await self._serve_connection(ws, cid, owned)
        except ConnectionClosed:
            pass  # client went away without a close frame (killed worker); its episodes are freed below
        finally:
            self.stats["clients"] -= 1
            for eid in owned:
                await self.run_engine(self.engine.end, eid)
            self.wake.set()

    async def _serve_connection(self, ws: ServerConnection, cid: int, owned: set[str]) -> None:
        async for raw in ws:
            try:
                if isinstance(raw, str):
                    raise ProtocolError("text frames are not supported (msgpack binary only)")
                resp = await self.dispatch(unpackb(raw), cid, owned)
            except ProtocolError as e:
                resp = error(str(e))
            except Exception as e:
                traceback.print_exc()
                resp = error(f"{type(e).__name__}: {e}")
            await ws.send(packb(resp))


# ---------------------------------------------------------------- setup


def resolve_ckpt(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "mock":
        return {
            "ckpt_id": registry.safe_id(args.ckpt.split("/")[-1]),
            "path_or_hub": args.ckpt,
            "adapter": None,
        }
    path = Path(args.registry) if args.registry else registry.registry_path(args.out_root)
    if registry.ensure_registry(path):
        print(f"[server] created {path} (pi05lib, pi05base)", flush=True)
    row = registry.resolve(args.ckpt, path)
    if args.libero_adapter == "none":
        row["adapter"] = None
    elif args.libero_adapter != "auto":
        row["adapter"] = args.libero_adapter
    elif not row.get("adapter"):
        from engine import auto_libero_adapter

        row["adapter"] = auto_libero_adapter(row["path_or_hub"])
    return row


def build_engine(args: argparse.Namespace) -> Any:
    import torch
    from common import exec_identity
    from engine import Engine, load_any, load_mock

    row = resolve_ckpt(args)
    device = "cpu" if args.backend == "mock" else args.device
    if args.backend == "mock":
        loaded = load_mock(row["path_or_hub"], args.n_action_steps)
    else:
        loaded = load_any(row["path_or_hub"], row["adapter"], args.suite, args.n_action_steps, device)
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    image_map = dict(kv.split("=", 1) for kv in args.image_map.split(",")) if args.image_map else None
    info = {
        "ckpt": row["path_or_hub"],
        "ckpt_id": row["ckpt_id"],
        "ckpt_registered": bool(row.get("registered")),
        "registry_row": {k: v for k, v in row.items() if k not in ("adapter", "registered")},
        "libero_adapter": row["adapter"],
        "backend": args.backend,
        "server": {
            "pid": os.getpid(),
            "device": device,
            "git_sha": git_sha(),
            "batch_wait_ms": args.batch_wait_ms,
        },
    }
    eng = Engine(loaded, device, args.n_action_steps, args.batch, info, image_map)
    eng.hello["exec"] = exec_identity(eng.exec_mode, gpu)
    # inference meta may hold torch dtypes / paths: hello must be msgpack-safe
    eng.hello = json.loads(json.dumps(eng.hello, default=str))
    return eng


async def amain(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    srv = Server(engine, args.batch_wait_ms)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    async with serve(
        srv.handler, args.host, args.port, max_size=None, compression=None, ping_interval=None
    ) as ws:
        port = next(iter(ws.sockets)).getsockname()[1]
        url = f"ws://{args.host}:{port}"
        batcher = asyncio.create_task(srv.batch_loop())
        h = engine.hello
        print(
            f"[server] {url} ckpt={h['ckpt_id']} ({h['ckpt']}) type={h['policy_type']} exec={h['exec_mode']} "
            f"adapter={h['libero_adapter']} gpu={h['exec']['gpu']}",
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
        batcher.cancel()
    print("[server] stopped", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="registry ckpt_id | local pretrained_model dir | hub id")
    ap.add_argument("--registry", default=None, help="default <out-root>/ckpt_registry.tsv")
    ap.add_argument("--out-root", default=None, help=f"default $SC_OUT_ROOT or {registry.DEFAULT_OUT_ROOT}")
    ap.add_argument(
        "--libero-adapter", default="auto", help="auto (registry notes / pi05 default) | none | <ckpt>"
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765, help="0 = any free port (see --ready-file)")
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1, help="1 = exec_mode single; N = batched-N across clients")
    ap.add_argument("--batch-wait-ms", type=float, default=100.0)
    ap.add_argument(
        "--suite", default="libero_goal", help="LiberoEnv config used to build features (same per suite)"
    )
    ap.add_argument("--image-map", default=None, help="image=<policy key>,image2=<policy key>")
    ap.add_argument("--backend", choices=("libero", "mock"), default="libero")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
