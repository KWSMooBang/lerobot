"""Observation frames for compare.py (run in the lerobot venv): raw-orientation LIBERO agentview / wrist frames.

    uv run python scripts/shortcut/serve/external/starvla/dump_frames.py --source libero-env \
        --suite libero_goal --task 1 --out f.npz          # real LIBERO render (GPU server; EGL)
    uv run python scripts/shortcut/serve/external/starvla/dump_frames.py --source hf-libero \
        --dataset /outputs/VLA-shortcut-exps/data/hf_libero --episode 0 --out f.npz   # CPU: training frames

libero-env: the env `bench/ours` uses (envproc._make_env, T009 fix env, 256×256 render), reset with the harness's
episode seed for init 0; frame k = the observation after 8·k LIBERO dummy actions (one frame per chunk).
hf-libero: HuggingFaceVLA/libero frames 8·k of one episode (stored upright) rotated back to the raw render.
npz keys: agentview, wrist (uint8 [chunks, H, W, 3], raw orientation), state (float32 [chunks, 8]),
instruction, seed, source.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[3]
sys.path[:0] = [p for p in (str(SHORTCUT), str(SHORTCUT / "parallel")) if p not in sys.path]


def from_env(suite: str, task: int, chunks: int, seed_base: int) -> dict:
    from bench.ours.worker import wire_obs
    from common import episode_seed
    from envproc import _make_env
    from run_eval import libero

    spec = {
        "suite": suite,
        "task_id": task,
        "max_steps": 600,
        "env_hw": [256, 256],
        "control_mode": "relative",
    }
    env = _make_env("libero", spec, {})
    seed = episode_seed(seed_base, 0, task, 0)
    np.random.seed(seed % 2**32)
    env.init_state_id = 0
    obs, _ = env.reset(seed=seed)
    agent, wrist, state = [], [], []
    for _k in range(chunks):
        agent.append(np.asarray(obs["pixels"]["image"]))
        wrist.append(np.asarray(obs["pixels"]["image2"]))
        state.append(wire_obs(obs)["state"])
        for _ in range(8):
            obs, *_ = env.step(np.asarray([0.0] * 6 + [-1.0]))
    instr = libero().get_suite(suite).get_task(task).language
    env.close()
    return {"agentview": agent, "wrist": wrist, "state": state, "instruction": instr, "seed": seed}


def from_dataset(root: Path, episode: int, chunks: int, seed_base: int) -> dict:
    import pyarrow.parquet as pq
    from PIL import Image

    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
    for f in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        t = pq.read_table(f, filters=[("episode_index", "=", episode)])
        if t.num_rows:
            break
    else:
        raise SystemExit(f"episode {episode} not found under {root}/data")
    df = t.to_pandas().sort_values("frame_index")
    rows = [df.iloc[8 * k] for k in range(chunks)]

    def raw(cell: dict) -> np.ndarray:  # stored upright (training convention) → raw LIBERO render
        up = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
        return np.ascontiguousarray(up[::-1, ::-1])

    ti = int(rows[0]["task_index"])
    instr = tasks.index[tasks["task_index"] == ti][0] if "task_index" in tasks else tasks.iloc[ti]["task"]
    return {
        "agentview": [raw(r["observation.images.image"]) for r in rows],
        "wrist": [raw(r["observation.images.image2"]) for r in rows],
        "state": [np.asarray(r["observation.state"], np.float32) for r in rows],
        "instruction": str(instr),
        "seed": seed_base * 1_000_000 + episode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("libero-env", "hf-libero"), required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--dataset", default="/outputs/VLA-shortcut-exps/data/hf_libero")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.source == "libero-env":
        d = from_env(a.suite, a.task, a.chunks, a.seed)
    else:
        d = from_dataset(Path(a.dataset), a.episode, a.chunks, a.seed)
    np.savez(
        a.out,
        agentview=np.stack(d["agentview"]).astype(np.uint8),
        wrist=np.stack(d["wrist"]).astype(np.uint8),
        state=np.stack(d["state"]).astype(np.float32),
        instruction=np.asarray(d["instruction"]),
        seed=np.asarray(int(d["seed"])),
        source=np.asarray(a.source),
    )
    print(
        f"[dump_frames] {a.out}: {a.source} {d['instruction']!r} seed {d['seed']} frames {len(d['agentview'])}"
    )


if __name__ == "__main__":
    main()
