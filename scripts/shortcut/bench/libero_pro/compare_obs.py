"""Observation-convention check: the PRO worker's wire obs vs the main harness' (lerobot LiberoEnv) for the same
clean task / init / seed. Run once per venv, then compare:

    .venvs/libero_pro/bin/python scripts/shortcut/bench/libero_pro/compare_obs.py pro  OUT/pro.npz
    uv run python                scripts/shortcut/bench/libero_pro/compare_obs.py main OUT/main.npz
    uv run python                scripts/shortcut/bench/libero_pro/compare_obs.py diff OUT/pro.npz OUT/main.npz

Both after reset → set_init_state → 10 no-op steps at 360 px (the policy server's image_hw).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(SHORTCUT),) if p not in sys.path]

import common  # noqa: E402

CASES = [("libero_spatial", 0, 0), ("libero_goal", 8, 1)]


def pro(out: str) -> None:
    from bench.core.client import robosuite_obs_to_wire
    from bench.libero_pro import catalog, sim

    arrs: dict[str, Any] = {}
    for suite, tid, init in CASES:
        env = sim.ProEnv(catalog.task_entry(catalog.pro_root(), suite, "none", "original", tid), hw=360)
        w = robosuite_obs_to_wire(env.reset(init, common.episode_seed(7, 0, tid, init)))
        arrs.update({f"{suite}_{tid}_{k}": v for k, v in w.items()})
        env.close()
    np.savez(out, **arrs)


def main_venv(out: str) -> None:
    common.ensure_libero_config()
    from bench.core.client import lerobot_libero_obs_to_wire
    from lerobot.envs.libero import LiberoEnv, _get_suite

    arrs: dict[str, Any] = {}
    for suite, tid, init in CASES:
        env = LiberoEnv(
            _get_suite(suite),
            tid,
            suite,
            obs_type="pixels_agent_pos",
            observation_height=360,
            observation_width=360,
        )
        env.init_state_id = init
        seed = common.episode_seed(7, 0, tid, init)
        np.random.seed(seed % 2**32)
        obs, _ = env.reset(seed=seed)
        w = lerobot_libero_obs_to_wire(obs)
        arrs.update({f"{suite}_{tid}_{k}": v for k, v in w.items()})
        env.close()
    np.savez(out, **arrs)


def diff(a: str, b: str) -> None:
    x, y = np.load(a), np.load(b)
    for k in sorted(x.files):
        d = np.abs(x[k].astype(np.float64) - y[k].astype(np.float64))
        extra = f", pixels differing {np.mean(d.max(-1) > 0) * 100:.2f}%" if x[k].dtype == np.uint8 else ""
        print(f"{k:<28} shape {x[k].shape} max |Δ| {d.max():.3g} mean |Δ| {d.mean():.3g}{extra}")


if __name__ == "__main__":
    mode = sys.argv[1]
    {
        "pro": lambda: pro(sys.argv[2]),
        "main": lambda: main_venv(sys.argv[2]),
        "diff": lambda: diff(sys.argv[2], sys.argv[3]),
    }[mode]()
