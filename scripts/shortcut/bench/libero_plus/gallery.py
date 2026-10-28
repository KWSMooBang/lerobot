"""Example gallery (T012): what each LIBERO-Plus axis changes, on fixed example tasks (Plus venv).

    .venvs/libero_plus/bin/python scripts/shortcut/bench/libero_plus/gallery.py \
        --out /outputs/VLA-shortcut-exps/analysis/examples/libero_plus [--tasks libero_spatial:0,libero_goal:8]

Per task: the original (Plus env, no perturbation) and, per axis, the registered variants at the lowest and
the highest difficulty level the task has (first by registry index; motion blur left out without
libMagickWand). Image = the first frame the policy sees (init state 0 + 10 settle steps), agentview | wrist
upright, 2 × 256 px; language variants get the instruction drawn under the scene (the scene is unchanged).
manifest.json: T005 format (id, title, caption, path) + suite, task_id, dimension, level, detail, variant,
instruction_original / instruction_given (what sc-bench gives in the default clean mode) / instruction_registry,
cue (what the axis takes away), robot_qpos_delta (max |Δ joint| vs the original, rad).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from bench.core.client import robosuite_obs_to_wire  # noqa: E402
from bench.libero_plus import lp_env, registry as reg  # noqa: E402

SUITE_TITLE = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}
AXIS_TITLE = {
    "none": "원본",
    "language": "언어",
    "layout": "물체 배치",
    "camera": "카메라 시점",
    "robot": "로봇 초기 자세",
    "lighting": "조명",
    "texture": "배경 질감",
    "noise": "센서 노이즈",
}
SIDE = 256


def picks(vs: list[reg.PlusVariant], task_id: int, dim: str, wand: bool) -> list[list[reg.PlusVariant]]:
    """Candidates for the lowest- and the highest-difficulty example, nearest level first, registry order inside
    a level; main() takes the first candidate whose frame differs from the axis' earlier picks (half of the
    light variants render the same unlit scene: both spotlights aim away from the table)."""
    pool = [v for v in vs if v.task_id == task_id and v.dimension == dim and (wand or not reg.motion_blur(v))]
    levels: list[int] = sorted({v.level for v in pool if v.level is not None})
    if not levels:
        return [sorted(pool, key=lambda v: v.index)]
    ends = [levels, levels[::-1]] if len(levels) > 1 else [levels]
    return [
        sorted(
            (v for v in pool if v.level is not None), key=lambda v: (order.index(int(v.level or 0)), v.index)
        )
        for order in ends
    ]


def same_image(a: np.ndarray, b: np.ndarray, tol: float = 2.0) -> bool:
    """Visually the same frame: mean |Δ| below `tol` (unlit light variants differ only by a specular glint)."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) < tol


Frame = tuple[np.ndarray, np.ndarray]  # (image, robot qpos)


def choose(
    cands: list[list[reg.PlusVariant]], render: Callable[[reg.PlusVariant], Frame], dim: str, tries: int = 12
) -> list[tuple[reg.PlusVariant, np.ndarray, np.ndarray]]:
    """One example per candidate list: its first variant whose frame is not (almost) the same as an earlier
    pick of the axis (language: the scene never changes, so just the first). Empty lists, and lists whose
    `tries` first candidates all repeat an earlier pick, give no example."""
    seen: list[np.ndarray] = []
    chosen: list[tuple[reg.PlusVariant, np.ndarray, np.ndarray]] = []
    for cs in cands:
        found = None
        for v in cs[:tries]:
            img, qpos = render(v)
            if dim == "language" or not any(same_image(img, x) for x in seen):
                found = (v, img, qpos)
                break
            print(f"[gallery] {v.name}: renders (almost) like an earlier pick, next candidate", flush=True)
        if found is None or (dim == "language" and any(c[0].name == found[0].name for c in chosen)):
            continue
        seen.append(found[1])
        chosen.append(found)
    return chosen


def text_band(width: int, lines: list[str]) -> np.ndarray:
    import cv2

    band: np.ndarray = np.full((10 + 18 * len(lines), width, 3), 255, np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(band, s, (6, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return band


def first_frame(v: reg.PlusVariant, hw: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    env = lp_env.PlusEnv(v, hw=hw)
    try:
        obs = env.reset(0, 7_000_000)
        wire = robosuite_obs_to_wire(obs)
        qpos = np.asarray(env.env.sim.data.qpos[env.env.robots[0]._ref_joint_pos_indexes], dtype=np.float64)
    finally:
        env.close()
    views = [cv2.resize(wire[k], (SIDE, SIDE), interpolation=cv2.INTER_AREA) for k in ("image", "image2")]
    return np.concatenate(views, axis=1), qpos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="libero_spatial:0,libero_goal:8")
    ap.add_argument("--hw", type=int, default=360, help="render size (policy server image_hw)")
    args = ap.parse_args(argv)
    import imageio

    b = lp_env.bootstrap()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for item in args.tasks.split(","):
        suite, t = item.split(":")
        task_id = int(t)
        vs = reg.load_variants(Path(b["src"]), suite)
        original = reg.language_from_name(reg.ORIGINAL_TASKS[suite][task_id])
        short = {"libero_spatial": "spat", "libero_object": "obj", "libero_goal": "goal", "libero_10": "l10"}[
            suite
        ]
        q0: np.ndarray | None = None
        for dim in reg.AXES:
            cands = (
                [[reg.original_variant(suite, task_id)]]
                if dim == "none"
                else picks(vs, task_id, dim, b["wand"])
            )
            chosen = choose(cands, lambda v: first_frame(v, args.hw), dim)
            for v, img, qpos in chosen:
                if dim == "none":
                    q0 = qpos
                registry_instr = lp_env.instruction(v)
                given = registry_instr if dim == "language" else original
                if dim == "language":
                    img = np.concatenate(
                        [img, text_band(img.shape[1], ["> " + s for s in textwrap.wrap(given, 70)][:4])],
                        axis=0,
                    )
                lv = f"L{v.level}" if v.level is not None else "L-"
                vid = "orig" if dim == "none" else f"{dim}_{lv}_{v.detail}"
                name = f"plus_{short}_t{task_id}_{vid}"
                path = out / f"{name}.png"
                imageio.imwrite(path, img)
                dq = None if q0 is None else float(np.max(np.abs(qpos - q0)))
                title = f"{SUITE_TITLE[suite]} t{task_id} · {AXIS_TITLE[dim]}" + (
                    "" if dim == "none" else f" {lv} ({v.detail})"
                )
                caption = reg.AXIS_CUE[dim]
                if dim == "language":
                    caption = f"지시문: “{given}” (원래: “{original}”). " + caption
                elif dim == "robot":
                    caption += f" (관절 최대 차이 {dq:.2f} rad)"
                manifest.append(
                    {
                        "id": name,
                        "section": "libero_plus",
                        "title": title,
                        "caption": caption,
                        "path": str(path),
                        "benchmark": "libero_plus",
                        "suite": suite,
                        "task_id": task_id,
                        "dimension": dim,
                        "level": v.level,
                        "detail": v.detail,
                        "variant": v.name,
                        "plus_index": v.index,
                        "params": v.params,
                        "instruction_original": original,
                        "instruction_given": given,
                        "instruction_registry": registry_instr,
                        "cue": reg.AXIS_CUE[dim],
                        "robot_qpos_delta": dq,
                        "bench_version": lp_env.bench_version(),
                        "bytes": path.stat().st_size,
                    }
                )
                print(f"[gallery] {name}: {v.name}  dq={dq}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"[gallery] {len(manifest)} images → {out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
