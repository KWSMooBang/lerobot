"""Do LIBERO-Plus visual variants actually change the image? (Plus venv; T012 review)

    .venvs/libero_plus/bin/python scripts/shortcut/bench/libero_plus/render_check.py \
        --out /outputs/VLA-shortcut-exps/smoke/T012/render_check [--tasks libero_spatial:0,libero_goal:8] [--max 0]

Per task: first frame of the original (init 0 + 10 settle steps, same seed) vs every registered variant of the
visual axes (lighting, texture, camera; --max N per axis to subsample). Writes render_check.csv (variant, level,
mean |Δ| / % changed pixels of agentview and wrist vs the original, md5 of agentview, the variant's scene XML and
its <light> lines) and prints per axis how many variants render identically to the original or to another
variant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from bench.libero_plus import lp_env, registry as reg  # noqa: E402

AXES = ("lighting", "texture", "camera")


def scene_xml_map(src: Path) -> dict[str, str]:
    """problem class name → scene_xml it loads (envs/problems/*.py)."""
    out = {}
    for f in (src / "libero" / "libero" / "envs" / "problems").glob("*.py"):
        text = f.read_text()
        for m in re.finditer(r"class (\w+)\(.*?\):(.*?)(?=\nclass |\Z)", text, re.S):
            x = re.search(r'"scene_xml":\s*"([^"]+)"', m.group(2))
            if x:
                out[m.group(1).lower()] = x.group(1)
    return out


def problem_name(bddl: Path) -> str:
    m = re.search(r"\(problem\s+(\w+)\)", bddl.read_text())
    return m.group(1) if m else ""


def light_lines(xml: Path) -> str:
    if not xml.exists():
        return ""
    return " | ".join(re.sub(r"\s+", " ", m) for m in re.findall(r"<light [^>]*>", xml.read_text()))


def render(v: reg.PlusVariant, hw: int) -> tuple[np.ndarray, np.ndarray]:
    env = lp_env.PlusEnv(v, hw=hw)
    try:
        obs = env.reset(0, 7_000_000)
        return np.asarray(obs["agentview_image"]).copy(), np.asarray(obs["robot0_eye_in_hand_image"]).copy()
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="libero_spatial:0,libero_goal:8")
    ap.add_argument("--axes", default=",".join(AXES))
    ap.add_argument("--max", type=int, default=0, help="variants per axis and task (0 = all)")
    ap.add_argument("--hw", type=int, default=128)
    args = ap.parse_args(argv)
    b = lp_env.bootstrap()
    src = Path(b["src"])
    scenes = scene_xml_map(src)
    assets = src / "libero" / "libero" / "assets"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.tasks.split(","):
        suite, t = item.split(":")
        task_id = int(t)
        vs = reg.load_variants(src, suite)
        a0, w0 = render(reg.original_variant(suite, task_id), args.hw)
        orig_md5 = hashlib.md5(a0.tobytes(), usedforsecurity=False).hexdigest()
        for axis in args.axes.split(","):
            pool = sorted(
                (v for v in vs if v.task_id == task_id and v.dimension == axis), key=lambda v: v.index
            )
            if args.max:
                pool = pool[:: max(1, len(pool) // args.max)][: args.max]
            for v in pool:
                a, w = render(v, args.hw)
                bddl = reg.variant_bddl(src, v)
                prob = problem_name(bddl)
                xml = scenes.get(prob.lower(), "")
                da = np.abs(a.astype(np.int16) - a0.astype(np.int16))
                dw = np.abs(w.astype(np.int16) - w0.astype(np.int16))
                rows.append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "axis": axis,
                        "variant": v.name,
                        "plus_index": v.index,
                        "level": v.level,
                        "detail": v.detail,
                        "agent_mean_abs_diff": round(float(da.mean()), 3),
                        "agent_pct_changed": round(100 * float((da.max(-1) > 0).mean()), 2),
                        "wrist_mean_abs_diff": round(float(dw.mean()), 3),
                        "agent_md5": hashlib.md5(a.tobytes(), usedforsecurity=False).hexdigest(),
                        "same_as_original": hashlib.md5(a.tobytes(), usedforsecurity=False).hexdigest()
                        == orig_md5
                        and not dw.any(),
                        "problem": prob,
                        "scene_xml": xml,
                        "lights": light_lines(assets / xml) if xml else "",
                    }
                )
                r = rows[-1]
                print(
                    f"[check] {suite} t{task_id} {axis:<8} L{v.level} {v.name[-40:]:<40} "
                    f"Δagent {r['agent_mean_abs_diff']:7.3f} ({r['agent_pct_changed']:5.1f}%) "
                    f"Δwrist {r['wrist_mean_abs_diff']:7.3f}",
                    flush=True,
                )
    with open(out / "render_check.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("\nsummary (per task × axis): n, identical to original, sharing an image with another variant")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["suite"], r["task_id"], r["axis"])].append(r)
    for k, rs in groups.items():
        dup = Counter(r["agent_md5"] for r in rs)
        shared = sum(c for c in dup.values() if c > 1)
        same = sum(r["same_as_original"] for r in rs)
        print(
            f"  {k[0]} t{k[1]} {k[2]:<8} n={len(rs):3d}  identical-to-original={same:3d}  in-duplicate-groups={shared:3d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
