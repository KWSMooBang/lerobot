"""LIBERO-PRO example gallery (PRO venv, CPU rendering is fine): first frame of each axis for fixed example tasks.

    .venvs/libero_pro/bin/python scripts/shortcut/bench/libero_pro/gallery.py \
        --out /outputs/VLA-shortcut-exps/analysis/examples/libero_pro

Per (task, axis, level): `<id>.png` (agentview as the policy sees it, after the 10 settle steps, init 0) and
`<id>_vs_orig.png` (clean | perturbed, captions), the given / original instruction, what changed in the BDDL
(object types, region swaps, table, goal), object drift over 50 no-op steps after settling (PRO warns that
non-main tables can let objects move), and one line on which cue the axis cuts. manifest.json = list of entries
in the T005 manifest style (id, category, benchmark, suite, task_id, init, conditions, instructions, why, png, ...).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from common import episode_seed  # noqa: E402

from bench.libero_pro import catalog, sim  # noqa: E402
from bench.libero_pro.instruction_table import word_edit  # noqa: E402

EXAMPLES = [("libero_spatial", 0), ("libero_goal", 8)]
# x/y shifts exist for libero_object only (PRO repo): shown on Object t0 at both ends of each axis
POSITION_EXTREMES = [("libero_object", 0, lv) for lv in ("x0.1", "x0.5", "y0.1", "y0.5")]
CUE = {
    "none": "기준 화면: 원래 LIBERO 태스크, 학습 지시문",
    "object": "물체 외형 단서를 끊는다: 물체 색·크기 교체({change}) — 학습 때 본 외형으로 대상을 찾는 지름길이면 실패",
    "position": "위치 단서를 끊는다: 물체들이 서로의 초기 자리로 바뀜({change}) — '그 자리 → 그 동작' 암기면 엉뚱한 물체·자리로 간다",
    "position_xy": "절대 위치 단서를 끊는다: 대상 물체의 초기 영역을 {axis}축으로 {level} 단위 이동({change}) — 작은(0.1)·큰(0.5) 이동 양 끝",
    "semantic": "문장 템플릿 단서를 끊는다: 장면·목표는 그대로 두고 지시문만 바꿔 씀({edit}) — 학습 문장 암기에 기대면 실패",
    "task": "장면→궤적 단서를 끊는다: 장면은 그대로, 목표·지시문을 바꿈({edit}) — 지시를 무시하고 익숙한 동작을 재생하면 원래 goal 달성(success_orig)",
    "environment": "배경 단서를 끊는다: 작업대 교체({change}) — 테이블 외형과 동작의 연합에 기대면 실패",
}
AXIS_TITLE = {
    "none": "원본",
    "object": "PRO 물체(object)",
    "position": "PRO 위치(position)",
    "semantic": "PRO 문장(semantic)",
    "task": "PRO 태스크(task)",
    "environment": "PRO 환경(environment)",
}
SUITE_TITLE = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}


def section(text: str, head: str) -> str:
    start = text.find(f"({head}")
    depth = 0
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return text[start : i + 1]
    return ""


def objects(text: str) -> dict[str, str]:
    return {m[0]: m[1] for m in re.findall(r"(\w+)\s+-\s+(\w+)", section(text, ":objects"))}


def on_regions(text: str) -> dict[str, str]:
    return dict(re.findall(r"\(On\s+(\w+)\s+(\w+)\)", section(text, ":init")))


def ranges(text: str) -> list[str]:
    return re.findall(r"\(:ranges\s*\(\s*\(([-\d.\s]+)\)", text)


def tables(text: str) -> list[str]:
    return sorted(
        set(re.findall(r"\b(main_table|kitchen_table|living_room_table|study_table|floor)\b", text))
    )


def pad_to(img: np.ndarray, h: int) -> np.ndarray:
    return np.concatenate([img, np.full((h - img.shape[0], img.shape[1], 3), 255, np.uint8)])


def describe_change(e: dict[str, Any]) -> dict[str, Any]:
    new, orig = Path(e["bddl"]).read_text(), Path(e["orig_bddl"]).read_text()
    out: dict[str, Any] = {}
    types_old, types_new = set(objects(orig).values()), set(objects(new).values())
    if types_old != types_new:
        out["object_types"] = {
            "removed": sorted(types_old - types_new),
            "added": sorted(types_new - types_old),
        }
    oi, ni = on_regions(orig), on_regions(new)
    moved = {k: (oi[k], ni[k]) for k in oi if k in ni and oi[k] != ni[k]}
    if moved:
        out["region_changes"] = moved
    if ranges(orig) != ranges(new):
        out["region_ranges"] = [(a, b) for a, b in zip(ranges(orig), ranges(new), strict=False) if a != b]
    if tables(orig) != tables(new):
        out["table"] = {"from": tables(orig), "to": tables(new)}
    g = catalog.bddl_goal
    if g(Path(e["bddl"])) != g(Path(e["orig_bddl"])):
        out["goal"] = {"original": g(Path(e["orig_bddl"])), "new": g(Path(e["bddl"]))}
    return out


def change_text(axis: str, ch: dict[str, Any]) -> str:
    if axis == "object" and "object_types" in ch:
        return f"{', '.join(ch['object_types']['removed'])} → {', '.join(ch['object_types']['added'])}"
    if axis == "position" and "region_changes" in ch:
        return ", ".join(
            f"{k}: {a.split('_', 2)[-1]}→{b.split('_', 2)[-1]}"
            for k, (a, b) in list(ch["region_changes"].items())[:3]
        )
    if axis == "position" and "region_ranges" in ch:
        a, b = ch["region_ranges"][0]
        return f"영역 ({a.strip()}) → ({b.strip()})"
    if axis == "environment" and "table" in ch:
        return f"{'/'.join(ch['table']['from'])} → {'/'.join(ch['table']['to'])}"
    return ""


def render(
    entry: dict[str, Any], seed: int, drift_steps: int = 50
) -> tuple[np.ndarray, float, str, list[np.ndarray]]:
    """First frame (after settling), worst object drift over drift_steps no-op steps, object xyz in BDDL order."""
    env = sim.ProEnv(entry, hw=360)
    obs = env.reset(0, seed)
    img = sim.upright(obs["agentview_image"])
    e = env.env.env
    bodies = {n: e.sim.model.body_name2id(o.root_body) for n, o in e.objects_dict.items()}
    p0 = {n: e.sim.data.body_xpos[b].copy() for n, b in bodies.items()}
    for _ in range(drift_steps):
        env.env.step(sim.DUMMY_ACTION)
    drift = {n: float(np.linalg.norm(e.sim.data.body_xpos[b] - p0[n])) for n, b in bodies.items()}
    worst = max(drift, key=lambda n: drift[n]) if drift else ""
    env.close()
    return img, drift.get(worst, 0.0), worst, list(p0.values())


def caption(img: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    h = 16
    lines = [w for t in lines for w in (textwrap.wrap(t, 56) or [""])]
    pad = np.full((8 + h * len(lines), img.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(pad, t, (6, 16 + i * h), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([img, pad], axis=0)


def main(argv: list[str] | None = None) -> int:
    import imageio.v2 as imageio

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/outputs/VLA-shortcut-exps/analysis/examples/libero_pro")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = catalog.pro_root()
    todo = [(s, t, a, lv) for s, t in EXAMPLES for a in catalog.AXES for lv in catalog.levels(s, a)]
    todo += [("libero_object", 0, "none", "original")] + [
        (s, t, "position", lv) for s, t, lv in POSITION_EXTREMES
    ]
    clean: dict[tuple[str, int], tuple[np.ndarray, list[np.ndarray]]] = {}
    manifest = []
    for suite, tid, axis, lv in todo:
        e = catalog.task_entry(root, suite, axis, lv, tid)
        seed = episode_seed(args.seed, 0, tid, 0)
        img, drift, worst, pos = render(e, seed)
        short = catalog.SUITE_SHORT[suite]
        eid = f"PRO_{short}_t{tid:02d}_{axis}_{lv}"
        png = out / f"{eid}.png"
        imageio.imwrite(png, img)
        if axis == "none":
            clean[suite, tid] = (img, pos)
        # placement vs the clean scene at the same init (objects in BDDL order; the object axis renames them)
        cpos = clean[suite, tid][1]
        moved_cm = (
            max((float(np.linalg.norm(a[:2] - b[:2])) for a, b in zip(pos, cpos, strict=False)), default=0.0)
            * 100
        )
        ch = describe_change(e)
        key = "position_xy" if axis == "position" and lv != "swap" else axis
        edit = word_edit(e["instruction_original"], e["instruction_given"]) or "변경 없음"
        why = CUE[key].format(change=change_text(axis, ch), axis=lv[:1], level=lv[1:], edit=edit)
        if axis not in ("none", "position") and moved_cm > 1:
            why += f" [init {e['init_source']}: 원본 대비 물체 배치도 최대 {moved_cm:.1f} cm 다름]"
        if drift > 0.01:
            why += f" (주의: settle 후 50 step 동안 {worst} 가 {drift * 100:.1f} cm 움직임)"
        pair = None
        if axis != "none":
            left = caption(clean[suite, tid][0], ["original (training sentence)", e["instruction_original"]])
            lines = [f"PRO {axis}/{lv} (init {e['init_source']})", e["instruction_given"]]
            if e["instruction_given"] != e["instruction_original"]:
                lines.append(f"vs training: {edit}")
            right = caption(img, lines)
            hgt = max(left.shape[0], right.shape[0])
            pair = out / f"{eid}_vs_orig.png"
            gap = np.full((hgt, 6, 3), 255, np.uint8)
            imageio.imwrite(pair, np.concatenate([pad_to(left, hgt), gap, pad_to(right, hgt)], axis=1))
        title = f"{SUITE_TITLE[suite]} t{tid} · {AXIS_TITLE[axis]}" + (
            f" {lv}" if lv not in (axis, "original") else ""
        )
        manifest.append(
            {
                "id": eid,
                "title": title,
                "caption": why,
                "path": str(pair or png),
                "category": "example",
                "benchmark": "libero_pro",
                "suite": suite,
                "task_id": tid,
                "init": 0,
                "seed": seed,
                "axis": axis,
                "level": lv,
                "dimension": catalog.DIMENSION[axis],
                "conditions": [f"pro:{axis}/{lv}"],
                "instructions": [e["instruction_given"]],
                "instruction_original": e["instruction_original"],
                "pro_instruction_original": e["pro_instruction_original"],
                "instruction_changed": e["instruction_given"] != e["instruction_original"],
                "instruction_edit_vs_training": edit
                if e["instruction_given"] != e["instruction_original"]
                else "",
                "init_source": e["init_source"],
                "placement_vs_clean_max_cm": round(moved_cm, 1),
                "bddl_changes": ch,
                "object_drift_50_steps_m": round(drift, 4),
                "object_drift_worst": worst,
                "type": "first_frame",
                "why": why,
                "png": str(png),
                "png_vs_orig": str(pair) if pair else None,
                "thumb": str(png),
                "bytes": png.stat().st_size,
                "pro_dir": e["dir"],
                "pro_task": e["task_name"],
                "bench_version": catalog.bench_version(root),
            }
        )
        print(
            f"[gallery] {eid}: {e['instruction_given']!r} drift {drift * 100:.1f} cm ({worst}) — {why}",
            flush=True,
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[gallery] {len(manifest)} entries → {out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
