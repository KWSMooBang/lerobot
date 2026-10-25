"""Aggregate episodes.jsonl from shortcut runs into model × suite × condition (× task) tables.

    uv run python scripts/shortcut/metrics.py /outputs/VLA-shortcut-exps/runs/sc-* --out results/shortcut

Each argument is a run dir (containing episodes.jsonl), an episodes.jsonl file, or a glob.

Columns (design doc §5, §7):
  sr_orig        original goal reached. C0 → SR(C0); C1–C3 → SR under that instruction;
                 C4–C6 → Shortcut Rate (original goal reached although the instruction asked otherwise)
  sr_new         new goal reached (C4–C6 only) = SR_new
  *_lo/*_hi      Wilson 95% CI
  chance         1/k, k = tasks in the suite sharing this task's scene (SR_new ceiling of a language-blind policy)
  d_vs_c0        sr_orig − SR(C0) on the same (task, init) set; for C1 this is −Language Sensitivity
  lang_sens      SR(C0) − SR(C1) (C1 rows)
  mcnemar_p      exact McNemar, success_orig of this condition vs C0 paired on (task, init) (repeat 0)
  grasp_match_c0 first grasped object equals C0's at the same (task, init)
  grasp_orig_tgt / grasp_new_tgt   C4: first grasp is the original / new object
  dtw            mean per-step DTW distance (m) of eef xyz vs C0 at the same (task, init), both truncated
                 to the shorter episode; dtw_c0_rep is the C0-vs-C0 (repeat 0 vs repeat k) baseline
  sec_per_ep     mean wall time per episode
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

DTW_MAX_POINTS = 120


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(pairs: list[tuple[bool, bool]]) -> float:
    b = sum(1 for a, c in pairs if a and not c)
    c = sum(1 for a, cc in pairs if not a and cc)
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-step DTW distance (Euclidean, path-length normalised)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    sa = max(1, math.ceil(len(a) / DTW_MAX_POINTS))
    sb = max(1, math.ceil(len(b) / DTW_MAX_POINTS))
    a, b = a[::sa], b[::sb]
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1), dtype=np.int64)  # warping-path length to each cell
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, k = d[i - 1, j - 1], steps[i - 1, j - 1]
            if d[i - 1, j] < best:
                best, k = d[i - 1, j], steps[i - 1, j]
            if d[i, j - 1] < best:
                best, k = d[i, j - 1], steps[i, j - 1]
            d[i, j] = best + cost[i - 1, j - 1]
            steps[i, j] = k + 1
    return float(d[n, m] / steps[n, m])


def load_episodes(inputs: Iterable[str]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for spec in inputs:
        for p in sorted(glob.glob(spec)) or [spec]:
            path = Path(p)
            files.append(path / "episodes.jsonl" if path.is_dir() else path)
    eps = []
    for f in files:
        if not f.exists():
            print(f"[skip] {f}")
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_root"] = str(f.parent)
                eps.append(r)
    return eps


def eef(r: dict[str, Any], cache: dict[str, np.ndarray]) -> np.ndarray:
    path = str(Path(r["_root"]) / r["steps_file"])
    if path not in cache:
        cache[path] = np.load(path)["eef_pos"]
    return cache[path]


def fmt(x: Any, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if pct:
        return f"{100 * x:.1f}"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def summarize(eps: list[dict[str, Any]], by_task: bool) -> list[dict[str, Any]]:
    # C0 reference per (model, suite, task, init) at repeat 0, and C0 repeats for the DTW baseline
    c0: dict[tuple, dict] = {}
    c0_rep: dict[tuple, list[dict]] = defaultdict(list)
    for r in eps:
        if r["condition"] != "C0":
            continue
        key = (r["model"], r["suite"], r["task_id"], r["init_idx"])
        if r.get("repeat_idx", 0) == 0:
            c0[key] = r
        else:
            c0_rep[key].append(r)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in eps:
        gk = (r["model"], r["suite"], r["task_id"] if by_task else "all", r["condition"])
        groups[gk].append(r)

    cache: dict[str, np.ndarray] = {}
    rows = []
    for (model, suite, task, cond), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        main = [r for r in rs if r.get("repeat_idx", 0) == 0] or rs
        n = len(main)
        k_orig = sum(bool(r["success_orig"]) for r in main)
        row: dict[str, Any] = {
            "model": model,
            "suite": suite,
            "task": task,
            "condition": cond,
            "n_variants": len({r["variant"] for r in main}),
            "n": n,
            "sr_orig": k_orig / n,
        }
        row["sr_orig_lo"], row["sr_orig_hi"] = wilson(k_orig, n)
        dual = [r for r in main if r.get("success_new") is not None]
        if dual:
            k_new = sum(bool(r["success_new"]) for r in dual)
            row["sr_new"] = k_new / len(dual)
            row["sr_new_lo"], row["sr_new_hi"] = wilson(k_new, len(dual))
            row["chance"] = float(np.mean([1.0 / r["scene_k"] for r in dual]))
            row["both_goals"] = float(
                np.mean([bool(r["success_orig"]) and bool(r["success_new"]) for r in dual])
            )

        paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for r in main:
            ref = c0.get((model, suite, r["task_id"], r["init_idx"]))
            if ref is not None:
                paired.append((r, ref))
        if paired and cond != "C0":
            ref_sr = float(np.mean([bool(ref["success_orig"]) for _, ref in paired]))
            this_sr = float(np.mean([bool(r["success_orig"]) for r, _ in paired]))
            row["sr_c0_paired"] = ref_sr
            row["d_vs_c0"] = this_sr - ref_sr
            if cond == "C1":
                row["lang_sens"] = ref_sr - this_sr
            row["mcnemar_p"] = mcnemar_exact(
                [(bool(r["success_orig"]), bool(ref["success_orig"])) for r, ref in paired]
            )
            g = [(r, ref) for r, ref in paired if ref.get("first_grasp_object")]
            if g:
                row["grasp_match_c0"] = float(
                    np.mean([r.get("first_grasp_object") == ref["first_grasp_object"] for r, ref in g])
                )
            d = []
            for r, ref in paired:
                a, b = eef(r, cache), eef(ref, cache)
                n_min = min(len(a), len(b))
                d.append(dtw(a[:n_min], b[:n_min]))
            row["dtw"] = float(np.nanmean(d)) if d else float("nan")
        if cond == "C0":
            d = []
            for r in main:
                for rep in c0_rep.get((model, suite, r["task_id"], r["init_idx"]), []):
                    a, b = eef(r, cache), eef(rep, cache)
                    n_min = min(len(a), len(b))
                    d.append(dtw(a[:n_min], b[:n_min]))
            row["dtw_c0_rep"] = float(np.nanmean(d)) if d else float("nan")
            row["n_repeats"] = 1 + max((len(v) for v in c0_rep.values()), default=0)
        swaps = [r for r in main if (r.get("variant_meta") or {}).get("orig_target")]
        if swaps and cond == "C4":
            row["grasp_orig_tgt"] = float(
                np.mean([r.get("first_grasp_object") == r["variant_meta"]["orig_target"] for r in swaps])
            )
            row["grasp_new_tgt"] = float(
                np.mean([r.get("first_grasp_object") == r["variant_meta"]["new_target"] for r in swaps])
            )
        row["grasp_none"] = float(np.mean([not r.get("first_grasp_object") for r in main]))
        row["sec_per_ep"] = float(np.mean([r["wall_time_s"] for r in main]))
        row["steps_per_ep"] = float(np.mean([r["n_steps"] for r in main]))
        rows.append(row)
    return rows


COLUMNS = [
    "model", "suite", "task", "condition", "n_variants", "n",
    "sr_orig", "sr_orig_lo", "sr_orig_hi", "sr_new", "sr_new_lo", "sr_new_hi", "chance", "both_goals",
    "sr_c0_paired", "d_vs_c0", "lang_sens", "mcnemar_p",
    "grasp_match_c0", "grasp_orig_tgt", "grasp_new_tgt", "grasp_none",
    "dtw", "dtw_c0_rep", "n_repeats", "sec_per_ep", "steps_per_ep",
]  # fmt: skip


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def ci(r: dict[str, Any], k: str) -> str:
    return f"{fmt(r.get(k), True)} [{fmt(r.get(k + '_lo'), True)}, {fmt(r.get(k + '_hi'), True)}]"


def markdown(rows: list[dict[str, Any]], title: str) -> str:
    head = (
        "| model | suite | task | cond | n | SR orig % [95% CI] | SR_new % [95% CI] | chance % | Δ vs C0 %p "
        "| LS %p | McNemar p | grasp=C0 % | grasp orig/new tgt % | DTW m (C0 rep) | s/ep |"
    )
    lines = [
        f"## {title}",
        "",
        "SR orig = SR(C0) on C0 rows, Shortcut Rate on C4–C6 rows.",
        "",
        head,
        "|" + "---|" * 15,
    ]
    for r in rows:
        new = ci(r, "sr_new") if "sr_new" in r else "–"
        tgt = (
            f"{fmt(r.get('grasp_orig_tgt'), True)}/{fmt(r.get('grasp_new_tgt'), True)}"
            if "grasp_orig_tgt" in r
            else "–"
        )
        dtw_s = fmt(r.get("dtw")) if r["condition"] != "C0" else f"({fmt(r.get('dtw_c0_rep'))})"
        d = r.get("d_vs_c0")
        lines.append(
            f"| {r['model']} | {r['suite']} | {r['task']} | {r['condition']} | {r['n']} | {ci(r, 'sr_orig')} | {new} "
            f"| {fmt(r.get('chance'), True)} | {f'{d * 100:+.1f}' if d is not None else '–'} "
            f"| {f'{r["lang_sens"] * 100:+.1f}' if 'lang_sens' in r else '–'} | {fmt(r.get('mcnemar_p'))} "
            f"| {fmt(r.get('grasp_match_c0'), True)} | {tgt} | {dtw_s} | {fmt(r.get('sec_per_ep'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    eps = load_episodes(args.inputs)
    if not eps:
        raise SystemExit("no episodes found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overall = summarize(eps, by_task=False)
    per_task = summarize(eps, by_task=True)
    write_csv(overall, out / "by_condition.csv")
    write_csv(per_task, out / "by_task.csv")
    md = markdown(overall, "model × suite × condition") + "\n" + markdown(per_task, "per task")
    (out / "summary.md").write_text(md)
    print(md)
    print(f"{len(eps)} episodes → {out}/by_condition.csv, by_task.csv, summary.md")


if __name__ == "__main__":
    main()
