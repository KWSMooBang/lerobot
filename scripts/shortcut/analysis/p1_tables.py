"""P1 pilot tables (T005): chart_sr.csv, qualitative.csv / qualitative_summary.csv, timing.csv, p2_budget.csv.

    uv run python scripts/shortcut/analysis/p1_tables.py /outputs/VLA-shortcut-exps/runs/sc-p1-* \
        --metrics /outputs/VLA-shortcut-exps/analysis/p1/metrics --out /outputs/VLA-shortcut-exps/analysis/p1

Run metrics.py into --metrics first (chart_sr.csv is read from its by_condition.csv).

Qualitative type per episode (first match wins), from episodes.jsonl (= step npz goal_orig/goal_new,
contact probe):
  both_goals     original and new goal both reached (C4–C6)
  orig_goal      original goal reached (C0/C1: success; C4/C5: shortcut)
  new_goal       new goal reached, original not
  grasp_orig_tgt C4: first grasp is the original object (variant_meta.orig_target), no goal
  grasp_new_tgt  C4: first grasp is the new object, no goal
  grasp_task_obj C0/C1/C5: first grasp is the object the original goal moves (C5 keeps the object), no goal
  no_grasp       nothing grasped, no goal
  other          grasped some other object, no goal
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import load_episodes  # noqa: E402

SHORTCUT_DIR = Path(__file__).resolve().parents[1]
TYPES = [
    "both_goals",
    "orig_goal",
    "new_goal",
    "grasp_orig_tgt",
    "grasp_new_tgt",
    "grasp_task_obj",
    "no_grasp",
    "other",
]
VIDEO_INITS = range(3)  # pi05lib runs kept videos for init 0–2 (--video first:3); pi05base kept all
SUITE_SHORT = {"libero_spatial": "spat", "libero_goal": "goal", "libero_object": "obj", "libero_10": "l10"}


def default_bddl_root() -> Path:
    import libero.libero

    return Path(libero.libero.__file__).parent / "bddl_files"


def task_objects(suites: set[str], bddl_root: Path) -> dict[tuple[str, int], str]:
    """(suite, task) → object the original goal acts on (first argument of its goal predicate)."""
    out = {}
    for suite in suites:
        spec = json.loads((SHORTCUT_DIR / "instructions" / f"{suite}.json").read_text())
        for t in spec["tasks"]:
            text = (bddl_root / t["bddl"]).read_text()
            m = re.search(r"\(:goal\s*\(And\s*\((\w+)\s+(\w+)", text)
            assert m, f"no goal in {t['bddl']}"
            out[(suite, t["task_id"])] = m.group(2)
    return out


def same_object(grasp: str | None, obj: str | None) -> bool:
    """grasp names are bodies (wooden_cabinet_1); goal arguments can be regions of them (..._middle_region)."""
    return bool(grasp and obj and (grasp == obj or obj.startswith(grasp + "_")))


def classify(r: dict[str, Any], task_obj: str) -> str:
    so, sn, g = bool(r["success_orig"]), bool(r.get("success_new")), r.get("first_grasp_object")
    meta = r.get("variant_meta") or {}
    if so and sn:
        return "both_goals"
    if so:
        return "orig_goal"
    if sn:
        return "new_goal"
    if not g:
        return "no_grasp"
    if meta.get("type") == "object":
        if same_object(g, meta.get("orig_target")):
            return "grasp_orig_tgt"
        if same_object(g, meta.get("new_target")):
            return "grasp_new_tgt"
        return "other"
    return "grasp_task_obj" if same_object(g, task_obj) else "other"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{path} ({len(rows)} rows)")


def chart_sr(metrics_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for r in csv.DictReader((metrics_dir / "by_condition.csv").open()):
        if r["task"] != "all":
            continue
        dual = r["condition"] in ("C4", "C5", "C6")
        metrics = [("ShortcutRate" if dual else "SR_orig", "sr_orig")]
        if dual:
            metrics.append(("SR_new", "sr_new"))
        for name, k in metrics:
            rows.append(
                {
                    "model": r["model"],
                    "suite": r["suite"],
                    "condition": r["condition"],
                    "metric": name,
                    "value": round(100 * float(r[k]), 2),
                    "ci_lo": round(100 * float(r[k + "_lo"]), 2),
                    "ci_hi": round(100 * float(r[k + "_hi"]), 2),
                    "n": int(r["n"]),
                }
            )
    return rows


def qualitative(eps: list[dict[str, Any]], objs: dict[tuple[str, int], str]) -> list[dict[str, Any]]:
    rows = []
    for r in eps:
        if r["init_idx"] not in VIDEO_INITS or not r.get("video"):
            continue
        meta = r.get("variant_meta") or {}
        rows.append(
            {
                "model": r["model"],
                "suite": r["suite"],
                "task_id": r["task_id"],
                "init": r["init_idx"],
                "condition": r["condition"],
                "variant": r["variant"],
                "type": classify(r, objs[(r["suite"], r["task_id"])]),
                "success_orig": int(bool(r["success_orig"])),
                "success_new": "" if r.get("success_new") is None else int(bool(r["success_new"])),
                "t_orig": r.get("t_orig") if r.get("t_orig") is not None else "",
                "t_new": r.get("t_new") if r.get("t_new") is not None else "",
                "first_grasp": r.get("first_grasp_object") or "",
                "first_grasp_t": r.get("first_grasp_t") if r.get("first_grasp_t") is not None else "",
                "first_contact": r.get("first_contact_object") or "",
                "task_obj": objs[(r["suite"], r["task_id"])],
                "orig_target": meta.get("orig_target", ""),
                "new_target": meta.get("new_target", ""),
                "n_steps": r["n_steps"],
                "instruction": r["instruction"],
                "video": str(Path(r["_root"]) / r["video"]),
            }
        )
    rows.sort(key=lambda x: (x["model"], x["suite"], x["condition"], x["task_id"], x["variant"], x["init"]))
    return rows


def qualitative_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, Counter] = defaultdict(Counter)
    for r in rows:
        groups[(r["model"], r["suite"], r["condition"])][r["type"]] += 1
    out = []
    for (m, s, c), cnt in sorted(groups.items()):
        n = sum(cnt.values())
        row: dict[str, Any] = {"model": m, "suite": s, "condition": c, "n": n}
        for t in TYPES:
            row[t] = round(100 * cnt[t] / n, 1)
        out.append(row)
    return out


def timing(runs: list[Path]) -> list[dict[str, Any]]:
    out = []
    for run in sorted(runs):
        eps = [json.loads(line) for line in (run / "episodes.jsonl").open()]
        t0 = datetime.fromisoformat((run / "started_at").read_text().strip())
        t1 = datetime.fromisoformat((run / "ended_at").read_text().strip())
        wall = (t1 - t0).total_seconds()
        ep_sum = sum(e["wall_time_s"] for e in eps)
        out.append(
            {
                "run_id": run.name,
                "model": eps[0]["model"],
                "suite": eps[0]["suite"],
                "condition": eps[0]["condition"],
                "n_ep": len(eps),
                "s_per_ep": round(ep_sum / len(eps), 2),
                "steps_per_ep": round(sum(e["n_steps"] for e in eps) / len(eps), 1),
                "ms_per_step": round(1000 * ep_sum / sum(e["n_steps"] for e in eps), 1),
                "run_wall_s": round(wall),
                "overhead_s": round(wall - ep_sum),
                "run_s_per_ep": round(wall / len(eps), 2),
            }
        )
    return out


def p2_budget(tim: list[dict[str, Any]], n_init: int = 50, c0_repeats: int = 3) -> list[dict[str, Any]]:
    """GPU-h for P2 on one GPU from measured ms/step. P1 has no object / libero_10 runs and no C2/C3/C6:
    object ≈ spatial (same max_steps 280, one scene per task), libero_10 = spatial ms/step × its own steps;
    C2 ≈ C1 (no usable language), C3 ≈ C0 (same task), C6 ≈ C4 (dual goal, runs to max_steps).
    pi05lib C0–C3 steps/ep scale with max_steps from the measured spatial/goal ratio."""
    max_steps = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
    proxy_suite = {
        "libero_spatial": "libero_spatial",
        "libero_object": "libero_spatial",
        "libero_goal": "libero_goal",
        "libero_10": "libero_spatial",
    }
    proxy_cond = {"C0": "C0", "C1": "C1", "C2": "C1", "C3": "C0", "C4": "C4", "C5": "C5", "C6": "C4"}
    meas = {(t["model"], t["suite"], t["condition"]): t for t in tim}
    overhead = sum(t["overhead_s"] for t in tim) / len(tim)
    rows = []
    for model in sorted({t["model"] for t in tim}):
        for suite in max_steps:
            spec = json.loads((SHORTCUT_DIR / "instructions" / f"{suite}.json").read_text())
            tasks = spec["tasks"]
            n_var = {
                "C0": len(tasks) * c0_repeats,
                "C1": len(tasks),
                "C2": len(tasks),
                "C3": sum(len(t["paraphrases"]) for t in tasks),
                "C4": sum(1 for t in tasks for s in t["swaps"] if s["type"] == "object" and s["feasible"]),
                "C5": sum(1 for t in tasks for s in t["swaps"] if s["type"] == "target" and s["feasible"]),
                "C6": sum(len(t["cross_task"]) for t in tasks),
            }
            for cond, nv in n_var.items():
                if nv == 0:
                    continue
                m = meas[(model, proxy_suite[suite], proxy_cond[cond])]
                steps = m["steps_per_ep"] * max_steps[suite] / max_steps[proxy_suite[suite]]
                n_ep = nv * n_init
                sec = n_ep * steps * m["ms_per_step"] / 1000
                n_runs = nv // len(tasks) if cond == "C0" else 1  # one run per (suite, cond, repeat)
                rows.append(
                    {
                        "model": model,
                        "suite": suite,
                        "condition": cond,
                        "n_ep": n_ep,
                        "steps_per_ep": round(steps, 1),
                        "ms_per_step": m["ms_per_step"],
                        "measured": suite == proxy_suite[suite] and cond == proxy_cond[cond],
                        "gpu_h": round((sec + n_runs * overhead) / 3600, 2),
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bddl-root", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = load_episodes(args.runs)
    bddl_root = Path(args.bddl_root) if args.bddl_root else default_bddl_root()
    objs = task_objects({r["suite"] for r in eps} | set(SUITE_SHORT), bddl_root)

    write_csv(chart_sr(Path(args.metrics)), Path(args.metrics) / "chart_sr.csv")
    q = qualitative(eps, objs)
    write_csv(q, out / "qualitative.csv")
    write_csv(qualitative_summary(q), out / "qualitative_summary.csv")
    tim = timing(sorted({Path(r["_root"]) for r in eps}))
    write_csv(tim, out / "timing.csv")
    bud = p2_budget(tim)
    write_csv(bud, out / "p2_budget.csv")
    for model in sorted({b["model"] for b in bud}):
        sel = [b for b in bud if b["model"] == model]
        print(f"P2 {model}: {sum(b['n_ep'] for b in sel)} ep, {sum(b['gpu_h'] for b in sel):.1f} GPU-h")


if __name__ == "__main__":
    main()
