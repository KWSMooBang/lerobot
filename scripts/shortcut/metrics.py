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
  n_pairs        (task, init) pairs with a repeat-0 C0 episode of the same seed
  mcnemar_p      exact McNemar, success_orig vs C0; variants aggregated per (task, init) by majority
  mcnemar_p_any  same, variants aggregated by any-success (per-variant tests: by_variant.csv)
  grasp_match_c0 first grasped object equals C0's at the same (task, init)
  grasp_orig_tgt / grasp_new_tgt   C4: first grasp is the original / new object
  dtw            mean per-step DTW distance (m) of eef xyz vs C0 at the same (task, init), both truncated
                 to the shorter episode; dtw_c0_rep is the C0-vs-C0 (repeat 0 vs repeat k) baseline
Also writes by_variant.csv (per-variant McNemar) and repeats.csv (C0 repeat-k vs repeat-0 agreement).
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


def pair_key(r: dict[str, Any], repeat_idx: int | None = None) -> tuple:
    """Episode identity for pairing: same checkpoint, suite, task, init, repeat and seed.

    The seed already encodes (base seed, repeat, task, init) and is identical across conditions at a
    given repeat, so a C4 episode at repeat 0 pairs with the C0 episode of the same seed at repeat 0.
    """
    rep = r.get("repeat_idx", 0) if repeat_idx is None else repeat_idx
    return (r["model"], r["ckpt"], r["suite"], r["task_id"], r["init_idx"], rep, r["seed"])


def index_c0(eps: list[dict[str, Any]]) -> tuple[dict[tuple, dict], dict[tuple, dict[int, dict]]]:
    """C0 reference (repeat_idx == 0, keyed by pair_key) and all C0 repeats keyed by (model, ckpt, suite,
    task, init). Duplicate episodes (e.g. two copies of a run passed in) are an error, never overwritten."""
    ref: dict[tuple, dict] = {}
    reps: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in eps:
        if r["condition"] != "C0":
            continue
        k = pair_key(r)
        rep = r.get("repeat_idx", 0)
        loc = (r["model"], r["ckpt"], r["suite"], r["task_id"], r["init_idx"])
        if rep in reps[loc]:
            raise ValueError(
                f"duplicate C0 episode {loc} repeat {rep}: {r['_root']} and {reps[loc][rep]['_root']}"
            )
        reps[loc][rep] = r
        if rep == 0:
            ref[k] = r
    return ref, reps


def check_unique(eps: list[dict[str, Any]]) -> None:
    seen: dict[tuple, str] = {}
    for r in eps:
        k = (*pair_key(r), r["condition"], r["variant"])
        if k in seen:
            raise ValueError(f"duplicate episode {k}: {r['_root']} and {seen[k]}")
        seen[k] = r["_root"]


def aggregate_variants(main: list[dict[str, Any]], c0_ref: dict[tuple, dict]) -> list[tuple[float, bool]]:
    """One pair per (task, init): (fraction of this condition's variants reaching the original goal,
    C0 success). Variants of C3–C6 share one C0 episode, so they must not count as separate pairs."""
    by_loc: dict[tuple, list[bool]] = defaultdict(list)
    for r in main:
        if pair_key(r, 0) in c0_ref:
            by_loc[pair_key(r, 0)].append(bool(r["success_orig"]))
    return [(float(np.mean(v)), bool(c0_ref[k]["success_orig"])) for k, v in by_loc.items()]


def summarize(eps: list[dict[str, Any]], by_task: bool) -> list[dict[str, Any]]:
    c0_ref, c0_reps = index_c0(eps)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in eps:
        gk = (r["model"], r["suite"], r["task_id"] if by_task else "all", r["condition"])
        groups[gk].append(r)

    cache: dict[str, np.ndarray] = {}
    rows = []
    for (model, suite, task, cond), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        main = [r for r in rs if r.get("repeat_idx", 0) == 0]  # C0 repeats > 0 go to the repeat table
        if not main:
            continue
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

        paired = [(r, c0_ref[pair_key(r, 0)]) for r in main if pair_key(r, 0) in c0_ref]
        if paired and cond != "C0":
            agg = aggregate_variants(main, c0_ref)
            ref_sr = float(np.mean([c for _, c in agg]))
            this_sr = float(np.mean([f for f, _ in agg]))
            row["n_pairs"] = len(agg)
            row["sr_c0_paired"] = ref_sr
            row["d_vs_c0"] = this_sr - ref_sr
            if cond == "C1":
                row["lang_sens"] = ref_sr - this_sr
            row["mcnemar_p"] = mcnemar_exact([(f >= 0.5, c) for f, c in agg])  # majority (ties → success)
            row["mcnemar_p_any"] = mcnemar_exact([(f > 0, c) for f, c in agg])
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
                loc = (r["model"], r["ckpt"], r["suite"], r["task_id"], r["init_idx"])
                for rep, other in c0_reps[loc].items():
                    if rep > 0:
                        a, b = eef(r, cache), eef(other, cache)
                        n_min = min(len(a), len(b))
                        d.append(dtw(a[:n_min], b[:n_min]))
            row["dtw_c0_rep"] = float(np.nanmean(d)) if d else float("nan")
            row["n_repeats"] = len({r.get("repeat_idx", 0) for r in rs})
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


def summarize_variants(eps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per variant (C3 paraphrases, C4/C5 swaps, C6 cross tasks): each variant pairs once with C0 per
    (task, init), so McNemar is valid per variant."""
    c0_ref, _ = index_c0(eps)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in eps:
        if r["condition"] != "C0" and r.get("repeat_idx", 0) == 0:
            groups[(r["model"], r["suite"], r["task_id"], r["condition"], r["variant"])].append(r)
    rows = []
    for (model, suite, task, cond, var), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        pairs = [
            (bool(r["success_orig"]), bool(c0_ref[pair_key(r, 0)]["success_orig"]))
            for r in rs
            if pair_key(r, 0) in c0_ref
        ]
        row: dict[str, Any] = {
            "model": model,
            "suite": suite,
            "task": task,
            "condition": cond,
            "variant": var,
            "instruction": rs[0]["instruction"],
            "n": len(rs),
            "sr_orig": float(np.mean([bool(r["success_orig"]) for r in rs])),
            "n_pairs": len(pairs),
        }
        if pairs:
            row["sr_c0_paired"] = float(np.mean([c for _, c in pairs]))
            row["mcnemar_p"] = mcnemar_exact(pairs)
        dual = [r for r in rs if r.get("success_new") is not None]
        if dual:
            row["sr_new"] = float(np.mean([bool(r["success_new"]) for r in dual]))
        rows.append(row)
    return rows


def summarize_repeats(eps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """C0 repeat k vs repeat 0 at the same (task, init): agreement of success and first grasp, McNemar, DTW."""
    _, c0_reps = index_c0(eps)
    groups: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    for (model, _ckpt, suite, _task, _init), reps in c0_reps.items():
        if 0 not in reps:
            continue
        for k, r in reps.items():
            if k > 0:
                groups[(model, suite, k)].append((reps[0], r))
    cache: dict[str, np.ndarray] = {}
    rows = []
    for (model, suite, k), pairs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        s0 = [bool(a["success_orig"]) for a, _ in pairs]
        sk = [bool(b["success_orig"]) for _, b in pairs]
        d = []
        for a, b in pairs:
            ea, eb = eef(a, cache), eef(b, cache)
            n_min = min(len(ea), len(eb))
            d.append(dtw(ea[:n_min], eb[:n_min]))
        rows.append(
            {
                "model": model,
                "suite": suite,
                "repeat": k,
                "n_pairs": len(pairs),
                "sr_r0": float(np.mean(s0)),
                "sr_rk": float(np.mean(sk)),
                "success_agree": float(np.mean([a == b for a, b in zip(s0, sk, strict=True)])),
                "grasp_agree": float(
                    np.mean([a.get("first_grasp_object") == b.get("first_grasp_object") for a, b in pairs])
                ),
                "mcnemar_p": mcnemar_exact(list(zip(sk, s0, strict=True))),
                "dtw": float(np.nanmean(d)) if d else float("nan"),
            }
        )
    return rows


COLUMNS = [
    "model", "suite", "task", "condition", "n_variants", "n",
    "sr_orig", "sr_orig_lo", "sr_orig_hi", "sr_new", "sr_new_lo", "sr_new_hi", "chance", "both_goals",
    "n_pairs", "sr_c0_paired", "d_vs_c0", "lang_sens", "mcnemar_p", "mcnemar_p_any",
    "grasp_match_c0", "grasp_orig_tgt", "grasp_new_tgt", "grasp_none",
    "dtw", "dtw_c0_rep", "n_repeats", "sec_per_ep", "steps_per_ep",
]  # fmt: skip


VARIANT_COLUMNS = [
    "model", "suite", "task", "condition", "variant", "instruction", "n", "sr_orig", "sr_new",
    "n_pairs", "sr_c0_paired", "mcnemar_p",
]  # fmt: skip
REPEAT_COLUMNS = [
    "model", "suite", "repeat", "n_pairs", "sr_r0", "sr_rk", "success_agree", "grasp_agree", "mcnemar_p", "dtw",
]  # fmt: skip

METHOD_NOTE = """Pairing: episodes pair on (model, checkpoint, suite, task, init, repeat, seed); the C0 reference is
repeat_idx = 0 (same seed as every condition at repeat 0). C0 repeats > 0 appear only in the repeat table.
McNemar per condition: C3–C6 have several variants per (task, init) that share one C0 episode, so variants
are first aggregated within each (task, init) into one pair. `McNemar p` uses majority (success if ≥ half
of the variants reach the original goal; ties count as success), `p (any)` uses any-variant success.
Per-variant McNemar (one pair per (task, init) per variant) is in the variant table. Δ vs C0 uses the
mean over variants per (task, init). DTW and grasp=C0 are descriptive means over all variant episodes."""


def write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str] = COLUMNS) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def ci(r: dict[str, Any], k: str) -> str:
    return f"{fmt(r.get(k), True)} [{fmt(r.get(k + '_lo'), True)}, {fmt(r.get(k + '_hi'), True)}]"


def markdown(rows: list[dict[str, Any]], title: str) -> str:
    head = (
        "| model | suite | task | cond | n | SR orig % [95% CI] | SR_new % [95% CI] | chance % | Δ vs C0 %p "
        "| LS %p | pairs | McNemar p (maj / any) | grasp=C0 % | grasp orig/new tgt % | DTW m (C0 rep) | s/ep |"
    )
    lines = [
        f"## {title}",
        "",
        "SR orig = SR(C0) on C0 rows, Shortcut Rate on C4–C6 rows.",
        "",
        head,
        "|" + "---|" * 16,
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
            f"| {f'{r["lang_sens"] * 100:+.1f}' if 'lang_sens' in r else '–'} | {fmt(r.get('n_pairs'))} "
            f"| {fmt(r.get('mcnemar_p'))} / {fmt(r.get('mcnemar_p_any'))} "
            f"| {fmt(r.get('grasp_match_c0'), True)} | {tgt} | {dtw_s} | {fmt(r.get('sec_per_ep'))} |"
        )
    return "\n".join(lines) + "\n"


def markdown_variants(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## per variant (McNemar vs C0, one pair per (task, init))",
        "",
        "| model | suite | task | cond | variant | instruction | n | SR orig % | SR_new % | SR C0 % | McNemar p |",
        "|" + "---|" * 11,
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['suite']} | {r['task']} | {r['condition']} | {r['variant']} | {r['instruction'] or '""'} "
            f"| {r['n']} | {fmt(r['sr_orig'], True)} | {fmt(r.get('sr_new'), True)} | {fmt(r.get('sr_c0_paired'), True)} "
            f"| {fmt(r.get('mcnemar_p'))} |"
        )
    return "\n".join(lines) + "\n"


def markdown_repeats(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## C0 repeat agreement (repeat k vs repeat 0, same (task, init))",
        "",
        "| model | suite | repeat | pairs | SR r0 % | SR rk % | success agree % | grasp agree % | McNemar p | DTW m |",
        "|" + "---|" * 10,
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['suite']} | {r['repeat']} | {r['n_pairs']} | {fmt(r['sr_r0'], True)} "
            f"| {fmt(r['sr_rk'], True)} | {fmt(r['success_agree'], True)} | {fmt(r['grasp_agree'], True)} "
            f"| {fmt(r['mcnemar_p'])} | {fmt(r['dtw'])} |"
        )
    if not rows:
        lines.append("| (no C0 repeats) |" + " |" * 9)
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    eps = load_episodes(args.inputs)
    if not eps:
        raise SystemExit("no episodes found")
    check_unique(eps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overall = summarize(eps, by_task=False)
    per_task = summarize(eps, by_task=True)
    write_csv(overall, out / "by_condition.csv")
    write_csv(per_task, out / "by_task.csv")
    variants = summarize_variants(eps)
    repeats = summarize_repeats(eps)
    write_csv(variants, out / "by_variant.csv", VARIANT_COLUMNS)
    write_csv(repeats, out / "repeats.csv", REPEAT_COLUMNS)
    md = "\n".join(
        [
            "# Shortcut metrics\n",
            METHOD_NOTE + "\n",
            markdown(overall, "model × suite × condition"),
            markdown_variants(variants),
            markdown_repeats(repeats),
            markdown(per_task, "per task"),
        ]
    )
    (out / "summary.md").write_text(md)
    print(md)
    print(f"{len(eps)} episodes → {out}/{{by_condition,by_task,by_variant,repeats}}.csv, summary.md")


if __name__ == "__main__":
    main()
