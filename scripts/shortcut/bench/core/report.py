"""sc-report: compare models over benchmark runs → tables, cross-model videos, manifest.json.

    scripts/shortcut/bench/sc-report --runs bench-pi05lib-quick,bench-tb-smolvla-a-quick [--out DIR]

--runs items: a run dir, a run id under <out-root>/runs/, or a ckpt id (→ every runs/bench-<id>-*). sc-bench
run dirs are read through jobs/*/episodes.jsonl; plain T001 run dirs (sc-p1-*) work too (standard fields
are derived from condition / success_orig / success_new).

Outputs (default <out-root>/reports/<model>_vs_<model>/):
  table.csv     long format: model, benchmark, suite (or ALL), dimension, level, detail, n, success (+ Wilson
                95% CI), shortcut_rate = original goal reached under a changed goal (C4–C6; + CI), sr_new (+ CI),
                d_vs_clean = success − clean (dimension none) success of the same (model, benchmark, suite)
  report.md     model-by-model table per benchmark, warnings, run list
  videos/       one clip per model side by side at the same (benchmark, suite, task, init, perturbation);
                cases where models disagree first. mp4 + GIF ≤ 3 MB + last-frame png in thumbs/
  manifest.json document manifest (T005 format: id, category, model, suite, task_id, init, conditions,
                instructions, outcomes, type, why, mp4, gif, gif_bytes, thumb)
Mixing rule (T004): a row whose models differ in exec_mode / GPU model / bench_version gets a warning (and the
`mixed` column); numbers still come from all episodes.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SHORTCUT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(SHORTCUT),) if p not in sys.path]

import imageio.v2 as imageio  # noqa: E402
from make_compare_video import hstack_clips, make_gif, read_clip  # noqa: E402
from metrics import wilson  # noqa: E402

from bench.core import registry  # noqa: E402
from bench.core.records import OURS_PERTURBATION, read_jsonl  # noqa: E402

SUITE_SHORT = {"libero_spatial": "spat", "libero_object": "obj", "libero_goal": "goal", "libero_10": "l10"}
COLUMNS = [
    "model", "benchmark", "suite", "dimension", "level", "detail", "n", "success", "success_lo", "success_hi",
    "shortcut_rate", "shortcut_lo", "shortcut_hi", "sr_new", "sr_new_lo", "sr_new_hi", "d_vs_clean",
    "exec", "bench_version", "mixed",
]  # fmt: skip


# ---------------------------------------------------------------- loading


def resolve_runs(items: list[str], root: Path) -> list[Path]:
    out: list[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out.append(p)
        elif (root / "runs" / it).is_dir():
            out.append(root / "runs" / it)
        else:
            hits = sorted(
                Path(x) for x in glob.glob(str(root / "runs" / f"bench-{it}-*")) if Path(x).is_dir()
            )
            if not hits:
                raise SystemExit(
                    f"--runs {it!r}: not a dir, not {root}/runs/{it}, no {root}/runs/bench-{it}-*"
                )
            out += hits
    uniq: dict[Path, Path] = {}
    for p in out:  # the same run given as a path and as a ckpt id / run id counts once
        uniq.setdefault(p.resolve(), p)
    return list(uniq.values())


def job_dirs(run: Path) -> list[Path]:
    if (run / "jobs").is_dir():
        return sorted(d for d in (run / "jobs").iterdir() if (d / "episodes.jsonl").exists())
    return [run] if (run / "episodes.jsonl").exists() else []


def normalize(r: dict[str, Any], meta: dict[str, Any], jdir: Path, run: Path) -> dict[str, Any]:
    """Fill the standard fields for T001 records (pilot runs) that predate them."""
    if "benchmark" not in r:
        dim, level = OURS_PERTURBATION[r["condition"]]
        dual = r.get("success_new") is not None
        sha = (meta.get("identity") or {}).get("instr_set_sha256", "")[:8]
        r.update(
            benchmark="ours",
            bench_version=f"ours-t001/instr-{sha}",
            variant_id=r["variant"],
            perturbation={"dimension": dim, "level": level, "detail": r["condition"]},
            instruction_original=r.get("original_instruction"),
            instruction_given=r.get("instruction"),
            success=bool(r["success_new"] if dual else r["success_orig"]),
        )
    ident = meta.get("identity") or {}
    r["_exec"] = f"{ident.get('exec_mode', 'single')}@{ident.get('gpu', meta.get('gpu'))}"
    r["_root"] = str(jdir)
    r["_run"] = run.name
    return r


OUTCOME_FIELDS = ("success", "success_orig", "success_new", "t_orig", "t_new")


def episode_key(r: dict[str, Any]) -> tuple:
    """One episode, wherever it was written: overlapping batteries (quick ⊂ full) or the same run given twice."""
    return (
        r["ckpt"], r["benchmark"], str(r["bench_version"]), r["suite"], r["task_id"], str(r["perturbation"]["detail"]),
        r["variant_id"], r["init_idx"], r.get("repeat_idx", 0), r["seed"], r["_exec"],
    )  # fmt: skip


def load(runs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Episodes of all runs, one per episode_key (the first in --runs order is used).
    Returns (episodes, duplicates [key, used run, dropped run, conflict], warnings)."""
    eps: dict[tuple, dict[str, Any]] = {}
    dups: list[dict[str, Any]] = []
    for run in runs:
        for jd in job_dirs(run):
            meta = json.loads((jd / "run_meta.json").read_text()) if (jd / "run_meta.json").exists() else {}
            for r in read_jsonl(jd / "episodes.jsonl"):
                r = normalize(r, meta, jd, run)
                k = episode_key(r)
                if k not in eps:
                    eps[k] = r
                    continue
                kept = eps[k]
                conflict = any(kept.get(f) != r.get(f) for f in OUTCOME_FIELDS)
                dups.append(
                    {
                        "key": list(k),
                        "used": f"{kept['_run']}/{Path(kept['_root']).name}",
                        "dropped": f"{r['_run']}/{jd.name}",
                        "conflict": conflict,
                    }
                )
                kept.setdefault("_dropped_from", []).append(r["_run"])
    warns = []
    if dups:
        n_conf = sum(d["conflict"] for d in dups)
        warns.append(
            f"{len(dups)} duplicate episodes across inputs counted once (first run in --runs order used; "
            f"duplicates.csv); {n_conf} of them have different outcomes"
        )
        warns += [
            f"conflicting duplicate {d['key']}: used {d['used']}, dropped {d['dropped']}"
            for d in dups
            if d["conflict"]
        ][:20]
    return list(eps.values()), dups, warns


# ---------------------------------------------------------------- tables


def rate(rs: list[dict[str, Any]], key: str) -> tuple[float | None, float | None, float | None]:
    vals = [bool(r[key]) for r in rs if r.get(key) is not None]
    if not vals:
        return None, None, None
    lo, hi = wilson(sum(vals), len(vals))
    return sum(vals) / len(vals), lo, hi


def pkey(r: dict[str, Any]) -> tuple[str, str, str]:
    p = r["perturbation"]
    return p["dimension"], str(p["level"]), str(p["detail"])


def table(eps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in eps:
        for suite in (r["suite"], "ALL"):
            groups[(r["model"], r["benchmark"], suite, *pkey(r))].append(r)
    clean = {(m, b, s): rate(rs, "success")[0] for (m, b, s, dim, *_), rs in groups.items() if dim == "none"}
    rows = []
    for (model, bench, suite, dim, level, detail), rs in groups.items():
        s, slo, shi = rate(rs, "success")
        dual = [r for r in rs if r.get("success_new") is not None]
        sc, sclo, schi = rate(dual, "success_orig") if dual else (None, None, None)
        sn, snlo, snhi = rate(dual, "success_new") if dual else (None, None, None)
        base = clean.get((model, bench, suite))
        rows.append(
            {
                "model": model, "benchmark": bench, "suite": suite, "dimension": dim, "level": level,
                "detail": detail, "n": len(rs), "success": s, "success_lo": slo, "success_hi": shi,
                "shortcut_rate": sc, "shortcut_lo": sclo, "shortcut_hi": schi,
                "sr_new": sn, "sr_new_lo": snlo, "sr_new_hi": snhi,
                "d_vs_clean": None if base is None or s is None or dim == "none" else s - base,
                "exec": "|".join(sorted({r["_exec"] for r in rs})),
                "bench_version": "|".join(sorted({str(r["bench_version"]) for r in rs})),
            }
        )  # fmt: skip
    # mixed: the same (benchmark, suite, perturbation) differs across models in exec / bench_version
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["benchmark"], row["suite"], row["dimension"], row["level"], row["detail"])].append(row)
    for cell in by_cell.values():
        execs = {r["exec"] for r in cell}
        bvs = {r["bench_version"] for r in cell}
        mixed = []
        if len(execs) > 1 or any("|" in e for e in execs):
            mixed.append("exec")
        if len(bvs) > 1 or any("|" in b for b in bvs):
            mixed.append("bench_version")
        for r in cell:
            r["mixed"] = ",".join(mixed)
    order = {"none": 0, "language": 1}
    return sorted(
        rows,
        key=lambda r: (
            r["benchmark"],
            r["suite"] == "ALL",
            r["suite"],
            order.get(r["dimension"], 2),
            r["detail"],
            r["level"],
            r["model"],
        ),
    )


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}"


def cell(r: dict[str, Any] | None) -> str:
    if r is None:
        return ""
    s = f"{pct(r['success'])} [{pct(r['success_lo'])}–{pct(r['success_hi'])}]"
    if r["shortcut_rate"] is not None:
        s += f"; SC {pct(r['shortcut_rate'])} [{pct(r['shortcut_lo'])}–{pct(r['shortcut_hi'])}]"
    if r["d_vs_clean"] is not None:
        s += f"; Δ {100 * r['d_vs_clean']:+.0f}"
    return s + f" (n={r['n']})"


def warnings(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    seen = set()
    for r in rows:
        if not r["mixed"] or r["suite"] == "ALL":
            continue
        k = (r["benchmark"], r["suite"], r["detail"], r["mixed"])
        if k in seen:
            continue
        seen.add(k)
        cells = [
            x
            for x in rows
            if (x["benchmark"], x["suite"], x["detail"], x["level"])
            == (r["benchmark"], r["suite"], r["detail"], r["level"])
        ]
        desc = "; ".join(f"{x['model']}: {x['exec']} / {x['bench_version']}" for x in cells)
        out.append(f"{r['benchmark']} {r['suite']} {r['detail']}: models differ in {r['mixed']} ({desc})")
    return out


def markdown(
    rows: list[dict[str, Any]], models: list[str], warns: list[str], runs: list[Path], eps: list[dict]
) -> str:
    lines = ["# Shortcut testbed report", ""]
    lines += [
        "Cells: success % [Wilson 95% CI]; SC = Shortcut Rate (original goal reached under a changed goal, "
        "C4–C6; then `success` = SR_new); Δ = success − clean success of the same model/benchmark/suite (pp).",
        "",
    ]
    if warns:
        lines += ["## Warnings (T004 mixing rule)", ""] + [f"- {w}" for w in warns] + [""]
    for bench in sorted({r["benchmark"] for r in rows}):
        lines += [
            f"## {bench}",
            "",
            "| suite | perturbation | " + " | ".join(models) + " |",
            "|---|---|" + "---|" * len(models),
        ]
        keys = []
        for r in rows:
            k = (r["suite"], r["dimension"], r["level"], r["detail"])
            if r["benchmark"] == bench and k not in keys:
                keys.append(k)
        idx = {
            (r["benchmark"], r["suite"], r["dimension"], r["level"], r["detail"], r["model"]): r for r in rows
        }
        for suite, dim, level, detail in keys:
            name = f"{detail} {level}" if detail != level else detail
            cells = [cell(idx.get((bench, suite, dim, level, detail, m))) for m in models]
            lines.append(f"| {SUITE_SHORT.get(suite, suite)} | {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Runs", "", "| run | model | episodes | exec | bench_version |", "|---|---|---|---|---|"]
    for run in runs:
        rs = [r for r in eps if r["_run"] == run.name]
        lines.append(
            f"| {run} | {','.join(sorted({r['model'] for r in rs}))} | {len(rs)} | "
            f"{','.join(sorted({r['_exec'] for r in rs}))} | {','.join(sorted({str(r['bench_version']) for r in rs}))} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items() if k in COLUMNS})


# ---------------------------------------------------------------- videos


def video_cases(
    eps: list[dict[str, Any]], models: list[str], n: int
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Same (benchmark, suite, task, init, perturbation, variant, repeat, seed) with a video for ≥ 2 models.
    Models that ran the same episode with another seed are not paired (warning). Disagreeing outcomes first,
    then shortcut-relevant (dual) cases; at most one case per (suite, detail) until every (suite, detail)
    had one."""
    by: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in eps:
        if r.get("video"):
            k = (
                r["benchmark"],
                r["suite"],
                r["task_id"],
                r["init_idx"],
                *pkey(r),
                r["variant_id"],
                r.get("repeat_idx", 0),
                r["seed"],
            )
            by[k].setdefault(r["model"], r)
    seeds: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for k, per in by.items():
        for m in per:
            seeds[k[:-1]][m].add(k[-1])
    warns = []
    for k, per_model in seeds.items():
        if len(per_model) > 1 and len({frozenset(v) for v in per_model.values()}) > 1:
            desc = ", ".join(f"{m} seed {sorted(v)}" for m, v in per_model.items())
            warns.append(
                f"not paired for videos, seeds differ: {k[0]} {k[1]} t{k[2]} init {k[3]} {k[6]}/{k[7]} ({desc})"
            )
    cands = []
    for k, per in by.items():
        if len(per) < 2:
            continue
        row = [per[m] for m in models if m in per]
        outcomes = {(bool(r["success"]), r.get("success_orig")) for r in row}
        dual = row[0].get("success_new") is not None
        cands.append((len(outcomes) == 1, not dual, k[2], k[3], k, row))
    cands.sort(key=lambda c: c[:4])
    picked: list[list[dict[str, Any]]] = []
    used: set[tuple] = set()
    for strict in (True, False):
        for c in cands:
            sd = (c[4][0], c[4][1], c[4][6])
            if (strict and sd in used) or any(c[5] is p for p in picked) or len(picked) >= n:
                continue
            used.add(sd)
            picked.append(c[5])
    return picked, warns


def case_type(row: list[dict[str, Any]]) -> str:
    succ = {bool(r["success"]) for r in row}
    if len(succ) > 1:
        return "disagree"
    if row[0].get("success_new") is not None and all(r.get("success_orig") for r in row):
        return "shortcut"
    return "all_success" if succ == {True} else "all_fail"


def make_videos(cases: list[list[dict[str, Any]]], out: Path, views: str) -> list[dict[str, Any]]:
    vdir, tdir = out / "videos", out / "thumbs"
    vdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in cases:
        r0 = row[0]
        p = r0["perturbation"]
        cid = (
            f"M_{r0['benchmark']}_{SUITE_SHORT.get(r0['suite'], r0['suite'])}_t{r0['task_id']:02d}_i{r0['init_idx']}"
            f"_{p['detail']}-{r0['variant_id']}_r{r0.get('repeat_idx', 0)}_s{r0['seed']}"
        ).replace("/", "-")
        clips = [read_clip(Path(r["_root"]) / r["video"], views) for r in row]
        frames = hstack_clips(clips)
        mp4, gif, thumb = vdir / f"{cid}.mp4", vdir / f"{cid}.gif", tdir / f"{cid}.png"
        imageio.mimwrite(mp4, frames, fps=10, codec="libx264", quality=7, macro_block_size=8)
        make_gif(mp4, gif, frames[0].shape[1])
        imageio.imwrite(thumb, frames[-1])
        typ = case_type(row)
        outcome = ", ".join(
            f"{r['model']} {'ok' if r['success'] else 'fail'}"
            + (
                f" (orig goal {'yes' if r['success_orig'] else 'no'})"
                if r.get("success_new") is not None
                else ""
            )
            for r in row
        )
        manifest.append(
            {
                "id": cid,
                "category": "models",
                "model": ",".join(r["model"] for r in row),
                "models": [r["model"] for r in row],
                "benchmark": r0["benchmark"],
                "suite": r0["suite"],
                "task_id": r0["task_id"],
                "init": r0["init_idx"],
                "repeat_idx": r0.get("repeat_idx", 0),
                "seed": r0["seed"],
                "conditions": [f"{p['detail']}/{r0['variant_id']}"] * len(row),
                "instructions": [r["instruction_given"] for r in row],
                "instruction_original": r0["instruction_original"],
                "perturbation": p,
                "outcomes": [
                    {
                        "model": r["model"],
                        "cond": f"{p['detail']}/{r['variant_id']}",
                        "success": r["success"],
                        "success_orig": r.get("success_orig"),
                        "success_new": r.get("success_new"),
                        "t_orig": r.get("t_orig"),
                        "t_new": r.get("t_new"),
                        "first_grasp": r.get("first_grasp_object"),
                        "video_src": str(Path(r["_root"]) / r["video"]),
                    }
                    for r in row
                ],
                "type": typ,
                "why": f"{r0['suite']} t{r0['task_id']} init {r0['init_idx']} {p['detail']}/{r0['variant_id']}: {outcome}",
                "mp4": str(mp4),
                "gif": str(gif),
                "gif_bytes": gif.stat().st_size,
                "thumb": str(thumb),
            }
        )
    return manifest


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="sc-report", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", required=True, help="comma list: run dirs, run ids, or ckpt ids")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--out", default=None, help="default <out-root>/reports/<models>")
    ap.add_argument("--videos", type=int, default=6, help="cross-model comparison videos (0 = none)")
    ap.add_argument("--views", choices=("agent", "both"), default="agent")
    args = ap.parse_args(argv)
    root = registry.out_root(args.out_root)
    runs = resolve_runs([x for x in args.runs.split(",") if x], root)
    eps, dups, load_warns = load(runs)
    if not eps:
        raise SystemExit("no episodes in " + ", ".join(map(str, runs)))
    models = list(dict.fromkeys(r["model"] for r in eps))
    out = Path(args.out) if args.out else root / "reports" / "_vs_".join(models)
    out.mkdir(parents=True, exist_ok=True)
    rows = table(eps)
    cases, video_warns = video_cases(eps, models, args.videos) if args.videos else ([], [])
    warns = load_warns + warnings(rows) + video_warns
    if dups:
        with open(out / "duplicates.csv", "w", newline="") as f:
            dw = csv.DictWriter(f, fieldnames=["key", "used", "dropped", "conflict"])
            dw.writeheader()
            dw.writerows({**d, "key": "|".join(map(str, d["key"]))} for d in dups)
    for w in warns:
        print(f"[warn] {w}")
    write_csv(rows, out / "table.csv")
    (out / "report.md").write_text(markdown(rows, models, warns, runs, eps))
    manifest = make_videos(cases, out, args.views)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"{out}: {len(eps)} episodes, models {models}, {len(rows)} rows, {len(manifest)} videos, {len(warns)} warnings"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
