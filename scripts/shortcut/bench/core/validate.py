"""Battery validation against the adapters' catalogs (T019), before anything is planned or started.

Errors (the run stops, every problem listed, with the closest known name): unknown benchmark, job key, suite,
condition / perturbation, level, detail / variant id, option value (instruction, instruction_source,
init_source), task id outside the suite.
Warnings (the run goes on): a job whose entry lacks an installed requirement is skipped (e.g. LIBERO-Plus
without its venv); a detail that lacks one (motion blur without libMagickWand) is left out of the job's
selection by the adapter; tasks without variants of a condition are skipped by the planner.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from .adapters import ADAPTERS, get_adapter
from .battery import EXPAND, expand_one, merged_jobs
from .catalog import CatalogEntry

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
OPTION_KEYS = ("instruction", "instruction_source", "init_source")


@dataclass
class Validation:
    jobs: list[dict[str, Any]] = field(default_factory=list)  # expanded jobs that will be planned
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # one line per dropped expanded job


def suggest(x: Any, known: Any) -> str:
    known = [str(k) for k in known]
    m = difflib.get_close_matches(str(x), known, n=1, cutoff=0.4)
    return f" (did you mean {m[0]!r}?)" if m else f" (known: {', '.join(known)})"


def as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else [v]


def parse_ids(spec: Any) -> list[int]:
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return [int(x) for x in spec]
    out: list[int] = []
    for part in str(spec).split(","):
        a, _, b = part.strip().partition("-")
        if a:
            out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def _lv(x: Any) -> str:
    return "None" if x is None else str(x)


def validate_battery(battery: dict[str, Any], backend: str = "libero") -> Validation:
    v = Validation()
    cats: dict[tuple[str, str, str], dict[str, CatalogEntry]] = {}  # (benchmark, suite, dir) → key → entry
    suites_of: dict[str, list[str]] = {}

    def entries(b: str, s: str, job: dict[str, Any] | None = None) -> dict[str, CatalogEntry]:
        # ours custom jobs (T020) may name their own variants_dir: validate against that catalog
        vdir = str((job or {}).get("variants_dir") or "")
        if (b, s, vdir) not in cats:
            ad = get_adapter(b)
            rows = ad.catalog_for_job(s, job) if vdir and hasattr(ad, "catalog_for_job") else ad.catalog(s)
            cats[b, s, vdir] = {e["key"]: e for e in rows}
        return cats[b, s, vdir]

    for i, job in enumerate(merged_jobs(battery)):
        where = f"job {i} ({job.get('benchmark')})"
        b = str(job["benchmark"])
        if b not in ADAPTERS:
            v.errors.append(f"{where}: unknown benchmark {b!r}{suggest(b, ADAPTERS)}")
            continue
        ad = get_adapter(b)
        if b not in suites_of:
            suites_of[b] = list(getattr(ad, "suites", SUITES))
        allowed = ad.allowed_job_keys() | set(EXPAND)
        for k in job:
            if k not in allowed:
                v.errors.append(f"{where}: unknown key {k!r}{suggest(k, sorted(allowed))}")
        kf = ad.key_field
        if kf not in job and f"{kf}s" not in job and kf == "condition":
            v.errors.append(f"{where}: needs `conditions:` (or `condition:`)")
            continue
        n_err = len(v.errors)
        exp = expand_one(job)
        suites = suites_of[b]
        for j in exp:
            j.setdefault(kf, "none")
            s, key = j.get("suite"), str(j[kf])
            if s not in suites:
                v.errors.append(f"{where}: unknown suite {s!r}{suggest(s, suites)}")
                continue
            e = entries(b, s, j).get(key)
            if e is None:
                keys = sorted(entries(b, s, j))
                v.errors.append(f"{where}: {s}: unknown {kf} {key!r}{suggest(key, keys)}")
                continue
            lvs = [_lv(x) for x in e.get("levels") or []]
            for lv in as_list(j.get("levels") or []):
                if _lv(lv) not in lvs:
                    v.errors.append(
                        f"{where}: {s} {key}: unknown level {lv!r}"
                        + (suggest(lv, lvs) if lvs else " (this entry has no levels)")
                    )
            dkey = "variants" if kf == "condition" else "details"
            dets = e.get("details") or []
            infeasible = e.get("infeasible") or {}
            with_infeasible = bool(j.get("include_infeasible"))
            if dkey in j and not (dkey == "details" and j[dkey] is None):
                vals = as_list(j[dkey])
                if dkey == "details" or not (len(vals) == 1 and str(vals[0]).isdigit()):
                    for d in vals:
                        if str(d) not in dets:
                            v.errors.append(
                                f"{where}: {s} {key}: unknown {dkey[:-1]} {d!r}{suggest(d, dets)}"
                            )
                        elif not with_infeasible:
                            bad = sorted(int(t) for t, ids in infeasible.items() if str(d) in ids)
                            if bad:
                                v.warnings.append(
                                    f"{where}: {s} {key}: variant {d!r} is marked infeasible (T002) in tasks {bad}; "
                                    "those run only with `include_infeasible: true`"
                                )
            for ok in OPTION_KEYS:
                if ok in j and ok in (e.get("options") or {}) and str(j[ok]) not in e["options"][ok]:
                    v.errors.append(f"{where}: {ok} {j[ok]!r}{suggest(j[ok], e['options'][ok])}")
            if "tasks" in j:
                try:
                    tids = parse_ids(j["tasks"])
                except ValueError:
                    v.errors.append(f"{where}: tasks {j['tasks']!r}: not a list / range of ints")
                    tids = []
                bad = [t for t in tids if t < 0 or t > 9]
                if bad:
                    v.errors.append(f"{where}: tasks {bad} outside 0-9")
                have = set(e.get("tasks") or range(10))
                if with_infeasible:
                    have |= {int(t) for t in infeasible}
                none = [t for t in tids if 0 <= t <= 9 and t not in have]
                if none:
                    v.warnings.append(
                        f"{where}: {s} {key}: tasks {none} have no variants (skipped by the planner)"
                    )
        if len(v.errors) > n_err:
            continue
        for j in exp:
            e = entries(b, j["suite"], j)[str(j[kf])]
            label = f"{b} {j['suite']} {j[kf]}"
            # mock backend (tests, e2e --mock): the adapters run their mock env in this interpreter, so missing
            # venvs are not a reason to skip; their own plan errors still apply
            missing = [] if backend == "mock" else list(e.get("missing") or [])
            if missing:
                v.skipped.append(f"{label}: SKIPPED, missing {', '.join(missing)}")
                continue
            want = (
                [str(d) for d in as_list(j["details"])] if j.get("details") else list(e.get("details") or [])
            )
            for d, m in (e.get("missing_by_detail") or {}).items():
                if m and d in want:
                    v.warnings.append(f"{label}: {d} variants left out (missing {', '.join(m)})")
            v.jobs.append(j)
    return v


def report(v: Validation, path: str) -> None:
    """Print warnings / skips; raise SystemExit listing every error."""
    for w in v.warnings:
        print(f"[sc-bench] warning: {w}", flush=True)
    for s in v.skipped:
        print(f"[sc-bench] warning: {s}", flush=True)
    if v.errors:
        raise SystemExit(f"battery {path}: {len(v.errors)} error(s)\n  " + "\n  ".join(v.errors))
