"""What can be selected in a battery: every adapter's `catalog(suite)` → `CatalogEntry` rows (T019).

One entry = one selectable (benchmark, suite, key): `key` is the value of the job field `key_field`
(ours: `condition` C0–C6 + T020's user variants; libero_plus / libero_pro: `perturbation` = axis).
`levels` / `details` are the values a job may restrict to (`levels:`, `details:` / ours `variants:`);
`count` is how many selectable items the entry holds in that suite (variants × tasks, or tasks × levels for PRO).
`requires` names what has to be installed for the whole entry, `requires_by_detail` for single sub-types
(LIBERO-Plus motion blur: libMagickWand); `missing` / `missing_by_detail` are what this machine lacks
(filled by the adapter from the venv / install markers; empty when everything is there).

Used by `sc-bench --list` (table / --json), by the battery validation (`validate.py`: unknown names → error with
a suggestion, missing requirements → warning + skip) and by the job time estimate.
"""

from __future__ import annotations

from typing import Any, TypedDict


class CatalogEntry(TypedDict, total=False):
    benchmark: str  # ours | libero_plus | libero_pro
    suite: str  # libero_spatial | libero_object | libero_goal | libero_10
    key_field: str  # job field that selects it: condition | perturbation
    key: str  # C0 … / language, layout, … / none, object, …
    levels: list[Any]  # allowed `levels:` values (Plus: difficulty 1–5 / None; PRO: swap, x0.1, …); [] = none
    details: list[str]  # allowed `details:` (Plus sub-types) or ours `variants:` ids; [] = none
    count: int | None  # selectable items in this suite (None = unknown: install missing)
    count_by_level: dict[str, int]  # optional breakdown
    count_by_detail: dict[str, int]  # optional breakdown
    tasks: list[int]  # task ids that have this entry
    description: str  # one line: what it changes / which cue it removes
    options: dict[str, list[str]]  # other job fields with a closed set of values (PRO instruction_source, …)
    requires: list[str]  # what the whole entry needs (human-readable)
    requires_by_detail: dict[str, list[str]]  # what single details need
    missing: list[str]  # subset of `requires` absent here
    missing_by_detail: dict[str, list[str]]  # subset of `requires_by_detail` absent here
    source: str  # where the entry comes from (builtin | user variant file | …)


def entry(
    benchmark: str,
    suite: str,
    key_field: str,
    key: str,
    *,
    description: str,
    levels: list[Any] | None = None,
    details: list[str] | None = None,
    count: int | None = None,
    **extra: Any,
) -> CatalogEntry:
    e: CatalogEntry = {
        "benchmark": benchmark,
        "suite": suite,
        "key_field": key_field,
        "key": key,
        "levels": list(levels or []),
        "details": list(details or []),
        "count": count,
        "description": description,
        "requires": [],
        "requires_by_detail": {},
        "missing": [],
        "missing_by_detail": {},
    }
    e.update(extra)  # type: ignore[typeddict-item]
    return e


def all_entries(benchmark: str | None = None, suite: str | None = None) -> list[CatalogEntry]:
    from .adapters import ADAPTERS, get_adapter

    out: list[CatalogEntry] = []
    for b in [benchmark] if benchmark else list(ADAPTERS):
        out += get_adapter(b).catalog(suite)
    return out


def _fmt_levels(xs: list[Any], limit: int = 8) -> str:
    s = [("-" if x is None else str(x)) for x in xs]
    return ",".join(s) if len(s) <= limit else ",".join(s[:limit]) + f",…(+{len(s) - limit})"


def format_table(entries: list[CatalogEntry]) -> str:
    """Human-readable table, one block per benchmark."""
    lines: list[str] = []
    by_bench: dict[str, list[CatalogEntry]] = {}
    for e in entries:
        by_bench.setdefault(e["benchmark"], []).append(e)
    for b, es in by_bench.items():
        kf = es[0]["key_field"]
        lines.append(f"== {b}  (battery: benchmark: {b}, {kf}s: [...])")
        lines.append(f"{'suite':<15} {kf:<13} {'count':>6}  {'levels':<26} {'details / variants':<50} needs")
        for e in es:
            need = []
            if e.get("missing"):
                need.append("MISSING " + ", ".join(e["missing"]))
            for d, m in (e.get("missing_by_detail") or {}).items():
                if m:
                    need.append(f"{d}: MISSING {', '.join(m)}")
            if not need and (e.get("requires") or e.get("requires_by_detail")):
                need.append("ok")
            cnt = "?" if e.get("count") is None else str(e["count"])
            lines.append(
                f"{e['suite']:<15} {e['key']:<13} {cnt:>6}  {_fmt_levels(e.get('levels') or []):<26} "
                f"{_fmt_levels(e.get('details') or [], 6):<50} {'; '.join(need)}"
            )
        seen: dict[str, str] = {}
        for e in es:
            seen.setdefault(e["key"], e["description"])
        lines.append("  " + "-" * 60)
        for k, d in seen.items():
            lines.append(f"  {k:<13} {d}")
        opts: dict[str, list[str]] = {}
        for e in es:
            for k, v in (e.get("options") or {}).items():
                opts.setdefault(k, v)
        for k, v in opts.items():
            lines.append(f"  option {k}: {' | '.join(v)}")
        lines.append("")
    return "\n".join(lines)
