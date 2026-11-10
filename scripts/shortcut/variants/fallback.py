"""Fallback table for custom-variant episodes (sc-report): did a failure do nothing, or a trained task?

Per (model, suite, variant): n, new-goal success, per tracked training goal `fallback` (reached in an episode
where the new goal was not: same basis as any_trained, so the per-goal rates of a variant are ≤ any_trained ≤ their
sum) and `reached_any` (reached at all, unconditional), `any_trained` (some tracked goal reached, new goal not), `nothing` (neither the new goal nor any
tracked goal). Goals that already held right after reset (goals_initial) do not count for that episode, and a
tracked goal identical to the variant goal is left out (it is the new goal, not a fallback).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

FIELDS = [
    "model",
    "suite",
    "variant",
    "group",
    "pair",
    "phrasing",
    "instruction",
    "feasibility",
    "n",
    "sr_new",
    "any_trained",
    "nothing",
    "goal",
    "goal_instruction",
    "fallback",  # this goal reached and the new goal not (episodes counted in any_trained)
    "reached_any",  # this goal reached at all, new goal or not
]


def _same_goal(a: list[list[str]], b: list[list[str]]) -> bool:
    norm = lambda g: sorted(" ".join([p[0].lower(), *p[1:]]) for p in g)  # noqa: E731
    return norm(a) == norm(b)


def fallback_rows(eps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(summary rows [one per model/suite/variant], long rows [one per tracked goal])."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in eps:
        if r.get("goals_reached") is not None:
            groups[(r["model"], r["suite"], r.get("variant_id", r.get("variant")))].append(r)
    summary, long = [], []

    def order(item: tuple) -> tuple:  # wordings of one goal next to each other
        (model, suite, vid), rs = item
        meta = rs[0].get("variant_meta") or {}
        return (model, suite, meta.get("pair") or vid, vid)

    for (model, suite, vid), rs in sorted(groups.items(), key=order):
        meta = rs[0].get("variant_meta") or {}
        goal = meta.get("goal") or []
        names = [
            n
            for n in rs[0]["goals_reached"]
            if not (goal and _same_goal(meta.get("track_goals", {}).get(n, []), goal))
        ]
        n = len(rs)
        hit = dict.fromkeys(names, 0)  # reached while the new goal was not (same basis as any_trained)
        hit_any = dict.fromkeys(names, 0)  # reached at all (unconditional)
        any_tr = nothing = new_ok = 0
        for r in rs:
            init = set(r.get("goals_initial") or [])
            reached = {k for k in names if r["goals_reached"].get(k) is not None and k not in init}
            new = bool(r.get("success_new"))
            for k in reached:
                hit_any[k] += 1
                hit[k] += not new
            new_ok += new
            any_tr += bool(reached) and not new
            nothing += not reached and not new
        summary.append(
            {
                "model": model, "suite": suite, "variant": vid, "group": meta.get("group"),
                "pair": meta.get("pair"), "phrasing": meta.get("phrasing"), "instruction": rs[0].get("instruction"),
                "feasibility": meta.get("feasibility"), "n": n,
                "sr_new": new_ok / n if goal else None, "any_trained": any_tr / n, "nothing": nothing / n,
                "fallbacks": sorted(((k, hit[k] / n) for k in names if hit[k]), key=lambda x: -x[1]),
                "instructions": meta.get("track_instructions", {}),
            }
        )  # fmt: skip
        for k in names:
            long.append(
                {
                    "model": model, "suite": suite, "variant": vid, "group": meta.get("group"),
                "pair": meta.get("pair"), "phrasing": meta.get("phrasing"), "instruction": rs[0].get("instruction"),
                "feasibility": meta.get("feasibility"), "n": n,
                    "sr_new": new_ok / n if goal else None, "any_trained": any_tr / n, "nothing": nothing / n,
                    "goal": k, "goal_instruction": meta.get("track_instructions", {}).get(k),
                    "fallback": hit[k] / n, "reached_any": hit_any[k] / n,
                }
            )  # fmt: skip
    return summary, long


def fallback_markdown(summary: list[dict[str, Any]]) -> list[str]:
    if not summary:
        return []
    pct = lambda x: "–" if x is None else f"{100 * x:.0f}"  # noqa: E731
    lines = [
        "## Custom variants: fallback",
        "",
        "new = variant goal reached; trained = a training-task goal reached instead (not the new goal); "
        "nothing = neither. Fallbacks: per training goal, % of episodes that reached it while the new goal was not "
        "reached (same basis as trained); goals true at reset excluded. Unconditional rates: fallback.csv reached_any.",
        "",
        "| model | suite | variant | instruction | feasibility | n | new % | trained % | nothing % | fallbacks |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        fb = ", ".join(
            f"{k} {pct(v)}%" + (f" ({s['instructions'][k]})" if s["instructions"].get(k) else "")
            for k, v in s["fallbacks"][:4]
        )
        lines.append(
            f"| {s['model']} | {s['suite']} | {s['variant']} | {s['instruction']} | {s.get('feasibility') or '–'} | {s['n']} | {pct(s['sr_new'])} | "
            f"{pct(s['any_trained'])} | {pct(s['nothing'])} | {fb or '—'} |"
        )
    return [*lines, ""]
