"""sc-variants: build / check / preview / list user-designed LIBERO variants (CPU only).

sc-variants build   --suite libero_goal [--ids a,b] [--variants-dir DIR] [--force]
sc-variants check   --suite libero_goal [--ids ...] [--workers 16] [--inits 3]
sc-variants preview --suite libero_goal [--ids ...] [--out /outputs/.../analysis/variants]
sc-variants all     ...      (build → check → preview)
sc-variants list    [--suite ...] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]  # no shadowing by this dir
sys.path.insert(0, str(HERE.parent))

from variants import preview as preview_mod  # noqa: E402  (sets CPU EGL before common.py is imported)
from variants.spec import (  # noqa: E402
    SUITES,
    find_spec,
    load_manifest,
    spec_paths,
    variant_rows,
    variants_dir,
)


def selected(args: argparse.Namespace) -> list[tuple[str, Path]]:
    suites = [args.suite] if args.suite else list(SUITES)
    ids = [x for x in (args.ids or "").split(",") if x]
    out = []
    for s in suites:
        if ids:
            out += [
                (s, find_spec(s, i, args.variants_dir))
                for i in ids
                if (variants_dir(args.variants_dir) / s / f"{i}.yaml").exists()
            ]
        else:
            out += [(s, p) for p in spec_paths(s, args.variants_dir)]
    if ids and len(out) != len(ids):
        found = {p.stem for _, p in out}
        raise SystemExit(f"unknown variant ids: {sorted(set(ids) - found)}")
    if not out:
        raise SystemExit("no variant specs selected")
    return out


def do_build(args: argparse.Namespace) -> list[dict]:
    from variants.generate import BuildError, build
    from variants.spec import SpecError, load_spec

    ms, bad = [], []
    for s, p in selected(args):
        try:
            m = build(load_spec(p), args.variants_dir, force=args.force)
        except (SpecError, BuildError) as e:
            print(f"[build] FAIL {s}/{p.stem}: {e}", flush=True)
            bad.append(p.stem)
            continue
        m = load_manifest(s, m["id"], args.variants_dir)
        chk = m["init_states"]["check"]
        print(
            f"[build] {s}/{m['id']}: init {m['init_states']['mode']} n={m['init_states']['n']}"
            + (f" (dropped {chk['n_failed']}/{chk['n_candidates']})" if chk.get("n_failed") else ""),
            flush=True,
        )
        ms.append(m)
    if bad:
        print(f"[build] {len(bad)} failed: {bad}", flush=True)
    return ms


def built(args: argparse.Namespace) -> list[dict]:
    return [load_manifest(s, p.stem, args.variants_dir) for s, p in selected(args)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="sc-variants", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("build", "check", "preview", "all", "list"))
    ap.add_argument("--suite", choices=SUITES)
    ap.add_argument("--ids", default=None, help="comma list of variant ids (default: all in the suite)")
    ap.add_argument("--variants-dir", default=None, help=f"default {HERE} (<dir>/<suite>/<id>.yaml)")
    ap.add_argument("--force", action="store_true", help="rebuild even if the manifest is up to date")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--inits", type=int, default=3, help="init states per feasibility check")
    ap.add_argument("--out", default=str(preview_mod.DEFAULT_OUT), help="preview / gallery dir")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        cat = variant_rows(args.suite, args.variants_dir)
        if args.json:
            print(json.dumps(cat, indent=1))
        else:
            print(f"{'suite':<14} {'id':<34} {'built':<6} {'n':>3} {'feasibility':<11} description")
            for c in cat:
                n = "" if c.get("count") is None else c["count"]
                print(
                    f"{c['suite']:<14} {c['details'][0]:<34} {('stale' if c.get('stale') else 'yes' if c.get('built') else 'no'):<6} "
                    f"{n!s:>3} {c.get('feasibility') or '-':<11} {c['description']}"
                )
        return 0
    ms = do_build(args) if args.cmd in ("build", "all") else built(args)
    if args.cmd in ("check", "all"):
        from variants.check import check

        check(ms, workers=args.workers, n_inits=args.inits)
        ms = (
            built(args)
            if args.cmd == "check"
            else [load_manifest(m["suite"], m["id"], args.variants_dir) for m in ms]
        )
    if args.cmd in ("preview", "all"):
        preview_mod.preview(ms, args.out, workers=args.workers)
    if args.cmd != "build":
        counts: dict[str, int] = {}
        for m in ms:
            st = m.get("feasibility", {}).get("status", "not checked")
            counts[st] = counts.get(st, 0) + 1
        print("[summary] feasibility:", json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
