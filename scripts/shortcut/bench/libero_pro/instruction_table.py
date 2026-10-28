"""PRO :language vs the LIBERO training sentence for all 40 tasks (stdlib; any venv).

    uv run python scripts/shortcut/bench/libero_pro/instruction_table.py --out DIR/instructions.csv

Columns: training sentence (= ours C0), PRO clean-suite :language, PRO file-name sentence, semantic and task
axis sentences, whether PRO clean = training (case / spacing ignored), the word-level edit PRO clean ← training,
and for the task axis the edit from the training sentence and a similarity ratio (how far it leaves the template).
"""

from __future__ import annotations

import argparse
import csv
import difflib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from bench.libero_pro import catalog  # noqa: E402


def norm(s: str) -> list[str]:
    return s.lower().replace(",", " ").split()


def word_edit(a: str, b: str) -> str:
    """'-removed +added' words turning a into b."""
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, norm(a), norm(b)).get_opcodes():
        if op in ("delete", "replace"):
            out.append("-" + " ".join(norm(a)[i1:i2]))
        if op in ("insert", "replace"):
            out.append("+" + " ".join(norm(b)[j1:j2]))
    return "; ".join(out)


def ratio(a: str, b: str) -> float:
    return round(difflib.SequenceMatcher(None, norm(a), norm(b)).ratio(), 3)


def rows(root: Path) -> list[dict[str, Any]]:
    out = []
    for suite in catalog.SUITES:
        for t in range(10):
            e = catalog.task_entry(root, suite, "none", "original", t)
            tr, pro = e["instruction_original"], e["pro_instruction_original"]
            sem = catalog.task_entry(root, suite, "semantic", "paraphrase", t)["instruction_given"]
            task = catalog.task_entry(root, suite, "task", "task", t)["instruction_given"]
            out.append(
                {"suite": suite, "task_id": t, "task_name": e["task_name"], "training": tr, "pro_clean": pro,
                 "pro_filename": catalog.filename_language(e["task_name"]), "pro_semantic": sem, "pro_task": task,
                 "pro_clean_equals_training": norm(pro) == norm(tr), "pro_clean_edit": word_edit(tr, pro),
                 "pro_clean_ratio": ratio(tr, pro), "filename_equals_training": norm(e["task_name"].replace("_", " ")) == norm(tr)
                 or norm(catalog.filename_language(e["task_name"])) == norm(tr),
                 "task_edit_vs_training": word_edit(tr, task), "task_ratio_vs_training": ratio(tr, task),
                 "semantic_ratio_vs_training": ratio(tr, sem)}
            )  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rs = rows(catalog.pro_root())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)
    same = sum(bool(r["pro_clean_equals_training"]) for r in rs)
    fn = sum(bool(r["filename_equals_training"]) for r in rs)
    edits = Counter(e for r in rs for e in str(r["pro_clean_edit"]).split("; ") if e)
    print(
        f"PRO clean :language = training sentence in {same}/{len(rs)}; file-name sentence = training in {fn}/{len(rs)}"
    )
    print("most common PRO-vs-training edits:", edits.most_common(8))
    for s in catalog.SUITES:
        sub = [r for r in rs if r["suite"] == s]
        print(
            f"{s}: PRO clean ratio mean {sum(float(r['pro_clean_ratio']) for r in sub) / 10:.2f}, "
            f"task ratio mean {sum(float(r['task_ratio_vs_training']) for r in sub) / 10:.2f}, "
            f"semantic ratio mean {sum(float(r['semantic_ratio_vs_training']) for r in sub) / 10:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
