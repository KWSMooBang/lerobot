"""CPU-only checks for the shortcut harness helpers: uv run pytest scripts/shortcut/tests -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import episode_seed, load_instruction_set, parse_range, variants_for, video_relpath  # noqa: E402
from metrics import dtw, mcnemar_exact, wilson  # noqa: E402

FIXTURE = HERE / "fixtures" / "libero_goal_2tasks.json"


class _Tok:
    def encode(self, s, add_special_tokens=False):
        return s.split()

    def get_vocab(self):
        return {"▁apple": 0, "▁river": 1, "▁stone": 2, "<pad>": 3, "▁X": 4}


def test_wilson_bounds():
    lo, hi = wilson(50, 50)
    assert hi == pytest.approx(1.0) and 0.92 < lo < 0.94
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.32


def test_mcnemar():
    assert mcnemar_exact([(True, True)] * 10) == 1.0
    assert mcnemar_exact([(True, False)] * 10) < 0.01


def test_dtw_identity_and_shift():
    a = np.zeros((50, 3))
    a[:, 0] = np.linspace(0, 0.5, 50)
    assert dtw(a, a) == pytest.approx(0.0)
    assert dtw(a, a + [0, 0, 0.1]) == pytest.approx(0.1, rel=1e-6)  # perpendicular offset


def test_parse_range_and_seed():
    assert parse_range("0-2,5") == [0, 1, 2, 5]
    assert episode_seed(7, 0, 8, 3) == episode_seed(7, 0, 8, 3)
    assert episode_seed(7, 1, 8, 3) != episode_seed(7, 0, 8, 3)
    assert video_relpath("libero_goal", 8, "C4", "C4-a", 3, 0) == "libero_goal/task08/C4_C4-a_init03"
    assert video_relpath("libero_goal", 8, "C0", "orig", 3, 2).endswith("C0_orig_r2_init03")


def test_variants_from_fixture():
    s = load_instruction_set(FIXTURE)
    s["_suite_tasks"] = {}
    t8 = s["_by_task"][8]
    tok = _Tok()
    assert [v.instruction for v in variants_for("C0", t8, s, tok)] == ["put the bowl on the plate"]
    assert variants_for("C1", t8, s, tok)[0].instruction == ""
    c2 = variants_for("C2", t8, s, tok)[0].instruction
    assert (
        len(c2.split()) == len(t8["original"].split()) and c2 == variants_for("C2", t8, s, tok)[0].instruction
    )
    assert len(variants_for("C3", t8, s, tok)) == 3
    c4 = variants_for("C4", t8, s, tok)
    assert [v.variant for v in c4] == ["C4-a"] and Path(c4[0].goal_bddl).exists()
    c6 = variants_for("C6", t8, s, tok)
    assert c6[0].instruction == "put the bowl on the stove" and c6[0].variant == "t1"
