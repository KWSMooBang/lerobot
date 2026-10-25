"""CPU-only checks for the shortcut harness helpers: uv run pytest scripts/shortcut/tests -q"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    check_resume,
    episode_seed,
    load_instruction_set,
    parse_range,
    variants_for,
    video_relpath,
)
from metrics import (  # noqa: E402
    aggregate_variants,
    check_unique,
    dtw,
    index_c0,
    mcnemar_exact,
    pair_key,
    wilson,
)

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


def _ep(cond, variant, task, init, ok, repeat=0, seed_base=7, root="r"):
    return {
        "model": "m",
        "ckpt": "c",
        "suite": "libero_goal",
        "condition": cond,
        "variant": variant,
        "task_id": task,
        "init_idx": init,
        "repeat_idx": repeat,
        "seed": episode_seed(seed_base, repeat, task, init),
        "success_orig": ok,
        "_root": root,
    }


def test_variants_aggregated_once_per_task_init():
    c0 = {pair_key(e): e for e in [_ep("C0", "orig", 0, i, True) for i in range(3)]}
    # 3 paraphrases on init 0 (2 fail) and init 1 (all fail), 1 on init 2 (success)
    c3 = [_ep("C3", f"p{j}", 0, 0, j == 0) for j in range(3)]
    c3 += [_ep("C3", f"p{j}", 0, 1, False) for j in range(3)] + [_ep("C3", "p0", 0, 2, True)]
    agg = aggregate_variants(c3, c0)
    assert len(agg) == 3  # one pair per (task, init), not one per variant episode
    assert sorted(f for f, _ in agg) == pytest.approx([0.0, 1 / 3, 1.0])


def test_c0_repeats_are_not_the_reference_and_duplicates_fail():
    eps = [_ep("C0", "orig", 0, 0, True), _ep("C0", "orig", 0, 0, False, repeat=1)]
    ref, reps = index_c0(eps)
    assert list(ref.values())[0]["success_orig"] is True
    assert set(reps[("m", "c", "libero_goal", 0, 0)]) == {0, 1}
    with pytest.raises(ValueError, match="duplicate"):
        index_c0(eps + [_ep("C0", "orig", 0, 0, True, root="other")])
    with pytest.raises(ValueError, match="duplicate"):
        check_unique([_ep("C4", "a", 0, 0, True), _ep("C4", "a", 0, 0, True, root="x")])


def test_resume_guard(tmp_path):
    ident = {"checkpoint": "c", "suite": "libero_goal", "condition": "C4", "repeat_idx": 0, "seed": 7}
    ep_path = tmp_path / "episodes.jsonl"
    check_resume(ident, None, ep_path)  # fresh dir
    rec = {
        "ckpt": "c",
        "suite": "libero_goal",
        "condition": "C4",
        "repeat_idx": 0,
        "task_id": 1,
        "init_idx": 2,
    }
    ep_path.write_text(json.dumps({**rec, "seed": episode_seed(7, 0, 1, 2)}) + "\n")
    check_resume(ident, {"identity": ident}, ep_path)  # same config resumes
    with pytest.raises(SystemExit, match="another configuration"):
        check_resume({**ident, "seed": 8}, {"identity": ident}, ep_path)
    with pytest.raises(SystemExit, match="another configuration"):
        check_resume({**ident, "checkpoint": "other"}, {"identity": ident}, ep_path)
    with pytest.raises(SystemExit, match="without run_meta"):
        check_resume(ident, None, ep_path)
    ep_path.write_text(json.dumps({**rec, "seed": 123}) + "\n")
    with pytest.raises(SystemExit, match="does not match"):
        check_resume(ident, {"identity": ident}, ep_path)
