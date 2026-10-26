"""CPU-only checks for the T007 overlap analysis: uv run pytest scripts/shortcut/tests/test_train_overlap.py -q"""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "analysis"))

from slots import mentions, parse  # noqa: E402
from train_overlap import Swap, component_counts, lcs_run, lev, strip_block  # noqa: E402


@pytest.mark.parametrize(
    "text, want",
    [
        (
            "pick up the black bowl next to the ramekin and place it on the plate",
            [("place", "black bowl", "black bowl next to ramekin", "on:plate")],
        ),
        (
            "pick up the black bowl between the plate and the ramekin and place it on the cookie box",
            [("place", "black bowl", "black bowl between plate and ramekin", "on:cookie box")],
        ),
        ("put the bowl on top of the cabinet", [("place", "black bowl", "black bowl", "on:wooden cabinet")]),
        ("push the plate to the front of the stove", [("push", "plate", "plate", "front:stove")]),
        (
            "open the top drawer and put the bowl inside",
            [
                ("open", "top drawer", "top drawer", ""),
                ("place", "black bowl", "black bowl", "in:top drawer"),
            ],
        ),
        (
            "put both the cream cheese box and the butter in the basket",
            [
                ("place", "cream cheese", "cream cheese", "in:basket"),
                ("place", "butter", "butter", "in:basket"),
            ],
        ),
        (
            "turn on the stove and put the moka pot on it",
            [("turn on", "stove", "stove", ""), ("place", "moka pot", "moka pot", "on:stove")],
        ),
        (
            "put the black bowl in the bottom drawer of the cabinet and close it",
            [
                ("place", "black bowl", "black bowl", "in:bottom drawer"),
                ("close", "bottom drawer", "bottom drawer", ""),
            ],
        ),
    ],
)
def test_parse(text, want):
    assert [(a.verb, a.obj, a.obj_ref, a.dest_str) for a in parse(text)] == want


def test_mentions_include_qualifier_landmarks():
    assert mentions("pick up the black bowl on the cookie box and place it on the plate") == {
        "black bowl",
        "cookie box",
        "plate",
    }


def test_string_metrics():
    assert lev("kitten", "sitting") == 3
    assert lcs_run(["a", "b", "c", "d"], ["x", "b", "c", "y"]) == 2
    assert strip_block("(a (:goal (And (On x y))) b)", "goal") == "(a  b)"


@pytest.mark.parametrize("suite", ["libero_spatial", "libero_goal"])
def test_all_swaps_match_bddl_goal(suite):
    """the last parsed action of every swap names the object/destination of its goal predicate"""
    spec = json.loads((HERE.parent / "instructions" / f"{suite}.json").read_text())
    for t in spec["tasks"]:
        for w in t["swaps"]:
            s = Swap(
                suite,
                t["task_id"],
                w["id"],
                w["type"],
                w["instruction"],
                t["original"],
                w["orig_target"],
                w["new_target"],
                w["goal"],
                t["bddl"],
                "",
                parse(w["instruction"]),
            )
            assert component_counts(s, [])["slot_check"] == "ok", s.swap_id
