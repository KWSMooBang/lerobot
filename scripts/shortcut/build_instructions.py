"""Build the LIBERO shortcut instruction sets (C2/C3/C4/C5/C6) and the swap goal BDDLs.

Writes, for each suite:
  instructions/<suite>.json        task list with original / paraphrases / swaps / cross_task
  instructions/vocab_<suite>.json  words of the suite's 10 training instructions
  bddl/<suite>/<orig>__<swap>.bddl original BDDL with only ``:goal`` and ``:language`` replaced

Swap sentences may only use words that appear in the suite's training instructions; this is
asserted here. ``feasible`` fields are filled in later by ``feasibility.py`` (existing values are
kept when the JSON is rebuilt and the swap is unchanged).

    uv run python scripts/shortcut/build_instructions.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
INSTR_DIR = HERE / "instructions"
BDDL_DIR = HERE / "bddl"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# --------------------------------------------------------------------------- paraphrases (C3)
# Meaning-preserving rewrites. Keys are the original (training) instructions. A third element
# ``review`` marks sentences whose meaning may have drifted and needs a human look.
PARA: dict[str, list[str]] = {
    # libero_spatial: the spatial descriptor of the target bowl must survive unchanged
    "pick up the black bowl between the plate and the ramekin and place it on the plate": [
        "grab the black bowl that sits between the plate and the ramekin and put it on the plate",
        "take the black bowl located between the ramekin and the plate and set it onto the plate",
        "lift the black bowl in between the plate and the ramekin, then put it down on the plate",
    ],
    "pick up the black bowl next to the ramekin and place it on the plate": [
        "grab the black bowl beside the ramekin and put it on the plate",
        "take the black bowl that is next to the ramekin and set it on the plate",
        "move the black bowl located next to the ramekin onto the plate",
    ],
    "pick up the black bowl from table center and place it on the plate": [
        "grab the black bowl in the center of the table and put it on the plate",
        "take the black bowl from the middle of the table and set it on the plate",
        "move the black bowl at the center of the table onto the plate",
    ],
    "pick up the black bowl on the cookie box and place it on the plate": [
        "grab the black bowl sitting on the cookie box and put it on the plate",
        "take the black bowl from the top of the cookie box and set it on the plate",
        "move the black bowl that is on the cookie box onto the plate",
    ],
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": [
        "grab the black bowl inside the top drawer of the wooden cabinet and put it on the plate",
        "take the black bowl from the wooden cabinet's top drawer and set it on the plate",
        "move the black bowl in the uppermost drawer of the wooden cabinet onto the plate",
    ],
    "pick up the black bowl on the ramekin and place it on the plate": [
        "grab the black bowl sitting on the ramekin and put it on the plate",
        "take the black bowl from on top of the ramekin and set it on the plate",
        "move the black bowl that is on the ramekin onto the plate",
    ],
    "pick up the black bowl next to the cookie box and place it on the plate": [
        "grab the black bowl beside the cookie box and put it on the plate",
        "take the black bowl that is next to the cookie box and set it on the plate",
        "move the black bowl located by the cookie box onto the plate",
    ],
    "pick up the black bowl on the stove and place it on the plate": [
        "grab the black bowl sitting on the stove and put it on the plate",
        "take the black bowl off the stove and set it on the plate",
        "move the black bowl that is on the stove onto the plate",
    ],
    "pick up the black bowl next to the plate and place it on the plate": [
        "grab the black bowl beside the plate and put it on the plate",
        "take the black bowl that is next to the plate and set it on top of the plate",
        "move the black bowl located next to the plate onto the plate",
    ],
    "pick up the black bowl on the wooden cabinet and place it on the plate": [
        "grab the black bowl sitting on top of the wooden cabinet and put it on the plate",
        "take the black bowl from the top of the wooden cabinet and set it on the plate",
        "move the black bowl that is on the wooden cabinet onto the plate",
    ],
    # libero_goal
    "open the middle drawer of the cabinet": [
        "pull open the middle drawer of the cabinet",
        "open the cabinet's middle drawer",
        "slide the middle drawer of the cabinet open",
    ],
    "put the bowl on the stove": [
        "place the bowl on the stove",
        "set the bowl on top of the stove",
        "move the bowl onto the stove",
    ],
    "put the wine bottle on top of the cabinet": [
        "place the wine bottle on the top of the cabinet",
        "set the wine bottle on the cabinet's top",
        "move the wine bottle onto the top of the cabinet",
    ],
    "open the top drawer and put the bowl inside": [
        "pull open the top drawer and place the bowl in it",
        "open the cabinet's top drawer, then put the bowl inside it",
        "open the uppermost drawer and set the bowl inside",
    ],
    "put the bowl on top of the cabinet": [
        "place the bowl on the top of the cabinet",
        "set the bowl on the cabinet's top",
        "move the bowl onto the top of the cabinet",
    ],
    "push the plate to the front of the stove": [
        "slide the plate to the front of the stove",
        "push the plate so that it is in front of the stove",
        "move the plate in front of the stove by pushing it",
    ],
    "put the cream cheese in the bowl": [
        "place the cream cheese in the bowl",
        "drop the cream cheese into the bowl",
        "move the cream cheese into the bowl",
    ],
    "turn on the stove": [
        "switch on the stove",
        "turn the stove on",
        "power on the stove",
    ],
    "put the bowl on the plate": [
        "place the bowl on the plate",
        "set the bowl on top of the plate",
        "move the bowl onto the plate",
    ],
    "put the wine bottle on the rack": [
        "place the wine bottle on the rack",
        "set the wine bottle in the wine rack",
        "move the wine bottle onto the rack",
    ],
    # libero_10
    "put both the alphabet soup and the tomato sauce in the basket": [
        "place the alphabet soup and the tomato sauce in the basket",
        "move both the tomato sauce and the alphabet soup into the basket",
        "drop the alphabet soup and the tomato sauce into the basket",
    ],
    "put both the cream cheese box and the butter in the basket": [
        "place the cream cheese box and the butter in the basket",
        "move both the butter and the cream cheese box into the basket",
        "drop the cream cheese box and the butter into the basket",
    ],
    "turn on the stove and put the moka pot on it": [
        "switch on the stove and place the moka pot on it",
        "turn the stove on, then put the moka pot on top of it",
        "put the moka pot on the stove after turning the stove on",
    ],
    "put the black bowl in the bottom drawer of the cabinet and close it": [
        "place the black bowl in the cabinet's bottom drawer and then close the drawer",
        "put the black bowl into the lowest drawer of the cabinet and shut it",
        "move the black bowl into the bottom drawer of the cabinet, then close that drawer",
    ],
    "put the white mug on the left plate and put the yellow and white mug on the right plate": [
        "place the white mug on the left plate and the yellow and white mug on the right plate",
        "set the white mug on the plate on the left, and set the yellow and white mug on the plate on the right",
        "put the yellow and white mug on the right plate and the white mug on the left plate",
    ],
    "pick up the book and place it in the back compartment of the caddy": [
        "grab the book and put it in the caddy's back compartment",
        "take the book and place it into the rear compartment of the caddy",
        "put the book in the back section of the caddy",
    ],
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": [
        "place the white mug on the plate and set the chocolate pudding on the right side of the plate",
        "put the white mug onto the plate, then place the chocolate pudding to the plate's right",
        "set the chocolate pudding to the right of the plate and the white mug on the plate",
    ],
    "put both the alphabet soup and the cream cheese box in the basket": [
        "place the alphabet soup and the cream cheese box in the basket",
        "move both the cream cheese box and the alphabet soup into the basket",
        "drop the alphabet soup and the cream cheese box into the basket",
    ],
    "put both moka pots on the stove": [
        "place both moka pots on the stove",
        "put the two moka pots on the stove",
        "set each of the moka pots on the stove",
    ],
    "put the yellow and white mug in the microwave and close it": [
        "place the yellow and white mug in the microwave and close the microwave door",
        "put the yellow and white mug inside the microwave, then shut it",
        "move the yellow and white mug into the microwave and close the door",
    ],
}
OBJECT_PARA = [
    "grab the {x} and put it in the basket",
    "take the {x} and drop it into the basket",
    "put the {x} inside the basket",
]
# paraphrase index -> reviewer memo
REVIEW: dict[tuple[str, int], str] = {
    ("pick up the black bowl on the wooden cabinet and place it on the plate", 1): (
        "'from the top of the wooden cabinet' could be read as the top drawer (task 4) instead of the cabinet top."
    ),
    ("pick up the black bowl next to the cookie box and place it on the plate", 2): (
        "'by the cookie box' is looser than 'next to'; check it does not also fit the bowl on the cookie box."
    ),
    ("put the wine bottle on the rack", 1): (
        "'in the wine rack' vs original 'on the rack': the goal predicate is On(wine_bottle, rack top_region); "
        "check the preposition change is acceptable."
    ),
    ("put the cream cheese in the bowl", 1): (
        "'drop ... into' may suggest releasing from height; goal (cream cheese On bowl) is unchanged."
    ),
    ("put the white mug on the plate and put the chocolate pudding to the right of the plate", 1): (
        "'to the plate's right' - right relative to the plate vs the viewer is as ambiguous as in the original; "
        "check it reads the same."
    ),
    ("put the yellow and white mug in the microwave and close it", 2): (
        "'close the door' (no 'microwave') - only door in the scene is the microwave, but confirm."
    ),
}

# --------------------------------------------------------------------------- object name phrases
NAME = {
    "akita_black_bowl": "black bowl",
    "glazed_rim_porcelain_ramekin": "ramekin",
    "cookies": "cookie box",
    "plate": "plate",
    "alphabet_soup": "alphabet soup",
    "cream_cheese": "cream cheese",
    "salad_dressing": "salad dressing",
    "bbq_sauce": "bbq sauce",
    "ketchup": "ketchup",
    "tomato_sauce": "tomato sauce",
    "butter": "butter",
    "milk": "milk",
    "chocolate_pudding": "chocolate pudding",
    "orange_juice": "orange juice",
}

# libero_spatial: where the *other* bowl (akita_black_bowl_2) is -> in-vocab descriptor
SPATIAL_DESC = {
    "main_table_between_plate_ramekin_region": "between the plate and the ramekin",
    "main_table_next_to_ramekin_region": "next to the ramekin",
    "main_table_next_to_box_region": "next to the cookie box",
    "main_table_next_to_plate_region": "next to the plate",
    "main_table_table_center": "from table center",
    "wooden_cabinet_1_top_side": "on the wooden cabinet",
    "wooden_cabinet_1_top_region": "in the top drawer of the wooden cabinet",
    "cookies_1": "on the cookie box",
    "glazed_rim_porcelain_ramekin_1": "on the ramekin",
    "flat_stove_1_cook_region": "on the stove",
}
# libero_spatial C5 place targets, in preference order: (phrase, BDDL target)
SPATIAL_TARGETS = [
    ("ramekin", "glazed_rim_porcelain_ramekin_1"),
    ("cookie box", "cookies_1"),
    ("stove", "flat_stove_1_cook_region"),
    ("wooden cabinet", "wooden_cabinet_1_top_side"),
]

# --------------------------------------------------------------------------- hand-written swaps
# (id, type, instruction, goal predicates, orig_target, new_target[, note])
GOAL_SWAPS: dict[int, list[tuple]] = {
    0: [
        (
            "C4-a",
            "object",
            "open the top drawer of the cabinet",
            "(Open wooden_cabinet_1_top_region)",
            "wooden_cabinet_1_middle_region",
            "wooden_cabinet_1_top_region",
        ),
    ],
    1: [
        (
            "C4-a",
            "object",
            "put the cream cheese on the stove",
            "(On cream_cheese_1 flat_stove_1_cook_region)",
            "akita_black_bowl_1",
            "cream_cheese_1",
        ),
        (
            "C4-b",
            "object",
            "put the wine bottle on the stove",
            "(On wine_bottle_1 flat_stove_1_cook_region)",
            "akita_black_bowl_1",
            "wine_bottle_1",
        ),
        (
            "C5-a",
            "target",
            "put the bowl in front of the stove",
            "(On akita_black_bowl_1 main_table_stove_front_region)",
            "flat_stove_1_cook_region",
            "main_table_stove_front_region",
        ),
    ],
    2: [
        (
            "C4-a",
            "object",
            "put the cream cheese on top of the cabinet",
            "(On cream_cheese_1 wooden_cabinet_1_top_side)",
            "wine_bottle_1",
            "cream_cheese_1",
        ),
        (
            "C5-a",
            "target",
            "put the wine bottle on the stove",
            "(On wine_bottle_1 flat_stove_1_cook_region)",
            "wooden_cabinet_1_top_side",
            "flat_stove_1_cook_region",
        ),
        (
            "C5-b",
            "target",
            "put the wine bottle in front of the stove",
            "(On wine_bottle_1 main_table_stove_front_region)",
            "wooden_cabinet_1_top_side",
            "main_table_stove_front_region",
        ),
    ],
    3: [
        (
            "C4-a",
            "object",
            "open the top drawer and put the cream cheese inside",
            "(In cream_cheese_1 wooden_cabinet_1_top_region)",
            "akita_black_bowl_1",
            "cream_cheese_1",
        ),
        (
            "C5-a",
            "target",
            "open the middle drawer and put the bowl inside",
            "(In akita_black_bowl_1 wooden_cabinet_1_middle_region)",
            "wooden_cabinet_1_top_region",
            "wooden_cabinet_1_middle_region",
        ),
    ],
    4: [
        (
            "C4-a",
            "object",
            "put the cream cheese on top of the cabinet",
            "(On cream_cheese_1 wooden_cabinet_1_top_side)",
            "akita_black_bowl_1",
            "cream_cheese_1",
        ),
        (
            "C5-a",
            "target",
            "put the bowl in front of the stove",
            "(On akita_black_bowl_1 main_table_stove_front_region)",
            "wooden_cabinet_1_top_side",
            "main_table_stove_front_region",
        ),
    ],
    5: [
        (
            "C4-a",
            "object",
            "push the bowl to the front of the stove",
            "(On akita_black_bowl_1 main_table_stove_front_region)",
            "plate_1",
            "akita_black_bowl_1",
        ),
        (
            "C4-b",
            "object",
            "push the cream cheese to the front of the stove",
            "(On cream_cheese_1 main_table_stove_front_region)",
            "plate_1",
            "cream_cheese_1",
        ),
    ],
    6: [
        (
            "C5-a",
            "target",
            "put the cream cheese on the plate",
            "(On cream_cheese_1 plate_1)",
            "akita_black_bowl_1",
            "plate_1",
        ),
        (
            "C5-b",
            "target",
            "put the cream cheese on the stove",
            "(On cream_cheese_1 flat_stove_1_cook_region)",
            "akita_black_bowl_1",
            "flat_stove_1_cook_region",
        ),
    ],
    7: [],
    8: [
        (
            "C4-a",
            "object",
            "put the cream cheese on the plate",
            "(On cream_cheese_1 plate_1)",
            "akita_black_bowl_1",
            "cream_cheese_1",
        ),
        (
            "C4-b",
            "object",
            "put the wine bottle on the plate",
            "(On wine_bottle_1 plate_1)",
            "akita_black_bowl_1",
            "wine_bottle_1",
        ),
        (
            "C5-a",
            "target",
            "put the bowl in front of the stove",
            "(On akita_black_bowl_1 main_table_stove_front_region)",
            "plate_1",
            "main_table_stove_front_region",
        ),
    ],
    9: [
        (
            "C4-a",
            "object",
            "put the cream cheese on the rack",
            "(On cream_cheese_1 wine_rack_1_top_region)",
            "wine_bottle_1",
            "cream_cheese_1",
        ),
        (
            "C5-a",
            "target",
            "put the wine bottle on the stove",
            "(On wine_bottle_1 flat_stove_1_cook_region)",
            "wine_rack_1_top_region",
            "flat_stove_1_cook_region",
        ),
        (
            "C5-b",
            "target",
            "put the wine bottle on the plate",
            "(On wine_bottle_1 plate_1)",
            "wine_rack_1_top_region",
            "plate_1",
        ),
    ],
}
GOAL_NOTES = {
    0: "Only one in-vocab swap: the other drawer reachable with 'open' that appears in training is the top one "
    "('bottom' is not in the libero_goal vocabulary).",
    7: "No swap: 'turn on' only ever applies to the stove in the libero_goal vocabulary, so there is no in-vocab "
    "object or target to swap to.",
}

L10_SWAPS: dict[int, list[tuple]] = {
    0: [
        (
            "C4-a",
            "object",
            "put both the alphabet soup and the butter in the basket",
            "(In alphabet_soup_1 basket_1_contain_region) (In butter_1 basket_1_contain_region)",
            "tomato_sauce_1",
            "butter_1",
        ),
        (
            "C4-b",
            "object",
            "put both the butter and the tomato sauce in the basket",
            "(In butter_1 basket_1_contain_region) (In tomato_sauce_1 basket_1_contain_region)",
            "alphabet_soup_1",
            "butter_1",
        ),
    ],
    1: [
        (
            "C4-a",
            "object",
            "put both the cream cheese box and the tomato sauce in the basket",
            "(In cream_cheese_1 basket_1_contain_region) (In tomato_sauce_1 basket_1_contain_region)",
            "butter_1",
            "tomato_sauce_1",
        ),
        (
            "C4-b",
            "object",
            "put both the alphabet soup and the butter in the basket",
            "(In alphabet_soup_1 basket_1_contain_region) (In butter_1 basket_1_contain_region)",
            "cream_cheese_1",
            "alphabet_soup_1",
        ),
    ],
    2: [],
    3: [],
    4: [
        (
            "C5-a",
            "target",
            "put the white mug on the right plate and put the yellow and white mug on the left plate",
            "(On porcelain_mug_1 plate_2) (On white_yellow_mug_1 plate_1)",
            "plate_1,plate_2",
            "plate_2,plate_1",
            "Both steps change: moving only one mug onto the other plate would put both mugs on one plate. "
            "Plates are exchanged, objects and verbs are unchanged.",
        ),
    ],
    5: [
        (
            "C4-a",
            "object",
            "pick up the yellow and white mug and place it in the back compartment of the caddy",
            "(In white_yellow_mug_1 desk_caddy_1_back_contain_region)",
            "black_book_1",
            "white_yellow_mug_1",
        ),
        (
            "C5-a",
            "target",
            "pick up the book and place it in the left compartment of the caddy",
            "(In black_book_1 desk_caddy_1_left_contain_region)",
            "desk_caddy_1_back_contain_region",
            "desk_caddy_1_left_contain_region",
        ),
        (
            "C5-b",
            "target",
            "pick up the book and place it in the right compartment of the caddy",
            "(In black_book_1 desk_caddy_1_right_contain_region)",
            "desk_caddy_1_back_contain_region",
            "desk_caddy_1_right_contain_region",
        ),
    ],
    6: [
        (
            "C5-a",
            "target",
            "put the white mug on the plate and put the chocolate pudding to the left of the plate",
            "(On porcelain_mug_1 plate_1) (On chocolate_pudding_1 living_room_table_plate_left_region)",
            "living_room_table_plate_right_region",
            "living_room_table_plate_left_region",
            "Only the second step changes (right -> left of the plate).",
        ),
    ],
    7: [
        (
            "C4-a",
            "object",
            "put both the tomato sauce and the cream cheese box in the basket",
            "(In tomato_sauce_1 basket_1_contain_region) (In cream_cheese_1 basket_1_contain_region)",
            "alphabet_soup_1",
            "tomato_sauce_1",
        ),
        (
            "C4-b",
            "object",
            "put both the alphabet soup and the tomato sauce in the basket",
            "(In alphabet_soup_1 basket_1_contain_region) (In tomato_sauce_1 basket_1_contain_region)",
            "cream_cheese_1",
            "tomato_sauce_1",
            "Sentence is verbatim the training instruction of task 0 (LIVING_ROOM_SCENE2); here it is asked in "
            "LIVING_ROOM_SCENE1, where both objects also exist.",
        ),
    ],
    8: [],
    9: [
        (
            "C4-a",
            "object",
            "put the white mug in the microwave and close it",
            "(In porcelain_mug_1 microwave_1_heating_region) (Close microwave_1)",
            "white_yellow_mug_1",
            "porcelain_mug_1",
        ),
    ],
}
L10_NOTES = {
    2: "No swap: the only other object (frying pan) has no name in the libero_10 vocabulary, and the stove is "
    "the only place the vocabulary can refer to in this scene.",
    3: "No swap: the wine bottle ('wine' is not in the libero_10 vocabulary) is the only other object and the other "
    "drawers ('top'/'middle') are not in the vocabulary either.",
    8: "No swap: the scene only contains the two moka pots and the stove; nothing else is nameable in vocabulary.",
}


# --------------------------------------------------------------------------- BDDL helpers
def bddl_root() -> Path:
    from libero.libero import get_libero_path

    return Path(get_libero_path("bddl_files"))


def section(txt: str, name: str) -> tuple[int, int]:
    i = txt.find("(:" + name)
    if i < 0:
        raise KeyError(name)
    depth = 0
    for j in range(i, len(txt)):
        if txt[j] == "(":
            depth += 1
        elif txt[j] == ")":
            depth -= 1
            if depth == 0:
                return i, j + 1
    raise ValueError(name)


def init_facts(txt: str) -> list[list[str]]:
    a, b = section(txt, "init")
    return [m.split() for m in re.findall(r"\(([A-Za-z]+ [^()]+)\)", txt[a:b])]


def rewrite(txt: str, goal: str, language: str) -> str:
    a, b = section(txt, "goal")
    txt = txt[:a] + f"(:goal\n    (And {goal})\n  )" + txt[b:]
    a, b = section(txt, "language")
    return txt[:a] + f"(:language {language})" + txt[b:]


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z]+", s.lower())


# --------------------------------------------------------------------------- per-suite swaps
def spatial_swaps(txt: str) -> tuple[list[tuple[str, ...]], str | None]:
    facts = init_facts(txt)
    loc = {f[1]: f[2] for f in facts if f[0] in ("On", "In")}
    d1 = SPATIAL_DESC[loc["akita_black_bowl_1"]]
    d2 = SPATIAL_DESC[loc["akita_black_bowl_2"]]
    swaps: list[tuple[str, ...]] = [
        (
            "C4-a",
            "object",
            f"pick up the black bowl {d2} and place it on the plate",
            "(On akita_black_bowl_2 plate_1)",
            "akita_black_bowl_1",
            "akita_black_bowl_2",
            f"Refers to the other black bowl, which in this layout is '{d2}'. The sentence is a training instruction "
            "of another libero_spatial task (other layout).",
        ),
    ]
    occupied = {v for k, v in loc.items() if k.startswith("akita_black_bowl")}
    if "glazed_rim_porcelain_ramekin_1" not in occupied:
        swaps.append(
            (
                "C4-b",
                "object",
                "pick up the ramekin and place it on the plate",
                "(On glazed_rim_porcelain_ramekin_1 plate_1)",
                "akita_black_bowl_1",
                "glazed_rim_porcelain_ramekin_1",
            )
        )
    n = 0
    for phrase, tgt in SPATIAL_TARGETS:
        if tgt in occupied or n == 2:
            continue
        swaps.append(
            (
                f"C5-{'ab'[n]}",
                "target",
                f"pick up the black bowl {d1} and place it on the {phrase}",
                f"(On akita_black_bowl_1 {tgt})",
                "plate_1",
                tgt,
            )
        )
        n += 1
    note = None
    if "glazed_rim_porcelain_ramekin_1" in occupied:
        note = "No ramekin object swap: a black bowl sits on the ramekin in this layout."
    return swaps, note


def object_swaps(txt: str, target_obj: str) -> list[tuple]:
    a, b = section(txt, "objects")
    objs = [w for w in txt[a:b].split() if w.endswith("_1")]
    others = [o for o in objs if o not in (target_obj, "basket_1")][:3]
    out = []
    for k, o in enumerate(others):
        phrase = NAME[o[:-2]]
        out.append(
            (
                f"C4-{'abc'[k]}",
                "object",
                f"pick up the {phrase} and place it in the basket",
                f"(In {o} basket_1_contain_region)",
                target_obj,
                o,
            )
        )
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    from libero.libero import benchmark

    root = bddl_root()
    bd = benchmark.get_benchmark_dict()
    INSTR_DIR.mkdir(parents=True, exist_ok=True)
    n_review = 0
    for si, suite in enumerate(SUITES):
        ts = bd[suite]()
        tasks = [ts.get_task(i) for i in range(ts.n_tasks)]
        train = [t.language for t in tasks]
        vocab = sorted({w for s in train for w in tokens(s)})
        vset = set(vocab)
        (INSTR_DIR / f"vocab_{suite}.json").write_text(
            json.dumps(
                {
                    "suite": suite,
                    "source": "task.language of the 10 training tasks",
                    "instructions": train,
                    "tokens": vocab,
                },
                indent=2,
            )
            + "\n"
        )
        old_path = INSTR_DIR / f"{suite}.json"
        old = json.loads(old_path.read_text()) if old_path.exists() else {"tasks": []}
        old_swaps = {(t["task_id"], s["id"]): s for t in old["tasks"] for s in t["swaps"]}
        old_tasks = {t["task_id"]: t for t in old["tasks"]}

        out_dir = BDDL_DIR / suite
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("*.bddl"):
            f.unlink()
        entries = []
        for tid, task in enumerate(tasks):
            src = root / task.problem_folder / task.bddl_file
            txt = src.read_text()
            orig = task.language
            note = None
            if suite == "libero_spatial":
                swaps, note = spatial_swaps(txt)
                paras = PARA[orig]
            elif suite == "libero_object":
                a, b = section(txt, "goal")
                found = re.search(r"\(In (\S+) ", txt[a:b])
                assert found is not None, src
                target_obj = found.group(1)
                swaps = object_swaps(txt, target_obj)
                note = "No C5: 'basket' is the only place word in the libero_object vocabulary."
                paras = [p.format(x=NAME[target_obj[:-2]]) for p in OBJECT_PARA]
            elif suite == "libero_goal":
                swaps = GOAL_SWAPS[tid]
                note = GOAL_NOTES.get(tid)
                paras = PARA[orig]
            else:
                swaps = L10_SWAPS[tid]
                note = L10_NOTES.get(tid)
                paras = PARA[orig]

            assert len(paras) == 3 and len(set(paras)) == 3 and orig not in paras, orig
            para_entries = []
            for k, p in enumerate(paras):
                memo = REVIEW.get((orig, k))
                para_entries.append({"text": p, "review": memo} if memo else {"text": p})
                n_review += memo is not None

            sw_entries = []
            for sw in swaps:
                sid, typ, instr, goal, orig_t, new_t = sw[:6]
                bad = [w for w in tokens(instr) if w not in vset]
                assert not bad, f"{suite} t{tid} {sid}: out-of-vocab words {bad} in '{instr}'"
                assert instr != orig, (suite, tid, sid)
                rel = BDDL_DIR.relative_to(REPO) / suite / f"{src.stem}__{sid}.bddl"
                (REPO / rel).write_text(rewrite(txt, goal, instr))
                e = {
                    "id": sid,
                    "type": typ,
                    "instruction": instr,
                    "goal": f"(And {goal})",
                    "goal_bddl": str(rel),
                    "feasible": None,
                    "orig_target": orig_t,
                    "new_target": new_t,
                }
                if len(sw) > 6:
                    e["note"] = sw[6]
                prev = old_swaps.get((tid, sid))
                if prev and prev["instruction"] == instr and prev["goal"] == e["goal"]:
                    for key in ("feasible", "feasible_trials", "feasible_note"):
                        if key in prev:
                            e[key] = prev[key]
                sw_entries.append(e)

            # libero_10 tasks 0/1 share LIVING_ROOM_SCENE2 -> pair them; otherwise the next task id
            cross = [1 - tid] if suite == "libero_10" and tid in (0, 1) else [(tid + 1) % 10]
            entry = {
                "task_id": tid,
                "bddl": str(Path(task.problem_folder) / task.bddl_file),
                "original": orig,
                "paraphrases": [p["text"] for p in para_entries],
                "review": {str(k): p["review"] for k, p in enumerate(para_entries) if "review" in p},
                "nonsense_seed": si * 100 + tid,
                "swaps": sw_entries,
                "cross_task": cross,
            }
            if suite != "libero_goal" and not (suite == "libero_10" and tid in (0, 1)):
                entry["cross_task_note"] = (
                    "Different scene: the cross-task sentence describes another layout."
                )
            elif suite == "libero_10":
                entry["cross_task_note"] = "Same scene (LIVING_ROOM_SCENE2) as the cross task."
            if note:
                entry["note"] = note
            prev_t = old_tasks.get(tid)
            if prev_t:
                for key in ("scripted_c0_success", "scripted_c0_trials"):
                    if key in prev_t:
                        entry[key] = prev_t[key]
            entries.append(entry)

        spec = {
            "suite": suite,
            "bddl_root": "libero/libero/bddl_files (hf-libero package)",
            "vocab": f"scripts/shortcut/instructions/vocab_{suite}.json",
            "tasks": entries,
        }
        old_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
        n_sw = sum(len(e["swaps"]) for e in entries)
        print(f"{suite}: {len(entries)} tasks, {n_sw} swaps, vocab {len(vocab)} words")
    print(f"paraphrases flagged for human review: {n_review}")


if __name__ == "__main__":
    os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    main()
