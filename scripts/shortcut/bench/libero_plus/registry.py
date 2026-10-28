"""LIBERO-Plus task registry → standard perturbation records (pure Python: main venv and the Plus venv).

LIBERO-Plus registers every perturbed variant of the 40 original LIBERO tasks as its own task
(`libero_suite_task_map.py`, 2402–2591 per suite); `task_classification.json` gives each registered task
(id = registry index + 1) its category (one of 7 axes) and difficulty_level (1–5; null for 121 Goal
light variants). The variant name = original task name + a suffix that encodes the perturbation:

  axis (category)                 dimension  suffix                                   detail
  Language Instructions           language   _language_<k>_view_0_0_100_0_0_initstate_0  rewrite
  Objects Layout                  layout     _add_<k>                                 add (confounding objects)
                                             [_moved]_level<l>_sample<s>              moved (object positions)
  Camera Viewpoints               camera     _view_<h>_<v>_<scale>_<rot>_<vert>_initstate_0  view
  Robot Initial States            robot      _view_0_0_100_0_0_initstate_<k>          initstate
  Light Conditions                lighting   _light_<k>                               light
  Background Textures             texture    _table_<k> | _tb_<k>                     table | tb
  Sensor Noise                    noise      _view_0_0_100_0_0_initstate_0_noise_<k>  motion_blur (1–10),
                                                                                      gaussian_blur, zoom_blur,
                                                                                      fog, glass_blur (…–50)

The clean baseline (dimension "none", detail "original") is the original task's own BDDL run in the
Plus env. Every registered variant keeps the original task's goal predicates (checked against the
BDDL files, see `goal_matches_original`), so `success` is the original goal judged in the perturbed
scene.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
# vanilla LIBERO task order (hf-libero libero_suite_task_map): task_id here = task_id in ours / HuggingFaceVLA/libero
ORIGINAL_TASKS: dict[str, list[str]] = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ],
}
CATEGORY_DIMENSION = {
    "Language Instructions": "language",
    "Objects Layout": "layout",
    "Camera Viewpoints": "camera",
    "Robot Initial States": "robot",
    "Light Conditions": "lighting",
    "Background Textures": "texture",
    "Sensor Noise": "noise",
}
AXES = ("none", *CATEGORY_DIMENSION.values())
NOISE_KINDS = ("motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur")  # noise 1–10, 11–20, ...
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}  # = lerobot
# what each axis takes away from the policy (gallery captions, docs)
AXIS_CUE = {
    "none": "교란 없음: Plus env 에서 원래 task (기준선)",
    "language": "지시문 표면형을 바꿔 (LLM 재작성, 군더더기 문맥) 학습 지시문 문자열 ↔ 동작 암기를 끊는다",
    "layout": "물체 추가·위치 이동으로 '장면 배치 → 궤적' 암기를 끊는다 (목표 물체는 같다)",
    "camera": "카메라 시점을 바꿔 픽셀 위치 ↔ 동작 대응을 끊는다",
    "robot": "로봇 초기 자세를 바꿔 고정 시작 자세에서 외운 궤적을 끊는다",
    "lighting": "조명 세기·방향·색을 바꿔 밝기·그림자 단서를 끊는다",
    "texture": "테이블·배경 질감을 바꿔 배경 외형 단서를 끊는다",
    "noise": "센서 노이즈 (흐림, 안개, 유리 왜곡) 로 세밀한 픽셀 단서를 끊는다 (agentview 만)",
}


def language_from_name(name: str) -> str:
    """LIBERO's instruction-from-filename rule (vanilla grab_language_from_filename)."""
    if name[0].isupper():  # LIBERO-100 scene prefix
        return " ".join(name[name.find("SCENE") + (8 if "SCENE10" in name else 7) :].split("_"))
    return " ".join(name.split("_"))


@dataclass(frozen=True)
class PlusVariant:
    suite: str
    index: int  # registry index in libero_task_map[suite] (task_classification id − 1); −1 = original
    name: str  # registered task name (BDDL stem, perturbation suffix included)
    task_id: int  # original task (vanilla order)
    dimension: str
    level: int | None  # difficulty_level
    detail: str  # sub-type, see module docstring
    params: dict[str, Any]

    @property
    def variant_id(self) -> str:
        return "orig" if self.index < 0 else f"p{self.index:04d}"

    def perturbation(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "level": self.level, "detail": self.detail}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def original_task_id(suite: str, name: str) -> int:
    """Longest original task name that `name` starts with (followed by '_' or the end)."""
    best, best_len = -1, -1
    for i, orig in enumerate(ORIGINAL_TASKS[suite]):
        if (name == orig or name.startswith(orig + "_")) and len(orig) > best_len:
            best, best_len = i, len(orig)
    if best < 0:
        raise ValueError(f"{suite}: {name!r} does not extend any original task")
    return best


def parse_variant(suite: str, index: int, name: str, category: str, level: int | None) -> PlusVariant:
    task_id = original_task_id(suite, name)
    dim = CATEGORY_DIMENSION[category]
    suffix = name[len(ORIGINAL_TASKS[suite][task_id]) :]
    params: dict[str, Any] = {}
    detail = dim
    if dim == "language":
        params["k"] = int(re.search(r"_language_(\d+)", suffix).group(1))  # type: ignore[union-attr]
        detail = "rewrite"
    elif dim == "layout":
        if m := re.fullmatch(r"_add_(\d+)", suffix):
            detail, params["k"] = "add", int(m.group(1))
        elif m := re.fullmatch(r"(_moved)?_level(\d+)_sample(\d+)", suffix):
            detail = "moved"
            params.update(level=int(m.group(2)), sample=int(m.group(3)))
        else:
            raise ValueError(f"layout variant {name!r}: unknown suffix {suffix!r}")
    elif dim in ("camera", "robot", "noise"):
        m = re.fullmatch(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate_(\d+)(?:_noise_(\d+))?", suffix)
        if not m:
            raise ValueError(f"{dim} variant {name!r}: unknown suffix {suffix!r}")
        h, v, scale, rot, vert, init, noise = m.groups()
        if dim == "camera":
            detail = "view"
            params.update(horizontal=int(h), vertical=int(v), scale=int(scale), rot=int(rot), vert=int(vert))
        elif dim == "robot":
            detail, params["initstate"] = "initstate", int(init)
        else:
            k = int(noise)
            detail, params["noise"] = NOISE_KINDS[min((k - 1) // 10, 4)], k
    elif dim == "lighting":
        detail, params["k"] = "light", int(re.fullmatch(r"_light_(\d+)", suffix).group(1))  # type: ignore[union-attr]
    elif dim == "texture":
        m = re.fullmatch(r"_(table|tb)_(\d+)", suffix)
        if not m:
            raise ValueError(f"texture variant {name!r}: unknown suffix {suffix!r}")
        detail, params["k"] = m.group(1), int(m.group(2))
    return PlusVariant(suite, index, name, task_id, dim, level, detail, params)


def original_variant(suite: str, task_id: int) -> PlusVariant:
    return PlusVariant(suite, -1, ORIGINAL_TASKS[suite][task_id], task_id, "none", None, "original", {})


def classification_path(src: Path) -> Path:
    return Path(src) / "libero" / "libero" / "benchmark" / "task_classification.json"


def load_variants(src: Path, suite: str) -> list[PlusVariant]:
    """All registered variants of a suite, from <LIBERO-plus src>/…/task_classification.json."""
    data = json.loads(classification_path(src).read_text())
    return [
        parse_variant(suite, int(x["id"]) - 1, x["name"], x["category"], x.get("difficulty_level"))
        for x in data[suite]
    ]


def motion_blur(v: PlusVariant) -> bool:
    return v.detail == "motion_blur"  # the only path that calls ImageMagick (Wand)


def select(
    variants: list[PlusVariant],
    *,
    dimension: str,
    tasks: list[int],
    n: int | str = "all",
    levels: list[int | None] | None = None,
    details: list[str] | None = None,
    seed: int = 0,
) -> dict[int, list[PlusVariant]]:
    """Variants per original task: the axis' variants (optionally only these levels / sub-types); `n` per
    task, stratified over levels (round-robin from the lowest level, a seeded shuffle inside each level),
    or every one with n="all". Deterministic for a given (seed, registry)."""
    out: dict[int, list[PlusVariant]] = {}
    for t in tasks:
        pool = [
            v
            for v in variants
            if v.task_id == t
            and v.dimension == dimension
            and (levels is None or v.level in levels)
            and (details is None or v.detail in details)
        ]
        if n == "all":
            out[t] = sorted(pool, key=lambda v: v.index)
            continue
        by_level: dict[Any, list[PlusVariant]] = {}
        for v in sorted(pool, key=lambda v: v.index):
            by_level.setdefault(v.level, []).append(v)
        for lv, vs in by_level.items():
            random.Random(f"{seed}/{vs[0].suite}/{dimension}/{t}/{lv}").shuffle(vs)
        order = sorted(by_level, key=lambda x: (x is None, x if x is not None else 0))
        picked: list[PlusVariant] = []
        while len(picked) < int(n) and any(by_level[lv] for lv in order):
            for lv in order:
                if by_level[lv] and len(picked) < int(n):
                    picked.append(by_level[lv].pop(0))
        out[t] = sorted(picked, key=lambda v: (v.level is None, v.level or 0, v.index))
    return out


def bddl_goal(path: Path) -> str:
    """Normalized (:goal ...) section of a BDDL file."""
    text = Path(path).read_text()
    i = text.find("(:goal")
    if i < 0:
        raise ValueError(f"{path}: no (:goal")
    return re.sub(r"\s+", " ", text[i:]).strip()


def variant_bddl(src: Path, v: PlusVariant) -> Path:
    """The BDDL the Plus env actually loads for a variant (ControlEnv strips `_view_…`)."""
    stem = v.name.split("_view_")[0] if "_view_" in v.name and "_initstate_" in v.name else v.name
    return Path(src) / "libero" / "libero" / "bddl_files" / v.suite / f"{stem}.bddl"


def goal_matches_original(src: Path, v: PlusVariant) -> bool:
    orig = original_variant(v.suite, v.task_id)
    return bddl_goal(variant_bddl(src, v)) == bddl_goal(variant_bddl(src, orig))
