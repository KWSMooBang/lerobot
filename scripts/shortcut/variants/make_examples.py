"""Write the shipped example specs (run once; the YAMLs are committed, edit those rather than this).

libero_goal/gNN_human, gNN_train (NN = 01…17 without 06, 12)  goal15: the human-written Goal variants (lead, 2026-11-10),
                     each in the human sentence (BDDL :language style) and translated to the training-data style
                     (HuggingFaceVLA/libero task strings); init keep. GOAL15 below is the translation table.
libero_goal/ctl_tNN_train, ctl_tNN_bddl  goal15 control: each training task with its training sentence (= C0)
                     and with its BDDL :language sentence (wording effect only); init keep, goal = the task goal.
libero_goal/n01…n23  new-instruction candidates N1–N4 of tasks/refs/goal-novel-candidates.md
                     (#18 close / #19 turn off start from an open drawer / a lit stove: fixture_state + patch).
                     base_task = the closest training task (pairing with its C0 episodes).
libero_object/c6_tNN  T014 C6: instruction of a trained object that is not in this scene and was trained at the
                      other target spot (A ↔ B); scene and init unchanged; no achievable goal.
libero_object/c7_tNN  T014 C7: target object and the object on the other target spot swap places (patch);
                      instruction and goal = the original task.
libero_object/c8_tNN  T014 C8: target object removed (patch); original instruction; no achievable goal.
Object spots: A = target_object_region of tasks 0,2,4,6,8 … (-0.12, -0.24), B = (0.05, -0.10); in every scene
other_object_region_0 is the other spot.

    uv run --extra libero python scripts/shortcut/variants/make_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(HERE.parent))

from variants.bddl_io import Problem  # noqa: E402

B, C, W, P = "akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1"
COOK, TOP_SIDE, RACK, FRONT = (
    "flat_stove_1_cook_region",
    "wooden_cabinet_1_top_side",
    "wine_rack_1_top_region",
    "main_table_stove_front_region",
)
DRAWER = {k: f"wooden_cabinet_1_{k}_region" for k in ("top", "middle", "bottom")}
STOVE = "flat_stove_1"

# (#, group, id slug, instruction, goal, base task, extra spec keys)
GOAL = [
    (1, "N1", "cheese_on_plate", "put the cream cheese on the plate", [f"On {C} {P}"], 8, {}),
    (2, "N1", "wine_on_plate", "put the wine bottle on the plate", [f"On {W} {P}"], 8, {}),
    (3, "N1", "cheese_on_stove", "put the cream cheese on the stove", [f"On {C} {COOK}"], 1, {}),
    (4, "N1", "wine_on_stove", "put the wine bottle on the stove", [f"On {W} {COOK}"], 1, {}),
    (5, "N1", "cheese_on_cabinet", "put the cream cheese on top of the cabinet", [f"On {C} {TOP_SIDE}"], 4, {}),
    (6, "N1", "cheese_on_rack", "put the cream cheese on the rack", [f"On {C} {RACK}"], 9, {}),
    (7, "N1", "bowl_front_of_stove", "put the bowl in front of the stove", [f"On {B} {FRONT}"], 5, {}),
    (8, "N1", "wine_front_of_stove", "put the wine bottle in front of the stove", [f"On {W} {FRONT}"], 5, {}),
    (9, "N1", "open_top_drawer", "open the top drawer of the cabinet", [f"Open {DRAWER['top']}"], 0, {}),
    (10, "N1", "middle_drawer_bowl", "open the middle drawer and put the bowl inside", [f"In {B} {DRAWER['middle']}"], 3, {}),
    (11, "N1", "top_drawer_cheese", "open the top drawer and put the cream cheese inside", [f"In {C} {DRAWER['top']}"], 3, {}),
    (12, "N2", "open_bottom_drawer", "open the bottom drawer of the cabinet", [f"Open {DRAWER['bottom']}"], 0, {}),
    (13, "N2", "bottom_drawer_bowl", "open the bottom drawer and put the bowl inside", [f"In {B} {DRAWER['bottom']}"], 3, {}),
    (14, "N2", "middle_drawer_cheese", "open the middle drawer and put the cream cheese inside", [f"In {C} {DRAWER['middle']}"], 3, {}),
    (15, "N3", "pick_up_cheese", "pick up the cream cheese", [f"Up {C}"], 6, {}),
    (16, "N3", "pick_up_wine", "pick up the wine bottle", [f"Up {W}"], 2, {}),
    (17, "N3", "plate_on_stove", "put the plate on the stove", [f"On {P} {COOK}"], 1, {}),
    (
        18, "N3", "close_top_drawer", "close the top drawer of the cabinet", [f"Close {DRAWER['top']}"], 3,
        {"scene": {"fixture_state": {DRAWER["top"]: "open"}}, "init": "patch"},
    ),
    (
        19, "N3", "turn_off_stove", "turn off the stove", [f"TurnOff {STOVE}"], 7,
        {"scene": {"fixture_state": {STOVE: "on"}}, "init": "patch"},
    ),
    (20, "N4", "bowl_plate_and_stove_on", "put the bowl on the plate and turn on the stove", [f"On {B} {P}", f"TurnOn {STOVE}"], 8, {}),
    (21, "N4", "wine_rack_and_bowl_stove", "put the wine bottle on the rack and put the bowl on the stove", [f"On {W} {RACK}", f"On {B} {COOK}"], 9, {}),
    (22, "N4", "middle_drawer_and_stove_on", "open the middle drawer and turn on the stove", [f"Open {DRAWER['middle']}", f"TurnOn {STOVE}"], 0, {}),
    # goal is the end state; listed bowl→plate first so the scripted check moves the bowl before filling it
    (23, "N4", "cheese_bowl_and_bowl_plate", "put the cream cheese in the bowl and put the bowl on the plate", [f"On {B} {P}", f"On {C} {B}"], 6, {}),
]  # fmt: skip


# goal15 (lead messages 2026-11-10; g06 / g12 = training tasks, dropped; ids kept): (#, human sentence, training-style sentence, goal, base task, note).
# Training style = the HuggingFaceVLA/libero task strings (instructions/vocab_libero_goal.json): "drawer of the
# cabinet", "top / middle / bottom drawer", "on top of the cabinet", BDDL "on the bowl" ↔ training "in the bowl",
# lower case (table set by the lead).
GOAL15 = [
    (1, "Open the bottom layer of the drawer", "open the bottom drawer of the cabinet", [f"Open {DRAWER['bottom']}"], 0, ""),
    (2, "Open the top layer of the drawer", "open the top drawer of the cabinet", [f"Open {DRAWER['top']}"], 0, ""),
    (3, "Put the wine bottle on the bowl", "put the wine bottle in the bowl", [f"On {W} {B}"], 2, "may be physically infeasible"),
    (4, "Put the wine bottle on the plate", "put the wine bottle on the plate", [f"On {W} {P}"], 2, ""),
    (5, "Put the wine bottle on the stove", "put the wine bottle on the stove", [f"On {W} {COOK}"], 2, ""),
    (7, "Open the top layer of the drawer and put the wine bottle inside", "open the top drawer and put the wine bottle inside",
     [f"In {W} {DRAWER['top']}"], 3, "may be physically infeasible (bottle in a drawer)"),
    (8, "Open the top layer of the drawer and put the cream cheese inside", "open the top drawer and put the cream cheese inside",
     [f"In {C} {DRAWER['top']}"], 3, ""),
    (9, "Open the middle layer of the drawer and put the bowl inside", "open the middle drawer and put the bowl inside",
     [f"In {B} {DRAWER['middle']}"], 3, ""),
    (10, "Open the middle layer of the drawer and put the wine bottle inside", "open the middle drawer and put the wine bottle inside",
     [f"In {W} {DRAWER['middle']}"], 3, "may be physically infeasible (bottle in a drawer)"),
    (11, "Open the middle layer of the drawer and put the cream cheese inside", "open the middle drawer and put the cream cheese inside",
     [f"In {C} {DRAWER['middle']}"], 3, ""),
    (13, "Put the bowl on the rack", "put the bowl on the rack", [f"On {B} {RACK}"], 9, "may be physically infeasible"),
    (14, "Put the cream cheese on the plate", "put the cream cheese on the plate", [f"On {C} {P}"], 8, ""),
    (15, "Put the cream cheese on the stove", "put the cream cheese on the stove", [f"On {C} {COOK}"], 1, ""),
    (16, "Put the cream cheese on the rack", "put the cream cheese on the rack", [f"On {C} {RACK}"], 9, ""),
    (17, "Put the cream cheese on top of the drawer", "put the cream cheese on top of the cabinet", [f"On {C} {TOP_SIDE}"], 4, ""),
]  # fmt: skip


def goal15_examples() -> None:
    from variants.generate import suite_tasks

    for num, human, train, goal, base, note in GOAL15:
        for phr, instr in (("bddl", human), ("train", train)):
            vid = f"g{num:02d}_{'human' if phr == 'bddl' else phr}"  # ids gNN_human / gNN_train; phrasing bddl | train
            spec = {"suite": "libero_goal", "base_task": base, "id": vid, "group": "goal15", "pair": f"g{num:02d}",
                    "phrasing": phr, "instruction": instr, "goal": goal, "init": "keep"}  # fmt: skip
            if note:
                spec["notes"] = note
            dump(HERE / "libero_goal" / f"{vid}.yaml", spec, f"goal15 #{num} ({phr} sentence)")
    for t in suite_tasks("libero_goal"):
        p = Problem.read(t["bddl"])
        for phr, instr in (("train", t["language"]), ("bddl", p.language)):
            vid = f"ctl_t{t['task_id']:02d}_{phr}"
            spec = {"suite": "libero_goal", "base_task": t["task_id"], "id": vid, "group": "goal15-control",
                    "pair": f"ctl_t{t['task_id']:02d}", "phrasing": phr, "instruction": instr,
                    "goal": [" ".join(g) for g in p.goal], "init": "keep"}  # fmt: skip
            dump(
                HERE / "libero_goal" / f"{vid}.yaml",
                spec,
                f"goal15 control: training task {t['task_id']} ({phr} sentence)",
            )


def dump(path: Path, spec: dict, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=120)
    path.write_text(f"# {header}\n" + text)


def goal_examples() -> None:
    for num, group, slug, instr, goal, base, extra in GOAL:
        vid = f"n{num:02d}_{slug}"
        spec = {
            "suite": "libero_goal",
            "base_task": base,
            "id": vid,
            "group": group,
            "instruction": instr,
            "goal": goal,
        }
        spec.update(extra)
        spec.setdefault("init", "keep")
        dump(HERE / "libero_goal" / f"{vid}.yaml", spec, f"goal-novel-candidates.md #{num} ({group})")


def object_examples() -> None:
    from variants.generate import suite_tasks

    tasks = suite_tasks("libero_object")
    probs = [Problem.read(t["bddl"]) for t in tasks]
    target = [p.goal[0][1] for p in probs]
    spot = [tuple(round(v, 3) for v in p.regions["floor_target_object_region"]["ranges"][0]) for p in probs]
    for k, (t, p) in enumerate(zip(tasks, probs, strict=True)):
        other = next(o for o in p.objects if p.placement(o) == ["On", o, "floor_other_object_region_0"])
        common = {"suite": "libero_object", "base_task": k}
        absent = [j for j in range(len(tasks)) if spot[j] != spot[k] and target[j] not in p.objects]
        j = absent[0]
        dump(
            HERE / "libero_object" / f"c6_t{k:02d}.yaml",
            {**common, "id": f"c6_t{k:02d}", "group": "C6", "instruction": tasks[j]["language"], "goal": [], "init": "keep",
             "notes": f"{target[j]} is not in this scene; trained (task {j}) at the other target spot, where {other} stands here"},
            f"T014 C6: absent object trained at the other spot (task {j}); goal none",
        )  # fmt: skip
        dump(
            HERE / "libero_object" / f"c7_t{k:02d}.yaml",
            {**common, "id": f"c7_t{k:02d}", "group": "C7", "instruction": t["language"],
             "goal": [" ".join(g) for g in p.goal], "scene": {"swap": [[target[k], other]]}, "init": "patch",
             "notes": f"{target[k]} and {other} (the other target spot) swap places"},
            "T014 C7: the two target spots swap objects; original instruction and goal",
        )  # fmt: skip
        dump(
            HERE / "libero_object" / f"c8_t{k:02d}.yaml",
            {**common, "id": f"c8_t{k:02d}", "group": "C8", "instruction": t["language"], "goal": [],
             "scene": {"remove": [target[k]]}, "init": "patch",
             "notes": f"{target[k]} removed; which object goes into the basket instead?"},
            "T014 C8: target object removed; original instruction; goal none",
        )  # fmt: skip


if __name__ == "__main__":
    goal15_examples()
    goal_examples()
    object_examples()
    print(
        "written:",
        len(list((HERE / "libero_goal").glob("*.yaml"))),
        "goal,",
        len(list((HERE / "libero_object").glob("*.yaml"))),
        "object",
    )
