"""CPU checks for user-designed variants (T020): spec validation, BDDL edits, multi-goal tracking (mock),
custom jobs through run_eval / sc-bench / sc-report, catalog; with LIBERO installed also the generator
(patch / keep), its state checks, the feasibility check and a few real env steps with goal tracking.

    uv run --extra libero --extra shortcut-bench pytest scripts/shortcut/tests/test_variants.py -q
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parent
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "parallel")]

import run_eval  # noqa: E402
from variants.bddl_io import Problem, parse, parse_predicate  # noqa: E402
from variants.fallback import fallback_rows  # noqa: E402
from variants.spec import SpecError, list_variants, parse_spec  # noqa: E402

GOAL_BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Put the bowl on the plate)
    (:regions
      (plate_region (:target main_table) (:ranges ((0.04 -0.03 0.06 -0.01))))
      (bowl_region (:target main_table) (:ranges ((-0.1 -0.01 -0.08 0.01))))
      (top_region (:target wooden_cabinet_1))
      (contain_region (:target basket_1))
    )
  (:fixtures main_table - table wooden_cabinet_1 - wooden_cabinet)
  (:objects akita_black_bowl_1 - akita_black_bowl plate_1 - plate basket_1 - basket)
  (:obj_of_interest akita_black_bowl_1 plate_1)
  (:init (On akita_black_bowl_1 main_table_bowl_region) (On plate_1 main_table_plate_region)
         (In plate_1 basket_1_contain_region))
  (:goal (And (On akita_black_bowl_1 plate_1)))
)
"""


def base_spec(**kw):
    d = {"suite": "libero_goal", "base_task": 8, "id": "v1", "instruction": "put the bowl on the stove",
         "goal": ["On akita_black_bowl_1 flat_stove_1_cook_region"]}  # fmt: skip
    d.update(kw)
    return d


# ---------------------------------------------------------------- spec


def test_spec_validation():
    s = parse_spec(base_spec())
    assert s.goal == [["On", "akita_black_bowl_1", "flat_stove_1_cook_region"]]
    assert s.init_mode == "keep" and s.track_goals == "default" and s.perturbation_dimension() == "language"
    s = parse_spec(base_spec(scene={"fixture_state": {"flat_stove_1": True}}, init="patch"))
    assert s.scene.fixture_state == {"flat_stove_1": "on"}  # YAML 1.1 bare `on`
    assert s.perturbation_dimension() == "layout"
    s = parse_spec(base_spec(scene={"remove": ["plate_1"]}, init={"resample": 5}, goal=[]))
    assert s.init_n == 5 and s.goal == [] and s.perturbation_dimension() == "object"
    s = parse_spec(base_spec(track_goals=[8, {"name": "x", "goal": ["(Up plate_1)"]}]))
    assert [t.name for t in s.track_goals] == ["t8", "x"]
    bad = [
        (base_spec(scene={"swap": [["a", "b"]]}), "init: keep"),
        (base_spec(goal=["Foo a b"]), "unknown predicate"),
        (base_spec(goal=["On a"]), "takes 2"),
        (
            base_spec(scene={"move": [{"object": "a", "xy": [0, 0]}], "remove": ["a"]}, init="patch"),
            "edited twice",
        ),
        (base_spec(scene={"move": [{"object": "a"}]}, init="patch"), "exactly one"),
        (base_spec(scene={"add": [{"object": "ketchup_1", "xy": [0, 0]}]}, init="patch"), "category"),
        (base_spec(init="later"), "init must be"),
        (base_spec(id="a b"), "file-name safe"),
        (base_spec(colour="red"), "unknown keys"),
    ]
    for d, msg in bad:
        with pytest.raises(SpecError, match=msg):
            parse_spec(d)


def test_shipped_example_specs_are_valid():
    cat = list_variants(None)
    assert len(cat) >= 53
    assert not [c for c in cat if c["description"].startswith("INVALID")]
    ids = {(c["suite"], c["details"][0]) for c in cat}
    assert len(ids) == len(cat)
    for c in cat:
        assert {"benchmark", "suite", "key", "levels", "details", "count", "description", "requires"} <= set(
            c
        )
        assert c["benchmark"] == "ours" and c["key"] == "custom"


# ---------------------------------------------------------------- BDDL


def test_bddl_roundtrip_and_edits():
    p = Problem(parse(GOAL_BDDL))
    assert parse(p.text()) == p.tree  # dump → parse is the identity on the tree
    assert p.objects == {"akita_black_bowl_1": "akita_black_bowl", "plate_1": "plate", "basket_1": "basket"}
    assert "basket_1_contain_region" in p.regions and p.goal == [["On", "akita_black_bowl_1", "plate_1"]]
    q = p.copy()
    q.set_language("put the plate on the bowl")
    q.set_goal([parse_predicate("On plate_1 akita_black_bowl_1")])
    dropped = q.remove_object("basket_1")
    assert dropped == ["basket_1_contain_region"]
    assert "basket_1" not in q.objects and all("basket_1_contain_region" not in x for x in q.init)
    q.add_object("ketchup_1", "ketchup")
    reg = q.add_region("ketchup_1_added_region", "main_table", [[0.1, 0.1, 0.11, 0.11]])
    q.set_placement("ketchup_1", reg)
    q.set_fixture_state("wooden_cabinet_1_top_region", "open")
    q.set_fixture_state("wooden_cabinet_1_top_region", "close")  # replaces, not appends
    r = Problem(parse(q.text()))
    assert r.language == "put the plate on the bowl"
    assert r.goal == [["On", "plate_1", "akita_black_bowl_1"]]
    assert r.placement("ketchup_1") == ["On", "ketchup_1", "main_table_ketchup_1_added_region"]
    assert [x for x in r.init if x[0] in ("Open", "Close")] == [["Close", "wooden_cabinet_1_top_region"]]
    assert r.scene_key() != p.scene_key()
    s = p.copy()
    s.set_language("anything")
    s.set_goal([])
    assert s.scene_key() == p.scene_key()  # language / goal do not change the scene
    with pytest.raises(ValueError):
        q.add_object("ketchup_1", "ketchup")


# ---------------------------------------------------------------- mock: run_eval / sc-bench / report


def write_mock_variant(root: Path, vid: str, base_task: int, instruction: str, track_instr: str) -> None:
    """A built-looking variant for --backend mock (no LIBERO): YAML + generated/<id>/manifest.json.
    The mock policy heads to text_pos(instruction); a tracked goal named after the instruction is
    therefore reached (fallback), the new goal (label <id>:new) is not."""
    d = root / "libero_goal"
    d.mkdir(parents=True, exist_ok=True)
    y = d / f"{vid}.yaml"
    y.write_text(
        f"suite: libero_goal\nbase_task: {base_task}\nid: {vid}\ninstruction: {instruction}\n"
        "goal: [On cream_cheese_1 plate_1]\ninit: keep\n"
    )
    g = d / "generated" / vid
    g.mkdir(parents=True, exist_ok=True)
    (g / f"{vid}.bddl").write_text("mock")
    (g / "init_states.npy").write_bytes(b"mock")
    m = {
        "id": vid, "suite": "libero_goal", "group": "test", "libero_env": "mock",
        "spec_sha256": hashlib.sha256(y.read_bytes()).hexdigest(),
        "instruction": instruction, "goal": [["On", "cream_cheese_1", "plate_1"]],
        "base_task": {"task_id": base_task, "goal": [["On", "akita_black_bowl_1", "plate_1"]], "goal_absent": []},
        "track_goals": {
            "t8": {"goal": [["On", "akita_black_bowl_1", "plate_1"]], "task": 8, "instruction": "put the bowl on the plate", "absent": []},
            track_instr: {"goal": [["On", "x", "y"]], "task": None, "instruction": track_instr, "absent": []},
            "gone": {"goal": [["In", "ketchup_1", "basket_1_contain_region"]], "task": None, "instruction": None, "absent": ["ketchup_1"]},
        },
        "edits": {"kinds": [], "dimension": "language"},
        "files": {"bddl": {"path": f"{vid}.bddl", "sha256": "a" * 64}, "init_states": {"path": "init_states.npy", "sha256": "b" * 64}},
        "init_states": {"mode": "keep", "n": 50},
    }  # fmt: skip
    (g / "manifest.json").write_text(json.dumps(m))


def run_custom(out: Path, vdir: Path, *extra: str) -> list[dict]:
    run_eval.main(
        ["--policy", "pi05", "--checkpoint", "mock/pi05lib", "--model-tag", "pi05lib", "--suite", "libero_goal",
         "--condition", "custom", "--backend", "mock", "--out", str(out), "--variants-dir", str(vdir),
         "--episodes", "2", "--video", "first:1", *extra]
    )  # fmt: skip
    return [json.loads(x) for x in (out / "episodes.jsonl").read_text().splitlines()]


def test_custom_run_eval_mock_multi_goal(tmp_path):
    vdir = tmp_path / "variants"
    write_mock_variant(
        vdir, "cheese_plate", 8, "put the cream cheese on the plate", "put the cream cheese on the plate"
    )
    write_mock_variant(vdir, "other", 1, "put the wine bottle on the plate", "unrelated")
    eps = run_custom(tmp_path / "run", vdir, "--variants", "cheese_plate,other")
    assert len(eps) == 4 and {e["variant"] for e in eps} == {"cheese_plate", "other"}
    for e in eps:
        assert e["condition"] == "custom" and e["early_stop"] is False and e["n_steps"] == e["max_steps"]
        other = "put the cream cheese on the plate" if e["variant"] == "cheese_plate" else "unrelated"
        assert set(e["goals_reached"]) == {"t8", other}
        assert "gone" not in e["goals_reached"] and e["variant_meta"]["track_absent"] == ["gone"]
        assert e["success_new"] is False and isinstance(e["goals_initial"], list)
        assert e["seed"] == run_eval.episode_seed(7, 0, e["task_id"], e["init_idx"])  # pairs with base C0
    hit = [e for e in eps if e["variant"] == "cheese_plate"]
    assert all(e["goals_reached"]["put the cream cheese on the plate"] is not None for e in hit)
    import numpy as np

    z = np.load(tmp_path / "run" / hit[0]["steps_file"])
    assert z["goal_track"].shape == (hit[0]["n_steps"], 2) and list(z["goal_track_names"])[0] == "t8"
    # fallback table: cheese_plate always ends up at a tracked goal, never the new one
    summary, long = fallback_rows(eps)
    s = {r["variant"]: r for r in summary}
    assert s["cheese_plate"]["any_trained"] == 1.0 and s["cheese_plate"]["nothing"] == 0.0
    assert s["cheese_plate"]["sr_new"] == 0.0
    assert {r["goal"] for r in long if r["variant"] == "cheese_plate"} == {
        "t8",
        "put the cream cheese on the plate",
    }
    # resume: same command adds nothing; regenerated files under the same id → refused
    assert len(run_custom(tmp_path / "run", vdir, "--variants", "cheese_plate,other")) == 4
    mp = vdir / "libero_goal" / "generated" / "other" / "manifest.json"
    m = json.loads(mp.read_text())
    m["files"]["init_states"]["sha256"] = "c" * 64
    mp.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="other generated files"):
        run_custom(tmp_path / "run", vdir, "--variants", "cheese_plate,other")
    # YAML edited after the build → refused until rebuilt
    (vdir / "libero_goal" / "cheese_plate.yaml").write_text(
        (vdir / "libero_goal" / "cheese_plate.yaml").read_text() + "notes: changed\n"
    )
    with pytest.raises(SystemExit, match="changed since it was built"):
        run_custom(tmp_path / "run2", vdir, "--variants", "cheese_plate")


def test_custom_catalog(tmp_path):
    vdir = tmp_path / "variants"
    write_mock_variant(vdir, "cheese_plate", 8, "put the cream cheese on the plate", "t")
    (vdir / "libero_goal" / "unbuilt.yaml").write_text(
        "suite: libero_goal\nbase_task: 8\nid: unbuilt\ninstruction: x\ngoal: []\n"
    )
    (vdir / "libero_goal" / "broken.yaml").write_text("suite: libero_goal\nid: broken\n")
    cat = {c["details"][0]: c for c in list_variants("libero_goal", vdir)}
    assert (
        cat["cheese_plate"]["built"]
        and cat["cheese_plate"]["count"] == 50
        and not cat["cheese_plate"]["requires"]
    )
    assert cat["unbuilt"]["requires"] == ["sc-variants build"] and cat["unbuilt"]["count"] is None
    assert cat["broken"]["description"].startswith("INVALID")


def test_custom_job_through_sc_bench_and_report(tmp_path):
    pytest.importorskip("websockets")
    from bench.core import report as sc_report, runner as sc_bench

    vdir = tmp_path / "variants"
    write_mock_variant(
        vdir, "cheese_plate", 8, "put the cream cheese on the plate", "put the cream cheese on the plate"
    )
    write_mock_variant(vdir, "other", 1, "put the wine bottle on the plate", "unrelated")
    bat = tmp_path / "b.yaml"
    bat.write_text(
        "name: customtest\njobs:\n"
        "  - {benchmark: ours, suite: libero_goal, conditions: [C0], tasks: '1,8', episodes: 2, video: 'first:1',"
        f" instr_set: '{HERE / 'fixtures' / 'libero_goal_2tasks.json'}'}}\n"
        "  - {benchmark: ours, suite: libero_goal, condition: custom, variants: [cheese_plate, other],"
        f" variants_dir: '{vdir}', episodes: 2, video: 'first:1'}}\n"
    )
    root = tmp_path / "root"
    argv = ["--ckpt", "mock/pi05lib", "--backend", "mock", "--battery", str(bat), "--out-root", str(root)]
    assert sc_bench.main(argv) == 0
    jd = root / "runs" / "bench-pi05lib-customtest" / "jobs" / "ours-goal-custom"
    eps = [json.loads(x) for x in (jd / "episodes.jsonl").read_text().splitlines()]
    assert len(eps) == 4
    for e in eps:
        assert e["perturbation"] == {
            "dimension": "language",
            "level": "custom",
            "detail": f"custom:{e['variant']}",
        }
        assert e["goals_reached"] and e["success"] == e["success_new"]
    meta = json.loads((jd / "run_meta.json").read_text())
    assert meta["identity"]["condition"] == "custom" and meta["identity"]["variants_dir"] == str(
        vdir.resolve()
    )
    out = tmp_path / "report"
    assert (
        sc_report.main(
            [
                "--runs",
                "bench-pi05lib-customtest",
                "--out-root",
                str(root),
                "--out",
                str(out),
                "--videos",
                "0",
            ]
        )
        == 0
    )
    md = (out / "report.md").read_text()
    assert "## Custom variants: fallback" in md and "cheese_plate" in md
    rows = (out / "fallback.csv").read_text().splitlines()
    assert rows[0].startswith("model,suite,variant") and len(rows) == 1 + 4


# ---------------------------------------------------------------- LIBERO (CPU)

LIBERO_TEST = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from variants import preview  # noqa: F401  (CPU EGL before common.py)


def main(vdir):
    import numpy as np
    import run_eval
    from run_eval import ContactProbe, EpisodeStepper
    from variants.check import check
    from variants.generate import build, load_init_states, suite_tasks
    from variants.runtime import goals_for, make_env, plan_custom
    from variants.spec import load_manifest, load_spec

    out = {}
    m = build(load_spec(vdir / "libero_object" / "swap.yaml"), vdir)
    out["swap"] = {"n": m["init_states"]["n"], "failures": len(m["init_states"]["check"]["failures"]),
                   "absent": m["base_task"]["goal_absent"]}
    m2 = build(load_spec(vdir / "libero_goal" / "keep.yaml"), vdir)
    base = load_init_states(suite_tasks("libero_goal")[8]["init_file"])
    out["keep_same"] = bool(np.array_equal(load_init_states(Path(m2["_dir"]) / "init_states.npy"), base))
    try:
        build(load_spec(vdir / "libero_object" / "bad.yaml"), vdir)
        out["bad"] = "built"
    except Exception as e:  # noqa: BLE001
        out["bad"] = f"{type(e).__name__}: {e}"[:300]
    # the state check flags an object that is not where the edit says it should be
    from variants.generate import check_state, control_env

    e = control_env(Path(m["_dir"]) / "swap.bddl")
    st = load_init_states(Path(m["_dir"]) / "init_states.npy")[0]
    res = m["edits"]["resolved"]
    out["check_ok"] = check_state(e, st, 0, {}, res, None, None)
    out["check_bad"] = check_state(e, st, 0, {"alphabet_soup_1": {"xy": [0.3, 0.3], "z": 0.0}}, res, None, None)
    e.close()
    check([load_manifest("libero_object", "swap", vdir)], workers=2, n_inits=1, log=lambda *a: None)
    out["feas"] = load_manifest("libero_object", "swap", vdir)["feasibility"]["status"]
    # a few real env steps through the eval stepper with multi-goal tracking
    [(tid, [v])] = plan_custom("libero_object", ["swap"], str(vdir), "libero", m["libero_env"])
    suite = run_eval.libero().get_suite("libero_object")
    env = make_env(v, suite, "libero_object", "libero", 5, [64, 64], "relative")
    new, orig, track = goals_for(v, "libero")
    ep = EpisodeStepper(env, v, 0, 123, 5, new, False, "t", ContactProbe, orig, track)
    ep.reset()
    while not ep.step(np.zeros(7, np.float32)):
        pass
    r = ep.result(0.0)
    out["steps"] = {"n": r["n_steps"], "tracked": sorted(r["goals_reached"]), "initial": r["goals_initial"],
                    "shape": list(r["arrays"]["goal_track"].shape), "orig": r["success_orig"], "new": r["success_new"]}
    env.close()
    print("@@" + json.dumps(out))


if __name__ == "__main__":
    main(Path(sys.argv[2]))
"""


@pytest.mark.skipif(os.environ.get("SC_SKIP_LIBERO") == "1", reason="SC_SKIP_LIBERO=1")
def test_libero_generate_check_and_track(tmp_path):
    pytest.importorskip("libero")
    pytest.importorskip("robosuite")
    vd = tmp_path / "v"
    (vd / "libero_object").mkdir(parents=True)
    (vd / "libero_goal").mkdir(parents=True)
    (vd / "libero_object" / "swap.yaml").write_text(
        "suite: libero_object\nbase_task: 0\nid: swap\n"
        "instruction: pick up the alphabet soup and place it in the basket\n"
        "goal: [In alphabet_soup_1 basket_1_contain_region]\n"
        "scene: {swap: [[alphabet_soup_1, salad_dressing_1]]}\ninit: patch\n"
    )
    (vd / "libero_goal" / "keep.yaml").write_text(
        "suite: libero_goal\nbase_task: 8\nid: keep\ninstruction: pick up the cream cheese\n"
        "goal: [Up cream_cheese_1]\n"
    )
    # moved onto the alphabet soup's spot → the two overlap: every state must fail the check
    (vd / "libero_object" / "bad.yaml").write_text(
        "suite: libero_object\nbase_task: 0\nid: bad\ninstruction: x\ngoal: []\n"
        "scene: {move: [{object: milk_1, to: floor_target_object_region}]}\ninit: patch\n"
    )
    script = tmp_path / "t.py"
    script.write_text(LIBERO_TEST)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    env.pop("CUDA_VISIBLE_DEVICES")  # robosuite's EGL device pick breaks on an empty value
    p = subprocess.run(
        [sys.executable, str(script), str(SHORTCUT), str(vd)],
        capture_output=True,
        text=True,
        env=env,
        timeout=1500,
    )
    line = [x for x in p.stdout.splitlines() if x.startswith("@@")]
    assert line, p.stdout[-2000:] + p.stderr[-4000:]
    out = json.loads(line[0][2:])
    assert out["swap"] == {"n": 50, "failures": 0, "absent": []}
    assert out["keep_same"] is True
    assert "placement sampler cannot place" in out["bad"], out["bad"]
    assert out["check_ok"] == [] and "from its target xy" in out["check_bad"][0]
    assert out["feas"] == "feasible"
    s = out["steps"]
    assert s["n"] == 5 and s["shape"][0] == 5 and s["orig"] is False and s["new"] is False
    assert "t0" in s["tracked"] and "t3" not in s["tracked"]  # bbq sauce (task 3) is not in this scene
