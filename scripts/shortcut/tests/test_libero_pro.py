"""CPU-only checks for the LIBERO-PRO adapter (T013): catalog on a fake PRO tree, the sc-bench flow through the
mock policy server with the mock PRO env (standard records, task-axis dual goals, empty init file, resume,
sc-report rows).

    uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_libero_pro.py -q
"""

import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack")

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parent
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "parallel")]

from bench.core import records, report as sc_report, runner as sc_bench  # noqa: E402
from bench.libero_pro import catalog  # noqa: E402
from bench.libero_pro.adapter import LiberoProAdapter  # noqa: E402

GOAL = [
    Path(t["bddl"]).stem
    for t in json.loads((SHORTCUT / "instructions" / "libero_goal.json").read_text())["tasks"]
]
assert GOAL[8] == "put_the_bowl_on_the_plate"


def bddl(language: str, goal: str) -> str:
    return f"(define (problem LIBERO_Tabletop_Manipulation)\n  (:language {language})\n  (:goal\n    (And {goal})\n  )\n)\n"


def init_file(path: Path, n: int) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(np.zeros((n, 4))))
        z.writestr("archive/version", b"1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


@pytest.fixture
def pro_tree(tmp_path, monkeypatch):
    """bddl_files / init_files of libero_goal: clean, _task (new goal + instruction), _swap (t1: empty inits).
    PRO's :language differs from the training sentence ("Pro ..."), as in the real PRO files."""
    root = tmp_path / "pro" / "libero" / "libero"
    (root / "benchmark").mkdir(parents=True)
    (root / "benchmark" / "libero_suite_task_map.py").write_text(
        f"libero_task_map = {{'libero_goal': {GOAL!r}}}\n"
    )
    for d, lang, goal in (
        ("libero_goal", "Pro {t}", "(On bowl_1 plate_1)"),
        ("libero_goal_task", "new {t}", "(On wine_bottle_1 plate_1)"),
        ("libero_goal_swap", "Pro {t}", "(On bowl_1 plate_1)"),
    ):
        for i, name in enumerate(GOAL):
            p = root / "bddl_files" / d / f"{name}.bddl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(bddl(lang.format(t=name.replace("_", " ")), goal))
            init_file(
                root / "init_files" / d / f"{name}.pruned_init", 0 if (d.endswith("swap") and i == 1) else 50
            )
    (tmp_path / "pro" / "SC_VERSION").write_text("libero-pro-test\n")
    monkeypatch.setenv("LIBERO_PRO_ROOT", str(root))
    catalog.suite_task_names.cache_clear()
    catalog.n_init_states.cache_clear()
    return root


def test_catalog_levels_dirs_and_entries(pro_tree):
    assert catalog.levels("libero_goal", "position") == ("swap",)
    assert catalog.levels("libero_object", "position")[:2] == ("swap", "x0.1") and "y0.5" in catalog.XY_LEVELS
    assert catalog.axis_dir("libero_object", "position", "x0.3") == "libero_object_temp_x0.3"
    assert catalog.axis_dir("libero_goal", "semantic", "paraphrase") == "libero_goal_lan"
    with pytest.raises(ValueError):
        catalog.axis_dir("libero_goal", "position", "x0.3")
    assert catalog.filename_language("KITCHEN_SCENE10_put_it_on") == "put it on"
    e = catalog.task_entry(pro_tree, "libero_goal", "task", "task", 8)
    assert e["instruction_original"] == "put the bowl on the plate"  # training sentence, always
    assert e["pro_instruction_original"] == "Pro put the bowl on the plate"
    assert e["instruction_given"] == "new put the bowl on the plate" and e["dual"] and e["goal_changed"]
    assert e["init_source"] == "clean" and e["init"].endswith(
        "init_files/libero_goal/put_the_bowl_on_the_plate.pruned_init"
    )
    f = catalog.task_entry(pro_tree, "libero_goal", "task", "task", 8, instruction_source="filename")
    assert (
        f["instruction_given"] == "put the bowl on the plate"
    )  # what PRO's Benchmark.get_task().language says
    p = catalog.task_entry(pro_tree, "libero_goal", "task", "task", 8, init_source="pro")
    assert p["init"].endswith("libero_goal_task/put_the_bowl_on_the_plate.pruned_init")
    s = catalog.task_entry(pro_tree, "libero_goal", "position", "swap", 8)
    assert (
        not s["dual"] and s["instruction_given"] == "put the bowl on the plate"
    )  # visual axis: training sentence
    s2 = catalog.task_entry(pro_tree, "libero_goal", "position", "swap", 8, instruction_source="pro_bddl")
    assert s2["instruction_given"] == "Pro put the bowl on the plate"
    assert catalog.task_entry(pro_tree, "libero_goal", "none", "original", 8)["instruction_given"] == (
        "put the bowl on the plate"
    )
    assert catalog.bench_version(pro_tree) == "libero-pro-test"
    with pytest.raises(FileNotFoundError, match="setup_venv"):
        catalog.task_entry(pro_tree, "libero_goal", "semantic", "paraphrase", 0)


def test_job_ids_and_level_checks(pro_tree, tmp_path):
    ad = LiberoProAdapter()
    assert ad.job_id({"suite": "libero_goal", "perturbation": "task"}) == "pro-goal-task"
    assert ad.job_id(
        {"suite": "libero_10", "perturbation": "semantic", "instruction_source": "filename", "repeat": 1}
    ) == ("pro-l10-semantic-filename-r1")
    assert (
        ad.job_id({"suite": "libero_goal", "perturbation": "task", "init_source": "clean"}) == "pro-goal-task"
    )
    assert (
        ad.job_id({"suite": "libero_goal", "perturbation": "task", "init_source": "pro"})
        == "pro-goal-task-initpro"
    )
    ctx = sc_bench.Context(tmp_path / "run", {"ckpt_id": "m", "path_or_hub": "m"}, "mock", SHORTCUT, {})
    with pytest.raises(SystemExit, match="levels"):
        ad.plan({"suite": "libero_goal", "perturbation": "position", "levels": ["x0.1"]}, ctx)
    plan = ad.plan({"suite": "libero_goal", "perturbation": "position", "tasks": "0-2", "episodes": 3}, ctx)
    assert [u["task_id"] for u in plan.units] == [0, 2] and plan.n_episodes == 6  # t1 has no init states


def write_battery(tmp_path: Path) -> Path:
    p = tmp_path / "pro.yaml"
    p.write_text(
        "name: protest\ndefaults: {seed: 7, video: 'first:1'}\njobs:\n"
        "  - {benchmark: libero_pro, suite: libero_goal, perturbations: [none, task, position], tasks: '1,8', episodes: 2}\n"
    )
    return p


def run_episodes(root: Path, run_id: str) -> list[dict]:
    out = []
    for f in sorted((root / "runs" / run_id / "jobs").glob("*/episodes.jsonl")):
        out += records.read_jsonl(f)
    return out


def test_sc_bench_flow_mock(pro_tree, tmp_path):
    bat, root = write_battery(tmp_path), tmp_path / "out"
    argv = ["--ckpt", "mock/pi05lib", "--backend", "mock", "--battery", str(bat), "--out-root", str(root)]
    assert sc_bench.main([*argv, "--clients-per-gpu", "2"]) == 0
    eps = run_episodes(root, "bench-pi05lib-protest")
    assert len(eps) == 2 * 2 + 2 * 2 + 1 * 2  # none, task: tasks 1, 8; position: task 1 has no init states
    for r in eps:
        assert records.missing_fields(r) == [], r
        assert r["benchmark"] == "libero_pro" and r["bench_version"] == "libero-pro-mock"
        assert r["perturbation"]["detail"] == f"pro:{r['axis']}" and r["variant_id"] == r["level"]
    task = [r for r in eps if r["axis"] == "task"]
    assert all(r["perturbation"]["dimension"] == "task" for r in task)
    assert all(r["success_new"] is not None and r["success_orig"] is not None for r in task)
    assert all(
        r["success"] == r["success_new"] and r["n_steps"] == r["max_steps"] for r in task
    )  # no early stop
    assert all(r["instruction_given"].startswith("new ") for r in task)
    assert all(r["init_source"] == "clean" for r in task)
    for r in eps:
        assert r["pro_instruction_original"].startswith("Pro ") and not r["instruction_original"].startswith(
            "Pro"
        )
        if r["axis"] != "task":
            assert r["instruction_given"] == r["instruction_original"]  # training sentence on visual axes
    clean = [r for r in eps if r["axis"] == "none"]
    assert all(r["success_new"] is None and r["success_orig"] == r["success"] for r in clean)
    assert any(r["video"] for r in eps) and all(
        (
            Path(root / "runs" / "bench-pi05lib-protest" / "jobs" / f"pro-goal-{r['axis']}") / r["video"]
        ).exists()
        for r in eps
        if r["video"]
    )
    meta = json.loads(
        (root / "runs" / "bench-pi05lib-protest" / "jobs" / "pro-goal-task" / "run_meta.json").read_text()
    )
    assert meta["identity"]["exec_mode"] == "batched-2" and meta["identity"]["axis"] == "task"

    # resume: nothing pending, no duplicates; another exec mode → refused
    assert sc_bench.main([*argv, "--clients-per-gpu", "2"]) == 0
    assert len(run_episodes(root, "bench-pi05lib-protest")) == len(eps)
    with pytest.raises(SystemExit, match="another configuration"):
        sc_bench.main(argv)

    out = tmp_path / "report"
    assert (
        sc_report.main(
            ["--runs", "bench-pi05lib-protest", "--out-root", str(root), "--out", str(out), "--videos", "0"]
        )
        == 0
    )
    rows = (out / "table.csv").read_text().splitlines()
    assert any("libero_pro" in r and ",task," in r and "pro:task" in r for r in rows)


def test_gen_env_axis_refuses_mixed_generation_settings(tmp_path):
    from bench.libero_pro import gen_env_axis as g

    root = tmp_path / "libero" / "libero"
    want13, want42 = g.wanted_meta(13, 50, "abc"), g.wanted_meta(42, 50, "abc")
    items = [("libero_goal", f"t{i}") for i in range(3)]
    for s, t in items:
        _, init, meta = g.out_paths(root, s, t)
        init_file(init, 50)
        meta.write_text(json.dumps({**want13, "table": "main_table->living_room_table"}))
    assert {g.cache_state(root, s, t, want13) for s, t in items} == {"ok"}
    assert g.existing(root) == items
    g.preflight(root, want13, force=False)  # same settings: reuse / fill gaps
    g.out_paths(root, *items[0])[1].unlink()  # one task deleted, the others are from seed 13
    assert g.cache_state(root, *items[0], want42) == "missing"
    # regenerating only that task (--tasks 0) with seed 42 would mix seeds across the axis: the check covers every
    # existing env-axis file, not just the requested ones
    with pytest.raises(SystemExit, match="another configuration"):
        g.preflight(root, want42, force=False)
    g.preflight(root, want42, force=True)  # --force: main() regenerates the whole axis
    g.preflight(root, want13, force=False)  # filling the gap with seed 13 is fine
    # a file of another suite made with another seed also blocks (same axis)
    _, init, meta = g.out_paths(root, "libero_spatial", "s0")
    init_file(init, 50)
    meta.write_text(json.dumps(want42))
    with pytest.raises(SystemExit, match="libero_spatial/s0"):
        g.preflight(root, want13, force=False)
    meta.write_text(json.dumps(want13))
    g.out_paths(root, *items[1])[2].unlink()  # no sidecar (older generator) → not trusted
    assert g.cache_state(root, *items[1], want13) == "mismatch"
    assert g.cache_state(root, *items[2], g.wanted_meta(13, 50, "other")) == "mismatch"
    assert g.cache_state(root, *items[2], {**want13, "script_version": "v0"}) == "mismatch"
