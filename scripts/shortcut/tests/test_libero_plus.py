"""CPU checks for the LIBERO-Plus adapter (T012): registry parsing / selection, the sc-bench flow with a mock
Plus env (fake fork checkout: task_classification.json + BDDL goals), resume, records, sc-report.

    uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_libero_plus.py -q
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack")

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parent
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "parallel")]

from bench.core import battery, records, report as sc_report, runner as sc_bench  # noqa: E402
from bench.core.adapters import Context  # noqa: E402
from bench.libero_plus import adapter as lp_adapter, registry as reg  # noqa: E402

GOAL = "(:goal (And (On akita_black_bowl_1 plate_1)))"
SUITE = "libero_goal"
T8 = reg.ORIGINAL_TASKS[SUITE][8]  # put_the_bowl_on_the_plate
T1 = reg.ORIGINAL_TASKS[SUITE][1]  # put_the_bowl_on_the_stove


def test_parse_variant_names():
    p = lambda n, c, lv=3: reg.parse_variant(SUITE, 5, n, c, lv)  # noqa: E731
    v = p(f"{T8}_language_7_view_0_0_100_0_0_initstate_0", "Language Instructions")
    assert (v.task_id, v.dimension, v.detail, v.params, v.variant_id) == (
        8,
        "language",
        "rewrite",
        {"k": 7},
        "p0005",
    )
    assert p(f"{T8}_add_12", "Objects Layout").detail == "add"
    m = p("turn_on_the_stove_moved_level5_sample4", "Objects Layout")
    assert (m.task_id, m.detail, m.params) == (7, "moved", {"level": 5, "sample": 4})
    assert p(f"{T8}_view_0_0_181_0_0_initstate_0", "Camera Viewpoints").params["scale"] == 181
    assert p(f"{T8}_view_0_0_100_0_0_initstate_19", "Robot Initial States").params == {"initstate": 19}
    noise = [
        p(f"{T8}_view_0_0_100_0_0_initstate_0_noise_{k}", "Sensor Noise").detail for k in (4, 13, 26, 37, 41)
    ]
    assert noise == list(reg.NOISE_KINDS)
    assert p(f"{T8}_light_3", "Light Conditions", None).level is None
    assert p(f"{T8}_tb_6", "Background Textures").detail == "tb"
    # longest-prefix match: "put_the_bowl_on_the_plate" must not be taken for another task
    assert reg.original_task_id("libero_spatial", reg.ORIGINAL_TASKS["libero_spatial"][0] + "_add_1") == 0
    with pytest.raises(ValueError):
        reg.original_task_id(SUITE, "not_a_task_add_1")
    assert (
        reg.language_from_name(reg.ORIGINAL_TASKS["libero_10"][2])
        == "turn on the stove and put the moka pot on it"
    )


def test_original_tasks_match_hf_libero():
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("hf-libero not installed")
    path = (
        Path(list(spec.submodule_search_locations)[0]) / "libero" / "benchmark" / "libero_suite_task_map.py"
    )
    ms = importlib.util.spec_from_file_location("libero_suite_task_map", path)
    assert ms is not None and ms.loader is not None
    mod = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(mod)
    for suite in reg.SUITES:
        assert mod.libero_task_map[suite] == reg.ORIGINAL_TASKS[suite]


def fake_fork(root: Path) -> Path:
    """A LIBERO-Plus checkout with just the registry and BDDL goals for two Goal tasks (5 levels × 2 per axis)."""
    src = root / "LIBERO-plus"
    bddl = src / "libero" / "libero" / "bddl_files" / SUITE
    bddl.mkdir(parents=True)
    entries: list[dict] = []

    def add(name: str, cat: str, level: int | None, bddl_stem: str | None = None) -> None:
        entries.append({"id": len(entries) + 1, "name": name, "category": cat, "difficulty_level": level})
        (bddl / f"{bddl_stem or name}.bddl").write_text(
            f"(define (problem x)\n  (:language {name})\n  {GOAL}\n)\n"
        )

    for t in (T1, T8):
        (bddl / f"{t}.bddl").write_text(f"(define (problem x)\n  (:language orig)\n  {GOAL}\n)\n")
        for lv in range(1, 6):
            for k in range(2):
                i = 10 * lv + k
                add(
                    f"{t}_language_{i}_view_0_0_100_0_0_initstate_0",
                    "Language Instructions",
                    lv,
                    f"{t}_language_{i}",
                )
                add(f"{t}_add_{i}", "Objects Layout", lv)
                add(f"{t}_view_0_0_100_0_0_initstate_0_noise_{1 + k * 10}", "Sensor Noise", lv, t)
    cls = src / "libero" / "libero" / "benchmark" / "task_classification.json"
    cls.parent.mkdir(parents=True)
    cls.write_text(json.dumps({SUITE: entries}))
    venv = root / "venv"
    venv.mkdir()
    (venv / "sc_env.json").write_text(
        json.dumps({"libero_config_path": str(root), "magick": False, "src": str(src), "sha": "abcdef0123"})
    )
    return venv


def test_select_and_goals(tmp_path):
    venv = fake_fork(tmp_path)
    src = Path(json.loads((venv / "sc_env.json").read_text())["src"])
    vs = reg.load_variants(src, SUITE)
    assert len(vs) == 2 * 5 * 2 * 3
    sel = reg.select(vs, dimension="language", tasks=[1, 8], n=5, seed=7)
    assert [v.level for v in sel[8]] == [1, 2, 3, 4, 5]  # one per level
    assert reg.select(vs, dimension="language", tasks=[1, 8], n=5, seed=7) == sel  # deterministic
    assert len(reg.select(vs, dimension="language", tasks=[8], n=7)[8]) == 7
    assert [v.level for v in reg.select(vs, dimension="layout", tasks=[8], n=4, levels=[1, 5])[8]] == [
        1,
        1,
        5,
        5,
    ]
    assert len(reg.select(vs, dimension="layout", tasks=[8], n="all")[8]) == 10
    assert all(reg.goal_matches_original(src, v) for v in vs)
    v = sel[8][0]
    reg.variant_bddl(src, v).write_text("(:language x)\n(:goal (And (On akita_black_bowl_1 stove_1)))")
    assert not reg.goal_matches_original(src, v)


def write_battery(tmp_path: Path) -> Path:
    p = tmp_path / "plus.yaml"
    p.write_text(
        "name: plustest\ndefaults: {seed: 7, video: 'first:1'}\njobs:\n"
        f"  - {{benchmark: libero_plus, suite: {SUITE}, perturbations: [none], tasks: '1,8', episodes: 2}}\n"
        f"  - {{benchmark: libero_plus, suite: {SUITE}, perturbations: [language, layout, noise], tasks: '1,8', "
        "variants: 3}\n"
    )
    return p


def argv(bat: Path, root: Path, *extra: str, ckpt: str = "mock/pi05lib") -> list[str]:
    return ["--ckpt", ckpt, "--backend", "mock", "--battery", str(bat), "--out-root", str(root), *extra]


def plus_episodes(root: Path, run_id: str) -> dict[tuple, dict]:
    out = {}
    for f in sorted((root / "runs" / run_id / "jobs").glob("plus-*/episodes.jsonl")):
        for r in records.read_jsonl(f):
            k = (r["perturbation"]["dimension"], r["task_id"], r["variant_id"], r["init_idx"])
            assert k not in out, f"duplicate {k}"
            out[k] = {
                x: v
                for x, v in r.items()
                if not x.startswith("time_") and x not in ("wall_time_s", "policy_server_ms")
            }
    return out


def test_plus_flow_records_resume_report(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBERO_PLUS_VENV", str(fake_fork(tmp_path)))
    bat = write_battery(tmp_path)
    root = tmp_path / "out"
    assert sc_bench.main(argv(bat, root, "--clients-per-gpu", "2")) == 0
    eps = plus_episodes(root, "bench-pi05lib-plustest")
    dims = {k[0] for k in eps}
    assert dims == {"none", "language", "layout", "noise"}
    assert sum(k[0] == "none" for k in eps) == 2 * 2  # 2 tasks × 2 inits
    assert sum(k[0] == "language" for k in eps) == 2 * 3
    # noise: the fake registry has motion blur (1) and gaussian blur (11); magick=False → motion blur skipped
    noise = [r for k, r in eps.items() if k[0] == "noise"]
    assert noise and all(r["perturbation"]["detail"] == "gaussian_blur" for r in noise)
    for (dim, task, _vid, _i), r in eps.items():
        assert records.missing_fields(r) == []
        assert r["benchmark"] == "libero_plus" and r["bench_version"] == "libero-plus-abcdef0/mock"
        assert r["instruction_original"] == reg.language_from_name(reg.ORIGINAL_TASKS[SUITE][task])
        if dim == "language":  # changed instruction is given, original kept
            assert r["instruction_given"] == r["plus_instruction_registry"] != r["instruction_original"]
        else:  # clean mode: the axis alone, no filename suffix in the instruction
            assert r["instruction_given"] == r["instruction_original"]
        if dim in ("none", "language", "layout"):
            assert r["success_orig"] == r["success"]
        else:
            assert r["success_orig"] is None
        assert r["success_new"] is None and isinstance(r["perturbation"]["level"], (int, type(None)))
    none8 = eps[("none", 8, "orig", 1)]
    assert none8["seed"] == 7 * 1_000_000 + 8 * 100 + 1  # = ours episode_seed (pairs with C0)
    run = root / "runs" / "bench-pi05lib-plustest"
    jdir = run / "jobs" / "plus-goal-language"
    meta = json.loads((jdir / "run_meta.json").read_text())
    assert meta["identity"]["exec_mode"] == "batched-2" and meta["identity"]["axis"] == "language"
    assert sum(1 for _ in (jdir / "videos").rglob("*.mp4")) == 2  # first:1 per unit (task)
    assert all((jdir / r["steps_file"]).exists() for r in records.read_jsonl(jdir / "episodes.jsonl"))

    # resume: drop half the language episodes → the rerun writes exactly those again
    lines = (jdir / "episodes.jsonl").read_text().splitlines()
    (jdir / "episodes.jsonl").write_text("\n".join(lines[:3]) + "\n")
    assert sc_bench.main(argv(bat, root, "--clients-per-gpu", "2")) == 0
    assert plus_episodes(root, "bench-pi05lib-plustest") == eps

    # another instruction mode in the same run dir → refused
    bat2 = tmp_path / "plus2.yaml"
    bat2.write_text(bat.read_text().replace("variants: 3}", "variants: 3, instruction: registry}"))
    bat2.write_text(bat2.read_text().replace("name: plustest", "name: plustest2"))
    assert sc_bench.main(argv(bat2, root, "--clients-per-gpu", "2", "--run-id", "reg")) == 0
    reg_eps = plus_episodes(root, "reg")
    lay = next(r for k, r in reg_eps.items() if k[0] == "layout")
    assert lay["instruction_given"] == lay["plus_instruction_registry"].strip()
    assert (
        lay["instruction_given"].startswith(lay["instruction_original"])
        and " add " in lay["instruction_given"]
    )
    with pytest.raises(SystemExit, match="another configuration"):
        sc_bench.main(argv(bat2, root, "--clients-per-gpu", "2", "--run-id", "bench-pi05lib-plustest"))

    out = tmp_path / "report"
    assert (
        sc_report.main(["--runs", "bench-pi05lib-plustest", "--out-root", str(root), "--out", str(out)]) == 0
    )
    rows = (out / "table.csv").read_text().splitlines()
    assert any(",libero_plus,libero_goal,language," in r for r in rows)
    assert any(",libero_plus,libero_goal,none," in r for r in rows)


def test_shipped_batteries_plan_plus_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBERO_PLUS_VENV", str(fake_fork(tmp_path)))
    quick = [j for j in battery.expand_jobs(battery.load_battery("quick")) if j["benchmark"] == "libero_plus"]
    assert {j["perturbation"] for j in quick} == {"none", "language", "layout"}
    assert {j["suite"] for j in quick} == set(reg.SUITES)
    full = [j for j in battery.expand_jobs(battery.load_battery("full")) if j["benchmark"] == "libero_plus"]
    assert {j["perturbation"] for j in full} == set(reg.AXES)
    ad = lp_adapter.LiberoPlusAdapter()
    ctx = Context(tmp_path / "run", {"ckpt_id": "x", "path_or_hub": "x"}, "mock", SHORTCUT.parents[1])
    job = next(j for j in quick if j["suite"] == SUITE and j["perturbation"] == "language")
    plan = ad.plan(job, ctx)
    assert plan.job_id == "plus-goal-language" and plan.n_episodes == 2 * 10  # fake fork: tasks 1, 8 only
    assert {u["task_id"] for u in plan.units} == {1, 8}


def test_resume_refuses_another_selection(tmp_path, monkeypatch):
    """levels [1] then levels [5] in the same run/job must not append level-5 episodes to the level-1 job."""
    monkeypatch.setenv("LIBERO_PLUS_VENV", str(fake_fork(tmp_path)))
    root = tmp_path / "out"

    def bat(name: str, extra: str) -> Path:
        p = tmp_path / f"{name}.yaml"
        p.write_text(
            f"name: sel\njobs:\n  - {{benchmark: libero_plus, suite: {SUITE}, perturbation: layout, tasks: '1,8', "
            f"variants: 2, video: none{extra}}}\n"
        )
        return p

    jdir = root / "runs" / "sel" / "jobs" / "plus-goal-layout"
    assert sc_bench.main(argv(bat("l1", ", levels: [1]"), root, "--run-id", "sel")) == 0
    before = (jdir / "episodes.jsonl").read_text()
    meta = json.loads((jdir / "run_meta.json").read_text())
    assert meta["selection"]["levels"] == [1] and meta["identity"]["selection_sha"]
    assert {r["perturbation"]["level"] for r in records.read_jsonl(jdir / "episodes.jsonl")} == {1}
    # same command again = resume, nothing new
    assert sc_bench.main(argv(bat("l1", ", levels: [1]"), root, "--run-id", "sel")) == 0
    assert (jdir / "episodes.jsonl").read_text() == before
    # sc-bench: an edited battery in the same run is already refused by the run identity (battery sha256) ...
    with pytest.raises(SystemExit, match="another configuration"):
        sc_bench.main(argv(bat("l1", ", levels: [5]"), root, "--run-id", "sel"))
    # ... and the job dir itself refuses another selection at plan time (before any server starts), whatever
    # the battery: levels, inits, tasks, instruction mode
    ad = lp_adapter.LiberoPlusAdapter()
    ctx = Context(
        root / "runs" / "sel",
        {"ckpt_id": "pi05lib", "path_or_hub": "mock/pi05lib"},
        "mock",
        SHORTCUT.parents[1],
    )
    base = {
        "benchmark": "libero_plus",
        "suite": SUITE,
        "perturbation": "layout",
        "tasks": "1,8",
        "variants": 2,
    }
    assert ad.plan({**base, "levels": [1]}, ctx).n_pending == 0  # same selection: resume
    for change in ({"levels": [5]}, {"levels": [1], "episodes": 2}, {"levels": [1], "tasks": "8"},
                   {"levels": [1], "instruction": "registry"}, {"levels": [1], "variants": 1}):  # fmt: skip
        with pytest.raises(SystemExit, match="another variant selection"):
            ad.plan({**base, **change}, ctx)
    assert (jdir / "episodes.jsonl").read_text() == before
    assert json.loads((jdir / "run_meta.json").read_text())["selection"]["levels"] == [1]
    # under another job name (tag) it plans as a separate job dir; in a new run it simply runs
    p5 = ad.plan({**base, "levels": [5], "tag": "L5"}, ctx)
    assert p5.job_id == "plus-goal-layout-L5" and p5.n_pending == 4
    assert {v["level"] for u in p5.units for v in u["variants"]} == {5}
    assert sc_bench.main(argv(bat("l5", ", levels: [5]"), root, "--run-id", "sel5")) == 0
    l5 = records.read_jsonl(root / "runs" / "sel5" / "jobs" / "plus-goal-layout" / "episodes.jsonl")
    assert {r["perturbation"]["level"] for r in l5} == {5}


def test_gallery_choose_empty_pool_and_duplicates(tmp_path):
    """gallery.choose with a fake renderer: an axis without variants gives no example (no stale pick from the
    previous axis), and a candidate that renders like an earlier pick is skipped."""
    from bench.libero_plus import gallery

    vs = reg.load_variants(Path(json.loads((fake_fork(tmp_path) / "sc_env.json").read_text())["src"]), SUITE)
    rendered: list[str] = []

    def render(v: reg.PlusVariant) -> tuple:
        rendered.append(v.name)
        img: np.ndarray = np.full((8, 8, 3), 10 * (v.level or 0), np.uint8)  # same level → same image
        return img, np.zeros(7)

    # no camera variants in the fake fork: picks() gives [[]] → nothing chosen, nothing rendered
    assert gallery.picks(vs, 8, "camera", True) == [[]]
    assert gallery.choose(gallery.picks(vs, 8, "camera", True), render, "camera") == []
    assert gallery.choose([], render, "camera") == [] and rendered == []
    # the axis before an empty one must not leak into it (the old loop reused v/img of the previous axis)
    lay = gallery.choose(gallery.picks(vs, 8, "layout", True), render, "layout")
    assert [v.level for v, _, _ in lay] == [1, 5]
    assert gallery.choose([[]], render, "camera") == []
    # duplicates: the second list starts with an image equal to the first pick → next candidate
    layout8 = [v for v in vs if v.task_id == 8 and v.dimension == "layout"]
    a, b = (v for v in layout8 if v.level == 1)  # two level-1 variants: same fake image
    c = next(v for v in layout8 if v.level == 2)
    got = gallery.choose([[a], [b, c]], render, "layout")
    assert [v.name for v, _, _ in got] == [a.name, c.name]
    # all candidates repeat the first pick → no second example
    assert [v.name for v, _, _ in gallery.choose([[a], [b]], render, "layout")] == [a.name]
    # language: the scene never changes, the first candidate of each list is taken (no repeats of a variant)
    lang = gallery.picks(vs, 8, "language", True)
    got = gallery.choose(lang, render, "language")
    assert [v.level for v, _, _ in got] == [1, 5]
