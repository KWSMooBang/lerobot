"""CPU checks for T019 (testbed selection): resolve_path without importing libero, catalogs, --list,
battery validation, dry-run GPU-h, n_action_steps auto, sc-train --then-bench / --cpu plan, render_fix.

    uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_testbed_select.py -q
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack")

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parent
REPO = SHORTCUT.parents[1]
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "parallel"), str(SHORTCUT / "train"), str(HERE)]

from bench import render_fix  # noqa: E402
from bench.core import runner as sc_bench  # noqa: E402
from bench.core.adapters import ADAPTERS, get_adapter  # noqa: E402
from bench.core.battery import load_battery  # noqa: E402
from bench.core.catalog import CatalogEntry, all_entries, format_table  # noqa: E402
from bench.core.validate import validate_battery  # noqa: E402

HAVE_LIBERO = importlib.util.find_spec("libero") is not None
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def clean_env(home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "LIBERO_CONFIG_PATH"}
    env["HOME"] = str(home)
    env.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))  # assets cache stays reachable
    return env


# ---------------------------------------------------------------- 1. libero config bug


@pytest.mark.skipif(not HAVE_LIBERO, reason="hf-libero not installed")
def test_resolve_path_does_not_import_libero(tmp_path):
    code = (
        "import sys; import common; "
        "p = common.resolve_path('libero_goal/put_the_bowl_on_the_plate.bddl', 'x.json'); "
        "assert p.endswith('libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl'), p; "
        "assert 'libero.libero' not in sys.modules; print('ok')"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=SHORTCUT, env=clean_env(tmp_path), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr[-2000:]
    assert not (tmp_path / ".libero").exists()


def tokenizer_dir() -> Path | None:
    from common import TOKENIZER_DIR, USER_TOKENIZER_DIR

    for d in (Path(TOKENIZER_DIR), USER_TOKENIZER_DIR):
        if (d / "tokenizer.model").exists():
            return d
    return None


@pytest.mark.skipif(not HAVE_LIBERO, reason="hf-libero not installed")
def test_empty_home_dry_run_and_plan_finish(tmp_path):
    """HOME empty, no LIBERO_CONFIG_PATH, stdin at EOF: sc-bench --dry-run (ours C0/C4 → resolve_path) and the
    run_multi plan finish instead of waiting in LIBERO's input() prompt / reading another venv's ~/.libero."""
    tok = tokenizer_dir()
    if tok is None:
        pytest.skip("no local PaliGemma tokenizer copy")
    home = tmp_path / "home"
    home.mkdir()
    env = {**clean_env(home), "SC_TOKENIZER_DIR": str(tok)}
    runs = [
        [sys.executable, str(SHORTCUT / "bench" / "core" / "runner.py"), "--ckpt", "pi05lib", "--battery", "smoke",
         "--dry-run", "--out-root", str(tmp_path / "out")],
        [sys.executable, str(SHORTCUT / "parallel" / "run_multi.py"), "--dry-run", "--conds", "C0,C4", "--suites",
         "goal", "--episodes", "2", "--tasks", "1,8", "--no-bookkeeping", "--out-root", str(tmp_path / "out")],
    ]  # fmt: skip
    for cmd in runs:
        r = subprocess.run(
            cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900
        )
        assert r.returncode == 0, (cmd, r.stdout[-1500:], r.stderr[-1500:])
        assert "total" in r.stdout
    assert not (home / ".libero").exists()


# ---------------------------------------------------------------- 2. catalog, --list, validation

REQUIRED = set(CatalogEntry.__annotations__) - {
    "count_by_level",
    "count_by_detail",
    "tasks",
    "options",
    "source",
}


def test_catalog_interface_all_adapters(monkeypatch, tmp_path):
    monkeypatch.setenv("LIBERO_PLUS_VENV", str(tmp_path / "no_plus"))
    monkeypatch.setenv("LIBERO_PRO_ROOT", str(tmp_path / "no_pro"))
    for b in ADAPTERS:
        es = get_adapter(b).catalog(None)
        assert es, b
        for e in es:
            assert set(e) >= REQUIRED, (b, REQUIRED - set(e))
            assert e["benchmark"] == b and e["suite"] in SUITES
            assert e["key_field"] == get_adapter(b).key_field
        assert {e["suite"] for e in es} == set(SUITES)
    ours = {(e["suite"], e["key"]): e for e in get_adapter("ours").catalog("libero_goal")}
    assert [k for _, k in ours] == ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    assert ours["libero_goal", "C4"]["details"] and all(
        d.startswith("C4-") for d in ours["libero_goal", "C4"]["details"]
    )
    plus = get_adapter("libero_plus").catalog("libero_goal")
    assert [e["key"] for e in plus] == [
        "none",
        "language",
        "layout",
        "camera",
        "robot",
        "lighting",
        "texture",
        "noise",
    ]
    assert all(e["missing"] for e in plus)  # no venv here → every entry reports it
    pro = get_adapter("libero_pro").catalog("libero_object")
    pos = next(e for e in pro if e["key"] == "position")
    assert pos["levels"][0] == "swap" and "y0.5" in pos["levels"] and pos["count"] == 110
    assert pro[0]["options"]["instruction_source"] == ["training", "pro_bddl", "filename"]


def test_plus_catalog_from_registry_and_motion_blur(monkeypatch, tmp_path):
    from test_libero_plus import fake_fork

    monkeypatch.setenv("LIBERO_PLUS_VENV", str(fake_fork(tmp_path)))
    es = {e["key"]: e for e in get_adapter("libero_plus").catalog("libero_goal")}
    assert es["language"]["count"] == 20 and es["language"]["levels"] == [1, 2, 3, 4, 5]
    assert es["noise"]["details"] == ["motion_blur", "gaussian_blur"]
    assert es["noise"]["missing_by_detail"]["motion_blur"]  # fake fork: magick false
    assert not es["language"]["missing"]
    # a suite the registry does not have: static entry, count unknown
    assert get_adapter("libero_plus").catalog("libero_10")[1]["count"] is None


def test_list_cli_table_and_json(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("LIBERO_PLUS_VENV", str(tmp_path / "no_plus"))
    assert sc_bench.main(["--list", "--benchmark", "ours", "--suite", "libero_goal"]) == 0
    out = capsys.readouterr().out
    assert "== ours" in out and "C4-a" in out and "libero_plus" not in out
    assert sc_bench.main(["--list", "--json", "--suite", "libero_goal,libero_10"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["benchmark"] for r in rows} == set(ADAPTERS)
    assert {r["suite"] for r in rows} == {"libero_goal", "libero_10"}
    assert "MISSING" in format_table(all_entries("libero_plus", "libero_goal"))


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / f"{name}.yaml"
    p.write_text(text)
    return p


def test_validation_errors_with_suggestions(tmp_path):
    bad = write(
        tmp_path,
        "bad",
        "name: bad\njobs:\n"
        "  - {benchmark: libero_plsu, suites: [libero_goal]}\n"
        "  - {benchmark: ours, suites: [libero_gaol], conditions: [C0]}\n"
        "  - {benchmark: ours, suites: [libero_goal], conditions: [C9, C4], variants: [C4-z], episods: 3}\n"
        "  - {benchmark: libero_plus, suites: [libero_goal], perturbations: [lightning]}\n"
        "  - {benchmark: libero_pro, suites: [libero_goal], perturbations: [position], levels: [x0.1],"
        " instruction_source: pro_bdl}\n"
        "  - {benchmark: ours, suites: [libero_goal], conditions: [C0], tasks: '8-11'}\n",
    )
    v = validate_battery(load_battery(str(bad)))
    text = "\n".join(v.errors)
    for want in (
        "unknown benchmark 'libero_plsu' (did you mean 'libero_plus'?)",
        "unknown suite 'libero_gaol' (did you mean 'libero_goal'?)",
        "unknown key 'episods' (did you mean 'episodes'?)",
        "unknown condition 'C9'",
        "unknown variant 'C4-z'",
        "unknown perturbation 'lightning' (did you mean 'lighting'?)",
        "unknown level 'x0.1' (known: swap)",
        "instruction_source 'pro_bdl' (did you mean 'pro_bddl'?)",
        "tasks [10, 11] outside 0-9",
    ):
        assert want in text, (want, text)
    with pytest.raises(SystemExit, match="error"):
        sc_bench.main(["--validate", "--battery", str(bad)])


def test_validation_skips_missing_installs_and_warns_motion_blur(monkeypatch, tmp_path):
    from test_libero_plus import fake_fork

    bat = write(
        tmp_path,
        "mix",
        "name: mix\njobs:\n"
        "  - {benchmark: ours, suites: [libero_goal], conditions: [C0]}\n"
        "  - {benchmark: libero_plus, suites: [libero_goal], perturbations: [noise], tasks: '8'}\n"
        "  - {benchmark: libero_pro, suites: [libero_goal], perturbations: [task]}\n",
    )
    monkeypatch.setenv("LIBERO_PLUS_VENV", str(fake_fork(tmp_path)))
    monkeypatch.setenv("LIBERO_PRO_ROOT", str(tmp_path / "no_pro"))
    v = validate_battery(load_battery(str(bat)), backend="libero")
    assert not v.errors
    assert [j["benchmark"] for j in v.jobs] == ["ours", "libero_plus"]
    assert any("libero_pro libero_goal task: SKIPPED" in s for s in v.skipped)
    assert any("motion_blur variants left out" in w for w in v.warnings)
    # mock backend (tests / e2e --mock): nothing is skipped for missing venvs
    assert len(validate_battery(load_battery(str(bat)), backend="mock").jobs) == 3


@pytest.mark.parametrize("name", ["quick", "full", "smoke", "custom_template", "e2e_smoke"])
def test_shipped_batteries_validate(name):
    v = validate_battery(load_battery(name), backend="mock")
    assert not v.errors, v.errors


def test_dry_run_prints_gpu_hours(capsys, tmp_path):
    assert sc_bench.main(
        ["--ckpt", "mock/pi05lib", "--backend", "mock", "--battery", "smoke", "--dry-run",
         "--out-root", str(tmp_path), "--clients-per-gpu", "4", "--est-step-ms", "20"]
    ) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "GPU-h (K=4)" in out and "incl. model load" in out and "env step 20.0 ms" in out
    line = next(x for x in out.splitlines() if x.startswith("ours-goal-C0"))
    assert float(line.split()[-1]) >= 0


# ---------------------------------------------------------------- 3. train → bench


def test_n_action_steps_auto(tmp_path):
    cfg = tmp_path / "ck"
    cfg.mkdir()
    (cfg / "config.json").write_text(json.dumps({"type": "sc_template", "chunk_size": 8}))
    row = {"ckpt_id": "x", "path_or_hub": str(cfg), "adapter": None, "notes": "preset y"}
    assert sc_bench.resolve_n_action_steps("auto", row, "libero") == 8
    (cfg / "config.json").write_text(json.dumps({"type": "pi05", "chunk_size": 50}))
    assert sc_bench.resolve_n_action_steps("auto", row, "libero") == 10
    assert sc_bench.resolve_n_action_steps("auto", {**row, "notes": "n_action_steps=5"}, "libero") == 5
    assert sc_bench.resolve_n_action_steps("7", row, "libero") == 7


def test_sc_train_cpu_plan_and_then_bench_cmd(tmp_path):
    import argparse

    import sc_train

    ns = argparse.Namespace(
        preset="sc_template_libero_all", run_id="t", gpus="0", steps=3, batch=4, effective_batch=4,
        variant_eval=None, out_root=str(tmp_path), data_variant=None, resume=False, dry_run=True, wandb="off",
        launcher="auto", ckpt_id=None, no_register=False, allow_any_gpu=False, notes="", cpu=True,
        then_bench="e2e_smoke", bench_args="--backend mock --episodes 1", extra=[],
    )  # fmt: skip
    plan = sc_train.build_plan(ns)
    assert plan["cpu"] and plan["gpus"] == [] and plan["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "--policy.device=cpu" in plan["cmd"] and plan["launcher"] == "python"
    cmd = sc_train.bench_cmd(ns, "t", Path(plan["out_root"]), plan)
    assert cmd[1].endswith("bench/core/runner.py")
    assert cmd[2:] == ["--ckpt", "t", "--battery", "e2e_smoke", "--out-root", plan["out_root"], "--backend",
                       "mock", "--episodes", "1"]  # fmt: skip


def test_engine_registers_local_plugins_and_checks_action_dim():
    pytest.importorskip("torch")
    sys.path.insert(0, str(SHORTCUT / "serve"))
    import engine

    from lerobot.configs.policies import PreTrainedConfig

    assert "lerobot_policy_sc_template" in engine.register_plugins()
    assert "sc_template" in PreTrainedConfig.get_known_choices()

    class Feat:
        shape = (14,)

    class Cfg:
        type = "x"
        output_features = {"action": Feat()}

    with pytest.raises(SystemExit, match="7-d actions"):
        engine.check_action_space(Cfg())


# ---------------------------------------------------------------- 4. render fix


def png(srgb: bool) -> bytes:
    import struct
    import zlib

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    body = chunk(b"IHDR", ihdr) + (chunk(b"sRGB", b"\x00") if srgb else b"")
    return (
        b"\x89PNG\r\n\x1a\n" + body + chunk(b"IDAT", zlib.compress(b"\x00\x01\x02\x03")) + chunk(b"IEND", b"")
    )


def test_render_fix_assets_in_mirror_and_init_patch(tmp_path):
    src = tmp_path / "fork" / "libero" / "libero"
    a = src / "assets"
    (a / "textures").mkdir(parents=True)
    (a / "scenes").mkdir()
    (a / "stable_hope_objects" / "salad_dressing").mkdir(parents=True)
    (a / "textures" / "floor.png").write_bytes(png(True))
    (a / "textures" / "wall.png").write_bytes(png(False))
    (a / "stable_hope_objects" / "salad_dressing" / "texture_map.png").write_bytes(png(True))
    (a / "scenes" / "libero_floor_base_style.xml").write_text('<texture file="../textures/floor.png"/>')
    (a / "scenes" / "libero_tabletop_base_style.xml").write_text('<texture file="../textures/floor.png"/>')
    (src / "bddl_files").mkdir()
    orig = {p: p.read_bytes() for p in a.rglob("*") if p.is_file()}
    dst = render_fix.mirror(src, tmp_path / "copy" / "libero" / "libero")
    assert render_fix.fix_tag(dst / "assets") == "raw"
    info = render_fix.apply_assets(dst / "assets")
    assert info["objects_stripped"] == ["stable_hope_objects/salad_dressing/texture_map.png"]
    assert info["floor_scenes"] == ["libero_floor_base_style.xml"] and info["textures_stripped"] == 1
    assert {p: p.read_bytes() for p in a.rglob("*") if p.is_file()} == orig  # the source is untouched
    fixed = dst / "assets"
    assert b"sRGB" not in (fixed / "stable_hope_objects" / "salad_dressing" / "texture_map.png").read_bytes()
    assert "textures_nosrgb" in (fixed / "scenes" / "libero_floor_base_style.xml").read_text()
    assert "textures_nosrgb" not in (fixed / "scenes" / "libero_tabletop_base_style.xml").read_text()
    assert (fixed / "textures_nosrgb" / "wall.png").is_symlink()
    import common

    assert render_fix.fix_tag(fixed) == f"fix-{common.LIBERO_FIX}-{render_fix.patch_key()}"
    assert render_fix.apply_assets(fixed) == info  # idempotent

    patch = json.loads(render_fix.INIT_PATCH.read_text())["files"]
    rel, lifts = next(iter(patch.items()))
    n = max(x["init"] for x in lifts) + 1
    d = max(x["state_index"] for x in lifts) + 1
    states = np.zeros((n, d))
    for x in lifts:
        states[x["init"], x["state_index"]] = x["z0"]
    out, k = render_fix.patch_init_states(rel, states)
    assert k == len(lifts) and not np.shares_memory(out, states)
    assert all(abs(out[x["init"], x["state_index"]] - x["z0"] - x["dz"]) < 1e-9 for x in lifts)
    same, k0 = render_fix.patch_init_states(rel, np.zeros((n, d)))  # other states (PRO re-sampled): untouched
    assert k0 == 0 and not same.any()
    assert render_fix.patch_init_states("libero_goal/x.pruned_init", states)[1] == 0
