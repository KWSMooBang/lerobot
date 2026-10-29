"""T009 LIBERO env fix: config selection (raw / fix, inherited LIBERO_CONFIG_PATH, cache key).

CPU only, no LIBERO install needed: uv run pytest scripts/shortcut/tests/test_libero_env.py -q
"""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402


@pytest.fixture
def fake_libero(tmp_path, monkeypatch):
    """Fake libero install + assets snapshot; HOME in tmp; a fresh 'process' (nothing configured yet)."""
    lib = tmp_path / "site" / "libero" / "libero"
    (lib / "init_files" / "libero_goal").mkdir(parents=True)
    (lib / "init_files" / "libero_goal" / "a.pruned_init").write_bytes(b"states-v1")
    assets = tmp_path / "hub" / "snapshots" / "rev1"
    (assets / "textures").mkdir(parents=True)
    (assets / "textures" / "floor.png").write_bytes(b"not a png")
    (assets / "scenes").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # ensure_libero_config writes os.environ directly: setenv first so monkeypatch restores the original
    for var in ("LIBERO_CONFIG_PATH", "SC_LIBERO_RAW"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    monkeypatch.setattr(common, "_libero_dir", lambda: str(lib))
    monkeypatch.setattr(common, "_assets_snapshot", lambda: assets)
    monkeypatch.setattr(common, "_redirect_libero_assets", lambda cfg_dir: None)
    monkeypatch.setattr(common, "_configured_env", None)
    return {"lib": lib, "assets": assets}


def new_process(monkeypatch):
    """A child process: inherits the environment, but nothing is configured in its interpreter yet."""
    monkeypatch.setattr(common, "_configured_env", None)


def init_states_of(cfg_dir: str) -> str:
    lines = Path(cfg_dir, "config.yaml").read_text().splitlines()
    return next(ln.split(": ", 1)[1] for ln in lines if ln.startswith("init_states:"))


def test_env_tag(monkeypatch):
    monkeypatch.setenv("SC_LIBERO_RAW", "1")
    assert common.libero_env_tag() == "raw"
    monkeypatch.setenv("SC_LIBERO_RAW", "0")
    assert common.libero_env_tag().startswith(f"fix-{common.LIBERO_FIX}-")


def test_fix_then_raw_ignores_inherited_fix_config(fake_libero, monkeypatch):
    fix_dir = common.ensure_libero_config()
    assert json.loads(Path(fix_dir, common.LIBERO_ENV_MARKER).read_text())["libero_env"].startswith("fix-")
    assert init_states_of(fix_dir) == str(Path(fix_dir, "init_files"))  # the patched mirror

    new_process(monkeypatch)  # LIBERO_CONFIG_PATH=fix_dir is inherited
    monkeypatch.setenv("SC_LIBERO_RAW", "1")
    raw_dir = common.ensure_libero_config()
    assert raw_dir != fix_dir
    assert json.loads(Path(raw_dir, common.LIBERO_ENV_MARKER).read_text())["libero_env"] == "raw"
    assert init_states_of(raw_dir) == str(fake_libero["lib"] / "init_files")  # the unpatched originals
    assert common.os.environ["LIBERO_CONFIG_PATH"] == raw_dir


def test_raw_then_fix_ignores_inherited_raw_config(fake_libero, monkeypatch):
    monkeypatch.setenv("SC_LIBERO_RAW", "1")
    raw_dir = common.ensure_libero_config()
    new_process(monkeypatch)
    monkeypatch.setenv("SC_LIBERO_RAW", "0")
    fix_dir = common.ensure_libero_config()
    assert fix_dir != raw_dir
    assert init_states_of(fix_dir) == str(Path(fix_dir, "init_files"))


def test_matching_inherited_config_is_reused(fake_libero, monkeypatch):
    first = common.ensure_libero_config()
    stamp = Path(first, "config.yaml").stat().st_mtime_ns
    new_process(monkeypatch)
    assert common.ensure_libero_config() == first
    assert Path(first, "config.yaml").stat().st_mtime_ns == stamp  # not rebuilt


def test_inherited_config_without_marker_is_ignored(fake_libero, monkeypatch, tmp_path):
    legacy = tmp_path / "legacy_cfg"  # a pre-T009 config dir (no marker), e.g. from an older worktree
    legacy.mkdir()
    (legacy / "config.yaml").write_text("init_states: /somewhere/else\n")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(legacy))
    assert common.ensure_libero_config() != str(legacy)


def test_mode_switch_in_one_interpreter_raises(fake_libero, monkeypatch):
    common.ensure_libero_config()
    monkeypatch.setenv("SC_LIBERO_RAW", "1")
    with pytest.raises(AssertionError, match="configured as fix-"):
        common.ensure_libero_config()


def test_source_change_gives_new_config(fake_libero, monkeypatch, tmp_path):
    first = common.ensure_libero_config()
    # same install path, reinstalled with other init states
    new_process(monkeypatch)
    (fake_libero["lib"] / "init_files" / "libero_goal" / "a.pruned_init").write_bytes(b"states-v2")
    second = common.ensure_libero_config()
    assert second != first
    # other assets revision
    new_process(monkeypatch)
    assets2 = tmp_path / "hub" / "snapshots" / "rev2"
    (assets2 / "textures").mkdir(parents=True)
    (assets2 / "scenes").mkdir()
    monkeypatch.setattr(common, "_assets_snapshot", lambda: assets2)
    third = common.ensure_libero_config()
    assert third not in (first, second)
    # other hf-libero version
    new_process(monkeypatch)
    from importlib import metadata

    monkeypatch.setattr(metadata, "version", lambda name: "9.9.9")
    assert common.ensure_libero_config() not in (first, second, third)
