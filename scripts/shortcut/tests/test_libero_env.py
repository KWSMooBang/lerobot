"""T009 LIBERO env fix: config selection (raw / fix, inherited LIBERO_CONFIG_PATH, cache key), the asset
redirect and init-state patch, resume and bench_version bookkeeping.

CPU only, no LIBERO install needed (hf-libero's libero.libero is replaced by a stand-in with the same
asset-path cache): uv run pytest scripts/shortcut/tests/test_libero_env.py -q
"""

import json
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parent
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "parallel")]

import common  # noqa: E402

PATCH = json.loads(common.INIT_PATCH.read_text())
PATCH_FILE, PATCH_LIFTS = next(iter(PATCH["files"].items()))  # a real entry of libero_init_patch.json


def png(srgb: bool) -> bytes:
    """1x1 RGB PNG, optionally with an sRGB chunk."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")

    ihdr = (1).to_bytes(4, "big") * 2 + bytes([8, 2, 0, 0, 0])
    body = chunk(b"IHDR", ihdr) + (chunk(b"sRGB", b"\x00") if srgb else b"")
    return (
        b"\x89PNG\r\n\x1a\n" + body + chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80")) + chunk(b"IEND", b"")
    )


def states_for(lifts: list[dict]) -> np.ndarray:
    """An init-state array whose patched entries hold the z0 the patch expects."""
    s = np.zeros((max(lf["init"] for lf in lifts) + 2, max(lf["state_index"] for lf in lifts) + 3))
    for lf in lifts:
        s[lf["init"], lf["state_index"]] = lf["z0"]
    return s


def fake_libero_pkg(default_assets: str) -> types.ModuleType:
    """Stand-in for hf-libero's libero.libero: get_assets_path() returns the cache if set, as hf-libero does."""
    mod = types.ModuleType("libero.libero")
    mod._assets_path_cache = None  # type: ignore[attr-defined]

    def get_assets_path() -> str:
        return mod._assets_path_cache or default_assets

    mod.get_assets_path = get_assets_path  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def fake_libero(tmp_path, monkeypatch):
    """Fake libero install + assets snapshot; HOME in tmp; a fresh 'process' (nothing configured yet)."""
    lib = tmp_path / "site" / "libero" / "libero"
    (lib / "init_files" / "libero_goal").mkdir(parents=True)
    (lib / "init_files" / "libero_goal" / "a.pruned_init").write_bytes(b"states-v1")
    patched = lib / "init_files" / PATCH_FILE
    patched.parent.mkdir(parents=True, exist_ok=True)
    torch.save(states_for(PATCH_LIFTS), patched)
    assets = tmp_path / "hub" / "snapshots" / "rev1"
    (assets / "textures").mkdir(parents=True)
    (assets / "textures" / "floor.png").write_bytes(png(srgb=True))
    (assets / "scenes").mkdir()
    (assets / "scenes" / "libero_floor_base_style.xml").write_text('<texture file="../textures/floor.png"/>')
    (assets / "scenes" / "libero_tabletop_base_style.xml").write_text(
        '<texture file="../textures/floor.png"/>'
    )
    (assets / "stable_hope_objects" / "milk").mkdir(parents=True)
    (assets / "stable_hope_objects" / "milk" / "texture_map.png").write_bytes(png(srgb=True))
    (assets / "stable_hope_objects" / "milk" / "plain.png").write_bytes(png(srgb=False))
    # ensure_libero_config writes os.environ directly: setenv first so monkeypatch restores the original
    for var in ("LIBERO_CONFIG_PATH", "SC_LIBERO_RAW"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(common, "_libero_dir", lambda: str(lib))
    monkeypatch.setattr(common, "_assets_snapshot", lambda: assets)
    monkeypatch.setattr(common, "_configured_env", None)
    pkg = fake_libero_pkg(str(tmp_path / "home" / ".cache" / "libero" / "assets"))
    monkeypatch.setitem(sys.modules, "libero", types.ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", pkg)
    return {"lib": lib, "assets": assets, "pkg": pkg, "default_assets": pkg.get_assets_path()}


def new_process(monkeypatch, fake=None):
    """A child process: inherits the environment, but nothing is configured or imported in it yet."""
    monkeypatch.setattr(common, "_configured_env", None)
    if fake is not None:
        fake["pkg"]._assets_path_cache = None


def init_states_of(cfg_dir: str) -> str:
    lines = Path(cfg_dir, "config.yaml").read_text().splitlines()
    return next(ln.split(": ", 1)[1] for ln in lines if ln.startswith("init_states:"))


def test_env_tag(monkeypatch):
    monkeypatch.setenv("SC_LIBERO_RAW", "1")
    assert common.libero_env_tag() == "raw"
    monkeypatch.setenv("SC_LIBERO_RAW", "0")
    assert common.libero_env_tag().startswith(f"fix-{common.LIBERO_FIX}-")


def test_fix_mirrors_redirect_and_patch(fake_libero):
    cfg = Path(common.ensure_libero_config())
    # hf-libero's asset path now points at the mirror
    assert fake_libero["pkg"].get_assets_path() == str(cfg / "assets")
    # object texture: sRGB chunk removed, other chunks kept; untagged textures stay symlinks
    tex = (cfg / "assets" / "stable_hope_objects" / "milk" / "texture_map.png").read_bytes()
    assert b"sRGB" not in tex and tex == common._strip_png_srgb(png(srgb=True))
    assert (cfg / "assets" / "stable_hope_objects" / "milk" / "plain.png").is_symlink()
    # floor scenes read textures_nosrgb/, table scenes are untouched (symlink to the original)
    assert "../textures_nosrgb/floor.png" in (cfg / "assets/scenes/libero_floor_base_style.xml").read_text()
    assert b"sRGB" not in (cfg / "assets" / "textures_nosrgb" / "floor.png").read_bytes()
    assert (cfg / "assets/scenes/libero_tabletop_base_style.xml").is_symlink()
    assert b"sRGB" in (cfg / "assets" / "textures" / "floor.png").read_bytes()
    # init states: the patched file is lifted entry by entry, the rest are symlinks to the originals
    assert init_states_of(str(cfg)) == str(cfg / "init_files")
    orig = states_for(PATCH_LIFTS)
    got = np.asarray(torch.load(cfg / "init_files" / PATCH_FILE, weights_only=False))
    want = orig.copy()
    for lf in PATCH_LIFTS:
        want[lf["init"], lf["state_index"]] += lf["dz"]
    assert np.array_equal(got, want) and not np.array_equal(got, orig)
    assert (cfg / "init_files" / "libero_goal" / "a.pruned_init").is_symlink()


def test_raw_uses_originals_and_no_redirect(fake_libero, monkeypatch):
    monkeypatch.setenv("SC_LIBERO_RAW", "1")
    cfg = common.ensure_libero_config()
    assert init_states_of(cfg) == str(fake_libero["lib"] / "init_files")
    assert fake_libero["pkg"].get_assets_path() == fake_libero["default_assets"]


def test_fix_then_raw_ignores_inherited_fix_config(fake_libero, monkeypatch):
    fix_dir = common.ensure_libero_config()
    new_process(monkeypatch, fake_libero)  # LIBERO_CONFIG_PATH=fix_dir is inherited
    monkeypatch.setenv("SC_LIBERO_RAW", "1")
    raw_dir = common.ensure_libero_config()
    assert raw_dir != fix_dir
    assert json.loads(Path(raw_dir, common.LIBERO_ENV_MARKER).read_text())["libero_env"] == "raw"
    assert init_states_of(raw_dir) == str(fake_libero["lib"] / "init_files")  # the unpatched originals
    assert fake_libero["pkg"].get_assets_path() == fake_libero["default_assets"]
    assert common.os.environ["LIBERO_CONFIG_PATH"] == raw_dir


def test_raw_then_fix_ignores_inherited_raw_config(fake_libero, monkeypatch):
    monkeypatch.setenv("SC_LIBERO_RAW", "1")
    raw_dir = common.ensure_libero_config()
    new_process(monkeypatch, fake_libero)
    monkeypatch.setenv("SC_LIBERO_RAW", "0")
    fix_dir = common.ensure_libero_config()
    assert fix_dir != raw_dir
    assert init_states_of(fix_dir) == str(Path(fix_dir, "init_files"))
    assert fake_libero["pkg"].get_assets_path() == str(Path(fix_dir, "assets"))


def test_matching_inherited_config_is_reused(fake_libero, monkeypatch):
    first = common.ensure_libero_config()
    stamp = Path(first, "config.yaml").stat().st_mtime_ns
    new_process(monkeypatch, fake_libero)
    assert common.ensure_libero_config() == first
    assert Path(first, "config.yaml").stat().st_mtime_ns == stamp  # not rebuilt


def test_inherited_config_without_marker_is_ignored(fake_libero, monkeypatch, tmp_path):
    legacy = tmp_path / "legacy_cfg"  # a pre-T009 config dir (no marker), e.g. from an older worktree
    legacy.mkdir()
    (legacy / "config.yaml").write_text("init_states: /somewhere/else\n")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(legacy))
    assert common.ensure_libero_config() != str(legacy)


def test_inherited_config_of_another_install_is_ignored(fake_libero, monkeypatch, tmp_path):
    """Another venv with the same hf-libero version, init files and assets revision (same source key)."""
    other = tmp_path / "other_venv" / "libero" / "libero"
    (other / "init_files").mkdir(parents=True)
    for f in (fake_libero["lib"] / "init_files").rglob("*"):
        if f.is_file():
            dst = other / "init_files" / f.relative_to(fake_libero["lib"] / "init_files")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(f.read_bytes())
    monkeypatch.setattr(common, "_libero_dir", lambda: str(other))
    theirs = common.ensure_libero_config()
    new_process(monkeypatch, fake_libero)
    monkeypatch.setattr(common, "_libero_dir", lambda: str(fake_libero["lib"]))
    ours = common.ensure_libero_config()
    assert ours != theirs
    assert str(fake_libero["lib"]) in Path(ours, "config.yaml").read_text()
    assert str(other) not in Path(ours, "config.yaml").read_text()


def test_mode_switch_in_one_interpreter_raises(fake_libero, monkeypatch):
    common.ensure_libero_config()
    monkeypatch.setenv("SC_LIBERO_RAW", "1")
    with pytest.raises(AssertionError, match="configured as fix-"):
        common.ensure_libero_config()


def test_source_change_gives_new_config(fake_libero, monkeypatch, tmp_path):
    first = common.ensure_libero_config()
    # same install path, reinstalled with other init states
    new_process(monkeypatch, fake_libero)
    (fake_libero["lib"] / "init_files" / "libero_goal" / "a.pruned_init").write_bytes(b"states-v2")
    second = common.ensure_libero_config()
    assert second != first
    # other assets revision
    new_process(monkeypatch, fake_libero)
    assets2 = tmp_path / "hub" / "snapshots" / "rev2"
    (assets2 / "textures").mkdir(parents=True)
    (assets2 / "scenes").mkdir()
    monkeypatch.setattr(common, "_assets_snapshot", lambda: assets2)
    third = common.ensure_libero_config()
    assert third not in (first, second)
    # other hf-libero version
    new_process(monkeypatch, fake_libero)
    from importlib import metadata

    monkeypatch.setattr(metadata, "version", lambda name: "9.9.9")
    assert common.ensure_libero_config() not in (first, second, third)


# ---------------------------------------------------------------- resume + bench_version


IDENT = {
    "policy": "pi05",
    "checkpoint": "c",
    "suite": "libero_goal",
    "condition": "C0",
    "repeat_idx": 0,
    "seed": 7,
}


def test_resume_refuses_raw_on_fix_and_fix_on_raw(tmp_path):
    ep = tmp_path / "episodes.jsonl"
    fix = {**IDENT, "libero_env": "fix-fix1-abcd"}
    with pytest.raises(SystemExit, match="libero_env"):
        common.check_resume(IDENT, {"identity": fix}, ep)  # fix dir, resumed with SC_LIBERO_RAW=1
    with pytest.raises(SystemExit, match="libero_env"):
        common.check_resume(fix, {"identity": IDENT}, ep)  # raw (P1) dir, resumed with the fix
    common.check_resume(IDENT, {"identity": dict(IDENT)}, ep)  # P1 raw dir, resumed raw: still fine
    common.check_resume(fix, {"identity": dict(fix)}, ep)


def test_bench_version_raw_unchanged_fix_suffixed(tmp_path, monkeypatch):
    from bench.ours import adapter

    instr = tmp_path / "i.json"
    instr.write_text("{}")
    monkeypatch.setenv("SC_LIBERO_RAW", "1")
    raw = adapter.bench_version(str(instr), "libero")
    assert raw.endswith(f"/hf-libero-{raw.rsplit('/hf-libero-', 1)[1]}")  # nothing after the version
    assert "/raw" not in raw and "/fix-" not in raw
    monkeypatch.setenv("SC_LIBERO_RAW", "0")
    assert adapter.bench_version(str(instr), "libero") == f"{raw}/{common.libero_env_tag()}"
