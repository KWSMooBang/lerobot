"""T009's LIBERO env fix for the LIBERO-Plus / LIBERO-PRO installs (T019). Runs in any venv (stdlib + numpy).

T009 fixed two MuJoCo-3.8 differences against the training renders in our env (common.py, LIBERO_FIX): PNG
textures with an sRGB chunk render linearised (Object-suite floor dark brown, salad dressings near-black), and
LIVING_ROOM_SCENE1/2 init states start objects inside the table so they are thrown off during the settle steps.
LIBERO-Plus and LIBERO-PRO ship the same assets / init files and MuJoCo 3.8.1 (T019 bench_render_compare.py:
PRO Object floor + salad dressings, Plus salad dressings, both L10 t0/t1/t7 thrown objects). Same fix here:

  assets  `apply_assets(<fork>/libero/libero/assets)`, in place (both forks build asset paths from their package
          dir, not from config.yaml, so a mirror would need patching the forks): object textures with an sRGB
          chunk are rewritten without it (same pixels), textures/ gets textures_nosrgb/ (sRGB chunk removed) and
          the floor scene XMLs point at it; table scenes untouched (as in T009). Originals are kept under
          `<assets>/../assets_srgb_orig/`; a marker `<assets>/.sc_render_fix.json` records it. Idempotent.
          The setup scripts run it (`render_fix.py apply --assets DIR`).
  inits   `patch_init_states(rel, states)` in the env loaders: the lifts of libero_init_patch.json for the
          original init files (`libero_10/LIVING_ROOM_SCENE…​.pruned_init`), applied only where the stored z
          still equals the patch's z0 (PRO's re-sampled axis inits are other states and stay as they are).

`fix_tag(assets)` = `fix-<LIBERO_FIX>-<patch hash>` (= common.libero_env_tag) when the assets carry the marker,
else "raw"; it is appended to the benchmark's bench_version so fixed and unfixed results never pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

SHORTCUT = Path(__file__).resolve().parents[1]
INIT_PATCH = SHORTCUT / "libero_init_patch.json"
MARKER = ".sc_render_fix.json"
FLOOR_SCENES = "scenes/libero_floor_*_style.xml"


def _common() -> Any:
    if str(SHORTCUT) not in sys.path:
        sys.path.insert(0, str(SHORTCUT))
    import common

    return common


def patch_key() -> str:
    return hashlib.sha1(INIT_PATCH.read_bytes(), usedforsecurity=False).hexdigest()[:8]


def fix_tag(assets: Path | str) -> str:
    m = Path(assets) / MARKER
    if not m.exists():
        return "raw"
    return f"fix-{json.loads(m.read_text())['fix']}-{patch_key()}"


def _write(path: Path, data: bytes) -> None:
    """Replace `path` (file or symlink) with a real file holding `data`, never writing through a symlink."""
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    path.unlink(missing_ok=True)
    tmp.replace(path)


def apply_assets(assets: Path | str) -> dict[str, Any]:
    """Apply the T009 asset fix to an asset dir in place; returns what changed. Idempotent."""
    common = _common()
    strip = common._strip_png_srgb
    assets = Path(assets)
    if (assets / MARKER).exists():
        return json.loads((assets / MARKER).read_text())
    backup = assets.parent / "assets_srgb_orig"
    changed: list[str] = []
    for p in sorted(assets.rglob("*.png")):
        rel = p.relative_to(assets)
        if rel.parts[0] in ("textures", "scenes", "textures_nosrgb") or rel.parts[0].startswith("."):
            continue
        data = p.read_bytes()
        stripped = strip(data)
        if stripped is None:
            continue
        b = backup / rel
        b.parent.mkdir(parents=True, exist_ok=True)
        if not b.exists():
            b.write_bytes(data)
        _write(p, stripped)
        changed.append(str(rel))
    nosrgb = assets / "textures_nosrgb"
    nosrgb.mkdir(exist_ok=True)
    n_tex = 0
    for p in sorted((assets / "textures").iterdir()):
        q = nosrgb / p.name
        stripped = strip(p.read_bytes()) if p.suffix.lower() == ".png" else None
        q.unlink(missing_ok=True)
        if stripped is None:
            q.symlink_to(p.resolve())
        else:
            q.write_bytes(stripped)
            n_tex += 1
    floors = []
    for scene in sorted(assets.glob(FLOOR_SCENES)):
        text = scene.read_text()
        if 'file="../textures/' not in text:
            continue
        b = backup / scene.relative_to(assets)
        b.parent.mkdir(parents=True, exist_ok=True)
        if not b.exists():
            b.write_text(text)
        _write(scene, text.replace('file="../textures/', 'file="../textures_nosrgb/').encode())
        floors.append(scene.name)
    info = {
        "fix": common.LIBERO_FIX,
        "objects_stripped": changed,
        "textures_stripped": n_tex,
        "floor_scenes": floors,
        "backup": str(backup),
    }
    (assets / MARKER).write_text(json.dumps(info, indent=1) + "\n")
    return info


_PATCH: dict[str, Any] | None = None


def patch_init_states(rel: str, states: Any) -> tuple[Any, int]:
    """`states` (n, D) of the init file `<suite>/<name>.pruned_init`, with the T009 lifts applied where the
    stored value equals the patch's z0. Returns (states, number of lifts applied)."""
    global _PATCH
    import numpy as np

    if _PATCH is None:
        _PATCH = json.loads(INIT_PATCH.read_text())
    lifts = _PATCH["files"].get(rel)
    if not lifts:
        return states, 0
    out = np.array(states, copy=True)
    n = 0
    for lift in lifts:
        i, j = lift["init"], lift["state_index"]
        if i < len(out) and j < out.shape[1] and abs(out[i, j] - lift["z0"]) < 1e-6:
            out[i, j] += lift["dz"]
            n += 1
    return out, n


def mirror(src: Path, dst: Path) -> Path:
    """Copy a fork's `libero/libero` package dir to `dst` with assets/ as a symlink farm (real dirs, symlinked
    files): `apply_assets(dst/assets)` then fixes the copy without touching `src` (verification runs)."""
    if dst.exists():
        return dst
    data = ("assets", "bddl_files", "init_files", "datasets", "__pycache__")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*data), symlinks=True)
    for d in data[1:4]:
        if (src / d).exists():
            (dst / d).symlink_to((src / d).resolve())
    for p in sorted((src / "assets").rglob("*")):
        q = dst / "assets" / p.relative_to(src / "assets")
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            q.symlink_to(p.resolve())
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apply", help="fix an asset dir in place (idempotent)")
    a.add_argument("--assets", type=Path, required=True)
    s = sub.add_parser("status")
    s.add_argument("--assets", type=Path, required=True)
    m = sub.add_parser("mirror", help="symlink-farm copy of a fork's libero/libero dir (for verification)")
    m.add_argument("src", type=Path)
    m.add_argument("dst", type=Path)
    args = ap.parse_args()
    if args.cmd == "apply":
        info = apply_assets(args.assets)
        print(
            f"[render_fix] {args.assets}: {fix_tag(args.assets)}; {len(info['objects_stripped'])} object textures, "
            f"{info['textures_stripped']} floor textures without sRGB, floor scenes {info['floor_scenes']}"
        )
    elif args.cmd == "status":
        print(fix_tag(args.assets))
    else:
        print(mirror(args.src, args.dst))


if __name__ == "__main__":
    main()
