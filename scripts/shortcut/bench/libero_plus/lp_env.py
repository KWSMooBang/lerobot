"""LIBERO-Plus env side (runs in the Plus venv, never in the main one).

    bootstrap()                         # before the first `import libero`: config, sys.path, EGL, Wand
    env = PlusEnv(variant, hw=360)      # one registered variant (or the original task)
    obs = env.reset(init_idx, seed)     # LIBERO init state + 10 settle steps, like lerobot's LiberoEnv
    obs, success = env.step(action)

Mirrors lerobot `LiberoEnv(is_libero_plus=True)`: OffScreenRenderEnv on the variant's registered BDDL name
(ControlEnv parses the `_view_…_initstate_…_noise_…` suffix itself), hard reset, `set_init_state` from the
original task's init file (`libero_newobj/` single state for `_add_` / `_level` layouts), 10 no-op settle
steps, relative (delta) control, success = `env.check_success()`. Observations stay raw (LIBERO's
upside-down render); `bench.core.client.robosuite_obs_to_wire` makes the wire obs.

`python lp_env.py selftest` = import + one env per axis (setup_libero_plus.sh runs it).
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parents[1]
REPO = SHORTCUT.parents[1]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from bench.libero_plus.registry import PlusVariant, original_variant  # noqa: E402

NUM_STEPS_WAIT = 10  # lerobot LiberoEnv default
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
_SUFFIX_RE = re.compile(r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+")  # lerobot get_task_init_states


def venv_dir() -> Path:
    return Path(os.environ.get("LIBERO_PLUS_VENV") or REPO / ".venvs" / "libero_plus")


def sc_env() -> dict[str, Any]:
    p = venv_dir() / "sc_env.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run scripts/shortcut/bench/libero_plus/setup_libero_plus.sh")
    return json.loads(p.read_text())


def src_dir() -> Path:
    return Path(sc_env()["src"])


def _use_nvidia_egl() -> None:
    """Same as common._use_nvidia_egl (common.py is not importable here): point glvnd at the NVIDIA EGL
    vendor library when it only lists mesa (llvmpipe rendering is ~100× slower)."""
    if "__EGL_VENDOR_LIBRARY_FILENAMES" in os.environ:
        return
    vendor_dirs = ("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d")
    if any(Path(d, "10_nvidia.json").exists() for d in vendor_dirs):
        return
    if not any(
        Path(d, "libEGL_nvidia.so.0").exists() for d in ("/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu")
    ):
        return
    vendor = Path.home() / ".cache" / "shortcut_libero" / "10_nvidia.json"
    if not vendor.exists():
        vendor.parent.mkdir(parents=True, exist_ok=True)
        vendor.write_text('{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n')
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor)


class WandMissingError(RuntimeError):
    pass


def _stub_wand() -> None:
    """env_wrapper.py imports Wand at module level, but only the motion-blur noise (1–10) calls it. Without
    libMagickWand (apt libmagickwand-6.q16-6) those variants raise WandMissingError; everything else runs."""

    def missing(*_a: Any, **_k: Any) -> Any:
        raise WandMissingError(
            "libMagickWand not installed (apt libmagickwand-6.q16-6): motion-blur noise unavailable"
        )

    class _Fn:
        argtypes: Any = None

        def __call__(self, *a: Any, **k: Any) -> Any:
            return missing()

    class _Lib:
        def __getattr__(self, name: str) -> _Fn:
            fn = _Fn()
            setattr(self, name, fn)
            return fn

    class _Image:
        def __init__(self, *a: Any, **k: Any) -> None:
            missing()

    wand = types.ModuleType("wand")
    api = types.ModuleType("wand.api")
    image = types.ModuleType("wand.image")
    api.library = _Lib()  # type: ignore[attr-defined]
    image.Image = _Image  # type: ignore[attr-defined]
    wand.api, wand.image = api, image  # type: ignore[attr-defined]
    sys.modules.update({"wand": wand, "wand.api": api, "wand.image": image})


_BOOTED: dict[str, Any] = {}


def bootstrap() -> dict[str, Any]:
    """Config, import path, EGL vendor and Wand; idempotent. Returns {"wand": bool, "src": str, "sha": str}."""
    if _BOOTED:
        return _BOOTED
    env = sc_env()
    os.environ["LIBERO_CONFIG_PATH"] = env["libero_config_path"]
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ.get("MUJOCO_GL") == "egl":
        _use_nvidia_egl()
    if env["src"] not in sys.path:  # the fork is a namespace package: an editable install alone is not enough
        sys.path.insert(0, env["src"])
    try:
        import wand.image  # noqa: F401

        has_wand = True
    except ImportError:
        _stub_wand()
        has_wand = False
    _BOOTED.update(wand=has_wand, src=env["src"], sha=env.get("sha", "unknown"))
    return _BOOTED


def bench_version() -> str:
    b = bootstrap()
    return f"libero-plus-{b['sha'][:7]}"


def bddl_name(v: PlusVariant) -> str:
    """Registered name → the bddl_file_name ControlEnv expects (suffix kept, it parses view/initstate/noise)."""
    from libero.libero import get_libero_path

    return os.path.join(get_libero_path("bddl_files"), v.suite, f"{v.name}.bddl")


def init_states(v: PlusVariant) -> np.ndarray:
    """lerobot get_task_init_states(is_libero_plus=True), numpy-only."""
    import torch
    from libero.libero import get_libero_path

    root = Path(get_libero_path("init_states"))
    fname = f"{v.name}.pruned_init"
    if "_add_" in fname or "_level" in fname:
        st = torch.load(root / "libero_newobj" / v.suite / fname, weights_only=False)  # nosec B614
        return np.asarray(st).reshape(1, -1)
    stem = _SUFFIX_RE.sub("", v.name)
    return np.asarray(torch.load(root / v.suite / f"{stem}.pruned_init", weights_only=False))  # nosec B614


def instruction(v: PlusVariant) -> str:
    """What LIBERO-Plus gives the policy: the `_language_` BDDL's sentence for language variants, else the
    original task's filename-derived instruction (the fork's grab_language_from_filename)."""
    from libero.libero.benchmark import grab_language_from_filename

    return str(grab_language_from_filename(v.suite, v.name + ".bddl"))


class PlusEnv:
    def __init__(self, v: PlusVariant, hw: int = 360, control_mode: str = "relative"):
        bootstrap()
        from libero.libero.envs import OffScreenRenderEnv

        self.v = v
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl_name(v), camera_heights=hw, camera_widths=hw, control_freq=20, hard_reset=True
        )
        self.init_states = init_states(v)
        self.control_mode = control_mode
        self.n_init = len(self.init_states)

    def reset(self, init_idx: int, seed: int) -> dict[str, Any]:
        np.random.seed(seed % 2**32)
        self.env.seed(seed)
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[init_idx % self.n_init])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = self.env.step(DUMMY_ACTION)
        for robot in self.env.robots:
            robot.controller.use_delta = self.control_mode == "relative"
        return obs

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], bool]:
        obs, _, _, _ = self.env.step(np.asarray(action, dtype=np.float64))
        return obs, bool(self.env.check_success())

    def close(self) -> None:
        self.env.close()


def selftest() -> None:
    from bench.core.client import robosuite_obs_to_wire
    from bench.libero_plus.registry import load_variants

    b = bootstrap()
    print(f"[selftest] LIBERO-Plus {b['sha'][:7]} from {b['src']}; Wand {'ok' if b['wand'] else 'STUBBED'}")
    vs = load_variants(Path(b["src"]), "libero_goal")
    picks = [original_variant("libero_goal", 8)]
    for dim in ("language", "layout", "camera", "robot", "lighting", "texture", "noise"):
        picks.append(
            next(v for v in vs if v.task_id == 8 and v.dimension == dim and v.detail != "motion_blur")
        )
    for v in picks:
        env = PlusEnv(v, hw=128)
        obs = env.reset(0, 7)
        w = robosuite_obs_to_wire(obs)
        obs, ok = env.step(np.zeros(7))
        print(f"[selftest] {v.dimension:<9} {v.name[:70]:<70} init {env.n_init:>2} img {w['image'].shape} ok")
        env.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        raise SystemExit("usage: lp_env.py selftest")
