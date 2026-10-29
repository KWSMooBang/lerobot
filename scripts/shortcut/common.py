"""Shared helpers for the LIBERO shortcut-observation harness.

LIBERO setup (non-interactive config), policy loading (same path as lerobot-eval), instruction-set
handling and per-condition instruction strings. Adapted from `scripts/pid_demo/{common,setup_libero}.py`
on `exp/pid-demo`.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")


def _use_nvidia_egl() -> None:
    """This container ships libEGL_nvidia but glvnd only lists mesa, so MuJoCo renders on llvmpipe
    (~100 ms per 360px frame vs <1 ms). Point glvnd at the NVIDIA vendor library via a user-level
    vendor JSON; no system change. Must run before EGL is first loaded."""
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


_use_nvidia_egl()

SUITE_SHORT = {"libero_spatial": "spat", "libero_object": "obj", "libero_goal": "goal", "libero_10": "l10"}
CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
DUAL_GOAL_CONDITIONS = ("C4", "C5", "C6")  # no early termination, original + new goal tracked every step
# --policy picks the architecture; --checkpoint picks the weights (pi05lib = LIBERO finetune,
# pi05base = lerobot/pi05_base, not trained on LIBERO).
POLICY_CKPT = {"pi05": "lerobot/pi05_libero_finetuned"}
BASE_CKPT = {"pi05": "lerobot/pi05_base"}
# google/paligemma-3b-pt-224 (the tokenizer named in both checkpoints) is gated and no HF token is
# available. Same fallback as exp/pid-demo: the public big_vision PaliGemma sentencepiece model that
# OpenPI uses, loaded with BOS on / EOS off like the HF tokenizer config.
TOKENIZER_DIR = os.environ.get("SC_TOKENIZER_DIR", "/cache/pid_demo_dl/paligemma_tok_hf")
# where setup_server.sh puts gs://big_vision/paligemma_tokenizer.model on a new server
USER_TOKENIZER_DIR = Path.home() / ".cache" / "shortcut_libero" / "paligemma_tok_hf"


# ---------------------------------------------------------------- LIBERO config


# The LIBERO files (assets, BDDL, init states) are byte-identical to the original LIBERO repo, but the
# MuJoCo that lerobot pins (3.8) renders two things differently from the MuJoCo (<= 3.3.2) that rendered
# HuggingFaceVLA/libero:
#   * >= 3.3.3 honours the PNG sRGB chunk (texture colorspace "auto"); the floor textures have one, are
#     linearised and turn dark brown. The floor fills the view in the floor scenes (Object suite); in the
#     table scenes only a 2-pixel strip behind the table shows it. Of the object textures only the two salad
#     dressings (Object suite only) have the chunk; they render near-black instead of green.
#   * >= 3.4 computes deeper box-box penetrations; milk / orange juice / ketchup start ~2.7 cm inside the
#     table in the LIVING_ROOM_SCENE1/2 init states and are thrown off it during the settle steps.
# Fix (LIBERO_FIX): a mirror of the asset dir in which the object textures have no sRGB chunk (same pixels)
# and the floor scenes (FLOOR_SCENES) take their textures from textures_nosrgb/ (likewise), and a mirror of init_files with the penetrating
# objects lifted by their penetration depth (libero_init_patch.json, made by libero_init_patch.py).
# The table scenes are left alone so Spatial/Goal renders stay identical to the P1 pilot.
# SC_LIBERO_RAW=1 keeps the unfixed env.
LIBERO_FIX = "fix1"  # bump when the fix changes: it keys the cached config + mirrors
FLOOR_SCENES = "scenes/libero_floor_*_style.xml"
INIT_PATCH = Path(__file__).with_name("libero_init_patch.json")


def libero_raw() -> bool:
    return os.environ.get("SC_LIBERO_RAW", "0") == "1"


def _strip_png_srgb(data: bytes) -> bytes | None:
    """PNG bytes without the sRGB chunk (pixel data untouched); None if there is no sRGB chunk."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    out, i, found = [data[:8]], 8, False
    while i < len(data):
        n = int.from_bytes(data[i : i + 4], "big")
        end = i + 12 + n
        if data[i + 4 : i + 8] == b"sRGB":
            found = True
        else:
            out.append(data[i:end])
        i = end
    return b"".join(out) if found else None


def _publish_dir(tmp: Path, dst: Path) -> None:
    """Atomically move a finished temp dir into place (another process may have won the race)."""
    import shutil

    try:
        tmp.replace(dst)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not dst.exists():
            raise


def _fixed_assets(src: Path, dst: Path) -> Path:
    """Mirror `src` with symlinked files, except object textures with an sRGB chunk (copied without it), plus
    textures_nosrgb/ (textures/ with the sRGB chunk removed) used by the FLOOR_SCENES XMLs. The arenas swap
    the floor/wall file name by style but keep the XML's directory, so redirecting the directory reaches the
    style textures too.

    Directories are real so that relative paths in the scene XMLs (`../textures/...`) stay in the mirror.
    """
    if dst.exists():
        return dst
    tmp = dst.with_name(f"{dst.name}.tmp{os.getpid()}")
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if rel.parts[0].startswith("."):
            continue
        q = tmp / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        own_texture = p.suffix.lower() == ".png" and rel.parts[0] not in ("textures", "scenes")
        stripped = _strip_png_srgb(p.read_bytes()) if own_texture else None
        if stripped is None:
            q.symlink_to(p.resolve())
        else:
            q.write_bytes(stripped)
    nosrgb = tmp / "textures_nosrgb"
    nosrgb.mkdir()
    for p in sorted((src / "textures").iterdir()):
        stripped = _strip_png_srgb(p.read_bytes()) if p.suffix.lower() == ".png" else None
        if stripped is None:
            (nosrgb / p.name).symlink_to(p.resolve())
        else:
            (nosrgb / p.name).write_bytes(stripped)
    for scene in sorted(src.glob(FLOOR_SCENES)):
        q = tmp / scene.relative_to(src)
        q.unlink()
        q.write_text(scene.read_text().replace('file="../textures/', 'file="../textures_nosrgb/'))
    _publish_dir(tmp, dst)
    return dst


def _fixed_init_states(src: Path, dst: Path) -> Path:
    """Mirror init_files (symlinks) with the lifts of libero_init_patch.json applied."""
    if dst.exists():
        return dst
    import numpy as np
    import torch

    patch = json.loads(INIT_PATCH.read_text())
    tmp = dst.with_name(f"{dst.name}.tmp{os.getpid()}")
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        q = tmp / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        lifts = patch["files"].get(str(rel))
        if not lifts:
            q.symlink_to(p.resolve())
            continue
        states = np.array(torch.load(p, weights_only=False))  # nosec B614 (LIBERO's own numpy pickles)
        for lift in lifts:
            i, j = lift["init"], lift["state_index"]
            assert abs(states[i, j] - lift["z0"]) < 1e-6, (rel, lift)
            states[i, j] += lift["dz"]
        torch.save(states, q)
    _publish_dir(tmp, dst)
    return dst


LIBERO_ENV_MARKER = "libero_env.json"  # in each config dir: {"libero_env": tag, "source": source key}
_configured_env: str | None = None  # libero_env tag this interpreter's LIBERO was configured with


def libero_env_tag() -> str:
    """'raw' (SC_LIBERO_RAW=1, the env of the P1 pilot) or 'fix-<LIBERO_FIX>-<hash of the init patch>'."""
    if libero_raw():
        return "raw"
    patch_key = hashlib.sha1(INIT_PATCH.read_bytes(), usedforsecurity=False).hexdigest()[:8]
    return f"fix-{LIBERO_FIX}-{patch_key}"


def _libero_dir() -> str:
    spec = importlib.util.find_spec("libero")
    assert spec is not None and spec.submodule_search_locations is not None
    return os.path.join(list(spec.submodule_search_locations)[0], "libero")


def _assets_snapshot() -> Path:
    """lerobot/libero-assets snapshot dir (its name is the dataset revision); downloads it if not cached."""
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset", local_files_only=True)
        )
    except Exception:  # nosec B110 (not cached yet)
        return Path(snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset"))


def _source_key(libero_dir: str) -> str:
    """What the mirrors are built from: hf-libero version, init_files content, libero-assets revision."""
    from importlib import metadata

    h = hashlib.sha1(usedforsecurity=False)
    try:
        h.update(metadata.version("hf-libero").encode())
    except metadata.PackageNotFoundError:
        h.update(b"hf-libero-unknown")
    for f in sorted(Path(libero_dir, "init_files").rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(libero_dir)).encode())
            h.update(f.read_bytes())
    h.update(_assets_snapshot().name.encode())
    return h.hexdigest()[:8]


def _marker(tag: str, source: str, libero_dir: str) -> dict[str, str]:
    return {"libero_env": tag, "source": source, "libero_dir": str(Path(libero_dir).resolve())}


def _marker_matches(cfg_dir: Path, tag: str, source: str, libero_dir: str) -> bool:
    """The config dir was built for this env tag, from this source, for this libero install (another venv
    with the same versions still has its own bddl/init paths)."""
    try:
        marker = json.loads((cfg_dir / LIBERO_ENV_MARKER).read_text())
    except (OSError, ValueError):
        return False
    return marker == _marker(tag, source, libero_dir) and (cfg_dir / "config.yaml").exists()


def _build_config(cfg_dir: Path, libero_dir: str, raw: bool) -> None:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    assets = _assets_snapshot()
    init_states = Path(libero_dir) / "init_files"
    if not raw:
        assets = _fixed_assets(assets, cfg_dir / "assets")
        init_states = _fixed_init_states(init_states, cfg_dir / "init_files")
    tmp = cfg_dir / f"config.yaml.tmp{os.getpid()}"
    tmp.write_text(
        f"assets: {assets}\n"
        f"bddl_files: {libero_dir}/bddl_files\n"
        f"benchmark_root: {libero_dir}\n"
        f"datasets: {libero_dir}/../datasets\n"
        f"init_states: {init_states}\n"
    )
    tmp.replace(cfg_dir / "config.yaml")


def ensure_libero_config() -> str:
    """Point LIBERO at a config.yaml for *this* interpreter's libero install (no input() prompt).

    Must run before the first `import libero`. The config dir is keyed by the libero install path, the
    source key (hf-libero version, init_files content, assets revision) and the env tag (libero_env_tag()).
    Unless SC_LIBERO_RAW=1 it points at the fixed init states and LIBERO's asset path is redirected to the
    fixed asset mirror (see LIBERO_FIX). An inherited LIBERO_CONFIG_PATH is reused only if its marker names
    the same tag and source, otherwise it is ignored. Asking for another tag in the same interpreter after
    LIBERO was configured raises: LIBERO keeps the paths it read first.
    """
    global _configured_env
    tag = libero_env_tag()
    if _configured_env is not None and _configured_env != tag:
        raise AssertionError(
            f"LIBERO is configured as {_configured_env} in this process, now asked for {tag}"
        )
    libero_dir = _libero_dir()
    source = _source_key(libero_dir)
    inherited = os.environ.get("LIBERO_CONFIG_PATH")
    if inherited and _marker_matches(Path(inherited), tag, source, libero_dir):
        cfg_dir = Path(inherited)
    else:
        if inherited:
            print(
                f"[libero] ignoring inherited LIBERO_CONFIG_PATH={inherited} (not a {tag} config of this "
                "libero install)",
                flush=True,
            )
        path_key = hashlib.sha1(libero_dir.encode(), usedforsecurity=False).hexdigest()[:10]
        cfg_dir = Path.home() / ".cache" / "shortcut_libero" / f"{path_key}-{source}-{tag}"
        if not _marker_matches(cfg_dir, tag, source, libero_dir):
            _build_config(cfg_dir, libero_dir, raw=tag == "raw")
            marker = _marker(tag, source, libero_dir)
            (cfg_dir / LIBERO_ENV_MARKER).write_text(json.dumps(marker) + "\n")
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)
    _redirect_libero_assets(cfg_dir)
    _configured_env = tag
    return str(cfg_dir)


def _redirect_libero_assets(cfg_dir: Path) -> None:
    """hf-libero ignores the config's `assets` key (its get_assets_path() uses a package-local dir or
    ~/.cache/libero/assets); point its path cache at the config's assets so the fixed mirror is used."""
    if not (cfg_dir / "assets").is_dir():
        return
    import libero.libero as libero_pkg

    libero_pkg._assets_path_cache = str(cfg_dir / "assets")
    assert libero_pkg.get_assets_path() == str(cfg_dir / "assets"), "hf-libero asset path API changed"


# ---------------------------------------------------------------- instruction set


@dataclass
class Variant:
    """One instruction given to the policy for a (task, condition)."""

    condition: str
    variant: str  # file-name-safe id, e.g. "orig", "blank", "p0", "C4-a", "t3"
    instruction: str
    goal_bddl: str | None = None  # new-goal BDDL (C4–C6); None for C0–C3
    meta: dict[str, Any] = field(default_factory=dict)


def load_instruction_set(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    data["_by_task"] = {int(t["task_id"]): t for t in data["tasks"]}
    data["_path"] = str(path)
    return data


def resolve_path(p: str, instr_set_path: str | Path) -> str:
    """Resolve a BDDL path from the JSON: absolute, repo-relative, or relative to the JSON file."""
    cand = [Path(p), Path.cwd() / p, Path(instr_set_path).parent / p]
    repo_root = Path(__file__).resolve().parents[2]
    cand.append(repo_root / p)
    try:  # original LIBERO BDDLs may be given relative to libero's bddl_files dir
        from libero.libero import get_libero_path

        cand.append(Path(get_libero_path("bddl_files")) / p)
    except ImportError:
        pass
    for c in cand:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError(f"BDDL not found: {p} (tried {[str(c) for c in cand]})")


def nonsense_instruction(original: str, tokenizer: Any, seed: int) -> str:
    """Random whole-word vocab pieces, as many as the original has tokens (no BOS).

    Restricting to `▁[a-z]{2,8}` pieces keeps re-tokenized length close to the original.
    """
    n = len(tokenizer.encode(original, add_special_tokens=False))
    vocab = sorted(t for t in tokenizer.get_vocab() if re.fullmatch(r"▁[a-z]{2,8}", t))
    rng = random.Random(seed)
    return " ".join(rng.choice(vocab)[1:] for _ in range(n))


def variants_for(
    condition: str,
    task_entry: dict[str, Any],
    instr_set: dict[str, Any],
    tokenizer: Any,
    include_infeasible: bool = False,
) -> list[Variant]:
    original = task_entry["original"]
    if condition == "C0":
        return [Variant("C0", "orig", original)]
    if condition == "C1":
        return [Variant("C1", "blank", "")]
    if condition == "C2":
        seed = int(task_entry.get("nonsense_seed", 0)) * 1000 + int(task_entry["task_id"])
        return [
            Variant("C2", "nonsense", nonsense_instruction(original, tokenizer, seed), meta={"seed": seed})
        ]
    if condition == "C3":
        return [Variant("C3", f"p{i}", s) for i, s in enumerate(task_entry.get("paraphrases", []))]
    if condition in ("C4", "C5"):
        want = "object" if condition == "C4" else "target"
        out = []
        for sw in task_entry.get("swaps", []):
            if sw.get("type") != want:
                continue
            if sw.get("feasible") is False and not include_infeasible:
                continue
            out.append(
                Variant(
                    condition,
                    sw["id"],
                    sw["instruction"],
                    goal_bddl=resolve_path(sw["goal_bddl"], instr_set["_path"]),
                    meta={k: sw.get(k) for k in ("type", "feasible", "orig_target", "new_target")},
                )
            )
        return out
    if condition == "C6":
        out = []
        for j in task_entry.get("cross_task", []):
            other = instr_set["_by_task"].get(int(j)) or instr_set["_suite_tasks"][int(j)]
            out.append(
                Variant(
                    "C6",
                    f"t{int(j)}",
                    other["original"],
                    goal_bddl=resolve_path(other["bddl"], instr_set["_path"]),
                    meta={"cross_task_id": int(j)},
                )
            )
        return out
    raise ValueError(f"unknown condition {condition}")


def parse_range(spec: str) -> list[int]:
    """'0-9' / '0,3,5' / '2' -> list of ints."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def episode_seed(base: int, repeat_idx: int, task_id: int, init_idx: int) -> int:
    """Same seed for every condition at a given (task, init, repeat) → paired comparisons."""
    return base * 1_000_000 + repeat_idx * 10_000 + task_id * 100 + init_idx


def check_resume(identity: dict[str, Any], prev_meta: dict[str, Any] | None, ep_path: Path) -> None:
    """Refuse to append to an out dir written with another configuration (checkpoint, seed, condition,
    suite, repeat, instruction-set hash, ...). Existing episodes must also match it one by one."""
    if prev_meta is not None:
        prev = prev_meta.get("identity")
        if prev is None:
            raise SystemExit(
                f"{ep_path.parent}: run_meta.json has no identity (older harness); use a new --out"
            )
        prev = legacy_exec_identity(prev, prev_meta, identity)
        # libero_env is only in fixed-env identities (raw ones stay as before T009): absent = raw, both ways
        prev = {**prev, "libero_env": prev.get("libero_env", "raw")}
        new = {**identity, "libero_env": identity.get("libero_env", "raw")}
        diff = {k: (prev.get(k), v) for k, v in new.items() if prev.get(k) != v}
        if diff:
            raise SystemExit(f"{ep_path.parent} was written with another configuration (old, new): {diff}")
    elif ep_path.exists() and ep_path.stat().st_size > 0:
        raise SystemExit(f"{ep_path} exists without run_meta.json; cannot verify its configuration")
    if not ep_path.exists():
        return
    for line in ep_path.read_text().splitlines():
        r = json.loads(line)
        want = episode_seed(identity["seed"], identity["repeat_idx"], r["task_id"], r["init_idx"])
        got = (r["ckpt"], r["suite"], r["condition"], r["repeat_idx"], r["seed"])
        exp = (identity["checkpoint"], identity["suite"], identity["condition"], identity["repeat_idx"], want)
        if got != exp:
            raise SystemExit(f"{ep_path}: episode {got} does not match this run {exp}")


EXEC_KEYS = ("exec_mode", "gpu", "torch", "cuda")


def exec_identity(exec_mode: str, gpu: str | None) -> dict[str, Any]:
    """Execution fields of a run identity. bf16 batched kernels are not bit-identical to batch 1, so
    episodes from different exec modes or GPU models are kept apart (resume guard, metrics pairing)."""
    import torch

    return {"exec_mode": exec_mode, "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda}


def legacy_exec_identity(
    prev: dict[str, Any], prev_meta: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    """Run dirs written before exec fields existed (T001) were single-env runs; their GPU and torch are
    in the top level of run_meta.json. The CUDA build was not recorded, so it is taken as unchanged."""
    if all(k in prev for k in EXEC_KEYS) or not any(k in identity for k in EXEC_KEYS):
        return prev
    filled = dict(prev)
    filled.setdefault("exec_mode", "single")
    filled.setdefault("gpu", prev_meta.get("gpu"))
    filled.setdefault("torch", prev_meta.get("torch"))
    filled.setdefault("cuda", identity.get("cuda"))
    return filled


def video_relpath(
    suite: str, task_id: int, condition: str, variant: str, init_idx: int, repeat_idx: int
) -> str:
    rep = f"_r{repeat_idx}" if repeat_idx else ""
    return f"{suite}/task{task_id:02d}/{condition}_{variant}{rep}_init{init_idx:02d}"


# ---------------------------------------------------------------- policy


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer, GemmaTokenizer

    for d in (Path(TOKENIZER_DIR), USER_TOKENIZER_DIR):
        if (d / "tokenizer.model").exists():
            return GemmaTokenizer.from_pretrained(str(d), add_bos_token=True, add_eos_token=False)
    return AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")


LIBERO_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def needs_libero_adapter(ckpt: str) -> bool:
    """True when the checkpoint was not trained on LIBERO keys (e.g. lerobot/pi05_base:
    base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb, 32-d state, no normalization stats)."""
    import lerobot.policies  # noqa: F401  (registers policy config types)
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    return not all(k in (cfg.input_features or {}) for k in LIBERO_IMAGE_KEYS)


def model_tag(ckpt: str) -> str:
    known = {"lerobot/pi05_libero_finetuned": "pi05lib", "lerobot/pi05_base": "pi05base"}
    return known.get(ckpt, re.sub(r"[^A-Za-z0-9]+", "-", ckpt.split("/")[-1]))


def load_policy(
    policy_type: str,
    ckpt: str,
    suite: str,
    n_action_steps: int,
    device: str = "cuda",
    libero_adapter: str | None = None,
    compile_model: bool = False,
) -> tuple:
    """Load like lerobot-eval (policy.path=ckpt) with a fixed n_action_steps. Returns
    (policy, pre, post, env_pre, env_post, tokenizer, inference_meta).

    With `libero_adapter` (a LIBERO-finetuned checkpoint of the same architecture), the weights come
    from `ckpt` but the config (input/output features: agentview→image, wrist→image2, empty third
    camera, 8-d state, 7-d action; MEAN_STD normalization; dtype) and the pre/post processors
    (HuggingFaceVLA/libero normalization stats, tokenizer settings) come from the adapter, so a
    non-LIBERO checkpoint sees exactly the inputs the LIBERO finetune sees.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors

    cfg_src = libero_adapter or ckpt
    cfg = PreTrainedConfig.from_pretrained(cfg_src)
    if cfg.type != policy_type:
        raise ValueError(f"--policy {policy_type} but checkpoint {cfg_src} is type {cfg.type}")
    adapter_meta = None
    if libero_adapter:
        own = PreTrainedConfig.from_pretrained(ckpt)
        if own.type != cfg.type:
            raise ValueError(f"adapter {libero_adapter} ({cfg.type}) != checkpoint {ckpt} ({own.type})")
        adapter_meta = {
            "weights_from": ckpt,
            "config_and_processors_from": libero_adapter,
            "matched": [
                "input features: observation.images.image (agentview), observation.images.image2 (wrist), "
                "observation.images.empty_camera_0 (padding), observation.state 8-d; replaces "
                "base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb (same slot order) and 32-d state",
                "output feature: action 7-d (model still predicts max_action_dim=32, first 7 used)",
                "normalization: MEAN_STD with HuggingFaceVLA/libero stats from the adapter's "
                "normalizer/unnormalizer (checkpoint ships none)",
                "pre/post processor pipeline and tokenizer settings of the adapter "
                "(no relative_actions/absolute_actions steps)",
                f"dtype {getattr(cfg, 'dtype', None)} (checkpoint stored {getattr(own, 'dtype', None)})",
            ],
            "checkpoint_own_input_features": {
                k: list(v.shape) for k, v in (own.input_features or {}).items()
            },
            "checkpoint_own_normalization_mapping": (
                {k: str(v) for k, v in own.normalization_mapping.items()}
                if own.normalization_mapping
                else None
            ),
        }
    meta = {
        "policy_type": cfg.type,
        "chunk_size": cfg.chunk_size,
        "n_action_steps_stored": cfg.n_action_steps,
        "n_action_steps": n_action_steps,
        "num_inference_steps": getattr(cfg, "num_inference_steps", None),
        "image_resolution": list(getattr(cfg, "image_resolution", []) or []),
        "tokenizer_max_length": getattr(cfg, "tokenizer_max_length", None),
        "dtype": getattr(cfg, "dtype", None),
        "temporal_ensembling": False,
        "compile_model": compile_model,
    }
    meta["libero_adapter"] = adapter_meta
    cfg.pretrained_path = Path(cfg_src)
    cfg.device = device
    cfg.n_action_steps = n_action_steps
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = compile_model
    env_cfg = LiberoEnvConfig(task=suite)
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, pretrained_path=ckpt)
    policy.eval()
    tokenizer = load_tokenizer()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg_src,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": {}},
            "tokenizer_processor": {"tokenizer": tokenizer},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    meta["env_observation_hw"] = [env_cfg.observation_height, env_cfg.observation_width]
    meta["control_mode"] = env_cfg.control_mode
    return policy, pre, post, env_pre, env_post, tokenizer, meta


@functools.cache  # hub calls once per checkpoint (run_multi plans many runs)
def checkpoint_training_info(ckpt: str) -> dict[str, Any]:
    """Mixed vs per-suite training, read from the checkpoint's train_config.json and model card."""
    from huggingface_hub import HfApi, ModelCard, hf_hub_download

    info: dict[str, Any] = {"checkpoint": ckpt}
    try:
        info["revision"] = HfApi().model_info(ckpt).sha
    except Exception as e:
        info["revision_error"] = repr(e)
    try:
        with open(hf_hub_download(ckpt, "train_config.json")) as f:
            tc = json.load(f)
        ds = tc.get("dataset") or {}
        info.update(
            dataset_repo_id=ds.get("repo_id"),
            dataset_episodes=ds.get("episodes"),
            steps=tc.get("steps"),
            batch_size=tc.get("batch_size"),
            init_from=(tc.get("policy") or {}).get("pretrained_path"),
        )
        # HuggingFaceVLA/libero holds all four suites (spatial/object/goal/10) in one dataset.
        mixed = ds.get("repo_id") == "HuggingFaceVLA/libero" and ds.get("episodes") is None
        info["training_mix"] = "mixed-4-suites" if mixed else "unknown"
    except Exception as e:
        info["train_config_error"] = repr(e)
        if ckpt in BASE_CKPT.values():
            info["training_mix"] = "not-libero-trained (pi05 base, no train_config.json on hub)"
    try:
        text = ModelCard.load(ckpt).text
        info["model_card_excerpt"] = " ".join(text.split()[:60])
    except Exception as e:
        info["model_card_error"] = repr(e)
    return info
