"""First-frame previews: base task vs variant (agentview | wrist, policy view = 180° rotated), one PNG per
variant + gallery.html / gallery.md. Renders on the CPU (Mesa llvmpipe EGL) unless SC_VARIANTS_GPU_RENDER=1.

The frames are what the policy sees at t=0 of init 0: lerobot LiberoEnv reset → set_init_state →
10 no-op steps (the eval path), with the T009 fixed assets (ensure_libero_config).
"""

from __future__ import annotations

import html
import json
import multiprocessing as mp
import os
import textwrap
from pathlib import Path
from typing import Any

MESA_EGL = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"


def cpu_render_env() -> None:
    """Must run before common.py is imported (it points EGL at the NVIDIA vendor library)."""
    if os.environ.get("SC_VARIANTS_GPU_RENDER") == "1":
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    if Path(MESA_EGL).exists():
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = MESA_EGL


cpu_render_env()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SIDE = 256
DEFAULT_OUT = Path(os.environ.get("SC_OUT_ROOT", "/outputs/VLA-shortcut-exps")) / "analysis" / "variants"


def render_first(
    suite: str, task_id: int, bddl: str | None, init_states: str | None, init: int = 0
) -> np.ndarray:
    """[agentview | wrist] at t=0 (after the eval settle), policy orientation, SIDE px each."""
    from .generate import lib, load_init_states

    lib()
    from lerobot.envs.libero import LiberoEnv, _get_suite

    env = LiberoEnv(
        _get_suite(suite), task_id, suite, obs_type="pixels_agent_pos",
        observation_height=SIDE, observation_width=SIDE,
    )  # fmt: skip
    if bddl:
        env._task_bddl_file = bddl
    if init_states:
        env._init_states = load_init_states(init_states)
    env.init_state_id = init
    obs, _ = env.reset(seed=0)
    env.close()
    views = [np.ascontiguousarray(obs["pixels"][k][::-1, ::-1]) for k in ("image", "image2")]
    return np.concatenate(views, axis=1)


def _caption(lines: list[tuple[str, tuple[int, int, int]]], width: int) -> np.ndarray:
    wrapped: list[tuple[str, tuple[int, int, int]]] = []
    for text, color in lines:
        wrapped += [(w, color) for w in textwrap.wrap(text, 78) or [""]]
    img: np.ndarray = np.full((8 + 15 * len(wrapped), width, 3), 30, np.uint8)
    for i, (text, color) in enumerate(wrapped):
        cv2.putText(img, text, (6, 16 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return img


def _render_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        img = render_first(job["suite"], job["task_id"], job.get("bddl"), job.get("init_states"))
        np.save(job["out"], img)
        return {"key": job["key"], "ok": True}
    except Exception as e:  # noqa: BLE001
        return {"key": job["key"], "ok": False, "error": f"{type(e).__name__}: {e}"}


def compose(m: dict[str, Any], base_img: np.ndarray, var_img: np.ndarray) -> np.ndarray:
    white, grey, green, amber = (255, 255, 255), (190, 190, 190), (120, 255, 120), (255, 200, 80)
    w = base_img.shape[1]
    goal = " & ".join("(" + " ".join(g) + ")" for g in m["goal"]) or "none (no achievable goal)"
    feas = m.get("feasibility", {})
    top = _caption([(f"BASE  t{m['base_task']['task_id']}: {m['base_task']['language']}", grey)], w)
    edits = ", ".join(m["edits"]["kinds"]) or "none"
    bottom = _caption(
        [
            (f"VARIANT {m['id']}: {m['instruction']}", white),
            (f"goal: {goal}", green),
            (
                f"edits: {edits} | init: {m['init_states']['mode']} ({m['init_states']['n']}) | "
                f"feasibility: {feas.get('status', 'not checked')} {feas.get('trials', '')}",
                amber,
            ),
        ],
        w,
    )
    return np.concatenate([top, base_img, bottom, var_img], axis=0)


def preview(
    manifests: list[dict[str, Any]], out: str | Path = DEFAULT_OUT, workers: int = 8, log: Any = print
) -> Path:
    """Render, compose, and write <out>/<suite>/<id>.png + <out>/gallery.{html,md}. Returns the html path."""
    from .spec import manifest_file

    out = Path(out)
    cache = out / ".frames"
    cache.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, dict[str, Any]] = {}
    for m in manifests:
        b = m["base_task"]
        bkey = f"base-{m['suite']}-t{b['task_id']}-{b['init_file_sha256'][:12]}"
        jobs.setdefault(bkey, {"key": bkey, "suite": m["suite"], "task_id": b["task_id"]})
        vkey = f"var-{m['suite']}-{m['id']}-{m['files']['bddl']['sha256'][:8]}{m['files']['init_states']['sha256'][:8]}"
        jobs[vkey] = {
            "key": vkey, "suite": m["suite"], "task_id": b["task_id"],
            "bddl": str(manifest_file(m, "bddl")), "init_states": str(manifest_file(m, "init_states")),
        }  # fmt: skip
        m["_frames"] = (bkey, vkey)
    for j in jobs.values():
        j["out"] = str(cache / f"{j['key']}.npy")
    todo = [j for j in jobs.values() if not Path(j["out"]).exists()]
    if todo:
        log(
            f"[preview] rendering {len(todo)} first frames on {'GPU' if os.environ.get('SC_VARIANTS_GPU_RENDER') == '1' else 'CPU'}"
        )
        with mp.get_context("spawn").Pool(max(1, min(workers, len(todo)))) as pool:
            for r in pool.imap_unordered(_render_job, todo):
                if not r["ok"]:
                    log(f"[preview] {r['key']}: {r['error']}")
    entries = []
    for m in manifests:
        bkey, vkey = m["_frames"]
        try:
            img = compose(m, np.load(cache / f"{bkey}.npy"), np.load(cache / f"{vkey}.npy"))
        except FileNotFoundError:
            log(f"[preview] {m['suite']}/{m['id']}: no frame")
            continue
        p = out / m["suite"] / f"{m['id']}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img[:, :, ::-1])
        entries.append((m, p.relative_to(out)))
    html_path = write_gallery(entries, out)
    log(f"[preview] {len(entries)} previews → {html_path}")
    return html_path


def write_gallery(entries: list[tuple[dict[str, Any], Path]], out: Path) -> Path:
    rows_md = [
        "# LIBERO variants — first frames (base task above, variant below)",
        "",
        "| suite | id | group | instruction | goal | edits | init | feasibility |",
        "|---|---|---|---|---|---|---|---|",
    ]
    cards = []
    for m, rel in entries:
        goal = " ∧ ".join("(" + " ".join(g) + ")" for g in m["goal"]) or "—"
        feas = m.get("feasibility", {})
        fs = f"{feas.get('status', 'not checked')} {feas.get('trials', '')}".strip()
        edits = ", ".join(m["edits"]["kinds"]) or "—"
        rows_md.append(
            f"| {m['suite']} | [{m['id']}]({rel}) | {m.get('group') or ''} | {m['instruction']} | {goal} | "
            f"{edits} | {m['init_states']['mode']} ({m['init_states']['n']}) | {fs} |"
        )
        e = html.escape
        cards.append(
            f'<div class="card"><img src="{e(str(rel))}" loading="lazy"><p><b>{e(m["suite"])}/{e(m["id"])}</b>'
            f" [{e(str(m.get('group') or ''))}]<br>{e(m['instruction'])}<br><code>{e(goal)}</code><br>"
            f"edits: {e(edits)} · init {e(m['init_states']['mode'])} · feasibility <b>{e(fs)}</b></p></div>"
        )
    (out / "gallery.md").write_text("\n".join(rows_md) + "\n")
    page = (
        "<!doctype html><meta charset=utf-8><title>LIBERO variants</title><style>"
        "body{font-family:sans-serif;background:#111;color:#ddd}.card{display:inline-block;vertical-align:top;"
        "width:520px;margin:8px;background:#222;padding:6px}.card img{width:512px}code{color:#9f9}</style>"
        f"<h1>LIBERO variants ({len(entries)})</h1>" + "".join(cards)
    )
    p = out / "gallery.html"
    p.write_text(page)
    (out / "gallery.json").write_text(
        json.dumps([{"suite": m["suite"], "id": m["id"], "png": str(rel)} for m, rel in entries], indent=1)
    )
    return p
