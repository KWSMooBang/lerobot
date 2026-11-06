"""T018 step 1: the fixed observation sets of the language probe (CPU; run once, then every server uses the files).

    uv run --extra shortcut-bench python scripts/shortcut/analysis/langprobe/make_obs.py \
        --out /outputs/VLA-shortcut-exps/analysis/langprobe/obs [--sets A,B] [--workers 20]

Set A (common, all models; 4 suites × 10 tasks × init_idx 0–4 = 200): the harness's initial observation. The
`ours` env (lerobot LiberoEnv, OSC, T009 fix mode unless SC_LIBERO_RAW=1) is reset exactly like EpisodeStepper
(np.random.seed(seed), init_state_id = init_idx, reset(seed) incl. LiberoEnv's 10 settle steps) with
seed = common.episode_seed(7, 0, task, init) (sc-bench's default --seed 7, repeat 0). Stored per observation:
  image_{256,360}, image2_{256,360}  agentview / wrist, upright (wire convention), rendered at 256 (StarVLA,
                                     DROID servers) and 360 (lerobot servers; LiberoEnvConfig) by two envs reset
                                     the same way (their sim states are checked to be identical)
  state (8)                          wire state, computed with lerobot's LiberoProcessorStep (bench/ours/worker.py)
  joint_pos (7)                      robot0_joint_pos (DROID server)
  sim_state, sim_state_len           MuJoCo flattened state (time, qpos, qvel), NaN-padded; qacc_warmstart (nv)
  gripper_action (2)                 Panda gripper controller's accumulated command (robosuite internal state)
  eef_pos (3), eef_quat (4, xyzw)    raw robosuite eef pose
and in set_A.meta.json per row: obs_id, suite, task_id, init_idx, seed, original instruction, object positions
(every entry of LIBERO's object_states_dict: objects and regions/sites, world m), sim layout (nq, nv).

Set B (auxiliary, no DROID: the dataset has no joint state; 40 tasks × 2 episodes × progress 25/50/75 % = 240):
frames of HuggingFaceVLA/libero (local copy of revision 86958911 at <out-root>/data/hf_libero, images upright
256 px as stored). Episodes: the two lowest episode_index of each task. Frame = round(p · (length − 1)).
Stored: image_256, image2_256, state (8), gt_action (10, 7: the dataset actions of frames t … t+9, edge-padded).

manifest.json: per set the npz / meta sha256, composition, renderer (MUJOCO_GL), LIBERO env tag, git sha.
`--backend mock`: tiny synthetic sets from parallel/mock_backend.py (tests; no LIBERO, no dataset).
Rendering: CPU (MUJOCO_GL=osmesa by default here); GPU EGL renders can differ from these pixels slightly, which
is why the files are fixed and shipped to the server instead of being re-rendered there.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess  # nosec B404
import sys
import time
import zipfile
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parents[1]
sys.path[:0] = [p for p in (str(HERE), str(SHORTCUT), str(SHORTCUT / "parallel")) if p not in sys.path]

import lp_common as lp  # noqa: E402

SIZES_A = (256, 360)
GT_ACTIONS = 10
VOLATILE = ("git_sha", "created_at")  # manifest only


def git_sha() -> str:
    try:
        return subprocess.check_output(  # nosec B603 B607
            ["git", "-C", str(SHORTCUT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- set A (LIBERO sim)


def _env_spec(suite: str, task_id: int, hw: int) -> dict[str, Any]:
    return {
        "suite": suite,
        "task_id": task_id,
        "max_steps": 600,
        "env_hw": [hw, hw],
        "control_mode": "relative",
    }


def sim_snapshot(raw_env: Any) -> dict[str, Any]:
    """MuJoCo state + the controller-side state the restore needs (see measure.restore_sim)."""
    e = raw_env.env if hasattr(raw_env, "env") else raw_env
    sim = e.sim
    robot = e.robots[0]
    return {
        "sim_state": np.asarray(sim.get_state().flatten(), dtype=np.float64),
        "qacc_warmstart": np.array(sim.data.qacc_warmstart, dtype=np.float64),
        "gripper_action": np.array(robot.gripper.current_action, dtype=np.float64),
        "nq": int(sim.model.nq),
        "nv": int(sim.model.nv),
    }


def object_positions(raw_env: Any) -> dict[str, list[float]]:
    e = raw_env.env if hasattr(raw_env, "env") else raw_env
    out = {}
    for name, st in e.object_states_dict.items():
        try:
            out[name] = [float(x) for x in st.get_geom_state()["pos"]]
        except Exception:  # noqa: S112  # nosec B112  (a state without a geom pose: skipped)
            continue
    return out


def task_obs_libero(suite: str, task_id: int, inits: list[int]) -> list[dict[str, Any]]:
    """Set A observations of one task (runs in a worker process)."""
    from bench.core.client import upright
    from bench.ours.worker import wire_obs
    from common import libero_env_tag
    from envproc import _make_env
    from run_eval import inner

    suites: dict[str, Any] = {}
    envs = {hw: _make_env("libero", _env_spec(suite, task_id, hw), suites) for hw in SIZES_A}
    out = []
    for init_idx in inits:
        seed = lp.episode_seed(lp.SEED_BASE["A"], 0, task_id, init_idx)
        obs: dict[int, Any] = {}
        snaps = {}
        for hw, env in envs.items():  # EpisodeStepper.reset
            np.random.seed(seed % 2**32)
            env.init_state_id = init_idx
            obs[hw], _ = env.reset(seed=seed)
            snaps[hw] = sim_snapshot(inner(env))
        a, b = (snaps[hw]["sim_state"] for hw in SIZES_A)
        if a.shape != b.shape or not np.array_equal(a, b):
            raise RuntimeError(
                f"{suite} task {task_id} init {init_idx}: sim state differs between render sizes"
            )
        o = obs[SIZES_A[0]]
        w = wire_obs(o)
        rs = o["robot_state"]
        rec: dict[str, Any] = {
            "suite": suite,
            "task_id": task_id,
            "init_idx": init_idx,
            "seed": seed,
            "env_tag": libero_env_tag(),
            "objects": object_positions(inner(envs[SIZES_A[0]])),
            "state": w["state"],
            "joint_pos": w["joint_pos"],
            "eef_pos": np.asarray(rs["eef"]["pos"], dtype=np.float64),
            "eef_quat": np.asarray(rs["eef"]["quat"], dtype=np.float64),
            **snaps[SIZES_A[0]],
        }
        for hw in SIZES_A:
            rec[f"image_{hw}"] = upright(obs[hw]["pixels"]["image"])
            rec[f"image2_{hw}"] = upright(obs[hw]["pixels"]["image2"])
        out.append(rec)
    for env in envs.values():
        env.close()
    return out


def task_obs_mock(suite: str, task_id: int, inits: list[int]) -> list[dict[str, Any]]:
    from bench.core.client import lerobot_libero_obs_to_wire
    from mock_backend import MockEnv, mock_suite, text_pos

    sets = lp.load_instruction_sets()
    entry = lp.task_entry(sets, suite, task_id)
    names = {entry.get("original", "")}
    for sw in entry.get("swaps", []):
        names |= {n for k in ("orig_target", "new_target") for n in str(sw.get(k) or "").split(",") if n}
    out = []
    for init_idx in inits:
        seed = lp.episode_seed(lp.SEED_BASE["A"], 0, task_id, init_idx)
        imgs = {}
        for hw in SIZES_A:
            env = MockEnv(mock_suite(suite).task_info[task_id], 50, (hw // 8, hw // 8))
            np.random.seed(seed % 2**32)
            env.init_state_id = init_idx
            o, _ = env.reset(seed=seed)
            imgs[hw] = o
        o = imgs[SIZES_A[0]]
        w = lerobot_libero_obs_to_wire(o)
        rec: dict[str, Any] = {
            "suite": suite,
            "task_id": task_id,
            "init_idx": init_idx,
            "seed": seed,
            "env_tag": "mock",
            "objects": {n: [float(x) for x in text_pos(n)] for n in sorted(names) if n},
            "state": w["state"],
            "joint_pos": np.zeros(7, np.float32),
            "eef_pos": np.asarray(o["robot_state"]["eef"]["pos"], dtype=np.float64),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "sim_state": np.r_[o["robot_state"]["eef"]["pos"], -1.0].astype(np.float64),
            "qacc_warmstart": np.zeros(1),
            "gripper_action": np.zeros(2),
            "nq": 3,
            "nv": 1,
        }
        for hw in SIZES_A:  # nearest-neighbour upscale of the tiny mock frame to the nominal size
            for slot in ("image", "image2"):
                small = np.asarray(imgs[hw]["pixels"][slot])
                rec[f"{slot}_{hw}"] = np.ascontiguousarray(
                    np.repeat(np.repeat(small, 8, 0), 8, 1)[::-1, ::-1]
                )
        out.append(rec)
    return out


def _task_job(job: tuple[str, str, int, list[int]]) -> list[dict[str, Any]]:
    backend, suite, task_id, inits = job
    import torch

    torch.set_num_threads(1)
    fn = task_obs_mock if backend == "mock" else task_obs_libero
    return fn(suite, task_id, inits)


def pad_stack(vals: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = max(len(v) for v in vals)
    out = np.full((len(vals), n), np.nan)
    for i, v in enumerate(vals):
        out[i, : len(v)] = v
    return out, np.array([len(v) for v in vals], dtype=np.int64)


def build_set_a(
    args: argparse.Namespace, tasks: list[tuple[str, int]]
) -> tuple[dict[str, np.ndarray], list[dict]]:
    sets = lp.load_instruction_sets()
    inits = list(range(args.inits))
    jobs = [(args.backend, s, t, inits) for s, t in tasks]
    t0 = time.time()
    if args.workers > 1:
        with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            parts = pool.map(_task_job, jobs, chunksize=1)
    else:
        parts = [_task_job(j) for j in jobs]
    recs = [r for p in parts for r in p]
    print(f"[make_obs] set A: {len(recs)} observations in {time.time() - t0:.0f} s", flush=True)
    arrays: dict[str, np.ndarray] = {}
    for hw in SIZES_A:
        for slot in ("image", "image2"):
            arrays[f"{slot}_{hw}"] = np.stack([r[f"{slot}_{hw}"] for r in recs]).astype(np.uint8)
    for k, dt in (
        ("state", np.float32),
        ("joint_pos", np.float32),
        ("eef_pos", np.float64),
        ("eef_quat", np.float64),
        ("gripper_action", np.float64),
    ):
        arrays[k] = np.stack([np.asarray(r[k]) for r in recs]).astype(dt)
    arrays["sim_state"], arrays["sim_state_len"] = pad_stack([r["sim_state"] for r in recs])
    arrays["qacc_warmstart"], arrays["qacc_warmstart_len"] = pad_stack([r["qacc_warmstart"] for r in recs])
    rows = []
    for i, r in enumerate(recs):
        entry = lp.task_entry(sets, r["suite"], r["task_id"])
        rows.append(
            {
                "obs_idx": i,
                "obs_id": f"A/{r['suite']}/t{r['task_id']}/i{r['init_idx']}",
                "suite": r["suite"],
                "task_id": r["task_id"],
                "init_idx": r["init_idx"],
                "seed": r["seed"],
                "bddl": entry.get("bddl"),
                "original": entry["original"],
                "env_tag": r["env_tag"],
                "nq": r["nq"],
                "nv": r["nv"],
                "objects": r["objects"],
            }
        )
    return arrays, rows


# ---------------------------------------------------------------- set B (training data)


def _decode_png(b: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def build_set_b(
    args: argparse.Namespace, tasks: list[tuple[str, int]]
) -> tuple[dict[str, np.ndarray], list[dict]]:
    if args.backend == "mock":
        return build_set_b_mock(tasks)
    import pyarrow.parquet as pq

    root = Path(args.dataset_root)
    sets = lp.load_instruction_sets()
    by_instr = {lp.task_entry(sets, s, t)["original"]: (s, t) for s, t in tasks}
    eps = pq.read_table(
        sorted((root / "meta" / "episodes").glob("*/*.parquet")),
        columns=["episode_index", "tasks", "length"],
    ).to_pylist()
    chosen: dict[tuple[str, int], list[dict]] = {k: [] for k in tasks}
    for e in sorted(eps, key=lambda e: e["episode_index"]):
        key = by_instr.get(e["tasks"][0])
        if key is not None and len(chosen[key]) < args.episodes_b:
            chosen[key].append(e)
    short = [k for k, v in chosen.items() if len(v) < args.episodes_b]
    if short:
        raise SystemExit(
            f"set B: no {args.episodes_b} episodes for {short} (dataset task strings ≠ originals?)"
        )
    # The episodes' data/file_index in meta does not match the local copy's files: index the files instead.
    ep_file: dict[int, Path] = {}
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        for ep in set(pq.read_table(path, columns=["episode_index"]).column(0).to_pylist()):
            ep_file.setdefault(int(ep), path)
    want: dict[Path, list[tuple]] = {}  # file → [(episode, frame, key, slot, p, length)]
    for (suite, tid), lst in chosen.items():
        for ei, e in enumerate(lst):
            for pi, p in enumerate(lp.SET_B_PROGRESS):
                frame = round(p * (e["length"] - 1))
                f = ep_file[int(e["episode_index"])]
                want.setdefault(f, []).append(
                    (e["episode_index"], frame, (suite, tid), ei * 10 + pi, p, e["length"])
                )
    recs: dict[tuple, dict] = {}
    files_read = {}
    for path, items in sorted(want.items()):
        files_read[str(path.relative_to(root))] = lp.sha256_file(path)
        ep_ids = sorted({it[0] for it in items})
        tab = pq.read_table(
            path,
            columns=[
                "episode_index",
                "frame_index",
                "observation.images.image",
                "observation.images.image2",
                "observation.state",
                "action",
            ],
            filters=[("episode_index", "in", ep_ids)],
        ).to_pandas()
        for ep, frame, key, slot, p, length in items:
            d = tab[tab["episode_index"] == ep].sort_values("frame_index")
            row = d[d["frame_index"] == frame].iloc[0]
            acts = np.stack(d["action"].to_numpy())
            idx = np.clip(np.arange(frame, frame + GT_ACTIONS), 0, len(acts) - 1)
            recs[(key, slot)] = {
                "suite": key[0],
                "task_id": key[1],
                "slot": slot,
                "episode_index": int(ep),
                "frame_index": int(frame),
                "length": int(length),
                "progress": p,
                "image_256": _decode_png(row["observation.images.image"]["bytes"]),
                "image2_256": _decode_png(row["observation.images.image2"]["bytes"]),
                "state": np.asarray(row["observation.state"], dtype=np.float32),
                "gt_action": acts[idx].astype(np.float32),
            }
    ordered = [recs[k] for k in sorted(recs, key=lambda k: (lp.SUITES.index(k[0][0]), k[0][1], k[1]))]
    arrays = {k: np.stack([r[k] for r in ordered]) for k in ("image_256", "image2_256", "state", "gt_action")}
    rows = [_row_b(i, r, sets) for i, r in enumerate(ordered)]
    rows[0]["_files_read"] = files_read  # moved into the manifest by main()
    return arrays, rows


def _row_b(i: int, r: dict[str, Any], sets: dict[str, Any]) -> dict[str, Any]:
    entry = lp.task_entry(sets, r["suite"], r["task_id"])
    return {
        "obs_idx": i,
        "obs_id": f"B/{r['suite']}/t{r['task_id']}/ep{r['episode_index']}/f{r['frame_index']}",
        "suite": r["suite"],
        "task_id": r["task_id"],
        "slot": r["slot"],
        "episode_index": r["episode_index"],
        "frame_index": r["frame_index"],
        "length": r["length"],
        "progress": r["progress"],
        "original": entry["original"],
    }


def build_set_b_mock(tasks: list[tuple[str, int]]) -> tuple[dict[str, np.ndarray], list[dict]]:
    sets = lp.load_instruction_sets()
    rng = np.random.default_rng(0)
    recs = []
    for suite, tid in tasks:
        for ei in range(lp.SET_B_EPISODES):
            for pi, p in enumerate(lp.SET_B_PROGRESS):
                recs.append(
                    {
                        "suite": suite,
                        "task_id": tid,
                        "slot": ei * 10 + pi,
                        "episode_index": 1000 * lp.SUITES.index(suite) + 10 * tid + ei,
                        "frame_index": int(100 * p),
                        "length": 100,
                        "progress": p,
                        "image_256": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
                        "image2_256": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
                        "state": rng.normal(0, 0.1, 8).astype(np.float32),
                        "gt_action": rng.normal(0, 0.3, (GT_ACTIONS, 7)).astype(np.float32),
                    }
                )
    arrays = {k: np.stack([r[k] for r in recs]) for k in ("image_256", "image2_256", "state", "gt_action")}
    return arrays, [_row_b(i, r, sets) for i, r in enumerate(recs)]


# ---------------------------------------------------------------- main


def parse_tasks(spec: str | None, suites: str | None) -> list[tuple[str, int]]:
    ss = [s for s in (suites or ",".join(lp.SUITES)).split(",") if s]
    tids = [int(t) for t in spec.split(",")] if spec else list(range(lp.N_TASKS))
    return [(s, t) for s in ss for t in tids]


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """np.savez_compressed with fixed zip timestamps (numpy stamps the current time): same arrays → same sha."""
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays):
            info = zipfile.ZipInfo(f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(arrays[k]), allow_pickle=False)
    tmp.replace(path)


def write_set(out: Path, name: str, arrays: dict[str, np.ndarray], rows: list[dict], info: dict) -> dict:
    npz, meta = lp.set_files(out, name)
    files_read = rows[0].pop("_files_read", None) if rows else None
    save_npz(npz, arrays)
    stable = {k: v for k, v in info.items() if k not in VOLATILE}  # meta (and its sha) reproducible
    meta.write_text(json.dumps({"set": name, **stable, "rows": rows}, indent=1, default=float) + "\n")
    entry = {
        "npz": npz.name,
        "meta": meta.name,
        "npz_sha256": lp.sha256_file(npz),
        "meta_sha256": lp.sha256_file(meta),
        "n_obs": len(rows),
        "arrays": {k: [str(v.dtype), list(v.shape)] for k, v in arrays.items()},
        **info,
    }
    if files_read:
        entry["dataset_files_sha256"] = files_read
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out", required=True, help="output dir (set_A.npz, set_B.npz, *.meta.json, manifest.json)"
    )
    ap.add_argument("--sets", default="A,B")
    ap.add_argument("--suites", default=None)
    ap.add_argument("--tasks", default=None, help="task ids (default 0-9)")
    ap.add_argument("--inits", type=int, default=lp.SET_A_INITS, help="set A: init_idx 0 … n-1")
    ap.add_argument("--episodes-b", type=int, default=lp.SET_B_EPISODES)
    ap.add_argument("--dataset-root", default=f"{lp.DEFAULT_ROOT}/data/hf_libero")
    ap.add_argument("--dataset-revision", default="86958911c0f959db2bbbdb107eb3e17c5f9c798e")
    ap.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    ap.add_argument("--backend", choices=("libero", "mock"), default="libero")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks, args.suites)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"sets": {}}
    common = {
        "git_sha": git_sha(),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "backend": args.backend,
        "tasks": [f"{s}/{t}" for s, t in tasks],
    }
    for name in args.sets.split(","):
        if name == "A":
            arrays, rows = build_set_a(args, tasks)
            info = {
                **common,
                "composition": f"{len(tasks)} tasks × init_idx 0-{args.inits - 1}",
                "seed": "common.episode_seed(7, 0, task, init) (sc-bench --seed 7, repeat 0)",
                "renderer": os.environ.get("MUJOCO_GL"),
                "env_tag": rows[0]["env_tag"] if rows else None,
                "image_sizes": list(SIZES_A),
                "env": "lerobot LiberoEnv OSC_POSE 20 Hz, 10 settle steps (= ours / EpisodeStepper.reset)",
            }
        elif name == "B":
            arrays, rows = build_set_b(args, tasks)
            info = {
                **common,
                "composition": f"{len(tasks)} tasks × {args.episodes_b} episodes (lowest episode_index) × "
                f"progress {list(lp.SET_B_PROGRESS)}",
                "dataset": "HuggingFaceVLA/libero",
                "dataset_revision": args.dataset_revision if args.backend != "mock" else "mock",
                "dataset_root": args.dataset_root if args.backend != "mock" else "mock",
                "image_sizes": [256],
            }
        else:
            raise SystemExit(f"unknown set {name!r}")
        manifest["sets"][name] = write_set(out, name, arrays, rows, info)
        e = manifest["sets"][name]
        print(
            f"[make_obs] set {name}: {e['n_obs']} obs → {out / e['npz']} sha256 {e['npz_sha256']}", flush=True
        )
    man_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if os.environ["MUJOCO_GL"] == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    sys.exit(main())
