"""Generate LIBERO-PRO's environment axis (PRO venv): bddl_files/<suite>_env + init_files/<suite>_env.

The PRO HF dataset ships object / swap / lan / task files but not the environment axis; PRO's
`perturbation.create_env` builds it at eval time. Same BDDL rewrite as PRO's EnvironmentReplacePerturbator
(text replace of the table name, problem class token, fixture type), with the target table from
catalog.ENV_TARGET (PRO hard-codes living_room_table, a no-op for scenes already on it; its README recommends
main_table because other tables can let objects drift). Init states: like PRO's
notebooks/generate_init_states.py (50 fresh placements per task, saved as torch-compatible .pruned_init)
but without a renderer (no GPU), seeded per task.

    .venvs/libero_pro/bin/python scripts/shortcut/bench/libero_pro/gen_env_axis.py [--suites ...] [--jobs 16]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import pickle  # nosec B403
import subprocess  # nosec B404
import sys
import zipfile
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from common import parse_range  # noqa: E402

from bench.libero_pro import catalog  # noqa: E402
from bench.libero_pro.sim import ensure_libero_config  # noqa: E402

# bump when the generation procedure changes (init sampling, BDDL rewrite, file format): cached files are refused
SCRIPT_VERSION = "gen_env_axis-v1"


def pro_perturbation(src: Path):  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("libero_pro_perturbation", src / "perturbation.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rewrite(src: Path, suite: str, task: str, content: str) -> tuple[str, str, str]:
    pert = pro_perturbation(src)
    p = pert.EnvironmentReplacePerturbator(
        pert.BDDLParser(content), str(src / "libero_ood" / "ood_environment.yaml")
    )
    cur = p._extract_current_env(suite, task)
    if cur not in p.ALLOWED_ENVS:
        raise ValueError(f"{suite}/{task}: table {cur!r} not in PRO's ood_environment.yaml")
    new = catalog.ENV_TARGET.get(cur, catalog.ENV_TARGET_DEFAULT)
    out = content.replace(cur, new)
    out = p._rewrite_problem_env_token(out, new)
    out = p._rewrite_fixtures_type(out, new, p.ENV_FIXTYPE[new])
    return out, cur, new


def save_init(path: Path, states: np.ndarray) -> None:
    """Same container as PRO's generate_init_states.py (torch.load / sim.load_init_states read it)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in (("archive/data.pkl", pickle.dumps(states)), ("archive/version", b"1")):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))  # fixed: same states → same bytes
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)


def pro_commit(src: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()  # nosec
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def out_paths(root: Path, suite: str, task: str) -> tuple[Path, Path, Path]:
    """(bddl, init, sidecar meta) of one generated task."""
    init = root / "init_files" / f"{suite}_env" / f"{task}.pruned_init"
    return root / "bddl_files" / f"{suite}_env" / f"{task}.bddl", init, init.with_suffix(".meta.json")


def wanted_meta(seed: int, n: int, commit: str) -> dict[str, Any]:
    """What a generated file depends on; stored next to it and compared before reusing it."""
    return {"seed": seed, "num_inits": n, "pro_commit": commit, "script_version": SCRIPT_VERSION}


def cache_state(root: Path, suite: str, task: str, want: dict[str, Any]) -> str:
    """'missing' (no init file), 'ok' (sidecar = want), 'mismatch' (no sidecar or another configuration)."""
    _, init, meta = out_paths(root, suite, task)
    if not init.exists():
        return "missing"
    if not meta.exists():
        return "mismatch"
    have = json.loads(meta.read_text())
    return "ok" if {k: have.get(k) for k in want} == want else "mismatch"


def existing(root: Path) -> list[tuple[str, str]]:
    """(suite, task) of every environment-axis init file already on disk, any suite."""
    return sorted(
        (p.parent.name.removesuffix("_env"), p.name.removesuffix(".pruned_init"))
        for p in (root / "init_files").glob("libero_*_env/*.pruned_init")
    )


def preflight(root: Path, want: dict[str, Any], force: bool) -> None:
    """The whole environment axis must come from one generation setting: refuse to (re)generate anything while any
    existing env-axis init file — requested or not, any suite — has another seed / PRO commit / script version or no
    sidecar. --force regenerates the whole axis instead (main)."""
    if force:
        return
    bad = [f"{s}/{t}" for s, t in existing(root) if cache_state(root, s, t, want) == "mismatch"]
    if bad:
        raise SystemExit(
            f"gen_env_axis: {len(bad)} existing environment-axis init files were generated with another configuration "
            f"than {want} (or have no .meta.json sidecar), e.g. {bad[:3]}; rerun with --force to regenerate the whole axis"
        )


def gen_task(args: tuple[str, str, str, int, int, bool, str]) -> str:
    root_s, suite, task, n, seed, force, commit = args
    root = Path(root_s)
    src = root.parents[1]
    out_bddl, out_init, out_meta = out_paths(root, suite, task)
    out_bddl.parent.mkdir(parents=True, exist_ok=True)
    out_init.parent.mkdir(parents=True, exist_ok=True)
    content, cur, new = rewrite(src, suite, task, (root / "bddl_files" / suite / f"{task}.bddl").read_text())
    want = wanted_meta(seed, n, commit)
    fresh = out_bddl.exists() and out_bddl.read_text() == content
    if not force and fresh and cache_state(root, suite, task, want) == "ok":
        return f"{suite}/{task}: {cur}->{new} (exists)"
    out_meta.unlink(missing_ok=True)  # no sidecar = not trusted, if we die mid-way
    out_bddl.write_text(content)
    ensure_libero_config()
    from libero.libero.envs.env_wrapper import ControlEnv

    np.random.seed((seed + zlib.crc32(f"{suite}/{task}".encode())) % 2**32)
    env = ControlEnv(bddl_file_name=str(out_bddl), has_offscreen_renderer=False, use_camera_obs=False)
    states = []
    for _ in range(n):
        env.reset()
        states.append(env.get_sim_state())
    env.close()
    save_init(out_init, np.asarray(states))
    bddl_sha = hashlib.sha256(content.encode()).hexdigest()
    out_meta.write_text(json.dumps({**want, "table": f"{cur}->{new}", "bddl_sha256": bddl_sha}, indent=2))
    return f"{suite}/{task}: {cur}->{new}, {n} inits"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", default=",".join(catalog.SUITES))
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--num-inits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    root = catalog.pro_root()
    tids = parse_range(args.tasks)
    commit = pro_commit(root.parents[1])
    items = [(s, catalog.suite_task_names(root, s)[t]) for s in args.suites.split(",") for t in tids]
    preflight(root, wanted_meta(args.seed, args.num_inits, commit), args.force)
    if args.force:  # one setting for the whole axis: every suite × task, not only the requested ones
        items = [(s, t) for s in catalog.SUITES for t in catalog.suite_task_names(root, s)]
        print(
            f"[gen_env_axis] --force: regenerating the whole environment axis ({len(items)} tasks)",
            flush=True,
        )
    work = [(str(root), s, t, args.num_inits, args.seed, args.force, commit) for s, t in items]
    with Pool(args.jobs) as pool:
        for line in pool.imap_unordered(gen_task, work):
            print("[gen_env_axis]", line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
