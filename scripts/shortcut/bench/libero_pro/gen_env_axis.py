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
import importlib.util
import io
import pickle  # nosec B403
import sys
import zipfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

from common import parse_range  # noqa: E402

from bench.libero_pro import catalog  # noqa: E402
from bench.libero_pro.sim import ensure_libero_config  # noqa: E402


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
        z.writestr("archive/data.pkl", pickle.dumps(states))
        z.writestr("archive/version", b"1")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)


def gen_task(args: tuple[str, str, str, int, int, bool]) -> str:
    root_s, suite, task, n, seed, force = args
    root = Path(root_s)
    src = root.parents[1]
    bddl_dir, init_dir = root / "bddl_files" / f"{suite}_env", root / "init_files" / f"{suite}_env"
    bddl_dir.mkdir(parents=True, exist_ok=True)
    init_dir.mkdir(parents=True, exist_ok=True)
    out_bddl, out_init = bddl_dir / f"{task}.bddl", init_dir / f"{task}.pruned_init"
    content, cur, new = rewrite(src, suite, task, (root / "bddl_files" / suite / f"{task}.bddl").read_text())
    if out_init.exists() and out_bddl.exists() and out_bddl.read_text() == content and not force:
        return f"{suite}/{task}: {cur}->{new} (exists)"
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
    work = [
        (str(root), s, catalog.suite_task_names(root, s)[t], args.num_inits, args.seed, args.force)
        for s in args.suites.split(",")
        for t in tids
    ]
    with Pool(args.jobs) as pool:
        for line in pool.imap_unordered(gen_task, work):
            print("[gen_env_axis]", line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
