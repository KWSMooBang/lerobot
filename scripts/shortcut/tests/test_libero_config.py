"""No LIBERO config prompt on a fresh server (T016 review): with an empty HOME (no ~/.libero/config.yaml), no
LIBERO_CONFIG_PATH and stdin at EOF, sc-bench --dry-run and the ours plan (C4/C5 variants resolve their BDDLs)
finish without libero's input() prompt, in both env modes (SC_LIBERO_RAW=0/1).

    uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_libero_config.py -q
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SHORTCUT = HERE.parent
RUNNER = SHORTCUT / "bench" / "core" / "runner.py"
FIXTURE = HERE / "fixtures" / "libero_goal_2tasks.json"
HAVE_LIBERO = importlib.util.find_spec("libero") is not None


def fresh_env(home: Path, raw: bool) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("LIBERO_CONFIG_PATH", "SC_LIBERO_RAW")}
    env.update(HOME=str(home), SC_LIBERO_RAW="1" if raw else "0", CUDA_VISIBLE_DEVICES="")
    # keep the shared caches (HF assets / uv): only LIBERO's own config must be missing
    env.setdefault("HF_HOME", str(Path(os.environ.get("HOME", "~")).expanduser() / ".cache" / "huggingface"))
    # the PaliGemma tokenizer (C4/C5 planning) also lives under ~/.cache: hand the real one over explicitly,
    # otherwise load_tokenizer falls back to the gated hub repo (401)
    tok = local_tokenizer_dir()
    if tok is not None:
        env["SC_TOKENIZER_DIR"] = str(tok)
    return env


def local_tokenizer_dir() -> Path | None:
    """A local PaliGemma tokenizer as common.load_tokenizer looks for it, resolved with the real HOME."""
    cands = [os.environ.get("SC_TOKENIZER_DIR"), "/cache/pid_demo_dl/paligemma_tok_hf"]
    cands.append(str(Path.home() / ".cache" / "shortcut_libero" / "paligemma_tok_hf"))
    for c in cands:
        if c and (Path(c) / "tokenizer.model").exists():
            return Path(c)
    return None


def run(argv: list[str], env: dict[str, str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *argv],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def battery(tmp: Path) -> Path:
    p = tmp / "b.yaml"
    p.write_text(
        "name: cfgtest\njobs:\n  - {benchmark: ours, suite: libero_goal, conditions: [C0, C4, C5], tasks: '1,8', "
        f"episodes: 2, instr_set: '{FIXTURE}'}}\n"
    )
    return p


def assert_no_prompt(r: subprocess.CompletedProcess) -> None:
    out = r.stdout + r.stderr
    assert "EOFError" not in out and "Do you want to specify a custom path" not in out, out[-3000:]
    assert r.returncode == 0, out[-3000:]


@pytest.mark.skipif(not HAVE_LIBERO, reason="needs the libero extra (the prompt lives in libero)")
@pytest.mark.parametrize("raw", [False, True])
def test_mock_dry_run_and_plan_without_libero_config(tmp_path, raw):
    """Mock backend: planning resolves C4/C5 goal BDDLs but must not import libero at all."""
    home = tmp_path / "home"
    home.mkdir()
    env = fresh_env(home, raw)
    bat = battery(tmp_path)
    r = run([str(RUNNER), "--ckpt", "mock/pi05lib", "--backend", "mock", "--battery", str(bat),
             "--out-root", str(tmp_path / "out"), "--dry-run"], env)  # fmt: skip
    assert_no_prompt(r)
    assert "ours-goal-C4" in r.stdout
    code = (  # argv: shortcut dir, parallel dir, instruction set, out dir
        "import sys; sys.path[:0] = sys.argv[1:3]\n"
        "import run_eval\n"
        "from common import load_instruction_set\n"
        "from types import SimpleNamespace\n"
        "from mock_backend import MockTokenizer\n"
        "args = run_eval.build_parser().parse_args(['--policy', 'pi05', '--suite', 'libero_goal', '--condition',"
        " 'C4', '--instr-set', sys.argv[3], '--tasks', '1,8', '--backend', 'mock', '--out', sys.argv[4]])\n"
        "instr = load_instruction_set(args.instr_set)\n"
        "_s, tasks, _k = run_eval.suite_tasks(args)\n"
        "instr['_suite_tasks'] = tasks\n"
        "planned = run_eval.plan_variants(SimpleNamespace(args=args, instr_set=instr), MockTokenizer())\n"
        "assert planned and all(v.goal_bddl for _, vs in planned for v in vs), planned\n"
        "assert 'libero.libero' not in sys.modules, 'planning imported libero'\n"
        "print('PLANNED', sum(len(vs) for _, vs in planned))\n"
    )
    argv = [str(SHORTCUT), str(SHORTCUT / "parallel"), str(FIXTURE), str(tmp_path / "plan")]
    r = run(["-c", code, *argv], env)
    assert_no_prompt(r)
    assert "PLANNED" in r.stdout


@pytest.mark.skipif(not HAVE_LIBERO, reason="needs the libero extra")
@pytest.mark.parametrize("raw", [False, True])
def test_libero_dry_run_builds_own_config(tmp_path, raw):
    """libero backend (pi05lib): the dry-run reads suite sizes / max steps from LIBERO, which must go through
    ensure_libero_config (a config for this interpreter's libero under the empty HOME), never the prompt."""
    try:  # the config is built from the cached lerobot/libero-assets snapshot; skip when it is not cached
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset", local_files_only=True)
    except Exception:
        pytest.skip("lerobot/libero-assets not in the HF cache")
    if local_tokenizer_dir() is None:
        pytest.skip(
            "no local PaliGemma tokenizer (SC_TOKENIZER_DIR / ~/.cache/shortcut_libero/paligemma_tok_hf)"
        )
    home = tmp_path / "home"
    home.mkdir()
    env = fresh_env(home, raw)
    bat = battery(tmp_path)
    r = run([str(RUNNER), "--ckpt", "pi05lib", "--battery", str(bat), "--out-root", str(tmp_path / "out"),
             "--registry", str(tmp_path / "reg.tsv"), "--dry-run"], env, timeout=1200)  # fmt: skip
    assert_no_prompt(r)
    tag = "raw" if raw else "fix-"
    cfgs = list((home / ".cache" / "shortcut_libero").glob("*-*-*"))
    assert any(c.name.endswith("raw") == raw and tag in c.name for c in cfgs), cfgs
    assert not (home / ".libero").exists()  # libero's own default config was never written


def test_resolve_path_uses_this_interpreters_bddl_dir(tmp_path, monkeypatch):
    """resolve_path finds suite BDDLs in this interpreter's libero install without importing libero, and an
    inherited LIBERO_CONFIG_PATH of another venv does not redirect it."""
    sys.path[:0] = [str(SHORTCUT)]
    import common

    if not HAVE_LIBERO:
        pytest.skip("needs the libero extra")
    bddl = common.libero_bddl_dir()
    assert bddl is not None and bddl.is_dir()
    rel = next(p for p in bddl.glob("libero_goal/*.bddl")).relative_to(bddl)
    other = tmp_path / "other_cfg"
    other.mkdir()
    (other / "config.yaml").write_text(f"bddl_files: {tmp_path / 'elsewhere'}\n")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(other))
    got = common.resolve_path(str(rel), FIXTURE)
    assert Path(got) == (bddl / rel).resolve()
    with pytest.raises(FileNotFoundError):
        common.resolve_path("libero_goal/does_not_exist.bddl", FIXTURE)
