"""sc-bench --stagger-start (T017 review): policy servers start one after another (the next only once the previous
one's ready file exists) for lerobot and external (starvla, droid) servers; without it all start at once (as
before). Mock backend, two fake GPUs, CPU only.

    uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_stagger_start.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("websockets")
pytest.importorskip("lerobot")

SHORTCUT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "parallel")]

from bench.core import runner  # noqa: E402

FIXTURE = SHORTCUT / "tests" / "fixtures" / "libero_goal_2tasks.json"


@pytest.fixture
def events(monkeypatch):
    """Record server starts and the moment each ready file was seen, in order."""
    log: list[tuple[str, str, bool]] = []
    init, wait = runner.ServerProc.__init__, runner.ServerProc.wait_ready

    def rec_init(self, gpu, args, ctx):
        # ready files of the servers started so far: with stagger, all of them must exist already
        prev_ready = all(p.exists() for p in started)
        log.append(("start", gpu, prev_ready))
        init(self, gpu, args, ctx)
        started.append(self.ready)

    def rec_wait(self):
        out = wait(self)
        log.append(("ready", self.gpu, self.ready.exists()))
        return out

    started: list[Path] = []
    monkeypatch.setattr(runner.ServerProc, "__init__", rec_init)
    monkeypatch.setattr(runner.ServerProc, "wait_ready", rec_wait)
    return log


def run(tmp_path: Path, name: str, server: str | None, stagger: bool) -> None:
    bat = tmp_path / f"{name}.yaml"
    bat.write_text(
        f"name: {name}\njobs:\n  - {{benchmark: ours, suite: libero_goal, conditions: [C0], tasks: '1', "
        f"episodes: 1, video: none, instr_set: '{FIXTURE}'}}\n"
    )
    argv = ["--ckpt", f"mock/{name}", "--backend", "mock", "--battery", str(bat), "--gpus", "0,1"]
    argv += ["--out-root", str(tmp_path / "out")]
    if server:
        argv += ["--server", server]
    if stagger:
        argv.append("--stagger-start")
    assert runner.main(argv) == 0


@pytest.mark.parametrize("server", [None, "starvla", "droid"])
def test_stagger_start_waits_for_each_ready_file(tmp_path, events, server):
    run(tmp_path, f"st-{server or 'lerobot'}", server, stagger=True)
    assert [(k, g) for k, g, _ in events] == [("start", "0"), ("ready", "0"), ("start", "1"), ("ready", "1")]
    assert all(ok for _k, _g, ok in events)  # each start saw every earlier ready file


def test_default_starts_all_servers_at_once(tmp_path, events):
    run(tmp_path, "all", None, stagger=False)
    # both servers launched before the first ready file is waited for
    assert [(k, g) for k, g, _ in events] == [("start", "0"), ("start", "1"), ("ready", "0"), ("ready", "1")]


def test_stagger_start_stops_at_a_failing_server(tmp_path, monkeypatch):
    """A server that dies while loading: no further server is started, the run fails."""
    starts: list[str] = []
    init = runner.ServerProc.__init__

    def rec_init(self, gpu, args, ctx):
        starts.append(gpu)
        init(self, gpu, args, ctx)

    def dead(self):
        raise SystemExit(f"policy server on GPU {self.gpu} exited (1)")

    monkeypatch.setattr(runner.ServerProc, "__init__", rec_init)
    monkeypatch.setattr(runner.ServerProc, "wait_ready", dead)
    with pytest.raises(SystemExit, match="GPU 0 exited"):
        run(tmp_path, "dead", None, stagger=True)
    assert starts == ["0"]
