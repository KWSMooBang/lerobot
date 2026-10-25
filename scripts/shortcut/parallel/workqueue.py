"""File-lock work queue shared by all workers of one multi-GPU session.

    <queue dir>/units.jsonl   one work unit per line (static, written by the coordinator)
    <queue dir>/log           append-only: "<op> <uid> <worker>" with op in claim|done|release|fail
    <queue dir>/lock          flock target; every read-modify-append of the log holds it

Each process replays the log incrementally (it remembers its byte offset), so a claim costs O(new lines).
Work stealing: a worker asks for a unit of its current policy group, preferring its env slot's current
(suite, task) so the env is reused, then any unit of the group; when its group has nothing pending it
loads the next group with pending units. The coordinator releases the claims of a worker that died.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

OPS = ("claim", "done", "release", "fail")


def write_queue(qdir: Path, units: list[dict[str, Any]]) -> None:
    qdir.mkdir(parents=True, exist_ok=True)
    tmp = qdir / "units.jsonl.tmp"
    with open(tmp, "w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")
    tmp.replace(qdir / "units.jsonl")
    (qdir / "log").write_bytes(b"")
    (qdir / "lock").touch()


class WorkQueue:
    def __init__(self, qdir: str | Path, wid: str):
        self.dir = Path(qdir)
        self.wid = wid
        self.units: dict[str, dict[str, Any]] = {}
        self.by_group: dict[str, list[str]] = defaultdict(list)
        self.by_loc: dict[tuple, list[str]] = defaultdict(list)
        for line in (self.dir / "units.jsonl").read_text().splitlines():
            u = json.loads(line)
            self.units[u["uid"]] = u
            self.by_group[u["group"]].append(u["uid"])
            self.by_loc[(u["group"], u["suite"], u["task_id"])].append(u["uid"])
        self.groups = list(self.by_group)  # first-seen order
        self.state: dict[str, tuple[str, str]] = {}  # uid -> (op, worker); absent = pending
        self.attempts: dict[str, int] = defaultdict(int)
        self._cursor: dict[Any, int] = defaultdict(int)
        self._off = 0
        self._tlock = threading.Lock()

    # ------------------------------------------------------------ log

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._tlock, open(self.dir / "lock", "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                self._sync()
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _sync(self) -> None:
        with open(self.dir / "log", "rb") as f:
            f.seek(self._off)
            data = f.read()
        end = data.rfind(b"\n") + 1  # complete lines only
        for line in data[:end].decode().splitlines():
            op, uid, wid = line.split(" ", 2)
            self._apply(op, uid, wid)
        self._off += end

    def _apply(self, op: str, uid: str, wid: str) -> None:
        if op == "claim":
            self.state[uid] = ("claim", wid)
            self.attempts[uid] += 1
        elif op == "release":
            self.state.pop(uid, None)
            self._cursor.clear()  # released units sit behind the cursors
        elif op in ("done", "fail"):
            self.state[uid] = (op, wid)
        else:
            raise ValueError(f"bad queue op {op!r}")

    def _append(self, op: str, uid: str) -> None:
        assert op in OPS and " " not in uid, (op, uid)
        with open(self.dir / "log", "ab") as f:
            f.write(f"{op} {uid} {self.wid}\n".encode())
            f.flush()
            os.fsync(f.fileno())
        self._sync()

    def _first_pending(self, key: Any, uids: list[str]) -> str | None:
        i = self._cursor[key]
        while i < len(uids) and uids[i] in self.state:
            i += 1
        self._cursor[key] = i
        return uids[i] if i < len(uids) else None

    # ------------------------------------------------------------ API

    def claim(self, group: str, prefer: tuple | None = None) -> dict[str, Any] | None:
        """Next pending unit of `group`, preferring (suite, task) == prefer. None when the group is empty."""
        with self._locked():
            uid = None
            if prefer is not None:
                loc = (group, *prefer)
                uid = self._first_pending(loc, self.by_loc.get(loc, []))
            if uid is None:
                uid = self._first_pending(group, self.by_group.get(group, []))
            if uid is None:
                return None
            self._append("claim", uid)
            return self.units[uid]

    def mark(self, uid: str, op: str) -> None:
        with self._locked():
            self._append(op, uid)

    def pending_groups(self) -> list[str]:
        with self._locked():
            return [g for g in self.groups if self._first_pending(g, self.by_group[g]) is not None]

    def release_worker(self, wid: str, max_attempts: int) -> tuple[list[str], list[str]]:
        """Coordinator: put the unfinished claims of a dead worker back (or fail them after
        max_attempts claims). Returns (released, failed)."""
        released, failed = [], []
        with self._locked():
            for uid, (op, who) in list(self.state.items()):
                if op == "claim" and who == wid:
                    if self.attempts[uid] >= max_attempts:
                        self._append("fail", uid)
                        failed.append(uid)
                    else:
                        self._append("release", uid)
                        released.append(uid)
        return released, failed

    def snapshot(self) -> dict[str, list[str]]:
        """uids by status: pending / claim / done / fail."""
        with self._locked():
            out: dict[str, list[str]] = {"pending": [], "claim": [], "done": [], "fail": []}
            for uid in self.units:
                out[self.state[uid][0] if uid in self.state else "pending"].append(uid)
            return out
