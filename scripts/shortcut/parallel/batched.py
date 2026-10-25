"""Continuous batching inside one worker: N env subprocesses, one policy call per chunk for all of them.

Every episode starts on a chunk boundary, so all running episodes need a new chunk at the same tick and
each policy call covers every busy slot. A slot whose episode ends mid-chunk (C0–C3 stop at the goal) idles
until the boundary (< n_action_steps steps), then takes the next unit; C4–C6 episodes run to max_steps in
the same batch. With pad=True every call is padded to N rows (copies of row 0), so kernel shapes, and with
them each row's numbers, do not depend on how many slots are busy.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from envproc import EnvClient, EnvError


@dataclass
class Slot:
    idx: int
    client: EnvClient
    unit: dict[str, Any] | None = None
    actions: deque = field(default_factory=deque)
    gen: torch.Generator | None = None
    obs: Any = None
    t0: float = 0.0
    t_policy: float = 0.0
    loc: tuple | None = None  # (suite, task) of the env this slot holds


@dataclass
class Stats:
    infer_calls: int = 0
    infer_s: float = 0.0
    rows: int = 0
    padded_rows: int = 0
    env_steps: int = 0
    step_s: float = 0.0
    episodes: int = 0
    failed: int = 0

    def line(self) -> str:
        per_call = 1000 * self.infer_s / max(1, self.infer_calls)
        per_step = 1000 * self.step_s / max(1, self.env_steps)
        return (
            f"episodes={self.episodes} failed={self.failed} env_steps={self.env_steps} "
            f"infer_calls={self.infer_calls} ({per_call:.0f} ms/call, rows {self.rows} + pad {self.padded_rows}) "
            f"step phase {per_step:.2f} ms/env-step"
        )


class BatchedRunner:
    def __init__(
        self,
        bundle: Any,
        slots: list[Slot],
        claim: Callable[[tuple | None], dict[str, Any] | None],
        on_done: Callable[[dict[str, Any], dict[str, Any], float], None],
        on_fail: Callable[[dict[str, Any], str], None],
        restart: Callable[[int], EnvClient],
        pad: bool = True,
        stats: Stats | None = None,
    ):
        self.bundle = bundle
        self.slots = slots
        self.claim, self.on_done, self.on_fail, self.restart = claim, on_done, on_fail, restart
        self.n = bundle.n_action_steps
        self.pad = pad
        self.stats = stats or Stats()

    def run(self) -> None:
        """Until claim() returns None and every slot is idle."""
        while True:
            self._fill()
            active = [s for s in self.slots if s.unit is not None]
            if not active:
                return
            self._infer(active)
            for _ in range(self.n):
                live = [s for s in active if s.unit is not None and s.actions]
                if not live:
                    break
                tic = time.perf_counter()
                sent = []
                for s in live:
                    a = s.actions.popleft()
                    try:
                        s.client.send("step", (a, not s.actions))
                        sent.append(s)
                    except EnvError as e:
                        self._fail(s, str(e))
                for s in sent:
                    try:
                        done, obs = s.client.recv()
                    except EnvError as e:
                        self._fail(s, str(e))
                        continue
                    if obs is not None:
                        s.obs = obs
                    if done:
                        self._finish(s)
                self.stats.env_steps += len(sent)
                self.stats.step_s += time.perf_counter() - tic

    def _payload(self, unit: dict[str, Any]) -> dict[str, Any]:
        meta = self.bundle.inf_meta
        return {
            **{k: unit[k] for k in ("suite", "task_id", "variant", "init_idx", "ep_seed", "max_steps")},
            "record_video": unit["record_video"],
            "tag": unit["tag"],
            "env_hw": meta["env_observation_hw"],
            "control_mode": meta["control_mode"],
        }

    def _fill(self) -> None:
        exhausted = False
        while not exhausted:
            started = []
            for s in self.slots:
                if s.unit is not None:
                    continue
                unit = self.claim(s.loc)
                if unit is None:
                    exhausted = True
                    break
                s.unit = unit
                try:
                    s.client.send("start", self._payload(unit))
                    started.append(s)
                except EnvError as e:
                    self._fail(s, str(e))
            if not started and not any(s.unit is None for s in self.slots):
                return
            for s in started:
                assert s.unit is not None
                try:
                    obs, done = s.client.recv()
                except EnvError as e:
                    self._fail(s, str(e))
                    continue
                s.obs, s.t0, s.t_policy = obs, time.time(), 0.0
                s.actions = deque()
                s.loc = (s.unit["suite"], s.unit["task_id"])
                s.gen = self.bundle.generator(int(s.unit["ep_seed"]))
                if done:
                    self._finish(s)
            if all(s.unit is not None for s in self.slots):
                return

    def _infer(self, rows: list[Slot]) -> None:
        obs = [s.obs for s in rows]
        instr = [s.unit["variant"]["instruction"] for s in rows if s.unit is not None]
        noise = torch.cat([self.bundle.draw_noise(s.gen) for s in rows])
        k = len(self.slots) - len(rows) if self.pad else 0
        if k > 0:
            obs += [obs[0]] * k
            instr += [instr[0]] * k
            noise = torch.cat([noise, noise[:1].expand(k, *noise.shape[1:])])
        tic = time.perf_counter()
        chunk = self.bundle.infer(obs, instr, noise)
        dt = time.perf_counter() - tic
        for i, s in enumerate(rows):
            s.actions = deque(chunk[i])
            s.t_policy += dt / len(rows)
        self.stats.infer_calls += 1
        self.stats.infer_s += dt
        self.stats.rows += len(rows)
        self.stats.padded_rows += k

    def _finish(self, s: Slot) -> None:
        unit = s.unit
        assert unit is not None
        try:
            s.client.send("result", s.t_policy)
            res = s.client.recv()
        except EnvError as e:
            self._fail(s, str(e))
            return
        s.unit = None
        s.actions = deque()
        self.stats.episodes += 1
        self.on_done(unit, res, time.time() - s.t0)

    def _fail(self, s: Slot, err: str) -> None:
        if s.unit is not None:
            self.on_fail(s.unit, err)
            self.stats.failed += 1
        s.unit = None
        s.actions = deque()
        if not s.client.alive():
            s.client = self.restart(s.idx)
            s.loc = None
