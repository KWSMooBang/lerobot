"""LIBERO-PRO env side (PRO venv: numpy, libero (PRO fork), robosuite, cv2, imageio — no torch, no lerobot).

`ProEpisode` mirrors lerobot `LiberoEnv` + T001 `EpisodeStepper` on a raw `OffScreenRenderEnv`:
reset = env.seed(seed) → reset → set_init_state(init) → 10 no-op steps → delta control; success = the PRO
BDDL's own goal (`env.check_success()`), early stop on it. Task axis (new goal): both the PRO goal and the
original suite BDDL's goal are evaluated every step, no early stop (T001 dual-goal rule), so
success_orig / success_new are measured the same way as ours C4–C6.

backend "mock" (tests): `MockProEnv`, numpy only, the policy server's mock policy heads to text_pos(instruction).
"""

from __future__ import annotations

import hashlib
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

SHORTCUT = Path(__file__).resolve().parents[2]
if str(SHORTCUT) not in sys.path:
    sys.path.insert(0, str(SHORTCUT))

import common  # noqa: E402,F401  (stdlib only: MUJOCO_GL=egl + NVIDIA EGL vendor before MuJoCo loads)

from bench.libero_pro import catalog  # noqa: E402

CONTROL_HZ = 20
VIDEO_FPS = 10
FRAME_SIDE = 256  # agentview | wrist → 512×256 like ours
NUM_STEPS_WAIT = 10  # lerobot LiberoEnv settle steps
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def ensure_libero_config() -> None:
    """LIBERO_CONFIG_PATH → <venv>/libero_config (setup_venv.sh writes it; never ~/.libero)."""
    if os.environ.get("LIBERO_CONFIG_PATH"):
        return
    cfg = Path(sys.prefix) / "libero_config"
    if not (cfg / "config.yaml").exists():
        raise SystemExit(f"{cfg}/config.yaml missing: run scripts/shortcut/bench/libero_pro/setup_venv.sh")
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)


load_init_states = catalog.load_init_states


# ---------------------------------------------------------------- frames


def upright(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img)[::-1, ::-1])


def header_lines(
    instruction: str, label: str, t: int, t_orig: int | None, t_new: int | None, dual: bool
) -> list[tuple[str, tuple[int, int, int]]]:
    white, green, red, grey = (255, 255, 255), (80, 255, 80), (255, 90, 90), (200, 200, 200)
    wrapped = textwrap.wrap(instruction or '""', 62)[:2] or ['""']
    lines = [(f"{label}  t={t}", white)]
    lines += [(f"> {w}" if i == 0 else f"  {w}", grey) for i, w in enumerate(wrapped)]

    def ok(tt: int | None) -> tuple[str, tuple[int, int, int]]:
        return (f"yes@{tt}", green) if tt is not None else ("no", red)

    if dual:
        (s, c), (s2, c2) = ok(t_orig), ok(t_new)
        lines += [(f"orig goal: {s}", c), (f"PRO goal:  {s2}", c2)]
    else:
        s, c = ok(t_new)
        lines.append((f"success: {s}", c))
    return lines


def compose_frame(agent: np.ndarray, wrist: np.ndarray, header: list) -> np.ndarray:
    import cv2

    views = [
        cv2.resize(upright(x), (FRAME_SIDE, FRAME_SIDE), interpolation=cv2.INTER_AREA) for x in (agent, wrist)
    ]
    frame = np.concatenate(views, axis=1)
    h = 14
    box = frame[: 6 + h * len(header)]
    box[:] = (box * 0.45).astype(np.uint8)
    for i, (txt, color) in enumerate(header):
        cv2.putText(frame, txt, (5, 13 + i * h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}{path.suffix}")
    imageio.mimwrite(tmp, frames, fps=VIDEO_FPS, codec="libx264", quality=7, macro_block_size=8)
    os.replace(tmp, path)


# ---------------------------------------------------------------- real env


class ContactProbe:
    """Objects the gripper touches; 'grasped' = touched by both fingers (run_eval.ContactProbe)."""

    def __init__(self, e: Any):
        m = e.sim.model
        roots = {m.body_name2id(o.root_body): n for n, o in {**e.objects_dict, **e.fixtures_dict}.items()}
        self.geom_obj: dict[int, str] = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            while b > 0 and b not in roots:
                b = int(m.body_parentid[b])
            if b in roots:
                self.geom_obj[g] = roots[b]
        grip = e.robots[0].gripper
        ids = lambda names: {m.geom_name2id(n) for n in names}  # noqa: E731
        self.grip_geoms = ids(grip.contact_geoms)
        self.left, self.right = (
            ids(grip.important_geoms["left_finger"]),
            ids(grip.important_geoms["right_finger"]),
        )
        self.sim = e.sim

    def __call__(self) -> tuple[list[str], list[str]]:
        d = self.sim.data
        touched, left, right = set(), set(), set()
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.grip_geoms and b in self.geom_obj:
                    o = self.geom_obj[b]
                    touched.add(o)
                    if a in self.left:
                        left.add(o)
                    if a in self.right:
                        right.add(o)
        return sorted(touched), sorted(left & right)


class ProEnv:
    """One PRO task (bddl + init file) as an env: reset(init_idx, seed) → raw obs; step(a) → (obs, pro, orig)."""

    def __init__(self, entry: dict[str, Any], hw: int = 360, control_freq: int = CONTROL_HZ):
        ensure_libero_config()
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.bddl_utils import robosuite_parse_problem

        self.entry = entry
        self.env = OffScreenRenderEnv(
            bddl_file_name=entry["bddl"], camera_heights=hw, camera_widths=hw, control_freq=control_freq
        )
        self.init_states = load_init_states(entry["init"])
        self.language = self.env.language_instruction
        self.orig_goal = robosuite_parse_problem(entry["orig_bddl"])["goal_state"] if entry["dual"] else None
        self.orig_missing: list[str] = []

    def reset(self, init_idx: int, seed: int) -> dict[str, Any]:
        np.random.seed(seed % 2**32)
        self.env.seed(seed)
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[init_idx])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = self.env.step(DUMMY_ACTION)
        for robot in self.env.env.robots:
            robot.controller.use_delta = True
        e = self.env.env
        if self.orig_goal is not None:
            self.orig_missing = sorted(
                {n for s in self.orig_goal for n in s[1:] if n not in e.object_states_dict}
            )
        self.probe = ContactProbe(e)
        return obs

    def step(self, a: np.ndarray) -> tuple[dict[str, Any], bool, bool | None]:
        obs, _, _, _ = self.env.step(np.asarray(a, dtype=np.float64))
        pro = bool(self.env.check_success())
        orig = None
        if self.orig_goal is not None and not self.orig_missing:
            e = self.env.env
            orig = all(e._eval_predicate(s) for s in self.orig_goal)
        return obs, pro, orig

    def contacts(self) -> tuple[list[str], list[str]]:
        return self.probe()

    def close(self) -> None:
        self.env.close()


# ---------------------------------------------------------------- mock env (tests)


def text_pos(s: str) -> np.ndarray:
    """Same as parallel/mock_backend.text_pos (where the mock policy heads for an instruction)."""
    h = hashlib.sha256(s.encode()).digest()
    return (np.frombuffer(h[:3], np.uint8).astype(np.float64) / 255.0 - 0.5) * 0.4


class MockProEnv:
    """Robosuite-like raw obs; PRO goal = near text_pos(PRO bddl language), orig goal = near text_pos(original)."""

    def __init__(self, entry: dict[str, Any], hw: int = 32):
        self.entry, self.hw = entry, hw
        self.language = entry["instruction_bddl"]
        self.goal = text_pos(self.language)
        self.orig = text_pos(entry["instruction_original"])
        self.init_states = np.zeros((50, 4))
        self.orig_missing: list[str] = []

    def reset(self, init_idx: int, seed: int) -> dict[str, Any]:
        np.random.seed(seed % 2**32)
        self.pos = self.goal + np.random.uniform(-0.25, 0.25, 3) + 0.001 * init_idx
        self.grip = -1.0
        return self._obs()

    def step(self, a: np.ndarray) -> tuple[dict[str, Any], bool, bool | None]:
        a = np.asarray(a, dtype=np.float64)
        self.pos = self.pos + 0.02 * np.clip(a[:3], -1, 1)
        self.grip = float(a[6])
        near = lambda p: bool(np.linalg.norm(self.pos - p) < 0.03)  # noqa: E731
        return self._obs(), near(self.goal), (near(self.orig) if self.entry["dual"] else None)

    def contacts(self) -> tuple[list[str], list[str]]:
        return [], []

    def _obs(self) -> dict[str, Any]:
        img: np.ndarray = np.empty((self.hw, self.hw, 3), np.uint8)
        img[:] = np.clip((self.pos + 0.5) * 255, 0, 255).astype(np.uint8)
        w = 0.04 if self.grip <= 0 else 0.01
        return {
            "agentview_image": img,
            "robot0_eye_in_hand_image": img[::-1].copy(),
            "robot0_eef_pos": self.pos.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([w, -w]),
        }

    def close(self) -> None:
        pass


def make_env(entry: dict[str, Any], backend: str, hw: int) -> Any:
    return MockProEnv(entry) if backend == "mock" else ProEnv(entry, hw)
