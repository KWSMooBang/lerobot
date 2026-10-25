"""Physical feasibility check for the C4/C5 swap goals with a scripted (privileged-state) policy.

For every task the scripted policy first tries the original goal (C0) so that policy bugs can be told
apart from infeasible swaps; then each swap goal BDDL is tried from the original init states 0-2.
The swap BDDLs only differ from the original in ``:goal`` / ``:language``, so loading the swap BDDL
builds the exact same MuJoCo model and the original init states can be restored into it.

The policy reads object poses from the simulator and drives the Panda with the default OSC_POSE
delta controller. Primitives: pick-and-place (onto objects or into BDDL regions), push, drawer
open/close, microwave close and stove turn-on. Each primitive retries a few grasp variants.

    export LIBERO_CONFIG_PATH=<dir with config.yaml>   # see scripts/pid_demo/setup_libero.py
    uv run python scripts/shortcut/feasibility.py --suites libero_goal --workers 32

Results go to ``instructions/feasibility_<suite>.json`` (per-attempt details) and the ``feasible``
field of every swap in ``instructions/<suite>.json`` (True if any of the init states succeeded).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INSTR_DIR = HERE / "instructions"
REPO = HERE.parent.parent
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

TABLE_CLEAR = 0.012  # min fingertip height above the support surface
MAX_STEPS = 4000
ROT_SPEED = 0.3  # max |rotation action| (x0.5 rad per step)


def _quat2axisangle(q: np.ndarray) -> np.ndarray:
    # q = (w, x, y, z)
    w = np.clip(q[0], -1.0, 1.0)
    if w < 0:
        q = -q
        w = -w
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3)
    return q[1:] / s * 2.0 * np.arccos(w)


def _mat2quat(r: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, r.reshape(-1))
    return q


def down_rot(theta: float) -> np.ndarray:
    """Gripper pointing down, fingers closing along world-xy direction ``theta``."""
    c0 = np.array([np.cos(theta), np.sin(theta), 0.0])
    c2 = np.array([0.0, 0.0, -1.0])
    c1 = np.cross(c2, c0)
    return np.stack([c0, c1, c2], axis=1)


DEBUG = bool(os.environ.get("FEAS_DEBUG"))


def log(*a) -> None:
    if DEBUG:
        print("[feas]", *a, flush=True)


class EpisodeDoneError(Exception):
    pass


class Scripted:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        self.m = env.sim.model
        self.mm = env.sim.model._model
        self.d = env.sim.data
        self.grip_site = self.m.site_name2id("gripper0_grip_site")
        self.grip = -1.0
        self.t = 0
        self.ever_success = False
        self.bad_thetas: list[float] = []  # wrist angles for which a place target was unreachable
        self.table_z = self._table_z()
        self.base = self.d.body_xpos[self.m.body_name2id("robot0_base")].copy()
        self.home_pos = self.eef
        self.home_rot = down_rot(self.closing_angle())
        self._robot_bodies = {
            i
            for i in range(self.m.nbody)
            if self.m.body_id2name(i).startswith(("robot0_", "gripper0_", "mount0_"))
        }

    # ------------------------------------------------------------------ state
    def _table_z(self) -> float:
        sites = {self.m.site_id2name(i) for i in range(self.m.nsite)}
        for name in ["table_top", "kitchen_table_top", "living_room_table_top", "study_table_top"]:
            if name in sites:
                return float(self.d.site_xpos[self.m.site_name2id(name)][2])
        # floor scenes (libero_object)
        return 0.0

    @property
    def eef(self) -> np.ndarray:
        return self.d.site_xpos[self.grip_site].copy()

    @property
    def eef_rot(self) -> np.ndarray:
        return self.d.site_xmat[self.grip_site].reshape(3, 3).copy()

    def closing_angle(self) -> float:
        c0 = self.eef_rot[:, 0]
        return float(np.arctan2(c0[1], c0[0]))

    def nearest_theta(self, theta: float) -> float:
        """Closing axis is symmetric under pi; pick the representative closest to the current one."""
        cur = self.closing_angle()
        best = theta
        for k in (-2, -1, 0, 1, 2):
            cand = theta + k * np.pi
            if abs(cand - cur) < abs(best - cur):
                best = cand
        return best

    def body_subtree(self, root: int) -> list[int]:
        out = []
        for b in range(self.m.nbody):
            p = b
            while p != 0 and p != root:
                p = self.mm.body_parentid[p]
            if p == root:
                out.append(b)
        return out

    def geoms_of_bodies(self, bodies: list[int], collision_only: bool = True) -> list[int]:
        bs = set(bodies)
        out = []
        for g in range(self.m.ngeom):
            if self.mm.geom_bodyid[g] in bs:
                if collision_only and self.mm.geom_contype[g] == 0 and self.mm.geom_conaffinity[g] == 0:
                    continue
                out.append(g)
        return out

    def geom_corners(self, geoms: list[int]) -> np.ndarray:
        pts = []
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        for g in geoms:
            aabb = self.mm.geom_aabb[g]
            c, h = aabb[:3], aabb[3:]
            loc = c + signs * h
            r = self.d.geom_xmat[g].reshape(3, 3)
            pts.append(self.d.geom_xpos[g] + loc @ r.T)
        return np.concatenate(pts, axis=0)

    def obj_body(self, name: str) -> int:
        return self.env.env.obj_body_id[name]

    def obj_geoms(self, name: str) -> list[int]:
        return self.geoms_of_bodies(self.body_subtree(self.obj_body(name)))

    def obj_pos(self, name: str) -> np.ndarray:
        return self.d.body_xpos[self.obj_body(name)].copy()

    def obj_shape(self, name: str) -> dict:
        pts = self.geom_corners(self.obj_geoms(name))
        xy = pts[:, :2]
        ctr = (xy.max(0) + xy.min(0)) / 2
        cov = np.cov((xy - xy.mean(0)).T)
        w, v = np.linalg.eigh(cov)
        minor, major = v[:, 0], v[:, 1]
        p_min = (xy - ctr) @ minor
        p_maj = (xy - ctr) @ major
        ctr = ctr + minor * (p_min.max() + p_min.min()) / 2 + major * (p_maj.max() + p_maj.min()) / 2
        return {
            "center": ctr,
            "minor": minor,
            "major": major,
            "w_minor": float(p_min.max() - p_min.min()),
            "w_major": float(p_maj.max() - p_maj.min()),
            "zmin": float(pts[:, 2].min()),
            "zmax": float(pts[:, 2].max()),
            "pts": pts,
        }

    def contacts_of(self, name: str) -> set[int]:
        """Root-body ids touching object ``name`` (excluding the robot)."""
        bodies = set(self.body_subtree(self.obj_body(name)))
        out = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.mm.geom_bodyid[c.geom1], self.mm.geom_bodyid[c.geom2]
            if b1 in bodies and b2 not in bodies and b2 not in self._robot_bodies:
                out.add(int(b2))
            elif b2 in bodies and b1 not in bodies and b1 not in self._robot_bodies:
                out.add(int(b1))
        return out

    def robot_touches(self, name: str) -> bool:
        bodies = set(self.body_subtree(self.obj_body(name)))
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.mm.geom_bodyid[c.geom1], self.mm.geom_bodyid[c.geom2]
            if (b1 in bodies and b2 in self._robot_bodies) or (b2 in bodies and b1 in self._robot_bodies):
                return True
        return False

    def gripper_width(self) -> float:
        q = self.d.qpos[self.mm.jnt_qposadr[self.m.joint_name2id("gripper0_finger_joint1")]]
        q2 = self.d.qpos[self.mm.jnt_qposadr[self.m.joint_name2id("gripper0_finger_joint2")]]
        return float(q - q2)

    # ---------------------------------------------------------------- control
    def step(self, pos: np.ndarray, rot: np.ndarray | None, grip: float, speed: float = 1.0) -> None:
        if self.t >= MAX_STEPS:
            raise EpisodeDoneError
        dp = (pos - self.eef) / 0.05 * 1.5
        n = np.linalg.norm(dp)
        if n > speed:
            dp = dp / n * speed
        if rot is None:
            dr = np.zeros(3)
        else:
            err = rot @ self.eef_rot.T
            dr = _quat2axisangle(_mat2quat(err)) / 0.5 * 1.5
            nr = np.linalg.norm(dr)
            if nr > ROT_SPEED:
                dr = dr / nr * ROT_SPEED
        action = np.concatenate([dp, dr, [grip]])
        self.env.step(action)
        self.t += 1
        if self.env.check_success():
            self.ever_success = True

    def move(
        self,
        pos,
        rot=None,
        tol: float = 0.006,
        rtol: float = 0.06,
        max_steps: int = 120,
        speed: float = 1.0,
        stop_fn=None,
    ) -> bool:
        pos = np.asarray(pos, dtype=float)
        for _ in range(max_steps):
            perr = float(np.linalg.norm(pos - self.eef))
            rerr = 0.0
            if rot is not None:
                rerr = float(np.linalg.norm(_quat2axisangle(_mat2quat(rot @ self.eef_rot.T))))
            if perr < tol and rerr < rtol:
                return True
            if stop_fn is not None and stop_fn():
                return True
            self.step(pos, rot, self.grip, speed)
        return False

    def gripper(self, close: bool, steps: int = 18) -> None:
        self.grip = 1.0 if close else -1.0
        hold = self.eef
        rot = self.eef_rot
        for _ in range(steps):
            self.step(hold, rot, self.grip)

    def gripper_partial(self, close_steps: int = 5) -> None:
        """Open fully, then close for a few steps and freeze (action 0 keeps the finger target)."""
        self.gripper(False, 10)
        self.gripper(True, close_steps)
        self.grip = 0.0
        self.hold(10)

    def hold(self, steps: int) -> None:
        p, r = self.eef, self.eef_rot
        for _ in range(steps):
            self.step(p, r, self.grip)

    def home(self) -> None:
        """Back to the start pose (top-down gripper) - used after side-approach primitives."""
        p = self.eef
        p[2] = max(p[2], self.home_pos[2])
        self.move(p, None, max_steps=60)
        self.move(self.home_pos, self.home_rot, max_steps=250, speed=0.6, rtol=0.04)

    def lift_to(self, z: float, rot=None) -> None:
        p = self.eef
        p[2] = max(p[2], z)
        self.move(p, rot, max_steps=80)

    # -------------------------------------------------------------- grasping
    def grasp_candidates(self, name: str) -> list[dict]:
        sh = self.obj_shape(name)
        h = sh["zmax"] - sh["zmin"]
        support = sh["zmin"]
        cands = []
        if sh["w_minor"] <= 0.07:
            th = float(np.arctan2(sh["minor"][1], sh["minor"][0]))
            # short boxes: middle of the side; tall bottles/cartons: just below the top
            z = support + max(h * 0.5, TABLE_CLEAR) if h <= 0.06 else sh["zmax"] - min(0.04, h * 0.4)
            xy = sh["center"]
            cands.append({"xy": xy, "theta": th, "z": z})
            cands.append({"xy": xy, "theta": th, "z": max(support + TABLE_CLEAR, z - 0.015)})
            if sh["w_major"] <= 0.07:
                cands.append({"xy": xy, "theta": th + np.pi / 2, "z": z})
        # height slices: footprint of the geoms crossing a given height (handles / lids excluded)
        geoms = self.obj_geoms(name)
        boxes = [self.geom_corners([g]) for g in geoms]
        for frac in (0.6, 0.4, 0.75):
            z = max(support + frac * h, support + TABLE_CLEAR)
            sl = [b for b in boxes if b[:, 2].min() <= z <= b[:, 2].max()]
            if not sl:
                continue
            xy = np.concatenate(sl)[:, :2]
            for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
                pr = xy @ axis
                other = np.array([-axis[1], axis[0]])
                po = xy @ other
                if pr.max() - pr.min() <= 0.072:
                    ctr = axis * (pr.max() + pr.min()) / 2 + other * (po.max() + po.min()) / 2
                    # along the long axis, centre on the part that is widest (the body, not the handle)
                    cands.append({"xy": ctr, "theta": float(np.arctan2(axis[1], axis[0])), "z": z})
        if not cands:
            # handles: thin geoms sticking out of the body (moka pot, mugs)
            prot = []
            for b in boxes:
                lo, hi = b.min(0), b.max(0)
                c = (lo + hi) / 2
                dist = float(np.linalg.norm(c[:2] - sh["center"]))
                w = hi[:2] - lo[:2]
                if w.min() <= 0.03 and dist > 0.03 and hi[2] - lo[2] >= 0.02:
                    k = int(np.argmin(w))
                    th = 0.0 if k == 0 else np.pi / 2
                    prot.append((dist, {"xy": c[:2], "theta": th, "z": float(c[2])}))
            prot.sort(key=lambda t: -t[0])
            cands += [c for _, c in prot[:3]]
        # rim grasp for wide / hollow objects (bowls, plates, baskets)
        ctr = sh["center"]
        to_base = self.base[:2] - ctr
        to_base /= np.linalg.norm(to_base) + 1e-9
        pts = sh["pts"]
        for ang in (0.0, 0.5, -0.5, np.pi / 2, -np.pi / 2):
            c, s = np.cos(ang), np.sin(ang)
            d = np.array([c * to_base[0] - s * to_base[1], s * to_base[0] + c * to_base[1]])
            r = float(((pts[:, :2] - ctr) @ d).max())
            xy = ctr + d * (r - 0.012)
            cands.append(
                {
                    "xy": xy,
                    "theta": float(np.arctan2(d[1], d[0])),
                    "z": max(sh["zmax"] - 0.02, support + TABLE_CLEAR),
                }
            )
        return cands

    def hand_clearance(self, name: str, xy: np.ndarray, theta: float, z: float) -> int:
        """Number of scene geoms (other than ``name``) overlapping the Panda hand footprint when the
        fingertips are at (xy, z) - used to rank grasp candidates in clutter."""
        own = set(self.body_subtree(self.obj_body(name)))
        c0 = np.array([np.cos(theta), np.sin(theta)])
        c1 = np.array([-c0[1], c0[0]])
        half = (0.11, 0.045)  # hand half-extent along / across the closing axis
        zlo, zhi = z + 0.045, z + 0.2
        rect = np.array([xy + sa * half[0] * c0 + sb * half[1] * c1 for sa in (-1, 1) for sb in (-1, 1)])
        n = 0
        for g in range(self.m.ngeom):
            b = self.mm.geom_bodyid[g]
            if b in own or b in self._robot_bodies or b == 0:
                continue
            if self.mm.geom_contype[g] == 0 and self.mm.geom_conaffinity[g] == 0:
                continue
            cr = self.geom_corners([g])
            if cr[:, 2].max() < zlo or cr[:, 2].min() > zhi:
                continue
            lo, hi = cr[:, :2].min(0), cr[:, :2].max(0)
            if (hi - lo).max() > 1.0:  # table / floor
                continue
            sep = False
            for ax in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), c0, c1):
                pr = rect @ ax
                pb = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]]) @ ax
                if pr.max() < pb.min() or pb.max() < pr.min():
                    sep = True
                    break
            n += not sep
        return n

    def lifted(self, name: str, z0: float) -> bool:
        return self.obj_shape(name)["zmin"] > z0 + 0.02 and self.robot_touches(name)

    def pick(self, name: str, carry_z: float, skip: int = 0) -> bool:
        cands = self.grasp_candidates(name)
        pen = [self.hand_clearance(name, c["xy"], c["theta"], c["z"]) for c in cands]
        cands = [cands[i] for i in sorted(range(len(cands)), key=lambda i: pen[i])]
        skip = skip % max(len(cands), 1)
        cands = cands[skip:] + cands[:skip]  # retries start from a different grasp

        def bad(th: float) -> bool:
            return any(abs(((th - b) + np.pi / 2) % np.pi - np.pi / 2) < 0.5 for b in self.bad_thetas)

        cands = [c for c in cands if not bad(c["theta"])] + [c for c in cands if bad(c["theta"])]
        for cand in cands:
            z0 = self.obj_shape(name)["zmin"]
            log(
                "pick",
                name,
                "cand",
                np.round(cand["xy"], 3),
                round(cand["z"], 3),
                round(cand["theta"], 2),
                "t",
                self.t,
            )
            th = self.nearest_theta(cand["theta"])
            rot = down_rot(th)
            above = np.array([cand["xy"][0], cand["xy"][1], max(cand["z"] + 0.12, self.eef[2])])
            self.gripper(False, 6)
            self.move(above, rot, max_steps=150)
            self.move([cand["xy"][0], cand["xy"][1], cand["z"] + 0.04], rot, max_steps=80)
            self.move([cand["xy"][0], cand["xy"][1], cand["z"]], rot, max_steps=80, speed=0.5)
            log("  at", np.round(self.eef, 3), "width", round(self.gripper_width(), 3))
            if np.linalg.norm(self.eef - np.array([cand["xy"][0], cand["xy"][1], cand["z"]])) > 0.025:
                self.lift_to(self.eef[2] + 0.08)  # blocked on the way down
                continue
            self.gripper(True, 20)
            log("  closed width", round(self.gripper_width(), 3))
            self.lift_to(self.eef[2] + 0.08, rot)
            log(
                "  lifted?",
                self.lifted(name, z0),
                "zmin",
                round(self.obj_shape(name)["zmin"], 3),
                "z0",
                round(z0, 3),
            )
            if self.lifted(name, z0):
                self.lift_to(carry_z, rot)
                if self.robot_touches(name):
                    return True
            self.gripper(False, 10)
            self.lift_to(self.eef[2] + 0.08)
        return False

    # -------------------------------------------------------------- placing
    def target_info(self, target: str) -> dict:
        """xy anchor + height of a place target (object name or BDDL region site)."""
        sites = self.env.env.object_sites_dict
        if target in sites:
            sid = self.m.site_name2id(target)
            pos = self.d.site_xpos[sid].copy()
            size = np.asarray(sites[target].size, dtype=float)
            parent = sites[target].parent_name
            # drawers: aim for the region bottom (descend until contact) so the object is not dropped
            # onto the drawer wall; elsewhere release just above the region top (drops into baskets)
            top = pos[2] - abs(size[2]) if sites[target].joints else pos[2] + size[2]
            return {"xy": pos[:2], "z": top, "site": True, "size": size, "parent": parent}
        sh = self.obj_shape(target)
        return {
            "xy": self.obj_pos(target)[:2],
            "z": sh["zmax"],
            "site": False,
            "size": None,
            "parent": target,
        }

    def place(self, name: str, target: str, xy_offset=np.zeros(2), release_clear: float = 0.02) -> bool:
        info = self.target_info(target)
        return self.place_at(name, info["xy"] + xy_offset, info["z"], release_clear)

    def place_at(self, name: str, xy: np.ndarray, z_surface: float, release_clear: float = 0.02) -> bool:
        """Carry the grasped object so that its body origin is above ``xy``, lower it until it
        touches something (or reaches ``z_surface``) and release."""
        rot = self.eef_rot
        start = self.eef
        sh = self.obj_shape(name)
        ref = self.obj_pos(name)
        off_xy = self.eef[:2] - ref[:2]
        hang = self.eef[2] - sh["zmin"]  # eef height above the object's bottom
        goal_xy = xy + off_xy
        z_release = z_surface + hang + release_clear
        lane = 0.5 * max(sh["w_major"], sh["w_minor"]) + 0.03
        carry = max(self.eef[2], z_release + 0.05, self.corridor_top(name, ref[:2], xy, lane) + hang + 0.04)
        self.move([self.eef[0], self.eef[1], carry], rot, max_steps=60)
        self.move([goal_xy[0], goal_xy[1], carry], rot, max_steps=200)
        # re-align: object may have swung in the grasp
        for _ in range(2):
            ref = self.obj_pos(name)
            goal_xy = xy + (self.eef[:2] - ref[:2])
            self.move([goal_xy[0], goal_xy[1], carry], rot, max_steps=60, tol=0.003)
        own = set(self.body_subtree(self.obj_body(name)))

        def touched() -> bool:
            # supporting contact only: something under the object (side contacts with walls/faces
            # while lowering do not count)
            zmin = self.obj_shape(name)["zmin"]
            for i in range(self.d.ncon):
                c = self.d.contact[i]
                b1, b2 = self.mm.geom_bodyid[c.geom1], self.mm.geom_bodyid[c.geom2]
                if (b1 in own) == (b2 in own):
                    continue
                other = b2 if b1 in own else b1
                if other in self._robot_bodies:
                    continue
                if c.pos[2] < zmin + 0.015:
                    return True
            return False

        reached = np.linalg.norm(self.eef[:2] - goal_xy) < 0.03
        if not reached:
            # e.g. self-collision with this wrist angle: put the object down here and let the caller
            # retry with another grasp
            log("place: target unreachable with this grasp", np.round(self.eef, 3), np.round(goal_xy, 3))
            self.bad_thetas.append(self.closing_angle())
            self.move([self.eef[0], self.eef[1], carry], rot, max_steps=60)
            self.move([start[0], start[1], carry], rot, max_steps=200)
            self.move([start[0], start[1], self.table_z], rot, max_steps=150, speed=0.4, stop_fn=touched)
            self.gripper(False, 15)
            self.lift_to(self.eef[2] + 0.1, rot)
            return False
        self.move([goal_xy[0], goal_xy[1], z_release], rot, max_steps=150, speed=0.4, stop_fn=touched)
        log(
            "place",
            name,
            "want",
            np.round(xy, 3),
            "obj",
            np.round(self.obj_pos(name), 3),
            "eef",
            np.round(self.eef, 3),
            "z_release",
            round(z_release, 3),
            "t",
            self.t,
        )
        self.gripper(False, 15)
        self.lift_to(self.eef[2] + 0.1, rot)
        return True

    def corridor_top(self, name: str, a: np.ndarray, b: np.ndarray, half_width: float = 0.07) -> float:
        """Highest obstacle top along the straight xy path a->b (carried object excluded)."""
        own = set(self.body_subtree(self.obj_body(name)))
        top = self.table_z
        ab = b - a
        seg_len = float(np.linalg.norm(ab))
        for lo, hi in self.scene_boxes(own):
            # distance from segment to box (sampled)
            for t in np.linspace(0.0, 1.0, max(2, int(seg_len / 0.02) + 1)):
                p = a + ab * t
                q = np.clip(p, lo[:2], hi[:2])
                if np.linalg.norm(p - q) < half_width:
                    top = max(top, float(hi[2]))
                    break
        return top

    def scene_boxes(self, exclude: set[int]) -> list[tuple[np.ndarray, np.ndarray]]:
        """World xy AABBs of all non-robot collision geoms except table/floor and ``exclude`` bodies."""
        out = []
        for g in range(self.m.ngeom):
            b = self.mm.geom_bodyid[g]
            if b in exclude or b in self._robot_bodies or b == 0:
                continue
            if self.mm.geom_contype[g] == 0 and self.mm.geom_conaffinity[g] == 0:
                continue
            cr = self.geom_corners([g])
            lo, hi = cr.min(0), cr.max(0)
            if (hi[:2] - lo[:2]).max() > 1.0 or hi[2] < self.table_z + 0.005:
                continue
            out.append((lo, hi))
        return out

    def free_spot(self, name: str, keep_out: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
        """A clear table spot for ``name`` (closest to where it is now), outside ``keep_out`` xy boxes."""
        own = set(self.body_subtree(self.obj_body(name)))
        sh = self.obj_shape(name)
        rad = 0.5 * max(sh["w_major"], sh["w_minor"]) + 0.03
        boxes = self.scene_boxes(own) + keep_out
        best, best_d = None, np.inf
        for x in np.arange(-0.32, 0.08, 0.02):
            for y in np.arange(-0.3, 0.32, 0.02):
                p = np.array([x, y])
                ok = True
                for lo, hi in boxes:
                    q = np.clip(p, lo[:2], hi[:2])
                    if np.linalg.norm(p - q) < rad:
                        ok = False
                        break
                dd = float(np.linalg.norm(p - sh["center"]))
                if ok and dd < best_d:
                    best, best_d = p, dd
        return best

    def drawer_sweep(self, site: str, travel: float = 0.18) -> tuple[np.ndarray, np.ndarray]:
        """xy box covered by the drawer of ``site`` once it is pulled open."""
        j = self._joint_of_site(site)
        body = self.mm.jnt_bodyid[j]
        open_dir = -(self.d.xmat[body].reshape(3, 3) @ self.mm.jnt_axis[j])
        pts = self.geom_corners(self.geoms_of_bodies([body]))
        pts = np.concatenate([pts, pts + open_dir * travel])
        lo, hi = pts.min(0), pts.max(0)
        lo[2] = -1.0
        hi[2] = 10.0
        return lo - 0.02, hi + 0.02

    # -------------------------------------------------------------- articulated
    def _joint_of_site(self, site: str) -> int:
        joints = self.env.env.object_sites_dict[site].joints
        return self.m.joint_name2id(joints[0])

    def _jq(self, j: int) -> float:
        return float(self.d.qpos[self.mm.jnt_qposadr[j]])

    def drawer(self, site: str, open_: bool) -> bool:
        """Slide drawer. The handle sits under the cabinet top, so it is approached horizontally:
        gripper pointing into the cabinet, fingers closing vertically around the handle bar."""
        j = self._joint_of_site(site)
        body = self.mm.jnt_bodyid[j]
        axis = self.d.xmat[body].reshape(3, 3) @ self.mm.jnt_axis[j]
        open_dir = -axis  # wooden/white cabinet: open when qpos < -0.14
        geoms = self.geoms_of_bodies([body])

        def handle() -> np.ndarray:
            proj = [self.d.geom_xpos[g] @ open_dir for g in geoms]
            return self.d.geom_xpos[geoms[int(np.argmax(proj))]].copy()

        c2 = -open_dir
        rots = []
        for sz in (1.0, -1.0):
            c0 = np.array([0.0, 0.0, sz])
            rots.append(np.stack([c0, np.cross(c2, c0), c2], axis=1))
        rots.sort(key=lambda r: np.linalg.norm(_quat2axisangle(_mat2quat(r @ self.eef_rot.T))))
        rot = rots[0]
        up = np.array([0.0, 0.0, 0.08])
        if open_:
            for dz in (0.0, 0.006, -0.006):
                if self._jq(j) < -0.15:
                    break
                h = handle() + np.array([0.0, 0.0, dz])
                self.gripper_partial(5)  # ~4 cm aperture: clears the cabinet top above the top-drawer handle
                log("drawer aperture", round(self.gripper_width(), 3))
                far = h + open_dir * 0.2 + up
                self.move(far, down_rot(self.nearest_theta(self.closing_angle())), max_steps=200, speed=0.6)
                r1 = self.move(far, rot, max_steps=250, speed=0.6)  # rotate in place, away from the cabinet
                if not r1:
                    rot = rots[1] if rot is rots[0] else rots[0]
                    r1 = self.move(far, rot, max_steps=250, speed=0.6)
                r2 = self.move(h + open_dir * 0.12, rot, max_steps=100, speed=0.5)
                r3 = self.move(h - open_dir * 0.005, rot, max_steps=120, speed=0.4)
                log("drawer approach", r1, r2, r3, np.round(self.eef, 3), np.round(h, 3))
                self.gripper(True, 20)
                start = self.eef
                for k in range(1, 10):
                    self.move(start + open_dir * 0.025 * k, rot, max_steps=25, speed=0.5)
                    if self._jq(j) < -0.155:
                        break
                log("drawer open", site, "q", round(self._jq(j), 3))
                self.gripper(False, 10)
                self.move(self.eef + open_dir * 0.06, rot, max_steps=40)
                self.lift_to(self.eef[2] + 0.1, rot)
            return self._jq(j) < -0.14
        # close: top-down closed fingers against the (protruding) handle, pushed back in
        self.gripper(True, 10)
        rot_d = down_rot(self.nearest_theta(float(np.arctan2(open_dir[1], open_dir[0])) + np.pi / 2))
        for it in range(5):
            if self.env.env.object_states_dict[site].is_close():
                break
            h = handle()
            z = h[2] + (0.0, 0.01, -0.008, 0.02, 0.0)[it]
            pre = h + open_dir * 0.04
            self.move([pre[0], pre[1], z + 0.12], rot_d, max_steps=200, speed=0.6)
            self.move([pre[0], pre[1], z], rot_d, max_steps=100, speed=0.4)
            end = h - open_dir * (-self._jq(j) + 0.04)
            self.move([end[0], end[1], z], rot_d, max_steps=150, speed=0.4)
            log("drawer close", site, "q", round(self._jq(j), 3))
            self.move(self.eef + open_dir * 0.05, rot_d, max_steps=40)
            self.lift_to(self.eef[2] + 0.12, rot_d)
            if self._jq(j) > -0.09:
                break
        # finish with the gripper pointing into the cabinet, hand flat (the top-down hand hits the
        # drawer above once the handle gets close to the cabinet face)
        c0 = np.cross(np.array([0.0, 0.0, 1.0]), open_dir)
        rots = [np.stack([sg * c0, np.cross(c2, sg * c0), c2], axis=1) for sg in (1.0, -1.0)]
        rots.sort(key=lambda r: np.linalg.norm(_quat2axisangle(_mat2quat(r @ self.eef_rot.T))))
        for it in range(4):
            if self.env.env.object_states_dict[site].is_close():
                break
            rot_h = rots[(it // 2) % 2]
            h = handle()
            far = h + open_dir * 0.2 + up
            self.move(far, rot_d, max_steps=200, speed=0.6)
            self.move(far, rot_h, max_steps=250, speed=0.6)
            self.move(h + open_dir * 0.06, rot_h, max_steps=150, speed=0.5)
            self.move(h - open_dir * (-self._jq(j) + 0.05), rot_h, max_steps=150, speed=0.4)
            log("drawer close (side)", site, "q", round(self._jq(j), 3))
            self.move(self.eef + open_dir * 0.1, rot_h, max_steps=60)
        return bool(self.env.env.object_states_dict[site].is_close())

    def _hinge_of(self, name: str) -> int:
        for jj in range(self.m.njnt):
            if self.m.joint_id2name(jj).startswith(name) and self.mm.jnt_type[jj] == 3:
                return jj
        raise KeyError(f"no hinge joint for {name}")

    def turn_on(self, stove: str) -> bool:
        """Rotate the knob bar counter-clockwise (qpos up) by pushing one of its ends sideways."""
        j = self._hinge_of(stove)
        body = self.mm.jnt_bodyid[j]
        zax = self.d.xmat[body].reshape(3, 3) @ self.mm.jnt_axis[j]
        self.gripper(True, 10)
        for it in range(8):
            if self._jq(j) > 0.7:
                break
            anchor = self.d.xanchor[j].copy()
            handle = [g for g in self.geoms_of_bodies([body]) if self.d.geom_xpos[g][2] > anchor[2] + 0.015]
            pts = self.geom_corners(handle)
            xy = pts[:, :2] - anchor[:2]
            w, v = np.linalg.eigh(np.cov(xy.T))
            major = v[:, 1]
            proj = xy @ major
            zmid = float((pts[:, 2].max() + pts[:, 2].min()) / 2 + 0.005)
            ends = [anchor[:2] + major * (proj.max() - 0.012), anchor[:2] + major * (proj.min() + 0.012)]
            end = ends[it % 2]
            rv = np.array([end[0] - anchor[0], end[1] - anchor[1], 0.0])
            tang = np.cross(zax, rv)
            tang /= np.linalg.norm(tang) + 1e-9
            th = self.nearest_theta(float(np.arctan2(tang[1], tang[0])) + np.pi / 2)
            rot = down_rot(th)
            pre = end - tang[:2] * 0.03
            self.move([pre[0], pre[1], zmid + 0.08], rot, max_steps=150)
            self.move([pre[0], pre[1], zmid], rot, max_steps=80, speed=0.4)
            for _k in range(4):
                cur = self.eef
                rv = np.array([cur[0] - anchor[0], cur[1] - anchor[1], 0.0])
                tg = np.cross(zax, rv)
                tg /= np.linalg.norm(tg) + 1e-9
                self.move(cur + tg * 0.025, rot, max_steps=30, speed=0.4)
            self.lift_to(zmid + 0.08, rot)
        return self._jq(j) >= 0.5

    def push(self, name: str, target: str) -> bool:
        info = self.target_info(target)
        self.gripper(True, 10)
        if self.obj_shape(name)["w_minor"] > 0.07:
            return self.drag_inside(name, info["xy"])
        for _ in range(8):
            sh = self.obj_shape(name)
            c = self.obj_pos(name)[:2]
            d = info["xy"] - c
            dist = float(np.linalg.norm(d))
            if dist < 0.01:
                break
            u = d / dist
            r = float(((sh["pts"][:, :2] - sh["center"]) @ (-u)).max())
            z = sh["zmin"] + 0.015
            th = self.nearest_theta(float(np.arctan2(u[1], u[0])) + np.pi / 2)
            rot = down_rot(th)
            pre = sh["center"] - u * (r + 0.03)
            self.move([pre[0], pre[1], sh["zmax"] + 0.08], rot, max_steps=150)
            if not self.move([pre[0], pre[1], z], rot, max_steps=80, speed=0.5, tol=0.01):
                log("push: pre-push pose blocked", np.round(self.eef, 3))
                self.lift_to(sh["zmax"] + 0.08, rot)
                break
            step = min(dist, 0.12)
            end = pre + u * (step + 0.03)
            self.move([end[0], end[1], z], rot, max_steps=120, speed=0.4)
            log("push", name, "obj", np.round(self.obj_pos(name), 3), "goal", np.round(info["xy"], 3))
            self.lift_to(sh["zmax"] + 0.08, rot)
        return True

    def drag_inside(self, name: str, goal_xy: np.ndarray) -> bool:
        """Move a wide/concave object (plate, bowl) by lowering the closed fingers inside it and
        dragging: the fingertips catch the inner rim on the far side."""
        for _ in range(4):
            sh = self.obj_shape(name)
            if np.linalg.norm(goal_xy - self.obj_pos(name)[:2]) < 0.01:
                break
            rot = down_rot(self.nearest_theta(self.closing_angle()))
            ctr = sh["center"]
            self.move([ctr[0], ctr[1], sh["zmax"] + 0.08], rot, max_steps=150)
            self.move(
                [ctr[0], ctr[1], sh["zmin"] + 0.003],
                rot,
                max_steps=100,
                speed=0.3,
                stop_fn=lambda: self.robot_touches(name),
            )
            z = self.eef[2] - 0.003  # press lightly: friction + inner rim drag the object
            for _k in range(40):
                d = goal_xy - self.obj_pos(name)[:2]
                if np.linalg.norm(d) < 0.008:
                    break
                stepv = d / max(np.linalg.norm(d), 1e-9) * min(np.linalg.norm(d), 0.02)
                tgt = self.eef + np.array([stepv[0], stepv[1], 0.0])
                tgt[2] = z
                self.move(tgt, rot, max_steps=8, speed=0.4)
            log("drag", name, "obj", np.round(self.obj_pos(name), 3), "goal", np.round(goal_xy, 3))
            self.lift_to(sh["zmax"] + 0.08, rot)
        return True

    def microwave(self, name: str, open_: bool) -> bool:
        """Swing the door by pushing near its free edge (closed fingers, tangential pushes)."""
        j = self._hinge_of(name)
        body = self.mm.jnt_bodyid[j]
        anchor = self.d.xanchor[j].copy()
        zax = self.d.xmat[body].reshape(3, 3) @ self.mm.jnt_axis[j]
        geoms = self.geoms_of_bodies(self.body_subtree(body))
        self.gripper(True, 10)
        for _ in range(5):
            q = self._jq(j)
            if (not open_ and q > -0.005) or (open_ and q < -1.3):
                break
            pts = self.geom_corners(geoms)
            rv = pts - anchor
            rv[:, 2] = 0
            far = pts[int(np.argmax(np.linalg.norm(rv, axis=1)))]
            rvec = far - anchor
            rvec[2] = 0
            # closing increases qpos -> rotate positively about the hinge axis
            sgn = 1.0 if not open_ else -1.0
            tang = sgn * np.cross(zax, rvec)
            tang /= np.linalg.norm(tang) + 1e-9
            pin = anchor + rvec * 0.8
            z = float(np.clip(anchor[2] + 0.05, self.table_z + 0.03, None))
            z = max(z, float(pts[:, 2].min()) + 0.04)
            th = self.nearest_theta(float(np.arctan2(tang[1], tang[0])) + np.pi / 2)
            rot = down_rot(th)
            pre = pin - tang * 0.05
            self.move([pre[0], pre[1], float(pts[:, 2].max()) + 0.05], rot, max_steps=150)
            self.move([pre[0], pre[1], z], rot, max_steps=100, speed=0.5)
            for _k in range(6):
                rvec = self.eef - anchor
                rvec[2] = 0
                tang = sgn * np.cross(zax, rvec)
                tang /= np.linalg.norm(tang) + 1e-9
                self.move(self.eef + tang * 0.06, rot, max_steps=25, speed=0.5)
            self.move(self.eef - tang * 0.04, rot, max_steps=30)
            self.lift_to(float(pts[:, 2].max()) + 0.08, rot)
        q = self._jq(j)
        return q > -0.005 if not open_ else q < -1.3


# ---------------------------------------------------------------------- planning
def make_plan(goal_state: list[list[str]], verb: str) -> list[tuple]:
    first, middle, last = [], [], []
    for pred in goal_state:
        p = pred[0].lower()
        if p == "open":
            first.append(("open", pred[1]))
        elif p == "turnon":
            first.append(("turnon", pred[1]))
        elif p == "close":
            last.append(("close", pred[1]))
        elif p in ("on", "in"):
            middle.append(("push" if verb == "push" else "pickplace", pred[1], pred[2], p))
        else:
            raise ValueError(f"unsupported predicate {pred}")
    return first + middle + last


def run_plan(pol: Scripted, plan: list[tuple]) -> None:
    placed_on: dict[str, int] = {}
    n_into: dict[str, int] = {}
    for st in plan:
        if st[0] in ("push", "pickplace"):
            n_into[st[2]] = n_into.get(st[2], 0) + 1
    for step in plan:
        kind = step[0]
        if kind == "open":
            site = step[1]
            if site in pol.env.env.object_sites_dict:
                pol.drawer(site, True)
            else:
                pol.microwave(site, True)
            pol.home()
        elif kind == "close":
            site = step[1]
            if site in pol.env.env.object_sites_dict and pol.env.env.object_sites_dict[site].joints:
                pol.drawer(site, False)
            else:
                pol.microwave(site, False)
            pol.home()
        elif kind == "turnon":
            pol.turn_on(step[1])
            pol.home()
        elif kind in ("push", "pickplace"):
            obj, tgt, pname = step[1], step[2], step[3]

            def done(pname: str = pname, obj: str = obj, tgt: str = tgt) -> bool:
                return bool(pol.env.env._eval_predicate([pname, obj, tgt]))

            if kind == "push":
                pol.push(obj, tgt)
                if done():
                    continue
                log("push failed, falling back to pick-and-place")
                pol.home()
            site = pol.env.env.object_sites_dict.get(tgt)
            if site is not None and site.joints and not pol.env.env.object_states_dict[tgt].is_open():
                # e.g. "open the top drawer and put the bowl inside": if the object sits where the drawer
                # will slide out, move it to a clear spot first so it can still be grasped afterwards
                lo, hi = pol.drawer_sweep(tgt)
                sh = pol.obj_shape(obj)
                pmin, pmax = sh["pts"].min(0), sh["pts"].max(0)
                if np.all(pmax[:2] > lo[:2]) and np.all(pmin[:2] < hi[:2]):
                    spot = pol.free_spot(obj, [(lo, hi)])
                    log("relocate", obj, "to", spot)
                    if spot is not None and pol.pick(obj, pol.table_z + 0.25):
                        pol.place_at(obj, spot, pol.table_z)
                    pol.home()
                pol.drawer(tgt, True)
                pol.home()
            n = placed_on.get(tgt, 0)
            placed_on[tgt] = n + 1
            for attempt in range(3):
                info = pol.target_info(tgt)
                sh = pol.obj_shape(obj)
                hgt = sh["zmax"] - sh["zmin"]
                carry_z = max(pol.table_z + 0.3, info["z"] + hgt + 0.12)
                offset = np.zeros(2)
                if n_into[tgt] > 1 and pname == "on":
                    # two objects into one region (both moka pots / both groceries): put them side by
                    # side along the object's narrow axis, on opposite sides of the region centre
                    half = 0.06 if info["size"] is None else 0.8 * float(min(info["size"][:2]))
                    amt = min(half, 0.5 * sh["w_minor"] + 0.006)
                    offset = (-1.0 if n == 0 else 1.0) * amt * sh["minor"]
                if site is not None and site.joints:
                    # drawer: keep clear of the cabinet face above the opening
                    j = pol._joint_of_site(tgt)
                    b = pol.mm.jnt_bodyid[j]
                    od = -(pol.d.xmat[b].reshape(3, 3) @ pol.mm.jnt_axis[j])
                    offset = offset + od[:2] * 0.025
                if pol.pick(obj, carry_z, skip=attempt):
                    pol.place(obj, tgt, offset)
                pol.gripper(False, 5)
                pol.lift_to(pol.table_z + 0.3)
                log("pickplace", obj, tgt, "attempt", attempt, "ok", done())
                if done():
                    break
                pol.home()
        # move away so contacts with the gripper do not affect predicates
        pol.gripper(False, 5)
        pol.lift_to(pol.table_z + 0.3)


def _make_env(bddl: str):
    from libero.libero.envs.env_wrapper import ControlEnv

    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        horizon=MAX_STEPS + 200,
        ignore_done=True,
    )
    env.seed(0)
    env.reset()
    return env


def run_one(job: dict) -> dict:
    """One rollout: (bddl, init state idx) -> goal reached?"""
    from libero.libero import benchmark

    t0 = time.time()
    out = dict(job)
    try:
        ts = benchmark.get_benchmark_dict()[job["suite"]]()
        init = ts.get_task_init_states(job["task_id"])[job["init_idx"]]
        env = _make_env(job["bddl_abs"])
        env.set_init_state(init)
        pol = Scripted(env)
        # let the scene settle like the official eval (zero actions)
        for _ in range(10):
            env.step(np.zeros(7).tolist()[:6] + [-1.0])
        plan = make_plan(env.env.parsed_problem["goal_state"], job.get("verb", ""))
        out["plan"] = [list(p) for p in plan]
        with contextlib.suppress(EpisodeDoneError):
            run_plan(pol, plan)
        for _ in range(20):
            env.step([0, 0, 0, 0, 0, 0, -1.0])
        # LIBERO counts an episode as solved as soon as the goal holds at any step (done=True); keep the
        # end-of-episode state as well for reference
        out["success_at_end"] = bool(env.check_success())
        out["success"] = bool(pol.ever_success or out["success_at_end"])
        out["steps"] = pol.t
        env.close()
    except Exception as e:  # noqa: BLE001
        out["success"] = False
        out["success_at_end"] = False
        out["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    out["sec"] = round(time.time() - t0, 1)
    return out


def build_jobs(suite: str, inits: list[int], only_tasks: list[int] | None, only_c0: bool) -> list[dict]:
    from libero.libero import benchmark, get_libero_path

    spec = json.loads((INSTR_DIR / f"{suite}.json").read_text())
    ts = benchmark.get_benchmark_dict()[suite]()
    jobs = []
    for t in spec["tasks"]:
        tid = t["task_id"]
        if only_tasks is not None and tid not in only_tasks:
            continue
        task = ts.get_task(tid)
        orig = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        verb = t["original"].split()[0]
        for i in inits:
            jobs.append(
                {"suite": suite, "task_id": tid, "swap": "C0", "bddl_abs": orig, "init_idx": i, "verb": verb}
            )
        if only_c0:
            continue
        for s in t["swaps"]:
            for i in inits:
                jobs.append(
                    {
                        "suite": suite,
                        "task_id": tid,
                        "swap": s["id"],
                        "bddl_abs": str(REPO / s["goal_bddl"]),
                        "init_idx": i,
                        "verb": s["instruction"].split()[0],
                    }
                )
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=SUITES)
    ap.add_argument("--inits", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tasks", nargs="+", type=int, default=None)
    ap.add_argument("--only-c0", action="store_true")
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    ap.add_argument("--no-write", action="store_true", help="do not update the instruction JSONs")
    ap.add_argument("--summarize", action="store_true", help="only rebuild JSON fields from saved rollouts")
    args = ap.parse_args()

    jobs = []
    for s in args.suites:
        jobs += [] if args.summarize else build_jobs(s, args.inits, args.tasks, args.only_c0)
    print(f"{len(jobs)} rollouts on {args.workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(args.workers, maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            results.append(r)
            tag = ("OK " if r["success_at_end"] else "ok*") if r["success"] else "-- "
            print(
                f"{tag} {r['suite']} t{r['task_id']} {r['swap']} init{r['init_idx']} {r['sec']}s "
                f"{r.get('error', '').splitlines()[0] if r.get('error') else ''}",
                flush=True,
            )

    summary: dict[str, dict] = {}
    for s in args.suites:
        rs = sorted(
            [r for r in results if r["suite"] == s], key=lambda r: (r["task_id"], r["swap"], r["init_idx"])
        )
        if not rs and not args.summarize:
            continue
        for r in rs:
            r.pop("bddl_abs", None)
        if args.no_write:
            continue
        path = INSTR_DIR / f"feasibility_{s}.json"
        old = json.loads(path.read_text())["rollouts"] if path.exists() else []
        key = lambda r: (r["task_id"], r["swap"], r["init_idx"])  # noqa: E731
        merged = {key(r): r for r in old}
        merged.update({key(r): r for r in rs})
        allr = sorted(merged.values(), key=key)
        path.write_text(json.dumps({"suite": s, "rollouts": allr}, indent=1) + "\n")
        spec_path = INSTR_DIR / f"{s}.json"
        spec = json.loads(spec_path.read_text())
        for t in spec["tasks"]:
            c0 = [r for r in allr if r["task_id"] == t["task_id"] and r["swap"] == "C0"]
            if c0:
                t["scripted_c0_success"] = sum(r["success"] for r in c0)
                t["scripted_c0_trials"] = len(c0)
            for sw in t["swaps"]:
                rr = [r for r in allr if r["task_id"] == t["task_id"] and r["swap"] == sw["id"]]
                if rr:
                    sw["feasible"] = any(r["success"] for r in rr)
                    sw["feasible_trials"] = f"{sum(r['success'] for r in rr)}/{len(rr)}"
                    sw.pop("feasible_note", None)
                    if not sw["feasible"] and t.get("scripted_c0_success") == 0:
                        sw["feasible_note"] = (
                            "scripted policy also fails the original task (C0) here, so this is a policy limitation, "
                            "not evidence that the swap is physically infeasible; needs teleop or a better primitive"
                        )
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
        summary[s] = {
            "c0_tasks_solved": sum(1 for t in spec["tasks"] if t.get("scripted_c0_success", 0) > 0),
            "c0_rollouts": f"{sum(t.get('scripted_c0_success', 0) for t in spec['tasks'])}/"
            f"{sum(t.get('scripted_c0_trials', 0) for t in spec['tasks'])}",
            "swaps": sum(len(t["swaps"]) for t in spec["tasks"]),
            "swaps_feasible": sum(1 for t in spec["tasks"] for sw in t["swaps"] if sw.get("feasible")),
            "swaps_infeasible": [
                f"t{t['task_id']} {sw['id']}"
                for t in spec["tasks"]
                for sw in t["swaps"]
                if sw.get("feasible") is False
            ],
            "paraphrases_for_review": sum(len(t.get("review", {})) for t in spec["tasks"]),
        }
    if summary:
        path = INSTR_DIR / "feasibility_summary.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        old.update(summary)
        path.write_text(json.dumps(old, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
