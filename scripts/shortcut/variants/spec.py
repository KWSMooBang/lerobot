"""Variant specs (`<variants-dir>/<suite>/<id>.yaml`), generated-file paths and the variant catalog.

    suite: libero_goal
    base_task: 8                       # scene, init states, seeds and pairing come from this training task
    id: n01_cream_cheese_on_plate      # unique in the suite, file-name safe
    instruction: put the cream cheese on the plate
    goal: [On cream_cheese_1 plate_1]  # conjunction; [] = no achievable goal (probe variants)
    phrasing: train                    # optional labels: wording style, and a key shared by the
    pair: g14                          #   wordings of one goal (reported side by side)
    track_goals: default               # default = every training task goal of the suite (t0 … t9)
    scene:                             # optional edits (any of)
      move: [{object: cream_cheese_1, to: main_table_plate_region}]   # or xy: [x, y] (region frame)
      swap: [[alphabet_soup_1, salad_dressing_1]]
      remove: [alphabet_soup_1]
      add: [{object: ketchup_1, category: ketchup, to: floor_other_object_region_1}]
      fixture_state: {wooden_cabinet_1_top_region: open, flat_stove_1: on}
    init: keep                         # keep | patch | {resample: 50}

Only this file is imported by the catalog (no LIBERO, no torch): `list_variants(suite)`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .bddl_io import parse_predicate

VARIANTS_DIR = Path(__file__).resolve().parent  # default --variants-dir (examples live here)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
INIT_MODES = ("keep", "patch", "resample")
FIXTURE_STATES = ("open", "close", "on", "off")
EDIT_KEYS = ("move", "swap", "remove", "add", "fixture_state")
SPEC_KEYS = {
    "suite", "base_task", "id", "instruction", "goal", "track_goals", "scene", "init", "group", "notes",
    "feasibility_hint", "phrasing", "pair",
}  # fmt: skip
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}")


class SpecError(ValueError):
    pass


@dataclass
class TrackGoal:
    name: str
    goal: list[list[str]] | None = None  # explicit predicates, or
    task: int | None = None  # a training task's goal


@dataclass
class SceneEdits:
    move: list[dict[str, Any]] = field(default_factory=list)  # {object, to | xy}
    swap: list[list[str]] = field(default_factory=list)  # [[a, b]]
    remove: list[str] = field(default_factory=list)
    add: list[dict[str, Any]] = field(default_factory=list)  # {object, category, to | xy}
    fixture_state: dict[str, str] = field(default_factory=dict)  # {state name: open|close|on|off}

    def empty(self) -> bool:
        return not (self.move or self.swap or self.remove or self.add or self.fixture_state)

    def changes_model(self) -> bool:
        """Edits that change the MuJoCo model (other objects) — init states cannot be reused as they are."""
        return bool(self.remove or self.add)

    def kinds(self) -> list[str]:
        return [k for k in EDIT_KEYS if getattr(self, k)]


@dataclass
class VariantSpec:
    suite: str
    base_task: int | str
    id: str
    instruction: str
    goal: list[list[str]]
    track_goals: str | list[TrackGoal] = "default"
    scene: SceneEdits = field(default_factory=SceneEdits)
    init_mode: str = "keep"
    init_n: int | None = None  # resample N
    group: str | None = None
    notes: str | None = None
    feasibility_hint: str | None = None
    phrasing: str | None = None  # free label, e.g. human | train | bddl (same goal, other wording)
    pair: str | None = None  # key shared by the phrasings of one goal (goal17: g01 … g17, ctl_t00 …)
    path: str | None = None
    sha256: str | None = None  # of the YAML file

    def perturbation_dimension(self) -> str:
        """records.DIMENSIONS value for episodes of this variant."""
        if self.scene.remove or self.scene.add:
            return "object"
        if not self.scene.empty():
            return "layout"
        return "language"

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("path", None)
        return d


def _as_list(x: Any, what: str) -> list[Any]:
    if x is None:
        return []
    if not isinstance(x, list):
        raise SpecError(f"{what} must be a list, got {type(x).__name__}")
    return x


def _placement(d: Any, what: str, need_category: bool = False) -> dict[str, Any]:
    if not isinstance(d, dict) or "object" not in d:
        raise SpecError(f"{what}: each entry needs `object` (and `to: <region>` or `xy: [x, y]`), got {d!r}")
    if ("to" in d) == ("xy" in d):
        raise SpecError(f"{what} {d['object']}: give exactly one of `to` (region) or `xy`")
    if "xy" in d and (not isinstance(d["xy"], list) or len(d["xy"]) != 2):
        raise SpecError(f"{what} {d['object']}: xy must be [x, y]")
    if need_category and "category" not in d:
        raise SpecError(f"{what} {d['object']}: needs `category` (asset name, e.g. ketchup)")
    unknown = set(d) - {"object", "to", "xy", "category", "yaw"}
    if unknown:
        raise SpecError(f"{what} {d['object']}: unknown keys {sorted(unknown)}")
    out = dict(d)
    if "xy" in out:
        out["xy"] = [float(v) for v in out["xy"]]
    return out


def parse_spec(data: dict[str, Any], path: str | None = None) -> VariantSpec:
    """Validate a YAML mapping (no LIBERO needed; scene names are checked by the generator)."""
    where = path or "<spec>"
    if not isinstance(data, dict):
        raise SpecError(f"{where}: top level must be a mapping")
    unknown = set(data) - SPEC_KEYS
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)} (known: {sorted(SPEC_KEYS)})")
    for k in ("suite", "base_task", "id", "instruction", "goal"):
        if k not in data:
            raise SpecError(f"{where}: missing `{k}`")
    if data["suite"] not in SUITES:
        raise SpecError(f"{where}: suite must be one of {SUITES}")
    vid = str(data["id"])
    if not _ID.fullmatch(vid):
        raise SpecError(f"{where}: id {vid!r} must be file-name safe ([A-Za-z0-9_.-], ≤ 81 chars)")
    try:
        goal = [parse_predicate(p) for p in _as_list(data["goal"], "goal")]
    except ValueError as e:
        raise SpecError(f"{where}: goal: {e}") from e

    track: str | list[TrackGoal]
    tg = data.get("track_goals", "default")
    if tg in (None, "default"):
        track = "default"
    elif tg == "none":
        track = []
    else:
        track = []
        for i, t in enumerate(_as_list(tg, "track_goals")):
            if isinstance(t, int):
                track.append(TrackGoal(name=f"t{t}", task=t))
            elif isinstance(t, dict) and ("goal" in t) != ("task" in t):
                name = str(t.get("name") or (f"t{t['task']}" if "task" in t else f"g{i}"))
                if "goal" in t:
                    try:
                        preds = [parse_predicate(p) for p in _as_list(t["goal"], f"track_goals[{i}].goal")]
                    except ValueError as e:
                        raise SpecError(f"{where}: track_goals[{i}]: {e}") from e
                    track.append(TrackGoal(name=name, goal=preds))
                else:
                    track.append(TrackGoal(name=name, task=int(t["task"])))
            else:
                raise SpecError(
                    f"{where}: track_goals[{i}] must be a task id or {{name, goal: [...]}} / {{name, task: k}}"
                )
        names = [t.name for t in track]
        if len(set(names)) != len(names):
            raise SpecError(f"{where}: track_goals names must be unique: {names}")

    sc = data.get("scene") or {}
    if not isinstance(sc, dict):
        raise SpecError(f"{where}: scene must be a mapping")
    bad = set(sc) - set(EDIT_KEYS)
    if bad:
        raise SpecError(f"{where}: unknown scene edits {sorted(bad)} (known: {EDIT_KEYS})")
    swaps = []
    for s in _as_list(sc.get("swap"), "scene.swap"):
        if not (isinstance(s, list) and len(s) == 2 and s[0] != s[1]):
            raise SpecError(f"{where}: scene.swap entries are [object_a, object_b], got {s!r}")
        swaps.append([str(s[0]), str(s[1])])
    fstate = sc.get("fixture_state") or {}
    if not isinstance(fstate, dict):
        raise SpecError(f"{where}: scene.fixture_state must be a mapping {{name: open|close|on|off}}")
    # YAML 1.1 reads bare on / off as booleans
    fstate = {k: ({True: "on", False: "off"}[v] if isinstance(v, bool) else v) for k, v in fstate.items()}
    for k, v in fstate.items():
        if v not in FIXTURE_STATES:
            raise SpecError(f"{where}: fixture_state {k}: {v!r} not in {FIXTURE_STATES}")
    edits = SceneEdits(
        move=[_placement(m, "scene.move") for m in _as_list(sc.get("move"), "scene.move")],
        swap=swaps,
        remove=[str(x) for x in _as_list(sc.get("remove"), "scene.remove")],
        add=[_placement(a, "scene.add", need_category=True) for a in _as_list(sc.get("add"), "scene.add")],
        fixture_state={str(k): str(v) for k, v in fstate.items()},
    )
    touched = [m["object"] for m in edits.move] + [x for s in edits.swap for x in s] + edits.remove
    if len(set(touched)) != len(touched):
        raise SpecError(f"{where}: an object is edited twice (move / swap / remove): {touched}")

    init = data.get("init", "keep")
    n = None
    if isinstance(init, dict):
        if set(init) != {"resample"}:
            raise SpecError(f"{where}: init must be keep | patch | {{resample: N}}")
        mode, n = "resample", int(init["resample"])
        if n < 1:
            raise SpecError(f"{where}: resample N must be ≥ 1")
    else:
        mode = str(init)
    if mode not in INIT_MODES:
        raise SpecError(f"{where}: init must be keep | patch | {{resample: N}}, got {init!r}")
    if mode == "keep" and not edits.empty():
        raise SpecError(
            f"{where}: init: keep reuses the base task's init states unchanged, but the spec edits the "
            f"scene ({edits.kinds()}); use init: patch (edit the original init states) or {{resample: N}}"
        )
    if mode == "patch" and edits.add and any("xy" not in a and "to" not in a for a in edits.add):
        raise SpecError(f"{where}: added objects need a place")
    return VariantSpec(
        suite=data["suite"],
        base_task=data["base_task"],
        id=vid,
        instruction=str(data["instruction"]).strip(),
        goal=goal,
        track_goals=track,
        scene=edits,
        init_mode=mode,
        init_n=n,
        group=data.get("group"),
        notes=data.get("notes"),
        feasibility_hint=data.get("feasibility_hint"),
        phrasing=data.get("phrasing"),
        pair=data.get("pair"),
        path=path,
    )


def load_spec(path: str | Path) -> VariantSpec:
    raw = Path(path).read_bytes()
    spec = parse_spec(yaml.safe_load(raw), str(path))
    spec.sha256 = hashlib.sha256(raw).hexdigest()
    suite_dir = Path(path).resolve().parent.name
    if suite_dir in SUITES and suite_dir != spec.suite:
        raise SpecError(f"{path}: lives under {suite_dir}/ but says suite: {spec.suite}")
    if Path(path).stem != spec.id:
        raise SpecError(f"{path}: file name must be <id>.yaml (id: {spec.id})")
    return spec


# ---------------------------------------------------------------- layout on disk


def variants_dir(d: str | Path | None = None) -> Path:
    return Path(d).resolve() if d else VARIANTS_DIR


def spec_paths(suite: str, d: str | Path | None = None) -> list[Path]:
    return sorted((variants_dir(d) / suite).glob("*.yaml"))


def generated_dir(suite: str, vid: str, d: str | Path | None = None) -> Path:
    """Where `sc-variants build` writes <id>.bddl, init_states.npy and manifest.json."""
    return variants_dir(d) / suite / "generated" / vid


def manifest_path(suite: str, vid: str, d: str | Path | None = None) -> Path:
    return generated_dir(suite, vid, d) / "manifest.json"


def load_manifest(suite: str, vid: str, d: str | Path | None = None) -> dict[str, Any]:
    p = manifest_path(suite, vid, d)
    if not p.exists():
        raise FileNotFoundError(
            f"variant {suite}/{vid} is not built ({p} missing): run "
            f"scripts/shortcut/variants/sc-variants build --suite {suite} --ids {vid}"
        )
    m = json.loads(p.read_text())
    m["_dir"] = str(p.parent)
    return m


def manifest_file(m: dict[str, Any], key: str) -> Path:
    """Absolute path of a generated file named in a manifest (stored relative to the manifest dir)."""
    return Path(m["_dir"]) / m["files"][key]["path"]


def find_spec(suite: str, vid: str, d: str | Path | None = None) -> Path:
    p = variants_dir(d) / suite / f"{vid}.yaml"
    if not p.exists():
        known = [x.stem for x in spec_paths(suite, d)]
        raise FileNotFoundError(f"no variant {suite}/{vid} in {variants_dir(d)} (known: {known})")
    return p


# ---------------------------------------------------------------- catalog


def variant_rows(suite: str | None = None, d: str | Path | None = None) -> list[dict[str, Any]]:
    """One row per user variant (`sc-variants list`): suite, details [id], count (init states, None until
    built), description, requires (what is missing before it can run), base_task, group, pair, phrasing,
    goal, edits, init, feasibility, built, stale, path."""
    out: list[dict[str, Any]] = []
    for s in [suite] if suite else list(SUITES):
        for p in spec_paths(s, d):
            try:
                spec = load_spec(p)
            except SpecError as e:
                out.append(
                    {
                        "benchmark": "ours", "suite": s, "key": "custom", "levels": [], "details": [p.stem],
                        "count": None, "description": f"INVALID: {e}", "requires": ["fix the YAML"],
                        "path": str(p),
                    }
                )  # fmt: skip
                continue
            mp = manifest_path(s, spec.id, d)
            m = json.loads(mp.read_text()) if mp.exists() else None
            stale = bool(m and m.get("spec_sha256") != spec.sha256)
            requires = []
            if m is None:
                requires.append("sc-variants build")
            elif stale:
                requires.append("sc-variants build (YAML changed since the last build)")
            feas = (m or {}).get("feasibility", {}).get("status")
            goal = " ∧ ".join("(" + " ".join(g) + ")" for g in spec.goal) or "— (no achievable goal)"
            edits = ", ".join(spec.scene.kinds()) or "none"
            out.append(
                {
                    "benchmark": "ours", "suite": s, "key": "custom", "levels": [], "details": [spec.id],
                    "count": (m or {}).get("init_states", {}).get("n"),
                    "description": f"{spec.instruction!r} | goal {goal} | scene edits: {edits} | init {spec.init_mode}",
                    "requires": requires, "base_task": spec.base_task, "group": spec.group,
                    "pair": spec.pair, "phrasing": spec.phrasing,
                    "goal": spec.goal, "edits": spec.scene.kinds(), "init": spec.init_mode,
                    "feasibility": feas, "built": m is not None, "stale": stale, "path": str(p),
                }
            )  # fmt: skip
    return out


def list_variants(suite: str | None = None, d: str | Path | None = None) -> list[dict[str, Any]]:
    """Catalog entries in the bench/core CatalogEntry format (T019): one entry per suite with user variants,
    key_field `condition`, key `custom`; `details` = the variant ids a job may list in `variants:` (all
    specs, built or not); `count` = built variants; `count_by_detail` = init states per built variant;
    `options` = other closed-set job fields (`phrasing`); `requires` / `missing` = variants still to build.
    Per-variant info: `variants` (variant_rows)."""
    out: list[dict[str, Any]] = []
    rows = variant_rows(suite, d)
    for s in [suite] if suite else list(SUITES):
        rs = [r for r in rows if r["suite"] == s]
        if not rs:
            continue
        unbuilt = [r["details"][0] for r in rs if r["requires"]]
        phr = sorted({r["phrasing"] for r in rs if r.get("phrasing")})
        out.append(
            {
                "benchmark": "ours", "suite": s, "key_field": "condition", "key": "custom", "levels": [],
                "details": [r["details"][0] for r in rs],
                "count": sum(1 for r in rs if not r["requires"]),
                "count_by_detail": {r["details"][0]: r["count"] for r in rs if r["count"] is not None},
                "tasks": sorted({int(r["base_task"]) for r in rs if str(r.get("base_task", "")).isdigit()}),
                "description": f"user-designed variants ({len(rs)}; scripts/shortcut/variants): own BDDL + init "
                "states, runs to max_steps, tracks every training goal (fallback table)",
                "options": {"phrasing": phr} if phr else {},
                "requires": ["hf-libero (main venv, --extra libero)"],
                "requires_by_detail": {i: ["built variant (sc-variants build)"] for i in [r["details"][0] for r in rs]},
                "missing": [],  # the adapter fills hf-libero; unbuilt variants are per detail (not a whole-job skip)
                "missing_by_detail": {i: [f"sc-variants build --suite {s} --ids {i}"] for i in unbuilt},
                "feasibility_by_detail": {r["details"][0]: r.get("feasibility") for r in rs},
                "source": f"user variant files ({variants_dir(d)})",
                "variants": rs,
            }
        )  # fmt: skip
    return out
