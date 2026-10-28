"""Training-data variants for shortcut experiments (`sc-train --data-variant ...`), without copying data.

Spec: one or more `key=value` items, separated by `;` or given as repeated `--data-variant` flags.
  suites=spatial,object,goal,10   train on a subset of the 4 suites (lerobot `dataset.episodes`)
  instr=orig|blank|paraphrase     task string given to the policy: unchanged / "" (design C9,
                                  vision-only) / a paraphrase from scripts/shortcut/instructions, picked per frame
                                  by hash(seed, frame index): reproducible across workers and --resume
  para_review=exclude|include     paraphrases still flagged for human review (6) are left out by default
  para_keep=<p>                   with instr=paraphrase, keep the original sentence with probability p (0)

`install(variant)` patches `lerobot.scripts.lerobot_train.make_train_eval_datasets` in the training
process: it sets `cfg.dataset.episodes` for `suites=` and switches the returned dataset's class to
`VariantLeRobotDataset`, which only rewrites `item["task"]`. Normalization stats stay those of the full
dataset (the metadata is shared).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

from lerobot.datasets import LeRobotDataset

INSTR_DIR = Path(__file__).resolve().parents[1] / "instructions"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "10": "libero_10",
    "long": "libero_10",
    **{s: s for s in SUITES},
}
INSTR_MODES = ("orig", "blank", "paraphrase")


@dataclass(frozen=True)
class DataVariant:
    suites: tuple[str, ...] | None = None  # None = all four
    instr: str = "orig"
    para_review: str = "exclude"
    para_keep: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.suites is None and self.instr == "orig"

    @property
    def id(self) -> str:
        """Canonical short string for registry/run_meta, e.g. `suites=goal;instr=blank` or `none`."""
        parts = []
        if self.suites is not None:
            short = {v: k for k, v in SUITE_ALIASES.items() if k in ("spatial", "object", "goal", "10")}
            parts.append("suites=" + ",".join(short[s] for s in self.suites))
        if self.instr != "orig":
            parts.append(f"instr={self.instr}")
            if self.instr == "paraphrase":
                if self.para_review != "exclude":
                    parts.append(f"para_review={self.para_review}")
                if self.para_keep:
                    parts.append(f"para_keep={self.para_keep:g}")
        return ";".join(parts) or "none"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str | None) -> DataVariant:
        if not s:
            return cls()
        d = json.loads(s)
        if d.get("suites") is not None:
            d["suites"] = tuple(d["suites"])
        return cls(**d)


def parse_variant(specs: list[str] | str | None) -> DataVariant:
    if specs is None:
        return DataVariant()
    if isinstance(specs, str):
        specs = [specs]
    kv: dict[str, str] = {}
    for spec in specs:
        for item in spec.split(";"):
            item = item.strip()
            if not item or item == "none":
                continue
            if "=" not in item:
                raise ValueError(f"data variant item {item!r} is not key=value")
            k, v = item.split("=", 1)
            kv[k.strip()] = v.strip()
    suites = None
    if "suites" in kv:
        names = [x.strip() for x in kv.pop("suites").split(",") if x.strip()]
        bad = [n for n in names if n not in SUITE_ALIASES]
        if bad or not names:
            raise ValueError(f"unknown suites {bad}; use spatial,object,goal,10")
        canon = {SUITE_ALIASES[n] for n in names}
        suites = None if canon == set(SUITES) else tuple(s for s in SUITES if s in canon)
    instr = kv.pop("instr", "orig")
    if instr not in INSTR_MODES:
        raise ValueError(f"instr must be one of {INSTR_MODES}, got {instr!r}")
    para_review = kv.pop("para_review", "exclude")
    if para_review not in ("exclude", "include"):
        raise ValueError("para_review must be exclude|include")
    para_keep = float(kv.pop("para_keep", 0.0))
    if not 0.0 <= para_keep <= 1.0:
        raise ValueError("para_keep must be in [0, 1]")
    if kv:
        raise ValueError(f"unknown data variant keys {sorted(kv)}")
    if instr != "paraphrase" and (para_review != "exclude" or para_keep):
        raise ValueError("para_review/para_keep only apply to instr=paraphrase")
    return DataVariant(suites=suites, instr=instr, para_review=para_review, para_keep=para_keep)


# ---------------------------------------------------------------------------- instruction tables
@cache
def instruction_table() -> dict[str, dict[str, Any]]:
    """original sentence -> {suite, task_id, paraphrases (all), review (paraphrase index -> memo)}."""
    table: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        for t in json.loads((INSTR_DIR / f"{suite}.json").read_text())["tasks"]:
            if t["original"] in table:
                raise ValueError(f"instruction {t['original']!r} appears in two suites")
            table[t["original"]] = {
                "suite": suite,
                "task_id": t["task_id"],
                "paraphrases": list(t.get("paraphrases", [])),
                "review": {int(k): v for k, v in (t.get("review") or {}).items()},
            }
    return table


def paraphrase_pool(include_review: bool) -> dict[str, list[str]]:
    pool = {}
    for orig, t in instruction_table().items():
        ps = [p for i, p in enumerate(t["paraphrases"]) if include_review or i not in t["review"]]
        pool[orig] = ps
    return pool


def suite_of(task: str) -> str:
    t = instruction_table().get(task)
    if t is None:
        raise KeyError(f"task {task!r} is not in scripts/shortcut/instructions/*.json")
    return t["suite"]


def episodes_for_suites(episodes_tasks: list[list[str]], suites: tuple[str, ...]) -> list[int]:
    """episodes_tasks[i] = task strings of episode i (lerobot meta.episodes['tasks'])."""
    keep = set(suites)
    return [i for i, tasks in enumerate(episodes_tasks) if all(suite_of(t) in keep for t in tasks)]


def _unit_hashes(seed: int, key: int) -> tuple[float, int]:
    """Two independent values from (seed, key): a float in [0, 1) and a 64-bit int."""
    h = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=16).digest()
    return int.from_bytes(h[:8], "little") / 2**64, int.from_bytes(h[8:], "little")


class TaskRemap:
    """Picklable, stateless task-string rewriter (runs inside dataloader workers).

    The paraphrase of a sample is a pure function of (seed, key), key = the frame's global `index`, so the
    same frame gets the same sentence in every process, worker count, epoch and after --resume.
    """

    def __init__(self, variant: DataVariant, seed: int = 0):
        self.variant = variant
        self.seed = seed
        self.pool = paraphrase_pool(variant.para_review == "include") if variant.instr == "paraphrase" else {}

    def __call__(self, task: str, key: int) -> str:
        if self.variant.instr == "blank":
            return ""
        if self.variant.instr == "paraphrase":
            ps = self.pool.get(task)
            if not ps:
                return task
            u, n = _unit_hashes(self.seed, key)
            if u < self.variant.para_keep:
                return task
            return ps[n % len(ps)]
        return task


class VariantLeRobotDataset(LeRobotDataset):
    """A LeRobotDataset whose `task` strings pass through `self._sc_remap`. Instances are made by
    switching `__class__` of an existing dataset (no reload, no copy)."""

    _sc_remap: TaskRemap

    def _sc_apply(self, item: dict) -> dict:
        if "task" in item:
            item["task"] = self._sc_remap(item["task"], int(item["index"]))
        return item

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if isinstance(out, list):
            return [self._sc_apply(x) for x in out]
        return self._sc_apply(out)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        return [self._sc_apply(x) for x in super().__getitems__(indices)]


def wrap_dataset(dataset: Any, variant: DataVariant, seed: int = 0) -> Any:
    if dataset is None or variant.instr == "orig":
        return dataset
    if type(dataset) is not LeRobotDataset:
        raise TypeError(f"instr variants need a map-style LeRobotDataset, got {type(dataset).__name__}")
    dataset.__class__ = VariantLeRobotDataset
    dataset._sc_remap = TaskRemap(variant, seed)
    return dataset


def install(variant: DataVariant, eval_variant: str = "same") -> None:
    """Patch lerobot-train's dataset factory in this process (call before `lerobot_train.main()`).

    `eval_variant` applies to lerobot-train's held-out split (`dataset.eval_split` > 0, offline validation
    loss every `eval_steps`; env-rollout eval does not use it): "same" = the training variant's task strings
    (validation loss on the training distribution), "none" = original instructions. The held-out episodes
    are the same either way (they come from the suite-filtered episode list).
    """
    if eval_variant not in ("same", "none"):
        raise ValueError(f"eval_variant must be same|none, got {eval_variant!r}")
    if variant.is_identity:
        return
    import lerobot.scripts.lerobot_train as lt
    from lerobot.datasets import (
        LeRobotDatasetMetadata,
        make_train_eval_datasets as orig,
    )

    def make_train_eval_datasets(cfg: Any) -> tuple[Any, Any]:
        if variant.suites is not None:
            meta = LeRobotDatasetMetadata(
                cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
            )
            eps = episodes_for_suites(list(meta.episodes["tasks"]), variant.suites)
            if cfg.dataset.episodes is not None:
                eps = [e for e in eps if e in set(cfg.dataset.episodes)]
            cfg.dataset.episodes = eps
        train_ds, eval_ds = orig(cfg)
        seed = cfg.seed or 0
        train_ds = wrap_dataset(train_ds, variant, seed)
        if eval_variant == "same":
            eval_ds = wrap_dataset(eval_ds, variant, seed)
        sample = [train_ds[i]["task"] for i in (0, len(train_ds) // 2, len(train_ds) - 1)]
        print(
            f"[sc-train] data variant {variant.id}: {train_ds.num_episodes} episodes, "
            f"{len(train_ds)} frames, sample tasks {sample}; held-out split: "
            f"{'none' if eval_ds is None else f'{eval_ds.num_episodes} episodes, variant {eval_variant}'}",
            flush=True,
        )
        return train_ds, eval_ds

    setattr(lt, "make_train_eval_datasets", make_train_eval_datasets)  # noqa: B010
