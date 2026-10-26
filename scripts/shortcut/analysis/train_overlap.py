"""T007: do pi05lib's failures to follow a swapped instruction relate to what the training data contained?

    uv run python scripts/shortcut/analysis/train_overlap.py \
        --runs /outputs/VLA-shortcut-exps/runs/sc-p1-pi05lib-{goal,spat}-{C4,C5} \
        --out /outputs/VLA-shortcut-exps/analysis/p1_train_overlap

CPU only. Downloads only dataset meta (tasks/episodes parquet, info.json) from the Hub, plus a small sentence
encoder (sentence-transformers/all-MiniLM-L6-v2) and the π0.5 language embedding rows of the checkpoint.

Training data. lerobot/pi05_libero_finetuned train_config.json: HuggingFaceVLA/libero, episodes=null, 6k steps
from lerobot/pi05_libero. lerobot/pi05_libero has no train_config (converted openpi checkpoint); openpi's
`pi05_libero` config trains on physical-intelligence/libero, whose meta is read too and compared task by task.
π0.5 base pretraining data is not public and out of scope.

Scene. A training task's scene = its LIBERO BDDL with :language, :obj_of_interest and :goal removed
(regions, fixtures, objects, init). Goal tasks share one scene; each Spatial task has its own layout.

Per swap (60 = Spatial 39 + Goal 21):
  exact          normalized instruction equals a training instruction; exact_scene = same / other scene
  sentence scores, max over the 40 training instructions (higher = closer):
    jaccard      token-set Jaccard
    lcs          longest common contiguous token run (tokens)
    edit_sim     1 - char Levenshtein / max length
    minilm       cosine, all-MiniLM-L6-v2 mean-pooled
    pi05emb      cosine of the mean π0.5 (PaliGemma) input token embedding over the instruction tokens
    *_nn / *_nn_scene  closest training instruction and whether it is in the swap's scene
    *_max_excl_orig    max over training instructions other than the swap's own original task
  components (slots.py; main action = last action of the sentence), episode counts in training data:
    obj_*        tasks moving (place/push) or operating (open/close/turn on) the same object, same role family
    dest_*       tasks with the same destination (relation, landmark), any verb
    combo_*      tasks with the same (role family, object, destination)
    *_any / *_scene   all training episodes / training episodes of the swap's scene
    new_*        the changed slot (C4: object, C5: destination): episodes in that role, and mention_* = the
                 word appears anywhere in a training instruction (any role)
  level          L1 exact·same scene > L2 exact·other scene > L3 combination seen, sentence new >
                 L4 all components seen, combination new > L5 some component never seen in its role
  outcome        sr_new, shortcut (= original goal reached), first grasp, failure type over the 10 episodes:
                 fail_orig_goal (original goal, new not), fail_grasp_new (C4: first grasp = new object;
                 C5: first grasp = the object to move, i.e. picked up but not put on the new target), fail_other
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import load_episodes, wilson  # noqa: E402
from slots import Action, mentions, normalize, parse  # noqa: E402

SHORTCUT_DIR = Path(__file__).resolve().parents[1]
TRAIN_REPO = "HuggingFaceVLA/libero"
PARENT_REPO = "physical-intelligence/libero"
FINETUNED_CKPT = "lerobot/pi05_libero_finetuned"
SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]  # priority when a sentence is in several
SWAP_SUITES = ["libero_spatial", "libero_goal"]
SHORT = {
    "libero_spatial": "spat",
    "libero_goal": "goal",
    "libero_object": "obj",
    "libero_10": "l10",
    "libero_90": "l90",
}
LEVELS = {
    "L1": "exact·same scene",
    "L2": "exact·other scene",
    "L3": "combo seen·sentence new",
    "L4": "components seen·combo new",
    "L5": "new component",
}
METRICS = ["jaccard", "lcs", "edit_sim", "minilm", "pi05emb"]
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
# BDDL goal arguments → slot names, to check the rule-based slots against the swap's goal predicate
BDDL_NAME = {
    "akita_black_bowl": "black bowl",
    "plate": "plate",
    "glazed_rim_porcelain_ramekin": "ramekin",
    "cookies": "cookie box",
    "flat_stove_1_cook_region": "stove",
    "wooden_cabinet_1_top_side": "wooden cabinet",
    "wooden_cabinet_1_top_region": "top drawer",
    "wooden_cabinet_1_middle_region": "middle drawer",
    "main_table_stove_front_region": "stove",
    "wine_rack_1_top_region": "wine rack",
    "cream_cheese": "cream cheese",
    "wine_bottle": "wine bottle",
}


# ---------------------------------------------------------------- data


@dataclass
class TrainTask:
    instruction: str
    suite: str
    task_id: int
    bddl: str
    scene_key: str
    n_episodes: int
    n_frames: int
    n_episodes_parent: int | None
    actions: list[Action] = field(default_factory=list)
    scene_id: str = ""


@dataclass
class Swap:
    suite: str
    task_id: int
    variant: str
    type: str
    instruction: str
    original: str
    orig_target: str
    new_target: str
    goal: str
    bddl: str
    scene_key: str
    actions: list[Action] = field(default_factory=list)

    @property
    def swap_id(self) -> str:
        return f"{SHORT[self.suite]}-t{self.task_id:02d}-{self.variant}"


def bddl_root() -> Path:
    import libero.libero

    return Path(libero.libero.__file__).parent / "bddl_files"


def strip_block(text: str, key: str) -> str:
    """Remove '(:key ...)' with balanced parentheses."""
    i = text.find(f"(:{key}")
    if i < 0:
        return text
    depth = 0
    for j in range(i, len(text)):
        depth += {"(": 1, ")": -1}.get(text[j], 0)
        if depth == 0:
            return text[:i] + text[j + 1 :]
    raise ValueError(f"unbalanced (:{key}")


def scene_key(path: Path) -> str:
    text = path.read_text()
    for key in ("language", "obj_of_interest", "goal"):
        text = strip_block(text, key)
    text = re.sub(r"\(define \(problem \w+\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def object_locations(path: Path) -> dict[str, tuple[str, tuple[float, float] | None]]:
    """Initial place of each object from the BDDL :init: (label, table xy of the region centre).
    Label = table region ('next_to_ramekin'), or what it stands on / is in ('on cookies_1', 'in wooden_cabinet_1_top_region')."""
    text = path.read_text()
    regs = {}
    for m in re.finditer(r"\((\w+)\s*\(:target (\w+)\)\s*\(:ranges \(\s*\(([-0-9. e]+)\)", text):
        x0, y0, x1, y1 = map(float, m.group(3).split())
        regs[f"{m.group(2)}_{m.group(1)}"] = ((x0 + x1) / 2, (y0 + y1) / 2)
    init = text[text.find("(:init") :]
    init = init[: init.find("(:goal")]
    rel = {m.group(2): (m.group(1), m.group(3)) for m in re.finditer(r"\((On|In) (\w+) (\w+)\)", init)}

    def xy(o: str, depth: int = 0) -> tuple[float, float] | None:
        if o in regs:
            return regs[o]
        if depth > 5:
            return None
        if o in rel:
            return xy(rel[o][1], depth + 1)
        base = re.sub(r"_(top|middle|bottom)_(region|side)$|_cook_region$", "", o)
        return xy(base, depth + 1) if base != o else None

    out = {}
    for o, (pred, sup) in rel.items():
        label = (
            sup.removeprefix("main_table_").removesuffix("_region")
            if sup in regs
            else f"{pred.lower()} {sup}"
        )
        out[o] = (label, xy(o))
    return out


def libero_tasks() -> dict[str, list[tuple[str, int, str]]]:
    """normalized language → [(suite, task_id, bddl relative path)] over the 4 trained suites."""
    from libero.libero import benchmark

    out: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    bd = benchmark.get_benchmark_dict()
    for suite in SUITES:
        b = bd[suite]()
        for i in range(b.n_tasks):
            t = b.get_task(i)
            out[normalize(t.language)].append((suite, i, f"{t.problem_folder}/{t.bddl_file}"))
    return out


def hub_meta(repo: str, files: list[str]) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    return {f: Path(hf_hub_download(repo, f, repo_type="dataset")) for f in files}


def load_training() -> tuple[list[TrainTask], dict[str, Any]]:
    import pandas as pd

    meta = hub_meta(
        TRAIN_REPO, ["meta/info.json", "meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet"]
    )
    info = json.loads(meta["meta/info.json"].read_text())
    eps = pd.read_parquet(
        meta["meta/episodes/chunk-000/file-000.parquet"], columns=["episode_index", "tasks", "length"]
    )
    assert len(eps) == info["total_episodes"], "episode meta is chunked; read all chunks"
    n_eps: Counter[str] = Counter()
    n_frames: Counter[str] = Counter()
    for tasks, length in zip(eps["tasks"], eps["length"], strict=True):
        assert len(tasks) == 1, f"multi-task episode {tasks}"
        n_eps[tasks[0]] += 1
        n_frames[tasks[0]] += int(length)

    parent = hub_meta(PARENT_REPO, ["meta/info.json", "meta/episodes.jsonl"])
    p_info = json.loads(parent["meta/info.json"].read_text())
    p_eps: Counter[str] = Counter()
    for line in parent["meta/episodes.jsonl"].read_text().splitlines():
        r = json.loads(line)
        p_eps[r["tasks"][0]] += 1

    lib = libero_tasks()
    root = bddl_root()
    tasks = []
    for text in sorted(n_eps):
        hits = lib.get(normalize(text))
        assert hits, f"training task not in LIBERO 4 suites: {text!r}"
        suite, tid, bddl = min(hits, key=lambda h: SUITES.index(h[0]))
        tasks.append(
            TrainTask(
                instruction=text,
                suite=suite,
                task_id=tid,
                bddl=bddl,
                scene_key=scene_key(root / bddl),
                n_episodes=n_eps[text],
                n_frames=n_frames[text],
                n_episodes_parent=p_eps.get(text),
                actions=parse(text),
            )
        )
    # readable scene ids: one per distinct scene_key, named by suite + first task
    ids: dict[str, str] = {}
    for t in sorted(tasks, key=lambda t: (SUITES.index(t.suite), t.task_id)):
        ids.setdefault(t.scene_key, f"{SHORT[t.suite]}-scene-t{t.task_id:02d}")
    for t in tasks:
        t.scene_id = ids[t.scene_key]
    prov = {
        "train_repo": TRAIN_REPO,
        "train_info": {
            k: info[k] for k in ("codebase_version", "total_episodes", "total_frames", "total_tasks", "fps")
        },
        "parent_repo": PARENT_REPO,
        "parent_info": {
            k: p_info[k] for k in ("codebase_version", "total_episodes", "total_frames", "total_tasks")
        },
        "parent_same_counts": dict(p_eps) == dict(n_eps),
        "suites": dict(Counter(t.suite for t in tasks)),
        "episodes_by_suite": {s: sum(t.n_episodes for t in tasks if t.suite == s) for s in SUITES},
        "n_scenes": len(ids),
    }
    return tasks, prov


def load_swaps(train: list[TrainTask]) -> list[Swap]:
    root = bddl_root()
    keys = {t.scene_key for t in train}
    swaps = []
    for suite in SWAP_SUITES:
        spec = json.loads((SHORTCUT_DIR / "instructions" / f"{suite}.json").read_text())
        for t in spec["tasks"]:
            sk = scene_key(root / t["bddl"])
            assert sk in keys, f"{suite} task {t['task_id']} scene not in training data"
            for w in t["swaps"]:
                if not w.get("feasible", True):
                    continue
                swaps.append(
                    Swap(
                        suite=suite,
                        task_id=t["task_id"],
                        variant=w["id"],
                        type=w["type"],
                        instruction=w["instruction"],
                        original=t["original"],
                        orig_target=w["orig_target"],
                        new_target=w["new_target"],
                        goal=w["goal"],
                        bddl=t["bddl"],
                        scene_key=sk,
                        actions=parse(w["instruction"]),
                    )
                )
    return swaps


# ---------------------------------------------------------------- outcomes


def same_object(grasp: str | None, obj: str | None) -> bool:
    """as p1_tables.py: grasp names are bodies, goal arguments can be regions of them."""
    return bool(grasp and obj and (grasp == obj or obj.startswith(grasp + "_")))


def goal_args(goal: str) -> tuple[str, str, str | None]:
    m = re.search(r"\(And \((\w+) (\w+)(?: (\w+))?\)\)", goal)
    assert m, goal
    return m.group(1), m.group(2), m.group(3)


def outcomes(runs: list[str], swaps: list[Swap]) -> dict[tuple[str, int, str], dict[str, Any]]:
    by: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in load_episodes(runs):
        if r["model"] != "pi05lib" or r["condition"] not in ("C4", "C5"):
            continue
        by[(r["suite"], r["task_id"], r["variant"])].append(r)
    out = {}
    for s in swaps:
        eps = by[(s.suite, s.task_id, s.variant)]
        assert eps, f"no episodes for {s.swap_id}"
        assert all(r["instruction"] == s.instruction for r in eps), s.swap_id
        moved = goal_args(s.goal)[1]  # object the new goal moves (C5: the original object)
        n = len(eps)
        k_new = sum(bool(r["success_new"]) for r in eps)
        k_orig = sum(bool(r["success_orig"]) for r in eps)
        g = [r.get("first_grasp_object") for r in eps]
        grasp_new = sum(same_object(x, s.new_target if s.type == "object" else moved) for x in g)
        fails = [r for r in eps if not r["success_new"]]
        f_orig = sum(bool(r["success_orig"]) for r in fails)
        f_grasp = sum(
            (not r["success_orig"])
            and same_object(r.get("first_grasp_object"), s.new_target if s.type == "object" else moved)
            for r in fails
        )
        out[(s.suite, s.task_id, s.variant)] = {
            "n_ep": n,
            "k_new": k_new,
            "k_orig": k_orig,
            "sr_new": k_new / n,
            "shortcut": k_orig / n,
            "first_grasp_new": grasp_new / n if s.type == "object" else math.nan,
            "first_grasp_moved_obj": sum(same_object(x, moved) for x in g) / n,
            "first_grasp_orig": sum(same_object(x, s.orig_target) for x in g) / n
            if s.type == "object"
            else math.nan,
            "fail_orig_goal": f_orig,
            "fail_grasp_new": f_grasp,
            "fail_other": len(fails) - f_orig - f_grasp,
        }
    return out


# ---------------------------------------------------------------- sentence scores


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lcs_run(a: list[str], b: list[str]) -> int:
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def minilm_embed(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MINILM)
    model = AutoModel.from_pretrained(MINILM).eval()
    with torch.no_grad():
        b = tok(texts, padding=True, return_tensors="pt")
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1)
    e = torch.nn.functional.normalize(e, dim=-1)
    return e.numpy()


def pi05_embed(texts: list[str]) -> np.ndarray:
    """Mean of π0.5's PaliGemma input embeddings (= tied lm_head rows) over the instruction tokens, as the
    processor cleans them (strip, '_' → ' '), without BOS and the 'Task: ...' template."""
    from common import load_tokenizer
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    tok = load_tokenizer()
    ids = [tok.encode(t.strip().replace("_", " "), add_special_tokens=False) for t in texts]
    path = hf_hub_download(FINETUNED_CKPT, "model.safetensors")
    with safe_open(path, "pt") as f:
        w = f.get_tensor("model.paligemma_with_expert.paligemma.lm_head.weight").float()
    # raw token embeddings share a large common direction (all sentence cosines ≈ 0.997); centre on the vocab mean
    w = (w - w.mean(0, keepdim=True)).numpy()
    e = np.stack([w[seq].mean(0) for seq in ids])
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def sentence_scores(swaps: list[Swap], train: list[TrainTask], embed: bool) -> list[dict[str, Any]]:
    tr_norm = [normalize(t.instruction) for t in train]
    sw_norm = [normalize(s.instruction) for s in swaps]
    sims: dict[str, np.ndarray] = {}
    sims["jaccard"] = np.array(
        [
            [len(set(a.split()) & set(b.split())) / len(set(a.split()) | set(b.split())) for b in tr_norm]
            for a in sw_norm
        ]
    )
    sims["lcs"] = np.array([[lcs_run(a.split(), b.split()) for b in tr_norm] for a in sw_norm], dtype=float)
    sims["edit_sim"] = np.array([[1 - lev(a, b) / max(len(a), len(b)) for b in tr_norm] for a in sw_norm])
    if embed:
        for name, fn in (("minilm", minilm_embed), ("pi05emb", pi05_embed)):
            e = fn([s.instruction for s in swaps] + [t.instruction for t in train])
            sims[name] = e[: len(swaps)] @ e[len(swaps) :].T
    rows = []
    for i, s in enumerate(swaps):
        same = np.array([t.scene_key == s.scene_key for t in train])
        own = np.array([normalize(t.instruction) == normalize(s.original) for t in train])
        exact = [t for t, n in zip(train, tr_norm, strict=True) if n == sw_norm[i]]
        row: dict[str, Any] = {
            "exact": bool(exact),
            "exact_scene": ("same" if any(t.scene_key == s.scene_key for t in exact) else "other")
            if exact
            else "",
            "exact_train_task": ";".join(f"{SHORT[t.suite]}-t{t.task_id:02d}" for t in exact),
        }
        for m in METRICS:
            if m not in sims:
                continue
            v = sims[m][i]
            j = int(np.argmax(v))
            row[f"{m}_max"] = round(float(v[j]), 4)
            row[f"{m}_nn"] = train[j].instruction
            row[f"{m}_nn_task"] = f"{SHORT[train[j].suite]}-t{train[j].task_id:02d}"
            row[f"{m}_nn_scene"] = "same" if same[j] else "other"
            row[f"{m}_max_same_scene"] = round(float(v[same].max()), 4)
            row[f"{m}_max_excl_orig"] = round(float(v[~own].max()), 4)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- components


def role(verb: str) -> str:
    return "move" if verb in ("place", "push") else "operate"


def component_counts(s: Swap, train: list[TrainTask]) -> dict[str, Any]:
    a = s.actions[-1]
    changed = "obj" if s.type == "object" else "dest"

    def eps(pred: Any, scene_only: bool = False) -> int:
        return sum(
            t.n_episodes
            for t in train
            if (not scene_only or t.scene_key == s.scene_key) and any(pred(b) for b in t.actions)
        )

    obj = lambda b: role(b.verb) == role(a.verb) and b.obj_ref == a.obj_ref  # noqa: E731
    obj_cat = lambda b: role(b.verb) == role(a.verb) and b.obj == a.obj  # noqa: E731
    dest = lambda b: a.dest is not None and b.dest == a.dest  # noqa: E731
    combo = lambda b: obj(b) and b.dest == a.dest and (a.dest is not None or b.verb == a.verb)  # noqa: E731
    verb = lambda b: b.verb == a.verb  # noqa: E731
    word = a.obj if changed == "obj" else (a.dest[1] if a.dest else "")
    row: dict[str, Any] = {
        "slot_verb": a.verb,
        "slot_obj": a.obj,
        "slot_obj_ref": a.obj_ref,
        "slot_dest": a.dest_str,
        "changed_slot": changed,
        "new_component": a.obj_ref if changed == "obj" else a.dest_str,
    }
    for name, pred in (("obj", obj), ("obj_cat", obj_cat), ("dest", dest), ("combo", combo), ("verb", verb)):
        row[f"{name}_eps_any"] = eps(pred)
        row[f"{name}_eps_scene"] = eps(pred, scene_only=True)
    if a.dest is None:  # open/close/turn on: no destination slot
        row["dest_eps_any"] = row["dest_eps_scene"] = ""
    row["new_eps_any"] = row[f"{changed}_eps_any"]
    row["new_eps_scene"] = row[f"{changed}_eps_scene"]
    row["new_word"] = word
    row["new_mention_eps_any"] = sum(t.n_episodes for t in train if word in mentions(t.instruction))
    row["new_mention_eps_scene"] = sum(
        t.n_episodes for t in train if t.scene_key == s.scene_key and word in mentions(t.instruction)
    )
    row["verb_in_scene"] = row["verb_eps_scene"] > 0
    row["obj_in_scene"] = row["obj_eps_scene"] > 0
    row["dest_in_scene"] = bool(row["dest_eps_scene"]) if a.dest else ""
    row["components_seen"] = (
        row["verb_eps_any"] > 0 and row["obj_eps_any"] > 0 and (a.dest is None or row["dest_eps_any"] > 0)
    )
    row["combo_seen"] = row["combo_eps_any"] > 0
    # slot check against the swap's BDDL goal predicate
    pred, x, y = goal_args(s.goal)
    want_obj = BDDL_NAME.get(re.sub(r"_\d+$", "", x), BDDL_NAME.get(x, x))
    want_dest = BDDL_NAME.get(y, BDDL_NAME.get(re.sub(r"_\d+$", "", y or ""), y)) if y else None
    if pred == "Open":
        ok = a.verb == "open" and a.obj == want_obj
    else:
        ok = a.obj == want_obj and a.dest is not None and a.dest[1] == want_dest
    row["slot_check"] = "ok" if ok else f"MISMATCH goal={s.goal}"
    return row


def location_cols(s: Swap, train: list[TrainTask]) -> dict[str, Any]:
    """C4 swaps: where the original and the new object start in this scene, and for an exact training
    sentence, where the object that sentence refers to started in its training scene."""
    if s.type != "object":
        return {}
    root = bddl_root()
    loc = object_locations(root / s.bddl)
    (lo, xo), (ln, xn) = loc.get(s.orig_target, ("", None)), loc.get(s.new_target, ("", None))
    row: dict[str, Any] = {
        "orig_loc": lo,
        "new_loc": ln,
        "orig_new_dist_m": round(math.dist(xo, xn), 3) if xo and xn else "",
    }
    exact = [
        t
        for t in train
        if normalize(t.instruction) == normalize(s.instruction) and t.scene_key != s.scene_key
    ]
    if exact:
        t = exact[0]
        text = (root / t.bddl).read_text()
        moved = re.search(r"\(:goal\s*\(And\s*\(\w+\s+(\w+)", text)
        assert moved
        row["trained_loc"] = object_locations(root / t.bddl)[moved.group(1)][0]
        row["new_at_trained_loc"] = row["trained_loc"] == ln
        row["orig_at_trained_loc"] = row["trained_loc"] == lo
    return row


def level(r: dict[str, Any]) -> str:
    if r["exact"]:
        return "L1" if r["exact_scene"] == "same" else "L2"
    if r["combo_seen"]:
        return "L3"
    if r["components_seen"]:
        return "L4"
    return "L5"


# ---------------------------------------------------------------- aggregation


def fmt_ci(k: int, n: int) -> str:
    if not n:
        return ""
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def by_level(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for suite in ["libero_spatial", "libero_goal", "all"]:
        for typ in ["all", "object", "target"]:
            for lv in LEVELS:
                sel = [
                    r
                    for r in rows
                    if (suite == "all" or r["suite"] == suite)
                    and (typ == "all" or r["type"] == typ)
                    and r["level"] == lv
                ]
                n = sum(r["n_ep"] for r in sel)
                kn = sum(r["k_new"] for r in sel)
                ko = sum(r["k_orig"] for r in sel)
                lo_n, hi_n = wilson(kn, n) if n else (math.nan, math.nan)
                lo_o, hi_o = wilson(ko, n) if n else (math.nan, math.nan)
                out.append(
                    {
                        "suite": suite,
                        "type": typ,
                        "level": lv,
                        "level_name": LEVELS[lv],
                        "n_swaps": len(sel),
                        "n_ep": n,
                        "k_new": kn,
                        "sr_new": round(kn / n, 4) if n else "",
                        "sr_new_lo": round(lo_n, 4) if n else "",
                        "sr_new_hi": round(hi_n, 4) if n else "",
                        "k_orig": ko,
                        "shortcut": round(ko / n, 4) if n else "",
                        "shortcut_lo": round(lo_o, 4) if n else "",
                        "shortcut_hi": round(hi_o, 4) if n else "",
                        "swap_mean_sr_new": round(float(np.mean([r["sr_new"] for r in sel])), 4)
                        if sel
                        else "",
                        "swaps": " ".join(r["swap_id"] for r in sel),
                    }
                )
    return out


SCORES = [f"{m}_max" for m in METRICS] + [
    "level_ord",
    "new_eps_any",
    "new_eps_scene",
    "new_mention_eps_any",
    "combo_eps_any",
]


def spearman(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups = [("all", "all")] + [
        (s, t) for s in ("libero_spatial", "libero_goal") for t in ("all", "object", "target")
    ]
    for suite, typ in groups:
        sel = [
            r for r in rows if (suite == "all" or r["suite"] == suite) and (typ == "all" or r["type"] == typ)
        ]
        for sc in SCORES:
            if sc not in sel[0]:
                continue
            x = np.array([float(r[sc]) for r in sel])
            for y_name in ("sr_new", "shortcut"):
                y = np.array([r[y_name] for r in sel])
                if np.ptp(x) == 0 or np.ptp(y) == 0:
                    rho, p = math.nan, math.nan
                else:
                    rho, p = spearmanr(x, y)
                out.append(
                    {
                        "suite": suite,
                        "type": typ,
                        "score": sc,
                        "outcome": y_name,
                        "n_swaps": len(sel),
                        "rho": round(float(rho), 3),
                        "p": round(float(p), 4),
                    }
                )
    return out


def find_videos(manifest: list[dict[str, Any]], r: dict[str, Any]) -> str:
    hits = [
        m["mp4"]
        for m in manifest
        if "pi05lib" in m["model"].split("+")
        and m["suite"] == r["suite"]
        and m["task_id"] == r["task_id"]
        and (r["variant"] in m["conditions"] or f"pi05lib {r['variant']}" in m["conditions"])
    ]
    return ";".join(hits)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{path} ({len(rows)} rows)")


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--manifest", type=Path, default=Path("/outputs/VLA-shortcut-exps/analysis/p1/manifest.json")
    )
    ap.add_argument(
        "--conclusions", type=Path, default=Path(__file__).with_name("train_overlap_conclusions.md")
    )
    ap.add_argument("--no-embed", action="store_true", help="skip MiniLM / π0.5 embedding scores")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = [p for g in args.runs for p in sorted(glob.glob(g))]

    train, prov = load_training()
    swaps = load_swaps(train)
    print(
        f"training: {len(train)} instructions, {sum(t.n_episodes for t in train)} episodes; swaps: {len(swaps)}"
    )

    write_csv(
        [
            {
                "instruction": t.instruction,
                "suite": t.suite,
                "task_id": t.task_id,
                "bddl": t.bddl,
                "scene_id": t.scene_id,
                "n_episodes": t.n_episodes,
                "n_frames": t.n_frames,
                "n_episodes_parent": t.n_episodes_parent,
                "actions": " ; ".join(f"{a.verb}({a.obj_ref} → {a.dest_str})" for a in t.actions),
            }
            for t in sorted(train, key=lambda t: (SUITES.index(t.suite), t.task_id))
        ],
        args.out / "train_tasks.csv",
    )

    # slots: every training instruction and every swap, one row per action, for human checking
    slot_rows = []
    for t in sorted(train, key=lambda t: (SUITES.index(t.suite), t.task_id)):
        for k, a in enumerate(t.actions):
            slot_rows.append(
                {
                    "role": "train",
                    "id": f"{SHORT[t.suite]}-t{t.task_id:02d}",
                    "action": k,
                    "instruction": t.instruction,
                    "verb_surface": a.surface,
                    "verb": a.verb,
                    "obj": a.obj,
                    "obj_ref": a.obj_ref,
                    "dest": a.dest_str,
                    "scene_id": t.scene_id,
                    "n_episodes": t.n_episodes,
                    "bddl_goal": "",
                    "slot_check": "",
                }
            )
    comp = {s.swap_id: component_counts(s, train) for s in swaps}
    scene_of = {t.scene_key: t.scene_id for t in train}
    for s in swaps:
        for k, a in enumerate(s.actions):
            slot_rows.append(
                {
                    "role": "swap",
                    "id": s.swap_id,
                    "action": k,
                    "instruction": s.instruction,
                    "verb_surface": a.surface,
                    "verb": a.verb,
                    "obj": a.obj,
                    "obj_ref": a.obj_ref,
                    "dest": a.dest_str,
                    "scene_id": scene_of[s.scene_key],
                    "n_episodes": "",
                    "bddl_goal": s.goal,
                    "slot_check": comp[s.swap_id]["slot_check"] if k == len(s.actions) - 1 else "",
                }
            )
    write_csv(slot_rows, args.out / "slots.csv")

    sent = sentence_scores(swaps, train, embed=not args.no_embed)
    outc = outcomes(runs, swaps)
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else []
    rows = []
    for s, sc in zip(swaps, sent, strict=True):
        r: dict[str, Any] = {
            "suite": s.suite,
            "swap_id": s.swap_id,
            "task_id": s.task_id,
            "variant": s.variant,
            "type": s.type,
            "instruction": s.instruction,
            "original": s.original,
            "orig_target": s.orig_target,
            "new_target": s.new_target,
            "scene_id": scene_of[s.scene_key],
        }
        r.update(sc)
        r.update(comp[s.swap_id])
        r.update(location_cols(s, train))
        r["level"] = level(r)
        r["level_name"] = LEVELS[r["level"]]
        r["level_ord"] = 6 - int(r["level"][1])  # 5 = L1 (most overlap) … 1 = L5
        r.update(outc[(s.suite, s.task_id, s.variant)])
        r["videos"] = find_videos(manifest, r)
        rows.append(r)
    bad = [r["swap_id"] for r in rows if r["slot_check"] != "ok"]
    assert not bad, f"slot check failed: {bad}"
    write_csv(rows, args.out / "swap_overlap.csv")

    lv_rows = by_level(rows)
    write_csv(lv_rows, args.out / "by_level.csv")
    sp_rows = spearman(rows)
    write_csv(sp_rows, args.out / "spearman.csv")

    chart = []
    for r in rows:
        for sc in SCORES:
            if sc in r:
                chart.append(
                    {
                        "suite": r["suite"],
                        "swap_id": r["swap_id"],
                        "type": r["type"],
                        "level": r["level"],
                        "metric": sc,
                        "score": r[sc],
                        "sr_new": r["sr_new"],
                        "shortcut": r["shortcut"],
                        "n": r["n_ep"],
                    }
                )
    write_csv(chart, args.out / "chart_overlap.csv")

    # cases: overlap order = (level, minilm or jaccard), descending
    key_m = "minilm_max" if "minilm_max" in rows[0] else "jaccard_max"
    order = sorted(rows, key=lambda r: (r["level_ord"], r[key_m], -r["sr_new"]), reverse=True)
    hi_fail = [r for r in order if r["sr_new"] <= 0.2][:5]
    lo_succ = [r for r in reversed(order) if r["sr_new"] >= 0.5][:5]
    case_rows = [dict(case="high_overlap_fail", **_case(r, key_m)) for r in hi_fail] + [
        dict(case="low_overlap_success", **_case(r, key_m)) for r in lo_succ
    ]
    write_csv(case_rows, args.out / "cases.csv")

    write_summary(args, prov, train, rows, lv_rows, sp_rows, case_rows)


def _case(r: dict[str, Any], key_m: str) -> dict[str, Any]:
    return {
        "swap_id": r["swap_id"],
        "instruction": r["instruction"],
        "level": r["level"],
        key_m: r[key_m],
        "nn": r[key_m.replace("_max", "_nn")],
        "sr_new": r["sr_new"],
        "shortcut": r["shortcut"],
        "fail_orig_goal": r["fail_orig_goal"],
        "fail_grasp_new": r["fail_grasp_new"],
        "fail_other": r["fail_other"],
        "video": r["videos"] or f"(no video) task {r['task_id']}, init 0-9, {r['variant']}",
    }


def write_summary(args, prov, train, rows, lv_rows, sp_rows, case_rows) -> None:  # type: ignore[no-untyped-def]
    lines = []
    lines.append("# T007 — pi05lib 지시 미이행 vs 학습 instruction 겹침\n")
    lines.append(
        f"생성: `scripts/shortcut/analysis/train_overlap.py` · 입력 {len(rows)} swap × 10 ep (P1 pi05lib C4/C5).\n"
    )
    if args.conclusions.exists():
        lines.append("## 결론\n")
        lines.append(args.conclusions.read_text().strip() + "\n")
    lines.append("## 학습 데이터\n")
    ti = prov["train_info"]
    lines.append(
        f"- pi05lib = `{FINETUNED_CKPT}` train_config: `{prov['train_repo']}` (episodes=null → 전체), "
        f"{ti['total_tasks']} instruction, {ti['total_episodes']} episode, {ti['total_frames']} frame. "
        f"suite: "
        + ", ".join(
            f"{s} {prov['suites'].get(s, 0)} task / {prov['episodes_by_suite'][s]} ep" for s in SUITES
        )
        + ". **LIBERO-90 없음.**"
    )
    lines.append(
        f"- 부모 `lerobot/pi05_libero` 에는 train_config 가 없다 (openpi 변환본). openpi `pi05_libero` 설정의 데이터셋 "
        f"`{prov['parent_repo']}` meta: {prov['parent_info']['total_tasks']} task / {prov['parent_info']['total_episodes']} ep, "
        f"task 별 episode 수 동일 = {prov['parent_same_counts']} → 같은 instruction 집합으로 본다."
    )
    lines.append(
        "- π0.5 base 사전학습 데이터는 비공개 → 범위 밖 (base 가 비슷한 문장을 봤을 가능성은 배제 못 함)."
    )
    lines.append(
        f"- 장면 (BDDL 의 regions·fixtures·objects·init): {prov['n_scenes']} 개. Goal 10 task = 1 장면, Spatial task = 각자 다른 배치.\n"
    )

    lines.append("## 겹침 수준별 (episode pooled, Wilson 95% CI)\n")
    lines.append(
        "suite 별 먼저. L1 같은 장면 완전 일치 > L2 다른 장면 완전 일치 > L3 (동사, 물체, 목적지) 조합은 학습에 있음·문장은 처음 > "
        "L4 구성요소는 각자 있음·조합은 처음 > L5 일부 구성요소가 그 역할로 처음.\n"
    )
    cols = [
        "suite",
        "type",
        "level",
        "level_name",
        "n_swaps",
        "n_ep",
        "sr_new_ci",
        "shortcut_ci",
        "swap_mean_sr_new",
    ]
    tab = []
    for r in lv_rows:
        if r["n_swaps"] == 0 or (r["suite"] == "all" and r["type"] != "all"):
            continue
        tab.append(
            {**r, "sr_new_ci": fmt_ci(r["k_new"], r["n_ep"]), "shortcut_ci": fmt_ci(r["k_orig"], r["n_ep"])}
        )
    lines.append(md_table(tab, cols) + "\n")

    lines.append("## Spatial C4-a — 학습 문장 그대로, 다른 배치 (L2)\n")
    lines.append(
        "Spatial 은 가구 (plate, cookie box, ramekin, cabinet, stove) 위치가 10 task 모두 같고 bowl 2개의 자리만 다르다. "
        "그래서 위치 수식어 ('next to the ramekin') 는 어느 장면에서나 같은 테이블 위치를 가리킨다. "
        "trained_loc = 그 문장의 학습 task 에서 집은 bowl 의 자리, new_loc / orig_loc = 이 장면의 새 / 원래 bowl 자리.\n"
    )
    l2 = [r for r in rows if r["level"] == "L2"]
    lines.append(
        md_table(
            l2,
            [
                "swap_id",
                "instruction",
                "exact_train_task",
                "orig_loc",
                "new_loc",
                "trained_loc",
                "new_at_trained_loc",
                "orig_new_dist_m",
                "sr_new",
                "shortcut",
                "first_grasp_new",
            ],
        )
        + "\n"
    )

    lines.append("## C5 — 새 목적지의 학습 노출\n")
    c5 = sorted(
        [r for r in rows if r["type"] == "target"],
        key=lambda r: (r["suite"], r["new_component"], r["swap_id"]),
    )
    lines.append(
        md_table(
            c5,
            [
                "swap_id",
                "instruction",
                "new_component",
                "dest_eps_any",
                "dest_eps_scene",
                "new_mention_eps_any",
                "level",
                "sr_new",
                "shortcut",
                "fail_grasp_new",
            ],
        )
        + "\n"
    )

    lines.append("## swap 60개\n")
    cols = [
        "swap_id",
        "type",
        "instruction",
        "level",
        "exact_train_task",
        "jaccard_max",
        "minilm_max",
        "pi05emb_max",
        "new_component",
        "new_eps_any",
        "new_eps_scene",
        "new_mention_eps_any",
        "combo_eps_any",
        "sr_new",
        "shortcut",
        "fail_orig_goal",
        "fail_grasp_new",
        "fail_other",
    ]
    lines.append(md_table(rows, [c for c in cols if c in rows[0]]) + "\n")

    lines.append("## Spearman (swap 단위)\n")
    lines.append(
        "swap 60개 (suite 별 39/21, type 별 더 적음) → 검정력 낮음. 같은 태스크·init 을 공유하는 swap 은 독립이 아니고, "
        "Spatial C4-a 는 전부 L2 라 수준과 swap 유형이 겹친다. p 는 참고용 (다중 비교 보정 없음).\n"
    )
    sel = [
        r
        for r in sp_rows
        if r["outcome"] == "sr_new"
        and r["score"] in ("level_ord", "jaccard_max", "minilm_max", "pi05emb_max", "new_eps_any")
    ]
    lines.append(md_table(sel, ["suite", "type", "score", "n_swaps", "rho", "p"]) + "\n")
    lines.append("전체 표: `spearman.csv` (shortcut 포함).\n")

    lines.append("## 사례\n")
    lines.append(md_table(case_rows, list(case_rows[0])) + "\n")

    lines.append("## 한계\n")
    lines.append(
        "- 학습 데이터는 LIBERO 4 suite 40 문장뿐이라 '부분 일치' 의 폭이 좁다: Spatial 은 전부 'pick up the black bowl … place it on the plate' 틀, "
        "Goal 은 같은 장면의 10 문장. 겹침 점수의 분산 대부분이 suite·swap 유형 (C4 물체 / C5 목적지) 차이에서 온다.\n"
        "- 슬롯은 규칙 기반 (`slots.py`); swap 60개는 BDDL goal 술어와 자동 대조 (전부 ok), 학습 40 문장은 `slots.csv` 로 사람이 확인.\n"
        "- 'black bowl' 처럼 같은 물체 이름이 다른 장면·다른 인스턴스를 가리키는 경우를 구분하지 않는다 (obj_ref 로 위치 수식어까지만 구분).\n"
        "- π0.5 임베딩 점수는 입력 토큰 임베딩 평균 (문맥 없음, vocab 평균으로 centering). 실제 모델 내부 표현 거리는 아니다. "
        "문장들이 토큰 대부분을 공유해 값이 0.99–1.00 에 몰리므로 순위로만 읽는다.\n"
        "- 'low_overlap_success' 사례는 L5 에 SR_new ≥ 50% swap 이 없어 (L5 최대 10%) 모두 Goal L4 다.\n"
        "- swap 당 10 ep (init 0–9, repeat 0). π0.5 base 사전학습 데이터 미공개.\n"
    )
    (args.out / "summary.md").write_text("\n".join(lines))
    print(f"{args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
