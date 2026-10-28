"""CPU checks for T011 sc-train: uv run pytest scripts/shortcut/tests/test_sc_train.py -q

The dataset tests read the local HuggingFaceVLA/libero copy (skipped when it is absent).
"""

import json
import pickle
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))

from data_variants import (  # noqa: E402
    DataVariant,
    TaskRemap,
    episodes_for_suites,
    instruction_table,
    paraphrase_pool,
    parse_variant,
    wrap_dataset,
)
from registry import COLUMNS, add_entries, ensure_seed_rows, lookup, read_registry  # noqa: E402
from sc_plugins import register_local_plugins  # noqa: E402
from sc_train import build_plan, load_preset, main, scale_for  # noqa: E402

DATA_ROOT = Path("/outputs/VLA-shortcut-exps/data/hf_libero")
needs_data = pytest.mark.skipif(
    not (DATA_ROOT / "meta" / "info.json").exists(), reason="no local LIBERO copy"
)


# ---------------------------------------------------------------------------- variant parsing
def test_parse_variant():
    assert parse_variant(None).is_identity
    assert parse_variant("none").id == "none"
    v = parse_variant(["suites=goal,spatial", "instr=blank"])
    assert v.suites == ("libero_spatial", "libero_goal") and v.instr == "blank"
    assert v.id == "suites=spatial,goal;instr=blank"
    assert parse_variant("suites=spatial,object,goal,10").suites is None  # all four = no subset
    assert parse_variant("instr=paraphrase;para_keep=0.5").id == "instr=paraphrase;para_keep=0.5"
    assert DataVariant.from_json(v.to_json()) == v
    for bad in ("suites=foo", "instr=nope", "bogus=1", "instr=blank;para_keep=0.3", "suites"):
        with pytest.raises(ValueError):
            parse_variant(bad)


def test_instruction_table_and_review_exclusion():
    table = instruction_table()
    assert len(table) == 40
    flagged = sum(len(t["review"]) for t in table.values())
    assert flagged == 6
    excl, incl = paraphrase_pool(False), paraphrase_pool(True)
    assert sum(map(len, incl.values())) - sum(map(len, excl.values())) == 6
    for orig, t in table.items():
        for i in t["review"]:
            assert t["paraphrases"][i] not in excl[orig]
            assert t["paraphrases"][i] in incl[orig]


def test_task_remap():
    orig = "put the bowl on the plate"
    assert TaskRemap(parse_variant("instr=blank"))(orig) == ""
    assert TaskRemap(parse_variant("instr=orig"))(orig) == orig
    para = TaskRemap(parse_variant("instr=paraphrase"), seed=3)
    outs = {para(orig) for _ in range(50)}
    assert orig not in outs and outs <= set(paraphrase_pool(False)[orig]) and len(outs) > 1
    keep = TaskRemap(parse_variant("instr=paraphrase;para_keep=1"))
    assert keep(orig) == orig
    assert para("an instruction not in the tables") == "an instruction not in the tables"
    assert pickle.loads(pickle.dumps(para))(orig) in paraphrase_pool(False)[orig]


def test_episodes_for_suites():
    tasks = list(instruction_table())
    by_suite = {}
    for t in tasks:
        by_suite.setdefault(instruction_table()[t]["suite"], []).append(t)
    eps = [[by_suite["libero_goal"][0]], [by_suite["libero_10"][0]], [by_suite["libero_goal"][3]]]
    assert episodes_for_suites(eps, ("libero_goal",)) == [0, 2]
    assert episodes_for_suites(eps, ("libero_10", "libero_goal")) == [0, 1, 2]


# ---------------------------------------------------------------------------- real batches
@needs_data
@pytest.mark.parametrize("spec", ["instr=blank", "instr=paraphrase", "suites=goal"])
def test_variant_changes_batch_tasks(spec):
    """The DataLoader batch (spawned workers, like lerobot-train) carries the rewritten task strings."""
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("HuggingFaceVLA/libero", root=DATA_ROOT)
    ep_tasks = list(meta.episodes["tasks"])
    v = parse_variant(spec)
    goal_eps = episodes_for_suites(ep_tasks, ("libero_goal",))
    if v.suites:
        assert len(goal_eps) == 428
        assert {instruction_table()[ep_tasks[e][0]]["suite"] for e in goal_eps} == {"libero_goal"}
    eps = goal_eps[:2] + episodes_for_suites(ep_tasks, ("libero_spatial",))[:2]
    if v.suites:
        eps = [e for e in eps if e in set(episodes_for_suites(ep_tasks, v.suites))]
    ds = LeRobotDataset("HuggingFaceVLA/libero", root=DATA_ROOT, episodes=eps, return_uint8=True)
    originals = {ep_tasks[e][0] for e in eps}
    ds = wrap_dataset(ds, v, seed=0)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=16, shuffle=True, num_workers=2, multiprocessing_context="spawn"
    )
    batch = next(iter(loader))
    tasks = batch["task"]
    assert len(tasks) == 16
    if v.instr == "blank":
        assert set(tasks) == {""}
    elif v.instr == "paraphrase":
        pool = paraphrase_pool(False)
        allowed = {p for o in originals for p in pool[o]}
        assert set(tasks) <= allowed and not set(tasks) & originals
    else:
        assert set(tasks) <= originals
        assert {instruction_table()[t]["suite"] for t in tasks} == {"libero_goal"}


# ---------------------------------------------------------------------------- registry
def test_registry(tmp_path):
    reg = tmp_path / "ckpt_registry.tsv"
    assert ensure_seed_rows(reg) == ["pi05lib", "pi05base"]
    assert ensure_seed_rows(reg) == []
    add_entries(reg, [{"ckpt_id": "a", "path_or_hub": "/x", "policy_type": "sc_template", "steps": 10}])
    with pytest.raises(ValueError):
        add_entries(reg, [{"ckpt_id": "a", "path_or_hub": "/y"}])
    add_entries(reg, [{"ckpt_id": "a", "path_or_hub": "/y", "notes": "tab\there"}], replace=True)
    rows = read_registry(reg)
    assert [r["ckpt_id"] for r in rows] == ["pi05lib", "pi05base", "a"]
    assert lookup(reg, "a")["path_or_hub"] == "/y" and lookup(reg, "a")["notes"] == "tab here"
    assert reg.read_text().splitlines()[0].split("\t") == list(COLUMNS)
    assert lookup(reg, "pi05lib")["path_or_hub"] == "lerobot/pi05_libero_finetuned"


# ---------------------------------------------------------------------------- presets / plan
@pytest.mark.parametrize(
    "name", ["sc_template_libero_all", "smolvla_libero_all", "pi05_libero_all", "pi0_libero_all"]
)
def test_presets_load(name):
    p = load_preset(name)
    assert p["policy_type"] and p["lerobot"] and p["scale"]
    assert scale_for(p, 4)["batch"] > 0


def _args(tmp_path, **kw):
    import argparse

    base = {
        "preset": "pi05_libero_all",
        "run_id": "x",
        "gpus": "0,1,2,3",
        "steps": None,
        "batch": None,
        "grad_accum": None,
        "out_root": str(tmp_path),
        "data_variant": None,
        "resume": False,
        "wandb": "off",
        "launcher": "auto",
        "extra": [],
        "dry_run": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_plan_pi05_4gpu(tmp_path):
    plan = build_plan(_args(tmp_path))
    assert plan["launcher"] == "accelerate" and "--num_processes=4" in plan["cmd"]
    assert plan["effective_batch"] == 256 and plan["optimizer_updates"] == 6000
    assert "--policy.pretrained_path=lerobot/pi05_libero" in plan["cmd"]
    assert "--policy.scheduler_decay_steps=12000" in plan["cmd"]
    assert plan["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_plan_variant_and_overrides(tmp_path):
    plan = build_plan(
        _args(
            tmp_path,
            preset="sc_template_libero_all",
            gpus="0",
            steps=123,
            batch=7,
            data_variant=["suites=goal", "instr=blank"],
            extra=["--policy.lang_cond=none"],
        )
    )
    assert plan["launcher"] == "python"
    assert "--steps=123" in plan["cmd"] and "--batch_size=7" in plan["cmd"]
    assert plan["cmd"][-1] == "--policy.lang_cond=none"
    assert json.loads(plan["env"]["SC_DATA_VARIANT"])["instr"] == "blank"
    assert plan["data_variant_id"] == "suites=goal;instr=blank"


def test_cli_register_external(tmp_path):
    assert (
        main(
            [
                "register",
                "--out-root",
                str(tmp_path),
                "--ckpt-id",
                "ext",
                "--path",
                "org/m",
                "--policy-type",
                "act",
            ]
        )
        == 0
    )
    assert lookup(tmp_path / "ckpt_registry.tsv", "ext")["policy_type"] == "act"


# ---------------------------------------------------------------------------- plugin
def test_sc_template_forward_and_loss_drops():
    register_local_plugins()
    from lerobot_policy_sc_template import ScTemplateConfig, ScTemplatePolicy

    from lerobot.configs.types import FeatureType, PolicyFeature

    cfg = ScTemplateConfig(
        device="cpu",
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        image_size=64,
        chunk_size=4,
        n_action_steps=2,
    )
    policy = ScTemplatePolicy(cfg)
    g = torch.Generator().manual_seed(0)
    b = 8
    batch = {
        "observation.images.image": torch.rand(b, 3, 64, 64, generator=g),
        "observation.images.image2": torch.rand(b, 3, 64, 64, generator=g),
        "observation.state": torch.randn(b, 8, generator=g),
        "action": torch.randn(b, 4, 7, generator=g),
        "action_is_pad": torch.zeros(b, 4, dtype=torch.bool),
        "task": ["put the bowl on the plate"] * 4 + [""] * 4,
    }
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        loss, info = policy.forward(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert "l1_loss" in info and losses[-1] < 0.7 * losses[0]
    policy.reset()
    a = policy.select_action({k: v[:1] for k, v in batch.items() if k != "task"} | {"task": "x"})
    assert a.shape == (1, 7) and len(policy._action_queue) == 1
