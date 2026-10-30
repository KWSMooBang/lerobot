"""Regression tests for the T015 review (codex, 53824742): concurrency, identity, resume, smoke verdict.

uv run pytest scripts/shortcut/sae/tests/test_review.py -q
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

SAE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAE_DIR))
sys.path.insert(0, str(SAE_DIR / "tests"))

import collect  # noqa: E402
import smoke  # noqa: E402
import store  # noqa: E402
from model import SAEConfig, TopKSAE  # noqa: E402
from stats import encode_all, episode_metrics  # noqa: E402


# ---------------------------------------------------------------------------- 1. plan.json race, PID 1 pods
def test_unique_tag_differs_with_same_pid(monkeypatch):
    monkeypatch.setattr(os, "getpid", lambda: 1)
    assert len({store.unique_tag() for _ in range(50)}) == 50


def test_plan_created_once_under_concurrency(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "getpid", lambda: 1)  # every k8s pod is PID 1
    path = tmp_path / "acts" / "plan.json"
    plan = {"identity": {"model": "m", "x": 1}, "parts": [{"part_id": "ep0-1"}] * 200}
    created, errors = [], []
    barrier = threading.Barrier(16)

    def go():
        barrier.wait()
        try:
            created.append(store.create_or_check_plan(path, plan))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=go) for _ in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors and sorted(created) == [False] * 15 + [True]
    assert json.loads(path.read_text()) == plan
    assert [p.name for p in path.parent.iterdir()] == ["plan.json"]  # no temp files left
    with pytest.raises(SystemExit, match=r"differs in \['x'\]"):
        store.create_or_check_plan(path, {**plan, "identity": {"model": "m", "x": 2}})


# ---------------------------------------------------------------------------- 2. identity: dtype + weights id
def test_local_checkpoint_fingerprint_changes_with_weights(tmp_path):
    ck = tmp_path / "pretrained_model"
    ck.mkdir()
    (ck / "config.json").write_text("{}")
    (ck / "model.safetensors").write_bytes(b"a" * 1000)
    f1 = collect.checkpoint_fingerprint.__wrapped__(str(ck))
    (ck / "model.safetensors").write_bytes(b"b" * 1000)
    f2 = collect.checkpoint_fingerprint.__wrapped__(str(ck))
    assert f1.startswith("sha256:") and f1 != f2


def test_plan_identity_has_dtype_and_revisions(monkeypatch):
    monkeypatch.setattr(collect, "checkpoint_fingerprint", lambda c: f"hub:{c}-sha")
    args = collect.main.__globals__["argparse"].Namespace(
        model="pi05base", episodes="379", layers="pg:5", ae_steps="all", parts=1, noise_seed=0
    )
    plan = collect.make_plan(args, Path("/outputs/VLA-shortcut-exps"))
    ident = plan["identity"]
    assert ident["ckpt_revision"] == "hub:lerobot/pi05_base-sha"
    assert ident["adapter_revision"] == "hub:lerobot/pi05_libero_finetuned-sha"
    assert "bfloat16" in ident["dtype"]


# ---------------------------------------------------------------------------- 3. per-writer tmp, lease, publish
def test_lease_fresh_blocks_stale_taken_over(tmp_path):
    assert store.try_lease(tmp_path, "ep1-2")
    assert not store.try_lease(tmp_path, "ep1-2")  # another writer, fresh
    old = time.time() - 2 * store.LEASE_TTL_S
    os.utime(store.lease_path(tmp_path, "ep1-2"), (old, old))
    assert store.try_lease(tmp_path, "ep1-2")  # stale → taken over
    store.release_lease(tmp_path, "ep1-2")
    assert not store.lease_path(tmp_path, "ep1-2").exists()


def test_publish_keeps_first_writer(tmp_path):
    final = tmp_path / "ep1-2"
    a, b = tmp_path / ".ep1-2.tmp-A", tmp_path / ".ep1-2.tmp-B"
    for d, tag in ((a, "A"), (b, "B")):
        d.mkdir()
        (d / "done.json").write_text(tag)
    assert store.publish_part(a, final) and (final / "done.json").read_text() == "A"
    assert not store.publish_part(b, final)
    assert (final / "done.json").read_text() == "A" and not b.exists()
    # a final dir without done.json (not made by this code) is set aside, not merged into
    junk = tmp_path / "ep3-4"
    junk.mkdir()
    (junk / "x").write_text("junk")
    c = tmp_path / ".ep3-4.tmp-C"
    c.mkdir()
    (c / "done.json").write_text("C")
    assert store.publish_part(c, junk) and (junk / "done.json").read_text() == "C"
    assert len(list(tmp_path.glob(".ep3-4.stale-*"))) == 1


# ---------------------------------------------------------------------------- 4. NaN / inf fail the part
def test_part_writer_nan_inf_fail(tmp_path):
    w = collect.PartWriter(tmp_path, ["pg00"], n_rows=2, n_steps=1)
    x = np.ones((1, 3, 2048), np.float32)
    x[0, 0, 0] = np.nan
    with pytest.raises(collect.NonFiniteActivationError):
        w.write("pg00", slice(0, 1), x)
    x[0, 0, 0] = np.inf
    with pytest.raises(collect.NonFiniteActivationError):
        w.write("pg00", slice(0, 1), x)
    assert w.overflow == []  # finite > fp16 max is the only case that turns a layer fp32


DATA_ROOT = Path("/outputs/VLA-shortcut-exps/data/hf_libero")
needs_data = pytest.mark.skipif(
    not (DATA_ROOT / "meta" / "info.json").exists(), reason="no local LIBERO copy"
)


@needs_data
def test_collect_nan_leaves_no_done(tmp_path, monkeypatch):
    from test_collect_mock import MockPolicy, mock_pre

    pol = MockPolicy()
    orig = pol.predict_action_chunk

    def nan_chunk(obs, noise):  # NaN from layer 16 on → layer 17 output, second batch
        if obs["frame_index"][0] >= 16:
            h = pol.model.lm.layers[16].register_forward_hook(lambda m, a, o: o * float("nan"))
            try:
                return orig(obs, noise)
            finally:
                h.remove()
        return orig(obs, noise)

    pol.predict_action_chunk = nan_chunk
    monkeypatch.setattr(collect, "load_for_collect", lambda i, d: (pol, mock_pre, lambda a: a, {}))
    argv = ["--model", "pi05lib", "--out-root", str(tmp_path), "--episodes", "379", "--parts", "1"]
    argv += ["--gpus", "cpu", "--workers", "0", "--batch", "16", "--layers", "pg:17"]
    with pytest.raises(collect.NonFiniteActivationError):
        collect.main(argv)
    pdir = tmp_path / "sae" / "pi05lib" / "acts" / "parts"
    assert list(pdir.iterdir()) == []  # no final part, no tmp dir, no lease


# ---------------------------------------------------------------------------- 5 + 6. train done / --force
@needs_data
def test_train_done_identity_and_force(tmp_path, monkeypatch, capsys):
    import train
    from test_collect_mock import MockPolicy, mock_pre

    pol = MockPolicy()
    monkeypatch.setattr(collect, "load_for_collect", lambda i, d: (pol, mock_pre, lambda a: a, {}))
    collect.main(
        [
            "--model",
            "pi05lib",
            "--out-root",
            str(tmp_path),
            "--episodes",
            "379",
            "--parts",
            "1",
            "--gpus",
            "cpu",
            "--workers",
            "0",
            "--layers",
            "pg:5",
        ]
    )
    base = [
        "--model",
        "pi05lib",
        "--out-root",
        str(tmp_path),
        "--gpus",
        "cpu",
        "--batch",
        "32",
        "--steps",
        "40",
        "--log-every",
        "10",
        "--ckpt-every",
        "10",
        "--wandb",
        "off",
        "--no-stats",
        "--layers",
        "pg:5",
    ]
    train.main(base)
    jd = tmp_path / "sae" / "pi05lib" / "sae" / "pooled_all" / "pg05" / "seed0"
    done = json.loads((jd / "done.json").read_text())
    assert done["config"]["parts"] == ["ep00379-00379"]
    assert done["config"]["collection_identity"]["ckpt_revision"].startswith("hub:")
    # 5: same config → skipped; another config → refused, the old result is kept
    capsys.readouterr()
    train.main(base)
    assert "all seeds done, skip" in capsys.readouterr().out
    with pytest.raises(SystemExit, match=r"another config \(differs in \['k'\]\)"):
        train.main([*base, "--k-pg", "50"])
    assert json.loads((jd / "done.json").read_text()) == done
    # 6: --force run that dies (--stop-after) leaves no done.json behind
    train.main([*base, "--k-pg", "50", "--force", "--stop-after", "20"])
    assert not (jd / "done.json").exists() and (jd / "ckpt.pt").exists()
    capsys.readouterr()
    train.main([*base, "--k-pg", "50"])  # resumes the forced run, then finishes
    assert "resume at step 20" in capsys.readouterr().out
    assert json.loads((jd / "done.json").read_text())["config"]["k"] == 50


# ---------------------------------------------------------------------------- 7. smoke verdict
GOOD_HOOK = {
    "a_actions_equal_with_hooks": True,
    "a_actions_equal_with_full_capture": True,
    "b_layer_out_equals_next_in": True,
    "b_emb_is_embed_prefix_output": True,
    "b_ae_recorder_max_abs_err": 0.0,
    "ae_steps_seen": 10,
    "c_empty_camera_tokens_masked": True,
    "c_group_mean_max_abs_err": 0.0,
    "d_weighted_mean_rel_err": 0.0,
    "e_pg05_project_out_max_component": 0.1,
    "e_pg05_bf16_component_scale": 11.7,
    "e_pg05_add_shift_err": 0.4,
    "e_ae05_project_out_max_component": 0.05,
    "e_ae05_bf16_component_scale": 0.8,
    "e_ae05_add_shift_err": 0.24,
}


def test_hook_checks_cover_every_subcheck():
    assert all(smoke.hook_checks(GOOD_HOOK).values())
    for key, bad in (
        ("a_actions_equal_with_full_capture", False),
        ("b_ae_recorder_max_abs_err", 0.5),
        ("e_pg05_add_shift_err", 100.0),
        ("e_ae05_project_out_max_component", 10.0),
        ("ae_steps_seen", 0),
    ):
        assert not all(smoke.hook_checks({**GOOD_HOOK, key: bad}).values()), key
    assert not all(smoke.hook_checks({}).values())


def test_summary_fails_on_step_exit_code_or_hook_verdict(tmp_path, capsys):
    hook = {**GOOD_HOOK, "checks": smoke.hook_checks(GOOD_HOOK), "pass": True}
    (tmp_path / "hook_check.json").write_text(json.dumps(hook))
    (tmp_path / "sae_bench.json").write_text(json.dumps({"pg": {"hours": 0.1}, "ae": {"hours": 1.0}}))
    names = ["hook_check", "collect", "collect_resume", "train_part1", "train_resume", "sae_bench"]
    (tmp_path / "steps.tsv").write_text("".join(f"{n}\t0\n" for n in names))
    smoke.summarize(tmp_path, "pi05lib")
    res = json.loads((tmp_path / "summary.json").read_text())
    assert all(res["checks"][f"step_{n}_exit0"] for n in names)
    assert all(v for k, v in res["checks"].items() if k.startswith("hook_"))
    # (no collection / SAE outputs here, so overall FAIL; each failure mode below must flip its own check)
    (tmp_path / "steps.tsv").write_text("".join(f"{n}\t{1 if n == 'train_resume' else 0}\n" for n in names))
    smoke.summarize(tmp_path, "pi05lib")
    assert not json.loads((tmp_path / "summary.json").read_text())["checks"]["step_train_resume_exit0"]
    (tmp_path / "hook_check.json").write_text(json.dumps({**hook, "pass": False}))
    smoke.summarize(tmp_path, "pi05lib")
    assert not json.loads((tmp_path / "summary.json").read_text())["checks"]["hook_check_pass"]
    (tmp_path / "steps.tsv").unlink()
    (tmp_path / "sae_bench.json").unlink()
    smoke.summarize(tmp_path, "pi05lib")
    c = json.loads((tmp_path / "summary.json").read_text())["checks"]
    assert not c["steps_recorded"] and not c["sae_bench_ok"]


# ---------------------------------------------------------------------------- 8. stats from fp32 codes
def test_stats_use_fp32_codes():
    sae = TopKSAE(SAEConfig(d=16, n_latents=16, k=2), seed=0)
    _, val = encode_all(sae, torch.randn(10, 16))
    assert val.dtype == np.float32
    # 0.100003 > τ_on but its fp16 rounding (0.099976) is not: the onset must be counted from the fp32 value
    v = np.float32(0.100003)
    assert np.float16(v) <= 0.1 < v
    em = episode_metrics(
        np.zeros((2, 1), np.int16), np.array([[0.0], [v]], np.float32), np.zeros(2, int), n=1
    )
    assert em["onsets"][0, 0] == 1
