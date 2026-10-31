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
from test_collect_mock import mock_loader  # noqa: E402,F401  (fixture)


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


def test_plan_identity_has_dtype_and_revisions():
    import argparse

    args = argparse.Namespace(
        model="pi05base", episodes="379", layers="pg:5", ae_steps="all", parts=1, noise_seed=0
    )
    plan = collect.make_plan(args, Path(collect.DEFAULT_OUT_ROOT))
    ident = plan["identity"]
    for k in ("ckpt_revision", "adapter_revision"):
        assert ident[k].startswith("hub:") and len(ident[k]) == 4 + 40
    assert "bfloat16" in ident["dtype"]
    # re-review 3: no locations in the identity (hub id / mount path live under "source")
    assert not {"ckpt", "adapter", "dataset_root", "acts_dir"} & set(ident)
    assert plan["source"]["ckpt"] == "lerobot/pi05_base"


# ---------------------------------------------------------------------------- re-review 1: pinned loading
def _ckpt(dirpath: Path, payload: bytes) -> Path:
    dirpath.mkdir(parents=True)
    (dirpath / "config.json").write_text("{}")
    (dirpath / "model.safetensors").write_bytes(payload)
    return dirpath


def test_load_uses_pinned_local_ckpt_and_reverifies(tmp_path, monkeypatch):
    import common

    ck = _ckpt(tmp_path / "ck", b"w1")
    rev = collect.checkpoint_fingerprint.__wrapped__(str(ck))
    ident = {"ckpt_revision": rev, "adapter_revision": None}
    seen = {}

    def fake_load(ptype, path, *a, **k):
        seen["path"] = path
        return "pol", "pre", "post", None, None, None, {}

    monkeypatch.setattr(common, "load_policy", fake_load)
    collect.load_for_collect(ident, "cpu", {"ckpt": str(ck), "adapter": None})
    assert seen["path"] == str(ck)

    # the weights change between planning and loading (or during the load) → refused
    def swap_then_load(*a, **k):
        (ck / "model.safetensors").write_bytes(b"w2")
        return fake_load(*a, **k)

    monkeypatch.setattr(common, "load_policy", swap_then_load)
    with pytest.raises(SystemExit, match="not mixing weights"):
        collect.load_for_collect(ident, "cpu", {"ckpt": str(ck), "adapter": None})


def test_hub_ckpt_loaded_at_recorded_commit(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake_snapshot(repo, revision=None, **k):
        calls.append((repo, revision))
        d = tmp_path / "snapshots" / revision
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    sha = "a" * 40
    path = collect.pinned_path("org/model", "hub:" + sha)
    assert calls == [("org/model", sha)] and Path(path).name == sha
    collect.verify_pinned(path, "hub:" + sha)
    with pytest.raises(SystemExit, match="not mixing weights"):
        collect.verify_pinned(str(tmp_path / "snapshots" / ("b" * 40)), "hub:" + sha)


# ---------------------------------------------------------------------------- re-review 3: paths not compared
def test_same_ckpt_at_two_mount_paths_same_identity(tmp_path):
    a = _ckpt(tmp_path / "mntA" / "ck", b"weights")
    b = _ckpt(tmp_path / "mntB" / "x" / "ck", b"weights")
    fp = collect.checkpoint_fingerprint.__wrapped__
    assert fp(str(a)) == fp(str(b))
    ident = {"model": "m", "ckpt_revision": fp(str(a))}
    path = tmp_path / "acts" / "plan.json"
    assert store.create_or_check_plan(path, {"identity": ident, "source": {"ckpt": str(a)}, "parts": []})
    # another pod, other mount path, same content: accepted
    assert not store.create_or_check_plan(path, {"identity": ident, "source": {"ckpt": str(b)}, "parts": []})


def test_train_config_independent_of_acts_mount_path(tmp_path):
    import argparse
    import shutil

    import train
    from test_sae import fake_acts

    acts = fake_acts(tmp_path / "podA")
    other = tmp_path / "podB" / "mnt" / "acts"
    shutil.copytree(acts, other)
    common = {
        "input": "pooled_all",
        "ae_steps": "all",
        "k_pg": 100,
        "k_ae": 64,
        "k_aux": 512,
        "aux_coef": 1 / 32,
        "dead_steps": 500,
        "lr": 1e-4,
        "batch": 4096,
        "epochs": 100,
        "steps": 0,
        "grad_clip": 1.0,
        "gm_samples": 10000,
        "expansion": 1.0,
        "sae_dir": str(tmp_path / "sae"),
        "force": False,
    }
    ca = train.job_config(argparse.Namespace(acts_dir=str(acts), **common), "pg05", 0)
    cb = train.job_config(argparse.Namespace(acts_dir=str(other), **common), "pg05", 0)
    assert ca == cb and "acts_dir" not in ca


# ---------------------------------------------------------------------------- 3. per-writer tmp, lease, publish
def test_lease_fresh_blocks_stale_taken_over(tmp_path):
    a, b = store.Lease(tmp_path, "ep1-2"), store.Lease(tmp_path, "ep1-2")
    assert a.acquire()
    assert not b.acquire()  # another writer, fresh
    old = time.time() - 2 * store.LEASE_TTL_S
    os.utime(store.lease_path(tmp_path, "ep1-2"), (old, old))
    assert b.acquire()  # stale → taken over
    b.release()
    assert not store.lease_path(tmp_path, "ep1-2").exists()


# ---------------------------------------------------------------------------- re-review 4: lease owner token
def test_lease_owner_token(tmp_path):
    lp = store.lease_path(tmp_path, "ep1-2")
    a, b = store.Lease(tmp_path, "ep1-2"), store.Lease(tmp_path, "ep1-2")
    assert a.acquire() and a.refresh() and a.owned()
    old = time.time() - 2 * store.LEASE_TTL_S
    os.utime(lp, (old, old))  # a stalls past the TTL
    assert b.acquire()  # b takes the part over
    assert not a.refresh() and a.lost.is_set()
    with pytest.raises(store.LeaseLostError):
        a.check()
    before = lp.read_text()
    a.release()  # not ours any more: b's lease is left alone
    assert lp.read_text() == before and b.owned()
    b.release()
    assert not lp.exists()
    # the lease file vanishing between check and touch is "lost", not an exception
    c = store.Lease(tmp_path, "ep1-2")
    assert c.acquire()
    lp.unlink()
    assert not c.refresh()


def test_lease_heartbeat_survives_errors_then_marks_lost(tmp_path, monkeypatch, capsys):
    a = store.Lease(tmp_path, "ep1-2", ttl=0.4)
    assert a.acquire()
    fails = {"n": 2}
    real = store.Lease.refresh

    def flaky(self):
        if fails["n"] > 0:  # two transient errors, then fine again
            fails["n"] -= 1
            raise OSError("ESTALE")
        return real(self)

    monkeypatch.setattr(store.Lease, "refresh", flaky)
    a.start_heartbeat(every=0.05)
    time.sleep(0.6)
    assert a._thread is not None and a._thread.is_alive() and not a.lost.is_set()  # recovered
    assert "heartbeat failed" in capsys.readouterr().out
    a.release()

    b = store.Lease(tmp_path, "ep3-4", ttl=0.3)
    assert b.acquire()
    monkeypatch.setattr(store.Lease, "refresh", lambda self: (_ for _ in ()).throw(OSError("EIO")))
    b.start_heartbeat(every=0.05)
    time.sleep(0.8)  # failing for longer than the TTL
    assert b.lost.is_set() and not b._thread.is_alive()
    monkeypatch.setattr(store.Lease, "refresh", real)
    with pytest.raises(store.LeaseLostError):
        b.check()
    b.release()


def test_lease_heartbeat_keeps_it_fresh(tmp_path):
    a = store.Lease(tmp_path, "ep1-2", ttl=0.5)
    assert a.acquire()
    a.start_heartbeat(every=0.05)
    time.sleep(1.0)  # well past the TTL with no main-loop refresh (e.g. a long data fetch)
    assert not store.Lease(tmp_path, "ep1-2", ttl=0.5).acquire()
    a.release()
    assert store.Lease(tmp_path, "ep1-2", ttl=0.5).acquire()


def test_publish_keeps_first_writer(tmp_path):
    final = tmp_path / "ep1-2"
    a, b = tmp_path / ".ep1-2.tmp-A", tmp_path / ".ep1-2.tmp-B"
    for d, tag in ((a, "A"), (b, "B")):
        d.mkdir()
        (d / "done.json").write_text(tag)
    assert store.publish_part(a, final) and (final / "done.json").read_text() == "A"
    assert not store.publish_part(b, final)
    assert (final / "done.json").read_text() == "A" and not b.exists()
    # re-review 2: a final dir without done.json (never made by this code) → clear error, nothing moved
    junk = tmp_path / "ep3-4"
    junk.mkdir()
    (junk / "x").write_text("junk")
    c = tmp_path / ".ep3-4.tmp-C"
    c.mkdir()
    (c / "done.json").write_text("C")
    with pytest.raises(SystemExit, match=r"ep3-4 exists without done.json"):
        store.publish_part(c, junk)
    assert sorted(x.name for x in junk.iterdir()) == ["x"] and (c / "done.json").exists()


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


OUT_ROOT = Path(collect.DEFAULT_OUT_ROOT)  # data lookup as in collect.py (SC_LIBERO_DATA_ROOT / SC_OUT_ROOT)
needs_data = pytest.mark.skipif(
    collect.local_data_root(OUT_ROOT) is None,
    reason="no local LIBERO copy ($SC_LIBERO_DATA_ROOT or $SC_OUT_ROOT/data/hf_libero)",
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
    monkeypatch.setattr(collect, "load_for_collect", lambda i, d, src: (pol, mock_pre, lambda a: a, {}))
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
    monkeypatch.setattr(collect, "load_for_collect", lambda i, d, src: (pol, mock_pre, lambda a: a, {}))
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


@needs_data
def test_worker_child_uses_parent_plan_and_lease_loss_drops_part(tmp_path, monkeypatch, capsys):
    """re-review 1: a --worker-parts child reads the verified plan.json (no make_plan). re-review 4: a writer
    whose lease is taken over mid-part drops its copy and leaves the new owner's lease alone."""
    from test_collect_mock import MockPolicy, mock_pre

    pol = MockPolicy()
    monkeypatch.setattr(collect, "load_for_collect", lambda i, d, src: (pol, mock_pre, lambda a: a, {}))
    argv = ["--model", "pi05lib", "--out-root", str(tmp_path), "--episodes", "379", "--parts", "1"]
    argv += ["--gpus", "cpu", "--workers", "0", "--batch", "16", "--layers", "pg:5"]
    collect.main(argv)
    acts = tmp_path / "sae" / "pi05lib" / "acts"
    pdir = acts / "parts"
    import shutil

    shutil.rmtree(pdir / "ep00379-00379")

    def no_plan(*a, **k):
        raise AssertionError("child must not recompute the plan")

    monkeypatch.setattr(collect, "make_plan", no_plan)
    orig = pol.predict_action_chunk

    def steal(obs, noise):  # another pod takes the lease over during the second batch
        if obs["frame_index"][0] >= 16:
            (pdir / "ep00379-00379.lease").write_text(json.dumps({"token": "other-pod"}))
        return orig(obs, noise)

    pol.predict_action_chunk = steal
    capsys.readouterr()
    collect.main([*argv, "--acts-dir", str(acts), "--worker-parts", "0"])
    log = capsys.readouterr().out
    assert "lease taken over by another writer" in log and "dropping this copy" in log
    assert not (pdir / "ep00379-00379").exists()
    assert json.loads((pdir / "ep00379-00379.lease").read_text())["token"] == "other-pod"
    assert not list(pdir.glob(".ep00379-00379.tmp-*"))


# ---------------------------------------------------------------------------- server smoke 2026-10-31
@needs_data
def test_collect_done_json_with_torch_dtype_meta(tmp_path, mock_loader):  # noqa: F811 (fixture)
    """GPU path: load_policy's meta has dtype=torch.bfloat16 (an object, not a str) → done.json must still be
    written (it raised TypeError: Object of type dtype is not JSON serializable)."""
    argv = ["--model", "pi05lib", "--out-root", str(tmp_path), "--episodes", "379", "--parts", "1"]
    argv += ["--gpus", "cpu", "--workers", "0", "--batch", "32", "--layers", "pg:5"]
    collect.main(argv)
    (part,) = store.done_parts(tmp_path / "sae" / "pi05lib" / "acts")
    assert part["inference"] == {"num_inference_steps": 10, "chunk_size": 5, "dtype": "torch.bfloat16"}


def test_json_default_handles_torch_numpy_path(tmp_path):
    obj = {
        "dtype": torch.bfloat16,
        "dev": torch.device("cpu"),
        "p": tmp_path,
        "n": np.float32(0.5),
        "i": np.int64(3),
        "a": np.arange(3),
        "t": torch.tensor([1.0, 2.0]),
    }
    back = json.loads(store.dumps(obj))
    assert back == {
        "dtype": "torch.bfloat16",
        "dev": "cpu",
        "p": str(tmp_path),
        "n": 0.5,
        "i": 3,
        "a": [0, 1, 2],
        "t": [1.0, 2.0],
    }
    store.write_json_atomic(tmp_path / "x.json", obj)
    assert json.loads((tmp_path / "x.json").read_text())["dtype"] == "torch.bfloat16"
    sae = TopKSAE(SAEConfig(d=4, n_latents=4, k=1), seed=0)
    sae.save(tmp_path / "sae", extra={"dtype": torch.bfloat16, "acts_dir": tmp_path})
    assert json.loads((tmp_path / "sae" / "config.json").read_text())["dtype"] == "torch.bfloat16"


def test_fingerprint_offline_reads_cache_refs(tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    ref = tmp_path / "models--org--model" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("c" * 40 + "\n")
    assert collect.checkpoint_fingerprint.__wrapped__("org/model") == "hub:" + "c" * 40
    with pytest.raises(SystemExit, match="HF_HUB_OFFLINE=1 and no cached revision for org/other"):
        collect.checkpoint_fingerprint.__wrapped__("org/other")
