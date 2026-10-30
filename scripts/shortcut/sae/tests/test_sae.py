"""CPU tests for the T015 SAE pipeline: uv run pytest scripts/shortcut/sae/tests -q

GPU checks on the real π0.5 (hook location, actions unchanged, group pooling) are in `smoke.py`.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

SAE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAE_DIR))

import store  # noqa: E402
from collect import PartWriter, parse_ae_steps, parse_episodes  # noqa: E402
from hooks import (  # noqa: E402
    Intervention,
    Steering,
    add,
    all_layers,
    parse_layers,
    project_out,
    replace_with,
)
from model import SAEConfig, TopKSAE, geometric_median  # noqa: E402
from stats import episode_metrics, feature_table  # noqa: E402


# ---------------------------------------------------------------------------- names / splitting
def test_parse_layers():
    assert parse_layers("all") == all_layers() and len(all_layers()) == 37
    assert parse_layers("pg:0-2,ae:5") == ["pg00", "pg01", "pg02", "ae05"]
    assert parse_layers("emb,pg5,ae:0,17") == ["emb", "pg05", "ae00", "ae17"]
    with pytest.raises(ValueError):
        parse_layers("pg:18")


def test_parse_steps_episodes():
    assert parse_ae_steps("all") == list(range(10))
    assert parse_ae_steps("first,last") == [0, 9]
    assert parse_ae_steps("5,0") == [0, 5]
    assert parse_episodes("3,1-2") == [1, 2, 3]


def test_parts_and_shards():
    rng = np.random.default_rng(0)
    eps = list(range(1693))
    lens = rng.integers(60, 500, size=len(eps)).tolist()
    parts = store.split_parts(eps, lens, 64)
    assert len(parts) == 64 and sum(parts, []) == eps
    sizes = [sum(lens[e] for e in p) for p in parts]
    assert max(sizes) < 1.5 * np.mean(sizes)
    for n_shards in (1, 3, 8):
        got = sorted(sum((store.shard_parts(64, i, n_shards) for i in range(n_shards)), []))
        assert got == list(range(64))
    assert store.part_id([3, 4, 5]) == "ep00003-00005"
    assert store.part_id([3, 7]).startswith("ep00003-00007_")


def test_pool_pg_weighted_mean():
    rng = np.random.default_rng(1)
    tokens = [rng.normal(size=(n, 8)) for n in (256, 256, 37)]
    arr = np.stack([t.mean(0) for t in tokens])[None].astype(np.float16)
    counts = np.array([[256, 256, 37]])
    direct = np.concatenate(tokens).mean(0)
    assert np.allclose(store.pool_pg(arr, counts, "pooled_all")[0], direct, atol=2e-3)
    assert np.allclose(store.pool_pg(arr, counts, "text")[0], tokens[2].mean(0), atol=2e-3)
    assert np.allclose(store.pool_pg(arr, counts, "img")[0], np.concatenate(tokens[:2]).mean(0), atol=2e-3)


def test_part_writer_fp16_overflow(tmp_path, capsys):
    w = PartWriter(tmp_path, ["pg00", "ae00"], n_rows=4, n_steps=2)
    w.write("pg00", slice(0, 2), np.ones((2, 3, 2048), np.float32))
    big = np.full((2, 3, 2048), 1e5, np.float32)
    w.write("pg00", slice(2, 4), big)
    w.write("ae00", slice(0, 4), np.ones((4, 2, 1024), np.float32))
    dtypes = w.close()
    assert dtypes == {"pg00": "float32", "ae00": "float16"} and w.overflow == ["pg00"]
    a = np.load(tmp_path / "pg00.npy")
    assert a.dtype == np.float32 and a[0, 0, 0] == 1 and a[3, 0, 0] == 1e5
    assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------- SAE
def test_sae_init_and_constraints():
    sae = TopKSAE(SAEConfig(d=32, n_latents=32, k=4), seed=0)
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(32), atol=1e-5)
    assert torch.allclose(sae.W_enc, sae.W_dec.T * (4 / 32) ** 0.5)
    x = torch.randn(64, 32) * 5 + 3
    xt, mu, nrm = sae.normalize(x)
    assert torch.allclose(xt.norm(dim=-1), torch.ones(64), atol=1e-5)
    assert torch.allclose(xt.mean(-1), torch.zeros(64), atol=1e-6)
    vals, idx = sae.encode(x)
    assert vals.shape == (64, 4) and (vals >= 0).all()
    # tangent projection: gradient orthogonal to each column
    sae.W_dec.grad = torch.randn_like(sae.W_dec)
    sae.project_decoder_grad()
    assert (sae.W_dec.grad * sae.W_dec).sum(0).abs().max() < 1e-5


def test_geometric_median_robust():
    x = torch.zeros(101, 3)
    x[:50, 0] = 1.0
    x[100] = 1000.0
    gm = geometric_median(x)
    assert gm.norm() < 1.5  # the mean would be pulled to ~10


def _sparse_data(n=4096, d=48, n_feat=96, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(d, n_feat, generator=g)
    dirs /= dirs.norm(dim=0)
    codes = torch.zeros(n, n_feat)
    idx = torch.randint(0, n_feat, (n, k), generator=g)
    codes.scatter_(1, idx, torch.rand(n, k, generator=g) + 0.5)
    return codes @ dirs.T + 2.0


def test_sae_loss_decreases_and_aux_runs():
    torch.manual_seed(0)
    x = _sparse_data()
    sae = TopKSAE(SAEConfig(d=48, n_latents=96, k=6, k_aux=16, dead_steps=5), seed=0)
    sae.init_from_data(x[:1000])
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    last = torch.zeros(96, dtype=torch.long)
    first = None
    for step in range(300):
        xb = x[torch.randint(0, len(x), (256,))]
        loss, parts, fired = sae.loss(xb, (step - last) >= 5)
        opt.zero_grad()
        loss.backward()
        sae.project_decoder_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae.renorm_decoder()
        last[fired] = step + 1
        first = first if first is not None else float(parts["mse"])
    m = sae.metrics(x)
    assert m["nmse"] < 0.5 * first and m["var_explained"] > 0.5
    # AuxK: with latents marked dead the aux term is > 0 and its gradient reaches the dead encoder rows
    dead = torch.zeros(96, dtype=torch.bool)
    dead[:40] = True
    sae.zero_grad()
    loss, parts, _ = sae.loss(x[:256], dead)
    assert float(parts["aux"]) > 0
    loss.backward()
    assert sae.W_enc.grad[:40].abs().sum() > 0
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(96), atol=1e-4)


def test_sae_save_load(tmp_path):
    sae = TopKSAE(SAEConfig(d=16, n_latents=16, k=2, c_mse=0.7), seed=3)
    sae.save(tmp_path, extra={"layer": "pg05"})
    back = TopKSAE.load(tmp_path)
    x = torch.randn(5, 16)
    assert torch.equal(sae.reconstruct(x), back.reconstruct(x)) and back.cfg.c_mse == 0.7
    assert json.loads((tmp_path / "config.json").read_text())["layer"] == "pg05"


# ---------------------------------------------------------------------------- hooks on a toy π0.5 layout
class ToyLayer(nn.Module):
    def __init__(self, d, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(d, d)

    def forward(self, hidden_states, **kw):
        return hidden_states + torch.tanh(self.lin(hidden_states))


def toy_policy(d_pg=16, d_ae=8, n=18):
    pg = nn.ModuleList([ToyLayer(d_pg, i) for i in range(n)])
    ae = nn.ModuleList([ToyLayer(d_ae, 100 + i) for i in range(n)])
    lm = SimpleNamespace(layers=pg)
    expert = nn.Module()
    expert.layers = ae
    pwe = SimpleNamespace(
        paligemma=SimpleNamespace(model=SimpleNamespace(language_model=lm)),
        gemma_expert=SimpleNamespace(model=expert),
    )
    return SimpleNamespace(model=SimpleNamespace(paligemma_with_expert=pwe)), pg, ae


def run_pg(pg, x):
    for layer in pg:
        x = layer(x)
    return x


def test_project_out_and_add():
    policy, pg, _ = toy_policy()
    x = torch.randn(2, 10, 16)
    v = torch.randn(16)
    got = []
    with project_out(policy, "pg05", v):
        h = pg[5].register_forward_hook(lambda m, a, o: got.append(o))
        run_pg(pg, x)
        h.remove()
    assert (got[0] @ (v / v.norm())).abs().max() < 1e-5
    ref = []
    h = pg[5].register_forward_hook(lambda m, a, o: ref.append(o))
    base = run_pg(pg, x)
    h.remove()
    got = []
    with add(policy, "pg05", v, 3.0):
        h = pg[5].register_forward_hook(lambda m, a, o: got.append(o))
        steered = run_pg(pg, x)
        h.remove()
    assert torch.allclose(got[0] - ref[0], 3.0 * v.expand_as(ref[0]), atol=1e-5)
    assert not torch.allclose(steered, base)
    assert torch.equal(run_pg(pg, x), base)  # removed on exit


def test_emb_intervention_changes_layer0_input():
    policy, pg, _ = toy_policy()
    x = torch.randn(1, 4, 16)
    v = torch.randn(16)
    out_steered = pg[0](x + 2.0 * v)
    with Steering(policy, [Intervention("emb", "add", vec=v, alpha=2.0)]):
        assert torch.allclose(pg[0](x), out_steered)
        assert torch.allclose(pg[0](hidden_states=x), out_steered)


def test_replace_with_matches_sae_metrics():
    torch.manual_seed(0)
    policy, pg, ae = toy_policy()
    sae = TopKSAE(SAEConfig(d=8, n_latents=8, k=2, c_mse=0.9), seed=0)
    y_seen, got = [], []
    x = torch.randn(3, 50, 8)
    with replace_with(policy, "ae03", sae) as st:
        h = ae[3].register_forward_hook(lambda m, a, o: got.append(o))
        h0 = ae[3].register_forward_pre_hook(lambda m, a: y_seen.append(a[0]))
        for layer in ae:
            x = layer(x)
        h.remove()
        h0.remove()
    iv = st.interventions[0]
    y = ae[3](y_seen[0])  # the unmodified layer output
    assert torch.allclose(got[0], sae.reconstruct(y), atol=1e-6)
    want = sae.metrics(y)
    assert iv.stats[0]["nmse"] == pytest.approx(want["nmse"], rel=1e-5)
    assert iv.stats[0]["var_explained"] == pytest.approx(want["var_explained"], rel=1e-5)
    # mean mode: token deviations kept, the token-mean replaced by its reconstruction
    iv2 = Intervention("ae03", "replace", sae=sae, mode="mean")
    out = iv2.apply(y)
    m = y.mean(1, keepdim=True)
    assert torch.allclose(out - y, sae.reconstruct(m) - m, atol=1e-5)


# ---------------------------------------------------------------------------- stats
def test_episode_metrics_hysteresis():
    # one feature, one episode: f = 0, .2(no onset: below tau from 0), .5(onset), .05(stay), 0(off), .3(onset), .3
    f = np.array([0, 0.05, 0.5, 0.05, 0, 0.3, 0.3], np.float32)
    idx = np.zeros((len(f), 1), np.int16)
    val = f[:, None].astype(np.float16)
    em = episode_metrics(idx, val, np.zeros(len(f), int), n=2)
    assert em["onsets"][0, 0] == 2
    assert em["active_steps"][0, 0] == 4  # t=2,3 and 5,6
    assert em["max"][0, 0] == pytest.approx(0.5, abs=1e-3)
    ft = feature_table(em)
    assert ft.loc[0, "ep_coverage"] == 1 and ft.loc[0, "mean_onsets"] == 2
    assert ft.loc[0, "rel_run_length"] == pytest.approx((4 / 2) / 7)
    assert ft.loc[1, "ep_coverage"] == 0 and np.isnan(ft.loc[1, "mean_onsets"])


# ---------------------------------------------------------------------------- end to end on fake activations
def fake_acts(root: Path, n_eps=6, length=40, steps=(0, 9)):
    rng = np.random.default_rng(0)
    acts = root / "acts"
    eps = list(range(n_eps))
    parts = store.split_parts(eps, [length] * n_eps, 2)
    identity = {"model": "toy", "ae_steps": list(steps), "layers": ["pg05", "ae05"]}
    plan = {
        "identity": identity,
        "parts": [{"part_id": store.part_id(p), "episodes": p, "n_frames": length * len(p)} for p in parts],
    }
    acts.mkdir(parents=True)
    store.write_json_atomic(acts / "plan.json", plan)
    dirs = torch.randn(2048, 64)
    for p in parts:
        d = acts / "parts" / store.part_id(p)
        d.mkdir(parents=True)
        n = length * len(p)
        codes = (rng.random((n, 3, 64)) < 0.05) * rng.random((n, 3, 64))
        np.save(d / "pg05.npy", (codes @ dirs.T.numpy() + 1.0).astype(np.float16))
        np.save(d / "ae05.npy", rng.normal(size=(n, len(steps), 1024)).astype(np.float16))
        np.save(d / "counts.npy", np.tile([256, 256, 30], (n, 1)).astype(np.int32))
        meta = pd.DataFrame(
            {
                "row": range(n),
                "index": range(n),
                "episode_index": np.repeat(p, length),
                "frame_index": np.tile(range(length), len(p)),
                "task_index": np.repeat(p, length) % 3,
                "suite": np.where(np.repeat(p, length) < 3, "libero_goal", "libero_10"),
                "instruction": "x",
            }
        )
        meta.to_parquet(d / "meta.parquet")
        store.write_json_atomic(d / "done.json", {"identity": store.identity_hash(identity), "rows": n})
    return acts


def test_train_resume_and_stats(tmp_path):
    import train

    acts = fake_acts(tmp_path)
    common = [
        "--model",
        "toy",
        "--acts-dir",
        str(acts),
        "--sae-dir",
        str(tmp_path / "sae"),
        "--gpus",
        "cpu",
        "--batch",
        "32",
        "--steps",
        "120",
        "--log-every",
        "10",
        "--ckpt-every",
        "20",
        "--wandb",
        "off",
        "--gm-samples",
        "200",
    ]
    train.main([*common, "--layers", "pg:5", "--stop-after", "40"])
    jd = tmp_path / "sae" / "pooled_all" / "pg05" / "seed0"
    assert (jd / "ckpt.pt").exists() and not (jd / "done.json").exists()
    train.main([*common, "--layers", "pg:5,ae:5"])
    log = [json.loads(x) for x in (jd / "log.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in log]
    assert steps == sorted(steps) and steps[0] == 10 and steps[-1] == 120 and 50 in steps
    assert log[-1]["nmse"] < log[0]["nmse"]
    m = json.loads((jd / "metrics.json").read_text())
    assert m["steps"] == 120 and 0 < m["var_explained"] <= 1
    ft = pd.read_parquet(jd / "features.parquet")
    assert len(ft) == 2048 and {"ep_coverage", "mean_onsets", "mean_max_act", "rel_run_length"} <= set(
        ft.columns
    )
    groups = pd.read_parquet(jd / "fire_by_group.parquet")
    assert set(groups["group_type"]) == {"suite", "task"}
    assert np.load(jd / "codes_idx.npy").shape == (240, 100)
    ja = tmp_path / "sae" / "ae-steps-all" / "ae05" / "seed0"
    fa = pd.read_parquet(ja / "features.parquet")
    assert sorted(fa["ae_step"].unique()) == [0, 9] and len(fa) == 2 * 1024
    assert np.load(ja / "codes_idx.npy").shape == (480, 64)
    # finished jobs are skipped
    before = (jd / "done.json").stat().st_mtime
    train.main([*common, "--layers", "pg:5"])
    assert (jd / "done.json").stat().st_mtime == before
