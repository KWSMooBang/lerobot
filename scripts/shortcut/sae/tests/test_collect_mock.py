"""CPU test of collect.py → store → train.py → stats.py with a mock π0.5 on the real LIBERO frames.

The mock has the module layout and call pattern `hooks.py` relies on (`embed_prefix` / `_embed_images`, prefix
once through `paligemma…language_model`, 10 calls of `gemma_expert.model`), with small image/text token
counts, so recording, token groups, part writing, shard split, resume, the reader and SAE training run
without a GPU. The real-model checks are `smoke.py` (server GPU, or `--device cpu --frames 2`).

    uv run pytest scripts/shortcut/sae/tests/test_collect_mock.py -q
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

import collect  # noqa: E402
import store  # noqa: E402
from hooks import ActivationRecorder  # noqa: E402

DATA_ROOT = Path("/outputs/VLA-shortcut-exps/data/hf_libero")
pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "meta" / "info.json").exists(), reason="no local LIBERO copy"
)
EPS = "379,807,1261"  # goal, object, spatial (365 frames)
IMG_TOK, TXT_LEN = 4, 12
D_PG, D_AE = 2048, 1024


class Layer(nn.Module):
    def __init__(self, d: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(torch.randn(d, 8, generator=g) / d**0.5)

    def forward(self, hidden_states, **kw):
        return hidden_states + torch.tanh(hidden_states @ self.w) @ self.w.T


class Stack(nn.Module):
    def __init__(self, d: int, seed: int):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d, seed + i) for i in range(18)])

    def forward(self, inputs_embeds, **kw):
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, attention_mask=None)
        return h


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_proj = nn.Linear(3 * 16, D_PG)
        self.tok_emb = nn.Embedding(64, D_PG)
        self.lm = Stack(D_PG, 0)
        self.expert = Stack(D_AE, 100)
        self.act_in = nn.Linear(32, D_AE)
        self.ctx = nn.Linear(D_PG, D_AE)
        self.act_out = nn.Linear(D_AE, 32)
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(model=SimpleNamespace(language_model=self.lm)),
            gemma_expert=SimpleNamespace(model=self.expert),
        )

    def _embed_images(self, images, img_masks):
        out = []
        for img in images:  # (B, 3, H, W) → 4 tokens from 2×2 pooled quadrants of a 4×4 grid
            g = nn.functional.adaptive_avg_pool2d(img, 4)  # (B, 3, 4, 4)
            q = g.unfold(2, 2, 2).unfold(3, 2, 2).reshape(img.shape[0], 3, IMG_TOK, 4)
            out.append(
                self.img_proj(q.permute(0, 2, 1, 3).reshape(img.shape[0], IMG_TOK, 12).repeat(1, 1, 4))
            )
        return out

    def embed_prefix(self, images, img_masks, tokens, masks, states=None, state_masks=None):
        embs = self._embed_images(images, img_masks)
        pads = [m[:, None].expand(-1, IMG_TOK) for m in img_masks]
        embs.append(self.tok_emb(tokens))
        pads.append(masks)
        return torch.cat(embs, 1), torch.cat(pads, 1), None


class MockPolicy:
    def __init__(self):
        self.model = MockModel()
        self.config = SimpleNamespace(chunk_size=5, max_action_dim=32, num_inference_steps=10)

    @torch.no_grad()
    def predict_action_chunk(self, obs, noise):
        imgs = [obs["observation.images.image"], obs["observation.images.image2"]]
        imgs.append(torch.full_like(imgs[0], -1.0))  # empty third camera, masked
        b = imgs[0].shape[0]
        masks = [
            torch.ones(b, dtype=torch.bool),
            torch.ones(b, dtype=torch.bool),
            torch.zeros(b, dtype=torch.bool),
        ]
        tok, tmask = obs["observation.language.tokens"], obs["observation.language.attention_mask"]
        embs, pad, _ = self.model.embed_prefix(imgs, masks, tok, tmask)
        h = self.model.lm.forward(embs)  # lerobot calls .forward() directly (no model-level hooks)
        ctx = (h * pad[..., None]).sum(1) / pad.sum(1, keepdim=True)
        x = noise
        for s in range(10):
            e = self.model.expert.forward(self.model.act_in(x) + self.model.ctx(ctx)[:, None] * (1 + s / 10))
            x = x - 0.1 * self.model.act_out(e)
        return x[:, :, :7]


def mock_pre(batch):
    lens = torch.tensor([min(len(t.split()), TXT_LEN) for t in batch["task"]])
    tokens = torch.zeros(len(lens), TXT_LEN, dtype=torch.long)
    mask = torch.arange(TXT_LEN)[None] < lens[:, None]
    for i, t in enumerate(batch["task"]):
        ids = [hash(w) % 63 + 1 for w in t.split()][:TXT_LEN]
        tokens[i, : len(ids)] = torch.tensor(ids)
    return {
        **batch,
        "observation.language.tokens": tokens,
        "observation.language.attention_mask": mask,
    }


@pytest.fixture
def mock_loader(monkeypatch):
    pol = MockPolicy()
    monkeypatch.setattr(
        collect,
        "load_for_collect",
        lambda ident, device: (pol, mock_pre, lambda a: a, {"num_inference_steps": 10}),
    )
    return pol


def test_collect_shards_resume_train(tmp_path, mock_loader, capsys):
    out = tmp_path / "out"
    base = [
        "--model",
        "pi05lib",
        "--out-root",
        str(out),
        "--episodes",
        EPS,
        "--parts",
        "3",
        "--gpus",
        "cpu",
        "--workers",
        "0",
        "--batch",
        "16",
        "--layers",
        "emb,pg:0,17,ae:0,17",
    ]
    collect.main([*base, "--shard", "0/2"])
    acts = out / "sae" / "pi05lib" / "acts"
    plan = store.read_plan(acts)
    assert [p["episodes"] for p in plan["parts"]] == [[379], [807], [1261]]
    assert len(store.done_parts(acts)) == 2  # shard 0/2 = parts 0, 1
    collect.main([*base, "--shard", "1/2"])
    parts = store.done_parts(acts)
    assert len(parts) == 3 and sum(p["rows"] for p in parts) == 365
    # resume: finished parts are skipped, a part without done.json (interrupted) is redone
    d1 = Path(parts[1]["dir"])
    (d1 / "done.json").unlink()
    keep = (Path(parts[0]["dir"]) / "done.json").stat().st_mtime
    capsys.readouterr()
    collect.main(base)
    log = capsys.readouterr().out
    assert "ep00379-00379 done, skip" in log and "ep00807-00807: 143 timesteps" in log
    assert (Path(parts[0]["dir"]) / "done.json").stat().st_mtime == keep
    # another configuration in the same dir is refused
    with pytest.raises(SystemExit, match="another configuration"):
        collect.main([*base[:-1], "emb,pg:0"])

    # stored values = recorder on the same frames; counts exclude padding and the empty camera
    d = Path(store.done_parts(acts)[0]["dir"])
    meta = pd.read_parquet(d / "meta.parquet")
    assert list(meta["episode_index"].unique()) == [379] and meta["frame_index"].tolist() == list(range(112))
    assert meta["suite"].iloc[0] == "libero_goal" and len(meta["action"].iloc[0]) == 7
    counts = np.load(d / "counts.npy")
    n_words = len(meta["instruction"].iloc[0].split())
    assert (counts == [IMG_TOK, IMG_TOK, min(n_words, TXT_LEN)]).all()
    pg17 = np.load(d / "pg17.npy")
    assert pg17.shape == (112, 3, D_PG) and pg17.dtype == np.float16
    assert np.load(d / "ae00.npy").shape == (112, 10, D_AE)
    done = json.loads((d / "done.json").read_text())
    assert done["overflow_fp32"] == [] and done["rows"] == 112

    # pooled_all from the stored groups = mean over all valid tokens, recomputed directly on frame 5
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo, root, rev = collect.dataset_source(out)
    ds = LeRobotDataset(repo, root=root, revision=rev, episodes=[379])
    item = torch.utils.data.default_collate([ds[5]])
    obs = mock_pre(item)
    pol = mock_loader
    cap = {}
    h = pol.model.lm.layers[17].register_forward_hook(lambda m, a, o: cap.setdefault("o", o))
    with ActivationRecorder(pol, ["pg17"]) as rec:
        pol.predict_action_chunk(obs, torch.zeros(1, 5, 32))
    h.remove()
    embs, pad, _ = pol.model.embed_prefix(
        [obs["observation.images.image"], obs["observation.images.image2"], obs["observation.images.image"]],
        [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)],
        obs["observation.language.tokens"],
        obs["observation.language.attention_mask"],
    )
    direct = (cap["o"] * pad[..., None]).sum(1) / pad.sum()
    pooled = store.pool_pg(pg17[5:6], counts[5:6], "pooled_all")
    assert np.allclose(pooled[0], direct[0].numpy(), atol=2e-2 * float(direct.abs().max()))
    assert torch.allclose(rec.pg["pg17"][0, 2], (cap["o"][0, 3 * IMG_TOK :][pad[0, 3 * IMG_TOK :]]).mean(0))

    x, m = store.load_layer(acts, "ae17")
    assert x.shape == (3650, D_AE) and sorted(m["ae_step"].unique()) == list(range(10))
    x, m = store.load_layer(acts, "pg00", "cam2")
    assert x.shape == (365, D_PG) and x.dtype == np.float32

    # train two layers briefly (+ resume) and write the stats
    import train

    targs = [
        "--model",
        "pi05lib",
        "--out-root",
        str(out),
        "--gpus",
        "cpu",
        "--batch",
        "64",
        "--steps",
        "60",
        "--log-every",
        "10",
        "--ckpt-every",
        "20",
        "--wandb",
        "off",
    ]
    train.main([*targs, "--layers", "pg:17", "--stop-after", "20"])
    train.main([*targs, "--layers", "pg:17,ae:17"])
    jd = out / "sae" / "pi05lib" / "sae" / "pooled_all" / "pg17" / "seed0"
    lg = [json.loads(x) for x in (jd / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lg] == [10, 20, 30, 40, 50, 60] and lg[-1]["nmse"] < lg[0]["nmse"]
    ft = pd.read_parquet(jd / "features.parquet")
    assert len(ft) == D_PG and ft["ep_coverage"].max() <= 1
    grp = pd.read_parquet(jd / "fire_by_group.parquet")
    assert set(grp.loc[grp["group_type"] == "suite", "group"]) == {
        "libero_goal",
        "libero_object",
        "libero_spatial",
    }
    ja = out / "sae" / "pi05lib" / "sae" / "ae-steps-all" / "ae17" / "seed0"
    assert sorted(pd.read_parquet(ja / "features.parquet")["ae_step"].unique()) == list(range(10))


def test_smoke_summary_on_mock(tmp_path, mock_loader, capsys):
    """smoke.sh steps 2-3 (all layers, pg05/ae05 stopped at 200 and resumed) with the mock → `smoke.py --summarize`."""
    import smoke
    import train

    out = tmp_path

    def logged(fn, argv, name):
        capsys.readouterr()
        fn(argv)
        (out / name).write_text(capsys.readouterr().out)

    base = ["--model", "pi05lib", "--out-root", str(out), "--episodes", EPS, "--parts", "3"]
    base += ["--gpus", "cpu", "--workers", "0", "--batch", "32"]
    logged(collect.main, base, "collect.log")
    logged(collect.main, base, "collect_resume.log")
    common = [
        "--model",
        "pi05lib",
        "--out-root",
        str(out),
        "--gpus",
        "cpu",
        "--batch",
        "64",
        "--steps",
        "600",
    ]
    common += ["--log-every", "20", "--ckpt-every", "100", "--wandb", "off", "--layers", "pg:5,ae:5"]
    logged(train.main, [*common, "--stop-after", "200"], "train_part1.log")
    logged(train.main, common, "train_resume.log")
    hook = {
        "a_actions_equal_with_hooks": True,
        "b_layer_out_equals_next_in": True,
        "b_emb_is_embed_prefix_output": True,
        "c_empty_camera_tokens_masked": True,
        "c_group_mean_max_abs_err": 0.0,
        "d_weighted_mean_rel_err": 0.0,
        "e_pg05_project_out_max_component": 0.0,
        "e_pg05_bf16_component_scale": 1.0,
        "e_ae05_project_out_max_component": 0.0,
        "e_ae05_bf16_component_scale": 1.0,
        "ae_steps_seen": 10,
    }
    (out / "hook_check.json").write_text(json.dumps(hook))
    (out / "sae_bench.json").write_text(json.dumps({"pg": {"hours": 0.1}, "ae": {"hours": 1.0}}))
    capsys.readouterr()
    assert smoke.summarize(out, "pi05lib")
    res = json.loads((out / "summary.json").read_text())
    assert res["pass"] and res["estimates"]["collect_full_storage_gb"] == pytest.approx(164.7, abs=0.5)
    assert res["estimates"]["sae_hours_per_layer_100_epochs"]["all_37_layers_1gpu"] == pytest.approx(19.9)
    (out / "hook_check.json").write_text(json.dumps({**hook, "a_actions_equal_with_hooks": False}))
    assert not smoke.summarize(out, "pi05lib")


def test_recorder_restores_outer_wrapper():
    pol = MockPolicy()
    outer = pol.model.embed_prefix  # bound class method

    def wrapped(*a, **k):
        return outer(*a, **k)

    pol.model.embed_prefix = wrapped
    with ActivationRecorder(pol, ["pg00"]):
        assert pol.model.embed_prefix is not wrapped
    assert pol.model.embed_prefix is wrapped and "_embed_images" not in pol.model.__dict__
    del pol.model.embed_prefix


def test_ae_step_subset(tmp_path, mock_loader):
    """--ae-steps first,last stores exactly steps 0 and 9 of the full 10-step recording."""
    common = ["--model", "pi05lib", "--episodes", "379", "--parts", "1", "--gpus", "cpu", "--workers", "0"]
    common += ["--layers", "ae:0,17"]
    collect.main([*common, "--out-root", str(tmp_path / "all")])
    collect.main([*common, "--out-root", str(tmp_path / "fl"), "--ae-steps", "first,last"])
    full = np.load(next((tmp_path / "all").glob("sae/*/acts/parts/ep*/ae17.npy")))
    sub = np.load(next((tmp_path / "fl").glob("sae/*/acts/parts/ep*/ae17.npy")))
    assert full.shape[1] == 10 and sub.shape[1] == 2
    assert np.array_equal(sub, full[:, [0, 9]])
    assert not np.array_equal(full[:, 0], full[:, 9])
