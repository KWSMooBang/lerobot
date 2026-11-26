"""CPU tests for the T034 crosscoder: uv run pytest scripts/shortcut/sae/tests/test_crosscoder.py -q"""

import sys
from pathlib import Path

import torch

SAE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAE_DIR))

from crosscoder import CrossCoder, CrossCoderConfig  # noqa: E402
from model import SAEConfig, TopKSAE  # noqa: E402


def _data(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * 3 + torch.randn(d, generator=g)


def test_one_layer_reduces_to_topk_sae():
    d, n, k = 32, 64, 4
    x = _data(256, d)
    sae = TopKSAE(SAEConfig(d=d, n_latents=n, k=k, k_aux=8), seed=3)
    cc = CrossCoder(CrossCoderConfig(layers=["L"], dims=[d], n_latents=n, k=k, k_aux=8), seed=3)
    # same initial draw
    assert torch.allclose(cc.W_dec.T, sae.W_dec) and torch.allclose(cc.W_enc, sae.W_enc)
    sae.init_from_data(x)
    cc.init_from_data(x)
    assert torch.allclose(cc.b_pre, sae.b_pre) and abs(cc.cfg.c_mse[0] - sae.cfg.c_mse) < 1e-6
    # random weights (decoder rows unit norm), copied across
    with torch.no_grad():
        sae.W_enc.normal_()
        sae.W_dec.normal_()
        sae.renorm_decoder()
        cc.W_enc.copy_(sae.W_enc)
        cc.W_dec.copy_(sae.W_dec.T)
    v1, i1 = sae.encode(x)
    v2, i2 = cc.encode(x)
    assert torch.equal(i1, i2) and torch.allclose(v1, v2)
    assert torch.allclose(sae.reconstruct(x), cc.reconstruct(x), atol=1e-5)
    dead = torch.zeros(n, dtype=torch.bool)
    dead[::3] = True
    l1, p1, f1 = sae.loss(x, dead)
    l2, p2, f2 = cc.loss(x, dead)
    assert torch.allclose(l1, l2, atol=1e-6) and torch.allclose(p1["aux"], p2["aux"], atol=1e-6)
    assert torch.equal(f1, f2)
    # gradients agree (decoder transposed)
    l1.backward()
    l2.backward()
    assert torch.allclose(sae.W_enc.grad, cc.W_enc.grad, atol=1e-5)
    assert torch.allclose(sae.W_dec.grad, cc.W_dec.grad.T, atol=1e-5)
    sae.project_decoder_grad()
    cc.project_decoder_grad()
    assert torch.allclose(sae.W_dec.grad, cc.W_dec.grad.T, atol=1e-5)


def _multi(dims=(16, 16, 8), n=40, k=5) -> tuple[CrossCoder, torch.Tensor]:
    cfg = CrossCoderConfig(layers=[f"l{i}" for i in range(len(dims))], dims=list(dims), n_latents=n, k=k, k_aux=6)
    cc = CrossCoder(cfg, seed=0)
    x = _data(128, cfg.D, seed=1)
    cc.init_from_data(x)
    return cc, x


def test_shapes_and_per_block_normalisation():
    for dims in ((16, 16, 16), (16, 16, 8)):
        cc, x = _multi(dims)
        xt, mu, nrm = cc.normalize(x)
        assert xt.shape == x.shape and mu.shape == (128, len(dims)) and nrm.shape == (128, len(dims))
        sq = cc.block_sq(xt)
        assert torch.allclose(sq, torch.ones_like(sq), atol=1e-5)  # every block unit norm
        o = cc.cfg.offsets
        for i in range(len(dims)):
            assert torch.allclose(xt[:, o[i] : o[i + 1]].mean(-1), torch.zeros(128), atol=1e-5)
        v, idx = cc.encode(x)
        assert v.shape == idx.shape == (128, 5) and (v >= 0).all()
        assert cc.reconstruct(x).shape == x.shape
        assert len(cc.cfg.c_mse) == len(dims)


def test_uniform_and_ragged_paths_agree():
    cc, x = _multi((16, 16, 16))
    a = cc.normalize(x)
    cc._uniform = False
    b = cc.normalize(x)
    for u, w in zip(a, b, strict=True):
        assert torch.allclose(u, w, atol=1e-6)
    assert torch.allclose(cc.block_sq(x), torch.stack([x[:, i * 16 : (i + 1) * 16].pow(2).sum(-1) for i in range(3)], -1))


def test_decoder_norm_constraint_and_profile():
    cc, x = _multi()
    opt = torch.optim.Adam(cc.parameters(), lr=1e-2)
    last = torch.zeros(cc.cfg.n_latents, dtype=torch.long)
    for step in range(20):
        loss, parts, fired = cc.loss(x, (step - last) >= 5)
        assert parts["mse_l"].shape == (3,)
        opt.zero_grad()
        loss.backward()
        cc.project_decoder_grad()
        assert torch.allclose((cc.W_dec.grad * cc.W_dec).sum(1), torch.zeros(cc.cfg.n_latents), atol=1e-5)
        opt.step()
        cc.renorm_decoder()
        last[fired] = step + 1
    assert torch.allclose(cc.W_dec.norm(dim=1), torch.ones(cc.cfg.n_latents), atol=1e-5)
    prof = cc.layer_profile()
    assert prof.shape == (cc.cfg.n_latents, 3) and torch.allclose(prof.sum(1), torch.ones(cc.cfg.n_latents), atol=1e-5)


def test_loss_decreases():
    cc, x = _multi()
    opt = torch.optim.Adam(cc.parameters(), lr=3e-3)
    first = None
    for _ in range(100):
        loss, parts, _ = cc.loss(x, None)
        first = float(parts["mse"]) if first is None else first
        opt.zero_grad()
        loss.backward()
        cc.project_decoder_grad()
        opt.step()
        cc.renorm_decoder()
    assert float(parts["mse"]) < 0.8 * first


def test_save_load_roundtrip(tmp_path):
    cc, x = _multi()
    cc.save(tmp_path / "cc", extra={"note": "t"})
    cc2 = CrossCoder.load(tmp_path / "cc")
    assert cc2.cfg.layers == cc.cfg.layers and cc2.cfg.dims == cc.cfg.dims and cc2.cfg.c_mse == cc.cfg.c_mse
    assert torch.equal(cc2.c_vec, cc.c_vec)
    for k, v in cc.state_dict().items():
        assert torch.equal(v, cc2.state_dict()[k])
    assert torch.allclose(cc.reconstruct(x), cc2.reconstruct(x))
    l1, _, _ = cc.loss(x, None)
    l2, _, _ = cc2.loss(x, None)
    assert torch.allclose(l1, l2)
