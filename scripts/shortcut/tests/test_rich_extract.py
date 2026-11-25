"""T028 rich extraction (serve/activations.RichHooks + analysis/rich_extract.py), CPU only.

A tiny pi05-shaped model (module names and call structure of PI05Pytorch: embed_prefix → VLM prefix pass with
a KV cache → euler_integrate over denoise_step on the action expert) runs through run_episode: keys and shapes,
x_N = the predicted chunk, probe row 0 = the main pass, internal consistency of the vectors, batching
invariance, PCA fit / projection.

    uv run pytest scripts/shortcut/tests/test_rich_extract.py -q
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

SHORTCUT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "serve"), str(SHORTCUT / "parallel"), str(SHORTCUT / "analysis")]

import rich_extract as RE  # noqa: E402, N812
from serve import activations as A  # noqa: E402, N812

from lerobot.policies.common.flow_matching import euler_integrate  # noqa: E402
from lerobot.policies.common.vla_utils import make_att_2d_masks  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE  # noqa: E402

D_V, D_L, D_KV, D_X, H, NL, LT, CH, AD, NS, NIMG, C = 8, 12, 4, 10, 2, 18, 64, 12, 8, 10, 3, 5


class CharTok:
    bos_token_id = 1

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> dict:
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class Mlp(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.up = nn.Linear(d, 2 * d)
        self.down_proj = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.tanh(self.up(x)))


class VisLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = Mlp(D_V)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.tanh(x + self.mlp(x)),)


class VlmAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(D_L, D_KV) for _ in range(3))
        self.o_proj = nn.Linear(D_KV, D_L)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor, cache: list) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        cache.append((k, v))
        s = (q @ k.transpose(-1, -2)).masked_fill(~key_mask.bool(), -1e9)  # (B, P, P) like make_att_2d_masks
        return self.o_proj(torch.softmax(s, -1) @ v)


class VlmLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = VlmAttn()
        self.mlp = Mlp(D_L)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor, cache: list) -> tuple[torch.Tensor]:
        x = x + self.self_attn(x, key_mask, cache)
        return (torch.tanh(x + self.mlp(x)),)


class ExpAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=H)
        self.q_proj = nn.Linear(D_X, H * D_KV)
        self.k_proj, self.v_proj = nn.Linear(D_X, D_KV), nn.Linear(D_X, D_KV)
        self.o_proj = nn.Linear(H * D_KV, D_X)

    def forward(self, x: torch.Tensor, kv: tuple, key_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, nq, _ = x.shape
        q = self.q_proj(x).view(b, nq, H, D_KV).transpose(1, 2)
        k = torch.cat([kv[0], self.k_proj(x)], 1)[:, None]
        v = torch.cat([kv[1], self.v_proj(x)], 1)[:, None]
        s = (q @ k.transpose(-1, -2)).masked_fill(~key_mask[:, None, None, :], -1e9)
        w = torch.softmax(s, -1)
        return self.o_proj((w @ v).transpose(1, 2).reshape(b, nq, -1)), w


class ExpLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = ExpAttn()
        self.mlp = Mlp(D_X)

    def forward(self, x: torch.Tensor, kv: tuple, key_mask: torch.Tensor) -> tuple[torch.Tensor]:
        a, _ = self.self_attn(x, kv, key_mask)
        x = x + a
        return (torch.tanh(x + self.mlp(x)),)


class Cache:
    def __init__(self, kv: list) -> None:
        self.kv = kv

    def batch_repeat_interleave(self, r: int) -> None:
        self.kv = [(k.repeat_interleave(r, 0), v.repeat_interleave(r, 0)) for k, v in self.kv]


class Pwe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma = nn.Module()
        self.paligemma.model = nn.Module()
        vt = self.paligemma.model.vision_tower = nn.Module()
        vt.vision_model = nn.Module()
        vt.vision_model.encoder = nn.Module()
        vt.vision_model.encoder.layers = nn.ModuleList(VisLayer() for _ in range(3))
        self.paligemma.model.language_model = nn.Module()
        self.paligemma.model.language_model.layers = nn.ModuleList(VlmLayer() for _ in range(NL))
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = nn.Module()
        self.gemma_expert.model.layers = nn.ModuleList(ExpLayer() for _ in range(NL))

    def forward(self, attention_mask: Any = None, position_ids: Any = None, past_key_values: Any = None,
                inputs_embeds: Any = None, use_cache: Any = None, adarms_cond: Any = None) -> Any:  # fmt: skip
        if inputs_embeds[1] is None:
            x, kv = inputs_embeds[0], []
            for layer in self.paligemma.model.language_model.layers:
                x = layer(x, attention_mask, kv)[0]
            return [x, None], Cache(kv)
        x = inputs_embeds[1]
        for i, layer in enumerate(self.gemma_expert.model.layers):
            x = layer(x, past_key_values.kv[i], attention_mask)[0]
        return [None, x], None


class TinyModel(nn.Module):
    use_typed_attention_masks = False
    precompute_denoise_times = False

    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = Pwe()
        self.patch = nn.Linear(3, D_V)
        self.v2l = nn.Linear(D_V, D_L)
        self.emb = nn.Embedding(200, D_L)
        self.a_in, self.t_in, self.a_out = nn.Linear(AD, D_X), nn.Linear(1, D_X), nn.Linear(D_X, AD)

    def sample_noise(self, shape: tuple, device: Any) -> torch.Tensor:
        return torch.randn(shape, device=device)

    def _prepare_attention_masks_4d(self, m: torch.Tensor, dtype: Any = None) -> torch.Tensor:
        return m

    def embed_prefix(self, images: list, img_masks: list, tokens: torch.Tensor, masks: torch.Tensor,
                     states: Any = None, state_masks: Any = None) -> tuple:  # fmt: skip
        b = tokens.shape[0]
        x = torch.cat(images)  # camera-major SigLIP call
        v = self.patch(x.flatten(2).transpose(1, 2)[:, : A.IMG_TOKENS])
        for layer in self.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers:
            v = layer(v)[0]
        v = self.v2l(v).view(NIMG, b, A.IMG_TOKENS, D_L).transpose(0, 1).reshape(b, NIMG * A.IMG_TOKENS, D_L)
        embs = torch.cat([v, self.emb(tokens.clamp(max=199))], 1)
        pad = torch.cat([m[:, None].expand(b, A.IMG_TOKENS) for m in img_masks] + [masks.bool()], 1)
        return embs, pad, torch.zeros_like(pad)

    def denoise_step(self, prefix_pad_masks: torch.Tensor, past_key_values: Any, x_t: torch.Tensor,
                     timestep: torch.Tensor) -> torch.Tensor:  # fmt: skip
        b = x_t.shape[0]
        h = self.a_in(x_t) + self.t_in(timestep[:, None, None].expand(b, CH, 1))
        key_mask = torch.cat([prefix_pad_masks, torch.ones(b, CH, dtype=torch.bool)], 1)
        out, _ = self.paligemma_with_expert.forward(
            attention_mask=key_mask, past_key_values=past_key_values, inputs_embeds=[None, h]
        )
        return self.a_out(out[1])

    def sample_actions(self, images: list, img_masks: list, tokens: torch.Tensor, masks: torch.Tensor,
                       states: Any = None, state_masks: Any = None, noise: Any = None) -> torch.Tensor:  # fmt: skip
        embs, pad, att = self.embed_prefix(images, img_masks, tokens, masks, states, state_masks)
        m2 = self._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        _, past = self.paligemma_with_expert.forward(attention_mask=m2, inputs_embeds=[embs, None], use_cache=True)
        return euler_integrate(
            lambda x, t: self.denoise_step(prefix_pad_masks=pad, past_key_values=past, x_t=x, timestep=t), noise, NS
        )


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.config = SimpleNamespace(
            chunk_size=CH, max_action_dim=AD, num_inference_steps=NS,
            image_features={"observation.images.image": 0, "observation.images.image2": 0, "empty": 0},
        )  # fmt: skip
        self.model = TinyModel()

    def _preprocess_images(self, batch: dict) -> tuple[list, list]:
        im = [batch["observation.images.image"], batch["observation.images.image2"]]
        b = im[0].shape[0]
        return im + [-torch.ones_like(im[0])], [torch.ones(b, dtype=torch.bool)] * 2 + [torch.zeros(b, dtype=torch.bool)]

    def _prepare_memory_states(self, batch: dict) -> tuple:
        return batch[OBS_STATE], None

    def predict_action_chunk(self, batch: dict, noise: torch.Tensor | None = None) -> torch.Tensor:
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        return self.model.sample_actions(images, img_masks, tokens, masks, noise=noise)[:, :, :7]


def pre(batch: dict) -> dict:
    tok = CharTok()
    rows, masks = [], []
    for i, task in enumerate(batch["task"]):
        st = " ".join(str(int(v * 10)) for v in batch[OBS_STATE][i][: 1 + i % 3].tolist())
        ids = [1, *tok(f"Task: {task}, State: {st};\nAction: ")["input_ids"]][:LT]
        rows.append(ids + [0] * (LT - len(ids)))
        masks.append([1] * len(ids) + [0] * (LT - len(ids)))
    return {**batch, OBS_LANGUAGE_TOKENS: torch.tensor(rows), OBS_LANGUAGE_ATTENTION_MASK: torch.tensor(masks, dtype=torch.bool)}  # noqa: E501


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("rich")
    policy = TinyPolicy().eval()
    L = SimpleNamespace(policy=policy, pre=pre, post=lambda a: a, env_post=lambda d: {ACTION: d[ACTION]})  # noqa: N806
    hooks = A.RichHooks(policy)
    path_rec = RE.PathRecorder(policy.model)
    rows = []
    for e in range(4):
        rng = np.random.default_rng(e)
        obs_p, steps_p = tmp / f"obs{e}.npz", tmp / f"steps{e}.npz"
        np.savez(obs_p, image=rng.integers(0, 255, (C, 16, 16, 3), dtype=np.uint8),
                 image2=rng.integers(0, 255, (C, 16, 16, 3), dtype=np.uint8),
                 state=rng.normal(size=(C, 8)).astype(np.float32), t=np.arange(C) * 10,
                 instruction=np.asarray("put the bowl on the plate"), seed=np.asarray(7 + e))  # fmt: skip
        np.savez(steps_p, action=rng.normal(size=(C * 10, 7)).astype(np.float32))
        rows.append({"episode": f"ep{e}", "set": "s", "job": "j", "obs_path": str(obs_p), "steps_path": str(steps_p),
                     "old_out_path": str(tmp / "none.npz"), "probe_calls": "0;1;3;4", "probe_anchor": "grasp",
                     "path": f"s/j/ep{e}.npz", "pilot": "1", "pca_fit": "1" if e < 3 else "0"})  # fmt: skip
    return {"tmp": tmp, "L": L, "hooks": hooks, "path_rec": path_rec, "rows": rows}


def run(s: dict, row: dict, bases: Any = None, raw: bool = True, max_batch: int = 32, **kw: Any) -> tuple[dict, dict]:
    return RE.run_episode(row, s["L"], s["hooks"], s["path_rec"], CharTok(), "cpu", bases, raw, max_batch, **kw)


def test_keys_shapes_and_paths(setup: dict) -> None:
    out, info = run(setup, setup["rows"][0])
    N = out["lang_ids"].shape[1]  # noqa: N806
    nt, nx, ns = len(A.RICH_TOK_LAYERS), len(A.RICH_EXP_TOK_LAYERS), len(A.RICH_STEPS)
    shapes = {
        "attn_mass": (C, NS, NL, 7), "attn_img": (C, 3, 512), "exp_h": (C, NS, NL), "vlm_h": (C, NL, 6),
        "sig_h": (C, NIMG, 3), "act_pred": (C, 10, 7), "t": (C,),
        "vlm_seg": (C, NL, 6, D_L), "k_seg": (C, NL, 6, D_KV), "v_seg": (C, NL, 6, D_KV),
        "vlm_tok": (C, nt, N, D_L), "k_tok": (C, nt, N, D_KV), "v_tok": (C, nt, N, D_KV),
        "exp_tok": (C, nx, ns, CH, D_X), "exp_mean": (C, ns, NL, D_X),
        "exp_mlp_mean": (C, ns, len(A.RICH_MLP_LAYERS), 2 * D_X),
        "den_x": (C, NS + 1, CH, AD), "den_v": (C, NS, CH, AD), "lang_seg": (C, N), "lang_valid": (C, N),
        "mn_calls": (4,), "mn_seeds": (4, RE.N_NOISE), "mn_chunk": (4, RE.N_NOISE, CH, AD),
        "mn_exp_last": (4, RE.N_NOISE, D_X), "mn_act": (4, RE.N_NOISE, CH, 7),
    }  # fmt: skip
    for k, shp in shapes.items():
        assert out[k].shape == shp, (k, out[k].shape, shp)
    for k in ("vlm_seg", "k_tok", "exp_tok", "exp_mean", "exp_mlp_mean", "mn_exp_last"):
        assert out[k].dtype == np.float16, k
    assert out["den_x"].dtype == np.float32 and out["mn_chunk"].dtype == np.float32
    # Euler path: x_{k+1} = x_k - v_k / n and x_N is the executed chunk
    np.testing.assert_allclose(out["den_x"][:, 1:], out["den_x"][:, :-1] - out["den_v"] / NS, atol=1e-5)
    np.testing.assert_allclose(out["den_x"][:, -1, :, :7][:, : out["act_pred"].shape[1]], out["act_pred"], atol=1e-6)
    # probe row 0 = original noise → reproduces the main pass; other rows differ
    assert info["mn_self_diff"] < 1e-5
    assert np.abs(out["mn_chunk"][:, 1] - out["mn_chunk"][:, 0]).max() > 1e-3
    assert (out["mn_seeds"][:, 0] == -1).all() and len(set(out["mn_seeds"][:, 1:].ravel())) == 4 * (RE.N_NOISE - 1)
    # x_0 = single-exec noise: manual_seed(episode seed), one draw per call in order
    torch.manual_seed(7)
    ref_noise = torch.cat([torch.randn(1, CH, AD) for _ in range(C)]).numpy()
    np.testing.assert_array_equal(out["den_x"][:, 0], ref_noise)
    meta = json.loads(str(out["meta"]))
    assert meta["tok_layers"] == list(A.RICH_TOK_LAYERS) and meta["raw"] is True


def test_vectors_consistent(setup: dict) -> None:
    out, _ = run(setup, setup["rows"][1])
    valid, seg = out["lang_valid"], out["lang_seg"]
    for j, li in enumerate(A.RICH_TOK_LAYERS):
        for sid in (3, 4, 5):  # task / state / template: segment mean = mean of that segment's language tokens
            for name in ("k", "v", "vlm"):
                tok = out[f"{name}_tok"][:, j].astype(np.float32)
                m = (seg == sid) & valid
                ref = (tok * m[..., None]).sum(1) / m.sum(1)[:, None]
                np.testing.assert_allclose(out[f"{name}_seg"][:, li, sid].astype(np.float32), ref, atol=2e-2, rtol=2e-2)
    for j, li in enumerate(A.RICH_EXP_TOK_LAYERS):
        np.testing.assert_allclose(
            out["exp_tok"][:, j].astype(np.float32).mean(-2), out["exp_mean"][:, :, li].astype(np.float32), atol=2e-3
        )
    # pad positions are zero; the old norm features match the new vectors
    assert (out["vlm_tok"][~np.broadcast_to(valid[:, None], out["vlm_tok"].shape[:3])] == 0).all()
    # exp_vec (T023, mean over action tokens at layers 5/11/17, all steps) == exp_mean at the kept steps
    for js, s in enumerate(A.RICH_STEPS):
        for jl, li in enumerate(A.VEC_LAYERS):
            np.testing.assert_allclose(out["exp_vec"][:, s, jl], out["exp_mean"][:, js, li], atol=2e-3)


def test_batching_invariance(setup: dict) -> None:
    a, _ = run(setup, setup["rows"][2], max_batch=32)
    b, _ = run(setup, setup["rows"][2], max_batch=2)
    for k in ("attn_mass", "vlm_seg", "k_tok", "exp_tok", "den_x", "mn_chunk", "act_pred", "lang_ids"):
        np.testing.assert_allclose(a[k].astype(np.float32), b[k].astype(np.float32), atol=2e-3, err_msg=k)


def test_pca_fit_and_projection(setup: dict) -> None:
    dims = {"vlm_tok": 6, "exp_tok": 5, "k_tok": 3, "vlm_seg": 7}
    out_dir = setup["tmp"] / "out"
    for r in setup["rows"]:
        arrays, _ = run(setup, r)
        RE.save_npz(out_dir / r["path"], arrays)
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(setup["rows"][0]))
        w.writeheader()
        w.writerows(setup["rows"])
    RE.cmd_fit_pca(SimpleNamespace(out=str(out_dir), threads=2))
    z = np.load(out_dir / "pca_bases.npz")
    rep = json.loads((out_dir / "pca_report.json").read_text())
    width = {"vlm_tok": D_L, "vlm_seg": D_L, "exp_tok": D_X, "k_tok": D_KV, "v_tok": D_KV}
    for s, (layers, _k, _m) in RE.PCA_STREAMS.items():
        for li in layers:
            comp = z[f"{s}/L{li}/components"]
            assert comp.shape == (width[s], width[s])  # max k > tiny widths: full basis
            np.testing.assert_allclose(comp @ comp.T, np.eye(comp.shape[0]), atol=1e-4)
            ev = z[f"{s}/L{li}/explained_variance"]
            assert (np.diff(ev) <= 1e-6).all()
            np.testing.assert_allclose(ev.sum(), z[f"{s}/L{li}/total_variance"], rtol=1e-5)
            curve = list(rep[s][f"L{li}"]["heldout_ratio"].values())
            assert all(0 <= c <= 1 + 1e-6 for c in curve) and curve == sorted(curve)
            assert abs(curve[-1] - 1) < 1e-3  # full basis captures everything
    (out_dir / "config.json").write_text(json.dumps({"pca": dims, "mlp_layers": [2, 5], "keep_vlm_vec": False}))
    RE.cmd_project_pilot(SimpleNamespace(out=str(out_dir), strip_fit_raw=False))
    d = np.load(out_dir / setup["rows"][3]["path"])
    for s, k in dims.items():
        assert d[f"{s}_pc"].shape == d[s].shape[:-1] + (k,)
    # main-mode projection on the fly == projection of the raw pilot file
    bases = RE.Bases(out_dir / "pca_bases.npz", "cpu", dims)
    m, _ = run(setup, setup["rows"][3], bases=bases, raw=False, dims=dims, mlp_keep=(2, 5), keep_vlm_vec=False)
    for s in dims:
        assert s not in m
        np.testing.assert_allclose(m[f"{s}_pc"].astype(np.float32), d[f"{s}_pc"].astype(np.float32), atol=2e-2)
    assert "v_tok" in m and "v_tok_pc" not in m and "vlm_vec" not in m
    np.testing.assert_allclose(m["exp_mlp_mean"], d["exp_mlp_mean"][:, :, [2, 5]], atol=2e-3)
    # manual projection
    j = 2
    x = d["exp_tok"][:, j].astype(np.float32)
    li = RE.PCA_STREAMS["exp_tok"][0][j]
    ref = (x - z[f"exp_tok/L{li}/mean"]) @ z[f"exp_tok/L{li}/components"][:5].T
    np.testing.assert_allclose(d["exp_tok_pc"][:, j].astype(np.float32), ref, atol=2e-2)



def test_probe_calls() -> None:
    assert RE.probe_calls(28, 5, None) == ([0, 3, 8, 27], "grasp")
    assert RE.probe_calls(28, 1, 9) == ([0, 0, 4, 27], "grasp")
    assert RE.probe_calls(28, None, 9) == ([0, 7, 12, 27], "contact")
    assert RE.probe_calls(30, None, None) == ([0, 13, 18, 29], "middle")
    assert RE.probe_calls(28, 26, None) == ([0, 24, 27, 27], "grasp")
