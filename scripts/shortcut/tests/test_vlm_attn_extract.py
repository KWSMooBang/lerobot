"""T035 vlm_attn_extract.reduce_attn / seg_from_full on random attention, CPU only.

    uv run pytest scripts/shortcut/tests/test_vlm_attn_extract.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SHORTCUT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "serve"), str(SHORTCUT / "parallel"), str(SHORTCUT / "analysis")]

import vlm_attn_extract as VA  # noqa: E402, N812
from serve.activations import IMG_TOKENS, PAD  # noqa: E402

B, H, LT = 3, 4, 20


def layout() -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """3 cameras (third empty) + LT language slots: BOS, 5 task, 3-5 state (varies per call), 4 template, pad."""
    P = 3 * IMG_TOKENS + LT  # noqa: N806
    seg = torch.full((B, P), PAD, dtype=torch.long)
    seg[:, :IMG_TOKENS], seg[:, IMG_TOKENS : 2 * IMG_TOKENS] = 0, 1
    o = 3 * IMG_TOKENS
    vpos = []
    for b in range(B):
        ns = 3 + b
        lang = [2] + [3] * 5 + [4] * ns + [5] * 4
        seg[b, o : o + len(lang)] = torch.tensor(lang)
        vpos.append(torch.cat([torch.arange(2 * IMG_TOKENS), torch.arange(o, o + len(lang))]))
    return seg, torch.arange(o + 1, o + 6), vpos


def attention(seg: torch.Tensor) -> torch.Tensor:
    P = seg.shape[1]  # noqa: N806
    s = torch.randn(B, H, P, P) * 3
    s = s.masked_fill((seg == PAD)[:, None, None, :], float("-inf"))
    return torch.softmax(s, -1)


def test_reduce_matches_loops() -> None:
    torch.manual_seed(0)
    seg, ipos, vpos = layout()
    w = attention(seg)
    V = max(len(v) for v in vpos)  # noqa: N806
    vidx = torch.zeros(B, V, dtype=torch.long)
    vvalid = torch.zeros(B, V, dtype=torch.bool)
    for b, v in enumerate(vpos):
        vidx[b, : len(v)], vvalid[b, : len(v)] = v, True
    r = VA.reduce_attn(w, seg, ipos, vidx, vvalid)
    assert r["seg_attn"].shape == (B, H, 6, 6)
    torch.testing.assert_close(r["seg_attn"].sum(-1), torch.ones(B, H, 6), atol=1e-5, rtol=0)
    assert float(r["pad_mass"].max()) == 0.0 and float(r["row_err"].max()) < 1e-5
    for b in range(B):
        for h in (0, H - 1):
            for qs in range(6):
                q = (seg[b] == qs).nonzero().flatten()
                for ks in range(6):
                    k = (seg[b] == ks).nonzero().flatten()
                    ref = w[b, h][q][:, k].sum(1).mean()
                    assert abs(float(r["seg_attn"][b, h, qs, ks] - ref)) < 1e-5
        wm = w[b].mean(0)
        torch.testing.assert_close(r["i2p"][b], wm[ipos][:, : 2 * IMG_TOKENS])
        torch.testing.assert_close(r["p2i"][b], wm[: 2 * IMG_TOKENS][:, ipos])
        n = len(vpos[b])
        torch.testing.assert_close(r["full"][b, :n, :n], wm[vpos[b]][:, vpos[b]])
        assert float(r["full"][b, n:].abs().sum() + r["full"][b, :, n:].abs().sum()) == 0
    # ① (head mean) rebuilt from ③
    full = r["full"][:, None].numpy()  # (B, 1 layer, V, V)
    fseg = np.zeros((B, V), np.int8)
    for b, v in enumerate(vpos):
        fseg[b, : len(v)] = seg[b, v].numpy()
    rebuilt = VA.seg_from_full(full, fseg, vvalid.numpy())
    np.testing.assert_allclose(rebuilt[:, 0], r["seg_attn"].mean(1).numpy(), atol=1e-5)


def test_goal_pool() -> None:
    assert VA.goal_pool({"set": "goal", "condition": "train", "outcome": ""}) == "train"
    assert VA.goal_pool({"set": "goal", "condition": "cand", "outcome": "fail_trained"}) == "cand_fallback"
    assert VA.goal_pool({"set": "goal", "condition": "cand", "outcome": "new_byproduct"}) == "cand_new"
    assert VA.goal_pool({"set": "goal", "condition": "cand", "outcome": "partial"}) == "cand_other"
    assert VA.goal_pool({"set": "libero3", "condition": "object", "outcome": ""}) == ""
