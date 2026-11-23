"""Policy-server activation recording (T023, serve/activations.py), CPU only.

A tiny pi0-shaped torch model (vision / VLM / expert layers with the module names pi05 uses) runs through
the real Engine: recording must not change actions, must write one npz per episode with the documented
shapes, must drop the padding rows of a batched call, and the segment ids must follow pi05's prompt.

    uv run --extra shortcut-bench pytest scripts/shortcut/tests/test_activations.py -q
"""

from __future__ import annotations

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
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "serve"), str(SHORTCUT / "parallel")]

from serve import activations as A  # noqa: E402, N812
from serve.engine import Engine, Loaded  # noqa: E402

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

D_V, D_L, D_X, H, NL, LT, CHUNK, NSTEP, NIMG = 8, 12, 10, 2, 18, 64, 6, 4, 3


class CharTok:
    """Character-level stand-in for the PaliGemma tokenizer (ids = code points, BOS = 1)."""

    bos_token_id = 1

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def __call__(
        self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False
    ) -> dict:
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class Mlp(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.up = nn.Linear(d, 2 * d)
        self.down_proj = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.tanh(self.up(x)))


class Attn(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=H)
        self.q, self.k = nn.Linear(d, H * 4), nn.Linear(d, H * 4)
        self.key_mask: torch.Tensor | None = None  # (B, K) True = attend (pi05 masks language padding)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, nq, _ = x.shape
        q = self.q(x).view(b, nq, H, 4).transpose(1, 2)
        k = self.k(kv).view(b, kv.shape[1], H, 4).transpose(1, 2)
        s = q @ k.transpose(-1, -2)
        if self.key_mask is not None:
            s = s.masked_fill(~self.key_mask[:, None, None, :], float("-inf"))
        w = torch.softmax(s, -1)  # (B, H, Q, K)
        return x + (w.mean(1) @ kv[..., : x.shape[-1]]), w


class Layer(nn.Module):
    def __init__(self, d: int, attn: bool = False):
        super().__init__()
        self.mlp = Mlp(d)
        self.self_attn = Attn(d) if attn else None

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> tuple[torch.Tensor]:
        if self.self_attn is not None:
            x, _ = self.self_attn(x, kv)
        return (torch.tanh(x + self.mlp(x)),)


class TinyPi(nn.Module):
    """predict_action_chunk: SigLIP on 3 cameras → VLM prefix pass → NSTEP expert passes over the
    chunk tokens attending to [prefix (projected), chunk]. Noise from the global RNG like pi05."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.config = SimpleNamespace(
            type="tinypi",
            chunk_size=CHUNK,
            max_action_dim=7,
            num_inference_steps=NSTEP,
            image_features={"observation.images.image": 0, "observation.images.image2": 0, "empty_cam": 0},
        )
        self.model = nn.Module()
        pwe = nn.Module()
        pwe.paligemma = nn.Module()
        pwe.paligemma.model = nn.Module()
        pwe.paligemma.model.vision_tower = nn.Module()
        pwe.paligemma.model.vision_tower.vision_model = nn.Module()
        pwe.paligemma.model.vision_tower.vision_model.encoder = nn.Module()
        pwe.paligemma.model.vision_tower.vision_model.encoder.layers = nn.ModuleList(
            Layer(D_V) for _ in range(3)
        )
        pwe.paligemma.model.language_model = nn.Module()
        pwe.paligemma.model.language_model.layers = nn.ModuleList(Layer(D_L) for _ in range(NL))
        pwe.gemma_expert = nn.Module()
        pwe.gemma_expert.model = nn.Module()
        pwe.gemma_expert.model.layers = nn.ModuleList(Layer(D_X, attn=True) for _ in range(NL))
        self.model.paligemma_with_expert = pwe
        self.patch = nn.Linear(3, D_V)
        self.v2l = nn.Linear(D_V, D_L)
        self.emb = nn.Embedding(200, D_L)
        self.l2x = nn.Linear(D_L, D_X)
        self.a_in, self.a_out = nn.Linear(7, D_X), nn.Linear(D_X, 7)

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch: dict[str, Any], noise: torch.Tensor | None = None) -> torch.Tensor:
        pwe = self.model.paligemma_with_expert
        imgs = [batch["observation.images.image"], batch["observation.images.image2"]]
        imgs.append(torch.zeros_like(imgs[0]))
        B = imgs[0].shape[0]  # noqa: N806
        x = torch.cat(imgs)  # (NIMG*B, 3, h, w), camera-major like pi05's SigLIP call
        tok = x.flatten(2).transpose(1, 2)[:, : A.IMG_TOKENS]
        v = self.patch(tok)
        for layer in pwe.paligemma.model.vision_tower.vision_model.encoder.layers:
            v = layer(v)[0]
        v = self.v2l(v).view(NIMG, B, A.IMG_TOKENS, D_L).transpose(0, 1).reshape(B, NIMG * A.IMG_TOKENS, D_L)
        prefix = torch.cat([v, self.emb(batch[OBS_LANGUAGE_TOKENS].clamp(max=199))], 1)
        for layer in pwe.paligemma.model.language_model.layers:
            prefix = layer(prefix)[0]
        kv_prefix = torch.tanh(self.l2x(prefix))
        key_mask = torch.cat(
            [
                torch.ones(B, 2 * A.IMG_TOKENS, dtype=torch.bool),
                torch.zeros(B, (NIMG - 2) * A.IMG_TOKENS, dtype=torch.bool),  # empty camera: masked
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                torch.ones(B, CHUNK, dtype=torch.bool),
            ],
            1,
        )
        for layer in pwe.gemma_expert.model.layers:
            layer.self_attn.key_mask = key_mask
        if noise is None:
            noise = torch.randn(B, CHUNK, 7)
        a = noise
        for _ in range(NSTEP):
            h = self.a_in(a)
            for layer in pwe.gemma_expert.model.layers:
                h = layer(h, torch.cat([kv_prefix, h], 1))[0]
            a = a - 0.1 * self.a_out(h)
        return a


class TokStep:
    def __init__(self) -> None:
        self.input_tokenizer = CharTok()


class Pre:
    steps = (TokStep(),)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        tok = CharTok()
        rows, masks = [], []
        for i, task in enumerate(batch["task"]):
            st = " ".join(str(int(v * 10)) for v in batch["observation.state"][i][:2].tolist())
            ids = [1, *tok(f"Task: {task}, State: {st};\nAction: ")["input_ids"]]
            ids = ids[:LT]
            rows.append(ids + [0] * (LT - len(ids)))
            masks.append([1] * len(ids) + [0] * (LT - len(ids)))
        return {
            **batch,
            OBS_LANGUAGE_TOKENS: torch.tensor(rows),
            OBS_LANGUAGE_ATTENTION_MASK: torch.tensor(masks, dtype=torch.bool),
        }

    def reset(self) -> None:
        pass


def make_engine(record_dir: Path | None, batch: int = 1) -> Engine:
    policy = TinyPi().eval()
    loaded = Loaded(
        policy, Pre(), lambda a: a, lambda d: {ACTION: d[ACTION]}, {"env_observation_hw": [32, 32]}, "tinypi"
    )
    return Engine(loaded, "cpu", 4, batch, {}, record_dir=str(record_dir) if record_dir else None)


def obs(k: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(k)
    return {
        "image": rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
        "image2": rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
        "state": rng.normal(size=8).astype(np.float32),
    }


def run(eng: Engine, eid: str, n: int, key: str | None = None) -> tuple[list[np.ndarray], str | None]:
    eng.reset(eid, 7, "put the bowl on the plate", record_key=key, record_meta={"job_id": "j"})
    acts = [eng.infer_one(eid, obs(k), "put the bowl on the plate", False) for k in range(n)]
    return acts, eng.end(eid)


def test_safe_key() -> None:
    assert (
        A.safe_key("job/libero_goal/task01/none_original_init03")
        == "job/libero_goal/task01/none_original_init03"
    )
    assert A.safe_key("../../etc/passwd") == "etc/passwd"
    assert A.safe_key("/abs/a b:c") == "abs/a_b_c"
    with pytest.raises(ValueError):
        A.safe_key("../..")


def test_lang_segments_follow_prompt() -> None:
    tok = CharTok()
    text = "Task: pick it, State: 1 2;\nAction: "
    seg = A.lang_segments(tok, [1, *tok(text)["input_ids"]])
    assert seg[0] == 2  # BOS
    body = seg[1:]
    assert "".join(c for c, s in zip(text, body, strict=True) if s == 3) == "pick it"
    assert "".join(c for c, s in zip(text, body, strict=True) if s == 4) == "1 2"
    assert set(body) == {3, 4, 5}


def test_recording_keeps_actions_and_writes_npz(tmp_path: Path) -> None:
    ref, none_path = run(make_engine(None), "e", 3)
    assert none_path is None
    eng = make_engine(tmp_path)
    got, path = run(eng, "c0/e", 3, key="job/suite/task01/none_original_init00")
    for a, b in zip(ref, got, strict=True):
        assert np.array_equal(a, b)  # bit-identical with recording on
    assert path is not None and path == str(tmp_path / "job/suite/task01/none_original_init00.npz")
    d = np.load(path)
    C = 3  # noqa: N806
    assert d["attn_mass"].shape == (C, NSTEP, NL, 7)
    np.testing.assert_allclose(d["attn_mass"].sum(-1), 1, atol=1e-5)  # every key is in one segment
    assert d["attn_head"].shape == (C, NL, H, 7)
    assert d["attn_img"].shape == (C, 3, 512) and d["attn_img"].dtype == np.float16
    assert d["exp_h"].shape == d["exp_mlp"].shape == (C, NSTEP, NL)
    assert d["vlm_h"].shape == d["vlm_mlp"].shape == (C, NL, 6)
    assert d["sig_h"].shape == (C, NIMG, 3)
    assert d["exp_vec"].shape == (C, NSTEP, 3, D_X) and d["exp_vec"].dtype == np.float16
    assert d["vlm_vec"].shape == (C, 3, 4, D_L)
    assert (d["t"] == [0, 4, 8]).all()
    np.testing.assert_array_equal(d["act_pred"], np.stack(got))
    ntok = d["seg_ntok"][0]
    assert ntok[0] == ntok[1] == A.IMG_TOKENS and ntok[2] == 1 and ntok[6] == CHUNK
    assert ntok[3] == len("put the bowl on the plate")
    assert all(np.isfinite(d[k]).all() for k in ("exp_h", "vlm_h", "sig_h", "exp_vec", "vlm_vec"))
    meta = json.loads(str(d["meta"]))
    assert meta["n_calls"] == C and meta["seed"] == 7 and meta["client"] == {"job_id": "j"}
    assert meta["record_key"] == "job/suite/task01/none_original_init00"
    assert not eng.recorder.buffers


def test_recorded_values_match_hook_free_replay(tmp_path: Path) -> None:
    """Same episode replayed in one batch (like attn_extract.py) gives the same per-call values."""
    eng = make_engine(tmp_path)
    _, path = run(eng, "e", 3)
    assert path is not None
    d = np.load(path)
    rec = eng.recorder
    batch = eng.l.pre(eng.wire_batch([obs(k) for k in range(3)], ["put the bowl on the plate"] * 3))
    torch.manual_seed(7)
    noise = torch.cat([torch.randn(1, CHUNK, 7) for _ in range(3)])
    rec.begin(batch)
    with torch.inference_mode():
        eng.l.policy.predict_action_chunk(batch, noise=noise)
    out = rec.hooks.stop()
    for k in ("attn_mass", "exp_h", "vlm_h", "sig_h", "vlm_mlp", "exp_mlp"):
        np.testing.assert_allclose(out[k], d[k], rtol=1e-4, atol=1e-5, err_msg=k)


def test_batched_drops_padding_rows(tmp_path: Path) -> None:
    eng = make_engine(tmp_path, batch=3)
    eng.reset("a", 1, "open the drawer", record_key="k/a")
    eng.reset("b", 2, "close the drawer", record_key="k/b")
    for k in range(2):
        eng.infer_batch(
            [("a", obs(k), "open the drawer", False), ("b", obs(k + 5), "close the drawer", False)]
        )
    pa, pb = eng.end("a"), eng.end("b")
    assert pa is not None and pb is not None
    da, db = np.load(pa), np.load(pb)
    assert da["attn_mass"].shape[0] == db["attn_mass"].shape[0] == 2
    assert da["seg_ntok"][0][3] == len("open the drawer") and db["seg_ntok"][0][3] == len("close the drawer")
    assert set(eng.recorder.buffers) == set()


def test_reset_without_end_drops_buffer(tmp_path: Path) -> None:
    eng = make_engine(tmp_path)
    eng.reset("e", 7, "x")
    eng.infer_one("e", obs(0), "x", False)
    eng.reset("e", 7, "x")  # restarted episode: old calls must not leak into the new npz
    eng.infer_one("e", obs(1), "x", False)
    p = eng.end("e")
    assert p is not None
    assert np.load(p)["attn_mass"].shape[0] == 1


def test_non_pi_policy_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="pi0-family"):
        A.Hooks(SimpleNamespace(model=nn.Linear(2, 2)))


# ---------------------------------------------------------------- review fixes (codex review of 8d29da28)


def test_truncated_prompt_skips_record_not_inference(tmp_path: Path) -> None:
    """A prompt cut before ';\\nAction' (tokenizer max length) must not fail the call: the record skips it."""
    long_task = "x" * (LT + 10)  # Pre truncates to LT tokens → no ';\nAction' in the prompt
    ref = make_engine(None)
    eng = make_engine(tmp_path)
    instr = ["short task", long_task, "short task"]
    for e in (ref, eng):
        e.reset("e", 7, "short task", record_key="k/e")
    a_ref = [ref.infer_one("e", obs(k), instr[k], False) for k in range(3)]
    a_rec = [eng.infer_one("e", obs(k), instr[k], False) for k in range(3)]
    for a, b in zip(a_ref, a_rec, strict=True):
        assert np.array_equal(a, b)
    p = eng.end("e")
    assert p is not None
    d = np.load(p)
    assert d["attn_mass"].shape[0] == 2 and list(d["call"]) == [0, 2] and list(d["t"]) == [0, 8]
    meta = json.loads(str(d["meta"]))
    assert meta["n_calls"] == 3 and meta["n_recorded"] == 2
    assert [s["call"] for s in meta["skipped_calls"]] == [1]
    assert "prompt segments" in meta["skipped_calls"][0]["reason"]


def test_record_key_cannot_escape_dir(tmp_path: Path) -> None:
    rec_dir, outside = tmp_path / "rec", tmp_path / "outside"
    rec_dir.mkdir()
    outside.mkdir()
    (rec_dir / "evil").symlink_to(outside, target_is_directory=True)
    eng = make_engine(rec_dir)
    _, p = run(eng, "e", 1, key="evil/x")
    assert p is None and not list(outside.iterdir())  # refused, nothing written outside
    # a symlink at the final path is replaced, never written through; temp files are unique and cleaned up
    (rec_dir / "ok").mkdir()
    victim = outside / "victim.npz"
    victim.write_bytes(b"keep")
    (rec_dir / "ok" / "y.npz").symlink_to(victim)
    _, p = run(eng, "e2", 1, key="ok/y")
    assert p == str((rec_dir / "ok" / "y.npz").resolve())
    assert victim.read_bytes() == b"keep" and not (rec_dir / "ok" / "y.npz").is_symlink()
    assert sorted(f.name for f in (rec_dir / "ok").iterdir()) == ["y.npz"]


def test_client_default_record_key_is_per_client() -> None:
    from bench.core.client import PolicyClient

    sent: list[dict] = []

    def fake_call(msg: dict) -> dict:
        sent.append(msg)
        return {"ok": True}

    for name in ("w1", "w2"):
        c = object.__new__(PolicyClient)
        c.name = name
        c._call = fake_call  # type: ignore[method-assign]
        c.reset("unit/i0", 1, "x")
    assert [m["record_key"] for m in sent] == ["w1/unit/i0", "w2/unit/i0"]


def _fake_pro_venvs(root: Path) -> Path:
    from test_testbed_select import png  # same fake sRGB textures as the render_fix test

    v = root / "venvs"
    (v / "libero_pro" / "bin").mkdir(parents=True)
    (v / "libero_pro" / "bin" / "python").symlink_to(sys.executable)
    src = v / "libero_pro_src"
    lib = src / "libero" / "libero"
    a = lib / "assets"
    (a / "textures").mkdir(parents=True)
    (a / "scenes").mkdir()
    (a / "textures" / "floor.png").write_bytes(png(True))
    (a / "scenes" / "libero_floor_base_style.xml").write_text('<texture file="../textures/floor.png"/>')
    (lib / "__init__.py").write_text("")
    (lib / "bddl_files").mkdir()
    (src / "libero" / "lifelong").mkdir()
    (src / "SC_VERSION").write_text("libero-pro-test\n")
    return v


def _snapshot(d: Path) -> dict:
    return {
        str(p): (p.is_symlink(), p.read_bytes() if p.is_file() and not p.is_symlink() else None)
        for p in d.rglob("*")
    }


def test_fixed_mirror_never_writes_the_original(tmp_path: Path) -> None:
    import subprocess  # nosec B404

    script = str(SHORTCUT / "bench" / "libero_pro" / "fixed_mirror.sh")
    v = _fake_pro_venvs(tmp_path)
    before = _snapshot(v)

    def sh(dst: Path) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", script, str(v), str(dst)], capture_output=True, text=True)  # nosec B603 B607

    ok = sh(tmp_path / "fix")
    assert ok.returncode == 0, ok.stderr
    assert "fix-" in ok.stdout and _snapshot(v) == before
    scene = tmp_path / "fix/src/libero/libero/assets/scenes/libero_floor_base_style.xml"
    assert "textures_nosrgb" in scene.read_text()
    # reused DST whose libero/libero is a symlink to the original → refused before apply
    bad = tmp_path / "bad" / "src" / "libero"
    bad.mkdir(parents=True)
    (bad / "libero").symlink_to(v / "libero_pro_src" / "libero" / "libero")
    r = sh(tmp_path / "bad")
    assert r.returncode != 0 and "refusing" in r.stderr
    # DST inside the original package → refused before anything is created
    r = sh(v / "libero_pro_src" / "libero" / "x")
    assert r.returncode != 0 and "refusing" in r.stderr
    assert _snapshot(v) == before
