"""T028 rich activation re-extraction: replay stored policy inputs through pi05 and keep *vectors*.

The replay inputs (per-call images, state, env step) written by attn_obs.py (T021) go back through pi05 with
the same noise as the original run (single exec: torch.manual_seed(episode seed), one draw per call in order),
like kubectl/libero90/_raw/attn_extract.py. Per episode we keep the T021/T023 features (norms, attention mass)
plus residual / K / V vectors, the action-token and denoising axes, and a 16-noise probe at 4 calls.
Keys, shapes and transforms: see README_KEYS below (also written to <out>/README.md by `readme`).

Subcommands (run in order; every step is resumable and never overwrites a finished file):
  manifest       build <out>/manifest.csv (episodes, labels, grasp/release calls, probe calls, pilot flag)
  extract        GPU. --phase pilot: raw token streams; --phase main: PCA projections (needs pca_bases.npz)
  fit-pca        CPU. PCA per (stream, layer) on the pilot fit episodes → <out>/pca_bases.npz + held-out
                 explained variance (<out>/pca_report.json)
  project-pilot  CPU. add the *_pc keys to the pilot (raw) files so every episode has the same keys
  weights        GPU/CPU. <out>/weights.npz (VLM k/v_proj + input norm, expert q/k/v_proj + adaRMS, action_out_proj)
  summary        GPU-h, disk, agreement with the earlier extraction from the extract logs

  python rich_extract.py manifest --out /outputs/VLA-shortcut-exps/rich_acts
  CUDA_VISIBLE_DEVICES=0 python rich_extract.py extract --out ... --phase pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

SHORTCUT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SHORTCUT), str(SHORTCUT / "serve"), str(SHORTCUT / "parallel")]

from serve.activations import (  # noqa: E402
    IMG_TOKENS,
    PAD,
    RICH_EXP_TOK_LAYERS,
    RICH_MLP_LAYERS,
    RICH_TOK_LAYERS,
    SEGMENTS,
    RichHooks,
    lang_segments,
)

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

K_ROOT = Path("/outputs/VLA-shortcut-exps/kubectl")
R_ROOT = Path("/workspace/VLA-shortcut-exps-kubectl/out/runs")
N_ACTION_STEPS = 10
N_NOISE = 16  # multi-noise probe: original noise + 15 fresh seeds
# stream → (layers, components kept in pca_bases.npz, samples = valid language tokens only)
PCA_STREAMS = {
    "vlm_tok": (RICH_TOK_LAYERS, 256, True),
    "exp_tok": (RICH_EXP_TOK_LAYERS, 128, False),
    "k_tok": (RICH_TOK_LAYERS, 256, True),
    "v_tok": (RICH_TOK_LAYERS, 256, True),
    "vlm_seg": (tuple(range(18)), 512, False),
}
PCA_DIMS = {"vlm_tok": 256, "exp_tok": 128}  # streams stored as projections in the main pass (default; --pca)
OLD_KEYS = ("attn_mass", "attn_head", "attn_img", "exp_h", "exp_mlp", "vlm_h", "vlm_mlp", "sig_h", "act_pred")
F16_MAX = 65000.0

L3 = K_ROOT / "libero-spatial-object-long/_raw/attn/obs"
GOAL = K_ROOT / "libero_goal_composition/_raw/attn/obs"
FAR = K_ROOT / "basket/far/_raw/attn/obs"
L90 = K_ROOT / "libero90/_raw/attn/obs"
# (set, job, obs dir, run dir, machine of the original run)
JOBS = [
    ("libero3", "ours-spat-custom-train10", L3, R_ROOT / "bench-pi05lib-libero3", "A100"),
    ("libero3", "ours-obj-custom-train10", L3, R_ROOT / "bench-pi05lib-libero3", "A100"),
    ("libero3", "ours-l10-custom-train10", L3, R_ROOT / "bench-pi05lib-libero3", "A100"),
    ("basket_a100", "ours-obj-custom-basket", L3, R_ROOT / "bench-pi05lib-libero3", "A100"),
    ("goal", "ours-goal-custom-train10", GOAL, R_ROOT / "bench-pi05lib-train10", "A100"),
    ("goal", "ours-goal-custom-cand31", GOAL, R_ROOT / "bench-pi05lib-cand31", "A100"),
    ("basket_far", "ours-obj-custom-train10", FAR, R_ROOT / "bench-pi05lib-basket_far", "4090"),
    ("basket_far", "ours-obj-custom-basket", FAR, R_ROOT / "bench-pi05lib-basket_far", "4090"),
    ("libero90", "ours-l90-custom-ctl10", L90, R_ROOT / "bench-pi05lib-libero90", "A100"),
]
L90_KINDS = ("part", "lscene")  # L10-scene tasks; 'other' only if budget remains (--l90-other)
PILOT_PER_SET = {"libero3": 12, "basket_a100": 8, "goal": 16, "basket_far": 9, "libero90": 15}  # 60 episodes
MANIFEST_COLS = [
    "episode", "set", "job", "machine", "replay_machine", "task", "variant", "init", "condition", "instruction",
    "success", "grasp_call", "release_call", "contact_call", "executed_task", "outcome", "n_calls",
    "probe_anchor", "probe_calls", "pilot", "pca_fit", "obs_path", "old_out_path", "steps_path", "path",
]  # fmt: skip


# ----------------------------------------------------------------------------------------------- manifest


def _read_csv(p: Path) -> list[dict[str, str]]:
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _int(x: Any) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _labels() -> dict[tuple[str, str], dict[str, Any]]:
    """(set/job, episode name) → labels from the T022/T025/T018 tables."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    ev = K_ROOT / "libero-spatial-object-long/8_phase_traj/events"
    jobs = {"S": "ours-spat-custom-train10", "O": "ours-obj-custom-train10", "L": "ours-l10-custom-train10"}
    for suite, name in (("S", "Spatial"), ("O", "Object"), ("L", "Long")):
        for r in _read_csv(ev / f"events_{name}.csv"):
            k = int(r["task"][1:]) - 1
            out["libero3/" + jobs[suite], f"ctl_t{k:02d}_train_i{int(r['init']):02d}"] = {
                "task": r["task"], "success": int(r["success"]), "grasp_call": _int(r["grasp_call"]),
                "release_call": _int(r["release_call"]), "contact_call": _int(r["contact_call"]),
                "executed_task": r["executed_training_tasks"], "outcome": r["judge_cat"],
            }  # fmt: skip
    for r in _read_csv(K_ROOT / "libero_goal_composition/8_phase_traj/events/events_Goal.csv"):
        if r["group"] == "train":
            key = "goal/ours-goal-custom-train10", f"ctl_t{int(r['task'][1:]) - 1:02d}_train_i{int(r['init']):02d}"
        else:
            key = "goal/ours-goal-custom-cand31", f"{r['task']}_i{int(r['init']):02d}"
        out[key] = {
            "task": r["task"], "success": int(r["success"]), "grasp_call": _int(r["grasp_call"]),
            "release_call": _int(r["release_call"]), "contact_call": _int(r["contact_call"]),
            "executed_task": r["executed_training_tasks"], "outcome": r["judge_cat"],
        }  # fmt: skip
    kinds = {
        ("near", "A100"): ("basket_a100/ours-obj-custom-basket", "bsk_near_t{t:02d}_i{i:02d}"),
        ("opp", "A100"): ("basket_a100/ours-obj-custom-basket", "bsk_opp_t{t:02d}_i{i:02d}"),
        ("ctl", "4090"): ("basket_far/ours-obj-custom-train10", "ctl_t{t:02d}_train_i{i:02d}"),
        ("far20", "4090"): ("basket_far/ours-obj-custom-basket", "bsk_c20y21_t{t:02d}_i{i:02d}"),
        ("farM", "4090"): ("basket_far/ours-obj-custom-basket", "bsk_c25y20_t{t:02d}_i{i:02d}"),
    }
    for r in _read_csv(K_ROOT / "basket/far/1_results/outcomes.csv"):
        km = kinds.get((r["kind"], r["machine"]))
        if km is None:
            continue
        t, i = int(r["task"]), int(r["init"])
        g, rel = _int(r["grasp_t"]), _int(r["release_t"])
        out[km[0], km[1].format(t=t, i=i)] = {
            "task": f"O{t + 1}", "success": int(r["success"]),
            "grasp_call": None if g is None else g // N_ACTION_STEPS,
            "release_call": None if rel is None else rel // N_ACTION_STEPS,
            "contact_call": None, "executed_task": "", "outcome": r["failure"],
        }  # fmt: skip
    for r in _read_csv(K_ROOT / "libero90/1_results/episodes.csv"):
        out["libero90/ours-l90-custom-ctl10", f"ctl_{r['task']}_{r['ep']}"] = {
            "task": r["task"], "success": int(r["success"] == "True"), "grasp_call": None, "release_call": None,
            "contact_call": None, "executed_task": r["reached_instead"], "outcome": r["outcome"],
            "kind": r["kind"],
        }  # fmt: skip
    return out


def _condition(set_: str, job: str, variant: str, kind: str | None) -> str:
    if set_ == "libero3":
        return {"spat": "spatial", "obj": "object", "l10": "long"}[job.split("-")[1]]
    if set_ == "basket_a100":
        return variant.split("_")[1]  # near / opp
    if set_ == "goal":
        return "train" if variant.startswith("ctl_") else "cand"
    if set_ == "basket_far":
        return "ctl4090" if variant.startswith("ctl_") else {"bsk_c20y21": "far20", "bsk_c25y20": "far25"}[variant[:10]]
    return kind or ""


def probe_calls(n_calls: int, grasp: int | None, contact: int | None) -> tuple[list[int], str]:
    """Multi-noise probe calls {0, g-2, g+3, last}; anchor g = grasp call, else first contact, else the middle."""
    if grasp is not None:
        g, anchor = grasp, "grasp"
    elif contact is not None:
        g, anchor = contact, "contact"
    else:
        g, anchor = n_calls // 2, "middle"
    calls = [0, g - 2, g + 3, n_calls - 1]
    return [min(max(c, 0), n_calls - 1) for c in calls], anchor


def build_manifest(out: Path, l90_other: bool = False, seed: int = 0) -> list[dict[str, Any]]:
    labels = _labels()
    rows = []
    for set_, job, obs_root, run_dir, machine in JOBS:
        jd = run_dir / "jobs" / job
        eps = {}
        for line in (jd / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            eps[f"{e['variant']}_i{int(e['init_idx']):02d}"] = e
        for f in sorted((obs_root / job).glob("*.npz")):
            name = f.stem
            lab = labels.get((f"{set_}/{job}", name))
            if lab is None:
                raise SystemExit(f"no labels for {set_}/{job}/{name}")
            if set_ == "libero90" and lab["kind"] not in L90_KINDS and not l90_other:
                continue
            e = eps[name]
            with np.load(f) as d:
                n_calls = len(d["t"])
            grasp = lab["grasp_call"]
            if grasp is None and set_ == "libero90" and e.get("first_grasp_t") is not None:
                grasp = int(e["first_grasp_t"]) // N_ACTION_STEPS
            succ = int(bool(e["success_new"]))
            if succ != lab["success"]:
                raise SystemExit(f"{set_}/{job}/{name}: success {succ} in episodes.jsonl vs {lab['success']} in tables")
            pc, anchor = probe_calls(n_calls, grasp, lab["contact_call"])
            rows.append({
                "episode": name, "set": set_, "job": job, "machine": machine, "replay_machine": "4090",
                "task": lab["task"], "variant": e["variant"], "init": int(e["init_idx"]),
                "condition": _condition(set_, job, e["variant"], lab.get("kind")), "instruction": e["instruction"],
                "success": succ, "grasp_call": grasp, "release_call": lab["release_call"],
                "contact_call": lab["contact_call"], "executed_task": lab["executed_task"], "outcome": lab["outcome"],
                "n_calls": n_calls, "probe_anchor": anchor, "probe_calls": ";".join(map(str, pc)),
                "pilot": 0, "pca_fit": 0, "obs_path": str(f), "old_out_path": str(f).replace("/attn/obs/", "/attn/out/"),
                "steps_path": str(jd / e["steps_file"]), "path": f"{set_}/{job}/{name}.npz",
            })  # fmt: skip
    # pilot: 1 init per task (distinct variants), stratified per set; every 5th pilot episode is held out of the fit
    rng = random.Random(seed)
    by_set: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_set[r["set"]][f"{r['job']}/{r['variant']}"].append(r)
    k = 0
    for set_, n in PILOT_PER_SET.items():
        variants = sorted(by_set[set_])
        # spread over the conditions: round-robin over condition groups
        groups: dict[str, list[str]] = defaultdict(list)
        for v in variants:
            groups[by_set[set_][v][0]["condition"]].append(v)
        for g in groups.values():
            rng.shuffle(g)
        order = [v for tup in zip(*[groups[c] + [None] * (len(variants) - len(groups[c])) for c in sorted(groups)], strict=True) for v in tup if v]  # noqa: E501
        for v in order[:n]:
            r = rng.choice(by_set[set_][v])
            r["pilot"] = 1
            r["pca_fit"] = int(k % 5 != 4)
            k += 1
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.csv"
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, MANIFEST_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r[c] is None else r[c]) for c in MANIFEST_COLS})
    os.replace(tmp, path)
    return rows


def read_manifest(out: Path) -> list[dict[str, str]]:
    return _read_csv(out / "manifest.csv")


# ----------------------------------------------------------------------------------------------- extraction


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)  # lossless zlib: -11% overall (zero-padded token arrays -30%)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def f16(name: str, x: np.ndarray | torch.Tensor, absmax: dict[str, float]) -> np.ndarray:
    a = x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, np.float32)
    m = float(np.abs(a).max()) if a.size else 0.0
    absmax[name] = max(absmax.get(name, 0.0), m)
    if m > F16_MAX:  # Gemma's massive activations: keep float32 for this array (load every key as f32)
        return a
    return a.astype(np.float16)


class Bases:
    """PCA bases on the device: stream → (mean (nL, D), first k components (nL, k, D))."""

    def __init__(self, path: Path, dev: str, dims: dict[str, int]):
        z = np.load(path)
        self.b = {}
        for s, k in dims.items():
            layers = PCA_STREAMS[s][0]
            mean = np.stack([z[f"{s}/L{li}/mean"] for li in layers])
            comp = np.stack([z[f"{s}/L{li}/components"][:k] for li in layers])
            if comp.shape[1] != k:
                raise SystemExit(f"pca_bases.npz has {comp.shape[1]} components for {s}, asked for {k}")
            self.b[s] = (torch.from_numpy(mean).float().to(dev), torch.from_numpy(comp).float().to(dev))

    def project(self, stream: str, x: torch.Tensor) -> torch.Tensor:
        """x: (B, nL, ..., D) → (B, nL, ..., k)."""
        mean, comp = self.b[stream]
        shape = [1, mean.shape[0]] + [1] * (x.dim() - 3) + [mean.shape[1]]
        xc = x - mean.view(shape)
        return torch.einsum("bl...d,lkd->bl...k", xc, comp)


def prompt_layout(tok: Any, tokens: torch.Tensor, masks: torch.Tensor, n_img: int, chunk: int):
    """Segment id per key (C, K) like attn_extract.py + language positions / ids / segments (C, N)."""
    C, Lt = tokens.shape  # noqa: N806
    K = n_img * IMG_TOKENS + Lt + chunk  # noqa: N806
    seg = torch.full((C, K), PAD, dtype=torch.long)
    for c in range(min(n_img, 2)):
        seg[:, c * IMG_TOKENS : (c + 1) * IMG_TOKENS] = c
    n_valid = masks.sum(1)
    N = int(n_valid.max())  # noqa: N806
    lang_pos = torch.zeros(C, N, dtype=torch.long)
    lang_valid = torch.zeros(C, N, dtype=torch.bool)
    lang_ids = np.full((C, N), -1, np.int32)
    lang_seg = np.full((C, N), PAD, np.int8)
    ntok = np.zeros((C, len(SEGMENTS)), np.int32)
    for b in range(C):
        ids = tokens[b][masks[b]].tolist()
        ls = lang_segments(tok, ids)
        pos = torch.nonzero(masks[b].cpu()).flatten() + n_img * IMG_TOKENS
        seg[b, pos] = torch.tensor(ls)
        seg[b, n_img * IMG_TOKENS + Lt :] = 6
        ntok[b] = np.bincount(seg[b].numpy(), minlength=PAD + 1)[:PAD]
        n = len(ids)
        lang_pos[b, :n], lang_valid[b, :n] = pos, True
        lang_ids[b, :n], lang_seg[b, :n] = ids, ls
    return seg, ntok, lang_pos, lang_valid, lang_ids, lang_seg


def _index_batch(batch: dict[str, Any], idx: list[int], C: int) -> dict[str, Any]:  # noqa: N803
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == C:
            out[k] = v[torch.as_tensor(idx, device=v.device)]
        elif isinstance(v, list) and len(v) == C:
            out[k] = [v[i] for i in idx]
        else:
            out[k] = v
    return out


class PathRecorder:
    """Instance-level wraps of model.denoise_step / model.sample_actions: x_k (k = 0..n) and v_k per call."""

    def __init__(self, model: Any):
        self.model = model
        self.on = False
        self._denoise, self._sample = model.denoise_step, model.sample_actions

        def denoise_step(*args: Any, **kwargs: Any) -> torch.Tensor:
            v = self._denoise(*args, **kwargs)
            if self.on:
                x = kwargs["x_t"] if "x_t" in kwargs else args[2]
                self.xs.append(x.float().clone())
                self.vs.append(v.float().clone())
            return v

        def sample_actions(*args: Any, **kwargs: Any) -> torch.Tensor:
            out = self._sample(*args, **kwargs)
            if self.on:
                self.final = out.float().clone()
            return out

        model.denoise_step = denoise_step
        model.sample_actions = sample_actions

    def start(self) -> None:
        self.xs: list[torch.Tensor] = []
        self.vs: list[torch.Tensor] = []
        self.final: torch.Tensor | None = None
        self.on = True

    def stop(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.on = False
        if self.final is None:
            raise RuntimeError("sample_actions was not called")
        return torch.stack([*self.xs, self.final], 1), torch.stack(self.vs, 1)


def multi_noise(
    policy: Any, batch: dict[str, Any], calls: list[int], C: int, base_noise: torch.Tensor, seed: int,  # noqa: N803
    exp_last_layer: Any, n_noise: int = N_NOISE,
) -> dict[str, np.ndarray]:  # fmt: skip
    """At each probe call: one prefix pass (KV cache), then n_noise denoising runs from different noise.
    Row 0 = the original noise of that call (single-exec draw); rows 1.. = torch.Generator(seed*10^5 + call*100 + r).
    Mirrors PI05Pytorch.sample_actions with the cache repeated over the noise rows."""
    from lerobot.policies.pi05.modeling_pi05 import euler_integrate, make_att_2d_masks

    model = policy.model
    cfg = policy.config
    sub = _index_batch(batch, calls, C)
    images, img_masks = policy._preprocess_images(sub)
    states, state_masks = policy._prepare_memory_states(sub)
    tokens, masks = sub[OBS_LANGUAGE_TOKENS], sub[OBS_LANGUAGE_ATTENTION_MASK]
    dev = tokens.device
    seeds = np.full((len(calls), n_noise), -1, np.int64)
    noise = []
    for j, c in enumerate(calls):
        noise.append(base_noise[c : c + 1].to(dev))
        for r in range(1, n_noise):
            seeds[j, r] = int(seed) * 100000 + c * 100 + r
            g = torch.Generator().manual_seed(int(seeds[j, r]))
            noise.append(torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=g).to(dev))
    noise_t = torch.cat(noise).float()
    last: list[torch.Tensor] = []
    h = exp_last_layer.register_forward_hook(
        lambda _m, _i, o: last.append((o[0] if isinstance(o, tuple) else o)[:, -cfg.chunk_size :].float().mean(1))
    )
    try:
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        att2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        mask_dtype = prefix_embs.dtype if getattr(model, "use_typed_attention_masks", False) else None
        m4 = model._prepare_attention_masks_4d(att2d, dtype=mask_dtype)
        lm = model.paligemma_with_expert.paligemma.model.language_model
        if hasattr(lm, "config"):
            lm.config._attn_implementation = "eager"  # noqa: SLF001
        _, past = model.paligemma_with_expert.forward(
            attention_mask=m4, position_ids=pos, past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True
        )
        past.batch_repeat_interleave(n_noise)
        ppm = prefix_pad_masks.repeat_interleave(n_noise, 0)
        x = euler_integrate(
            lambda x_t, t: model.denoise_step(prefix_pad_masks=ppm, past_key_values=past, x_t=x_t, timestep=t),
            noise_t,
            cfg.num_inference_steps,
            precompute_times=bool(getattr(model, "precompute_denoise_times", False)),
        )
    finally:
        h.remove()
    n = len(calls)
    return {
        "mn_calls": np.asarray(calls, np.int32),
        "mn_seeds": seeds,
        "mn_chunk": x.float().cpu().numpy().reshape(n, n_noise, cfg.chunk_size, -1),
        "mn_exp_last": last[-1].cpu().numpy().reshape(n, n_noise, -1),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def load_obs(row: dict[str, Any]) -> dict[str, np.ndarray]:
    """Replay input of one episode, read fully (runs in a prefetch thread during extraction)."""
    with np.load(row["obs_path"]) as z:
        return {k: z[k] for k in z.files}


def run_episode(
    row: dict[str, str], L: Any, hooks: RichHooks, path_rec: PathRecorder, tok: Any, dev: str,  # noqa: N803
    bases: Bases | None, raw: bool, max_batch: int, dims: dict[str, int] | None = None,
    mlp_keep: tuple[int, ...] | None = None, keep_vlm_vec: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:  # fmt: skip
    """raw: keep every stream raw (pilot / smoke). bases + dims: streams in dims are stored as `<stream>_pc`
    (first dims[s] PCA components) and, unless raw, not raw. mlp_keep: expert MLP layers kept in exp_mlp_mean."""
    from serve.engine import LIBERO_IMAGE_MAP, Engine

    dims = dict(PCA_DIMS if dims is None else dims)
    mlp_keep = tuple(hooks.mlp_layers if mlp_keep is None else mlp_keep)
    mlp_idx = [hooks.mlp_layers.index(i) for i in mlp_keep]

    d = row.get("_obs") or load_obs(row)
    C = len(d["t"])  # noqa: N806
    instruction = str(d["instruction"])
    obs = [{"image": d["image"][k], "image2": d["image2"][k], "state": d["state"][k]} for k in range(C)]
    batch = L.pre(Engine.wire_batch(SimpleNamespace(image_map=LIBERO_IMAGE_MAP), obs, [instruction] * C))
    policy, model = L.policy, L.policy.model
    cfg = policy.config
    torch.manual_seed(int(d["seed"]))
    noise = torch.cat([model.sample_noise((1, cfg.chunk_size, cfg.max_action_dim), dev) for _ in range(C)])
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
    n_img = len(cfg.image_features)
    seg, ntok, lang_pos, lang_valid, lang_ids, lang_seg = prompt_layout(tok, tokens, masks, n_img, cfg.chunk_size)
    seg, lang_pos, lang_valid = seg.to(dev), lang_pos.to(dev), lang_valid.to(dev)
    n_steps = int(cfg.num_inference_steps)
    absmax: dict[str, float] = {}
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    chunks = []
    n_sub = -(-C // max_batch)
    size = -(-C // n_sub)  # even sub-batches (52 calls, max 32 → 26 + 26)
    timing = {"main": 0.0, "probe": 0.0}
    t0 = time.time()
    for lo in range(0, C, size):
        idx = list(range(lo, min(C, lo + size)))
        sub = _index_batch(batch, idx, C) if len(idx) < C else batch
        hooks.start_rich(seg[idx], n_img, n_steps, cfg.chunk_size, lang_pos[idx], lang_valid[idx])
        path_rec.start()
        try:
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(sub, noise=noise[idx])
        except BaseException:
            hooks.active = False
            path_rec.on = False
            raise
        light, rich = hooks.stop_rich()
        xs, vs = path_rec.stop()
        if abs(float(light["attn_mass"].sum(-1).mean()) - 1) > 1e-2:
            raise RuntimeError(f"attention mass per query does not sum to 1: {float(light['attn_mass'].sum(-1).mean())}")
        if not torch.equal(xs[:, -1, :, : chunk.shape[-1]], chunk.float()):
            raise RuntimeError("recorded x_N differs from the predicted chunk")
        chunks.append(chunk[:, :N_ACTION_STEPS])
        for k, v in light.items():
            if k != "vlm_vec" or keep_vlm_vec:
                parts[k].append(v)
        rich["exp_mlp_mean"] = rich["exp_mlp_mean"][:, :, mlp_idx]
        for s in ("vlm_seg", "k_seg", "v_seg", "exp_mean", "exp_mlp_mean", "k_tok", "v_tok", "vlm_tok", "exp_tok"):
            if raw or s not in dims:
                parts[s].append(f16(s, rich[s], absmax))
            if bases is not None and s in dims:
                pc = bases.project(s, rich[s])
                if PCA_STREAMS[s][2]:  # language tokens: zeros at the padding positions
                    pc = pc * lang_valid[idx][:, None, :, None]
                parts[f"{s}_pc"].append(f16(f"{s}_pc", pc, absmax))
        parts["den_x"].append(xs.cpu().numpy())
        parts["den_v"].append(vs.cpu().numpy())
        del rich, xs, vs
    out = {k: np.concatenate(v) for k, v in parts.items()}  # f16 + f32 parts (overflow in one sub-batch) give f32
    info_f32 = sorted(k for k, v in out.items() if v.dtype == np.float32 and absmax.get(k, 0) > F16_MAX)
    out["attn_img"] = out["attn_img"].astype(np.float16)
    out["exp_vec"] = out["exp_vec"].astype(np.float16)
    timing["main"] = time.time() - t0
    chunk = torch.cat(chunks)
    act = Engine._post(SimpleNamespace(l=L), chunk)
    out.update(
        seg_ntok=ntok, act_pred=act, t=d["t"], lang_ids=lang_ids, lang_seg=lang_seg, lang_valid=lang_valid.cpu().numpy()
    )
    calls = [int(c) for c in row["probe_calls"].split(";")]
    t0 = time.time()
    with torch.inference_mode():
        mn = multi_noise(policy, batch, calls, C, noise, int(d["seed"]), hooks.exp_layers[-1])
    timing["probe"] = time.time() - t0
    mn["mn_exp_last"] = f16("mn_exp_last", mn["mn_exp_last"], absmax)
    mn_act = Engine._post(SimpleNamespace(l=L), torch.from_numpy(mn["mn_chunk"].reshape(-1, cfg.chunk_size, mn["mn_chunk"].shape[-1])[..., :7]).to(dev))  # noqa: E501
    mn["mn_act"] = mn_act.reshape(len(calls), N_NOISE, cfg.chunk_size, -1)
    out.update(mn)
    # checks: probe row 0 reproduces the main pass; recorded actions; earlier extraction
    mn_self = float(np.abs(mn["mn_chunk"][:, 0] - out["den_x"][calls, -1]).max())
    info: dict[str, Any] = {
        "calls": C, "n_lang": int(lang_ids.shape[1]), "mn_self_diff": mn_self, "f32_keys": info_f32,
        "sec_main": round(timing["main"], 2), "sec_probe": round(timing["probe"], 2),
    }  # fmt: skip
    st = np.load(row["steps_path"])["action"]
    ref = np.stack([st[k : k + N_ACTION_STEPS] for k in d["t"]])
    info["act_max_diff_recorded"] = float(np.abs(act - ref).max())
    info["act_frac_close_recorded"] = float((np.abs(act - ref).max(-1) < 0.05).mean())
    old_p = Path(row["old_out_path"])
    if old_p.exists():
        old = np.load(old_p)
        # old attn_img also lists the empty camera's 256 patches (C, 3, 768); T023 keeps the two real ones
        info["r_old"] = {k: _pearson(old[k][..., : out[k].shape[-1]], out[k]) for k in OLD_KEYS if k in old}
        info["act_max_diff_old"] = float(np.abs(old["act_pred"] - act).max())
    info["absmax"] = {k: round(v, 1) for k, v in absmax.items()}
    meta = {
        "episode": row["episode"], "set": row["set"], "job": row["job"], "instruction": instruction,
        "seed": int(d["seed"]), "segments": SEGMENTS, "tok_layers": list(hooks.tok_layers),
        "exp_tok_layers": list(hooks.exp_tok_layers), "steps": list(hooks.steps), "mlp_layers": list(mlp_keep),
        "pca_dims": dims if bases is not None else {}, "raw": raw, "n_action_steps": N_ACTION_STEPS, "n_noise": N_NOISE,
        "probe_anchor": row["probe_anchor"], "replay_machine": "VGI-102 RTX 4090", "info": info,
    }  # fmt: skip
    out["meta"] = np.asarray(json.dumps(meta, default=str))
    return out, info


def load_model(dev: str) -> tuple[Any, Any]:
    from serve.engine import load_any
    from transformers import AutoTokenizer

    L = load_any(os.environ.get("SC_CKPT", "lerobot/pi05_libero_finetuned"), None, "libero_goal", N_ACTION_STEPS, dev)  # noqa: N806
    tok = AutoTokenizer.from_pretrained(os.environ["SC_TOKENIZER_DIR"])
    return L, tok


def parse_dims(spec: str | None) -> dict[str, int]:
    """'vlm_tok:256,exp_tok:128,k_tok:64' → dict; None → PCA_DIMS."""
    if not spec:
        return dict(PCA_DIMS)
    out = {}
    for item in spec.split(","):
        s, k = item.split(":")
        if s not in PCA_STREAMS:
            raise SystemExit(f"--pca: unknown stream {s!r} (have {list(PCA_STREAMS)})")
        out[s] = int(k)
    for s in PCA_DIMS:
        if s not in out:
            raise SystemExit(f"--pca must keep {s} (its raw form is never stored in the main pass)")
    return out


def main_config(a: argparse.Namespace) -> dict[str, Any]:
    mlp = tuple(range(18)) if a.mlp_layers == "all" else tuple(int(x) for x in a.mlp_layers.split(","))
    return {"pca": parse_dims(a.pca), "mlp_layers": list(mlp), "keep_vlm_vec": bool(a.keep_vlm_vec)}


def check_config(out: Path, cfg: dict[str, Any]) -> None:
    """The main-pass key config is fixed at the first run (out/config.json); a resume must use the same."""
    p = out / "config.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old != cfg:
            raise SystemExit(f"config differs from {p}: {old} vs {cfg}")
    else:
        p.write_text(json.dumps(cfg, indent=1))


def cmd_extract(a: argparse.Namespace) -> None:
    out = Path(a.out)
    rows = read_manifest(out)
    if a.phase == "pilot":
        rows = [r for r in rows if r["pilot"] == "1"]
    elif a.phase == "main":
        rows = [r for r in rows if r["pilot"] != "1"]
    if a.sets:
        rows = [r for r in rows if r["set"] in a.sets.split(",")]
    if a.episodes:
        want = set(a.episodes.split(","))
        rows = [r for r in rows if r["episode"] in want or r["path"].removesuffix(".npz") in want]
    if a.limit:
        rows = rows[: a.limit]
    dst = Path(a.dst) if a.dst else out
    todo = [r for r in rows if not (dst / r["path"]).exists()]
    raw = a.phase in ("pilot", "smoke")
    cfg = main_config(a)
    if a.phase == "main":
        if not (out / "pca_bases.npz").exists():
            raise SystemExit(f"{out / 'pca_bases.npz'} missing: run fit-pca first")
        check_config(out, cfg)
    else:  # pilot / smoke: everything raw, all 18 MLP layers, T023 vlm_vec kept
        cfg = {"pca": {}, "mlp_layers": list(RICH_MLP_LAYERS), "keep_vlm_vec": True}
    dev = "cuda"
    tic = time.time()
    L, tok = load_model(dev)  # noqa: N806
    bases = Bases(out / "pca_bases.npz", dev, cfg["pca"]) if a.phase == "main" else None
    hooks = RichHooks(L.policy)
    path_rec = PathRecorder(L.policy.model)
    print(f"[rich] {a.phase}: {len(todo)}/{len(rows)} episodes to do; load {time.time() - tic:.0f}s; config {cfg}",
          flush=True)  # fmt: skip
    log_p = dst / f"extract_{a.phase}.jsonl"
    log_p.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    io = ThreadPoolExecutor(2)  # prefetch the next replay input / write the previous episode while the GPU runs
    nxt = io.submit(load_obs, todo[0]) if todo else None
    pending: Future | None = None

    def finish(r: dict[str, Any], info: dict[str, Any], t0: float, t1: float, i: int, log: Any) -> None:
        mb = (dst / r["path"]).stat().st_size / 2**20
        line = {"episode": r["episode"], "set": r["set"], "job": r["job"], **info,
                "sec_gpu": round(t1 - t0, 2), "sec_total": round(time.time() - t0, 2), "mb": round(mb, 1),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
        log.write(json.dumps(line) + "\n")
        log.flush()
        r_old = info.get("r_old", {})
        print(f"[rich] {i}/{len(todo)} {r['set']}/{r['episode']} {line['sec_gpu']}s "
              f"(main {info['sec_main']} probe {info['sec_probe']}) {mb:.0f}MB "
              f"act_rec={info['act_max_diff_recorded']:.3g} mn_self={info['mn_self_diff']:.2g} "
              f"r_old_min={min(r_old.values()) if r_old else float('nan'):.5f}", flush=True)  # fmt: skip

    with open(log_p, "a") as log:
        for i, r in enumerate(todo, 1):
            if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
                print(f"[rich] stop: --max-hours {a.max_hours} reached", flush=True)
                break
            t0 = time.time()
            r = {**r, "_obs": nxt.result()}
            nxt = io.submit(load_obs, todo[i]) if i < len(todo) else None
            arrays, info = run_episode(
                r, L, hooks, path_rec, tok, dev, bases, raw, a.max_batch,
                dims=cfg["pca"], mlp_keep=tuple(cfg["mlp_layers"]), keep_vlm_vec=cfg["keep_vlm_vec"],
            )  # fmt: skip
            t1 = time.time()
            if pending is not None:
                pending.result()  # one write in flight at most (bounded memory, errors surface)
            r.pop("_obs")
            pending = io.submit(lambda r=r, arrays=arrays, info=info, t0=t0, t1=t1, i=i: (
                save_npz(dst / r["path"], arrays), finish(r, info, t0, t1, i, log)))  # fmt: skip
            if a.peak_mem and i == 1:
                print(f"[rich] peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if pending is not None:
            pending.result()
    io.shutdown()


# ----------------------------------------------------------------------------------------------- PCA


def _rows(x: np.ndarray, j: int, valid: np.ndarray | None) -> np.ndarray:
    """Samples (n, D) of layer slot j: valid language tokens (x: (C, nL, N, D)), or every leading index
    (exp_tok (C, nL, S, T, D): call × step × token; vlm_seg (C, L, 6, D): call × segment)."""
    xl = x[:, j].astype(np.float32)
    if valid is not None:
        return xl[valid]
    return xl.reshape(-1, xl.shape[-1])


def _stream_rows(z: Any, s: str, j: int) -> np.ndarray:
    return _rows(z[s], j, z["lang_valid"] if PCA_STREAMS[s][2] else None)


def cmd_fit_pca(a: argparse.Namespace) -> None:
    """Exact PCA from streamed covariances (float64) on the pilot fit episodes; held-out curves on the rest."""
    out = Path(a.out)
    rows = [r for r in read_manifest(out) if r["pilot"] == "1"]
    fit = [r for r in rows if r["pca_fit"] == "1"]
    held = [r for r in rows if r["pca_fit"] != "1"]
    torch.set_num_threads(a.threads)
    acc: dict[tuple[str, int], list[Any]] = {}
    for n_f, r in enumerate(fit, 1):
        z = np.load(out / r["path"])
        for s, (layers, _k, _m) in PCA_STREAMS.items():
            for j, li in enumerate(layers):
                X = torch.from_numpy(_stream_rows(z, s, j)).double()  # noqa: N806
                d = X.shape[1]
                a_ = acc.setdefault((s, li), [0, torch.zeros(d, dtype=torch.float64), torch.zeros(d, d, dtype=torch.float64)])  # noqa: E501
                a_[0] += X.shape[0]
                a_[1] += X.sum(0)
                a_[2] += X.T @ X
        print(f"[fit-pca] {n_f}/{len(fit)} {r['path']}", flush=True)
    bases: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {"fit_episodes": [r["path"] for r in fit], "heldout_episodes": [r["path"] for r in held]}
    for (s, li), (n, sx, sxx) in acc.items():
        mean = sx / n
        cov = (sxx - n * torch.outer(mean, mean)) / (n - 1)
        ev, evec = torch.linalg.eigh(cov)
        ev, evec = ev.flip(0).clamp(min=0), evec.flip(1)
        k = min(PCA_STREAMS[s][1], len(ev))
        bases[f"{s}/L{li}/mean"] = mean.float().numpy()
        bases[f"{s}/L{li}/components"] = evec[:, :k].T.contiguous().float().numpy()  # (k, D), rows = PCs
        bases[f"{s}/L{li}/explained_variance"] = ev[:k].float().numpy()
        bases[f"{s}/L{li}/total_variance"] = np.asarray(float(ev.sum()), np.float64)
        bases[f"{s}/L{li}/n_samples"] = np.asarray(n, np.int64)
        report.setdefault(s, {})[f"L{li}"] = {
            "n_fit": int(n), "fit_ratio": {str(m): round(float(ev[:m].sum() / ev.sum()), 4) for m in _ms(k)},
        }  # fmt: skip
    # held-out: share of the variance (about the fit mean) captured by the first m components
    for s, (layers, _k, _m) in PCA_STREAMS.items():
        for j, li in enumerate(layers):
            mean = torch.from_numpy(bases[f"{s}/L{li}/mean"]).double()
            comp = torch.from_numpy(bases[f"{s}/L{li}/components"]).double()
            ms = _ms(comp.shape[0])
            tot, cap = 0.0, np.zeros(len(ms))
            for r in held:
                X = torch.from_numpy(_stream_rows(np.load(out / r["path"]), s, j)).double() - mean  # noqa: N806
                P = X @ comp.T  # noqa: N806
                tot += float((X * X).sum())
                cap += np.array([float((P[:, :m] ** 2).sum()) for m in ms])
            report[s][f"L{li}"]["heldout_ratio"] = {str(m): round(float(c / tot), 4) for m, c in zip(ms, cap, strict=True)}
    tmp = out / "pca_bases.tmp.npz"
    np.savez(tmp, **bases)
    os.replace(tmp, out / "pca_bases.npz")
    (out / "pca_report.json").write_text(json.dumps(report, indent=1))
    for s in PCA_STREAMS:
        print(s, {li: v["heldout_ratio"] for li, v in report[s].items()}, flush=True)


def _ms(k: int) -> list[int]:
    return [m for m in (8, 16, 32, 64, 128, 256) if m < k] + [k]


def cmd_project_pilot(a: argparse.Namespace) -> None:
    """Add the main-pass `<stream>_pc` keys (config.json) to the pilot files, which keep their raw streams."""
    out = Path(a.out)
    cfg = json.loads((out / "config.json").read_text())
    z = np.load(out / "pca_bases.npz")
    for r in read_manifest(out):
        if r["pilot"] != "1":
            continue
        p = out / r["path"]
        d = dict(np.load(p))
        if all(f"{s}_pc" in d for s in cfg["pca"]):
            continue
        absmax: dict[str, float] = {}
        for s, k in cfg["pca"].items():
            layers = PCA_STREAMS[s][0]
            mean = np.stack([z[f"{s}/L{li}/mean"] for li in layers])
            comp = np.stack([z[f"{s}/L{li}/components"][:k] for li in layers])
            x = torch.from_numpy(d[s].astype(np.float32))
            shape = [1, len(layers)] + [1] * (x.dim() - 3) + [mean.shape[1]]
            pc = torch.einsum("bl...d,lkd->bl...k", x - torch.from_numpy(mean).view(shape), torch.from_numpy(comp))
            if PCA_STREAMS[s][2]:
                pc = pc * torch.from_numpy(d["lang_valid"])[:, None, :, None]
            d[f"{s}_pc"] = f16(f"{s}_pc", pc, absmax)
        meta = json.loads(str(d["meta"]))
        meta["pca_dims"] = cfg["pca"]
        d["meta"] = np.asarray(json.dumps(meta, default=str))
        save_npz(p, d)
        print(f"[project-pilot] {r['path']} {absmax}", flush=True)


# ----------------------------------------------------------------------------------------------- weights


def cmd_weights(a: argparse.Namespace) -> None:
    out = Path(a.out)
    p = out / "weights.npz"
    if p.exists():
        print(f"{p} exists")
        return
    L, _tok = load_model("cuda" if torch.cuda.is_available() else "cpu")  # noqa: N806
    m = L.policy.model
    pwe = m.paligemma_with_expert
    vlm = pwe.paligemma.model.language_model.layers
    exp = pwe.gemma_expert.model.layers
    w: dict[str, np.ndarray] = {}

    def put(k: str, t: torch.Tensor) -> None:
        w[k] = t.detach().float().cpu().numpy()

    for i, layer in enumerate(vlm):
        put(f"vlm/L{i}/k_proj", layer.self_attn.k_proj.weight)  # (256, 2048): K = W x (no bias)
        put(f"vlm/L{i}/v_proj", layer.self_attn.v_proj.weight)
        put(f"vlm/L{i}/input_layernorm", layer.input_layernorm.weight)  # Gemma RMSNorm: x̂ (1 + w)
    for i, layer in enumerate(exp):
        put(f"exp/L{i}/q_proj", layer.self_attn.q_proj.weight)  # (2048, 1024)
        put(f"exp/L{i}/k_proj", layer.self_attn.k_proj.weight)
        put(f"exp/L{i}/v_proj", layer.self_attn.v_proj.weight)
        n = layer.input_layernorm
        put(f"exp/L{i}/input_layernorm_dense_w", n.dense.weight)  # adaRMS: (scale, shift, gate) = dense(cond)
        put(f"exp/L{i}/input_layernorm_dense_b", n.dense.bias)
    put("action_out_proj_w", m.action_out_proj.weight)  # (32, 1024)
    put("action_out_proj_b", m.action_out_proj.bias)
    w["meta"] = np.asarray(json.dumps({
        "ckpt": os.environ.get("SC_CKPT", "lerobot/pi05_libero_finetuned"),
        "dtype_in_model": str(vlm[0].self_attn.k_proj.weight.dtype),
        "rms_eps_vlm": float(vlm[0].input_layernorm.eps), "rms_eps_exp": float(exp[0].input_layernorm.eps),
        "vlm_rope_theta": float(getattr(pwe.paligemma.model.language_model.config, "rope_theta", 10000.0)),
        "note": "PyTorch Linear layout (out, in): y = x @ W.T (+ b). adaRMS: x̂ = x/rms(x); "
                "scale, shift, gate = chunk(dense(cond), 3); y = x̂ * (1 + scale) + shift; gate scales the residual.",
    }))  # fmt: skip
    tmp = out / "weights.tmp.npz"
    np.savez(tmp, **w)
    os.replace(tmp, p)
    print(f"{p}: {len(w)} arrays, {p.stat().st_size / 2**20:.0f} MB")


# ----------------------------------------------------------------------------------------------- summary


def cmd_summary(a: argparse.Namespace) -> None:
    out = Path(a.out)
    lines = []
    for p in sorted(out.glob("extract_*.jsonl")):
        lines += [json.loads(x) | {"phase": p.stem[8:]} for x in p.read_text().splitlines()]
    if not lines:
        print("no logs")
        return
    sec = sum(x["sec_total"] for x in lines)
    du = sum(f.stat().st_size for f in out.rglob("*.npz"))
    r_old = defaultdict(list)
    for x in lines:
        for k, v in x.get("r_old", {}).items():
            r_old[k].append(v)
    rec = np.array([x["act_max_diff_recorded"] for x in lines])
    close = np.array([x["act_frac_close_recorded"] for x in lines])
    old = np.array([x.get("act_max_diff_old", np.nan) for x in lines])
    s = {
        "episodes": len(lines), "gpu_h_episodes": round(sec / 3600, 3), "disk_gb": round(du / 2**30, 2),
        "mb_per_ep": round(float(np.mean([x["mb"] for x in lines])), 1),
        "sec_per_ep": round(sec / len(lines), 2),
        "r_old_min": {k: round(float(np.nanmin(v)), 6) for k, v in r_old.items()},
        "r_old_median": {k: round(float(np.nanmedian(v)), 6) for k, v in r_old.items()},
        "act_max_diff_recorded": {"median": float(np.median(rec)), "p95": float(np.percentile(rec, 95)), "max": float(rec.max())},
        "act_calls_within_0.05_of_recorded": round(float(close.mean()), 4),
        "act_max_diff_old_extraction": {"median": float(np.nanmedian(old)), "max": float(np.nanmax(old))},
        "mn_self_diff_max": float(max(x["mn_self_diff"] for x in lines)),
        "absmax": {k: max(x["absmax"].get(k, 0) for x in lines) for k in lines[-1]["absmax"]},
    }  # fmt: skip
    print(json.dumps(s, indent=1))
    if a.write:
        (out / "summary.json").write_text(json.dumps(s, indent=1))


def main() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))  # task rule: ≤ 8 CPU workers
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--l90-other", action="store_true")
    p = sp.add_parser("extract")
    p.add_argument("--out", required=True)
    p.add_argument("--dst", default=None, help="write episodes here instead of --out (smoke)")
    p.add_argument("--phase", choices=["smoke", "pilot", "main"], required=True)
    p.add_argument("--sets", default=None)
    p.add_argument("--episodes", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--peak-mem", action="store_true")
    p.add_argument("--pca", default=None, help="main pass: stream:k list, default vlm_tok:256,exp_tok:128")
    p.add_argument("--mlp-layers", default="all", help="main pass: expert MLP layers kept ('all' or 2,5,8)")
    p.add_argument("--keep-vlm-vec", action="store_true", help="main pass: keep T023 vlm_vec (in vlm_seg)")
    p = sp.add_parser("fit-pca")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=8)
    p = sp.add_parser("project-pilot")
    p.add_argument("--out", required=True)
    p = sp.add_parser("weights")
    p.add_argument("--out", required=True)
    p = sp.add_parser("summary")
    p.add_argument("--out", required=True)
    p.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.cmd == "manifest":
        rows = build_manifest(Path(a.out), a.l90_other)
        print(f"{len(rows)} episodes, pilot {sum(r['pilot'] for r in rows)}; by set:",
              {s: sum(r["set"] == s for r in rows) for s in PILOT_PER_SET})  # fmt: skip
    elif a.cmd == "extract":
        cmd_extract(a)
    elif a.cmd == "fit-pca":
        cmd_fit_pca(a)
    elif a.cmd == "project-pilot":
        cmd_project_pilot(a)
    elif a.cmd == "weights":
        cmd_weights(a)
    elif a.cmd == "summary":
        cmd_summary(a)


if __name__ == "__main__":
    main()
