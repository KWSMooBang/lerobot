"""Checks of the hook location on the real π0.5 (T015 smoke). Writes `<out>/hook_check.json`.

    uv run python scripts/shortcut/sae/smoke.py --out <dir> --episodes 379,807,1261           # GPU (server)
    CUDA_VISIBLE_DEVICES= uv run python scripts/shortcut/sae/smoke.py --out <dir> --device cpu --frames 2

(a) recording does not change the action (same noise, bitwise)
(b) the hooked output of PaliGemma layer L is the input of layer L+1 (and layer 17's output is what the final
    norm gets); the same for the action expert at every denoising step; `emb` = embed_prefix output
(c) recorder group means = masked means computed from the full token tensors; padding text tokens and the
    empty third camera are excluded, cam1/cam2 have 256 tokens each
(d) count-weighted mean of the 3 group means = mean over all valid tokens
(e) a live `project_out` leaves no component along v in the hooked layer's output; `add` shifts by alpha·v
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
for p in (str(HERE), str(HERE.parent), str(HERE.parent / "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import store  # noqa: E402
import torch  # noqa: E402
from collect import DEFAULT_OUT_ROOT, dataset_source, gpu_lock, noise_for, resolve_model  # noqa: E402
from hooks import ActivationRecorder, add, ae_layers, all_layers, pg_layers, project_out  # noqa: E402


def bench_sae(out: Path, steps: int = 200) -> dict:
    """Seconds per optimizer step at batch 4096 (paper setting) → hours per full layer SAE."""
    import time

    from model import SAEConfig, TopKSAE

    res = {}
    n_t = 273_465
    for fam, d, k, n_samples in (("pg", 2048, 100, n_t), ("ae", 1024, 64, n_t * 10)):
        x = torch.randn(65536, d, device="cuda")
        sae = TopKSAE(SAEConfig(d=d, n_latents=d, k=k), seed=0).cuda()
        sae.init_from_data(x[:10000])
        opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
        last = torch.zeros(d, dtype=torch.long, device="cuda")
        for i in range(steps + 20):
            if i == 20:
                torch.cuda.synchronize()
                t0 = time.time()
            xb = x[torch.randint(0, len(x), (4096,), device="cuda")]
            loss, _, fired = sae.loss(xb, (i - last) >= 10)  # many dead latents: AuxK active (worst case)
            opt.zero_grad()
            loss.backward()
            sae.project_decoder_grad()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            sae.renorm_decoder()
            last[fired] = i + 1
        torch.cuda.synchronize()
        sec = (time.time() - t0) / steps
        total_steps = 100 * -(-n_samples // 4096)
        res[fam] = {"sec_per_step": sec, "steps_100_epochs": total_steps, "hours": sec * total_steps / 3600}
    res["gpu"] = torch.cuda.get_device_name(0)
    print(store.dumps(res, indent=1))
    (out / "sae_bench.json").write_text(store.dumps(res, indent=1) + "\n")
    return res


N_TIMESTEPS = 273_465  # HuggingFaceVLA/libero, 1,693 episodes


def hook_checks(res: dict) -> dict[str, bool]:
    """Every sub-check of the hook test. bf16 tolerance: `e_*_bf16_component_scale` = max|y|·2⁻⁸ (one bf16 ulp
    at the largest value of the layer output); edits are computed in fp32 and cast back to bf16."""
    out = {
        "a_actions_equal_with_hooks": res.get("a_actions_equal_with_hooks") is True,
        "a_actions_equal_with_full_capture": res.get("a_actions_equal_with_full_capture") is True,
        "b_layer_out_equals_next_in": res.get("b_layer_out_equals_next_in") is True,
        "b_emb_is_embed_prefix_output": res.get("b_emb_is_embed_prefix_output") is True,
        "b_ae_recorder_matches_capture": res.get("b_ae_recorder_max_abs_err", 1.0) < 1e-2,
        "b_ae_10_denoising_steps_counted": res.get("ae_steps_seen") == 10,
        "c_empty_camera_tokens_masked": res.get("c_empty_camera_tokens_masked") is True,
        "c_group_means_match_masked_means": res.get("c_group_mean_max_abs_err", 1.0) < 1e-2,
        "d_weighted_group_mean_eq_all_tokens": res.get("d_weighted_mean_rel_err", 1.0) < 1e-4,
    }
    for la in ("pg05", "ae05"):
        raw = res.get(f"e_{la}_bf16_component_scale")
        scale = float(raw) if isinstance(raw, float | int) and raw > 0 else 0.0
        ok = scale > 0
        out[f"e_{la}_project_out_no_v_component"] = (
            ok and res.get(f"e_{la}_project_out_max_component", 1e9) <= 4 * scale
        )
        out[f"e_{la}_add_shifts_by_alpha_v"] = ok and res.get(f"e_{la}_add_shift_err", 1e9) <= 2 * scale
    return out


def summarize(out: Path, model: str) -> bool:
    """PASS/FAIL of every smoke.sh step + full-scale estimates from the measured numbers → summary.json."""
    import pandas as pd
    from collect import bytes_per_timestep

    checks: dict[str, bool] = {}
    est: dict[str, object] = {}

    def read(p: Path) -> str:
        return p.read_text() if p.exists() else ""

    # every smoke.sh step must have exited 0 (steps.tsv: name, exit code)
    step_codes = [ln.split("\t") for ln in read(out / "steps.tsv").splitlines() if ln.strip()]
    checks["steps_recorded"] = len(step_codes) >= 6
    for name, rc in step_codes:
        checks[f"step_{name}_exit0"] = rc.strip() == "0"
    hc = json.loads(read(out / "hook_check.json") or "{}")
    checks["hook_check_pass"] = hc.get("pass") is True
    recomputed = hook_checks(hc)
    for k, v in recomputed.items():
        checks[f"hook_{k}"] = v and (hc.get("checks") or {}).get(k) is True
    acts = out / "sae" / model / "acts"
    parts = store.done_parts(acts) if (acts / "plan.json").exists() else []
    rows = sum(p["rows"] for p in parts)
    checks["collect_3_episodes_365_timesteps"] = len(parts) == 3 and rows == 365
    checks["collect_resume_skips_done_parts"] = read(out / "collect_resume.log").count("done, skip") == 3
    if parts:
        plan = store.read_plan(acts)
        secs = sum(p["seconds"] for p in parts)
        tps_all = rows / secs
        tps_best = max(p["timesteps_per_s"] for p in parts)
        bpt = bytes_per_timestep(plan["identity"]["layers"], len(plan["identity"]["ae_steps"]))
        est["gpu"] = parts[0]["gpu"]
        est["collect_timesteps_per_s"] = {"overall_incl_warmup": round(tps_all, 2), "best_part": tps_best}
        est["collect_full_hours_1gpu"] = round(N_TIMESTEPS / tps_best / 3600, 2)
        est["collect_full_storage_gb"] = round(N_TIMESTEPS * bpt / 1e9, 1)
        est["fp16_overflow_layers"] = sorted({x for p in parts for x in p.get("overflow_fp32", [])})
        meta = pd.concat([pd.read_parquet(Path(p["dir"]) / "meta.parquet") for p in parts])
        a = np.stack(meta["action"].to_numpy())
        a0 = np.stack(meta["policy_action0"].to_numpy())
        est["policy_action0_vs_dataset_mae"] = float(
            np.nanmean(np.abs(a - a0))
        )  # prefix-only parts: first batch only
        est["dataset_action_abs_mean"] = float(np.abs(a).mean())
    for lay, tag in (("pg05", "pooled_all"), ("ae05", "ae-steps-all")):
        jd = out / "sae" / model / "sae" / tag / lay / "seed0"
        log = [json.loads(x) for x in read(jd / "log.jsonl").splitlines() if x]
        steps = [r["step"] for r in log]
        checks[f"sae_{lay}_loss_decreases"] = len(log) > 2 and log[-1]["nmse"] < log[0]["nmse"]
        checks[f"sae_{lay}_resumed_at_200"] = (
            f"{lay} seed0: resume at step 200" in read(out / "train_resume.log")
            and steps == sorted(steps)
            and steps[-1:] == [600]
        )
        checks[f"sae_{lay}_stats_parquet"] = all(
            (jd / f).exists() for f in ("features.parquet", "fire_by_group.parquet", "codes_idx.npy")
        )
        m = json.loads(read(jd / "done.json") or "{}").get("metrics", {})
        checks[f"sae_{lay}_done_metrics_finite"] = all(
            isinstance(m.get(k), float) and np.isfinite(m[k]) for k in ("nmse", "var_explained")
        )
        if (jd / "metrics.json").exists():
            est[f"sae_{lay}_smoke_metrics"] = {
                k: v for k, v in json.loads((jd / "metrics.json").read_text()).items() if k != "stats"
            }
    bench = json.loads(read(out / "sae_bench.json") or "{}")
    checks["sae_bench_ok"] = all(
        isinstance(bench.get(f, {}).get("hours"), float | int) and 0 < bench[f]["hours"] < 1e4
        for f in ("pg", "ae")
    )
    if checks["sae_bench_ok"]:
        est["sae_hours_per_layer_100_epochs"] = {
            "pg_or_emb": round(bench["pg"]["hours"], 2),
            "ae_all_10_steps": round(bench["ae"]["hours"], 2),
            "all_37_layers_1gpu": round(19 * bench["pg"]["hours"] + 18 * bench["ae"]["hours"], 1),
        }
    ok = all(checks.values())
    print(store.dumps({"checks": checks, "estimates": est}, indent=1))
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("T015 SMOKE", "PASS" if ok else "FAIL")
    (out / "summary.json").write_text(
        store.dumps({"pass": ok, "checks": checks, "estimates": est}, indent=1) + "\n"
    )
    return ok


PREFIX_CONFIGS = {  # the VLM collection batches + all of them at once
    "a: emb,pg:0-5": "emb,pg:0-5",
    "b: pg:6-11": "pg:6-11",
    "c: pg:12-17": "pg:12-17",
    "all: emb,pg:0-17": "emb,pg:0-17",
}


def prefix_check(policy: Any, obs: dict, noise: Any, reps: int = 2) -> dict:
    """Prefix-only (cut after the deepest requested PaliGemma layer) vs full forward: recorded emb/PaliGemma
    activations must be bit-identical, nothing may stay attached afterwards; wall time of each."""
    import time

    from hooks import PrefixDone, parse_layers

    def run(layers: list[str], stop: bool) -> tuple[dict, float]:
        rec = ActivationRecorder(policy, layers, stop_after_prefix=stop)
        t0 = time.perf_counter()
        try:
            with torch.inference_mode(), rec:
                policy.predict_action_chunk(obs, noise=noise)
            done = True
        except PrefixDone:
            done = False
        dt = time.perf_counter() - t0
        if done == stop:
            raise RuntimeError(f"stop={stop} but the forward {'completed' if done else 'was cut'}")
        assert rec.counts is not None
        return {k: v.clone() for k, v in rec.pg.items()} | {"counts": rec.counts.clone()}, dt

    all_pg = parse_layers("emb,pg:0-17")
    with torch.inference_mode():
        a_before = policy.predict_action_chunk(obs, noise=noise)
    run(all_pg, False)  # warm-up
    t_full = []
    for _ in range(reps):
        ref, dt = run(all_pg, False)
        t_full.append(dt)
    res: dict = {"full_s": min(t_full), "configs": {}}
    for name, spec in PREFIX_CONFIGS.items():
        layers = parse_layers(spec)
        ts: list[float] = []
        got: dict[str, Any] = {}
        for _ in range(reps):
            got, dt = run(layers, True)
            ts.append(dt)
        equal = {k: bool(torch.equal(got[k], ref[k])) for k in got}
        res["configs"][name] = {
            "prefix_s": min(ts),
            "full_over_prefix": min(t_full) / min(ts),
            "bit_identical": all(equal.values()),
            "layers_compared": len(equal) - 1,
        }
    # nothing left behind by the interrupted calls: hooks, wrappers, next forward
    mods = list(pg_layers(policy)) + list(ae_layers(policy))
    res["no_hooks_left"] = all(not m._forward_hooks and not m._forward_pre_hooks for m in mods)
    res["no_wrappers_left"] = not {"embed_prefix", "_embed_images"} & set(policy.model.__dict__)
    with torch.inference_mode():
        a_after = policy.predict_action_chunk(obs, noise=noise)
    res["action_after_interrupts_bit_identical"] = bool(torch.equal(a_before, a_after))
    res["pass"] = (
        all(c["bit_identical"] for c in res["configs"].values())
        and res["no_hooks_left"]
        and res["no_wrappers_left"]
        and res["action_after_interrupts_bit_identical"]
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="pi05lib")
    ap.add_argument("--episodes", default="379,807,1261")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument(
        "--frames", type=int, default=8, help="frames in the test batch (0 and 60 of each episode)"
    )
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--gpu", type=int, default=0, help="nvidia-smi index (locked in this container)")
    ap.add_argument("--bench-sae", action="store_true", help="only time SAE training steps")
    ap.add_argument("--summarize", action="store_true", help="summary of a smoke.sh output dir (no GPU)")
    ap.add_argument(
        "--prefix-check", action="store_true", help="prefix-only vs full forward: identical values + timing"
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        sys.exit(0 if summarize(out, args.model) else 1)
    if args.device == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    if args.bench_sae:
        with ExitStack() as stack:
            gpu_lock(stack, [args.gpu], False)
            bench_sae(out)
        return
    eps = [int(x) for x in args.episodes.split(",")]

    dev = args.device
    with ExitStack() as stack:
        if dev == "cuda":
            gpu_lock(stack, [args.gpu], False)
        from common import load_policy
        from torch.utils.data import DataLoader, Subset

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        m = resolve_model(args.model, Path(args.out_root))
        policy, pre, post, *_ = load_policy("pi05", m["ckpt"], "libero_spatial", 10, dev, m["adapter"])
        repo, root, rev = dataset_source(Path(args.out_root))
        ds = LeRobotDataset(repo, root=root, revision=rev, episodes=eps)
        # frame 0 of each episode, then frame 60: the batch mixes suites and instruction lengths
        cols = ds.hf_dataset.select_columns(["episode_index", "frame_index"]).to_pandas()
        sel = cols[cols["frame_index"].isin([0, 60])].sort_values(["frame_index", "episode_index"])
        pick = sel.index.tolist()[: args.frames]
        batch = next(iter(DataLoader(Subset(ds, pick), batch_size=len(pick))))
        obs = pre(batch)
        cfg = policy.config
        noise = noise_for(
            batch["episode_index"].numpy(),
            batch["frame_index"].numpy(),
            (cfg.chunk_size, cfg.max_action_dim),
            0,
        ).to(dev)
        res: dict = {"model": m, "frames": len(pick), "episodes": eps}
        if args.prefix_check:
            res |= prefix_check(policy, obs, noise)
            res["device"] = dev
            print(json.dumps(res, indent=1))
            (out / "prefix_check.json").write_text(json.dumps(res, indent=1) + "\n")
            print("PREFIX CHECK", "PASS" if res["pass"] else "FAIL")
            sys.exit(0 if res["pass"] else 1)

        with torch.inference_mode():
            a_plain = policy.predict_action_chunk(obs, noise=noise)
            with ActivationRecorder(policy, all_layers()) as rec:
                a_rec = policy.predict_action_chunk(obs, noise=noise)
            res["a_actions_equal_with_hooks"] = bool(torch.equal(a_plain, a_rec))
            res["a_max_abs_diff"] = float((a_plain - a_rec).abs().max())

            # full tensors: inputs and outputs of every decoder layer + final norms
            cap: dict[str, list] = {}
            hs = []

            def keep(key):
                def f(_m, args, kwargs, *rest):
                    x = rest[0] if rest else (args[0] if args else kwargs["hidden_states"])
                    cap.setdefault(key, []).append(x.detach().float().clone())

                return f

            for fam, layers in (("pg", pg_layers(policy)), ("ae", ae_layers(policy))):
                for i, layer in enumerate(layers):
                    hs.append(layer.register_forward_pre_hook(keep(f"{fam}{i:02d}.in"), with_kwargs=True))
                    hs.append(layer.register_forward_hook(keep(f"{fam}{i:02d}.out"), with_kwargs=True))
            lm = policy.model.paligemma_with_expert.paligemma.model.language_model
            ex = policy.model.paligemma_with_expert.gemma_expert.model
            hs.append(lm.norm.register_forward_pre_hook(keep("pg.norm.in"), with_kwargs=True))
            hs.append(ex.norm.register_forward_pre_hook(keep("ae.norm.in"), with_kwargs=True))
            orig_ep = policy.model.embed_prefix
            pref: dict = {}

            def embed_prefix(*a, **k):
                r = orig_ep(*a, **k)
                pref["embs"], pref["pad"] = r[0].detach().float().clone(), r[1].clone()
                return r

            policy.model.embed_prefix = embed_prefix
            with ActivationRecorder(policy, all_layers()) as rec:
                a_cap = policy.predict_action_chunk(obs, noise=noise)
            del policy.model.embed_prefix
            for h in hs:
                h.remove()
            res["a_actions_equal_with_full_capture"] = bool(torch.equal(a_plain, a_cap))

            # (b)
            chain = {}
            for fam, n in (("pg", 18), ("ae", 18)):
                for i in range(n - 1):
                    outs, ins = cap[f"{fam}{i:02d}.out"], cap[f"{fam}{i + 1:02d}.in"]
                    chain[f"{fam}{i:02d}->{i + 1:02d}"] = len(outs) == len(ins) and all(
                        torch.equal(o, x) for o, x in zip(outs, ins, strict=True)
                    )
                last, norm_in = cap[f"{fam}17.out"], cap[f"{fam}.norm.in"]
                chain[f"{fam}17->final_norm"] = len(last) == len(norm_in) and all(
                    torch.equal(o, x) for o, x in zip(last, norm_in, strict=True)
                )
            res["b_layer_out_equals_next_in"] = all(chain.values())
            res["b_failed"] = [k for k, v in chain.items() if not v]
            res["b_calls"] = {"pg00": len(cap["pg00.out"]), "ae00": len(cap["ae00.out"])}
            res["b_emb_is_embed_prefix_output"] = bool(
                torch.equal(cap["pg00.in"][0], pref["embs"].to(torch.bfloat16).float())
            )

            # (c) + (d)
            pad = pref["pad"]
            n_txt = obs["observation.language.attention_mask"].shape[1]
            total = pad.shape[1]
            n_img = (total - n_txt) // 3
            groups = {"cam1": (0, n_img), "cam2": (n_img, 2 * n_img), "text": (3 * n_img, total)}
            errs, errs_all = [], []
            assert rec.counts is not None
            for name in ["emb"] + [f"pg{i:02d}" for i in range(18)]:
                full = cap["pg00.in"][0] if name == "emb" else cap[f"{name}.out"][0]
                for g, (s, e) in groups.items():
                    msk = pad[:, s:e].float()
                    ref = (full[:, s:e] * msk[..., None]).sum(1) / msk.sum(1, keepdim=True)
                    errs.append(float((rec.pg[name][:, ["cam1", "cam2", "text"].index(g)] - ref).abs().max()))
                allm = pad.float()
                ref_all = (full * allm[..., None]).sum(1) / allm.sum(1, keepdim=True)
                c = rec.counts.float()
                wavg = (rec.pg[name] * c[..., None]).sum(1) / c.sum(1, keepdim=True)
                errs_all.append(float((wavg - ref_all).abs().max() / ref_all.abs().max()))
            res["c_counts_first_frame"] = rec.counts[0].tolist()
            res["c_text_tokens_valid_vs_mask"] = [
                int(x) for x in obs["observation.language.attention_mask"].sum(1).tolist()
            ]
            res["c_empty_camera_tokens_masked"] = bool((~pad[:, 2 * n_img : 3 * n_img]).all())
            res["c_text_padding_present"] = bool((~pad[:, 3 * n_img :]).any())
            res["c_group_mean_max_abs_err"] = max(errs)
            res["d_weighted_mean_rel_err"] = max(errs_all)
            # AE recorder vs captured
            ae_err = max(
                float((rec.ae[f"ae{i:02d}"][:, s] - cap[f"ae{i:02d}.out"][s].mean(1)).abs().max())
                for i in range(18)
                for s in range(10)
            )
            res["b_ae_recorder_max_abs_err"] = ae_err
            res["ae_steps_seen"] = rec.n_steps

            # (e) live interventions on pg05 / ae05
            torch.manual_seed(0)
            for layer in ("pg05", "ae05"):
                d = 2048 if layer.startswith("pg") else 1024
                v = torch.randn(d, device=dev)
                v = v / v.norm()
                got: list = []
                mod = pg_layers(policy)[5] if layer.startswith("pg") else ae_layers(policy)[5]
                with project_out(policy, layer, v):
                    h = mod.register_forward_hook(lambda _m, _a, o, got=got: got.append(o.float()))
                    policy.predict_action_chunk(obs, noise=noise)
                    h.remove()
                res[f"e_{layer}_project_out_max_component"] = max(float((g @ v).abs().max()) for g in got)
                res[f"e_{layer}_bf16_component_scale"] = max(float(g.abs().max()) for g in got) * 2**-8
                ref_out = cap[f"{layer}.out"]
                got = []
                with add(policy, layer, v, 10.0):
                    h = mod.register_forward_hook(lambda _m, _a, o, got=got: got.append(o.float()))
                    policy.predict_action_chunk(obs, noise=noise)
                    h.remove()
                # first call has the same input as without steering
                res[f"e_{layer}_add_shift_err"] = float(((got[0] - ref_out[0]) - 10.0 * v).abs().max())

        res["post_action0_vs_dataset_l1"] = float(
            (post(a_plain[:, 0]).float().cpu() - batch["action"].float()).abs().mean()
        )
    print(store.dumps(res, indent=1))
    (out / "hook_check.json").write_text(store.dumps(res, indent=1) + "\n")
    res["checks"] = hook_checks(res)
    res["pass"] = ok = all(res["checks"].values())
    (out / "hook_check.json").write_text(store.dumps(res, indent=1) + "\n")
    for k, v in res["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("HOOK CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
