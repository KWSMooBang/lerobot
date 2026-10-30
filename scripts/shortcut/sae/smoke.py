"""GPU checks of the hook location on the real π0.5 (T015 smoke). Writes `<out>/hook_check.json`.

    uv run python scripts/shortcut/sae/smoke.py --out /outputs/VLA-shortcut-exps/smoke/T015 --episodes 379,807,1261

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

HERE = Path(__file__).resolve().parent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 only (its lock is taken below)
for p in (str(HERE), str(HERE.parent), str(HERE.parent / "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from collect import dataset_source, gpu_lock, noise_for, resolve_model  # noqa: E402
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
    print(json.dumps(res, indent=1))
    (out / "sae_bench.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="pi05lib")
    ap.add_argument("--episodes", default="379,807,1261")
    ap.add_argument("--out-root", default="/outputs/VLA-shortcut-exps")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--bench-sae", action="store_true", help="only time SAE training steps")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.bench_sae:
        with ExitStack() as stack:
            gpu_lock(stack, [0], False)
            bench_sae(out)
        return
    eps = [int(x) for x in args.episodes.split(",")]

    with ExitStack() as stack:
        gpu_lock(stack, [0], False)
        from common import load_policy
        from torch.utils.data import DataLoader, Subset

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        m = resolve_model(args.model, Path(args.out_root))
        policy, pre, post, *_ = load_policy("pi05", m["ckpt"], "libero_spatial", 10, "cuda", m["adapter"])
        repo, root, rev = dataset_source(Path(args.out_root))
        ds = LeRobotDataset(repo, root=root, revision=rev, episodes=eps)
        # frames 0 and 60 of each episode, so the batch mixes suites and instruction lengths
        cols = ds.hf_dataset.select_columns(["episode_index", "frame_index"]).to_pandas()
        pick = cols.index[cols["frame_index"].isin([0, 60])].tolist()[: args.batch * 2]
        batch = next(iter(DataLoader(Subset(ds, pick), batch_size=len(pick))))
        obs = pre(batch)
        cfg = policy.config
        noise = noise_for(
            batch["episode_index"].numpy(),
            batch["frame_index"].numpy(),
            (cfg.chunk_size, cfg.max_action_dim),
            0,
        ).cuda()
        res: dict = {"model": m, "frames": len(pick), "episodes": eps}

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
                v = torch.randn(d, device="cuda")
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
    print(json.dumps(res, indent=1))
    (out / "hook_check.json").write_text(json.dumps(res, indent=1) + "\n")
    ok = (
        res["a_actions_equal_with_hooks"]
        and res["a_actions_equal_with_full_capture"]
        and res["b_layer_out_equals_next_in"]
        and res["b_emb_is_embed_prefix_output"]
        and res["c_empty_camera_tokens_masked"]
        and res["c_group_mean_max_abs_err"] < 1e-2
        and res["d_weighted_mean_rel_err"] < 1e-4
        and res["b_ae_recorder_max_abs_err"] < 1e-2
    )
    print("HOOK CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
