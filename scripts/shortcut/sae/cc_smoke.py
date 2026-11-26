#!/usr/bin/env python
"""T034 smoke (GPU 0): memory + throughput of candidate crosscoder sizes on random data of the real widths, and the
T031 single-layer SAE config as the throughput reference (T031: 3 100 steps in 12.2 s = 254 steps/s on this GPU).

  run_gpu.sh smoke python cc_smoke.py   → /outputs/VLA-shortcut-exps/sae/crosscoder/smoke/probe.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from crosscoder import CrossCoder, CrossCoderConfig  # noqa: E402
from model import SAEConfig, TopKSAE  # noqa: E402

OUT = Path("/outputs/VLA-shortcut-exps/sae/crosscoder/smoke")
T031_STEPS_PER_S = 3100 / 12.2


def step_loop(model, x, steps: int, amp: bool) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    n = model.cfg.n_latents
    last = torch.zeros(n, dtype=torch.long, device="cuda")
    for s in range(steps + 5):
        if s == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, _, fired = model.loss(x, (s - last) >= 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        model.project_decoder_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.renorm_decoder()
        last[fired] = s + 1
    torch.cuda.synchronize()
    return steps / (time.time() - t0)


def probe(name: str, dims: list[int], n: int, k: int, batch: int, steps: int = 30) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec: dict = {"name": name, "D": sum(dims), "n_latents": n, "k": k, "batch": batch}
    try:
        cfg = CrossCoderConfig(layers=[str(i) for i in range(len(dims))], dims=dims, n_latents=n, k=k)
        cc = CrossCoder(cfg, seed=0).cuda()
        x = torch.randn(batch, sum(dims), device="cuda", dtype=torch.float16)
        cc.init_from_data(x[:2000].float())
        rec["steps_per_s"] = step_loop(cc, x, steps, amp=True)
        rec["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        rec["ok"] = True
        del cc, x
    except torch.OutOfMemoryError as e:
        rec["ok"] = False
        rec["error"] = f"OOM: {str(e)[:200]}"
    torch.cuda.empty_cache()
    print(rec, flush=True)
    return rec


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"gpu": torch.cuda.get_device_name(), "time": time.strftime("%F %T")}
    # reference: T031 setting (fp32 SAE d 1024, n 2048, k 64, batch 4096), same step loop
    sae = TopKSAE(SAEConfig(d=1024, n_latents=2048, k=64), seed=0).cuda()
    x = torch.randn(4096, 1024, device="cuda")
    sae.init_from_data(x)
    sps = step_loop(sae, x, 500, amp=False)
    res["t031_ref"] = {"steps_per_s": sps, "t031_measured": T031_STEPS_PER_S, "ratio": sps / T031_STEPS_PER_S}
    print(res["t031_ref"], flush=True)
    del sae, x
    probes = []
    for n in (8192, 16384):
        for k in (64, 128):
            probes.append(probe(f"expert_n{n}_k{k}", [1024] * 18, n, k, 4096))
    probes.append(probe("vlm_n8192_k128", [2048] * 18, 8192, 128, 4096))
    probes.append(probe("vlm_n16384_k64", [2048] * 18, 16384, 64, 4096))
    probes.append(probe("joint_n8192_k128_b4096", [2048] * 18 + [1024] * 18, 8192, 128, 4096))
    probes.append(probe("joint_n8192_k128_b2048", [2048] * 18 + [1024] * 18, 8192, 128, 2048))
    res["probes"] = probes
    (OUT / "probe.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
