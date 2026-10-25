"""Default worker/env counts from CPU cores and GPU memory, and the dry-run throughput model."""

from __future__ import annotations

import math
import os
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from typing import Any

# π0.5 (PaliGemma 3B + 300M action expert) in bf16, measured on the 4090 path; per-env cost = batch-row
# activations (≈ 1k prefix tokens, KV cache) + the env's EGL context/framebuffers on the same GPU.
WEIGHTS_GB = 8.0
WORKER_BASE_GB = 3.0  # CUDA context, allocator slack, batch-1 activations
PER_ENV_GB = 0.7
N_CAP = 16  # lockstep: every tick waits for the slowest env; beyond ~16 the straggler cost dominates
GPU_MEM_HEADROOM = 0.9


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def gpu_info(ids: list[str]) -> dict[str, dict[str, Any]]:
    """{id: {name, mem_gb, uuid}} from nvidia-smi (PCI order = CUDA_DEVICE_ORDER=PCI_BUS_ID). Empty if
    nvidia-smi is missing or an id is not a GPU (mock runs with fake ids)."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.check_output(  # nosec B603 B607
            ["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=30,
        )
    except Exception:
        return {}
    info = {}
    for line in out.strip().splitlines():
        idx, name, mem, uuid = (x.strip() for x in line.split(","))
        info[idx] = {"name": name, "mem_gb": float(mem) / 1024, "uuid": uuid}
    return {i: info[i] for i in ids if i in info}


def choose(
    gpus: list[str], procs: str, envs: str, mem_gb: float | None, cores: int | None = None
) -> tuple[int, int, list[str]]:
    """(procs per GPU, envs per worker, reasons). 'auto' or an int for each."""
    cores = cores or cpu_count()
    why = []
    if procs == "auto":
        p = 2 if mem_gb is not None and mem_gb >= 40 else 1
        why.append(
            f"procs-per-gpu={p}: GPU memory {mem_gb:.0f} GB "
            f"({'>=' if p == 2 else '<'} 40 GB; 2 workers/GPU overlap one's env steps with the other's inference)"
            if mem_gb is not None
            else f"procs-per-gpu={p}: GPU memory unknown"
        )
    else:
        p = int(procs)
        why.append(f"procs-per-gpu={p} (given)")
    workers = len(gpus) * p
    if envs == "auto":
        n_cpu = max(1, (cores - workers) // workers)
        parts = [f"cpu: ({cores} cores - {workers} worker mains) // {workers} workers = {n_cpu}"]
        n = n_cpu
        if mem_gb is not None:
            n_mem = math.floor((mem_gb * GPU_MEM_HEADROOM / p - WEIGHTS_GB - WORKER_BASE_GB) / PER_ENV_GB)
            parts.append(
                f"gpu mem: ({mem_gb:.0f} GB x {GPU_MEM_HEADROOM} / {p} - {WEIGHTS_GB} weights - "
                f"{WORKER_BASE_GB} base) / {PER_ENV_GB} GB per env = {n_mem}"
            )
            n = min(n, n_mem)
        parts.append(f"cap {N_CAP} (lockstep straggler cost)")
        n = max(1, min(n, N_CAP))
        why.append(f"num-envs={n} = min(" + "; ".join(parts) + ")")
    else:
        n = int(envs)
        why.append(f"num-envs={n} (given)")
    return p, n, why


@dataclass
class ThroughputModel:
    """Steps/s of the batched path. Defaults are from the T001 smoke on the RTX 4090 (45 ms/step single:
    ≈10 ms env step + ≈300 ms per chunk inference over 10 steps + per-step preprocessing). Replace them
    with measured values (--est-*) once a GPU check has run on the target server."""

    env_ms: float = 10.0  # one env step incl. 360 px render (EGL on the GPU)
    infer1_ms: float = 300.0  # one chunk inference, batch 1, incl. preprocessing
    beta: float = 0.1  # infer_ms(N) = infer1_ms * (1 + beta * (N - 1))
    single_ms_per_step: float = 45.0
    straggler: float = 0.1  # lockstep tick = env_ms * (1 + straggler * ln N)
    align_waste: float = 0.03  # slots idle until the next chunk boundary
    load_s: float = 90.0  # model load per worker per policy group

    def infer_ms(self, n: int) -> float:
        return self.infer1_ms * (1 + self.beta * (n - 1))

    def gpu_steps_per_s(self, p: int, n: int, n_action_steps: int) -> float:
        tick = self.env_ms * (1 + self.straggler * math.log(max(n, 1))) + self.infer_ms(n) / n_action_steps
        env_bound = p * n / tick  # each worker alternates env steps and its share of inference
        gpu_bound = n * n_action_steps / self.infer_ms(n)  # GPU busy with inference all the time
        return 1000 * min(env_bound, gpu_bound) * (1 - self.align_waste)

    def report(self, steps: float, g: int, p: int, n: int, n_action_steps: int, n_groups: int) -> list[str]:
        rate = g * self.gpu_steps_per_s(p, n, n_action_steps)
        hours = steps / rate / 3600 + n_groups * self.load_s / 3600
        single_h = steps * self.single_ms_per_step / 1000 / 3600
        return [
            f"model: env {self.env_ms} ms/step, chunk inference {self.infer1_ms} ms at batch 1, "
            f"x(1 + {self.beta}(N-1)) at batch N, straggler {self.straggler} ln N, align waste {self.align_waste}",
            f"per GPU: {self.gpu_steps_per_s(p, n, n_action_steps):.0f} env-steps/s (P={p}, N={n}); "
            f"{g} GPUs: {rate:.0f} env-steps/s",
            f"estimate: {steps / 1e6:.2f} M env-steps -> {hours:.1f} h wall ({hours * g:.0f} GPU-h); "
            f"single path at {self.single_ms_per_step} ms/step: {single_h:.0f} GPU-h",
        ]
