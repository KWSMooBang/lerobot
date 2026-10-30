"""TopK sparse autoencoder with AuxK (Gao et al. 2024), as configured in Swann et al. 2026 App. B.1.

  x̃ = ((x − b_pre) − μ) / ‖(x − b_pre) − μ‖₂        μ = scalar mean over the d dims, per sample
  z  = ReLU(TopK(W_enc x̃))                           no encoder bias
  x̂̃ = W_dec z,   x̂ = x̂̃·‖·‖ + μ + b_pre               no decoder bias, unit-norm decoder columns
  L  = ‖x̃ − x̂̃‖² / C + (1/32)·‖ẽ − ê_aux‖² / C        ẽ = x̃ − x̂̃ (detached), ê_aux from the top-k_aux
                                                      dead latents (no firing in the last 500 steps)

Both loss terms are taken in the normalized space (where the AuxK residual lives); C (C_MSE) is the
variance of the centered normalized activations at initialization, so the first term is the normalized
MSE (fraction of variance unexplained). `metrics()` also reports variance explained in the raw space.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass
class SAEConfig:
    d: int
    n_latents: int
    k: int
    k_aux: int = 512
    aux_coef: float = 1.0 / 32
    dead_steps: int = 500
    c_mse: float = 1.0  # set from the data at init
    eps: float = 1e-6

    @classmethod
    def load(cls, path: str | Path) -> SAEConfig:
        d = json.loads(Path(path).read_text())
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def geometric_median(x: Tensor, iters: int = 100, tol: float = 1e-6) -> Tensor:
    """Weiszfeld's algorithm on rows of x (N, d), in float32."""
    x = x.float()
    y = x.mean(0)
    for _ in range(iters):
        w = 1.0 / (x - y).norm(dim=1).clamp_min(1e-8)
        y_new = (w[:, None] * x).sum(0) / w.sum()
        if (y_new - y).norm() <= tol * y.norm().clamp_min(1.0):
            return y_new
        y = y_new
    return y


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        w_dec = torch.randn(cfg.d, cfg.n_latents, generator=g)
        w_dec = w_dec / w_dec.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(w_dec)  # (d, n), columns = feature directions
        self.W_enc = nn.Parameter(w_dec.T.clone() * math.sqrt(cfg.k / cfg.d))  # (n, d)
        self.b_pre = nn.Parameter(torch.zeros(cfg.d))

    # ---------------------------------------------------------------- core
    def normalize(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        xc = x.float() - self.b_pre
        mu = xc.mean(-1, keepdim=True)
        xc = xc - mu
        nrm = xc.norm(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        return xc / nrm, mu, nrm

    def pre_acts(self, xt: Tensor) -> Tensor:
        return xt @ self.W_enc.T

    def topk(self, pre: Tensor, k: int | None = None) -> tuple[Tensor, Tensor]:
        vals, idx = pre.topk(k or self.cfg.k, dim=-1)
        return vals.relu(), idx

    def decode_sparse(self, vals: Tensor, idx: Tensor) -> Tensor:
        z = torch.zeros(*vals.shape[:-1], self.cfg.n_latents, device=vals.device, dtype=vals.dtype)
        z.scatter_(-1, idx, vals)
        return z @ self.W_dec.T

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x (..., d) raw activations → (values, indices) of the k active latents (values ≥ 0)."""
        xt, _, _ = self.normalize(x)
        return self.topk(self.pre_acts(xt))

    def reconstruct(self, x: Tensor, return_metrics: bool = False):
        """x (..., d) → x̂ in the raw space (same shape). With return_metrics: (x̂, metrics(x))."""
        xt, mu, nrm = self.normalize(x)
        vals, idx = self.topk(self.pre_acts(xt))
        xt_hat = self.decode_sparse(vals, idx)
        x_hat = xt_hat * nrm + mu + self.b_pre
        if not return_metrics:
            return x_hat
        return x_hat, self._metrics(x.float(), x_hat, xt, xt_hat)

    def _metrics(self, x: Tensor, x_hat: Tensor, xt: Tensor, xt_hat: Tensor) -> dict[str, float]:
        x2, xh2 = x.reshape(-1, x.shape[-1]), x_hat.reshape(-1, x.shape[-1])
        xt2, xth2 = xt.reshape(-1, xt.shape[-1]), xt_hat.reshape(-1, xt.shape[-1])
        raw_var = (x2 - x2.mean(0)).pow(2).sum(-1).mean()
        return {
            "nmse": float((xt2 - xth2).pow(2).sum(-1).mean() / self.cfg.c_mse),
            "var_explained": float(1 - (x2 - xh2).pow(2).sum(-1).mean() / raw_var.clamp_min(1e-12)),
            "n": int(x2.shape[0]),
        }

    @torch.no_grad()
    def metrics(self, x: Tensor) -> dict[str, float]:
        return self.reconstruct(x, return_metrics=True)[1]

    # ---------------------------------------------------------------- training
    def loss(self, x: Tensor, dead: Tensor | None) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        """x (B, d) raw. dead: (n,) bool mask of dead latents. Returns (loss, parts, fired (n,) bool)."""
        xt, _, _ = self.normalize(x)
        pre = self.pre_acts(xt)
        vals, idx = self.topk(pre)
        xt_hat = self.decode_sparse(vals, idx)
        mse = (xt - xt_hat).pow(2).sum(-1).mean() / self.cfg.c_mse
        fired = torch.zeros(self.cfg.n_latents, dtype=torch.bool, device=x.device)
        fired[idx[vals > 0]] = True
        aux = torch.zeros((), device=x.device)
        n_dead = int(dead.sum()) if dead is not None else 0
        if dead is not None and n_dead > 0 and self.cfg.aux_coef > 0:
            k_aux = min(self.cfg.k_aux, n_dead)
            pre_dead = pre.masked_fill(~dead, float("-inf"))
            a_vals, a_idx = pre_dead.topk(k_aux, dim=-1)
            a_vals = a_vals.relu().nan_to_num(0.0)
            e_hat = self.decode_sparse(a_vals, a_idx)
            e = (xt - xt_hat).detach()
            aux = (e - e_hat).pow(2).sum(-1).mean() / self.cfg.c_mse
        loss = mse + self.cfg.aux_coef * aux
        return loss, {"mse": mse.detach(), "aux": aux.detach()}, fired

    @torch.no_grad()
    def init_from_data(self, sample: Tensor) -> None:
        """b_pre = geometric median of the sample; C_MSE = variance of the centered normalized sample."""
        self.b_pre.copy_(geometric_median(sample).to(self.b_pre.device))
        xt, _, _ = self.normalize(sample.to(self.b_pre.device))
        self.cfg.c_mse = float((xt - xt.mean(0)).pow(2).sum(-1).mean())

    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        """Remove the gradient component along each unit-norm decoder column (tangent plane)."""
        g = self.W_dec.grad
        if g is not None:
            g -= (g * self.W_dec).sum(0, keepdim=True) * self.W_dec

    @torch.no_grad()
    def renorm_decoder(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)

    # ---------------------------------------------------------------- io
    def save(self, out_dir: str | Path, extra: dict | None = None) -> None:
        from safetensors.torch import save_file

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}, out / "sae.safetensors"
        )
        (out / "config.json").write_text(json.dumps({**asdict(self.cfg), **(extra or {})}, indent=1) + "\n")

    @classmethod
    def load(cls, out_dir: str | Path, device: str = "cpu") -> TopKSAE:
        from safetensors.torch import load_file

        out = Path(out_dir)
        sae = cls(SAEConfig.load(out / "config.json"))
        sae.load_state_dict(load_file(out / "sae.safetensors"))
        return sae.to(device)
