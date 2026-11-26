"""Acausal sparse crosscoder (Lindsey et al. 2024) with the TopK + AuxK recipe of model.py (T034).

One feature dictionary shared by L layers (blocks). Block l has d_l dims; the input is the concatenation
x = [x_0; …; x_{L-1}] (D = Σ d_l).

  x̃_l = ((x_l − b_pre_l) − μ_l) / ‖·‖₂               per block, exactly model.py's normalisation
  z   = ReLU(TopK(W_enc x̃))  = TopK(Σ_l W_enc^l x̃_l)  one encoder reads every block, no encoder bias
  x̂̃_l = W_dec^l z                                    per-block decoders, no decoder bias
  L   = Σ_l ‖x̃_l − x̂̃_l‖² / C_l + (1/32)·Σ_l ‖ẽ_l − ê_aux,l‖² / C_l     (AuxK on dead latents, as model.py)

Decoder constraint: each feature's concatenated decoder row [d_i^0; …; d_i^{L-1}] has unit norm (tangent-plane
gradient + renorm, as model.py), so ‖d_i^l‖² is feature i's share in block l (its *layer profile*, sums to 1).

Storage: W_dec is (n_latents, D) (rows = features). W_enc is (n_latents, D) like model.py. With L = 1 this is
model.TopKSAE with W_dec transposed (tests/test_crosscoder.py checks the reduction).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

try:
    from model import geometric_median
except ImportError:  # imported as a package module
    from .model import geometric_median


@dataclass
class CrossCoderConfig:
    layers: list[str]
    dims: list[int]
    n_latents: int
    k: int
    k_aux: int = 512
    aux_coef: float = 1.0 / 32
    dead_steps: int = 500
    c_mse: list[float] = field(default_factory=list)  # per block, set from the data at init
    eps: float = 1e-6

    def __post_init__(self) -> None:
        assert len(self.layers) == len(self.dims), (self.layers, self.dims)
        if not self.c_mse:
            self.c_mse = [1.0] * len(self.dims)

    @property
    def D(self) -> int:
        return int(sum(self.dims))

    @property
    def offsets(self) -> list[int]:
        o = [0]
        for d in self.dims:
            o.append(o[-1] + d)
        return o

    @classmethod
    def load(cls, path: str | Path) -> CrossCoderConfig:
        d = json.loads(Path(path).read_text())
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class CrossCoder(nn.Module):
    def __init__(self, cfg: CrossCoderConfig, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        D, n = cfg.D, cfg.n_latents
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        # same draw as model.TopKSAE (randn (D, n), columns normalised), stored transposed
        w = torch.randn(D, n, generator=g)
        w = w / w.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(w.T.contiguous())  # (n, D), rows = concatenated feature directions
        self.W_enc = nn.Parameter(w.T.clone() * math.sqrt(cfg.k / D))  # (n, D)
        self.b_pre = nn.Parameter(torch.zeros(D))
        self._uniform = len(set(cfg.dims)) == 1
        self.register_buffer("c_vec", torch.tensor(cfg.c_mse, dtype=torch.float32), persistent=False)

    # ---------------------------------------------------------------- blocks
    def _blocks(self, x: Tensor) -> list[Tensor]:
        o = self.cfg.offsets
        return [x[..., o[i] : o[i + 1]] for i in range(len(self.cfg.dims))]

    def normalize(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x (..., D) raw → x̃ (..., D), μ (..., L), ‖·‖ (..., L)."""
        xc = x.float() - self.b_pre
        if self._uniform:
            L, d = len(self.cfg.dims), self.cfg.dims[0]
            xb = xc.reshape(*xc.shape[:-1], L, d)
            mu = xb.mean(-1, keepdim=True)
            xb = xb - mu
            nrm = xb.norm(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
            return (xb / nrm).reshape(xc.shape), mu[..., 0], nrm[..., 0]
        outs, mus, nrms = [], [], []
        for b in self._blocks(xc):
            mu = b.mean(-1, keepdim=True)
            b = b - mu
            nrm = b.norm(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
            outs.append(b / nrm)
            mus.append(mu)
            nrms.append(nrm)
        return torch.cat(outs, -1), torch.cat(mus, -1), torch.cat(nrms, -1)

    def denormalize(self, xt_hat: Tensor, mu: Tensor, nrm: Tensor) -> Tensor:
        o = self.cfg.offsets
        out = torch.empty_like(xt_hat)
        for i in range(len(self.cfg.dims)):
            out[..., o[i] : o[i + 1]] = xt_hat[..., o[i] : o[i + 1]] * nrm[..., i : i + 1] + mu[..., i : i + 1]
        return out + self.b_pre

    def block_sq(self, r: Tensor) -> Tensor:
        """r (B, D) → (B, L) per-block squared norms."""
        if self._uniform:
            return r.reshape(r.shape[0], len(self.cfg.dims), -1).pow(2).sum(-1)
        return torch.stack([b.pow(2).sum(-1) for b in self._blocks(r)], -1)

    # ---------------------------------------------------------------- core
    def pre_acts(self, xt: Tensor) -> Tensor:
        return xt @ self.W_enc.T

    def topk(self, pre: Tensor, k: int | None = None) -> tuple[Tensor, Tensor]:
        vals, idx = pre.float().topk(k or self.cfg.k, dim=-1)
        return vals.relu(), idx

    def decode_sparse(self, vals: Tensor, idx: Tensor) -> Tensor:
        """(..., k) values / indices → (..., D) = Σ_j vals_j · W_dec[idx_j] (normalised space). Dense z @ W_dec
        (compute-bound under bf16 autocast; a row gather is memory-bound for the AuxK k_aux = 512 path)."""
        z = torch.zeros(*vals.shape[:-1], self.cfg.n_latents, device=vals.device, dtype=vals.dtype)
        z.scatter_(-1, idx, vals)
        return z @ self.W_dec

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x (..., D) raw → (values, indices) of the k active latents (values ≥ 0)."""
        xt, _, _ = self.normalize(x)
        return self.topk(self.pre_acts(xt))

    def reconstruct(self, x: Tensor) -> Tensor:
        xt, mu, nrm = self.normalize(x)
        vals, idx = self.topk(self.pre_acts(xt))
        return self.denormalize(self.decode_sparse(vals, idx), mu, nrm)

    def layer_profile(self) -> Tensor:
        """(n_latents, L): ‖d_i^l‖² (rows sum to 1 under the unit-norm constraint)."""
        with torch.no_grad():
            return self.block_sq(self.W_dec.float())

    # ---------------------------------------------------------------- training
    def loss(self, x: Tensor, dead: Tensor | None) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        """x (B, D) raw. dead: (n,) bool. Returns (loss, parts, fired (n,) bool). parts['mse_l'] is (L,)."""
        xt, _, _ = self.normalize(x)
        pre = self.pre_acts(xt)
        vals, idx = self.topk(pre)
        xt_hat = self.decode_sparse(vals, idx).float()
        mse_l = self.block_sq(xt - xt_hat).mean(0) / self.c_vec
        mse = mse_l.sum()
        fired = torch.zeros(self.cfg.n_latents, dtype=torch.bool, device=x.device)
        fired[idx[vals > 0]] = True
        aux = torch.zeros((), device=x.device)
        n_dead = int(dead.sum()) if dead is not None else 0
        if dead is not None and n_dead > 0 and self.cfg.aux_coef > 0:
            k_aux = min(self.cfg.k_aux, n_dead)
            pre_dead = pre.float().masked_fill(~dead, float("-inf"))
            a_vals, a_idx = pre_dead.topk(k_aux, dim=-1)
            a_vals = a_vals.relu().nan_to_num(0.0)
            e_hat = self.decode_sparse(a_vals, a_idx).float()
            e = (xt - xt_hat).detach()
            aux = (self.block_sq(e - e_hat).mean(0) / self.c_vec).sum()
        loss = mse + self.cfg.aux_coef * aux
        return loss, {"mse": mse.detach(), "mse_l": mse_l.detach(), "aux": aux.detach()}, fired

    @torch.no_grad()
    def init_from_data(self, sample: Tensor) -> None:
        """b_pre_l = geometric median of block l; C_l = variance of the centred normalised block l."""
        o = self.cfg.offsets
        for i in range(len(self.cfg.dims)):
            self.b_pre[o[i] : o[i + 1]] = geometric_median(sample[:, o[i] : o[i + 1]]).to(self.b_pre.device)
        xt, _, _ = self.normalize(sample.to(self.b_pre.device))
        c = self.block_sq(xt - xt.mean(0)).mean(0)
        self.cfg.c_mse = [float(v) for v in c]
        self.c_vec = c.float().to(self.b_pre.device)

    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        g = self.W_dec.grad
        if g is not None:
            g -= (g * self.W_dec).sum(1, keepdim=True) * self.W_dec

    @torch.no_grad()
    def renorm_decoder(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)

    # ---------------------------------------------------------------- io
    def save(self, out_dir: str | Path, extra: dict | None = None) -> None:
        from safetensors.torch import save_file

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}, out / "crosscoder.safetensors"
        )
        cfg = {**asdict(self.cfg), **(extra or {})}
        (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str) + "\n")

    @classmethod
    def load(cls, out_dir: str | Path, device: str = "cpu") -> CrossCoder:
        from safetensors.torch import load_file

        out = Path(out_dir)
        cc = cls(CrossCoderConfig.load(out / "config.json"))
        cc.load_state_dict(load_file(out / "crosscoder.safetensors"))
        return cc.to(device)
