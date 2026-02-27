"""GELU77 – Output-Space Surprise × Cosine Gate.

MOTIVATION:
    gelu71 measures surprise as: ||x[b,t] - ema_x|| (input deviation magnitude).
    The input x is the pre-GELU linear projection of the residual stream.
    
    But surprise at the INPUT doesn't always produce surprise at the OUTPUT.
    The GELU nonlinearity is ~linear for large |x| and suppresses negative x.
    
    Example: if x changes from 2.0 to 4.0 (100% deviation at input), 
    GELU(2.0)=1.955, GELU(4.0)=3.9996 — output changes 105% (amplified).
    
    Example: if x changes from -0.5 to -2.0 (300% deviation at input),
    GELU(-0.5)=-0.154, GELU(-2.0)=-0.046 — output changes only 70% (compressed!).
    
    The OUTPUT deviation directly reflects what the next layer ACTUALLY RECEIVES —
    it's the signal that matters for the residual connection and attention layers.
    
    Testing: does using output-space deviation give a better novelty signal than input-space?

IMPLEMENTATION:
    ema_out_raw[d]  = EMA of E[GELU(x)[d]]   per channel (raw value, not normalized)
    surp[b, t]      = tanh(σ × ||out[b,t] - ema_out_raw|| / norm_out_raw)
    
    gate = exp(-τ × cos(out, ema_out_dir)) × (1 + w × surp)
    
    where ema_out_dir = normalized unit vector for cosine (same as gelu71's ema_out).
    The cosine gate operates on DIRECTION familiarity.
    The surprise operates on MAGNITUDE DEVIATION of the raw output.
    
    These are different aspects of the output: direction vs magnitude deviation.

NOTE ON DIFFERENCE FROM GELU71:
    gelu71: surprise = tanh(σ × ||x - ema_x|| / ||ema_x||)   ← deviates from input mean
    gelu77: surprise = tanh(σ × ||out - ema_out_raw|| / ||ema_out_raw||)  ← deviates from output mean
    
    The denominator is ||ema_out_raw|| ≈ mean GELU output magnitude.
    Since GELU(x) ≥ 0 for typical x distributions, ema_out_raw[d] > 0 generally.
    The normalization is numerically stable.

Params: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars (same count as gelu71).
State: _ema_out_dir (D,) unit vector; _ema_out_raw (D,) raw mean; _ema_out_norm scalar.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU77(nn.Module):
    """Output-space surprise × cosine gate: deviation from output mean, not input mean."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_out_dir:  torch.Tensor = None   # unit vector (direction EMA)
        self._ema_out_raw:  torch.Tensor = None   # raw value EMA (for deviation)
        self._ema_out_norm: float = 1.0
        self._ready = False

    def reset_state(self):
        self._ema_out_dir  = None
        self._ema_out_raw  = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                om = out.detach().flatten(0,1).mean(0)
                self._ema_out_raw  = om.clone()
                self._ema_out_norm = om.norm().item() + self.eps
                self._ema_out_dir  = F.normalize(om, dim=0).clone()
                self._ready        = True
            return out

        # Cosine familiarity gate (direction of output vs historical mean direction)
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        # Output-deviation surprise (magnitude of output vs historical mean output)
        ema_or  = self._ema_out_raw.view(1, 1, D)
        delta   = (out.detach() - ema_or).norm(dim=-1)
        surprise = torch.tanh(sigma * delta / (self._ema_out_norm + self.eps))

        gate = gate_cos * (1.0 + w * surprise)
        output = out * gate.unsqueeze(-1)

        with torch.no_grad():
            om = out.detach().flatten(0,1).mean(0)
            self._ema_out_raw  = d_val * self._ema_out_raw  + (1.0-d_val) * om
            self._ema_out_norm = self._ema_out_raw.norm().item() + self.eps
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1.0-d_val) * F.normalize(om, dim=0)

        return output
