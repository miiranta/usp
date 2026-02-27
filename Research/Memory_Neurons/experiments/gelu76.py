"""GELU76 – Surprise × Double Cosine (gelu31 backbone + surprise boost).

THE IDEA:
    gelu31 (val PPL 177.51) uses a DOUBLE cosine gate:
        gate = exp(-τ_in × cos(x, ema_in)) × exp(-τ_out × cos(out, ema_out))
    
    gelu71 (val PPL 174.9 — CURRENT BEST) uses a SINGLE cosine gate + surprise:
        gate = exp(-τ × cos(out, ema_out)) × (1 + w × surprise_in)
    
    These are two ORTHOGONAL improvements over gelu28's single cosine:
    - gelu31:  directional familiarity from input space AND output space
    - gelu71:  directional familiarity + input deviation magnitude
    
    gelu76 combines ALL THREE signals:
        gate = exp(-τ_in × cos(x, ema_in)) × exp(-τ_out × cos(out, ema_out))
               × (1 + w × surprise)
        
    Then contrast normalize for energy preservation.

SIGNAL DECOMPOSITION:
    exp(-τ_in × cos(x, ema_in)):
        Suppresses tokens whose INPUT DIRECTION matches the historical mean direction.
        Even if the magnitude is unusual, if the direction is familiar → attenuated.
    
    exp(-τ_out × cos(out, ema_out)):
        Suppresses tokens whose OUTPUT DIRECTION (after GELU nonlinearity) is familiar.
        Catches cases where an unusual input produces a familiar-looking output.
    
    (1 + w × surprise):
        Restores transmission for tokens where the input MAGNITUDE deviates,
        regardless of direction. "Even if this direction is common, the value is
        unusually large/small → signal this."

WHEN IS GATE HIGH? (most open transmission)
    - Input direction: novel (cos_in ≈ -1) → first term ≈ exp(+τ_in) >> 1 ← BUT amplified by contrast norm
    - Output direction: novel (cos_out ≈ -1) → second term ≈ exp(+τ_out) >> 1
    - Surprise: input deviates from history → third term > 1
    
    Triple novelty = maximum gate → with contrast norm, significantly above 1.0.

WHEN IS GATE LOW? (most suppressed)
    - Input direction: familiar (cos_in ≈ 1) → first term ≈ exp(-τ_in) ≈ 0
    - Output direction: familiar (cos_out ≈ 1) → second term ≈ exp(-τ_out) ≈ 0
    - Surprise: expected magnitude → third term ≈ 1 (no rescue)
    
    Triple familiarity = near-zero gate → maximally suppressed.

COMPLEXITY:
    6 parameters total (logit_decay, log_tau_in, log_tau_out, log_blend, log_sigma_raw, log_w_raw).
    State: _ema_in (D,), _ema_out (D,), both unit vectors.
    One extra state vs gelu71: _ema_in tracks input direction EMA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU76(nn.Module):
    """gelu31 backbone (double cosine) × gelu71 surprise boost, contrast-normalized."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau_in    = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_tau_out   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_in:     torch.Tensor = None   # unit vector, input direction
        self._ema_out:    torch.Tensor = None   # unit vector, output direction
        self._ema_x:      torch.Tensor = None   # raw, for deviation
        self._ema_x_norm: float = 1.0
        self._ready = False

    def reset_state(self):
        self._ema_in  = None
        self._ema_out = None
        self._ema_x   = None
        self._ready   = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val  = torch.sigmoid(self.logit_decay).detach().item()
        tau_in = self.log_tau_in.exp()
        tau_out = self.log_tau_out.exp()
        sigma  = F.softplus(self.log_sigma_raw)
        w      = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xm = x.detach().flatten(0,1).mean(0)
                om = out.detach().flatten(0,1).mean(0)
                self._ema_in     = F.normalize(xm, dim=0).clone()
                self._ema_out    = F.normalize(om, dim=0).clone()
                self._ema_x      = xm.clone()
                self._ema_x_norm = xm.norm().item() + self.eps
                self._ready      = True
            return out

        x_n    = F.normalize(x.detach(), dim=-1)
        out_n  = F.normalize(out.detach(), dim=-1)
        ei_n   = F.normalize(self._ema_in,  dim=0).view(1, 1, D)
        eo_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)

        cos_in  = (x_n   * ei_n ).sum(-1).clamp(-1, 1)
        cos_out = (out_n * eo_n ).sum(-1).clamp(-1, 1)

        # Double cosine gate (gelu31 backbone)
        gate_cos = torch.exp(-(tau_in * cos_in + tau_out * cos_out))

        # Input-deviation surprise (gelu71 signal)
        ema_x = self._ema_x.view(1, 1, D)
        delta = (x.detach() - ema_x).norm(dim=-1)
        surprise = torch.tanh(sigma * delta / (self._ema_x_norm + self.eps))

        gate_raw = gate_cos * (1.0 + w * surprise)

        # Contrast normalize
        with torch.no_grad():
            gate_mean = gate_raw.mean().clamp(min=self.eps)
        gate_norm = gate_raw / gate_mean
        output = out * gate_norm.unsqueeze(-1)

        with torch.no_grad():
            xm = x.detach().flatten(0,1).mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_in     = d_val * self._ema_in  + (1.0 - d_val) * F.normalize(xm, dim=0)
            self._ema_out    = d_val * self._ema_out + (1.0 - d_val) * F.normalize(om, dim=0)
            self._ema_x      = d_val * self._ema_x   + (1.0 - d_val) * xm
            self._ema_x_norm = self._ema_x.norm().item() + self.eps

        return output
