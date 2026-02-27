"""GELU73 – Contrast-Normalized Surprise×Cosine Gate (gelu71 + novelty amplification).

WHAT gelu71 STILL LACKS:
    gelu71's gate = cos_gate × (1 + w × surprise)
    
    When surprise ≈ 0 (all tokens are expected): gate ≈ cos_gate ≤ 1 always.
    The AVERAGE gate < 1, meaning total output energy shrinks below plain GELU.
    Familiar tokens are suppressed but novel tokens are merely "not suppressed."
    
    Example (τ=2, w=1, surprise uniform in [0,0.5]):
        familiar token: cos=0.9 → cos_gate=0.16 → gate ≈ 0.16 × 1.25 = 0.20
        novel token:    cos=0.0 → cos_gate=1.0  → gate ≈ 1.0  × 1.25 = 1.25
        mean gate ≈ 0.7 → net energy loss ~30%
    
    The residual stream partially compensates but at the cost of diluting the
    novelty contrast — when everything is scaled by 0.7, the contrast ratio
    between familiar (0.20) and novel (1.25) is 6.25x, but only 4.4x after scaling.

THE FIX — DIVIDE BY MEAN:
    gate_norm[b,t] = gate_raw[b,t] / mean_{b,t}(gate_raw)
    output[b,t]    = GELU(x[b,t]) × gate_norm[b,t]
    
    Now:
    - mean output energy = mean GELU(x) energy (no net energy change)
    - familiar token: gate_norm ≈ 0.20/0.70 = 0.29 (MORE suppressed relatively)
    - novel token:    gate_norm ≈ 1.25/0.70 = 1.79 (AMPLIFIED above gelu baseline!)
    - contrast ratio: 6.25x (same) but now expressed as 0.29 vs 1.79 around 1.0
    
    The model sees novel tokens with 79% BOOST rather than just "not suppressed."
    This is a fundamentally better novelty detector signal for downstream attention.

ANALOGY — RETINAL GANGLION CELLS:
    Center-surround receptive fields subtract the mean luminance from their field.
    A pixel at mean luminance → zero response.
    A bright pixel on dark background → strong positive response (amplified).
    A dark pixel on bright background → strong negative response (suppressed).
    The mean luminance is the "EMA level"; deviations are contrast-enhanced.

IMPLEMENTATION NOTE:
    mean is computed WITHOUT gradients — it's a normalization constant only.
    Gradient flows through output = GELU(x) × gate_raw/mean = GELU(x) × gate_raw × (1/mean).
    Since (1/mean) is a constant (no grad), the gradient w.r.t. gate parameters
    is identical to gelu71, so the same learning dynamics apply.
    The only difference is the output scale, which is more stable.

Params: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars (same as gelu71).
State: _ema_x (D,), _ema_out (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU73(nn.Module):
    """gelu71 + contrast normalization: familiar suppressed, novel amplified above 1.0."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_x:      torch.Tensor = None
        self._ema_out:    torch.Tensor = None
        self._ema_x_norm: float = 1.0
        self._ready = False

    def reset_state(self):
        self._ema_x    = None
        self._ema_out  = None
        self._ready    = False

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
                xm = x.detach().flatten(0,1).mean(0)
                self._ema_x      = xm.clone()
                self._ema_x_norm = xm.norm().item() + self.eps
                self._ema_out    = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
                self._ready      = True
            return out

        # Cosine familiarity gate
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        # Input-deviation surprise
        ema_x = self._ema_x.view(1, 1, D)
        delta = (x.detach() - ema_x).norm(dim=-1)
        surprise = torch.tanh(sigma * delta / (self._ema_x_norm + self.eps))

        # Combined gate (same as gelu71)
        gate_raw = gate_cos * (1.0 + w * surprise)

        # ── Contrast normalization: mean over B×T = 1.0 ────────────────────
        with torch.no_grad():
            gate_mean = gate_raw.mean().clamp(min=self.eps)
        gate_norm = gate_raw / gate_mean          # (B, T) — mean=1, novel > 1

        output = out * gate_norm.unsqueeze(-1)

        with torch.no_grad():
            xm = x.detach().flatten(0,1).mean(0)
            self._ema_x      = d_val * self._ema_x + (1.0 - d_val) * xm
            self._ema_x_norm = self._ema_x.norm().item() + self.eps
            bm = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
            self._ema_out = d_val * self._ema_out + (1.0 - d_val) * bm

        return output
