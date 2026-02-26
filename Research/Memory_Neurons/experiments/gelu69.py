"""GELU69 – Contrast-Normalized Double Cosine Gate (gelu31 + Novelty Amplification).

THE LIMITATION OF GELU31:
    gelu31's gate = exp(-τ_in * cos(x, ema_in)) * exp(-τ_out * cos(out, ema_out))
    
    This gate is ALWAYS ≤ 1. The best case (novel token) is gate ≈ 1.0 — pass-through.
    Familiar tokens get gate ≈ 0 — suppressed.
    
    So gelu31 only SUPPRESSES. Novel tokens are not amplified, they're just "not hurt."
    The net effect: average output energy is REDUCED relative to plain GELU.
    The model compensates by up-weighting FF outputs in the residual, but this
    means the novelty signal is diluted by the scale adaptation.

THE FIX — CONTRAST NORMALIZATION:
    Normalize the gate so its MEAN over the batch×time is 1.0:
    
        gate_raw[b, t]   = exp(-τ_in * cos(x[b,t], ema_in)) * exp(-τ_out * cos(out[b,t], ema_out))
        gate_norm[b, t]  = gate_raw[b, t] / mean_{b',t'}(gate_raw)   → mean = 1.0
        output[b, t]     = GELU(x[b,t]) * gate_norm[b, t]

    NOW:
    - Familiar token: gate_raw ≈ 0   → gate_norm ≈ 0 → suppressed (same as before)
    - Average token:  gate_raw ≈ μ   → gate_norm ≈ 1.0 → unchanged
    - NOVEL token:    gate_raw ≈ max  → gate_norm > 1.0 → AMPLIFIED  ← new behavior!
    
    The model now has a CONTRAST-ENHANCED representation:
    familiar patterns are damped, novel patterns are boosted relative to mean.
    This is what column-level lateral inhibition achieves in primary visual cortex:
    the most active (familiar) columns suppress neighbors, amplifying rare patterns.

KEY PROPERTY — ZERO NET ENERGY CHANGE:
    mean(output) ≈ mean(GELU(x)) because mean(gate_norm) = 1.
    The residual connection doesn't need to compensate for energy loss.
    This is fundamentally more stable than gelu31 at training start.

BIOLOGICAL ANALOGY:
    Retinal ganglion cells: center-surround receptive fields create contrast enhancement.
    The "surround" suppresses familiar/average stimuli, the "center" passes novel ones.
    Net result: the MEAN luminance is subtracted; only deviations (edges, transients) pass.
    This is exactly what contrast normalization achieves.

IMPLEMENTATION DETAILS:
    - gate_raw uses the SAME cosine gate as gelu31 (proven to work at ep5)
    - Contrast normalization divides by the BATCH×TIME mean (not per-channel)
    - init: same EMA init as gelu31 (from first batch actuals)
    - The normalization mean is computed WITHOUT GRADIENTS (prevents gradient 
      flow through the normalization constant)
    - Fallback: if all gate_raw ≈ 0 (first call before EMA warms up), skip normalization

Params: logit_decay (EMA), log_tau_in, log_tau_out, log_blend = 4 scalars (same as gelu31).
State:  _ema_in (D,), _ema_out (D,), both cross-batch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU69(nn.Module):
    """Contrast-normalized double cosine gate: novelty amplified, familiarity suppressed,
    zero-mean energy change."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau_in   = nn.Parameter(torch.tensor(math.log(2.0)))   # sharpness for input cos
        self.log_tau_out  = nn.Parameter(torch.tensor(math.log(2.0)))   # sharpness for output cos
        self.log_blend    = nn.Parameter(torch.tensor(math.log(1.0)))   # how strongly to apply

        self._ema_in:  torch.Tensor = None
        self._ema_out: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_in  = None
        self._ema_out = None
        self._ready   = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau_in  = self.log_tau_in.exp()
        tau_out = self.log_tau_out.exp()
        blend   = torch.sigmoid(self.log_blend)   # ∈ (0,1): how strongly to apply gate

        out = self._gelu(x)   # (B, T, D)

        # ── Init (same as gelu31) ────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._ema_in  = F.normalize(x.detach().flatten(0, 1).mean(0), dim=0)
                self._ema_out = F.normalize(out.detach().flatten(0, 1).mean(0), dim=0)
                self._ready   = True
            return out   # first call: no suppression

        # ── Compute double cosine gate ───────────────────────────────────────
        x_n   = F.normalize(x.detach(), dim=-1)                    # (B, T, D)
        out_n = F.normalize(out.detach(), dim=-1)                   # (B, T, D)
        ei_n  = F.normalize(self._ema_in, dim=0).view(1, 1, D)
        eo_n  = F.normalize(self._ema_out, dim=0).view(1, 1, D)

        cos_in  = (x_n   * ei_n).sum(-1).clamp(-1, 1)             # (B, T)
        cos_out = (out_n * eo_n).sum(-1).clamp(-1, 1)             # (B, T)

        gate_raw = torch.exp(-(tau_in * cos_in + tau_out * cos_out))  # (B, T) ∈ (0,∞)

        # ── Contrast normalization: mean over B×T = 1.0 ─────────────────────
        with torch.no_grad():
            gate_mean = gate_raw.mean().clamp(min=self.eps)
        gate_norm = gate_raw / gate_mean                            # (B, T) mean = 1.0

        # ── Blend: blend=1 → full contrast gate; blend=0 → plain GELU ───────
        gate_final = (1.0 - blend) + blend * gate_norm              # (B, T)
        output = out * gate_final.unsqueeze(-1)                     # (B, T, D)

        # ── Update EMA states ────────────────────────────────────────────────
        with torch.no_grad():
            bm_in  = F.normalize(x.detach().flatten(0, 1).mean(0), dim=0)
            bm_out = F.normalize(out.detach().flatten(0, 1).mean(0), dim=0)
            self._ema_in  = d_val * self._ema_in  + (1.0 - d_val) * bm_in
            self._ema_out = d_val * self._ema_out + (1.0 - d_val) * bm_out

        return output
