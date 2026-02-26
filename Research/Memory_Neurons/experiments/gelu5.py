"""GELU5 – Per-channel EMA habituation (correct sign).

Fixes both prior failures:
  • GELU3 bug: novelty = exp(-τ·deviation) → suppressed NOVEL instead of FAMILIAR
  • GELU4: learned linear gate → completely different mechanism, extra parameters

Core idea (faithful to habituation hypothesis):
  familiarity_j = exp(-τ · |x_j - μ_j| / (ρ_j + ε))   ← HIGH when x is near EMA mean
  novelty_j     = 1 - familiarity_j                       ← HIGH when x is far from mean
  scale_j       = (1 - α·d) + α·d · novelty_j            ← suppress familiar, pass novel
  output        = GELU(x · scale)

EMA state (not parameters, no gradients):
  μ_j  = EMA of x_j              (per-channel mean)
  ρ_j  = EMA of |x_j|            (per-channel magnitude, used for scale normalization)

Learnable parameters (same 3 as GELU2, same counts):
  logit_decay → d ∈ (0,1) : EMA memory length
  log_tau     → τ > 0     : suppression sharpness
  log_blend   → α ∈ (0,1) : max suppression depth

Parameter count vs control: +3 scalars per layer (identical to GELU2).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU5(nn.Module):
    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._mu:  torch.Tensor = None   # (D,) per-channel EMA mean
        self._rho: torch.Tensor = None   # (D,) per-channel EMA of |x| (for normalisation)
        self._ready = False

        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )

    def reset_state(self):
        self._mu    = None
        self._rho   = None
        self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)
        d     = torch.sigmoid(self.logit_decay)
        d_val = d.detach().item()

        x_flat = x.detach().flatten(0, -2)          # (B*T, D)
        x_mean = x_flat.mean(0)                      # (D,)
        x_rho  = x_flat.abs().mean(0).clamp(min=1e-6)  # (D,)

        if not self._ready:
            self._mu    = x_mean
            self._rho   = x_rho
            self._ready = True
            return self._gelu(x)

        # Normalised deviation: near 0 when x ≈ μ, grows as x diverges
        norm_dev    = (x - self._mu).abs() / (self._rho + 1e-6)     # (B, T, D)

        # familiarity ∈ (0,1]: HIGH when x is near its expected value
        familiarity = torch.exp(-tau * norm_dev)                     # (B, T, D)

        # novelty ∈ [0,1): HIGH when x deviates from expectation
        novelty = 1.0 - familiarity

        # gate: familiar activations are scaled down, novel ones pass through
        scale   = (1.0 - alpha * d) + alpha * d * novelty           # (B, T, D)
        blended = x * scale

        # Update EMA buffers
        self._mu  = d_val * self._mu  + (1.0 - d_val) * x_mean
        self._rho = d_val * self._rho + (1.0 - d_val) * x_rho

        return self._gelu(blended)
