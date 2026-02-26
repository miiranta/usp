"""GELU3 – Per-channel EMA habituation.

Each feature dimension j maintains its own EMA mean (μ_j) and variance (σ²_j).
Novelty for token (b,t) in channel j:

    novelty_j = exp(-τ * (x[b,t,j] - μ_j)² / σ²_j)

Scale gate per channel:
    scale_j = (1 - α·d) + α·d · novelty_j

output = GELU(x * scale)

This is much more fine-grained than GELU2's single prototype vector:
it independently characterises "expected" vs "novel" for every feature dimension.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU3(nn.Module):
    """Per-channel EMA habituation GELU.

    Tracks per-channel running mean and variance.
    Suppresses channels whose activation is near their expected value;
    lets novel (off-mean) activations through at full strength.

    Learnable parameters (same as GELU2):
        logit_decay → d ∈ (0,1)
        log_tau     → τ > 0
        log_blend   → α ∈ (0,1)
    """

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._mu:  torch.Tensor = None   # (D,)  per-channel mean
        self._var: torch.Tensor = None   # (D,)  per-channel variance
        self._ready = False

        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend   = nn.Parameter(
            torch.tensor(math.log(0.3 / 0.7))   # logit of 0.3
        )
        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )

    def reset_state(self):
        self._mu    = None
        self._var   = None
        self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)
        d     = torch.sigmoid(self.logit_decay)
        d_val = d.detach().item()

        # Per-channel stats over (B, T) tokens
        x_flat   = x.detach().flatten(0, -2)          # (B*T, D)
        x_mean   = x_flat.mean(0)                      # (D,)
        x_var    = x_flat.var(0, unbiased=False).clamp(min=1e-6)  # (D,)

        if not self._ready:
            self._mu    = x_mean
            self._var   = x_var
            self._ready = True
            return self._gelu(x)

        # Per-token, per-channel novelty
        # deviation: how many "sigma" away is each activation from its expected value
        deviation = (x - self._mu).pow(2) / self._var   # (B, T, D) via broadcast
        novelty   = torch.exp(-tau * deviation)           # (B, T, D)

        scale   = (1.0 - alpha * d) + alpha * d * novelty   # (B, T, D)
        blended = x * scale

        # Update EMA
        self._mu  = d_val * self._mu  + (1.0 - d_val) * x_mean
        self._var = d_val * self._var + (1.0 - d_val) * x_var

        return self._gelu(blended)
