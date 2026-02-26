"""GELU21 – Cross-Position Channel Uniqueness.

A channel's value at position t is "familiar" if it's UNIFORM across positions
in this sequence — it's carrying the same information everywhere, i.e., a
global feature with no local discriminative power.

A channel's value at position t is "novel" if it DEVIATES from that channel's
mean across positions — it's carrying position-specific, discriminative information.

    mu_d     = mean_t x_d[t]              (B, D)   per-channel sequence mean
    sigma_d  = std_t  x_d[t] + ε         (B, D)   per-channel std across positions
    novelty_d[t] = |x_d[t] - mu_d| / sigma_d      (B, T, D)  z-score per channel

    gate_d[t]    = 1 - exp(-τ · novelty_d[t])     (B, T, D)  sigmoid-like ∈ [0,1)
    scale_d[t]   = (1-α) + α · gate_d[t]
    output       = GELU(x * scale_d)

Suppressed: channels whose values are similar across all positions
            (they're encoding document-level or global features uniformly)
Amplified:  channels that deviate from their sequence average
            (they're encoding position-specific, local, discriminative features)

Why this is different from everything else:
  • No cross-batch EMA, no cross-sequence statistics
  • Familiarity is defined WITHIN THIS SEQUENCE, per-CHANNEL
  • A channel can simultaneously be familiar in one sequence and novel in another
  • This detects "what is unique about THIS position in THIS context"

Unlike GELU18 (which compares token vectors via cosine), GELU21 compares
each channel to its OWN mean, giving per-channel rather than per-token novelty.

The sequence mean/std are computed with gradient flow off (detached),
so the gate is purely a modulation signal with no interference with the
backpropagation of the language model loss through x.

Params per layer: log_tau, log_blend (2 scalars). Zero extra memory.
Compute: one mean and std over T — O(T·D) = same as one LayerNorm.
"""

import math
import torch
import torch.nn as nn


class GELU21(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))  # EMA d=0.9
        self._ema_mean: torch.Tensor = None
        self._ema_var:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None; self._ema_var = None; self._ready = False

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
        d_val = torch.sigmoid(self.logit_decay).detach().item()

        # Cross-batch EMA of per-channel mean and variance — causally valid
        with torch.no_grad():
            batch_mean = x.detach().mean(dim=(0, 1))   # (D,)
            batch_var  = x.detach().var(dim=(0, 1), correction=0).clamp(min=1e-8)
            if not self._ready:
                self._ema_mean = batch_mean.clone()
                self._ema_var  = batch_var.clone()
                self._ready    = True
            else:
                self._ema_mean = d_val * self._ema_mean + (1 - d_val) * batch_mean
                self._ema_var  = d_val * self._ema_var  + (1 - d_val) * batch_var

        mu    = self._ema_mean.unsqueeze(0).unsqueeze(0)               # (1, 1, D)
        sigma = self._ema_var.sqrt().unsqueeze(0).unsqueeze(0)         # (1, 1, D)

        z_score = (x - mu).abs() / sigma                               # (B, T, D)
        novelty = 1.0 - torch.exp(-tau * z_score)
        scale   = (1.0 - alpha) + alpha * novelty
        return self._gelu(x * scale)
