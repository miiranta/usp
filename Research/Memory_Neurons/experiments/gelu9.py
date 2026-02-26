"""GELU9 – Causal within-sequence habituation.

Most faithful interpretation of the habituation hypothesis for a causal LM:
at position t, suppress activations that are familiar relative to what this
sequence has shown so far (positions 0..t-1).

    causal_mean[t] = mean(x[0], ..., x[t-1])     # cumulative prefix mean
                   = 0 at t=0 (no history yet)

    norm_dev[t]    = |x[t] - causal_mean[t]| / (causal_ref[t] + ε)
                     where causal_ref = cumulative mean of |x|

    familiarity[t] = exp(-τ * norm_dev[t])        # HIGH when x[t] ≈ prefix mean
    novelty[t]     = 1 - familiarity[t]           # HIGH when x[t] deviates
    scale[t]       = (1 - α) + α * novelty[t]
    output[t]      = GELU(x[t] * scale[t])

Properties:
  • Causal: at t=0 no history → no suppression → plain GELU (correct fallback)
  • Gradients flow freely (no detach needed except for cum-mean computation)
  • O(T·D) cost: just a cumsum + normalization, no attention
  • No cross-batch state
  • 2 learnable params per layer (+0 vs GELU6, same as GELU2 without decay)
"""

import math
import torch
import torch.nn as nn


class GELU9(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.5 / 0.5)))  # α=0.5 init

    def reset_state(self):
        pass  # stateless

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
        B, T, D = x.shape
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)

        with torch.no_grad():
            # Cumulative sum shifted by 1: prefix[t] = mean(x[0..t-1])
            # cumsum shape: (B, T, D) — pad a zero at the front, drop the last
            x_cs      = x.cumsum(dim=1)                           # (B, T, D)
            x_abs_cs  = x.abs().cumsum(dim=1)                     # (B, T, D)
            counts    = torch.arange(1, T + 1, device=x.device,
                                     dtype=x.dtype).view(1, T, 1)  # (1, T, 1)

            # Shift: prefix mean at position t = sum of 0..t-1 / t
            # = cumsum shifted right by 1 (first position has no history → 0)
            cum_mean  = torch.cat([
                torch.zeros(B, 1, D, device=x.device, dtype=x.dtype),
                x_cs[:, :-1]
            ], dim=1) / counts.clamp(min=1)                       # (B, T, D)

            cum_ref   = torch.cat([
                torch.ones(B, 1, D, device=x.device, dtype=x.dtype) * 1e-6,
                x_abs_cs[:, :-1]
            ], dim=1) / counts.clamp(min=1)                       # (B, T, D)
            cum_ref   = cum_ref.clamp(min=1e-6)

        # Normalised deviation from causal prefix mean
        norm_dev    = (x - cum_mean).abs() / cum_ref               # (B, T, D)

        familiarity = torch.exp(-tau * norm_dev)                   # (B, T, D)
        novelty     = 1.0 - familiarity                            # (B, T, D)
        scale       = (1.0 - alpha) + alpha * novelty              # (B, T, D)

        return self._gelu(x * scale)
