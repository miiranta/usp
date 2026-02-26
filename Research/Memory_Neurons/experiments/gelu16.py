"""GELU16 – Causal Temporal-Difference Surprise.

Biological inspiration: neural adaptation to change. A sensory neuron fires
strongly when its input CHANGES and suppresses its response to static inputs.
The retinal "on-center / off-center" ganglion cells are literally computing
temporal derivatives.

Within a sequence, this translates to:

    surprise[0]   = 1.0                        (no predecessor → fully novel)
    surprise[t]   = ||x[t] - x[t-1]||₂
                    ──────────────────          (cosine-inspired normalisation)
                    ||x[t]||₂ + ε

    novelty[t]  = σ(τ · surprise[t])          scalar ∈ (0,1)
    scale[t]    = (1 - α) + α · novelty[t]    scalar gate per token
    output[t]   = GELU(x[t] · scale[t])

Properties:
  • Still frame → surprise→0 → suppress (repetition suppression)
  • Sharp change → surprise large → amplify (change detection)
  • No EMA, no batch statistics, no prototype memory
  • Purely within-sequence causal: depends only on x[t] and x[t-1]
  • Works as a learned "velocity detector" on the token stream

Trainable params per layer: log_tau, log_blend (just 2 scalars). Minimal.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU16(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))   # sharpness
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # max gate strength

    def reset_state(self):
        pass   # stateless across sequences

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

        # ── Temporal difference (causal, zero-padded at t=0) ──────────
        # x_prev[t] = x[t-1]; x_prev[0] = 0 → large surprise for first token
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device, dtype=x.dtype),
                            x[:, :-1, :]], dim=1)           # (B, T, D)

        diff      = x - x_prev                              # (B, T, D)
        diff_norm = diff.norm(dim=-1)                       # (B, T)
        x_norm    = x.norm(dim=-1).clamp(min=1e-8)         # (B, T)

        surprise  = diff_norm / x_norm                      # (B, T) ≥ 0

        # Novelty in (0,1): large surprise → near 1
        novelty   = torch.sigmoid(tau * surprise)           # (B, T)

        scale     = (1.0 - alpha) + alpha * novelty         # (B, T) scalar per token
        scale     = scale.unsqueeze(-1)                      # (B, T, 1)

        return self._gelu(x * scale)
