"""GELU10 – Per-channel EMA sign-agreement habituation.

GELU2's weakness: cosine similarity collapses the (B,T,D) familiarity to (B,T,1)
— all 1024 channels get the same gate, even if only 50 of them are "familiar."

GELU5's weakness: used |x_j - μ_j| / ρ_j — fine-grained, but the EMA mean μ_j
converges to the DC offset of each channel, not to the "expected direction," so 
a large-but-typical activation gets called "novel" when it's actually familiar.

Fix: use sign agreement between x and μ as the familiarity signal.
  - Channel j is FAMILIAR when x_j and μ_j have the SAME sign AND are both large
  - Channel j is NOVEL when the activation direction is unexpected

Smooth formulation:
    agree_j     = tanh(β · x_j) · tanh(β · μ_j)   ∈ (-1, 1)
                  ≈ sgn(x_j)·sgn(μ_j) for large β (hard sign agreement)
    familiarity_j = agree_j.clamp(min=0)            ∈ [0, 1)   zero if opposite sign
    novelty_j   = exp(-τ · familiarity_j)           ∈ (0, 1]   low when familiar
    scale_j     = (1 - α·d) + α·d · novelty_j
    output      = GELU(x · scale)

Why this is better than GELU5:
  - GELU5: |x_j - μ_j| / |μ_j| reacts to magnitude deviations. A channel that
    normally outputs 0.5 and now outputs 1.0 is "novel" even if same direction.
  - GELU10: sign agreement is DIRECTION-based. A channel firing in the expected
    direction is familiar (regardless of magnitude). Only unexpected sign-flip = novel.

Direction is more meaningful for rectified activations (most values are small;
large-magnitude events in unexpected directions are the true novelty).

Parameters: 3 per layer (same as GELU2) — logit_decay, log_tau, log_blend.
Extra params vs control: +3 scalars per layer × 4 layers = 12 scalars total.
"""

import math
import torch
import torch.nn as nn


class GELU10(nn.Module):
    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._mu:  torch.Tensor = None   # (D,) per-channel EMA mean
        self._ready = False

        # β=5 → tanh slope ≈ sgn but smooth gradient near 0
        self.log_beta    = nn.Parameter(torch.tensor(math.log(5.0)))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )

    def reset_state(self):
        self._mu    = None
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
        beta  = self.log_beta.exp()
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)
        d     = torch.sigmoid(self.logit_decay)
        d_val = d.detach().item()

        x_mean = x.detach().flatten(0, -2).mean(0)    # (D,)

        if not self._ready:
            self._mu    = x_mean
            self._ready = True
            return self._gelu(x)

        # Per-channel sign agreement: tanh(β*x_j) * tanh(β*μ_j) ∈ (-1, 1)
        agree       = torch.tanh(beta * x) * torch.tanh(beta * self._mu)  # (B, T, D)

        # Positive agreement = familiar; negative agreement = novel (unexpected sign)
        familiarity = agree.clamp(min=0.0)                                # (B, T, D) ∈ [0,1)
        novelty     = torch.exp(-tau * familiarity)                       # (B, T, D) ∈ (0,1]
        scale       = (1.0 - alpha * d) + alpha * d * novelty            # (B, T, D)

        # Update EMA mean
        self._mu = d_val * self._mu + (1.0 - d_val) * x_mean

        return self._gelu(x * scale)
