"""GELU35 – Position-Progressive Suppression.

MOTIVATION:
All existing experiments apply the SAME familiarity gate to ALL positions in a
sequence (T=64). But linguistically, the role of a token changes with its position:

  - EARLY tokens (pos 0-10): establish context, topics, entities
  - LATE tokens (pos 50-64): should suppress repetition of context more aggressively
                              since they "already know" what this document is about

INSIGHT: The familiarity suppression should SCALE UP with position. Later positions
have already processed the context, so their outputs are MORE likely to be habitual
responses to the established context.

MECHANISM:
    out_raw  = GELU(x)                                       (B, T, D)
    out_sim  = cosine(out_raw, ema_out)                      (B, T)

    # Position-dependent exponent: positions 0→0, T-1→1 linearly
    pos_weight = arange(T) / (T - 1)                        (T,)  ∈ [0, 1]

    # Progressive novelty: novelty^(1 + γ·pos_weight)
    # When pos_weight=0 (first token): novelty^1 (standard)
    # When pos_weight=1 (last token):  novelty^(1+γ) (stronger suppression)
    # Since novelty ∈ (0,1), higher exponent → more suppression
    γ         = softplus(log_gamma)                          scalar ≥ 0 (learned)
    exponent  = 1.0 + γ · pos_weight                        (T,)
    novelty   = exp(-τ · out_sim)                            (B, T)
    scaled_nov = novelty ** exponent.unsqueeze(0)            (B, T)

    α         = sigmoid(log_blend)
    scale     = (1 - α) + α · scaled_nov                    (B, T)
    output    = out_raw · scale.unsqueeze(-1)               (B, T, D)

When γ→0: same as gelu28 (uniform suppression)
When γ>0: late tokens suppressed more aggressively

ZERO EXTRA TRAINING PARAMS vs gelu28 (only adds log_gamma = 1 extra param).
State: ema_out (D_FF,). Overall: 4 params total.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU35(nn.Module):
    """Position-progressive familiarity suppression: later positions → stronger gate."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema_out: torch.Tensor = None
        self._ready = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        # γ≥0: how much extra suppression at the last position
        # init: log(1) = 0 → γ=softplus(0)=0.693 (moderate progression)
        self.log_gamma   = nn.Parameter(torch.tensor(0.0))

    def reset_state(self):
        self._ema_out = None
        self._ready   = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out_raw = self._gelu(x)   # (B, T, D)
        d = torch.sigmoid(self.logit_decay)

        # --- Update EMA ---
        with torch.no_grad():
            out_mean = out_raw.detach().mean(dim=(0, 1))
            if not self._ready:
                self._ema_out = out_mean.clone()
                self._ready   = True
            else:
                self._ema_out = d * self._ema_out + (1 - d) * out_mean

        # --- Output cosine similarity ---
        out_flat = out_raw.reshape(B * T, D)
        out_sim  = F.cosine_similarity(
            out_flat, self._ema_out.unsqueeze(0).expand(B * T, -1), dim=-1
        ).reshape(B, T)   # (B, T)

        # --- Base novelty ---
        tau     = torch.exp(self.log_tau)
        novelty = torch.exp(-tau * out_sim)   # (B, T)  ∈ (0, 1)

        # --- Position-progressive exponent ---
        # pos_weight: 0 at pos=0, 1 at pos=T-1
        pos_weight = torch.arange(T, device=x.device, dtype=x.dtype) / max(T - 1, 1)  # (T,)
        gamma      = F.softplus(self.log_gamma)         # ≥ 0
        exponent   = 1.0 + gamma * pos_weight           # (T,)  ∈ [1, 1+γ]

        # Apply progressive exponent: later positions get ^(1+γ) instead of ^1
        scaled_nov = novelty ** exponent.unsqueeze(0)   # (B, T)

        alpha  = torch.sigmoid(self.log_blend)
        scale  = (1.0 - alpha) + alpha * scaled_nov     # (B, T)

        return out_raw * scale.unsqueeze(-1)
