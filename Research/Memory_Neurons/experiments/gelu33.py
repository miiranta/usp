"""GELU33 – Adaptive Learnable Suppression Curve.

MOTIVATION:
gelu28 uses: scale = (1-α) + α·exp(-τ·sim)
The shape of exp(-τ·sim) as a function of sim is FIXED — exponential decay.
The model can only learn the AMPLITUDE (τ, α) of this fixed curve.

INSIGHT: Let the model LEARN THE SHAPE of the suppression curve.
Replace exp(-τ·sim) with a learnable sigmoid: sigmoid(−w·sim − b)
This is a generalized logistic gate parameterized by:
  w > 0: suppress familiar (sim=+1→ low gate), amplify novel (sim=−1→ high gate)
  b:     threshold/bias of the familiarity boundary
  w < 0: reverse (anti-familiarity mode — gradient will discover direction)

MECHANISM:
    out_raw  = GELU(x)                                  (B, T, D)
    out_sim  = cosine(out_raw, ema_out)                 (B, T)

    # Learnable suppression curve (generalized sigmoid gate)
    novelty  = sigmoid(-w · out_sim - b)                (B, T)
                ↑ when sim→+1: sigmoid(-w-b) → low (suppress familiar)
                ↑ when sim→-1: sigmoid(+w-b) → high (preserve novel)

    # Blend: guarantee minimum passthrough (1-α floor)
    α        = sigmoid(log_blend)                        scalar
    scale    = (1 - α) + α · novelty                   (B, T)
    output   = out_raw · scale

The key is that (w, b) jointly determine:
  - WHERE the suppression threshold falls on the sim axis
  - HOW STEEP the transition from "novel" to "familiar" is
  - Whether it's inverted (w<0 → amplify familiar, suppress novel)

Gradient flow teaches the model exactly HOW to respond to familiarity.
Different layers may learn different w values → richer per-layer habituation.

Params: logit_decay, w, b, log_blend = 4 scalars (same as gelu2_k1).
State:  ema_out (D_FF,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU33(nn.Module):
    """Learnable sigmoid suppression curve — adaptive familiarity gate."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema_out: torch.Tensor = None
        self._ready = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        # w>0: suppress familiar. Init at 2.0 (moderate slope)
        self.w         = nn.Parameter(torch.tensor(2.0))
        # b: bias of suppression threshold. Init at 0 (symmetric around sim=0)
        self.b         = nn.Parameter(torch.tensor(0.0))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

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
        # x: (B, T, D)
        B, T, D = x.shape
        out_raw = self._gelu(x)   # (B, T, D)

        # --- Update output EMA ---
        with torch.no_grad():
            out_mean = out_raw.detach().mean(dim=(0, 1))   # (D,)
            d = torch.sigmoid(self.logit_decay)
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

        # --- Learnable suppression curve ---
        # novelty ∈ (0,1): high when sim is LOW (novel), low when sim is HIGH (familiar)
        novelty  = torch.sigmoid(-self.w * out_sim - self.b)   # (B, T)

        # --- Blend with floor ---
        alpha  = torch.sigmoid(self.log_blend)
        scale  = (1.0 - alpha) + alpha * novelty               # (B, T)

        return out_raw * scale.unsqueeze(-1)
