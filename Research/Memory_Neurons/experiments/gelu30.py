"""GELU30 – Binary Hard-Threshold Suppression (Median Split Gate).

PROBLEM WITH ALL SOFT GATES (gelu2–gelu29):
Every prior experiment uses a SOFT monotone gate:
    novelty = exp(-τ · sim), or sigmoid(-λ · z), etc.
    scale = (1-α) + α · novelty   ∈ (1-α, 1]

The soft gate is:
  1. REDUNDANT for most tokens: when sim is moderate, novelty≈0.5, scale≈0.87 —
     a mild 13% suppression that probably has negligible effect on learning.
  2. SENSITIVE to τ calibration: with median cosine sim ≈ 0.5, exp(-2*0.5)=0.37,
     scale≈(1-0.3*0.9)+0.3*0.9*0.37≈0.73+0.1=0.83. Mild suppression on EVERYTHING.
  3. The gradient signal from the gate is spread across all tokens.

GELU30 uses HARD BINARY SUPPRESSION via median split:
    • Compute cosine similarity of each token to the EMA: sim (B, T)
    • Find the median sim across the batch
    • Tokens with sim > median: FAMILIAR → scale = 1 - α    (suppressed)
    • Tokens with sim ≤ median: NOVEL    → scale = 1         (full pass)

Result: EXACTLY 50% of tokens get suppressed at full strength α.
This is MAXIMUM CONTRAST:
  - The familiar half gets maximally suppressed
  - The novel half gets zero suppression (not even mild)
  - No wasted suppression on "middle" tokens

Why this might beat gelu2:
  gelu2's soft gate spreads mild suppression (scale≈0.78–0.90) across most tokens.
  Hard gate concentrates strong suppression (scale=1-α=0.7) on EXACTLY the top-50%
  most familiar tokens and leaves the rest completely undisturbed.
  This is a much sharper signal for the model to learn from:
  "these patterns are habitual, these patterns are not."

Why hard gates are safe (no gradient issues):
  The scale computation is FULLY DETACHED from the gradient:
    sim.detach(), median threshold = no-gradient
  Gradients flow through: loss → output → x * scale → x (where scale is constant).
  Same as all prior work.

A learnable α controls the suppression strength for the familiar half.
The median threshold is computed on-the-fly per batch (no stored state needed).

However, to stabilize the threshold, we ALSO maintain an EMA prototype:
    - The EMA gives a consistent reference direction (like gelu2)
    - The median split gives a consistent fraction (UNLIKE gelu2's variable soft gate)

Params per layer: logit_decay (d), log_blend (α) = 2 scalars (fewer than gelu2's 3)
State: 1 × D EMA vector (same as gelu2_k1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU30(nn.Module):
    """Binary hard-threshold gate: top-50% similar tokens suppressed at full strength."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema:   torch.Tensor = None   # (D,) EMA prototype
        self._ready: bool = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        # α: suppression strength for the familiar half, init at 0.3
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

    def reset_state(self):
        self._ema   = None
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
        d     = torch.sigmoid(self.logit_decay)
        alpha = torch.sigmoid(self.log_blend)
        d_val = d.detach().item()

        x_mean = x.detach().flatten(0, -2).mean(0)   # (D,) batch mean

        # ── Warm-up ──────────────────────────────────────────────────
        if not self._ready:
            self._ema   = x_mean.clone()
            self._ready = True
            return self._gelu(x)

        # ── Cosine similarity to EMA prototype ───────────────────────
        x_norm   = F.normalize(x, dim=-1)                             # (B, T, D)
        ema_norm = F.normalize(self._ema.unsqueeze(0), dim=-1)        # (1, D)
        sim      = (x_norm * ema_norm).detach().sum(-1)               # (B, T) detached

        # ── Median-split binary gate ──────────────────────────────────
        # Threshold: median cosine similarity across the batch
        threshold = sim.median()                                       # scalar
        # Binary: 1 if familiar (sim > median), 0 if novel
        is_familiar = (sim > threshold).float()                        # (B, T) ∈ {0.0, 1.0}
        # Scale: familiar tokens get (1-α), novel tokens get 1
        scale    = 1.0 - alpha * is_familiar                          # (B, T)
        scale    = scale.unsqueeze(-1)                                 # (B, T, 1)

        # ── Update EMA ────────────────────────────────────────────────
        self._ema = d_val * self._ema + (1.0 - d_val) * x_mean

        return self._gelu(x * scale)
