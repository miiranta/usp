"""GELU23 – EMA Threshold Shift (Adaptive Spike Threshold).

EVERY prior GELU variant (gelu2–21) uses the same mathematical pattern:
    scale = f(familiarity)   ∈ (0, 1]
    output = GELU(x · scale)

They differ only in how familiarity is computed. But the OPERATION is always
scalar multiplication before GELU.

GELU23 uses a fundamentally different operation: SUBTRACTION to shift the
THRESHOLD of the GELU nonlinearity.

    output = GELU(x − α · EMA_mean)

Biological grounding (spike threshold adaptation):
  Real neurons do not just reduce their gain — they raise their FIRING THRESHOLD.
  When a neuron receives persistent input at level μ, its threshold rises to μ,
  so only inputs EXCEEDING μ produce firing. This is "adaptation" or "habituation"
  at the membrane level.

  Mathematically: GELU(x) has its main transition near x=0 (the "threshold").
  By subtracting α·EMA from the input, we shift this threshold to α·EMA:
    - x ≈ EMA_mean → input to GELU ≈ (1-α)·x ≈ near zero → suppressed
    - x >> EMA_mean → input to GELU ≈ x → full GELU output → passed
    - x << EMA_mean → input to GELU is negative/small → doubly suppressed

Why this is different from multiplication:
  Multiplication (GELU(x·s)) preserves the SHAPE of the input, just shrinks it.
  Subtraction (GELU(x−shift)) CHANGES WHICH VALUES CROSS THE THRESHOLD.
  A token with x = EMA gets full zero output, not just a smaller version.
  This is qualitatively different: the nonlinearity is applied to the DEVIATION
  from the familiar mean, not to a scaled version of x.

  Analogy: sensory adaptation. After staring at grey, a slight colour appears
  vivid. The threshold has shifted to the adapted state.

Params per layer: logit_decay (d), log_blend (α) = 2 scalars  ← fewer than gelu2's 3
State: D-dimensional EMA mean vector
"""

import math
import torch
import torch.nn as nn


class GELU23(nn.Module):
    """EMA Threshold Shift: GELU(x − α·EMA_mean)."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema:   torch.Tensor = None   # (D,) running mean
        self._ready: bool = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_blend = nn.Parameter(
            torch.tensor(math.log(0.3 / 0.7))   # α = 0.3 init
        )

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

        x_mean = x.detach().flatten(0, -2).mean(0)   # (D,)

        # ── Warm-up ──────────────────────────────────────────────────
        if not self._ready:
            self._ema   = x_mean
            self._ready = True
            return self._gelu(x)

        # ── Shift GELU threshold by α × EMA_mean ────────────────────
        # Shift shape: (D,) → broadcasts across (B, T, D)
        shift  = alpha * self._ema             # (D,) — how much to lift the threshold
        output = self._gelu(x - shift)          # suppress activations near/below mean

        # ── Update EMA ───────────────────────────────────────────────
        self._ema = d_val * self._ema + (1.0 - d_val) * x_mean

        return output
