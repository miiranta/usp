"""GELU11 – Deviation Amplification: amplify novel, let GELU suppress familiar.

All previous approaches modified h via a multiplicative scale ≤ 1 (suppression).
The flaw: models learn to compensate by increasing weights — suppression is neutralised.
GELU8 (gradient-trained prototypes) was WORSE because prototypes learned useful patterns
and then suppressed exactly the information the model needed.

The insight: instead of suppressing familiar patterns, AMPLIFY deviations from expected.
The GELU nonlinearity then naturally handles suppression of below-expected activations.

Math:
    enhanced = h + α·d · (h - μ)       # α·d ∈ (0,1): blend of EMA decay × strength
             = (1 + α·d)·h - α·d·μ

Behaviour:
    h >> μ  (novel, above-mean):     enhanced > h  → GELU region shifts right → AMPLIFIED ✓
    h  < μ  (familiar, below-mean):  enhanced < h  → GELU pushes more negative → SUPPRESSED ✓
    h ≈ μ:  no change ✓

Key differences from all prior approaches:
    • enhanced ≥ h for above-mean activations (signal never removed)
    • GELU's existing nonlinearity is the gate — no extra multiplicative gate fighting gradients
    • Functionally equivalent to "soft de-biasing": subtract some expected mean
    • μ is per-channel EMA: correct reference, not noisy batch variance

Parameters: 3 (same as GELU2): logit_decay → d, log_blend → α, log_tau (kept for EMA τ init).
"""

import math
import torch
import torch.nn as nn


class GELU11(nn.Module):
    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._mu: torch.Tensor = None
        self._ready = False

        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))   # α init=0.3
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
        alpha = torch.sigmoid(self.log_blend)
        d     = torch.sigmoid(self.logit_decay)
        d_val = d.detach().item()
        blend = alpha * d                          # total amplification strength

        x_mean = x.detach().flatten(0, -2).mean(0)   # (D,)

        if not self._ready:
            self._mu    = x_mean
            self._ready = True
            return self._gelu(x)

        if self._mu.device != x.device:
            self._mu = self._mu.to(x.device)

        # Amplify deviation from expected: novel channels get pushed further into GELU
        # enhanced = (1 + blend)*x - blend*mu
        enhanced = x + blend * (x - self._mu)      # (B, T, D)

        self._mu = d_val * self._mu + (1.0 - d_val) * x_mean

        return self._gelu(enhanced)
