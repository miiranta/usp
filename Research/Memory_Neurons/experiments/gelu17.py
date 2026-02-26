"""GELU17 – Per-Channel Predictive Coding Gate.

Predictive coding is the leading neuroscience theory of cortical computation:
the brain continuously predicts its own inputs, and only the PREDICTION ERROR
(the "surprise") propagates upward. Familiar patterns generate small errors
and get suppressed. Novel patterns generate large errors and get amplified.

Here we implement a learned first-order autoregressive predictor for each
activation channel d:

    pred_d[t]  = w_d · x_d[t-1]           (AR(1) with learned per-channel gain)
    error_d[t] = x_d[t] - pred_d[t]       (prediction residual)
    novelty_d  = 1 - exp(-τ · |error_d| / (|x_d[t]| + ε))   ∈ [0,1]
    scale_d    = (1 - α) + α · novelty_d  (per-channel gate)
    output     = GELU(x · scale_d)

For t=0 (no predecessor): pred = 0 → error = x → high novelty → pass-through.

Why per-channel?
  • Some dimensions are highly predictable (positional/syntactic structure)
  • Others are volatile (content-bearing)
  • The predictor learns which is which and suppresses only the predictable ones

The AR weight w_d is learned per channel — this means the model discovers
HOW to predict each channel from its previous value. w_d≈1 = stable channel
(suppress), w_d≈0 = unpredictable channel (never suppress), w_d≈-1 =
oscillating channel (suppress when oscillating normally).

Params per layer: D (predictor weights) + log_tau + log_blend = D+2.
With D=1024 and 4 layers: 4×1026 ≈ 4k extra params (~0.04% overhead).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU17(nn.Module):
    def __init__(self):
        super().__init__()
        self._w_pred: nn.Parameter = None  # lazily initialised: (D,)

        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

    def reset_state(self):
        pass   # fully stateless: prediction is computed within the sequence

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

        # ── Lazy init of predictor weights ────────────────────────────
        if self._w_pred is None:
            # Init near 1: the trivial predictor is "predict same as before"
            self._w_pred = nn.Parameter(
                torch.ones(D, device=x.device, dtype=x.dtype)
            )

        if self._w_pred.device != x.device:
            self._w_pred = nn.Parameter(self._w_pred.to(x.device))

        w = self._w_pred                                    # (D,)

        # ── Causal AR(1) prediction ───────────────────────────────────
        # x_prev[t] = x[t-1], x_prev[0] = 0 (no prior context)
        x_prev = torch.cat([
            torch.zeros(B, 1, D, device=x.device, dtype=x.dtype),
            x[:, :-1, :]
        ], dim=1)                                           # (B, T, D)

        pred  = w * x_prev                                  # (B, T, D)  AR(1)
        error = x - pred                                    # (B, T, D)  residual

        # Per-channel normalised absolute error
        abs_err = error.abs()                               # (B, T, D)
        abs_x   = x.abs().clamp(min=1e-8)                  # (B, T, D)
        rel_err = abs_err / abs_x                          # (B, T, D)  ∈ [0, ∞)

        # Novelty: saturates to 1 for large errors, near 0 for perfect prediction
        novelty = 1.0 - torch.exp(-tau * rel_err)           # (B, T, D)  ∈ [0, 1)

        scale   = (1.0 - alpha) + alpha * novelty           # (B, T, D)

        return self._gelu(x * scale)
