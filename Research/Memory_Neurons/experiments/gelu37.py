"""GELU37 – Predictive Coding Error Amplification.

MOTIVATION:
Every previous experiment gates DOWN familiar signals (multiplicative suppression).
This experiment takes the opposite route: it AMPLIFIES the prediction error.

NEUROSCIENCE INSPIRATION:
Predictive coding (Rao & Ballard 1999): neurons don't transmit raw activations – they
transmit RESIDUALS from top-down predictions.  What is sent upward is the SURPRISE.

MECHANISM:
    out_raw  = GELU(x)                                      (B, T, D)

    # Track running expectation (what does the typical output look like?)
    ema_out ← d * ema_out + (1-d) * mean_BT(out_raw)       (D,)

    # Compute prediction error (deviation from expectation)
    error   = out_raw - ema_out                             (B, T, D)

    # Output = raw signal + alpha * error
    #        = (1 + alpha) * out_raw - alpha * ema_out
    output  = out_raw + alpha * error

WHY THIS IS DIFFERENT FROM ALL PREVIOUS EXPERIMENTS:
- gelu28 and friends: output = out_raw * scale  (scale ≤ 1, dampening)
- gelu37:             output = out_raw + alpha * error  (AMPLIFYING the novel part)
  At alpha=0: plain GELU.
  At alpha>0: each token's output is BOOSTED proportionally to how much it
              DEVIATES from the running average.  Familiar tokens stay near
              the EMA → small boost.  Novel tokens deviate strongly → large boost.

This turns the gate into an ERROR AMPLIFIER rather than a FAMILIARITY SUPPRESSOR.
Same information theoretically, but the gradient landscape is very different —
the model is incentivized to make outputs that differ from the mean, encouraging
more informative and discriminative representations.

Params: logit_decay (1 scalar), log_alpha (1 scalar) = 2 learnable scalars.
State:  ema_out (D,) – cross-batch EMA of GELU output mean.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU37(nn.Module):
    """Predictive Coding Error Amplifier: output = GELU(x) + alpha*(GELU(x)-EMA)."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema_out: torch.Tensor = None
        self._ready = False

        # Learnable EMA decay  d = sigmoid(logit_decay) ∈ (0,1)
        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        # Learnable amplification coefficient  alpha = softplus(log_alpha_raw)
        # initialised to alpha ≈ 0.3
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))

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
        out_raw = self._gelu(x)            # (B, T, D)

        d     = torch.sigmoid(self.logit_decay)
        alpha = F.softplus(self.log_alpha_raw)  # ≥ 0

        # First forward: initialise EMA and return plain GELU
        with torch.no_grad():
            out_mean = out_raw.detach().mean(dim=(0, 1))  # (D,)
            if not self._ready:
                self._ema_out = out_mean.clone()
                self._ready   = True
                return out_raw
            # Update EMA
            d_val = d.detach().item()
            self._ema_out = d_val * self._ema_out + (1.0 - d_val) * out_mean

        # Amplify prediction error
        # error: (B, T, D)  –  how much this token deviates from the expected pattern
        error  = out_raw - self._ema_out.unsqueeze(0).unsqueeze(0)
        output = out_raw + alpha * error   # = (1+alpha)*out_raw - alpha*ema

        return output
