"""GELU44 – Pre-activation Sequence Contrast.

gelu39 contrasts AFTER the nonlinearity:  output = GELU(x) + alpha*(GELU(x) - mu_gelu)
GELU44 contrasts BEFORE the nonlinearity: output = GELU(x + alpha*(x - mu_x))
                                                  = GELU((1+alpha)*x - alpha*mu_x)

WHY THE PLACEMENT MATTERS:
The nonlinearity GELU is not linear, so:
    GELU(x + alpha*dev_x)  ≠  GELU(x) + alpha*dev_gelu

Pre-activation contrast changes WHICH ACTIVATION REGIME each neuron operates in.
If position t has a larger-than-average x value, pre-contrast pushes it further
into the saturated/gain region of GELU.  If below average, it pushes toward the
negative-gain (suppressed) region.

This creates a soft WINNER-TAKE-ALL effect at the nonlinearity:
- Tokens with ABOVE-AVERAGE pre-activations are pushed into higher GELU gain
- Tokens with BELOW-AVERAGE pre-activations are pushed toward zero / suppression

In biological terms: lateral inhibition.  The "loudest" neurons in the sequence
suppress the quieter ones at the point of activation, not after.

CONTRAST WITH GELU39:
    gelu39 adds deviation AFTER GELU → additive output correction
    gelu44 adds deviation BEFORE GELU → changes the operating point of each neuron
    Both use the deviation from sequence mean, but acting at different moments.

Params: log_alpha_raw (1 scalar), alpha = softplus(raw) ≥ 0, init ≈ 0.3.
State:   None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU44(nn.Module):
    """Pre-activation EMA contrast: contrast x against cross-batch EMA before GELU."""

    def __init__(self):
        super().__init__()
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(9.0)))  # d=0.9
        self._ema: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema = None; self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.log_alpha_raw)
        d_val = torch.sigmoid(self.logit_decay).detach().item()

        # Cross-batch EMA of x (pre-activation) — causally valid
        first_step = False
        with torch.no_grad():
            bm = x.detach().mean(dim=(0, 1))    # (D,)
            if not self._ready:
                self._ema = bm.clone(); self._ready = True
                first_step = True
            else:
                self._ema = d_val * self._ema + (1 - d_val) * bm

        if first_step:
            return self._gelu(x)  # outside no_grad: gradients preserved

        mu_x         = self._ema.unsqueeze(0).unsqueeze(0)
        dev_x        = x - mu_x
        x_contrasted = x + alpha * dev_x
        return self._gelu(x_contrasted)
