"""GELU7 – Batch-discriminativeness habituation.

Core insight: an informative/novel channel is one that fires **differently** for
different tokens in the current batch (high cross-token variance). A channel that
fires similarly for all tokens is "familiar background" and should be suppressed.

    var_j   = Var over (B*T) of x_j                # (D,) per-channel batch variance
    rho_j   = E[|x_j|] + ε                         # (D,) mean magnitude for normalization
    disc_j  = var_j / rho_j                         # (D,) normalized discriminativeness
    novelty = 1 - exp(-τ · disc_j)                 # (D,) high for discriminative channels
    scale   = (1 - α) + α · novelty                # (D,) broadcast to (B, T, D)
    output  = GELU(x · scale)

Properties:
  • No cross-batch EMA state — computed fresh from the current batch
  • Per-channel gate: suppresses "constant" channels, amplifies discriminative ones
  • Naturally acts as a soft sparse feature selector
  • Scale-invariant: normalised by mean magnitude
  • Stabilised: gate computed with torch.no_grad() for clean gradient flow
  • 2 learnable params per layer: log_tau, log_blend (fewer than GELU2's 3)
  • Params vs control: +2 per layer (≈ same total model params)
"""

import math
import torch
import torch.nn as nn


class GELU7(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau   = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.5 / 0.5)))  # logit(0.5): start at α=0.5

    def reset_state(self):
        pass  # stateless

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
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)

        with torch.no_grad():
            x_flat  = x.flatten(0, -2)                           # (B*T, D)
            var_j   = x_flat.var(dim=0, unbiased=False)          # (D,) per-channel variance
            rho_j   = x_flat.abs().mean(dim=0).clamp(min=1e-6)   # (D,) mean magnitude

            disc    = var_j / rho_j                              # (D,) discriminativeness
            novelty = 1.0 - torch.exp(-tau * disc)               # (D,) high = informative
            scale   = (1.0 - alpha) + alpha * novelty            # (D,)

        return self._gelu(x * scale)
