"""GELU41 – Causal Within-Sequence Predictive Coding Gate.

MOTIVATION:
GELU39 subtracts the sequence mean (all T positions) – it uses future context.
For a language model we ideally want causal operations.

More importantly: the CAUSAL CUMULATIVE MEAN captures "what has been seen so far
in this sequence".  The deviation of position t from the mean of positions 0..t-1
is the causal prediction error: how surprising is token t given the prior context?

MECHANISM:
    out_raw    = GELU(x)                                     (B, T, D)

    # Causal cumulative mean: for position t, mean over [0, t-1]
    # (position 0 has no prior → its causal mean = out_raw[0] → dev = 0)
    cum_sum    = cumsum(out_raw, dim=1)                      (B, T, D)
    counts     = arange(1, T+1).float()                      (T,)
    cum_mean   = cum_sum / counts[None, :, None]             (B, T, D)

    # Shift right by 1 to get "mean of preceding tokens"
    # Position 0: no prior → shifted mean = 0 vector (no boost)
    prior_mean = pad(cum_mean[:, :-1, :], left=1 zero-step)  (B, T, D)

    dev    = out_raw - prior_mean                            (B, T, D)
    output = out_raw + alpha * dev

INTERPRETATION:
    At position t: the model amplifies HOW MUCH this token differs from the
    cumulative average of all preceding tokens in the sequence.
    - First token: prior_mean = 0, dev = out_raw  → slight boost
    - Later tokens: dev = difference from running mean → novel = boosted, redundant = dampened
    - This is a CAUSAL within-context contrast, appropriate for LM training.

WHY NOT JUST USE GELU39 (non-causal version)?
    Causal version backpropagates richer temporal gradients: the cumulative mean
    at position t depends on all positions 0..t-1, creating structured gradient flow.
    The non-causal version's mean depends on all positions simultaneously (less targeted).

ZERO STATE.  1 learnable parameter (log_alpha_raw).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU41(nn.Module):
    """Causal cumulative-mean contrast: amplify deviation from prior sequence context."""

    def __init__(self):
        super().__init__()
        # alpha = softplus(raw) ≥ 0, initialised ≈ 0.3
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))

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
        B, T, D = x.shape
        out_raw = self._gelu(x)                                 # (B, T, D)
        alpha   = F.softplus(self.log_alpha_raw)

        # Causal cumulative mean
        cum_sum = torch.cumsum(out_raw, dim=1)                  # (B, T, D)
        counts  = torch.arange(1, T + 1, dtype=x.dtype, device=x.device)
        cum_mean = cum_sum / counts.unsqueeze(-1)               # (B, T, D)

        # Shift cum_mean right by 1: position t sees mean of [0..t-1]
        # prior_mean[:, 0, :] = 0 (no prior for first token)
        prior_mean = F.pad(cum_mean[:, :-1, :], (0, 0, 1, 0))  # (B, T, D)

        dev    = out_raw - prior_mean                           # (B, T, D)
        output = out_raw + alpha * dev

        return output
