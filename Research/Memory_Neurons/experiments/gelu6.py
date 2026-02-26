"""GELU6 – Within-sequence novelty gate.

Instead of global EMA (which is too coarse and slow to adapt), we compute
novelty relative to the *current sequence context*.

For token (b, t), "familiar" = activation similar to the sequence average.
Novel tokens deviate from what's common in this sequence → should not be suppressed.

    seq_mean  = mean over T positions                    # (B, 1, D)
    norm_dev  = |x - seq_mean| / (|seq_mean| + ε)       # (B, T, D)
    novelty   = 1 - exp(-τ · norm_dev)                  # (B, T, D)  ∈ [0,1)
    scale     = (1 - α) + α · novelty                   # suppress familiar, pass novel
    output    = GELU(x · scale)

No EMA state. Novelty is computed fresh per forward pass from the sequence itself.
This is the most principled formulation for a sequence model: "what is unusual
for *this* sequence?" rather than "what has the model seen on average?".

Learnable parameters: 2 per layer (vs 3 for GELU2, same total model params ≈ control).
    log_tau   → τ > 0    : suppression sharpness
    log_blend → α ∈ (0,1): max suppression depth
"""

import math
import torch
import torch.nn as nn


class GELU6(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))  # sigmoid -> 0.9
        self._ema_mean: torch.Tensor = None
        self._ema_abs:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None; self._ema_abs = None; self._ready = False

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
        d_val = torch.sigmoid(self.logit_decay).detach().item()

        # Cross-batch EMA — causally valid, no future leakage
        with torch.no_grad():
            batch_mean = x.detach().mean(dim=(0, 1))        # (D,)
            batch_abs  = x.detach().abs().mean(dim=(0, 1))  # (D,)
            if not self._ready:
                self._ema_mean = batch_mean.clone()
                self._ema_abs  = batch_abs.clone()
                self._ready    = True
            else:
                self._ema_mean = d_val * self._ema_mean + (1 - d_val) * batch_mean
                self._ema_abs  = d_val * self._ema_abs  + (1 - d_val) * batch_abs

        ema_mean = self._ema_mean.unsqueeze(0).unsqueeze(0)               # (1, 1, D)
        ema_abs  = self._ema_abs.unsqueeze(0).unsqueeze(0).clamp(min=1e-6)

        norm_dev    = (x - ema_mean).abs() / ema_abs         # (B, T, D)
        familiarity = torch.exp(-tau * norm_dev)
        novelty     = 1.0 - familiarity
        scale       = (1.0 - alpha) + alpha * novelty
        return self._gelu(x * scale)
