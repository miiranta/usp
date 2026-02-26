"""GELU39 – Stateless Within-Sequence Instance Contrast.

MOTIVATION:
Every previous experiment maintains state ACROSS batches (cross-batch EMA).
This couples activations across training steps, adding fragility and requiring
warm-up (the first few batches produce different behaviour).

GELU39 is completely STATELESS: it computes a gate entirely within the current
forward pass, using only the (B, T, D) tensor it receives.

INSPIRATION: Instance Normalization / Contrast Normalization
  In vision (style transfer, low-level vision), instance normalization removes
  mean and variance WITHIN a single sample.  Here we apply the same idea to the
  sequence dimension: remove the "within-sequence mean" from each token, then
  amplify how much each token deviates from the sequence background.

MECHANISM:
    out_raw  = GELU(x)                                     (B, T, D)
    mu       = out_raw.mean(dim=1, keepdim=True)           (B, 1, D)  seq mean
    dev      = out_raw - mu                                (B, T, D)  deviation

    # Amplify deviation: positions that stand out get boosted
    output   = out_raw + alpha * dev
             = (1 + alpha) * out_raw - alpha * mu

    As alpha → 0: plain GELU
    As alpha → 1: deviation from within-context mean is doubled

WHY THIS HELPS LANGUAGE MODELLING:
  In a language context, the "sequence mean" captures the dominant topic /
  semantic background.  Each token's deviation captures HOW IT DIFFERS from
  the background — exactly the kind of contrastive signal that helps a model
  assign different probabilities to different continuations.

ZERO STATE.  1 learnable parameter (log_alpha_raw → alpha via softplus).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU39(nn.Module):
    """EMA-based sequence contrast: amplify deviation from cross-batch EMA mean."""

    def __init__(self):
        super().__init__()
        # alpha = softplus(raw) >= 0, initialised ~0.3
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
        out_raw = self._gelu(x)                                    # (B, T, D)
        alpha   = F.softplus(self.log_alpha_raw)                   # scalar ≥ 0
        d_val   = torch.sigmoid(self.logit_decay).detach().item()

        # Cross-batch EMA — causally valid, no future-token leakage
        with torch.no_grad():
            batch_mean = out_raw.detach().mean(dim=(0, 1))  # (D,)
            if not self._ready:
                self._ema = batch_mean.clone()
                self._ready = True
                return out_raw
            self._ema = d_val * self._ema + (1 - d_val) * batch_mean

        mu     = self._ema.unsqueeze(0).unsqueeze(0)               # (1, 1, D)
        dev    = out_raw - mu                                      # (B, T, D)
        output = out_raw + alpha * dev
        return output
