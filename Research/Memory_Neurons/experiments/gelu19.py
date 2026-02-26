"""GELU19 – Divisive Normalization (Biological Gain Control).

The canonical neuroscience model of gain control (Carandini & Heeger 2012):
every neuron's response is divided by a pooled measure of nearby activity.
This is how the visual cortex handles contrast, how auditory cortex handles
loudness, and how neurons avoid saturation in general.

Applied here at the FFN activation:

    pool[t] = RMS(x[t]) = sqrt(mean_d(x_d²))     (total signal energy, scalar)
    gain[t] = target_rms / (pool[t] + ε)          (homeostatic scalar gate ∈ ℝ>0)
    scale[t] = (1-α) + α · clamp(gain[t], lo, hi)
    output   = GELU(x * scale[t])

Semantics:
  • High-energy tokens (all channels firing strongly, well-encoded familiar
    features) → pool large → gain < 1 → suppressed
  • Low-energy tokens (uncertain, novel, or sparse) → pool small → gain > 1
    → amplified into GELU's sensitive range

Why this is conceptually different from every previous GELU variant:
  ✗ No EMA / cross-batch state
  ✗ No prototype memory
  ✗ No cross-token comparisons
  ✓ Purely within-token, instantaneous energy normalization
  ✓ The "familiar" detector is the magnitude of x itself — if activation is
    large across many channels, the pattern is already strongly encoded and
    needs no further amplification; if small, it's being underutilized.

This is essentially the mechanism behind Layer Norm applied AFTER the FFN
input rather than before, but with a learnable soft gate rather than hard
normalization.

Params per layer: log_target_rms (1), log_blend (1), log_lo (1), log_hi (1) = 4.
"""

import math
import torch
import torch.nn as nn


class GELU19(nn.Module):
    def __init__(self):
        super().__init__()
        # Target RMS (learnable): what "normal" energy should be
        self.log_target  = nn.Parameter(torch.tensor(0.0))   # target RMS ≈ 1
        # Blend strength: how much homeostatic gain to apply
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        # Gain clamp (learnable lo/hi prevent runaway amplification)
        self.log_lo      = nn.Parameter(torch.tensor(math.log(0.2)))   # min gain ≈ 0.2
        self.log_hi      = nn.Parameter(torch.tensor(math.log(5.0)))   # max gain ≈ 5.0

    def reset_state(self):
        pass   # stateless

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
        target = self.log_target.exp()                        # positive RMS target
        alpha  = torch.sigmoid(self.log_blend)
        lo     = self.log_lo.exp()                            # min gain
        hi     = self.log_hi.exp()                            # max gain

        # Per-token RMS energy
        rms    = x.pow(2).mean(dim=-1).sqrt()                 # (B, T)
        rms    = rms.clamp(min=1e-8)

        # Homeostatic gain: tokens with high energy get suppressed, low get amplified
        gain   = target / rms                                 # (B, T)
        gain   = gain.clamp(lo.item(), hi.item())             # safety clamp (no grad through lo/hi here)
        
        # Blend: scale = 1 when alpha=0, fully corrected when alpha=1
        scale  = (1.0 - alpha) + alpha * gain                 # (B, T)
        scale  = scale.unsqueeze(-1)                          # (B, T, 1)

        return self._gelu(x * scale)
