"""GELU38 – Per-Channel Frequency Habituation.

MOTIVATION:
All previous experiments compute a SCALAR gate per token (one number per position).
This means the same scale is applied uniformly across all D dimensions.

But individual channels have very different roles:
- Some channels are ALWAYS highly active (common syntactic/semantic features).
- Some channels are RARELY active (specialised, specific-to-context features).

Suppressing a token uniformly (scalar gate) is a blunt instrument.
A channel that is ALWAYS active (channel-level habituation) should be suppressed
independently of whether the current TOKEN is familiar.

MECHANISM:
    out_raw   = GELU(x)                                  (B, T, D)

    # Per-channel running mean of |activation|
    ema_ch ← d * ema_ch + (1-d) * mean_BT(|out_raw|)    (D,)  – cross-batch

    # Per-channel gate: channels with high average |activity| get suppressed
    # gate_d = 1 - alpha * sigmoid(k * ema_ch)          (D,)
    # At ema_ch → 0:   gate_d ≈ 1 - alpha/2  (mild suppression)
    # At ema_ch → ∞:   gate_d ≈ 1 - alpha    (maximal suppression)
    # We normalise ema_ch by its own median to make k scale-free.

    output = out_raw * gate_d                            (D-broadcast)

KEY DIFFERENCES FROM PREVIOUS EXPERIMENTS:
- Gate is D-dimensional (per channel), not scalar
- Based on channel MAGNITUDE history, not cosine similarity
- Suppresses FEATURES that are always firing, regardless of the current token
- Zero per-token computation: gate is constant within a forward pass

Params: logit_decay (1 scalar), log_alpha (1 scalar) = 2 learnable scalars.
State:  ema_ch (D,) – per-channel EMA of activation magnitude.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU38(nn.Module):
    """Per-channel habituation: suppress always-active channels, preserve rare ones."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema_ch: torch.Tensor = None
        self._ready = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        # Maximum suppression alpha = sigmoid(log_alpha_logit) ∈ (0,1)
        # initialised to alpha ≈ 0.5
        self.log_alpha_logit = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5

    def reset_state(self):
        self._ema_ch = None
        self._ready  = False

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
        alpha = torch.sigmoid(self.log_alpha_logit)   # max suppression strength

        with torch.no_grad():
            ch_mean = out_raw.detach().abs().mean(dim=(0, 1))   # (D,) per-channel activity
            if not self._ready:
                self._ema_ch = ch_mean.clone()
                self._ready  = True
                return out_raw
            d_val = d.detach().item()
            self._ema_ch = d_val * self._ema_ch + (1.0 - d_val) * ch_mean

        # Normalise EMA by its median so the sigmoid threshold is scale-free
        # gate = 1 - alpha * sigmoid(ema_ch / (median + eps))
        median = self._ema_ch.median().clamp(min=1e-6)
        gate   = 1.0 - alpha * torch.sigmoid(self._ema_ch / median)   # (D,)

        # Broadcast gate across B and T
        output = out_raw * gate.unsqueeze(0).unsqueeze(0)  # (B, T, D)
        return output
