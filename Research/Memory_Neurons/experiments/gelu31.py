"""GELU31 – Double Cosine Gate (input + output).

MOTIVATION:
gelu2_k1  works via INPUT cosine gate  → modest improvement (+2.5%)
gelu28    works via OUTPUT cosine gate → BEATS gelu2_k1 at every epoch

INSIGHT: Both independently improve things. If the two familiarity signals are
complementary (input direction ≠ output direction, generally), combining them
multiplicatively should capture richer familiarity structure.

MECHANISM:
    out_raw  = GELU(x)                                     (B, T, D)

    # Two independent EMA vectors:
    in_sim   = cosine(x,       ema_in)                     (B, T)  ← input cosine
    out_sim  = cosine(out_raw, ema_out)                    (B, T)  ← output cosine

    # Joint novelty: product of exp-gates (multiplicative independence)
    novelty  = exp(-τ_in * in_sim) * exp(-τ_out * out_sim) (B, T)
    scale    = (1 - α) + α · novelty                       (B, T, 1)
    output   = out_raw · scale

    # Both EMAs updated with batch-mean of their respective vectors
    ema_in  ← d·ema_in  + (1-d)·mean(x)
    ema_out ← d·ema_out + (1-d)·mean(out_raw)

Why multiplicative?  exp(-a)*exp(-b) = exp(-(a+b)).
The combined gate is the exp of the SUM of both familiarity scores.
Only truly novel in BOTH input AND output space passes un-suppressed.
If EITHER space is novel, suppression is reduced (exp near 1 for that term).

Params: logit_decay (shared), log_tau_in, log_tau_out, log_blend = 4 scalars.
State:  ema_in (D_FF,), ema_out (D_FF,) — non-trainable buffers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU31(nn.Module):
    """Double cosine gate: joint input + output cosine familiarity."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema_in: torch.Tensor  = None
        self._ema_out: torch.Tensor = None
        self._ready = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_tau_in  = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_tau_out = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

    def reset_state(self):
        self._ema_in  = None
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
        # x: (B, T, D)
        B, T, D = x.shape
        d = torch.sigmoid(self.logit_decay)

        # --- GELU forward ---
        out_raw = self._gelu(x)          # (B, T, D)

        # --- Build / update EMAs ---
        with torch.no_grad():
            x_mean   = x.detach().mean(dim=(0, 1))      # (D,)
            out_mean = out_raw.detach().mean(dim=(0, 1)) # (D,)

            if not self._ready:
                self._ema_in  = x_mean.clone()
                self._ema_out = out_mean.clone()
                self._ready   = True
            else:
                self._ema_in  = d * self._ema_in  + (1 - d) * x_mean
                self._ema_out = d * self._ema_out + (1 - d) * out_mean

        # --- Input cosine similarity ---
        x_flat   = x.reshape(B * T, D)                              # (BT, D)
        in_sim   = F.cosine_similarity(
            x_flat, self._ema_in.unsqueeze(0).expand(B * T, -1), dim=-1
        ).reshape(B, T)                                              # (B, T)

        # --- Output cosine similarity ---
        out_flat = out_raw.reshape(B * T, D)
        out_sim  = F.cosine_similarity(
            out_flat, self._ema_out.unsqueeze(0).expand(B * T, -1), dim=-1
        ).reshape(B, T)

        # --- Joint novelty gate ---
        tau_in  = torch.exp(self.log_tau_in)
        tau_out = torch.exp(self.log_tau_out)
        novelty = torch.exp(-tau_in * in_sim) * torch.exp(-tau_out * out_sim)  # (B, T)

        alpha = torch.sigmoid(self.log_blend)
        scale = (1.0 - alpha) + alpha * novelty                      # (B, T)

        return out_raw * scale.unsqueeze(-1)
