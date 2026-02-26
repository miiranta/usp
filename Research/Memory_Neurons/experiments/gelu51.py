"""GELU51 – Per-Channel Adaptive Novelty Gate.

ALL PREVIOUS EXPERIMENTS suppress at the TOKEN level (1 scalar per token):
    gate[b, t] = f(familiarity_of_whole_vector[b, t])     ← single scalar

This misses channel-level structure: within one token, some channels might
be firing in a novel pattern while others are completely predictable.

GELU51 computes familiarity INDEPENDENTLY PER CHANNEL:

    out[b, t, d] = GELU(x)[b, t, d]
    z[b, t, d]   = (out[b, t, d] - μ[d]) / (σ[d] + ε)    ← per-channel z-score
    novelty[b, t, d] = tanh(|z[b, t, d]| / temp)          ∈ (0, 1)
                     = 0 for channels near their mean (familiar)
                     = 1 for channels far from mean  (novel)
    gate[b, t, d] = 1 - α * (1 - novelty[b, t, d])        ∈ (1-α, 1)
    output = out * gate

WHERE:
    μ[d], σ[d] are per-channel EMA statistics (D-dimensional running mean/std).
    temp        is a learned sharpness scalar.
    α           is a learned max-suppression strength scalar.

WHAT THIS GIVES:
    Each channel d independently decides: "am I novel or familiar right now?"
    Novel channels (|z|>>0) gate → 1.0 (no suppression, pass through)
    Familiar channels (|z|≈0)  gate → 1-α   (suppressed by up to α)

    For a token where 90% of channels are firing normally but 10% fire unusually,
    only the 10% novel channels contribute strongly to downstream processing.
    Scalar gates (all prior experiments) would average these and suppress everything.

    This is D-dimensional channel selection vs 1-dimensional token selection.

RELATIONSHIP TO BATCH NORM:
    The z-score uses per-channel EMA (not batch statistics), so it's causally safe.
    The gate *amplifies* deviation rather than normalizing it away (opposite of BN).

Params: log_alpha_raw (1), log_temp_raw (1) = 2 scalars.
State:  μ (D_FF,), log_var (D_FF,) — EMA, no grad. logit_decay learned.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU51(nn.Module):
    """Per-channel EMA z-score novelty gate — D-dimensional signal selector."""

    def __init__(self, ema_decay: float = 0.95, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # α ≈ 0.5
        self.log_temp_raw  = nn.Parameter(torch.tensor(0.0))   # temp = softplus(0) ≈ 0.69

        self._ema_mean: torch.Tensor = None   # (D,)
        self._ema_var:  torch.Tensor = None   # (D,)
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_var  = None
        self._ready    = False

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

        out = self._gelu(x)   # (B, T, D)

        alpha = F.softplus(self.log_alpha_raw).clamp(max=1.0)   # max suppression ≤ 1.0
        temp  = F.softplus(self.log_temp_raw)                   # sharpness > 0
        d_val = torch.sigmoid(self.logit_decay).detach().item()

        # ── Initialise EMA on first call ──────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                flat = out.detach().flatten(0, 1)               # (B*T, D)
                self._ema_mean = flat.mean(0).clone()
                self._ema_var  = flat.var(0, correction=0).clamp(min=1e-4).clone()
                self._ready    = True
            return out   # warm-up: return plain GELU

        # ── Per-channel z-score ───────────────────────────────────────────────
        mu  = self._ema_mean.view(1, 1, D)            # (1, 1, D)
        std = self._ema_var.sqrt().view(1, 1, D)      # (1, 1, D)

        z       = (out - mu) / (std + self.eps)       # (B, T, D) — per-channel z-score
        novelty = torch.tanh(z.abs() / temp)          # (B, T, D) ∈ (0, 1)

        # gate: 1 for perfectly novel, (1-alpha) for perfectly familiar
        gate   = 1.0 - alpha * (1.0 - novelty)        # (B, T, D) ∈ (1-α, 1)
        output = out * gate                            # (B, T, D)

        # ── Update EMA (no grad) ──────────────────────────────────────────────
        with torch.no_grad():
            flat = out.detach().flatten(0, 1)         # (B*T, D)
            bm = flat.mean(0)
            bv = flat.var(0, correction=0).clamp(min=1e-4)
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * bm
            self._ema_var  = d_val * self._ema_var  + (1 - d_val) * bv

        return output
