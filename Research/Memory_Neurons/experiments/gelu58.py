"""GELU58 – Oja's Rule Online PCA Decorrelation.

THE FUNDAMENTAL PROBLEM WITH MEAN-BASED SUPPRESSION:
    All cosine-EMA experiments suppress similarity to the MEAN DIRECTION:
        familiar[t] ≈ cosine(out[t], EMA_mean)
    
    But the mean direction is not the most "familiar" direction — it's just the
    average. The DOMINANT VARIANCE DIRECTION (first principal component) is what
    captures the most systematic, repeated pattern in the activation space.

    Example: if tokens alternate between "the bank" and "the river", the mean
    direction is between them. But neither token is similar to the mean.
    The FIRST PC is the direction "bank vs river" — and BOTH tokens lie along it.
    Suppressing PC₁ removes this dominant repeated dimension from all tokens.

GELU58: ONLINE PCA VIA OJA'S RULE
    Oja's update (1982):
        w ← normalize(w + lr * (out_mean · y) · (out_mean - y · w))
        where y = out_mean · w   (projection coefficient)
    
    This converges to the first eigenvector of the covariance of out_mean.
    lr is set as (1 - d): when d → 1, slow update (stable); d → 0, fast tracking.

    Suppression:
        y    = einsum('btd,d->bt', out, w)     (B, T) projection coefficient
        proj = einsum('bt,d->btd', y, w)       (B, T, D) component along w
        output = out  +  alpha * (out - proj)  = (1 + alpha) * out - alpha * proj
               = out with the FIRST PC direction partially removed

    alpha ∈ (0, 1): learned suppression strength.
    When alpha=0: identity. When alpha=1: PC₁ fully removed.

WHY THIS IS DIFFERENT FROM GELU50 (orthogonal subspace):
    GELU50 used K EMA MEAN PROTOTYPES → Gram-Schmidt orthogonal basis.
    - Prototypes = directions of high average activation (mean-based).
    GELU58 uses the FIRST PC of out's COVARIANCE:
    - PC₁ = direction of MAXIMUM VARIANCE across tokens/batches.
    - These are fundamentally different geometric objects.
    - Covariance PC captures the most VARIABLE familiar signal,
      not just the average familiar signal.

WHY THIS WORKS BETTER FOR LANGUAGE:
    In text, the most variable direction is often shared syntactic patterns
    (noun phrases, verb phrases) which are systemically overrepresented.
    PCA finds this direction precisely. Removing it forces the downstream
    attention layers to work harder on less predictable, more lexically rich features.

STABILITY:
    w is unit-norm by construction (Oja rule normalizes).
    alpha < 1 so output norm is bounded.
    lr = 1 - d ≈ 0.05 for d=0.95 → slow, stable EMA-like convergence.

Params: log_alpha_raw (1 scalar), logit_decay (1 scalar) = 2 total.
State:  w (D_FF,) — online first principal component, unit norm. No grad.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU58(nn.Module):
    """Oja's rule online PCA decorrelation: suppress dominant variance direction."""

    def __init__(self, ema_decay: float = 0.95):
        super().__init__()
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # α ≈ 0.3

        self._w: torch.Tensor = None   # (D,) first principal component, unit norm
        self._ready = False

    def reset_state(self):
        self._w     = None
        self._ready = False

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

        out   = self._gelu(x)    # (B, T, D)
        alpha = F.softplus(self.log_alpha_raw).clamp(max=1.0)
        d_val = torch.sigmoid(self.logit_decay).detach().item()
        lr    = 1.0 - d_val      # Oja learning rate = EMA update speed

        # Batch mean of GELU output (used for Oja update)
        out_mean = out.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Initialise w to first batch mean direction ────────────────────────
        if not self._ready:
            self._w     = F.normalize(out_mean, dim=0)
            self._ready = True
            return out   # warm-up

        # ── Project out onto first PC ─────────────────────────────────────────
        w = self._w   # (D,) unit vector

        # y[b, t] = out[b, t, :] · w
        y    = torch.einsum('btd,d->bt', out, w)           # (B, T)
        # proj[b, t, :] = y[b, t] * w
        proj = torch.einsum('bt,d->btd', y, w)             # (B, T, D)

        # Suppress: remove alpha fraction of PC₁ component
        output = out + alpha * (out - proj)                 # = (1+α)*out - α*proj
        # Equivalent: out * (1 + alpha) - alpha * proj
        # At alpha=0: identity. At alpha=1: PC₁ zero'd.

        # ── Oja's rule update to w (no grad) ──────────────────────────────────
        with torch.no_grad():
            y_mean = (out_mean @ w).item()                  # scalar projection
            # Oja: w ← w + lr * y * (out_mean - y * w)
            w_new  = w + lr * y_mean * (out_mean - y_mean * w)
            self._w = F.normalize(w_new, dim=0)             # keep unit norm

        return output
