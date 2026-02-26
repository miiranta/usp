"""GELU26 – Diagonal-Gaussian NLL Surprise Gate.

THE CORE PROBLEM WITH ALL PRIOR EMA-BASED GATING (gelu2–gelu25):
They all estimate familiarity using a single statistic per token or direction:
  - gelu2:   cosine similarity to mean vector     → direction only, ignores magnitude
  - gelu5/3: |x_j - μ_j| / mean|x_j|             → per-channel MAD, ignores variance
  - gelu20:  projection onto mean direction       → direction only
  - gelu25:  per-channel |GELU(x) - GELU(EMA)|   → output MAD, ignores variance

None of them compute the proper JOINT surprise under a distribution:
  NLL(x; μ, σ²) = 0.5 * Σ_j (x_j - μ_j)² / σ²_j      (log-det term constant)

GELU26 tracks the DIAGONAL GAUSSIAN EMA:
  - Per-channel running mean   μ  (D,)
  - Per-channel running variance σ² (D,)

Then computes the normalized NLL per token as the familiarity signal:

    nll[b,t]      = mean_j [(x[b,t,j] - μ_j)² / (σ²_j + ε)]  ←  Mahalanobis² / D
    familiarity_j = exp(−τ · nll)                               scalar per token ∈ (0,1]
    scale         = (1 − α) + α · (1 − familiarity)
    output        = GELU(x · scale)

Why NLL beats cosine:
  • Cosine: two activations with the SAME DIRECTION but DIFFERENT magnitudes
    score identically. A token with x = 10 * μ_norm is incorrectly flagged as
    "very familiar" even though it's unusually large.
  • NLL: a token is only familiar if it's near μ AND its per-channel deviation
    is proportional to what the model normally sees (σ²). Unusual MAGNITUDE in
    any channel, even if the direction matches, → high NLL → novel → not suppressed.
  • This is the proper principled measure under a Gaussian model.

Practical normalization: divide NLL by D (so the scalar is dimension-independent)
and use mean_j instead of sum_j. This gives a scale-free surprise score.

Competitive EMA update rule: same as gelu2 (updates both μ and σ², competitive
update to the nearest prototype — here just single prototype so always update).

Params per layer: logit_decay, log_tau, log_blend = 3 scalars (same as gelu2_k1)
State: 2 × D vectors (μ, σ²), all non-trainable.
"""

import math
import torch
import torch.nn as nn


class GELU26(nn.Module):
    """Diagonal-Gaussian NLL Surprise: per-token NLL under EMA Gaussian → gate."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._mu:  torch.Tensor = None   # (D,) EMA per-channel mean
        self._var: torch.Tensor = None   # (D,) EMA per-channel variance
        self._ready = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

    def reset_state(self):
        self._mu    = None
        self._var   = None
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
        # x: (B, T, D)
        d     = torch.sigmoid(self.logit_decay)
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)
        d_val = d.detach().item()

        x_flat   = x.detach().flatten(0, -2)                      # (B*T, D)
        x_mean   = x_flat.mean(0)                                  # (D,) batch mean
        x_var    = x_flat.var(0, unbiased=False).clamp(min=1e-8)   # (D,) batch variance

        # ── Warm-up ──────────────────────────────────────────────────
        if not self._ready:
            self._mu    = x_mean
            self._var   = x_var
            self._ready = True
            return self._gelu(x)

        # ── Normalised NLL per token (Mahalanobis² / D) ──────────────
        # mahal²_j = (x_j - μ_j)² / σ²_j   per channel per token
        # nll = mean_j mahal²_j               scalar per (b,t) — dimensionless
        diff_sq  = (x - self._mu).pow(2)                           # (B, T, D)
        nll      = (diff_sq / (self._var + 1e-8)).mean(dim=-1)     # (B, T)  ≥ 0

        # familiarity: ~1 when nll≈0 (x near Gaussian mean), ~0 when nll big (novel)
        familiarity = torch.exp(-tau * nll)                        # (B, T)

        # suppress familiar tokens; let novel ones through at full strength
        novelty = 1.0 - familiarity                                # (B, T)
        scale   = (1.0 - alpha) + alpha * novelty                  # (B, T)
        scale   = scale.unsqueeze(-1)                              # (B, T, 1)

        # ── Update EMA μ and σ² ──────────────────────────────────────
        self._mu  = d_val * self._mu  + (1.0 - d_val) * x_mean
        self._var = d_val * self._var + (1.0 - d_val) * x_var

        return self._gelu(x * scale)
