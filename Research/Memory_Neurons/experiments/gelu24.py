"""GELU24 – Z-Score Calibrated Familiarity Suppression.

PROBLEM WITH ALL PRIOR APPROACHES:
  Every GELU variant (gelu2–23) uses a fixed-τ threshold:
      novelty = exp(-τ · familiarity)
  For this to work, you need τ to be calibrated to the cosine similarity scale.
  If τ is too small: cosine sims are large (≈0.8–1.0) → novelty is always tiny → no suppression
  If τ is too large: novelty collapses to 0 for everything → full suppression → loss of signal

  The τ optimum depends on the DISTRIBUTION of cosine similarities, which shifts
  throughout training as the model learns and the EMA stabilises. A fixed τ (even
  learned) is always chasing a moving target.

GELU24 SOLUTION: Z-SCORE NORMALISE the familarity before gating.
  Instead of working on raw cosine similarity, normalise it by the RUNNING
  STATISTICS of the cosine similarity distribution:

      sim   = cosine(x_token, EMA_prototype)          (B, T)   raw familiarity
      z     = (sim − EMA[sim]) / sqrt(Var[sim] + ε)   (B, T)   self-calibrated Z-score
      novelty = σ(-λ · z)                              (B, T)   ∈ (0, 1)
      scale   = (1 − α) + α · novelty                 (B, T, 1)
      output  = GELU(x · scale)

INTERPRETATION:
  • z >> 0 → this token is MORE familiar than average → novelty → 0 → suppressed
  • z  ≈ 0 → this token is AVERAGELY familiar → novelty ≈ 0.5 → partial gate
  • z << 0 → this token is LESS familiar than average → novelty → 1 → full pass

  λ controls how sharply the z-score translates to suppression (a soft steepness).
  Because Z-scores are normalised, λ=1 is already a sensible starting point.

WHY THIS BEATS gelu2_k1:
  1. Self-calibrating: no τ tuning needed; works regardless of cosine sim magnitude
  2. Always has a constant fraction of tokens suppressed (those above-average in familiarity)
     → more consistent gradient signal throughout training
  3. As the model improves, cos sims change, but the Z-score re-normalises automatically
  4. The effective τ-equivalent adapts dynamically to the current EMA stability

STATE (per layer):
  • D-dimensional EMA prototype (float32)
  • scalar EMA of [mean cosine similarity]
  • scalar EMA of [variance of cosine similarity]
  All updated EMA style (non-differentiable, detached).

PARAMS per layer: logit_decay, log_lambda, log_blend = 3 scalars (same as gelu2_k1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU24(nn.Module):
    """Z-Score Calibrated Familiarity Suppression."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema:     torch.Tensor = None   # (D,) prototype vector
        self._sim_mu:  torch.Tensor = None   # scalar: running mean of cosine similarities
        self._sim_var: torch.Tensor = None   # scalar: running var  of cosine similarities
        self._ready = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_lambda = nn.Parameter(torch.tensor(0.0))          # λ=1 init: z-score steepness
        self.log_blend  = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # α=0.3 init

    def reset_state(self):
        self._ema     = None
        self._sim_mu  = None
        self._sim_var = None
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
        d     = torch.sigmoid(self.logit_decay)
        lam   = self.log_lambda.exp()           # z-score scale (positive)
        alpha = torch.sigmoid(self.log_blend)
        d_val = d.detach().item()

        x_mean = x.detach().flatten(0, -2).mean(0)   # (D,) current batch mean

        # ── Warm-up ──────────────────────────────────────────────────
        if not self._ready:
            self._ema     = x_mean.clone()
            self._sim_mu  = torch.zeros(1, device=x.device, dtype=x.dtype)
            self._sim_var = torch.ones(1,  device=x.device, dtype=x.dtype)
            self._ready   = True
            return self._gelu(x)

        # ── Cosine similarity of every token to the EMA prototype ────
        x_norm = F.normalize(x, dim=-1)                              # (B, T, D)
        p_norm = F.normalize(self._ema.unsqueeze(0), dim=-1)         # (1, D)
        sim    = (x_norm * p_norm).sum(-1)                           # (B, T) ∈ [-1, 1]

        # ── Z-score normalise against running distribution ───────────
        # z: positive → above-average familiar, negative → below-average (novel)
        z       = (sim - self._sim_mu) / (self._sim_var.sqrt() + 1e-8)   # (B, T)

        # Novelty: sigmoid(-λ·z)
        #   z >> 0 (very familiar) → σ(-large) ≈ 0 → suppressed
        #   z << 0 (very novel)    → σ(+large) ≈ 1 → passed through
        novelty = torch.sigmoid(-lam * z)                            # (B, T)

        scale   = (1.0 - alpha) + alpha * novelty                    # (B, T)
        scale   = scale.unsqueeze(-1)                                # (B, T, 1)

        # ── Update EMA prototype ──────────────────────────────────────
        self._ema = d_val * self._ema + (1.0 - d_val) * x_mean

        # ── Update running statistics of similarity distribution ──────
        sim_mean_batch = sim.detach().mean()
        sim_var_batch  = sim.detach().var().clamp(min=1e-8)
        self._sim_mu  = d_val * self._sim_mu  + (1.0 - d_val) * sim_mean_batch
        self._sim_var = d_val * self._sim_var + (1.0 - d_val) * sim_var_batch

        return self._gelu(x * scale)
