"""GELU20 – Orthogonal Familiarity Subtraction.

Every previous GELU variant uses a SCALAR gate: output = GELU(x * s) where
s ∈ ℝ is the same for all channels. This means "suppress this whole token
uniformly."

GELU20 does something geometrically richer: it identifies the FAMILIAR
DIRECTION in D-dimensional space (the EMA of past activations, normalised
to a unit vector) and selectively suppresses only the COMPONENT OF X THAT
ALIGNS WITH THIS FAMILIAR DIRECTION. The orthogonal complement (everything
that is novel, unexpected, not captured by the familiar prototype) passes
through at FULL amplitude.

        fam_dir = normalize(EMA)                    (D,) — learned familiar axis
        proj    = (x · fam_dir) * fam_dir           (B,T,D) — familiar component
        novel   = x - proj                          (B,T,D) — orthogonal residual
        novelty = exp(-τ · cosine(x, fam_dir))      (B,T) — scalar familiar measure
        blend   = (1-α) + α · novelty               (B,T) — gate for familiar comp
        x_out   = novel + blend * proj              (B,T,D) — suppress familiar part
        output  = GELU(x_out)

When the familiar direction captures a strong component of x:
  • The projection is large along fam_dir
  • cosine similarity is high → novelty is LOW → blend → 0 → projection suppressed
  • Only the orthogonal (novel) residual survives

When x is novel (orthogonal to fam_dir):
  • Projection is near-zero anyway
  • blend doesn't matter; full x passes

This is DIRECTIONAL SUPPRESSION: the model can learn a single axis in
representation space that encodes "this has been seen" and zero it out.
Previous variants only ever scaled the whole vector.

Params per layer: 3 scalars (log_tau, log_blend, logit_decay) + D EMA (not params).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU20(nn.Module):
    def __init__(self, ema_decay: float = 0.99):
        super().__init__()
        self._ema:   torch.Tensor = None
        self._ready: bool = False

        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )

    def reset_state(self):
        self._ema   = None
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
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)
        d     = torch.sigmoid(self.logit_decay)
        d_val = d.detach().item()

        x_mean = x.detach().flatten(0, -2).mean(0)   # (D,)

        # Warm-up
        if not self._ready:
            self._ema   = x_mean
            self._ready = True
            return self._gelu(x)

        # ── Familiar direction ─────────────────────────────────────────
        fam_dir = F.normalize(self._ema.unsqueeze(0).unsqueeze(0), dim=-1)  # (1,1,D)

        # ── Projection onto familiar direction ─────────────────────────
        # proj_scalar[b,t] = x[b,t] · fam_dir   (signed scalar projection)
        proj_scalar = (x * fam_dir).sum(dim=-1, keepdim=True)  # (B, T, 1)
        proj        = proj_scalar * fam_dir                     # (B, T, D) familiar comp
        novel       = x - proj                                  # (B, T, D) novel residual

        # ── Novelty: how much does x align with the familiar direction? ─
        x_norm   = F.normalize(x, dim=-1)                       # (B, T, D)
        cos_sim  = (x_norm * fam_dir).sum(dim=-1)               # (B, T)
        novelty  = torch.exp(-tau * cos_sim)                    # (B, T) ∈ (0,1]

        # ── Blend: suppress familiar component proportional to its cosine sim ─
        blend = (1.0 - alpha) + alpha * novelty                 # (B, T) scalar
        blend = blend.unsqueeze(-1)                             # (B, T, 1)

        # ── Reconstruct: keep novel intact, gate familiar component ────
        x_out  = novel + blend * proj                           # (B, T, D)

        # ── EMA update ─────────────────────────────────────────────────
        self._ema = d_val * self._ema + (1.0 - d_val) * x_mean

        return self._gelu(x_out)
