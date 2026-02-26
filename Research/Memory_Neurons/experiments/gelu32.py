"""GELU32 – Familiar-Direction Projection Suppression.

MOTIVATION:
gelu28 computes cosine(out_raw, ema_out) and scales the WHOLE vector by a
scalar. This suppresses familiar-pointing vectors uniformly.

INSIGHT: We can be MORE SURGICAL. Decompose the GELU output into:
    1. The component ALONG the EMA direction (= the "familiar" part)
    2. The component ORTHOGONAL to the EMA direction (= the "novel" part)

Then suppress ONLY the familiar component, leaving the novel component intact.

MECHANISM:
    out_raw   = GELU(x)                               (B, T, D)
    ema_dir   = ema_out / ||ema_out||                 (D,)  unit familiar direction

    # Scalar projection onto familiar direction (per token)
    proj      = (out_raw · ema_dir)                   (B, T)

    # Decompose into familiar + novel parts
    fam_part  = proj.unsqueeze(-1) * ema_dir          (B, T, D)  ← along EMA
    nov_part  = out_raw - fam_part                    (B, T, D)  ← orthogonal

    # Scale familiar component by β ∈ (0, 1)
    β         = sigmoid(logit_β)                       scalar ← learned
    output    = nov_part + β · fam_part               (B, T, D)

When β→0: only novel directions survive (harsh suppression of familiar component)
When β→1: identity (no effect)
When β>1: amplify familiar component (learned to be counter-habituating)

WHY THIS IS DIFFERENT from gelu28:
  gelu28: same scalar scale on ENTIRE vector
  gelu32: DIFFERENT scales on familiar vs novel subspaces
          Novel subspace: scale = 1.0 (fully preserved)
          Familiar subspace: scale = β   (learned suppression)

  This preserves more information: even if the familiar direction is suppressed,
  the orthogonal complement (potentially carrying the novel content) is untouched.

Params: logit_decay (EMA speed), logit_beta (suppression strength) = 2 scalars.
        Minimal param overhead!
State:  ema_out (D_FF,) — non-trainable buffer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU32(nn.Module):
    """Project GELU output onto EMA direction; suppress familiar component only."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema_out: torch.Tensor = None
        self._ready = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        # β=0.5 init: suppress the familiar component by half at start
        self.logit_beta = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5

    def reset_state(self):
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
        out_raw = self._gelu(x)          # (B, T, D)

        # --- Update EMA of GELU output ---
        with torch.no_grad():
            out_mean = out_raw.detach().mean(dim=(0, 1))   # (D,)
            d = torch.sigmoid(self.logit_decay)
            if not self._ready:
                self._ema_out = out_mean.clone()
                self._ready   = True
            else:
                self._ema_out = d * self._ema_out + (1 - d) * out_mean

        # --- Familiar direction: unit vector along EMA ---
        ema_norm = F.normalize(self._ema_out.unsqueeze(0), dim=-1)  # (1, D)

        # --- Scalar projection of each token onto familiar direction ---
        # out_raw: (B, T, D), ema_norm: (1, D)
        proj     = (out_raw * ema_norm).sum(dim=-1, keepdim=True)   # (B, T, 1)

        # --- Decompose ---
        fam_part = proj * ema_norm            # (B, T, D)  — familiar component
        nov_part = out_raw - fam_part         # (B, T, D)  — orthogonal / novel

        # --- Suppress familiar, preserve novel ---
        beta   = torch.sigmoid(self.logit_beta)   # ∈ (0, 1)
        output = nov_part + beta * fam_part        # (B, T, D)

        return output
