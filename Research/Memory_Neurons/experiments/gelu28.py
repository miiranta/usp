"""GELU28 – Output-Cosine EMA Gate.

MOTIVATION (combining insights from gelu2 and gelu25):

gelu2_k1 works by:
    sim = cosine(x_input, EMA_input)     ← cosine on INPUT, whole-vector
    novelty = exp(-τ·sim)
    output = GELU(x · scale)

gelu25 works by (output-side, survival to epoch 11):
    out_raw = GELU(x)
    novelty = per-channel deviation of out_raw from EMA_output
    output = out_raw · scale

GELU28 combines the BEST element of each:
    - gelu2's WHOLE-VECTOR COSINE (most effective familiarity signal)
    - gelu25's OUTPUT-SIDE GATING (more principled biological mechanism)

    out_raw  = GELU(x)                             (B, T, D)
    sim_out  = cosine(out_raw, EMA_out)            (B, T)  ← cosine on OUTPUT, whole-vector
    novelty  = exp(-τ · sim_out)
    scale    = (1 - α) + α · novelty              (B, T, 1)
    output   = out_raw · scale

The key difference from gelu25:
  gelu25 used |out_raw[j] - EMA_out[j]| / |EMA_out[j]| per channel (MAD),
  which is sensitive to amplitude and breaks the holistic vector encoding.
  GELU28 uses cosine similarity of the WHOLE gelu output vector to the EMA
  of gelu outputs — DIRECTION-based familiarity at the OUTPUT level.

The key difference from gelu2:
  gelu2 gates the input to GELU (before nonlinearity).
  GELU28 gates the output of GELU (after nonlinearity).

Why "output cosine" might be better than "input cosine":
  The GELU output is a nonlinear transformation of the input. Two different
  inputs can produce similar GELU outputs if they share the same "above threshold"
  channels. Cosine familiarity of the OUTPUT captures whether the MODEL RESPONSE
  (which channels fired and at what relative strength) is familiar — not whether
  the INPUT was familiar. This is closer to the biological definition of habituation.

Params per layer: log_tau, log_blend, logit_decay = 3 scalars (same as gelu2_k1)
State: 1 × D_FF EMA vector (non-trainable)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU28(nn.Module):
    """Output-Cosine EMA gate: cosine familiarity on GELU output, gate applied to output."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema_out: torch.Tensor = None   # (D,) EMA of GELU output mean vector
        self._ready = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

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
        d     = torch.sigmoid(self.logit_decay)
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)
        d_val = d.detach().item()

        # ── Compute GELU output first ─────────────────────────────────
        out_raw = self._gelu(x)              # (B, T, D)

        # ── EMA of the batch-mean GELU output ────────────────────────
        out_mean = out_raw.detach().flatten(0, -2).mean(0)   # (D,) batch mean of output

        if not self._ready:
            self._ema_out = out_mean.clone()
            self._ready   = True
            return out_raw   # first batch: pass through

        # ── Cosine similarity of each token's GELU output to EMA ─────
        out_norm = F.normalize(out_raw, dim=-1)                        # (B, T, D)
        ema_norm = F.normalize(self._ema_out.unsqueeze(0), dim=-1)    # (1, D)
        sim_out  = (out_norm * ema_norm).sum(-1)                      # (B, T) ∈ [-1, 1]

        # Novelty: high when GELU output is unusual (no habitual response)
        novelty = torch.exp(-tau * sim_out)                            # (B, T)

        # Gate: suppress habitual GELU responses
        scale   = (1.0 - alpha) + alpha * novelty                      # (B, T)
        scale   = scale.unsqueeze(-1)                                  # (B, T, 1)

        # ── Update EMA ────────────────────────────────────────────────
        self._ema_out = d_val * self._ema_out + (1.0 - d_val) * out_mean

        return out_raw * scale
