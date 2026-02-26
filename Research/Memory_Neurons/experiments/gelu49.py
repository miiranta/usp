"""GELU49 – Batch-Dimension Contrast (Cross-Sequence DC Removal).

THE KEY INSIGHT: which dimension is safe to aggregate over?

    GELU39/42 used  mean(dim=1)  — average over the T (time) dimension.
    This leaks future tokens WITHIN a sequence → causal violation.

    GELU49 uses    mean(dim=0)  — average over the B (batch) dimension.
    This averages the SAME TIME-STEP t across B DIFFERENT, INDEPENDENT sequences.
    There is no causal relationship between sequences in a batch, so this is
    100% causally valid.

MECHANISM:
    out    = GELU(x)                                                   (B, T, D)
    mu_bt  = out.mean(dim=0, keepdim=True)                             (1, T, D)
             = "population baseline" for each position t across the batch
    output = out + alpha * (out − mu_bt)                               (B, T, D)

    This is INSTANCE CONTRAST in the batch direction: each sequence deviates
    from the AVERAGE sequence (at each time step) rather than from its own
    temporal mean.

WHY THIS SHOULD CAPTURE GELU39's BENEFIT:
    The mean over B at position t is a summary of "what the model typically
    activates at this point in the sequence" across all current training
    sequences.  Subtracting it removes position-dependent DC bias, leaving
    only the unique information each sequence carries.  This is similar to
    gelu39's sequence-mean subtraction in purpose (DC removal, context contrast)
    but operates on a DIFFERENT and SAFE axis.

TRAIN / INFERENCE MISMATCH:
    At inference (batch size 1), mu_bt = out itself → no contrast.
    To handle this, when B == 1 (or during eval-time small batches), we fall
    back to a CROSS-BATCH EMA mean: a running average of the per-position
    mean from training, which approximates the expected batch mean.
    We ALWAYS blend:
        context = (1 - w) * mu_bt  +  w * ema_mean
    where w = sigmoid(logit_blend) lets the model adjust how much it trusts
    the cross-batch EMA vs the live batch mean.  During training the batch
    mean dominates; the blend parameter allows graceful generalisation.

CAUSAL GUARANTEE:
    output[b, t, :] depends on out[b, 0..t, :] (direct path) and on
    out[0..B-1, t, :] (batch-mean path).  The batch-mean path involves other
    sequences at the SAME time step t — not future time steps of sequence b.
    A future token x[b, t+1] cannot affect output[b, t] through this mechanism.

    The EMA update uses stop-gradient and is performed AFTER the forward pass.

Params: alpha (scalar, learned), logit_blend (scalar, learned) — 2 parameters.
State:  EMA of per-position mean  (T_nominal, D)  — initialised lazily.
        NOTE: T can vary across calls; the EMA is reused for matching T only.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU49(nn.Module):
    """Batch-dimension contrast: subtract mean across batch (not across time)."""

    def __init__(self):
        super().__init__()
        self.log_alpha_raw = nn.Parameter(torch.tensor(0.0))   # alpha via softplus
        self.logit_blend   = nn.Parameter(torch.tensor(-2.0))  # w = sigmoid → ~0.12  (mostly live batch mean)
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(9.0)))  # EMA decay for state
        self._ema_mean: torch.Tensor = None                    # (T, D) or None
        self._ready = False
        self._ema_T = None   # the T for which _ema_mean was created

    def reset_state(self):
        self._ema_mean = None
        self._ready    = False
        self._ema_T    = None

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

        # ── Initialise EMA on first call ──────────────────────────────────────
        if not self._ready or self._ema_T != T:
            with torch.no_grad():
                self._ema_mean = out.detach().mean(dim=0).clone()  # (T, D)
                self._ema_T    = T
                self._ready    = True
            return out   # warm-up step

        # ── Live batch mean at each position ─────────────────────────────────
        batch_mean = out.mean(dim=0, keepdim=True)  # (1, T, D) — mean over B sequences

        # ── Blend with cross-batch EMA (for train/eval consistency) ──────────
        w       = torch.sigmoid(self.logit_blend)
        ema_ctx = self._ema_mean.unsqueeze(0)       # (1, T, D)
        context = (1.0 - w) * batch_mean + w * ema_ctx   # (1, T, D)

        # ── Contrast output ──────────────────────────────────────────────────
        alpha  = F.softplus(self.log_alpha_raw)
        output = out + alpha * (out - context)      # (B, T, D)

        # ── Update EMA of per-position mean ──────────────────────────────────
        d_val = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            bm = out.detach().mean(dim=0)           # (T, D)
            self._ema_mean = d_val * self._ema_mean + (1.0 - d_val) * bm

        return output
