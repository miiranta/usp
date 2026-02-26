"""GELU47 – Causal Online Instance Normalization.

INSIGHT FROM GELU42 (causal analogue):
    GELU42 normalized GELU(x) against the full-sequence mean/var — one stat per channel
    for the whole T dimension.  That stat included FUTURE positions: causal violation.

    GELU47 performs the same normalization but STRICTLY CAUSALLY:
    Each position t is normalized against the running mean & variance of
    positions 0 .. t-1 only.

MECHANISM:
    out  = GELU(x)                                              (B, T, D)
    mu_{t-1}  = cumsum(out, dim=1)[:, t-1, :] / t              (past mean up to t-1)
    var_{t-1} = E[x²]_{t-1} - mu²_{t-1}                       (online Welford-style)
    out_norm  = (out - mu_{t-1}) / (std_{t-1} + eps)
    output    = gamma * out_norm + beta

    For position 0 (no past exists), the cross-batch EMA of mean & variance is used
    as a prior — this gives position 0 a meaningful normalization anchor from the
    distribution of past training batches.

WHY THIS SHOULD WORK:
    The power of gelu42 was that it whitened activations along the T axis, giving
    every position a zero-mean, unit-variance representation relative to the current
    sequence's statistics.  This is the maximal causal version of that operation:
    it progressively accumulates statistics so that each position is normalized
    against everything it could legitimately observe.  By the end of the sequence
    (large t), the running stats closely approximate the full-sequence stats,
    recovering most of gelu42's benefit.

CAUSAL GUARANTEE:
    output[t] depends only on x[0..t] — the cumsum at position t-1 never touches x[t+1..T-1].
    The EMA state is updated after the forward pass with stop-grad, so it never
    creates gradient paths to future tokens.

Params: gamma (D,), beta (D,)  → 2 * D_FF additional parameters.
State:  EMA mean & var (no grad), logit_decay (scalar, learned).
"""

import math
import torch
import torch.nn as nn


class GELU47(nn.Module):
    """Causal online instance normalization — normalize pos t against stats of 0..t-1."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))  # sigmoid→0.9
        self.gamma: nn.Parameter = None
        self.beta:  nn.Parameter = None
        self._ema_mean: torch.Tensor = None
        self._ema_var:  torch.Tensor = None
        self._ready = False

    def _init_affine(self, D: int, device):
        self.gamma = nn.Parameter(torch.ones(D,  device=device))
        self.beta  = nn.Parameter(torch.zeros(D, device=device))

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

        if self.gamma is None:
            self._init_affine(D, x.device)

        out = self._gelu(x)   # (B, T, D)

        # ── Initialise EMA on first call ──────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._ema_mean = out.detach().mean(dim=(0, 1)).clone()
                self._ema_var  = out.detach().var(dim=(0, 1), correction=0).clamp(min=0.01).clone()
                self._ready    = True
            return out          # warm-up step: no normalization yet

        # ── Build causal prefix mean/variance via cumsum ──────────────────────
        # counts[t] = number of observations BEFORE position t  (0-based)
        #   = 0 at t=0 (no past), 1 at t=1, ..., T-1 at t=T-1
        counts = torch.arange(0, T, device=x.device, dtype=out.dtype).view(1, T, 1)  # (1,T,1)

        # Causal sums: sum/sq_sum of out[0..t-1]
        # cumsum gives sum of [0..t]; shift right by 1 to get [0..t-1]
        full_sum    = out.cumsum(dim=1)            # (B, T, D)   sum[0..t]
        full_sq_sum = out.pow(2).cumsum(dim=1)     # (B, T, D)

        zero = torch.zeros(B, 1, D, device=x.device, dtype=out.dtype)
        past_sum    = torch.cat([zero, full_sum[:, :-1, :]],    dim=1)   # sum of [0..t-1]
        past_sq_sum = torch.cat([zero, full_sq_sum[:, :-1, :]], dim=1)   # sq_sum of [0..t-1]

        # Empirical mean & variance of [0..t-1]  (counts=0 means undefined → blended below)
        safe_counts = counts.clamp(min=1)                                 # avoid div/0
        emp_mean = past_sum    / safe_counts                              # (B, T, D)
        emp_sq   = past_sq_sum / safe_counts
        emp_var  = (emp_sq - emp_mean.pow(2)).clamp(min=0)               # (B, T, D)

        # ── Blend empirical stats with EMA prior, weighted by observation count ──
        # prior_weight = 1 at t=0 (no data), decays as K/(K+counts)  (K=4)
        K = 4.0
        prior_w  = K / (K + counts)                                      # (1, T, 1)
        empir_w  = 1.0 - prior_w

        ema_mu  = self._ema_mean.view(1, 1, D)                           # (1, 1, D)
        ema_var = self._ema_var.view(1, 1, D)

        past_mean = empir_w * emp_mean + prior_w * ema_mu                # (B, T, D)
        past_var  = empir_w * emp_var  + prior_w * ema_var               # (B, T, D)
        past_std  = past_var.sqrt().clamp(min=1e-5)                      # (B, T, D)

        out_norm = (out - past_mean) / (past_std + self.eps)             # (B, T, D)
        output   = self.gamma * out_norm + self.beta

        # ── Update EMA with current batch statistics (no grad) ───────────────
        d_val = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            bm = out.detach().mean(dim=(0, 1))
            bv = out.detach().var(dim=(0, 1), correction=0).clamp(min=0.01)
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * bm
            self._ema_var  = d_val * self._ema_var  + (1 - d_val) * bv

        return output
