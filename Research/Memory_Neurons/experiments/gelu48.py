"""GELU48 – Exponentially-Weighted Causal Mean Subtraction.

INSIGHT FROM GELU39:
    GELU39 subtracted the uniform sequence mean from each position:
        output = GELU(x) + alpha * (GELU(x) - mean_T(GELU(x)))
    That mean used ALL T positions (including future) → causal violation.

    GELU41 (the causal fix attempt) used the UNIFORM cumulative mean:
        mu_t = mean(out[0..t])
    This was causally correct but got EARLY STOPPED — because the uniform
    cumulative mean is NOISY at the start (position 0 has no past; position 1
    only sees one neighbour) and changes non-smoothly as t grows.

    GELU48 replaces the uniform cumulative mean with an EXPONENTIALLY-WEIGHTED
    causal mean (EWA — exponentially weighted average):
        ewa_t = sum_{s < t}  d^(t-1-s) * (1-d) * out[s]       (past-only, causal)
    This gives recent positions higher weight and decays older ones, producing
    a SMOOTH, STABLE context signal throughout the sequence.

MECHANISM:
    out   = GELU(x)                                                    (B, T, D)
    W     = lower-triangular T×T matrix  where W[i,j] = d^(i-1-j) * (1-d)
            (j < i only — strict past; row 0 is all zeros → use EMA fallback)
            rows are normalised by their sum.
    ewa   = einsum('ij, bjd -> bid', W, out)                          (B, T, D)
    context[pos=0] = cross-batch EMA (no past available in sequence)
    output = out + alpha * (out − context)

    alpha is a learned scalar; d = sigmoid(logit_decay) is a learned scalar.

WHY THIS SHOULD WORK BETTER THAN GELU41:
    Uniform cumsum gives equal weight to a token from 60 steps ago and one from
    1 step ago.  The exponential weighting emphasises RECENT context, which is
    richer and more relevant for next-token prediction.  At later positions in
    the sequence the EWA approximates the full-sequence mean closely (as d→0.9,
    the effective window is ~10 positions), giving the DC-removal effect of gelu39
    without any future leakage.

CAUSAL GUARANTEE:
    W[i, j] = 0 for j >= i (strict past only).  output[t] therefore depends only
    on out[0..t-1] (plus the current out[t] via the direct term), and out[s]
    depends only on x[s], so there is no path from future x to current output.

    For position 0 (first row of W is all zeros), the cross-batch EMA is substituted
    as the context — a prior from past training distributions, containing zero
    information about the current sequence.

    The EMA state is updated after the forward pass under torch.no_grad(), creating
    no gradient path to any token in the sequence.

Params: alpha (scalar, learned), logit_decay (scalar, learned) — 2 parameters total.
State:  EMA of mean output (D,), no grad.
"""

import math
import torch
import torch.nn as nn


class GELU48(nn.Module):
    """Exponentially-weighted causal mean subtraction inside the FF activation."""

    def __init__(self):
        super().__init__()
        self.log_alpha_raw = nn.Parameter(torch.tensor(0.0))    # alpha via softplus
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(9.0)))  # d = sigmoid → 0.9
        self._ema_mean: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ready    = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )

    def _build_ewa_matrix(self, T: int, d: float, device, dtype) -> torch.Tensor:
        """Build a strict-past exponential weight matrix  W  of shape (T, T).

        W[i, j] = (1-d) * d^(i-1-j)   for j < i   (strict causal past)
                = 0                     for j >= i
        Rows are normalised so that sum(W[i, :]) == 1 for i > 0.
        Row 0 is all zeros (no past available).
        """
        i_idx = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)
        j_idx = torch.arange(T, device=device, dtype=dtype).unsqueeze(0)  # (1, T)
        dist  = i_idx - j_idx - 1.0                                        # (T, T)

        # Past-only mask (strict): j < i  ↔  dist >= 0
        mask   = (dist >= 0).float()
        raw_w  = (d ** dist) * (1.0 - d) * mask                           # (T, T)
        row_sum = raw_w.sum(dim=1, keepdim=True).clamp(min=1e-8)           # (T, 1)
        # For row 0 (sum == 0), division by clamp gives 0/eps ≈ 0: fine.
        W = raw_w / row_sum                                                 # (T, T) normalised
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        out = self._gelu(x)   # (B, T, D)

        # ── Initialise EMA on first call ──────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._ema_mean = out.detach().mean(dim=(0, 1)).clone()
                self._ready    = True
            return out   # warm-up step

        # ── Build EWA matrix (not cached — d is a learned scalar) ────────────
        d_val = torch.sigmoid(self.logit_decay).detach().item()
        W     = self._build_ewa_matrix(T, d_val, x.device, out.dtype)  # (T, T)

        # ewa_mean[b, t, d] = weighted sum of out[b, 0..t-1, d]
        ewa_mean = torch.einsum('ij,bjd->bid', W, out)                  # (B, T, D)

        # Position 0 row of W is all-zero → ewa_mean[:, 0, :] == 0
        # Replace with cross-batch EMA as context anchor
        has_past = (W.sum(dim=1) > 0).view(1, T, 1)                    # (1, T, 1)
        ema_ctx  = self._ema_mean.view(1, 1, D).expand(B, T, D)
        context  = torch.where(has_past.expand(B, T, D), ewa_mean, ema_ctx)

        alpha    = torch.nn.functional.softplus(self.log_alpha_raw)
        output   = out + alpha * (out - context)                        # (B, T, D)

        # ── Update EMA ────────────────────────────────────────────────────────
        with torch.no_grad():
            bm = out.detach().mean(dim=(0, 1))
            self._ema_mean = d_val * self._ema_mean + (1.0 - d_val) * bm

        return output
