"""GELU52 – Learned Low-Rank Interference Cancellation.

THE IDEAS SO FAR:
    gelu31 (best): 4.4% improvement — scalar gate from joint cosine of EMA
    gelu50: geometric subspace suppression via EMA prototypes (data-driven)
    gelu51: per-channel EMA z-score gate (channel-level selectivity)

GELU52 INTRODUCES A FUNDAMENTALLY DIFFERENT AXIS:
    All previous experiments suppress based on SIMILARITY TO EMA (statistical memory).
    GELU52 learns — via backprop — WHICH direction to suppress for optimal LM quality.

MOTIVATION:
    In signal processing, an "interference canceller" learns a reference signal and
    subtracts it from the target signal.  The reference direction u and detector
    direction v are learned from the data — not hand-designed or EMA-accumulated.

    Applied here:
        out     = GELU(x)                              (B, T, D)
        coef    = out · V     → (B, T, r)              "how much interference detected?"
        cancel  = coef @ U^T  → (B, T, D)              "estimated interference signal"
        output  = out - alpha * cancel                  "interference-cancelled output"

    where U ∈ R^(D, r) and V ∈ R^(D, r) are LEARNED, not EMA.

    The gradient of the language-modelling loss will push U and V to identify
    the D-dimensional subspace which, when subtracted from GELU outputs, MOST
    REDUCES PERPLEXITY.  This is a task-driven version of suppression:
    suppress whatever isn't useful for predicting the next token.

WHY "INTERFERENCE"?
    Think of GELU(x) as containing two components:
        1. "signal" — novel, token-specific, informative for LM prediction
        2. "interference" — common-mode background, redundant across tokens,
                             could be position bias, DC offset, shared syntax noise
    
    U, V jointly learn to isolate the interference subspace.  Since they are trained
    end-to-end, they don't need to track what's "familiar" statistically — they
    directly optimise for information useful downstream.

RELATION TO PRIOR EXPERIMENTS:
    gelu50: EMA prototypes find familiar subspace. Suppresses via projection.
            Data-driven, no gradient through the suppression direction.
    gelu32: Single EMA prototype, project & suppress. Gradient flows through scaling.
            Used COSINE SIM gate, rank-1 EMA only.
    GELU52: FULLY LEARNED suppression directions U, V (rank r=4).
            Gradient flows through U, V, alpha — task-driven, not distribution-driven.
            The model learns "what to suppress" rather than observing "what's common".

PARAMETER COUNT:
    2 × r × D + 1 = 2 × 4 × 1024 + 1 = 8193  extra scalars per layer.
    Negligible vs 9.8M total params.

CAUSAL SAFETY:
    output[t] = GELU(x[t]) - alpha * V @ (U^T @ GELU(x[t]))
    This is a LINEAR FUNCTION of x[t] alone — no aggregation over T.
    Strictly causal.

Params: U (D, r), V (D, r), log_alpha_raw (scalar) — all learned via backprop.
State:  None. Fully stateless.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU52(nn.Module):
    """Learned low-rank interference cancellation — task-driven subspace suppression."""

    def __init__(self, rank: int = 4):
        super().__init__()
        self._rank = rank
        # Lazy-initialised so D is discovered from first forward pass
        self._U: nn.Parameter = None   # (D, r) — suppression direction
        self._V: nn.Parameter = None   # (D, r) — detector direction
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.2) - 1.0)))  # α ≈ 0.2

    def reset_state(self):
        # No EMA state; U/V are parameters — don't reset them between experiments
        pass

    def _init_params(self, D: int, device):
        r = self._rank
        # Small random init: both U, V start near zero so output ≈ GELU(x) initially
        scale = 1.0 / math.sqrt(D)
        self._U = nn.Parameter(torch.randn(D, r, device=device) * scale)
        self._V = nn.Parameter(torch.randn(D, r, device=device) * scale)

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

        if self._U is None:
            self._init_params(D, x.device)

        out   = self._gelu(x)            # (B, T, D)
        alpha = F.softplus(self.log_alpha_raw)

        # ── Detect interference: project out onto V columns ───────────────────
        # coef[b, t, r] = out[b, t, :] · V[:, r]
        coef   = torch.einsum('btd,dr->btr', out, self._V)   # (B, T, r)

        # ── Reconstruct interference signal ───────────────────────────────────
        # cancel[b, t, d] = sum_r coef[b, t, r] * U[d, r]
        cancel = torch.einsum('btr,dr->btd', coef, self._U)  # (B, T, D)

        # ── Subtract interference ─────────────────────────────────────────────
        output = out - alpha * cancel    # (B, T, D)

        return output
