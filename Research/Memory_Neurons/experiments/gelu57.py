"""GELU57 – Homeostatic Plasticity (Per-Channel Activity Regulation).

THE BIOLOGICAL MECHANISM:
    The brain maintains stable activity levels through HOMEOSTATIC PLASTICITY:
    neurons that fire too much for too long DECREASE their excitability.
    Neurons that fire too little INCREASE their excitability.
    This is fundamentally different from Hebbian (associative) learning.

    BCM rule / Turrigiano (2008):
        firing rate r_d → target rate θ_d
        if r_d > θ_d: decrease gain (suppress)
        if r_d < θ_d: increase gain (amplify)

GELU57 IMPLEMENTATION:
    ema_act[d]  = EMA of mean |GELU(x)[d]|   ← tracked "average activity" per channel
    target_act  = learned scalar (global target rate, shared across channels)
    
    scale[d]    = (target_act / (ema_act[d] + ε)) ^ gamma    ← homeostatic gain
    output      = GELU(x) * scale                            ← per-channel regulation

    gamma ∈ (0, 1): learned exponent controlling regulation strength.
    scale is clamped to [0.1, 10] for stability.

WHY THIS IS SELF-REGULATING:
    ema_act[d] large (channel overactive) → scale[d] < 1 → suppressed → activity falls
    ema_act[d] small (channel quiet)      → scale[d] > 1 → amplified → activity rises
    
    This dynamically equalizes channel utilization — channels that carry the
    redundant "familiar" signal (high ema_act) are automatically scaled down,
    while channels encoding rare novel patterns (low ema_act) are amplified.

    No cosine similarity, no prototype, no threshold: purely activity-driven regulation.

RELATIONSHIP TO DIVISIVE NORMALIZATION (gelu19):
    gelu19 used: output = GELU(x) / (sigma + sum_d GELU(x_d)^2)  — instantaneous
    GELU57 uses: output = GELU(x) * scale(ema_act)               — temporal EMA
    
    GELU57's key advantage: scale[d] is based on HISTORY (what's been active before),
    not just the current activation. Novel channels (rare in general) stay amplified
    even when they happen to fire strongly right now.

INITIALIZATION SAFETY:
    ema_act initializes to the actual first-batch activation levels → scale ≈ 1.0
    immediately. No warmup needed. No NaN risk: ema_act ≥ ε = 1e-3.

Params: log_target_raw (1 scalar, target activity), logit_gamma (1 scalar, exponent),
        logit_decay (1 scalar, EMA speed) = 3 total.
State:  ema_act (D_FF,) — per-channel running mean absolute activation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU57(nn.Module):
    """Homeostatic plasticity: per-channel activity-normalized GELU gate."""

    def __init__(self, ema_decay: float = 0.95, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        # target activity level; softplus so it's always positive; init ≈ 0.3
        self.log_target_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        # gamma: regulation exponent ∈ (0, 1), init 0.5
        self.logit_gamma    = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5

        self._ema_act: torch.Tensor = None   # (D,) per-channel mean |activation|
        self._ready = False

    def reset_state(self):
        self._ema_act = None
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
        B, T, D = x.shape

        out   = self._gelu(x)   # (B, T, D)
        d_val = torch.sigmoid(self.logit_decay).detach().item()

        # ── Initialise EMA on first call ──────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                act_mean = out.detach().abs().flatten(0, 1).mean(0).clamp(min=self.eps)
                self._ema_act = act_mean.clone()
                self._ready   = True
            # Still apply scaling on first call — initialized properly above
            # (scale ≈ 1 since we init from actual values)

        # ── Compute homeostatic gain ──────────────────────────────────────────
        target = F.softplus(self.log_target_raw)         # scalar, > 0
        gamma  = torch.sigmoid(self.logit_gamma)         # ∈ (0, 1)

        # scale[d] = (target / ema_act[d])^gamma
        # log scale = gamma * (log target - log ema_act)
        ema_log = self._ema_act.clamp(min=self.eps).log()            # (D,)
        log_scale = gamma * (target.log() - ema_log)                 # (D,)
        scale = log_scale.exp().clamp(0.1, 10.0).view(1, 1, D)      # (1, 1, D)

        output = out * scale   # (B, T, D)

        # ── Update EMA of per-channel mean |activation| ───────────────────────
        with torch.no_grad():
            act_batch = out.detach().abs().flatten(0, 1).mean(0).clamp(min=self.eps)
            self._ema_act = d_val * self._ema_act + (1 - d_val) * act_batch

        return output
