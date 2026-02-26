"""GELU64 – Fast-Slow Synaptic Depression (Two-Pool Model).

BIOLOGICAL REALITY:
    The Tsodyks-Markram synapse actually has TWO distinct pools of resources:
    
    1. FAST (readily releasable pool, RRP): small, depletes in ~1-5 spikes,
       recovers in ~200ms. Handles burst suppression.
    2. SLOW (reserve pool, RP): large, depletes over seconds of sustained activity,
       recovers in ~1-10s. Handles chronic adaptation.

    GELU56 has only one pool. This means:
    - Short tau_rec: catches rapid local repetition but ignores sustained patterns
    - Long tau_rec: catches sustained patterns but ignores rapid local repetition
    - Can't do both simultaneously.

GELU64: TWO INDEPENDENT RESOURCE POOLS
    r_fast[t]: depletes and recovers fast  (tau_fast ≈ 2 steps)
    r_slow[t]: depletes and recovers slow  (tau_slow ≈ 16 steps)
    
    Combined gate:
        r_combined = w_fast * r_fast + w_slow * r_slow   (B, T, D)
    
    w_fast, w_slow are learned scalars (sum need not be 1).
    Then contrast-normalized as in GELU62.

WHY TWO POOLS BEATS ONE:
    At early positions (t=0..4): r_fast dominates (slow pool still full).
    At mid positions (t=10..20): r_fast has recovered; r_slow is depleting.
    At late positions (t=50..63): both pools track different timescales.

    For a word appearing twice quickly: r_fast fires (short spacing).
    For a word used throughout a paragraph: r_slow fires (cumulative).
    Both signals contribute, and the model can weight them via w_fast/w_slow.

CROSS-SEQUENCE MEMORY:
    Both pools carry EMA resource levels as sequence initialization.
    The slow pool's cross-sequence EMA provides genuine long-term adaptation:
    channels chronically overused across many passages start the next passage
    already partially depleted in the slow pool.

Params: logit_U_fast, log_tau_fast, logit_U_slow, log_tau_slow,
        logit_w_fast, logit_w_slow, logit_decay = 7 scalars.
State:  _ema_res_fast (D,), _ema_res_slow (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU64(nn.Module):
    """Two-pool synaptic depression: fast (burst) and slow (sustained) resource pools."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # Fast pool: small utilization, fast recovery (τ ≈ 2 steps)
        self.logit_U_fast    = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # U ≈ 0.3
        self.log_tau_fast    = nn.Parameter(torch.tensor(math.log(math.exp(2.0) - 1.0)))  # τ ≈ 2

        # Slow pool: smaller utilization, slow recovery (τ ≈ 16 steps)
        self.logit_U_slow    = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))  # U ≈ 0.1
        self.log_tau_slow    = nn.Parameter(torch.tensor(math.log(math.exp(16.0) - 1.0)))  # τ ≈ 16

        # Combination weights (free-signed after tanh: can subtract)
        self.logit_w_fast    = nn.Parameter(torch.tensor(0.0))   # init: equal weight
        self.logit_w_slow    = nn.Parameter(torch.tensor(0.0))

        # Cross-sequence EMA decay
        self.logit_decay     = nn.Parameter(torch.tensor(math.log(0.95 / 0.05)))

        self._ema_res_fast: torch.Tensor = None
        self._ema_res_slow: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_res_fast = None
        self._ema_res_slow = None
        self._ready        = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def _run_pool(self, out: torch.Tensor, r_init: torch.Tensor,
                  U: float, rec_rate: float) -> torch.Tensor:
        """Run one resource pool along T dimension. Returns r_gates (B, T, D)."""
        B, T, D = out.shape
        r = r_init.unsqueeze(0).expand(B, D).clone()   # (B, D)
        trace = []
        with torch.no_grad():
            for t in range(T):
                trace.append(r.clone())
                firing = torch.tanh(out[:, t, :].detach().abs())   # bounded ∈ (0,1)
                used   = (U * r * firing).clamp(max=r * 0.99)
                rec    = (1.0 - r) * rec_rate
                r      = (r - used + rec).clamp(0.01, 1.0)
        return torch.stack(trace, dim=1), r   # (B, T, D), (B, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        U_fast     = torch.sigmoid(self.logit_U_fast).detach().item()
        tau_fast   = F.softplus(self.log_tau_fast).clamp(min=0.5).detach().item()
        U_slow     = torch.sigmoid(self.logit_U_slow).detach().item()
        tau_slow   = F.softplus(self.log_tau_slow).clamp(min=0.5).detach().item()
        d_val      = torch.sigmoid(self.logit_decay).detach().item()
        rec_fast   = 1.0 - math.exp(-1.0 / tau_fast)
        rec_slow   = 1.0 - math.exp(-1.0 / tau_slow)

        # Combination weights: sigmoid so each ∈ (0, 1)
        w_fast     = torch.sigmoid(self.logit_w_fast)
        w_slow     = torch.sigmoid(self.logit_w_slow)

        out = self._gelu(x)   # (B, T, D)

        # ── Initialise ────────────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._ema_res_fast = torch.ones(D, device=x.device, dtype=out.dtype)
                self._ema_res_slow = torch.ones(D, device=x.device, dtype=out.dtype)
                self._ready = True
            return out

        # ── Run both pools ────────────────────────────────────────────────────
        r_fast_gates, r_fast_end = self._run_pool(out, self._ema_res_fast, U_fast, rec_fast)
        r_slow_gates, r_slow_end = self._run_pool(out, self._ema_res_slow, U_slow, rec_slow)

        # ── Combine ───────────────────────────────────────────────────────────
        r_combined = w_fast * r_fast_gates + w_slow * r_slow_gates   # (B, T, D)

        # Contrast normalization: mean gate = 1 per token → energy-preserving
        r_norm = r_combined / (r_combined.mean(dim=-1, keepdim=True) + self.eps)
        output = out * r_norm

        # ── Update EMA resource levels ────────────────────────────────────────
        with torch.no_grad():
            self._ema_res_fast = d_val * self._ema_res_fast + (1 - d_val) * r_fast_end.mean(0)
            self._ema_res_slow = d_val * self._ema_res_slow + (1 - d_val) * r_slow_end.mean(0)

        return output
