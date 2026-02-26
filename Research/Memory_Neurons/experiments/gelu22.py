"""GELU22 – Dual-Timescale EMA Familiarity Suppression.

Every previous approach used a SINGLE EMA with ONE decay rate. This conflates
two fundamentally different types of familiarity:

  • Short-term familiarity: "I saw this pattern in the last few batches"
    → fast EMA (d ≈ 0.5: forgets quickly, very responsive)
  • Long-term familiarity: "This is structurally common in the entire dataset"
    → slow EMA (d ≈ 0.99: stable across thousands of batches)

A token is "maximally familiar" if EITHER recognition system flags it.
A token is "maximally novel" if BOTH are surprised by it.

    sim_fast   = cosine(x_token, EMA_fast)         (B, T)
    sim_slow   = cosine(x_token, EMA_slow)         (B, T)
    familiarity = max(sim_fast, sim_slow)           (B, T) — OR-logic suppression

    novelty = exp(-τ * familiarity)                (B, T)
    scale   = (1 - α) + α · novelty               (B, T, 1)
    output  = GELU(x · scale)

The fast EMA captures: sudden recurring patterns, local syntactic habits,
high-frequency n-gram statistics within recent batches.
The slow EMA captures: deep structural patterns baked in from early training,
common syntactic scaffolding, stop-word patterns.

Why this should beat gelu2_k1 (single EMA):
  gelu2_k1's single EMA with d≈0.9 is a compromise between fast and slow.
  It misses patterns that appear then disappear then reappear (slow EMA catches them)
  and is slow to suppress freshly-common patterns (fast EMA catches them).
  Using the maximum gives the union of both recognition systems.

Params per layer: log_tau, log_blend, logit_fast, logit_slow = 4 scalars
State: 2 × D-dimensional EMA vectors (non-trainable buffers)
Extra params vs gelu2_k1: +1 decay scalar per layer = +4 total (tiny)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU22(nn.Module):
    """Dual-Timescale EMA: fast (short-term) + slow (long-term) familiarity."""

    def __init__(self):
        super().__init__()
        self._ema_fast: torch.Tensor = None   # (D,)
        self._ema_slow: torch.Tensor = None   # (D,)
        self._ready = False

        self.log_tau    = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend  = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # α = 0.3
        # Fast EMA: d ≈ 0.5 (short memory, ~1 batch)
        self.logit_fast = nn.Parameter(torch.tensor(math.log(0.5 / 0.5)))  # logit(0.5)
        # Slow EMA: d ≈ 0.99 (long memory, ~100 batches)
        self.logit_slow = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))

    def reset_state(self):
        self._ema_fast = None
        self._ema_slow = None
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
        tau    = self.log_tau.exp()
        alpha  = torch.sigmoid(self.log_blend)
        d_fast = torch.sigmoid(self.logit_fast)
        d_slow = torch.sigmoid(self.logit_slow)
        df_val = d_fast.detach().item()
        ds_val = d_slow.detach().item()

        x_mean = x.detach().flatten(0, -2).mean(0)   # (D,) current batch mean

        # ── Warm-up: initialise both EMAs to first batch mean ──────────
        if not self._ready:
            self._ema_fast = x_mean.clone()
            self._ema_slow = x_mean.clone()
            self._ready    = True
            return self._gelu(x)

        # ── Cosine similarity of every token to each EMA ───────────────
        x_norm    = F.normalize(x, dim=-1)                              # (B, T, D)
        fast_norm = F.normalize(self._ema_fast.unsqueeze(0), dim=-1)   # (1, D)
        slow_norm = F.normalize(self._ema_slow.unsqueeze(0), dim=-1)   # (1, D)

        sim_fast    = (x_norm * fast_norm).sum(-1)     # (B, T)
        sim_slow    = (x_norm * slow_norm).sum(-1)     # (B, T)

        # OR-logic: suppressed if recognised by EITHER system
        familiarity = torch.max(sim_fast, sim_slow)    # (B, T)

        novelty = torch.exp(-tau * familiarity)         # (B, T)
        scale   = (1.0 - alpha) + alpha * novelty       # (B, T)
        scale   = scale.unsqueeze(-1)                   # (B, T, 1)

        # ── Update both EMAs with current batch mean ───────────────────
        self._ema_fast = df_val * self._ema_fast + (1 - df_val) * x_mean
        self._ema_slow = ds_val * self._ema_slow + (1 - ds_val) * x_mean

        return self._gelu(x * scale)
