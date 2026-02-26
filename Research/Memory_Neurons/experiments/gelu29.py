"""GELU29 – Contrastive Dual-EMA Gate (Recent–Global Contrast).

MOTIVATION:

gelu22 tried dual-timescale EMA but used MAX(sim_fast, sim_slow):
  "suppress if recognised by EITHER system" → OR logic
  This failed because it was too aggressive — flagged everything as familiar.

GELU29 uses a CONTRASTIVE approach:
  contrast = sim_short - sim_long

  sim_short: cosine similarity to RECENT EMA (fast, d≈0.5, ~2 batches memory)
  sim_long:  cosine similarity to GLOBAL EMA (slow, d≈0.99, ~100 batches memory)

contrast > 0 → this token is MORE similar to recent history than global history
              → it's a "locally-habituated" pattern that's been recurring lately
              → SUPPRESS IT (it's recently repetitive, clogging local representation)

contrast < 0 → this token is MORE similar to global history than recent history
              → it's "globally familiar but locally novel" (e.g., a pattern that
                 hasn't appeared in a while but is globally common)
              → DON'T suppress (it's been absent recently, allow fresh access)

contrast ≈ 0 → both EMAs agree → ambiguous familiarity → mild gate

The suppression:
    familiarity = relu(contrast)       (only positive contrast drives suppression)
    novelty     = exp(-τ · familiarity)
    scale       = (1 - α) + α · novelty
    output      = GELU(x · scale)

WHY THIS IS FUNDAMENTALLY DIFFERENT FROM gelu22:
  gelu22: OR(fast_familiar, slow_familiar) → "suppress if EITHER recognizes you"
  gelu29: only suppress RECENTLY-but-not-globally familiar patterns
          = suppress "currently clogging patterns" only
          = adaptive habituation to the current discourse topic

For a language model on WikiText-2:
  - Short EMA tracks patterns in the current article/topic being trained on
  - Long EMA tracks patterns across the whole dataset
  - Contrast isolates topic-specific repetition (suppress → force novel expression)

Params per layer: log_tau, log_blend, logit_fast, logit_slow = 4 scalars
State: 2 × D EMA vectors (fast and slow)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU29(nn.Module):
    """Contrastive dual-EMA: suppress tokens that are MORE familiar recently than globally."""

    def __init__(self):
        super().__init__()
        self._ema_fast: torch.Tensor = None   # (D,) recent-history EMA
        self._ema_slow: torch.Tensor = None   # (D,) global-history EMA
        self._ready = False

        self.log_tau    = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend  = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        # Fast EMA: d≈0.5 — forgets in ~2 batches
        self.logit_fast = nn.Parameter(torch.tensor(0.0))               # logit(0.5)
        # Slow EMA: d≈0.99 — remembers ~100 batches
        self.logit_slow = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))

    def reset_state(self):
        self._ema_fast = None
        self._ema_slow = None
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
        # x: (B, T, D)
        tau    = self.log_tau.exp()
        alpha  = torch.sigmoid(self.log_blend)
        d_fast = torch.sigmoid(self.logit_fast)
        d_slow = torch.sigmoid(self.logit_slow)
        df_val = d_fast.detach().item()
        ds_val = d_slow.detach().item()

        x_mean = x.detach().flatten(0, -2).mean(0)   # (D,) current batch mean

        # ── Warm-up ──────────────────────────────────────────────────
        if not self._ready:
            self._ema_fast = x_mean.clone()
            self._ema_slow = x_mean.clone()
            self._ready    = True
            return self._gelu(x)

        # ── Cosine similarities ───────────────────────────────────────
        x_norm    = F.normalize(x, dim=-1)                             # (B, T, D)
        fast_norm = F.normalize(self._ema_fast.unsqueeze(0), dim=-1)   # (1, D)
        slow_norm = F.normalize(self._ema_slow.unsqueeze(0), dim=-1)   # (1, D)

        sim_fast = (x_norm * fast_norm).sum(-1)   # (B, T) ∈ [-1, 1]
        sim_slow = (x_norm * slow_norm).sum(-1)   # (B, T) ∈ [-1, 1]

        # ── Contrastive signal: recently familiar MINUS globally familiar ──
        # Positive: currently habituated (recently common but not globally dominant)
        contrast    = sim_fast - sim_slow           # (B, T) can be negative
        familiarity = F.relu(contrast)              # only positive contrast suppresses

        novelty = torch.exp(-tau * familiarity)     # (B, T)
        scale   = (1.0 - alpha) + alpha * novelty   # (B, T)
        scale   = scale.unsqueeze(-1)               # (B, T, 1)

        # ── Update both EMAs ──────────────────────────────────────────
        self._ema_fast = df_val * self._ema_fast + (1.0 - df_val) * x_mean
        self._ema_slow = ds_val * self._ema_slow + (1.0 - ds_val) * x_mean

        return self._gelu(x * scale)
