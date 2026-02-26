"""GELU25 – Output-Side EMA Familiarity Gate.

ALL prior GELU variants (2–24) share one mathematical structure:
    scale = f(familiar(x))
    output = GELU(x · scale)

They differ only in how familiar(x) is computed.

GELU25 uses a DIFFERENT structure:
    out_raw = GELU(x)               ← standard GELU
    scale   = g(familiar(out_raw))  ← familiarity of the GELU OUTPUT
    output  = out_raw · scale       ← gate the OUTPUT, not the input

Biological grounding: spike-rate adaptation.
In neuroscience, neural adaptation acts on the OUTPUT side: a neuron's FIRING RATE
decreases when it fires repeatedly in its "habitual" pattern. The neuron doesn't
receive a reduced input — it fires less in RESPONSE to the same input.

The EMA here tracks the typical GELU OUTPUT per channel (not the input vector).
A channel that consistently produces values near its usual output is "habituated"
and gets suppressed. A channel that suddenly produces an unusual output is
"aroused" and passes through.

Key differences from all prior work:
  1. The gate operates AFTER the nonlinearity, not before it.
  2. The EMA monitors output values, not input directions.
  3. Suppression is per-CHANNEL, but based on output deviation (not input deviation).
     (Per-channel input approaches  gelu3/5/7/21 all failed — per-channel OUTPUT
     tracking is conceptually distinct because GELU creates a nonlinear mapping.)

    out_raw   = GELU(x)                                    (B, T, D)
    out_mean  = EMA of out_raw over batches                (D,)
    out_rho   = EMA of |out_raw| over batches              (D,)  normalization
    norm_dev  = |out_raw − out_mean| / (out_rho + ε)      (B, T, D)
    familiarity = exp(−τ · norm_dev)                       (B, T, D)  high=habitual
    scale       = (1−α) + α · (1 − familiarity)           (B, T, D)  suppress habitual
    output      = out_raw · scale

Params per layer: log_tau, log_blend, logit_decay = 3 scalars (same as gelu2_k1)
State: 2×D EMA vectors for out_mean, out_rho (non-trainable)
"""

import math
import torch
import torch.nn as nn


class GELU25(nn.Module):
    """Output-side EMA familiarity gate: suppress neurons whose OUTPUT is habitual."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema_out:  torch.Tensor = None   # (D,) EMA of GELU output (per channel)
        self._ema_rho:  torch.Tensor = None   # (D,) EMA of |GELU output|
        self._ready = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

    def reset_state(self):
        self._ema_out = None
        self._ema_rho = None
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

        # ── Standard GELU output ──────────────────────────────────────
        out_raw = self._gelu(x)              # (B, T, D)

        # ── EMA statistics over batch/time ────────────────────────────
        out_flat = out_raw.detach().flatten(0, -2)        # (B*T, D)
        out_mean = out_flat.mean(0)                        # (D,) batch mean of GELU output
        out_rho  = out_flat.abs().mean(0).clamp(min=1e-6)  # (D,) mean absolute output

        if not self._ready:
            self._ema_out = out_mean
            self._ema_rho = out_rho
            self._ready   = True
            return out_raw   # first batch: pass through, no gate yet

        # ── How novel is this batch's output? ─────────────────────────
        # deviation of each (b,t,j) output relative to channel j's typical output
        norm_dev    = (out_raw - self._ema_out).abs() / (self._ema_rho + 1e-6)  # (B,T,D)

        # familiarity: 1 when output = typical; 0 when very different
        familiarity = torch.exp(-tau * norm_dev)           # (B, T, D)

        # suppress familiar outputs; pass novel ones
        novelty = 1.0 - familiarity                        # (B, T, D)
        scale   = (1.0 - alpha) + alpha * novelty          # (B, T, D)

        # ── Update EMA ────────────────────────────────────────────────
        self._ema_out = d_val * self._ema_out + (1.0 - d_val) * out_mean
        self._ema_rho = d_val * self._ema_rho + (1.0 - d_val) * out_rho

        return out_raw * scale
