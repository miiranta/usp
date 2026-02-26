"""GELU55 – Multi-Timescale Filter Bank (signal processing analogue).

SIGNAL PROCESSING FRAMING:
    A single EMA is a 1-pole IIR low-pass filter with cutoff frequency fc ∝ (1-d).
    When we compute (out - ema), we're computing the HIGH-PASS residual.
    gelu37/gelu39 both did this but with fixed or no decay.

    In audio/DSP, you don't use a single filter — you use a FILTER BANK:
    split the signal into multiple frequency bands, process each differently,
    recombine. Each band carries different information.

GELU55: 5-POLE FILTER BANK on the temporal stream of activations:
    EMA_k[d] tracked with decay rate d_k  (k=0..4, log-spaced from fast to slow)
        k=0: d=0.50 → "last 2 batches"     (very fast, current context)
        k=1: d=0.80 → "last 5 batches"
        k=2: d=0.90 → "last 10 batches"
        k=3: d=0.97 → "last 33 batches"
        k=4: d=0.99 → "last 100 batches"  (slow, long-term average)

    For each token, compute per-timescale RESIDUAL:
        r_k[b,t,d] = out[b,t,d] - EMA_k[d]       (B, T, D)  per timescale

    Learned combination of residuals:
        output = out + sum_k w_k * r_k            (B, T, D)
               = out * (1 + sum_k w_k) - sum_k w_k * EMA_k

    where w_k are 5 learned scalar weights (can be positive: amplify novelty,
    or negative: suppress that band's contribution).

    With all w_k > 0: subtract multi-band background → residual amplification.
    With mixed signs: model can decide which timescale's baseline to trust more.

WHY THIS IS BETTER THAN SINGLE EMA:
    - Different semantic patterns repeat at different timescales.
    "the" repeats every ~5 tokens (fast timescale).
    A topic like "banking" persists for paragraphs (slow timescale).
    - A single EMA can only track one scale; the filter bank tracks all 5.
    - The model can learn that "suppress slow-timescale familiarity but amplify
      fast-timescale changes" — adapting suppression to the linguistic structure.

RELATIONSHIP TO DUAL-EMA (gelu22, gelu29):
    gelu22 used 2 timescales (fast+slow); gelu29 used (recent EMA - global EMA).
    GELU55 uses 5 timescales with free-signed learned weights — strictly more general.

CAUSAL GUARANTEE:
    Each EMA_k is updated from PAST batches only, under no_grad.
    output[t] depends only on x[t] and EMA state from past batches.
    Zero intra-sequence aggregation over T.

Params: w (5 scalars), logit_decays (5 scalars — decay rates learned) = 10 total.
State:  5 EMA vectors (D_FF,) each.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 5 initial decay values (fast to slow), log-spaced
_INIT_DECAYS = [0.50, 0.80, 0.90, 0.97, 0.99]


class GELU55(nn.Module):
    """5-timescale filter bank: multi-pole IIR residual amplification."""

    def __init__(self):
        super().__init__()
        K = len(_INIT_DECAYS)
        self._K = K

        # Per-timescale learned decay (all trainable — model tunes the frequencies)
        self.logit_decays = nn.Parameter(torch.tensor(
            [math.log(d / (1.0 - d)) for d in _INIT_DECAYS]
        ))   # (K,)

        # Per-timescale mixing weights (init near 0 → safe startup)
        self.weights = nn.Parameter(torch.zeros(K))   # (K,)

        # Per-timescale EMA state
        self._emas: list = [None] * K
        self._ready = False

    def reset_state(self):
        self._emas  = [None] * self._K
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
        B, T, D = x.shape

        out = self._gelu(x)   # (B, T, D)

        batch_mean = out.detach().flatten(0, 1).mean(0)   # (D,) for EMA update

        # ── Initialise all EMA banks on first call ────────────────────────────
        if not self._ready:
            with torch.no_grad():
                for k in range(self._K):
                    self._emas[k] = batch_mean.clone()
            self._ready = True
            return out   # warm-up

        # ── Compute per-timescale residuals and weighted sum ──────────────────
        # weights are free-signed scalars; tanh bounds them to (-1, +1)
        w = torch.tanh(self.weights)   # (K,)  ∈ (-1, 1)

        # Accumulate: output_correction = sum_k w_k * (out - EMA_k)
        correction = torch.zeros_like(out)   # (B, T, D)
        for k in range(self._K):
            ema_k = self._emas[k].view(1, 1, D)              # (1, 1, D)
            residual_k = out - ema_k                          # (B, T, D)
            correction = correction + w[k] * residual_k

        output = out + correction   # (B, T, D)

        # ── Update all EMA banks (no grad) ────────────────────────────────────
        d_vals = torch.sigmoid(self.logit_decays).detach()   # (K,)
        with torch.no_grad():
            for k in range(self._K):
                dk = d_vals[k].item()
                self._emas[k] = dk * self._emas[k] + (1 - dk) * batch_mean

        return output
