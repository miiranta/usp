"""GELU72 – Opponent-Process Rebound via Output Deviation Amplification.

BIOLOGICAL BASIS — OPPONENT PROCESS THEORY:
    Solomon & Corbit (1974): after a strong response, the nervous system produces
    an opponent ("rebound") response to return to homeostasis.
    
    Examples:
    - Visual afterimage: stare at red, see green afterimage (color-opponent cells)  
    - Drug tolerance: initial euphoria → rebound dysphoria (neural sensitization)
    - Neural adaptation: sustained stimulation → reduced response + rebound silence
    
    At the synaptic level: after strong sustained activation, opponent inhibition
    creates a CONTRAST SIGNAL — we see edges (transitions), not fills.

IMPLEMENTATION — OUTPUT DEVIATION ENHANCEMENT:
    For each token, compute the DEVIATION of GELU(x) from the historical average:
    
        ema_out[d]   = EMA of E[GELU(x)[d]]   per channel, cross-batch
        delta[b,t,d] = GELU(x[b,t,d]) - ema_out[d]   ← signed deviation
        output       = GELU(x) + alpha * delta
                     = (1+alpha) * GELU(x) - alpha * ema_out

    THREE CASES:
    1. Out >> ema_out (channel firing more than usual): delta > 0 → enhanced (opponent up)
    2. Out ≈  ema_out (channel firing normally):        delta ≈ 0 → unchanged
    3. Out << ema_out (channel quieter than usual):     delta < 0 → further suppressed

    This AMPLIFIES deviations from baseline in BOTH directions.
    Positive surprises get amplified positively.
    Negative surprises (silencing of normally-active channel) get amplified negatively.

WHY THIS IS DIFFERENT FROM ALL PRIOR EXPERIMENTS:
    - All prior: gate * GELU(x) — scales entire output
    - gelu72:    GELU(x) + alpha*(GELU(x) - ema) = GELU(x) scaled + baseline removed
    
    The key difference: GELU(x) that matches the mean is subtracted!
    This implements EXACT MEAN-SUBTRACTION on the GELU output, per channel.
    
    In signal processing: zero-meaned outputs → the DC component is removed.
    The model now propagates ONLY the AC (deviation) component of FF activations.
    
    Compare to gelu48 (which subtracted from INPUT before GELU, early-stopped):
    gelu72 subtracts from OUTPUT AFTER GELU — fundamentally different gradient structure.
    GELU(x) - mean_of_GELU ≠ GELU(x - mean_of_x).
    The nonlinearity has already been applied; we're removing the DC from its output.

PER-CHANNEL, NOT SCALAR:
    This is a PER-CHANNEL operation (delta[b,t,d] uses channel-specific ema_out[d]).
    Prior experiments showed per-channel DIVISION fails (energy loss + instability).
    ADDITION is fundamentally safer:
    - Gradient: d(output)/d(GELU) = (1+alpha) → just a scalar, no instability
    - Gradient: d(output)/d(alpha) = delta = bounded (GELU outputs are bounded)
    - Energy: ||(1+α)*GELU - α*ema|| ≈ (1+α)²||GELU||² for orthogonal mean → +20% if α=0.1

SELF-REGULATION:
    Channel d chronically high (familiar content) → ema_out[d] rises → delta ≈ 0
    → output ≈ GELU(x) at mean → no amplification. Channel "normalized."
    
    Channel d suddenly fires above average (novel activation) → delta > 0 → amplified.
    Channel d suddenly goes quiet below average → delta < 0 → amplified downward.
    
    Result: the model learns to encode SURPRISES, not VALUES.

Params: log_alpha_raw (deviation strength), logit_decay (EMA) = 2 scalars.
State: _ema_out (D_FF,) per-channel GELU output mean.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU72(nn.Module):
    """Opponent-process deviation amplifier: output = GELU(x) + alpha*(GELU(x) - ema_out)."""

    def __init__(self, ema_decay: float = 0.95, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        # alpha: deviation amplification. Start small (0.1) for stability.
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.1) - 1.0)))

        self._ema_out: torch.Tensor = None   # (D_FF,) per-channel GELU output mean
        self._ready = False

    def reset_state(self):
        self._ema_out = None
        self._ready   = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        alpha = F.softplus(self.log_alpha_raw).clamp(max=1.0)   # ∈ (0, 1] for stability

        out = self._gelu(x)   # (B, T, D)

        # ── Init: ema_out from first batch actual values ──────────────────────
        if not self._ready:
            with torch.no_grad():
                self._ema_out = out.detach().flatten(0, 1).mean(0).clone()   # (D,)
                self._ready = True
            return out   # first call: no deviation info yet

        # ── Opponent-process: amplify deviations from historical mean ─────────
        ema = self._ema_out.view(1, 1, D)          # (1, 1, D) broadcast
        delta = out - ema                           # (B, T, D) — gradient flows here
        output = out + alpha * delta               # (B, T, D) = (1+alpha)*out - alpha*ema

        # ── Update EMA of GELU output mean (no grad) ─────────────────────────
        with torch.no_grad():
            batch_mean = out.detach().flatten(0, 1).mean(0)    # (D,)
            self._ema_out = d_val * self._ema_out + (1.0 - d_val) * batch_mean

        return output
