"""GELU67 – Adaptive Firing Threshold (Spike Threshold Accommodation).

THE BIOLOGICAL MECHANISM:
    In real neurons, the action potential threshold is NOT fixed.
    Sustained depolarization causes sodium channel inactivation and slow K+ activation,
    gradually raising the threshold required to trigger a spike (accommodation).
    
    Result: a steady input → threshold rises → neuron stops firing even though
            input is still applied. Only CHANGES (novel or sudden inputs) fire.

    This is distinct from Ca-AHP (which reduces post-spike hyperpolarization) —
    threshold adaptation happens PRE-SPIKE, changing what CAN fire.

IMPLEMENTATION:
    θ[d] = per-channel EMA of the INPUT x (not output) → the "expected input level"
    
    output = GELU(x - θ)    ← apply nonlinearity to THRESHOLD-SHIFTED input

    Channels where x ≈ θ (usual value) → argument of GELU ≈ 0 → GELU output ≈ 0
    Channels where x >> θ (unusual spike) → GELU fires strongly
    Channels where x << θ (below average) → GELU still fires (just less)

    θ tracks EMA of:  batch_mean_x[d] = mean over (B, T) of x[b, t, d]
    decay rate learned; init θ = 0 (GELU(x - 0) = GELU(x) at start)

WHY THIS IS POWERFUL:
    The GELU nonlinearity has maximum sensitivity near zero (the "knee"):
        GELU'(0) = 0.5 + 1/sqrt(2π) ≈ 0.9  (steep)
        GELU'(3) ≈ 1.0  (flat, already saturated)
    
    By centering x around 0 via θ subtraction, EACH CHANNEL is operated near
    its maximum sensitivity point — the model always works on DEVIATIONS FROM
    EXPECTED, not on raw values. This is exactly what barrel cortex neurons do:
    they respond to whisker deflection (change), not whisker position (DC).

    Crucially: GELU(x - θ) is nonlinearly different from GELU(x) - GELU(θ).
    The threshold shifts WHERE the nonlinearity operates, changing the gradient
    landscape for the whole training process.

RELATIONSHIP TO PRIOR EXPERIMENTS:
    gelu37: output = GELU(x) + alpha*(GELU(x) - EMA_output) — post-GELU subtraction
    GELU67: output = GELU(x - θ)                            — pre-GELU threshold shift
    
    gelu67 affects the GRADIENT of the activation; gelu37 does not.
    When θ[d] ≈ mean(x[d]), the effective GELU slope for channel d is maximized.

SELF-REGULATION:
    x[d] chronically high → θ[d] rises to match → GELU(x-θ) stays moderate
    x[d] drops suddenly (novel) → θ[d] still high → GELU fires ≈ 0 (below threshold)
    Wait a few steps → θ decays toward new low level → sensitivity restored
    This is PERFECT repetition suppression: the threshold chases the signal.

Params: logit_decay (EMA speed), log_scale_raw (optional output scale) = 2 scalars.
State:  _ema_x (D_FF,) per-channel input threshold.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU67(nn.Module):
    """Adaptive firing threshold: subtract per-channel EMA of input before GELU."""

    def __init__(self, ema_decay: float = 0.95):
        super().__init__()
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        # Optional learned output scale (init 1.0, so at start = standard GELU)
        self.log_scale_raw = nn.Parameter(torch.tensor(0.0))  # softplus(0) ≈ 0.69 → but we add 0.31

        self._ema_x: torch.Tensor = None   # (D,) per-channel input threshold
        self._ready = False

    def reset_state(self):
        self._ema_x = None
        self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        # scale output so energy is conserved: init ~1.0
        scale = F.softplus(self.log_scale_raw) + 0.31   # softplus(0)+0.31 ≈ 1.0

        # ── Init: θ = batch mean of x so first call has GELU(x - mean(x)) ────
        if not self._ready:
            with torch.no_grad():
                self._ema_x = x.detach().flatten(0, 1).mean(0).clone()   # (D,)
                self._ready = True
            # First call: apply threshold immediately (well-initialized)

        # ── Threshold-shifted GELU ────────────────────────────────────────────
        theta = self._ema_x.view(1, 1, D)    # (1, 1, D) broadcast
        output = self._gelu(x - theta) * scale   # (B, T, D)

        # ── Update per-channel EMA of input (no grad) ─────────────────────────
        with torch.no_grad():
            x_batch = x.detach().flatten(0, 1).mean(0)   # (D,)
            self._ema_x = d_val * self._ema_x + (1.0 - d_val) * x_batch

        return output
