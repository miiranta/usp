"""GELU66 – Shunting Inhibition (Activity-Gated Divisive Normalization).

THE BIOLOGICAL MECHANISM:
    Inhibitory interneurons in cortex release GABA, opening Cl⁻/K⁺ channels.
    Unlike excitatory synapses (which ADD current), these increase membrane
    CONDUCTANCE — they "shunt" the excitatory current to ground.

    Math: instead of  V += I_exc - I_inh  (subtractive)
          the cell sees V = I_exc / (g_leak + g_inh)  (divisive — shunting)

    KEY PROPERTY: the inhibition is SILENT at rest (no effect if cell is quiet)
    but strongly attenuates large responses. This is GAIN CONTROL, not subtraction.

THE CRITICAL DIFFERENCE FROM ALL PRIOR EXPERIMENTS:
    All suppression so far: output = GELU(x) - something  OR  GELU(x) * scalar
    Shunting: output[d] = GELU(x)[d] / (1 + g[d])

    Division naturally preserves DYNAMIC RANGE — it compresses large familiar
    signals without cutting off small novel signals.
    Addition/subtraction can accidentally zero-out or invert signals.

IMPLEMENTATION:
    g[d] = per-channel conductance = EMA of |GELU(x)[d]| over past batches
         = "how much has this channel been active historically?"
    
    output[b,t,d] = GELU(x[b,t,d]) / (1 + alpha * g[d])
    
    THEN: contrast-normalize so mean divisor = 1:
        effective_g = alpha * g / mean(alpha * g)
        output = GELU(x) / (1 + effective_g - 1)   -- avoids pure division-by-mean drift

    Simpler stable form:
        w[d] = g[d] / mean(g)                    relative conductance (mean=1)
        output[d] = GELU(x)[d] / (1 + alpha * (w[d] - 1))
                  = GELU(x)[d] * 1/(1 + alpha*(w[d]-1))

    Channels with above-average conductance (familiar, historically active) →
    w[d] > 1 → denominator > 1 → suppressed.
    Channels with below-average conductance (quiet, novel) →
    w[d] < 1 → denominator < 1 → AMPLIFIED.

SELF-REGULATION:
    Channel d fires a lot → g[d] rises → w[d] > 1 → output reduced
    → less contribution to embedding → reduced firing → g[d] decays
    Equilibrium: all channels stabilize to equal conductance (homeostasis).

STABILITY GUARANTEE:
    1 + alpha*(w-1) ≥ 1 - alpha (since w ≥ 0). Clamp alpha ≤ 0.9 → denom ≥ 0.1.
    No explosion possible.

Params: log_alpha_raw (shunting strength), logit_decay (EMA rate) = 2 scalars.
State:  _ema_g (D_FF,) per-channel EMA conductance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU66(nn.Module):
    """Shunting inhibition: divisive conductance gate with relative normalization."""

    def __init__(self, ema_decay: float = 0.95, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        # alpha: shunting strength ∈ (0, 0.9)
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # ≈ 0.3

        self._ema_g: torch.Tensor = None   # (D,) per-channel conductance
        self._ready = False

    def reset_state(self):
        self._ema_g = None
        self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        alpha = F.softplus(self.log_alpha_raw).clamp(max=0.9)   # ∈ (0, 0.9]
        d_val = torch.sigmoid(self.logit_decay).detach().item()

        out = self._gelu(x)   # (B, T, D)

        # ── Init from actual first-batch activity levels ───────────────────────
        if not self._ready:
            with torch.no_grad():
                g_init = out.detach().abs().flatten(0, 1).mean(0).clamp(min=self.eps)
                self._ema_g = g_init.clone()
                self._ready = True
            return out   # warm-up: g = actual activity, gate ≈ 1 immediately

        # ── Relative conductance: mean = 1 across channels ────────────────────
        g_rel = self._ema_g / (self._ema_g.mean() + self.eps)   # (D,) mean = 1.0

        # Shunting divisor: high-conductance channels (g_rel > 1) are suppressed
        # divisor[d] = 1 + alpha * (g_rel[d] - 1)  ∈ (1-alpha, 1+alpha)
        divisor = (1.0 + alpha * (g_rel - 1.0)).clamp(min=0.1).view(1, 1, D)  # (1,1,D)
        output  = out / divisor                                  # (B, T, D)

        # ── Update EMA conductance (no grad) ──────────────────────────────────
        with torch.no_grad():
            g_batch = out.detach().abs().flatten(0, 1).mean(0).clamp(min=self.eps)
            self._ema_g = d_val * self._ema_g + (1.0 - d_val) * g_batch

        return output
