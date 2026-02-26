"""GELU62 – Contrast-Normalized Synaptic Depression.

DIAGNOSIS OF GELU56:
    output = GELU(x) * r_gates   where r_gates ∈ (0.01, 1.0)
    
    TWO PROBLEMS:
    1. Energy destruction: multiplying by r < 1 reduces mean activation energy,
       shrinking effective hidden state magnitude and slowing learning.
    2. Unclamped firing: depletion uses raw |GELU(x)| which can be 0-5+,
       causing rapid catastrophic depletion for large activations.

FIX 1 — Bounded firing signal:
    firing = tanh(|GELU(x)|)   ∈ (0, 1)   instead of raw |GELU(x)|
    Depletion is now bounded regardless of activation magnitude.

FIX 2 — Contrast normalization (energy conservation):
    Instead of: output = out * r_gates
    Use:        output = out * (r_gates / mean_D(r_gates))
    
    This normalizes the gate so mean across channels = 1.0.
    Depleted channels are suppressed; novel channels are RELATIVELY AMPLIFIED.
    Total energy (mean gate) stays at 1.0 → no systematic energy loss.
    
    This is the critical change: it converts "suppress familiar"
    into "suppress familiar AND amplify novel" — the desired signal separation.

BIOLOGICAL ANALOGY:
    In visual cortex, lateral inhibition means active neurons suppress neighbors.
    The net effect is CONTRAST ENHANCEMENT, not just suppression.
    The most active neuron looks even more distinct after normalization.

Params: logit_U (utilization), log_tau_rec (recovery), logit_decay (EMA init) = 3.
State:  _ema_resource (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU62(nn.Module):
    """Contrast-normalized synaptic depression: energy-preserving channel competition."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_U      = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))   # U ≈ 0.2
        self.log_tau_rec  = nn.Parameter(torch.tensor(math.log(math.exp(4.0) - 1.0)))  # τ ≈ 4
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(0.95 / 0.05)))

        self._ema_resource: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_resource = None
        self._ready        = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        U_val   = torch.sigmoid(self.logit_U).detach().item()
        tau_val = F.softplus(self.log_tau_rec).clamp(min=0.5).detach().item()
        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        rec_rate = 1.0 - math.exp(-1.0 / tau_val)

        out = self._gelu(x)   # (B, T, D)

        if not self._ready:
            with torch.no_grad():
                self._ema_resource = torch.ones(D, device=x.device, dtype=out.dtype)
                self._ready = True
            return out

        r = self._ema_resource.unsqueeze(0).expand(B, D).clone()   # (B, D)
        r_trace = []

        with torch.no_grad():
            for t in range(T):
                r_trace.append(r.clone())
                # FIX 1: tanh-bounded firing signal ∈ (0, 1)
                firing = torch.tanh(out[:, t, :].detach().abs())   # (B, D)
                used   = (U_val * r * firing).clamp(max=r * 0.99)
                rec    = (1.0 - r) * rec_rate
                r      = (r - used + rec).clamp(0.01, 1.0)

        r_gates = torch.stack(r_trace, dim=1)   # (B, T, D)

        # FIX 2: contrast normalization — mean gate = 1.0 per token
        r_norm = r_gates / (r_gates.mean(dim=-1, keepdim=True) + self.eps)  # (B, T, D)
        output = out * r_norm

        with torch.no_grad():
            self._ema_resource = d_val * self._ema_resource + (1 - d_val) * r.mean(0)

        return output
