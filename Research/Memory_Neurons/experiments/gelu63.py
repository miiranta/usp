"""GELU63 – Depression × EMA Cosine: Dual-Axis Suppression.

THE ORTHOGONALITY INSIGHT:
    gelu28/31 suppress based on DIRECTION similarity to cross-batch EMA:
        familiar if current GELU output direction ≈ average past direction
        → catches: "is this the kind of activation I usually produce?"

    gelu56 suppresses based on TEMPORAL FIRING RECENCY in this sequence:
        familiar if this channel fired heavily in the last few positions
        → catches: "has this channel been active recently in this passage?"

    These are ORTHOGONAL signals:
    - A common function word ("the") is directionally familiar (gelu28 fires)
      but may not be spatially consecutive (gelu56 misses it).
    - Within one passage, "bank" repeating 3x is temporally familiar (gelu56 fires)
      but may be an unusual word overall (gelu28 misses it).

GELU63: MULTIPLY BOTH GATES
    out         = GELU(x)
    r_gate[t]   = synaptic_depression(out[0..t])            (B, T, D) ∈ (0,1]
    cos_gate[t] = (1 - α) + α * exp(-τ * cosine(out, ema)) (B, T, 1) scalar
    output      = out * r_gate_normalized * cos_gate

    r_gate_normalized = r_gate / mean_D(r_gate)   (contrast-normalized, from GELU62)

    The two gates suppress along different axes:
    cos_gate is scalar per token (suppresses whole vector based on direction)
    r_gate is per-channel (suppresses channels that have been recently active)

    Combined: a token that's both directionally familiar AND has recently active
    channels gets doubly suppressed. A genuinely novel token passes both gates.

EMA update: ema_out tracks mean GELU output direction across batches.
Depression update: ema_resource tracks mean channel resource per batch.

Params: logit_U, log_tau_rec, logit_decay_res (depression) +
        logit_decay_cos, log_tau_cos, log_blend (cosine) = 6 scalars.
State:  _ema_resource (D,), _ema_out (D,) — two independent EMA vectors.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU63(nn.Module):
    """Synaptic depression × EMA cosine gate — orthogonal dual-axis suppression."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # Depression arm (from GELU62)
        self.logit_U         = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))
        self.log_tau_rec     = nn.Parameter(torch.tensor(math.log(math.exp(4.0) - 1.0)))
        self.logit_decay_res = nn.Parameter(torch.tensor(math.log(0.95 / 0.05)))

        # Cosine arm (from gelu28)
        self.logit_decay_cos = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))
        self.log_tau_cos     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend       = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # α ≈ 0.3

        self._ema_resource: torch.Tensor = None
        self._ema_out:      torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_resource = None
        self._ema_out      = None
        self._ready        = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        U_val      = torch.sigmoid(self.logit_U).detach().item()
        tau_val    = F.softplus(self.log_tau_rec).clamp(min=0.5).detach().item()
        d_res      = torch.sigmoid(self.logit_decay_res).detach().item()
        rec_rate   = 1.0 - math.exp(-1.0 / tau_val)

        d_cos      = torch.sigmoid(self.logit_decay_cos).detach().item()
        tau_cos    = self.log_tau_cos.exp()
        alpha      = torch.sigmoid(self.log_blend)

        out = self._gelu(x)   # (B, T, D)

        # ── Initialise both EMA states on first call ──────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._ema_resource = torch.ones(D, device=x.device, dtype=out.dtype)
                self._ema_out = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
                self._ready = True
            return out

        # ── Arm 1: contrast-normalized synaptic depression ────────────────────
        r = self._ema_resource.unsqueeze(0).expand(B, D).clone()
        r_trace = []
        with torch.no_grad():
            for t in range(T):
                r_trace.append(r.clone())
                firing = torch.tanh(out[:, t, :].detach().abs())
                used   = (U_val * r * firing).clamp(max=r * 0.99)
                rec    = (1.0 - r) * rec_rate
                r      = (r - used + rec).clamp(0.01, 1.0)

        r_gates = torch.stack(r_trace, dim=1)   # (B, T, D)
        r_norm  = r_gates / (r_gates.mean(dim=-1, keepdim=True) + self.eps)  # (B, T, D)

        # ── Arm 2: EMA cosine gate (scalar per token, from gelu28) ───────────
        out_norm = F.normalize(out, dim=-1)                             # (B, T, D)
        ema_norm = F.normalize(self._ema_out, dim=0).view(1, 1, D)     # (1, 1, D)
        cos_sim  = (out_norm * ema_norm).sum(-1)                        # (B, T)
        novelty  = torch.exp(-tau_cos * cos_sim)                        # (B, T)
        cos_gate = ((1.0 - alpha) + alpha * novelty).unsqueeze(-1)     # (B, T, 1)

        # ── Combine: multiply both gates ─────────────────────────────────────
        output = out * r_norm * cos_gate                                # (B, T, D)

        # ── Update both EMA states ────────────────────────────────────────────
        with torch.no_grad():
            self._ema_resource = d_res * self._ema_resource + (1 - d_res) * r.mean(0)
            bm = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
            self._ema_out = d_cos * self._ema_out + (1 - d_cos) * bm

        return output
