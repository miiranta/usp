"""GELU83 – Perpendicular-Deviation Surprise × Cosine Gate.

THE KEY INSIGHT:
    gelu71's surprise signal: ||x[b,t] - ema_x|| / ||ema_x||
    
    This includes ALL components of the deviation vector:
        dev = x - ema_x  = dev_parallel + dev_perpendicular
    
    where:
        dev_parallel = <dev, ê_x> ê_x   (component along mean direction)
        dev_perp     = dev - dev_parallel  (component PERPENDICULAR to mean)
    
    PROBLEM: dev_parallel just means "x is unusually large/small along its usual direction."
    A token twice as intense in the same direction as average is NOT semantically novel —
    it's the same type of content, just more strongly activated.
    
    The TRULY novel component is dev_perp: the token is going in a DIFFERENT direction.
    Direction change is what matters for semantic distinction.

THE RELATIONSHIP TO COSINE GATE:
    The cosine gate captures: cos(x, ê_x) → direction familiarity.
    Perpendicular surprise captures: ||dev_perp|| → magnitude of DIRECTION CHANGE.
    
    These are exactly complementary:
    cos(x, ê_x) = 1 - ||dev_perp||²/(2||x||²) approximately for small deviations.
    But cosine is NORMALIZED (doesn't care about magnitude), while perp magnitude DOES.
    
    Together: cos gate says "is the direction familiar?"
              perp surprise says "how MUCH did the direction change (in absolute terms)?"

IMPLEMENTATION:
    ê_x = ema_x / ||ema_x||     ← unit mean direction
    dev = x[b,t] - ema_x
    dev_par  = (dev · ê_x) × ê_x   ← projection
    dev_perp = dev - dev_par         ← perpendicular component
    
    surp_perp = tanh(σ × ||dev_perp|| / ||ema_x||)
    
    gate = exp(-τ × cos(out, ema_out)) × (1 + w × surp_perp)
    
    NOTE: We use OUTPUT cosine (as in gelu71) but PERPENDICULAR INPUT deviation as surprise.
    The cosine is on the output because that's what downstream layers receive.
    The perpendicular is on the input because that's where the direction change is clearest.

WHAT THIS REMOVES:
    - Tokens that are more intense than usual IN THE USUAL DIRECTION: no surprise boost.
    - A word that frequently appears in strong positive contexts and is now even stronger:
      dev is mostly parallel to ema_x → small dev_perp → low surprise.
    
    This ONLY boosts tokens that are going in different directions from usual:
    - A usually-positive channel is now pulling negative (antonym context)
    - A usually-uniform activation now has sharp per-channel variation
    - Tokens from a completely different semantic domain

Params: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars (same as gelu71).
State: _ema_x (D,) raw mean (for direction), _ema_out (D,) unit for cosine.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU83(nn.Module):
    """Perpendicular-deviation surprise: only directional change (not magnitude) boosts gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_x:    torch.Tensor = None   # (D,) raw mean for perpendicular projection
        self._ema_out:  torch.Tensor = None   # (D,) unit vector for cosine
        self._ema_x_norm: float = 1.0
        self._ready = False

    def reset_state(self):
        self._ema_x   = None
        self._ema_out = None
        self._ready   = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xm = x.detach().flatten(0,1).mean(0)
                om = out.detach().flatten(0,1).mean(0)
                self._ema_x      = xm.clone()
                self._ema_x_norm = xm.norm().item() + self.eps
                self._ema_out    = F.normalize(om, dim=0).clone()
                self._ready      = True
            return out

        # ── Perpendicular deviation surprise ────────────────────────────────
        ema_x_unit = F.normalize(self._ema_x, dim=0).view(1, 1, D)   # ê_x (1,1,D)
        dev = x.detach() - self._ema_x.view(1, 1, D)                  # (B,T,D)
        # Parallel component: scalar projection onto ê_x
        dev_par_mag = (dev * ema_x_unit).sum(-1, keepdim=True)        # (B,T,1)
        dev_par     = dev_par_mag * ema_x_unit                        # (B,T,D)
        dev_perp    = dev - dev_par                                    # (B,T,D)
        
        surp_perp = torch.tanh(sigma * dev_perp.norm(dim=-1) / (self._ema_x_norm + self.eps))

        # ── Cosine familiarity gate (output direction vs EMA) ────────────────
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_on  = F.normalize(self._ema_out, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_on).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surp_perp)
        output = out * gate.unsqueeze(-1)

        with torch.no_grad():
            xm = x.detach().flatten(0,1).mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_x      = d_val * self._ema_x   + (1-d_val) * xm
            self._ema_x_norm = self._ema_x.norm().item() + self.eps
            self._ema_out    = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
