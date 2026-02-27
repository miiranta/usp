"""GELU79 – Temporal Velocity Surprise × Cosine Gate.

MOTIVATION:
    All prior surprise signals (gelu71, gelu77, gelu78) compare the CURRENT token
    to a CROSS-BATCH historical average. They ask: "Is this token unusual globally?"
    
    But language has a fundamentally different kind of novelty: TRANSITIONS.
    A word can be perfectly ordinary globally but mark a critical topic transition:
        "...the quarterly earnings were strong. HOWEVER, the debt ratio..."
        "HOWEVER" triggers a contrast; it's common globally but locally surprising.
    
    Within-sequence VELOCITY captures this:
        vel[b, t] = ||out[b, t] - out[b, t-1]||   (or vs causal EMA of recent outputs)
    
    High velocity = the activation just CHANGED significantly in this sequence.
    This is the direct FF-layer analog of temporal derivative coding.

BIOLOGICAL BASIS — CHANGE DETECTORS:
    Many sensory neurons are CHANGE DETECTORS that respond to onset/offset:
    - Auditory cortex: strong response at note onset, reduced during sustained tone
    - Visual motion cells: respond to moving edges, not static luminance
    - Somatosensory: firing at pressure change, not steady pressure
    
    These neurons don't encode "what" but "what CHANGED." Combining them with
    "what is familiar" (cosine gate) captures both static and dynamic novelty.

IMPLEMENTATION:
    ema_local[b, d]: per-sample causal running average, initialized from cross-batch EMA.
    Updates each step: ema_local[b, t+1] = α_l × ema_local[b, t] + (1-α_l) × out[b, t]
    
    vel[b, t] = tanh(σ_v × ||out[b,t] - ema_local[b, t-1]|| / scale)  ← causal!
    
    Combined gate (same as gelu71 with vel replacing surp):
        gate = exp(-τ × cos(out, ema_cross)) × (1 + w_v × vel)
    
    PLUS: add cross-batch input-deviation surprise from gelu71:
        gate = exp(-τ × cos(out, ema_cross)) × (1 + w_v × vel + w_g × global_surp)
    
    Two signals:
    - vel: local transition detector (within-sequence)
    - global_surp: rare token detector (cross-batch)
    Both are multiplicative boosts on the cosine suppression gate.

CAUSAL GUARANTEE:
    ema_local at t uses only out at 0..t-1.
    At t=0: ema_local = ema_cross (warm start), velocity = ||out[0] - ema_cross|| 
    (which is essentially global surprise on the first step → graceful initialization).

Params: logit_decay_g (cross), logit_decay_l (local), log_tau, log_sigma_v, log_sigma_g, log_w_v, log_w_g = 7.
State: _ema_cross_out (D,) direction; _ema_cross_x (D,) raw; _ema_cross_x_norm; _ema_cross_out_raw (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU79(nn.Module):
    """Temporal velocity (within-seq) + global surprise × cosine gate."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay_g  = nn.Parameter(torch.tensor(math.log(0.9/0.1)))
        self.logit_decay_l  = nn.Parameter(torch.tensor(math.log(0.7/0.3)))  # local, faster
        self.log_tau        = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_v    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))  # vel sensitivity
        self.log_sigma_g    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))  # global surp
        self.log_w_v        = nn.Parameter(torch.tensor(math.log(math.exp(0.7) - 1.0)))  # vel weight
        self.log_w_g        = nn.Parameter(torch.tensor(math.log(math.exp(0.7) - 1.0)))  # global weight

        self._ema_cross_out: torch.Tensor = None   # (D,) unit vector, output direction
        self._ema_cross_x:   torch.Tensor = None   # (D,) raw input mean
        self._ema_cross_out_raw: torch.Tensor = None  # (D,) raw output mean
        self._ema_cross_x_norm:   float = 1.0
        self._ema_cross_out_norm: float = 1.0
        self._ready = False

    def reset_state(self):
        self._ema_cross_out     = None
        self._ema_cross_x       = None
        self._ema_cross_out_raw = None
        self._ready             = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_g   = torch.sigmoid(self.logit_decay_g).detach().item()
        d_l   = torch.sigmoid(self.logit_decay_l).detach().item()
        tau   = self.log_tau.exp()
        sig_v = F.softplus(self.log_sigma_v)
        sig_g = F.softplus(self.log_sigma_g)
        w_v   = F.softplus(self.log_w_v)
        w_g   = F.softplus(self.log_w_g)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xm = x.detach().flatten(0,1).mean(0)
                om = out.detach().flatten(0,1).mean(0)
                self._ema_cross_out     = F.normalize(om, dim=0).clone()
                self._ema_cross_x       = xm.clone()
                self._ema_cross_out_raw = om.clone()
                self._ema_cross_x_norm   = xm.norm().item() + self.eps
                self._ema_cross_out_norm = om.norm().item() + self.eps
                self._ready             = True
            return out

        # ── Cosine familiarity gate ─────────────────────────────────────────
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_on  = F.normalize(self._ema_cross_out, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_on).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        # ── Global surprise (cross-batch input deviation, as gelu71) ─────────
        ema_x   = self._ema_cross_x.view(1, 1, D)
        delta_g = (x.detach() - ema_x).norm(dim=-1)
        surp_g  = torch.tanh(sig_g * delta_g / (self._ema_cross_x_norm + self.eps))

        # ── Local velocity (within-sequence causal) ───────────────────────────
        ema_local = self._ema_cross_out_raw.unsqueeze(0).expand(B, D).clone()  # (B, D)
        vel_list  = []
        with torch.no_grad():
            scale = self._ema_cross_out_norm
            for t in range(T):
                delta_v = (out.detach()[:, t, :] - ema_local).norm(dim=-1)    # (B,)
                vel_t   = torch.tanh(sig_v.detach() * delta_v / (scale + self.eps))
                vel_list.append(vel_t)
                ema_local = d_l * ema_local + (1.0 - d_l) * out.detach()[:, t, :]

        vel = torch.stack(vel_list, dim=1)   # (B, T)

        # ── Combined gate ─────────────────────────────────────────────────────
        gate   = gate_cos * (1.0 + w_v * vel + w_g * surp_g)
        output = out * gate.unsqueeze(-1)

        # ── Update global EMA states ──────────────────────────────────────────
        with torch.no_grad():
            xm = x.detach().flatten(0,1).mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_cross_x       = d_g * self._ema_cross_x       + (1-d_g) * xm
            self._ema_cross_x_norm  = self._ema_cross_x.norm().item() + self.eps
            self._ema_cross_out     = d_g * self._ema_cross_out      + (1-d_g) * F.normalize(om, dim=0)
            self._ema_cross_out_raw = d_g * self._ema_cross_out_raw  + (1-d_g) * om
            self._ema_cross_out_norm = self._ema_cross_out_raw.norm().item() + self.eps

        return output
