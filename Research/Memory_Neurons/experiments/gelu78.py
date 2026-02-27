"""GELU78 – Asymmetric Cosine Gate (Anti-Correlated Patterns Amplified).

THE LIMITATION OF ALL PRIOR GATES:
    Every gate so far: gate ∈ (0, 1] or gate ∈ (0, 1+δ] where δ is small.
    The maximum gate value = 1.0 (or slightly above with surprise boost).
    
    This means familiar tokens are suppressed, novel tokens are "let through."
    But tokens that are ANTI-CORRELATED with the EMA — i.e., output direction is
    OPPOSITE to the mean — are treated the same as orthogonal tokens (cos = -1 → same as 0).
    
    exp(-τ × cos(-1)) = exp(+τ) >> 1 — BUT we're currently NOT taking advantage of this!
    The exp(-τ × cos) for cos = -1 gives a gate >> 1, but the model compensates 
    by learning small τ values, because outputs >> 1 create gradient instability.

THE FIX — ASYMMETRIC GATE WITH EXPLICIT AMPLIFICATION:
    Instead of exp(-τ × cos) which can explode for large τ and negative cos,
    split the cosine gate into two monotonic, bounded arms:
    
    Suppression arm:  when cos > 0 (familiar direction)
        supp_gate = exp(-τ_s × cos)     ∈ (exp(-τ_s), 1]
    
    Amplification arm: when cos < 0 (anti-correlated = most novel)
        ampl_gate = 1 + w_a × relu(-cos)  ∈ [1, 1+w_a]
    
    Combined:
        gate = supp_gate × ampl_gate × (1 + w_surp × surprise)
             = exp(-τ_s × relu(cos)) × (1 + w_a × relu(-cos)) × (1 + w_surp × surp)
    
    BEHAVIOR:
    cos = +1 (very familiar): gate = exp(-τ_s) × 1 × (1+w_surp*surp) → suppressed
    cos =  0 (orthogonal):    gate = 1 × 1 × (1+surp) → surprise-boosted only  
    cos = -1 (anti-correlated): gate = 1 × (1+w_a) × (1+surp) → amplified AND boosted
    
    This creates a FULL asymmetry: familiar→suppress, novel→passthrough, anti-correlated→amplify.
    
    BIOLOGICAL ANALOGY:
    Off-center retinal cells: respond to DECREASE in luminance below mean, firing STRONGLY
    when illumination is the opposite of "what they expect." 
    On-center: fire to increases (familiar), suppressed by decreases.
    Together they create opponent coding of ALL deviations.
    
    gelu78 implements this: two populations — suppressed by familiar (supp_arm), 
    amplified by anti-correlated (ampl_arm). Together richer representation.

STABILITY:
    supp_gate: exp(-τ_s × relu(cos)) ∈ (0, 1] — always ≤ 1.
    ampl_gate: 1 + w_a × relu(-cos) ∈ [1, 1+w_a] — w_a init 0.5 → max = 1.5.
    Combined max: 1 × 1.5 × (1+w_surp) — controlled, no explosion.

Params: logit_decay, log_tau_s, log_w_a, log_sigma_raw, log_w_surp = 5 scalars.
State: _ema_out (D,) unit, _ema_x (D,) raw, _ema_x_norm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU78(nn.Module):
    """Asymmetric cosine gate: familiar suppressed, anti-correlated amplified, surprise-boosted."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau_s     = nn.Parameter(torch.tensor(math.log(2.0)))          # suppression sharpness
        self.log_w_a_raw   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # ampl weight ≈0.5
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_surp    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_out: torch.Tensor = None
        self._ema_x:   torch.Tensor = None
        self._ema_x_norm: float = 1.0
        self._ready = False

    def reset_state(self):
        self._ema_out  = None
        self._ema_x    = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val  = torch.sigmoid(self.logit_decay).detach().item()
        tau_s  = self.log_tau_s.exp()
        w_a    = F.softplus(self.log_w_a_raw)
        sigma  = F.softplus(self.log_sigma_raw)
        w_surp = F.softplus(self.log_w_surp)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xm = x.detach().flatten(0,1).mean(0)
                om = out.detach().flatten(0,1).mean(0)
                self._ema_out    = F.normalize(om, dim=0).clone()
                self._ema_x      = xm.clone()
                self._ema_x_norm = xm.norm().item() + self.eps
                self._ready      = True
            return out

        # Cosine similarity
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)   # (B, T)

        # Asymmetric gate
        supp_arm = torch.exp(-tau_s * cos_sim.clamp(min=0))          # ∈ (0, 1]
        ampl_arm = 1.0 + w_a * (-cos_sim).clamp(min=0)               # ∈ [1, 1+w_a]

        # Input-deviation surprise (same as gelu71)
        ema_x   = self._ema_x.view(1, 1, D)
        delta   = (x.detach() - ema_x).norm(dim=-1)
        surprise = torch.tanh(sigma * delta / (self._ema_x_norm + self.eps))

        gate   = supp_arm * ampl_arm * (1.0 + w_surp * surprise)
        output = out * gate.unsqueeze(-1)

        with torch.no_grad():
            xm = x.detach().flatten(0,1).mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_x      = d_val * self._ema_x   + (1.0-d_val) * xm
            self._ema_x_norm = self._ema_x.norm().item() + self.eps
            self._ema_out    = d_val * self._ema_out  + (1.0-d_val) * F.normalize(om, dim=0)

        return output
