"""GELU82 – Input + Output Combined Surprise, No Contrast Norm.

MOTIVATION:
    gelu74 combined input + output deviation BUT used contrast normalization,
    which caused early-stopping failure (ep5 just barely missed threshold).
    
    gelu77 uses output surprise ALONE — same structure as gelu71, just output space.
    
    This experiment: COMBINE both input and output surprise WITHOUT contrast norm.
    The hypothesis: two independent orthogonal surprise signals (input and output space)
    provide richer novelty detection than either alone.

WHY TWO SURPRISE SIGNALS ARE COMPLEMENTARY:
    Input surprise:  ||x[b,t] - ema_x||        — deviation in pre-GELU space
    Output surprise: ||out[b,t] - ema_out_raw|| — deviation in post-GELU space
    
    These are NOT always correlated because GELU is nonlinear:
    - A large input gain might hit GELU saturation → output barely changes (low output surprise)
    - A small input change at the inflection point → large output response (high output surprise)
    
    Combined, they cover the full range of FF layer dynamics:
    high input + high output surprise → maximum novelty
    high input but low output  → large input but GELU absorbed it
    low input but high output  → small input at sensitive GELU region
    low input + low output     → familiar, suppress

FORMULA (same as gelu71 but with two additive surprise terms):
    gate = exp(-τ × cos(out, ema_out_dir))
           × (1 + w_in  × tanh(σ_in  × ||x - ema_x||       / norm_x))
           × (1 + w_out × tanh(σ_out × ||out - ema_out_raw|| / norm_out))
    
    The product form (not sum) ensures both must "agree" to get maximum boost:
    gate is large iff: (1) not familiar direction AND (2) unusually different input
    AND (3) unusually different output. Triple conjunction. Very selective.
    
    Alternative sum form: gate = cos_gate × (1 + w_in×surp_in + w_out×surp_out)
    For simplicity and consistency with gelu71, use the SUM form.
    
    Both forms are valid; sum is less selective (OR logic), product is more selective (AND logic).
    Starting with SUM for close comparison with gelu71 (w_in=w_out≈1 → 2x boost vs gelu71's 1x).

NO CONTRAST NORM:
    Omitting contrast norm for early-stop safety.
    Mean gate ≈ mean(cos_gate) × (1 + two small boosts) < 1.
    Energy reduction compensated by residual learning as usual.

Params: logit_decay, log_tau, log_sigma_in, log_sigma_out, log_w_in, log_w_out = 6 scalars.
State: _ema_out_dir (D,) unit; _ema_out_raw (D,) raw; _ema_x (D,) raw.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU82(nn.Module):
    """Input + output combined surprise gate, sum form, no contrast norm."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        # Init each surprise weight at ~0.5 (combined = ~1.0, matching gelu71)
        self.log_sigma_in  = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_sigma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_in      = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_w_out     = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

        self._ema_out_dir:  torch.Tensor = None
        self._ema_out_raw:  torch.Tensor = None
        self._ema_x:        torch.Tensor = None
        self._ema_out_norm: float = 1.0
        self._ema_x_norm:   float = 1.0
        self._ready = False

    def reset_state(self):
        self._ema_out_dir = None
        self._ema_out_raw = None
        self._ema_x       = None
        self._ready       = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val    = torch.sigmoid(self.logit_decay).detach().item()
        tau      = self.log_tau.exp()
        sig_in   = F.softplus(self.log_sigma_in)
        sig_out  = F.softplus(self.log_sigma_out)
        w_in     = F.softplus(self.log_w_in)
        w_out    = F.softplus(self.log_w_out)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xm = x.detach().flatten(0,1).mean(0)
                om = out.detach().flatten(0,1).mean(0)
                self._ema_x        = xm.clone()
                self._ema_x_norm   = xm.norm().item() + self.eps
                self._ema_out_dir  = F.normalize(om, dim=0).clone()
                self._ema_out_raw  = om.clone()
                self._ema_out_norm = om.norm().item() + self.eps
                self._ready        = True
            return out

        # Cosine familiarity
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        # Input surprise
        ema_x = self._ema_x.view(1, 1, D)
        surp_in = torch.tanh(sig_in * (x.detach() - ema_x).norm(dim=-1) / (self._ema_x_norm + self.eps))

        # Output surprise
        ema_or  = self._ema_out_raw.view(1, 1, D)
        surp_out = torch.tanh(sig_out * (out.detach() - ema_or).norm(dim=-1) / (self._ema_out_norm + self.eps))

        gate   = gate_cos * (1.0 + w_in * surp_in + w_out * surp_out)
        output = out * gate.unsqueeze(-1)

        with torch.no_grad():
            xm = x.detach().flatten(0,1).mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_x        = d_val * self._ema_x       + (1-d_val) * xm
            self._ema_x_norm   = self._ema_x.norm().item() + self.eps
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)
            self._ema_out_raw  = d_val * self._ema_out_raw  + (1-d_val) * om
            self._ema_out_norm = self._ema_out_raw.norm().item() + self.eps

        return output
