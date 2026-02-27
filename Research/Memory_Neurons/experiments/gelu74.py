"""GELU74 – Dual-Axis Surprise (Input + Output Deviation) × Cosine Gate.

THE ADDITIONAL SIGNAL NOT USED IN GELU71:
    gelu71 measures surprise on the INPUT x: tanh(σ * ||x - ema_x|| / norm)
    
    But the GELU nonlinearity can convert a moderate input surprise into a large
    output surprise, or dilute a large input surprise if it's in the saturated region.
    
    Example:
        x = 3.0 (normal), ema_x = 2.8 (close)  → small input surprise
        GELU(3.0) = 2.994, GELU(2.8) = 2.627    → large output surprise (14% diff)
        
        x = -0.5 (normal), ema_x = 0.0          → moderate input surprise
        GELU(-0.5) = -0.154, GELU(0.0) = 0.0    → similar output surprise
    
    The nonlinearity amplifies surprises near its inflection point (x near 0)
    and compresses surprises in the saturated regions (|x| >> 0).
    
    Using BOTH gives a more complete picture of "how unusual is this activation."

IMPLEMENTATION:
    input_surprise  = tanh(σ_in  × ||x[b,t] - ema_x[d]|| / norm_x)   ← gelu71 signal
    output_surprise = tanh(σ_out × ||out[b,t] - ema_out_raw[d]|| / norm_out)  ← new
    
    gate = gate_cos × (1 + w_in × input_surprise + w_out × output_surprise)
    
    NOTE: ema_out_raw[d] = EMA of E[GELU(x)[d]] as a RAW VECTOR (not normalized)
          This is different from gelu71's ema_out that is kept as unit vector.
          The output deviation is in the same space as the output values.

    Then CONTRAST NORMALIZE (from gelu73's analysis):
    gate_norm = gate / mean(gate)
    output    = GELU(x) × gate_norm

WHY THIS BEATS gelu71:
    The model now responds to surprise in TWO independent spaces:
    1. Pre-GELU (linear feature space) — what did the FF layer receive?
    2. Post-GELU (processed feature space) — what is the FF layer outputting?
    
    A token that is simultaneously surprising in BOTH spaces (novel content AND
    novel activation pattern) gets the maximum gate boost.
    
    A token that is surprising in input but not in output (the nonlinearity
    absorbed the deviation) gets a partial boost.
    
    This creates a FINER-GRAINED novelty signal than either alone.

Params: logit_decay, log_tau, log_sigma_in, log_sigma_out, log_w_in, log_w_out = 6 scalars.
State: _ema_x (D,), _ema_x_norm, _ema_out_dir (D, unit), _ema_out_raw (D,), _ema_out_norm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU74(nn.Module):
    """Dual-axis surprise: (input deviation + output deviation) × cosine gate, contrast-norm."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay    = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau        = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_in   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_sigma_out  = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_in       = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_w_out      = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

        self._ema_x:       torch.Tensor = None
        self._ema_out_dir: torch.Tensor = None   # unit vector for cosine
        self._ema_out_raw: torch.Tensor = None   # raw mean for deviation
        self._ema_x_norm:  float = 1.0
        self._ema_out_norm: float = 1.0
        self._ready = False

    def reset_state(self):
        self._ema_x        = None
        self._ema_out_dir  = None
        self._ema_out_raw  = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val    = torch.sigmoid(self.logit_decay).detach().item()
        tau      = self.log_tau.exp()
        sigma_in = F.softplus(self.log_sigma_in)
        sigma_out = F.softplus(self.log_sigma_out)
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

        # ── Cosine familiarity gate ────────────────────────────────────────
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        # ── Input-deviation surprise ───────────────────────────────────────
        ema_x   = self._ema_x.view(1, 1, D)
        delta_in = (x.detach() - ema_x).norm(dim=-1)
        surp_in  = torch.tanh(sigma_in * delta_in / (self._ema_x_norm + self.eps))

        # ── Output-deviation surprise ──────────────────────────────────────
        ema_or  = self._ema_out_raw.view(1, 1, D)
        delta_out = (out.detach() - ema_or).norm(dim=-1)
        surp_out  = torch.tanh(sigma_out * delta_out / (self._ema_out_norm + self.eps))

        # ── Combined gate ──────────────────────────────────────────────────
        gate_raw = gate_cos * (1.0 + w_in * surp_in + w_out * surp_out)

        # Contrast normalize
        with torch.no_grad():
            gate_mean = gate_raw.mean().clamp(min=self.eps)
        gate_norm = gate_raw / gate_mean
        output = out * gate_norm.unsqueeze(-1)

        # ── Update EMA states ──────────────────────────────────────────────
        with torch.no_grad():
            xm = x.detach().flatten(0,1).mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_x        = d_val * self._ema_x + (1.0 - d_val) * xm
            self._ema_x_norm   = self._ema_x.norm().item() + self.eps
            self._ema_out_dir  = d_val * self._ema_out_dir + (1.0 - d_val) * F.normalize(om, dim=0)
            self._ema_out_raw  = d_val * self._ema_out_raw + (1.0 - d_val) * om
            self._ema_out_norm = self._ema_out_raw.norm().item() + self.eps

        return output
