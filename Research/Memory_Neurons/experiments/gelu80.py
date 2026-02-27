"""GELU80 – Per-Channel Z-Score Gate (Channel-Normalized Surprise).

THE KEY INSIGHT NOT CAPTURED BY GELU71:
    gelu71's surprise = tanh(σ × ||x[b,t] - ema_x|| / ||ema_x||)
    
    This is a SINGLE SCALAR per token — it collapses all D channels into one number.
    The result: channel-dominant directions drive the surprise signal.
    
    In high dimensions, the L2 norm is dominated by the ~sqrt(D) channels with largest
    activation variance. Channels with smaller absolute activations but proportionally
    large deviations are INVISIBLE to the scalar surprise.
    
    Example (D=4): ema_x = [10, 0.1, 0.1, 0.1], x = [10.5, 0.5, 0.5, 0.5]
    - Scalar surprise: ||x - ema|| / ||ema|| = sqrt(0.25 + 0.16*3) / sqrt(100+0.03) ≈ 0.084 (low!)
    - Channel-0 deviation: 0.5/10 = 5% (small)
    - Channels 1-3 deviation: 0.4/0.1 = 400% each (HUGE but invisible!)
    
    Per-channel z-score captures this:
    z[d] = (x[b,t,d] - ema_x[d]) / (std_x[d] + eps)
    
    Each channel is normalized by its OWN historical standard deviation.  
    A 400% deviation on a low-variance channel is just as "surprising" as a proportional
    deviation on a high-variance channel. This is statistically principled.

IMPLEMENTATION:
    Track per-channel statistics cross-batch (no grad):
        ema_mean[d]  = EMA of x̄[d]         — already in gelu71
        ema_sq[d]    = EMA of x²[d]         — new: needed for std
        std[d]       = sqrt(ema_sq[d] - ema_mean[d]² + eps_var)
    
    Per-channel z-score:
        z[b,t,d]    = (x[b,t,d] - ema_mean[d]) / (std[d] + eps)   — (B,T,D)
    
    Aggregate surprise (mean absolute z-score):
        surp[b,t]   = tanh(σ × mean_d(|z[b,t,d]|))      — (B,T)
    
    Gate (same structure as gelu71):
        gate = exp(-τ × cos(out[b,t], ema_out)) × (1 + w × surp[b,t])
    
    WHY MEAN ABSOLUTE Z-SCORE:
    - |z[d]| = how many std devs did channel d deviate?
    - mean_d(|z|) = average surprise across all channels
    - tanh(σ × mean|z|) ∈ (0,1) — bounded, smooth, positive
    - High when MANY channels deviate simultaneously OR when a FEW deviate enormously
    - Captures both diffuse and concentrated surprise

STABILITY:
    At training start: ema_mean ≈ actual mean (init from first batch), std is small.
    Small std → large z-scores initially. But tanh caps at 1, and init w small.
    The model quickly adjusts σ downward if surprise is uniformly high.
    
    Gradient: d(output)/d(x[d]) = (1+α) × GELU'(x[d]) — same as if std were fixed.
    The z-score computation is done without grad; gradients only flow through GELU(x).

Params: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars.
State: _ema_mean (D,), _ema_sq (D,), _ema_out (D,) unit vector.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU80(nn.Module):
    """Per-channel z-score surprise gate: channel-normalized individual deviations."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4   # variance floor to prevent div-by-zero on low-var channels
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        # sigma: z-score sensitivity. Init small (σ≈0.3) so mean|z|≈1 doesn't explode gate.
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean: torch.Tensor = None   # (D,) per-channel input mean
        self._ema_sq:   torch.Tensor = None   # (D,) per-channel mean of x²
        self._ema_out:  torch.Tensor = None   # (D,) unit vector, output direction
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_out  = None
        self._ready    = False

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
                xf = x.detach().flatten(0,1)     # (B*T, D)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Per-channel standard deviation from EMA ────────────────────────
        var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
        std  = var.sqrt()                                           # (D,)

        # ── Per-channel z-score ────────────────────────────────────────────
        mu_  = self._ema_mean.view(1, 1, D)
        std_ = std.view(1, 1, D)
        z    = (x.detach() - mu_) / (std_ + self.eps)              # (B, T, D)
        mean_abs_z = z.abs().mean(dim=-1)                           # (B, T)
        surprise   = torch.tanh(sigma * mean_abs_z)                 # (B, T) ∈ (0,1)

        # ── Cosine familiarity gate ────────────────────────────────────────
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surprise)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics (no grad) ───────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0,1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
