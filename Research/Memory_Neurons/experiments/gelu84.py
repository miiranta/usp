"""GELU84 – Mahalanobis Surprise × Cosine Gate (Per-Channel Variance Normalization).

THE MATHEMATICAL MOTIVATION:
    gelu71's L2 surprise: ||x - μ||₂ / ||μ||₂
    
    This treats all D dimensions EQUALLY. But in practice:
    - Some channels have high variance (fire strongly in many contexts) → individual deviations are expected
    - Some channels have low variance (rarely fire strongly) → any deviation is surprising
    
    The STATISTICALLY CORRECT measure of "how surprising is x?" is Mahalanobis distance:
    d_M(x; μ, Σ) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
    
    With diagonal Σ (per-channel variance, simplest assumption):
    d_M = sqrt(Σ_d (x_d - μ_d)² / σ_d²) = sqrt(mean_d(z_d²)) × sqrt(D)
    
    where z_d = (x_d - μ_d) / σ_d is the per-channel z-score.
    
    NORMALIZED version: mean_d(z_d²) = mean squared z-score (= 1 for standard Gaussians)
    
DIFFERENCE FROM GELU80 (mean absolute z-score):
    gelu80: surp = tanh(σ × mean_d(|z_d|))          ← mean absolute z-score
    gelu84: surp = tanh(σ × sqrt(mean_d(z_d²)))      ← RMS z-score = norm of z-vector / sqrt(D)
    
    RMS z-score is more sensitive to CONCENTRATED large deviations (few channels but very large).
    Mean |z| is more sensitive to DIFFUSE moderate deviations (many channels slightly off).
    
    For a Gaussian: E[|z|] = sqrt(2/π) ≈ 0.798, sqrt(E[z²]) = 1.0.
    So mean|z| ≈ 0.8 while RMS z ≈ 1.0 for a typical token. More robust baseline.

PER-CHANNEL VARIANCE TRACKING:
    ema_mean[d] = EMA of mean_batch(x[b,t,d])
    ema_var[d]  = EMA of mean_batch((x[b,t,d] - ema_mean[d])²)
    
    To avoid division by near-zero variance on inactive channels:
    std[d] = sqrt(max(ema_var[d], eps_var))
    
    z[d]   = (x[b,t,d] - ema_mean[d]) / std[d]
    surp   = tanh(σ × sqrt(mean_d(z²)))

STABILITY NOTE:
    Unlike mean|z|, mean(z²) has infinite variance for heavy-tailed distributions.
    BUT: tanh(σ × sqrt(·)) bounds the output ∈ (0,1) for any input.
    The gradient through sqrt may be unstable if mean(z²) ≈ 0 (all near mean).
    SOLUTION: all z computations done with torch.no_grad(); gradients only flow through GELU(x).

Params: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars.
State: _ema_mean (D,), _ema_var (D,), _ema_out (D,) unit vector.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU84(nn.Module):
    """Mahalanobis/RMS-z surprise gate: per-channel variance normalization of deviation."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4   # variance floor
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        # Init sigma small (0.3) since RMS z ≈ 1.0 at neutral → tanh(0.3) ≈ 0.29 initial surprise
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean: torch.Tensor = None   # (D,) per-channel input mean
        self._ema_var:  torch.Tensor = None   # (D,) per-channel input variance
        self._ema_out:  torch.Tensor = None   # (D,) unit vector output direction
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_var  = None
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
                xf = x.detach().flatten(0,1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_var  = xf.var(0, unbiased=False).clamp(min=self.eps_var).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Per-channel z-score and RMS aggregation ────────────────────────
        with torch.no_grad():
            mu_  = self._ema_mean.view(1, 1, D)
            std_ = self._ema_var.clamp(min=self.eps_var).sqrt().view(1, 1, D)
            z    = (x - mu_) / (std_ + self.eps)      # (B, T, D)
            rms_z = z.pow(2).mean(dim=-1).sqrt()       # (B, T) — RMS z-score
        surprise = torch.tanh(sigma * rms_z)           # (B, T) ∈ (0,1)

        # ── Cosine familiarity gate ────────────────────────────────────────
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surprise)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics ─────────────────────────────────────────
        with torch.no_grad():
            xf   = x.detach().flatten(0,1)
            xm   = xf.mean(0)
            xvar = (xf - xm.unsqueeze(0)).pow(2).mean(0)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xm
            self._ema_var  = (d_val * self._ema_var  + (1-d_val) * xvar).clamp(min=self.eps_var)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_out  = d_val * self._ema_out   + (1-d_val) * F.normalize(om, dim=0)

        return output
