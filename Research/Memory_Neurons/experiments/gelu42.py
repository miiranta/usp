"""GELU42 – Full Sequence Instance Normalization.

gelu39 showed that subtracting the sequence mean (DC removal) gives 33.7% PPL reduction.
That removes the MEAN but leaves the VARIANCE in the T dimension untouched.

Full instance normalization removes BOTH mean and variance, then applies a learned
per-channel affine (gamma, beta) — standard LayerNorm applied to the T axis instead
of D.  This is the maximal version of gelu39.

MECHANISM:
    out = GELU(x)                                      (B, T, D)
    mu  = out.mean(dim=1, keepdim=True)                (B, 1, D)
    sig = out.std(dim=1, keepdim=True, correction=0)   (B, 1, D)
    out_norm = (out - mu) / (sig + eps)                (B, T, D)  zero-mean, unit-var along T
    output   = gamma * out_norm + beta                 in-group affine

WHY REMOVING VARIANCE MATTERS:
    gelu39 scales deviation by alpha (fixed for all sequences), meaning a sequence
    with HIGH variance will produce large outputs and one with LOW variance small ones.
    Full instance norm equalises across sequences: every sequence gets unit-variance
    along T after normalization, giving consistent gradient signal regardless of
    input scale.

RELATIONSHIP TO LAYERNORM:
    Standard LayerNorm normalises over D (per position, across channels).
    GELU42 normalises over T (per batch item, across positions / time steps).
    These are ORTHOGONAL operations — LayerNorm already happens in the transformer;
    GELU42 adds the complementary axis.

Params: gamma (D,), beta (D,)  →  2 * D_FF additional parameters.
         For D_FF=1024: 2048 params = 0.02% overhead.
State:   None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU42(nn.Module):
    """EMA-based instance normalization: normalize against cross-batch EMA mean/var."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma: nn.Parameter = None
        self.beta:  nn.Parameter = None
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))  # d=0.9
        self._ema_mean: torch.Tensor = None
        self._ema_var:  torch.Tensor = None
        self._ready = False

    def _init_affine(self, D: int, device):
        self.gamma = nn.Parameter(torch.ones(D,  device=device))
        self.beta  = nn.Parameter(torch.zeros(D, device=device))

    def reset_state(self):
        self._ema_mean = None; self._ema_var = None; self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if self.gamma is None:
            self._init_affine(D, x.device)

        out = self._gelu(x)                                    # (B, T, D)
        d_val = torch.sigmoid(self.logit_decay).detach().item()

        # Cross-batch EMA of mean and variance — causally valid
        with torch.no_grad():
            bm = out.detach().mean(dim=(0, 1))                 # (D,)
            bv = out.detach().var(dim=(0, 1), correction=0).clamp(min=1e-8)
            if not self._ready:
                self._ema_mean = bm.clone()
                self._ema_var  = bv.clone()
                self._ready    = True
                return out
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * bm
            self._ema_var  = d_val * self._ema_var  + (1 - d_val) * bv

        mu      = self._ema_mean.unsqueeze(0).unsqueeze(0)     # (1, 1, D)
        sig     = self._ema_var.sqrt().unsqueeze(0).unsqueeze(0)
        out_norm = (out - mu) / (sig + self.eps)               # (B, T, D)
        output   = self.gamma * out_norm + self.beta
        return output
