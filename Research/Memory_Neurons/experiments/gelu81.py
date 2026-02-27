"""GELU81 – K=2 Dual-Prototype Cosine Gate + Input Surprise.

THE WEAKNESS OF A SINGLE-MEAN EMA (gelu71):
    gelu71's cosine gate uses ONE prototype direction: the mean output direction.
    
    But language tokens cluster into distinct SEMANTIC GROUPS:
    - Content words (nouns, verbs): tend to fire high-magnitude, broad activations
    - Function words (the, a, of): tend to fire low-magnitude, sparse activations
    - Punctuation/special tokens: activate specific channels only
    
    A single mean direction = the centroid of ALL these clusters.
    A token that is "familiar within its cluster" but far from the global centroid
    will have LOW cosine to the single mean → NOT suppressed → information leak.
    
    With K=2 prototypes:
    - proto_1 might anchor to content-word patterns
    - proto_2 might anchor to function-word patterns
    
    FAMILIARITY = max cosine to ANY prototype = "familiar to its own cluster."
    Gate = exp(-τ × max_cos) × (1 + w × surprise)
    
    Only tokens that are novel to BOTH clusters pass through fully.

PROTOTYPE UPDATE (soft competitive):
    At each batch, compute assignment: a_k[b] = cos(out[b], p_k)
    Soft-normalize: s_k = softmax([a_1, a_2], dim=0)   (per sample)
    Update: p_k += (1-d) × (s_k × (out_unit - p_k))
    
    The closer prototype gets a larger update (tracks its own cluster).
    Both prototypes drift toward their respective cluster centroids over training.
    
    INITIALIZATION: both prototypes start at the same initial batch mean.
    They diverge naturally as different token patterns are encountered.
    To help divergence: add small random noise to one prototype at init.

EARLY STABILITY:
    For the first few epochs, both prototypes are near the mean direction.
    max_cos ≈ mean cosine to single prototype (≈ gelu71 behavior).
    As prototypes diverge, the gate becomes more selective.
    Gradual warmup → safe for early stopping barrier.

Params: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars (same as gelu71).
State: _proto (2, D) unit vectors; _ema_x (D,) raw for surprise; _ema_x_norm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU81(nn.Module):
    """K=2 dual-prototype cosine gate: familiarity = max cosine to nearest prototype."""

    def __init__(self, ema_decay: float = 0.9, K: int = 2, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.K   = K
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._proto:       torch.Tensor = None   # (K, D) unit vectors
        self._ema_x:       torch.Tensor = None   # (D,) raw input mean
        self._ema_x_norm:  float = 1.0
        self._ready = False

    def reset_state(self):
        self._proto    = None
        self._ema_x    = None
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
                xm = x.detach().flatten(0,1).mean(0)
                om = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
                # Init both prototypes at mean, perturb one slightly for divergence
                noise = torch.randn_like(om) * 0.01
                p0 = F.normalize(om + noise, dim=0)
                p1 = F.normalize(om - noise, dim=0)
                self._proto    = torch.stack([p0, p1], dim=0)    # (K, D)
                self._ema_x    = xm.clone()
                self._ema_x_norm = xm.norm().item() + self.eps
                self._ready    = True
            return out

        # ── Dual-prototype cosine familiarity ──────────────────────────────
        out_n = F.normalize(out.detach(), dim=-1)   # (B, T, D)
        # Cosine to each prototype: (B, T, K)
        cos_k = torch.einsum('btd,kd->btk', out_n, self._proto)  # (B, T, K)
        max_cos, _ = cos_k.max(dim=-1)                             # (B, T)
        gate_cos  = torch.exp(-tau * max_cos.clamp(-1, 1))         # (B, T)

        # ── Input-deviation surprise ────────────────────────────────────────
        ema_x = self._ema_x.view(1, 1, D)
        delta = (x.detach() - ema_x).norm(dim=-1)
        surprise = torch.tanh(sigma * delta / (self._ema_x_norm + self.eps))

        gate   = gate_cos * (1.0 + w * surprise)
        output = out * gate.unsqueeze(-1)

        # ── Update prototypes (soft competitive) ───────────────────────────
        with torch.no_grad():
            # Mean over B*T, assignment-weighted update
            out_unit_flat = out_n.flatten(0, 1)   # (B*T, D)
            cos_flat = cos_k.flatten(0, 1)        # (B*T, K)
            # Soft assignment weights: normalize over K = which prototype owns this sample
            assign = torch.softmax(cos_flat, dim=-1)  # (B*T, K)
            for k in range(self.K):
                w_k = assign[:, k].mean()        # scalar mean assignment weight
                self._proto[k] = d_val * self._proto[k] + (1-d_val) * w_k * out_unit_flat.mean(0)
                self._proto[k] = F.normalize(self._proto[k], dim=0)

            xm = x.detach().flatten(0,1).mean(0)
            self._ema_x    = d_val * self._ema_x + (1-d_val) * xm
            self._ema_x_norm = self._ema_x.norm().item() + self.eps

        return output
