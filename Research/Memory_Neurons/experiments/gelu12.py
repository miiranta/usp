"""GELU12 – Neuron Memory: Winner-Take-All Prototype Clustering.

Biological intuition: A neuron learns to recognize typical patterns of input
in its context. It maintains a "memory" of K learned prototypes representing
the manifold of normal activation patterns for this neuron.

When a new activation arrives:
  1. Find the CLOSEST prototype (winner-take-all, hard assignment)
  2. Measure distance to that prototype
  3. Small distance → familiar pattern from memory → suppress gently
  4. Large distance → novel pattern not in memory → amplify boldly

This is fundamentally different from EMA (which forgets) or batch-stats
(which are noisy). Prototypes are **learned and persistent**, trained by
backprop to cluster the distribution of activations the network uses.

Mathematical formulation:
    P               = K learned prototypes                    (K, D) 
    p_star, dist    = argmin_k ||x - P_k||, min_k ||x - P_k|| per-token
    novelty         = 1 - exp(-τ · dist)                     (B, T)
    scale           = (1 - α) + α · novelty
    output          = GELU(x * scale)

Why this works:
  • Prototypes learn what's "normal" via the loss — self-consistent
  • Hard assignment (argmax) gives sharper gradients than soft (softmax)
  • Distance-based novelty is interpretable and smooth
  • No EMA decay needed — learned distribution is stable
  • Works across the full activation space, not just small changes

Parameters:
  • K×D prototypes (learned via backprop)
  • log_tau, log_blend (2 additional scalars per layer)
  
With K=8, D=1024, 4 layers: 8×1024×4 = 32,768 extra params (~0.33% of 9.87M).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU12(nn.Module):
    def __init__(self, n_prototypes: int = 8):
        super().__init__()
        self.K = n_prototypes
        self._prototypes: nn.Parameter = None   # lazily initialized

        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

    def reset_state(self):
        pass   # prototypes are persistent learned parameters

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)

        # ── Lazy prototype init ────────────────────────────────────────
        if self._prototypes is None:
            D = x.shape[-1]
            # Initialize from standard normal, then normalize
            P = torch.randn(self.K, D, device=x.device, dtype=x.dtype)
            P = F.normalize(P, dim=-1)
            self._prototypes = nn.Parameter(P)

        # ── Winner-take-all: find nearest prototype for each token ──
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)                           # (B*T, D)
        x_norm = F.normalize(x_flat, dim=-1, eps=1e-8)      # (B*T, D)
        P_norm = F.normalize(self._prototypes, dim=-1, eps=1e-8)  # (K, D)

        # Cosine distance to all prototypes
        sims = torch.mm(x_norm, P_norm.T)                   # (B*T, K) ∈ (-1, 1)
        sims = torch.clamp(sims, -1.0, 1.0)                 # numerical stability
        dists = 1.0 - sims                                  # (B*T, K) ∈ (0, 2)
        min_dist, _ = torch.min(dists, dim=-1)              # (B*T,)
        min_dist = torch.clamp(min_dist, 0.0, 2.0)          # clip to valid range

        # Novelty: 1 - exp(-τ * dist)
        # At dist=0: novelty=0 (familiar)
        # As dist→∞: novelty→1 (novel)
        novelty = 1.0 - torch.exp(-tau * min_dist)          # (B*T,)
        novelty = novelty.view(B, T)                        # (B, T)

        # Smooth gating: familiar (low novelty) → scale≤1; novel (high novelty) → scale≥1
        scale   = (1.0 - alpha) + alpha * novelty.unsqueeze(-1)   # (B, T, 1)
        scale   = torch.clamp(scale, 0.1, 10.0)             # prevent extreme scaling

        return self._gelu(x * scale)
