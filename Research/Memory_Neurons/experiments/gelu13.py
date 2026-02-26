"""GELU13 – Momentum-Updated Neuron Memory.

GELU12 learns prototypes purely via backprop gradients from the loss.
Weaknesses:
  • Prototypes can drift if loss has blind spots
  • Slow to adapt to actual activation distribution shifts
  • Gradients through prototypes compete with task gradients

Fix: Momentum update. Like momentum encoders in contrastive learning (MoCo),
update prototypes with a slow EMA of cluster centers computed in each batch.

For each batch:
  1. Assign each token to nearest prototype (hard assignment)
  2. Compute cluster center for each prototype (tokens assigned to it)
  3. Update prototypes: P' = β·P + (1-β)·centroid
     β ≈ 0.999 (very slow update) keeps prototypes stable
  4. Compute novelty based on distance to updated prototypes
  5. Backprop only through novelty → gating, not through prototype updates

This decouples two roles:
  • Backprop: learns task-relevant activation transformations
  • Momentum: learns the actual distribution of activations

Properties:
  • Prototypes automatically track what "normal" means for this network
  • More stable than pure EMA (uses actual cluster centers, not just moving mean)
  • More robust than pure backprop (not competed by task gradients)
  • Interpreted as "neuron memory" refreshed by experience

Parameters:
  • K×D prototypes (updated via momentum, not gradients)
  • log_tau, log_blend, logit_momentum (3 scalars per layer)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU13(nn.Module):
    def __init__(self, n_prototypes: int = 8, momentum: float = 0.999):
        super().__init__()
        self.K = n_prototypes
        self.momentum = momentum

        self._prototypes: torch.Tensor = None   # (K, D) not a Parameter — no gradients
        self._ready = False

        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

    def reset_state(self):
        self._prototypes = None
        self._ready = False

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

        B, T, D = x.shape
        x_flat = x.detach().flatten(0, -2)                  # (B*T, D) detached for momentum update

        # ── Lazy init ──────────────────────────────────────────────────
        if self._prototypes is None:
            P = torch.randn(self.K, D, device=x.device, dtype=x.dtype)
            P = F.normalize(P, dim=-1)
            self._prototypes = P
            self._ready = True
            return self._gelu(x)

        if self._prototypes.device != x.device:
            self._prototypes = self._prototypes.to(x.device)

        P_norm = F.normalize(self._prototypes, dim=-1)      # (K, D)

        # ── Hard assignment: nearest prototype for each token ──
        x_flat_norm = F.normalize(x_flat, dim=-1, eps=1e-8)  # (B*T, D) normalise for cosine
        sims = torch.mm(x_flat_norm, P_norm.T)               # (B*T, K) true cosine ∈ (-1,1)
        sims = torch.clamp(sims, -1.0, 1.0)
        assignments = torch.argmax(sims, dim=-1)             # (B*T,)

        # ── Compute cluster centroids ──────────────────────────────────
        centroids = []
        for k in range(self.K):
            mask = (assignments == k)
            if mask.sum() > 0:
                centroid = x_flat[mask].mean(0)             # (D,)
                centroids.append(centroid)
            else:
                # No tokens for this prototype; keep it
                centroids.append(self._prototypes[k])

        new_P = torch.stack(centroids, dim=0)               # (K, D)
        new_P = F.normalize(new_P, dim=-1)

        # ── Momentum update: slow update with new centroids ──
        self._prototypes = (self.momentum * self._prototypes +
                           (1.0 - self.momentum) * new_P)

        # ── Novelty based on distance to (updated) prototypes ──
        P_norm = F.normalize(self._prototypes, dim=-1, eps=1e-8)
        sims2 = torch.mm(x_flat_norm, P_norm.T)             # (B*T, K) cosine
        sims2 = torch.clamp(sims2, -1.0, 1.0)
        dists = 1.0 - sims2.max(dim=-1)[0]                 # (B*T,) ∈ [0, 2]
        dists = torch.clamp(dists, 0.0, 2.0)

        novelty = 1.0 - torch.exp(-tau * dists)             # (B*T,) ∈ [0, 1]
        novelty = novelty.view(B, T)                        # (B, T)

        scale = (1.0 - alpha) + alpha * novelty.unsqueeze(-1)  # (B, T, 1)
        scale = torch.clamp(scale, 0.1, 10.0)

        return self._gelu(x * scale)
