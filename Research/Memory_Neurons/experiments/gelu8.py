"""GELU8 – Gradient-trained familiarity bank.

The core weakness of GELU2: EMA updates bypass backprop entirely, so prototypes
converge to the running mean of activations regardless of whether suppressing
them actually helps. The model can't learn *which* patterns are truly "familiar
noise" vs "familiar signal worth keeping".

Fix: make prototypes learnable parameters trained end-to-end.

The loss naturally teaches the bank:
  - If suppressing pattern X helps predictions → X gets encoded in a prototype
  - If suppressing pattern X hurts predictions → X stays novel (high novelty score)

Architecture:
    P          = K learnable prototype vectors      (K, D)  ← trained by gradient
    sim        = cosine(x_token, P)                (B, T, K)
    weights    = softmax(sim * τ_assign)           (B, T, K)  soft assignment
    familiarity = (weights * sim).sum(-1)          (B, T)    ∈ (-1, 1)
    novelty    = exp(-τ * familiarity.clamp(min=0)) (B, T)   ∈ (0, 1]
    scale      = (1 - α) + α * novelty            (B, T, 1)
    output     = GELU(x * scale)

Prototypes are lazily initialized from the first batch (kaiming-like unit sphere)
and then trained by backprop — self-consistent: they converge to represent exactly
the patterns whose suppression minimises the loss.

Extra parameters vs control: K×D per layer.
With K=4, D=1024, 4 layers: 4×1024×4 = 16,384 extra (~0.17% of 9.87M).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU8(nn.Module):
    def __init__(self, n_prototypes: int = 4):
        super().__init__()
        self.K = n_prototypes

        # Lazily set in first forward; __setattr__ auto-registers as Parameter
        self._prototypes: nn.Parameter = None

        self.log_tau    = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend  = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        self.log_t_asgn = nn.Parameter(torch.tensor(math.log(5.0)))  # softmax sharpness

    def reset_state(self):
        pass   # no EMA state; prototypes persist across calls as Parameters

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
        tau    = self.log_tau.exp()
        alpha  = torch.sigmoid(self.log_blend)
        t_asgn = self.log_t_asgn.exp()

        # ── Lazy prototype init ────────────────────────────────────────
        if self._prototypes is None:
            D = x.shape[-1]
            P = torch.randn(self.K, D, device=x.device, dtype=x.dtype)
            P = F.normalize(P, dim=-1)
            # Register as Parameter via __setattr__ so it joins model.parameters()
            self._prototypes = nn.Parameter(P)

        # ── Soft familiarity ──────────────────────────────────────────
        x_norm = F.normalize(x, dim=-1)                                  # (B, T, D)
        p_norm = F.normalize(self._prototypes, dim=-1)                   # (K, D)
        sim    = torch.einsum('btd,kd->btk', x_norm, p_norm)            # (B, T, K)

        # Soft assignment: concentrate weight on nearest prototype
        weights    = torch.softmax(sim * t_asgn, dim=-1)                 # (B, T, K)
        familiarity = (weights * sim).sum(-1)                            # (B, T) ∈ (-1,1)

        # Suppress only tokens that are positively similar to a prototype
        novelty = torch.exp(-tau * familiarity.clamp(min=0.0))           # (B, T)
        scale   = (1.0 - alpha) + alpha * novelty.unsqueeze(-1)         # (B, T, 1)

        return self._gelu(x * scale)
