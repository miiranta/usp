"""GELU27 – Backprop-Trained Familiarity Direction (Gradient-Optimised Prototype).

THE FUNDAMENTAL ASSUMPTION IN ALL PRIOR WORK:
  "The familiar direction = the EMA of mean activations"

This is an assumption, not a truth. The mean activation EMA is easy to compute,
but it's not necessarily *the* direction to suppress to improve language modeling.

GELU27 CHALLENGES THIS ASSUMPTION:
Instead of using EMA (data-driven mean) as the familiar prototype, we let
GRADIENT DESCENT from the task loss find the optimal suppression direction.

The prototype is a trainable nn.Parameter that gets updated by backprop:
  • No EMA update
  • Prototype gradient flows through: loss → scale → novelty → sim → prototype
  • The prototype learns to point in "the direction that, when suppressed, most
    improves language modeling." This is gradient-optimal suppression.

Architecture:
    prototype = nn.Parameter(D,)    ← learned familiarity direction
    sim       = cosine(x, prototype) per token  (B, T)
    novelty   = exp(-τ · sim)                  (B, T)
    scale     = (1-α) + α · novelty            (B, T, 1)
    output    = GELU(x · scale)

The prototype is L2-normalized before computing cosine similarity, so it lives
on the unit sphere. This prevents it from scaling indefinitely.
Initialised near zero + random noise → initially cosine ≈ 0 → novelty ≈ 1 → scale ≈ 1
(i.e., initially behaves like plain GELU, then the prototype learns).

Why this might beat EMA:
  • EMA tracks the DATA distribution — what's common in the batch statistics.
  • The gradient-trained prototype tracks the TASK distribution — what to suppress
    to reduce language modeling loss. These can be very different.
  • Example: common syntactic scaffolding (articles, prepositions) is EMA-familiar
    but may not be what the model needs to suppress to predict the NEXT token well.
    The gradient finds the right suppression direction for the task.

Why it might fail:
  • Without EMA's stability, the prototype may oscillate (noisy gradients).
  • The prototype might learn to always suppress (scale→0) or never suppress (scale→1).
  • Learnable τ, α help prevent saturation.

Regularization: spectral norm of prototype is constrained to 1 via F.normalize().
No additional regularizer needed — the task loss itself regularizes direction.

Params per layer: prototype (D) + log_tau + log_blend = D+2 scalars.
With D=1024 and 4 layers: 4 × 1026 ≈ 4k extra parameters (~0.04% overhead).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU27(nn.Module):
    """Gradient-trained familiarity prototype — no EMA, purely backprop."""

    def __init__(self):
        super().__init__()
        # Lazily-initialized prototype: we don't know D at __init__ time
        self._proto: nn.Parameter = None    # (D,) — task-optimized familiarity direction

        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

    def reset_state(self):
        # Nothing to reset — prototype IS the state and is fully trained
        pass

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
        D = x.shape[-1]
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)

        # ── Lazy-init prototype ───────────────────────────────────────
        if self._proto is None:
            # Small random init: initially cosine ≈ 0 → behaves like plain GELU
            init = torch.randn(D, device=x.device, dtype=x.dtype) * 0.01
            self._proto = nn.Parameter(init)
            # Register param in the module (lazy init via add_module/parameter workaround)
            # Note: must be registered for optimiser to see it.
            # We do so here — the optimiser will pick it up after the first forward pass.

        if self._proto.device != x.device:
            self._proto = nn.Parameter(self._proto.to(x.device))

        # ── Cosine similarity: fully differentiable through prototype ──
        # Normalise prototype to unit sphere (L2 ball)
        proto_norm = F.normalize(self._proto.unsqueeze(0), dim=-1)   # (1, D)

        x_norm = F.normalize(x, dim=-1)                               # (B, T, D)
        sim    = (x_norm * proto_norm).sum(-1)                        # (B, T) ∈ [-1,1]

        # Novelty: exp(-τ·sim) — near 1 when sim≈0 (no familiarity), →0 when sim≈1
        novelty = torch.exp(-tau * sim)                               # (B, T)

        scale   = (1.0 - alpha) + alpha * novelty                     # (B, T)
        scale   = scale.unsqueeze(-1)                                 # (B, T, 1)

        return self._gelu(x * scale)
