"""GELU34 – Multi-Head Familiarity Gate.

MOTIVATION:
gelu28 uses ONE global cosine familiarity signal over the entire D_FF=1024 vector.
This means all 1024 dimensions are treated equally — a token is "familiar" or
"novel" globally.

INSIGHT: The FFN hidden space D_FF=1024 encodes MULTIPLE semantic properties
simultaneously. A token might be familiar in its syntactic subspace but novel in
its semantic subspace. A single cosine score AVERAGES this out, losing information.

GELU34 splits D_FF into H=8 groups of 128 dims each, computes a SEPARATE
EMA and cosine-familiarity score per group, and applies a per-group gate.

MECHANISM:
    out_raw  = GELU(x)                                        (B, T, D)
    D//H chunks, each of size 128:
      out_g  = out_raw[..., g*128:(g+1)*128]                  (B, T, 128)
      sim_g  = cosine(out_g, ema_g)                           (B, T)
      nov_g  = exp(-τ · sim_g)                                (B, T)
      scale_g = (1-α) + α · nov_g                             (B, T)
      out_g_gated = out_g * scale_g.unsqueeze(-1)             (B, T, 128)
    output = concat([out_0_gated, ..., out_7_gated], dim=-1)  (B, T, D)

The EMAs: ema_g ∈ R^128, total of H EMA vectors total = D_FF extra state.
Each group independently discovers what "familiar" means for its subspace.

WHY THIS IS POWERFUL:
  Syntactic group (e.g.) may have HIGH familiarity for common function words
    → suppresses them → makes room for content words
  Semantic group may have LOW familiarity for domain-specific tokens
    → does NOT suppress them
  The token's total gate is a PER-GROUP product, giving richer signal.

Params: logit_decay, H log_tau values, H log_blend values = 1 + 2H = 17 scalars
        (H=8 groups, each with independent τ and α)
State:  H EMA vectors of size D//H each = 1 × D_FF worth of EMA state

Note: logit_decay is SHARED across groups (single EMA update speed).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU34(nn.Module):
    """Multi-head familiarity: per-group cosine EMA gate on GELU output."""

    def __init__(self, ema_decay: float = 0.9, n_heads: int = 8):
        super().__init__()
        self.H = n_heads
        self._ema_groups = None   # will be (H, D//H) once we know D
        self._ready = False

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        # Per-head learnable tau and blend
        self.log_tau   = nn.Parameter(torch.full((n_heads,), math.log(2.0)))
        self.log_blend = nn.Parameter(torch.full((n_heads,), math.log(0.3 / 0.7)))

    def reset_state(self):
        self._ema_groups = None
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
        B, T, D = x.shape
        H = self.H
        assert D % H == 0, f"D={D} not divisible by H={H}"
        G = D // H   # group size

        out_raw = self._gelu(x)   # (B, T, D)
        d = torch.sigmoid(self.logit_decay)

        # --- Initialize EMA groups ---
        with torch.no_grad():
            out_mean = out_raw.detach().mean(dim=(0, 1))   # (D,)
            out_groups_mean = out_mean.reshape(H, G)       # (H, G)

            if not self._ready:
                self._ema_groups = out_groups_mean.clone()
                self._ready = True
            else:
                self._ema_groups = d * self._ema_groups + (1 - d) * out_groups_mean

        # --- Per-group cosine similarity and gating ---
        out_groups = out_raw.reshape(B, T, H, G)           # (B, T, H, G)
        tau   = torch.exp(self.log_tau)                    # (H,)
        alpha = torch.sigmoid(self.log_blend)              # (H,)

        # cosine sim per group: (B, T, H)
        out_flat = out_groups.reshape(B * T, H, G)         # (BT, H, G)
        ema_exp  = self._ema_groups.unsqueeze(0).expand(B * T, -1, -1)  # (BT, H, G)
        
        # Cosine similarity along G dimension
        sim = F.cosine_similarity(out_flat, ema_exp, dim=-1)  # (BT, H)
        sim = sim.reshape(B, T, H)                            # (B, T, H)

        # Per-group gate
        novelty = torch.exp(-tau.unsqueeze(0).unsqueeze(0) * sim)   # (B, T, H)
        scale   = (1.0 - alpha) + alpha * novelty                   # (B, T, H)

        # Apply group gates
        output  = out_groups * scale.unsqueeze(-1)                  # (B, T, H, G)
        output  = output.reshape(B, T, D)

        return output
