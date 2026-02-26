"""GELU46 – Self-Similarity Weighted Sequence Contrast.

gelu39 subtracts the UNIFORM mean of the sequence:
    mu[t] = (1/T) * sum_s(GELU(x)[s])      — equal weight to all positions

This is crude: position t gets contrasted against everything, even positions with
completely different content.

GELU46 subtracts a SIMILARITY-WEIGHTED mean: position t is contrasted primarily
against positions that are MOST SIMILAR to it.  Positions with high cosine
similarity to t contribute MORE to t's background estimate.

    out    = GELU(x)                                   (B, T, D)
    S      = cosine_sim(out, out^T)                    (B, T, T)  pairwise cosines
    W      = softmax(beta * S, dim=-1)                 (B, T, T)  similarity weights
    mu_sim = W @ out                                   (B, T, D)  sim-weighted background
    dev    = out - mu_sim                              (B, T, D)
    output = out + alpha * dev

WHAT THIS COMPUTES:
    For each position t, the background mu_sim[t] is a blend of all other positions
    weighted by how similar they are to t.  Very similar positions get high weight.
    
    The deviation from this background is t's "uniqueness relative to its contextual
    type" — not just "uniqueness relative to everything".

    Example: if "the bank" appears 3x in a 64-token window, each "bank" gets
    contrasted mainly against the other "bank" occurrences → strong suppression
    of the repeated pattern.  A unique word gets contrasted against itself (low
    similarity to others) → mild suppression, representation preserved.

RELATIONSHIP TO SELF-ATTENTION:
    W @ out is exactly one step of single-head self-attention on the GELU OUTPUT,
    without learned Q/K/V projections — using identity projections instead.
    This is a 0-learned-parameter sparse information routing that is automatically
    tuned by the model through alpha and beta.
    
    This smuggles a free attention-like operation inside the FF activation.

Params: log_beta_raw (1 scalar, temperature), log_alpha_raw (1 scalar, contrast strength).
State:  None.
Compute: O(T²·D) per layer — for T=64, D=1024: 4M ops per layer, negligible vs attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU46(nn.Module):
    """Similarity-weighted sequence contrast: contrast each position against its
    most contextually similar neighbors in the current sequence."""

    def __init__(self):
        super().__init__()
        # Temperature beta: higher = more peaked similarity weights
        # Init beta ≈ 1 (softplus(0) ≈ 0.69 → + 0.5 = 1.19)
        self.log_beta_raw  = nn.Parameter(torch.tensor(0.0))
        # Contrast strength alpha, init ≈ 0.3
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))

    def reset_state(self):
        pass  # stateless

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
        beta  = F.softplus(self.log_beta_raw).clamp(max=5.0) + 0.5   # temperature in [0.5, 5.5]
        alpha = F.softplus(self.log_alpha_raw)                         # contrast strength ≥ 0

        out = self._gelu(x)                             # (B, T, D)

        # Compute similarity weights without backprop through normalize
        # (avoids gradient explosion via 1/||v|| in very-small-norm vectors)
        with torch.no_grad():
            out_norm = F.normalize(out.detach(), dim=-1, eps=1e-6)     # (B, T, D)
            S = torch.bmm(out_norm, out_norm.transpose(1, 2))          # (B, T, T)
            S = S / math.sqrt(D)                                        # scale like attention

            # CAUSAL mask: position t can only attend to positions s <= t (past + self)
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )  # True for future positions s > t
            S = S.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

            # Softmax over causally-valid positions
            W = torch.softmax(beta * S, dim=-1)             # (B, T, T)
            W = torch.nan_to_num(W, nan=0.0)               # safety: 0/0 -> 0
            W[:, 0, 0] = 1.0                                # position 0: attend to self

        # Causal similarity-weighted background (gradients flow through out)
        mu_sim = torch.bmm(W, out)                          # (B, T, D)

        dev    = out - mu_sim
        output = out + alpha * dev
        return output
