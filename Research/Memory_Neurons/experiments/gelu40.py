"""GELU40 – Gradient-Trained Associative Memory Gate (Hopfield-inspired).

MOTIVATION:
EMA-based prototypes (gelu2, gelu12, gelu13, gelu28 etc.) track history but
have NO GRADIENT FLOWING THROUGH THEM — they can't learn to represent the most
USEFUL "familiar patterns" for reducing loss.

This experiment uses a gradient-trained memory bank (K prototypes as nn.Parameter),
updated fully via backprop.  The memory learns to encode the MOST informative
"expected" patterns, so the residual (surprise) is maximally useful for prediction.

INSPIRATION: Modern Hopfield Networks (Ramsauer et al., 2020) use stored patterns
and energy-based retrieval.  Here we use a soft-attention retrieval to find the
nearest stored pattern, then amplify the residual from that retrieval.

MECHANISM:
    out_raw    = GELU(x)                                    (B, T, D)

    # Soft-attention retrieval (Hopfield-style)
    # M: (K, D)  – memory matrix (learned via gradient)
    scores     = out_raw @ M^T / sqrt(D)                   (B, T, K)
    weights    = softmax(beta * scores, dim=-1)             (B, T, K) - soft attention
    retrieved  = weights @ M                               (B, T, D)

    # Amplify prediction error (residual from retrieved memory)
    error  = out_raw - retrieved                           (B, T, D)

    # Output: plain GELU + alpha * error
    output = out_raw + alpha * error

WHY K PROTOTYPES ARE BETTER THAN ONE EMA:
- EMA single-prototype: tracks global average – everything familiar looks the same.
- K gradient-trained prototypes: each prototype specialises on a DISTINCT type
  of familiar pattern (e.g. punctuation, function words, topic words).
  The model learns which combination to retrieve, maximising prediction accuracy.

STABILITY:
  Prototypes are L2-normalised before use (unit sphere), preventing collapse.
  Memory retrieval temperature beta is learned.

Params:  M (K×D), log_beta (1), log_alpha (1)  = K*D + 2  params.
         For K=8, D=1024: 8194 extra params (~0.2% overhead for D_FF=1024).
State:   None (fully gradient-trained, no EMA).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU40(nn.Module):
    """Hopfield-inspired gradient-trained memory: amplify residual from nearest prototype."""

    def __init__(self, n_prototypes: int = 8):
        super().__init__()
        self._K = n_prototypes
        self._D = None  # set on first forward

        # Will be initialised lazily on first forward (don't know D yet)
        self.memory   = None   # (K, D) nn.Parameter
        self.log_beta = nn.Parameter(torch.tensor(0.0))      # retrieval temperature
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))

    def _init_memory(self, D: int, device):
        self._D = D
        # Random unit-sphere initialisation
        raw = torch.randn(self._K, D, device=device)
        raw = F.normalize(raw, dim=-1)
        self.memory = nn.Parameter(raw)

    def reset_state(self):
        pass   # no EMA state — but keep memory (it's learned parameters)

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
        out_raw = self._gelu(x)    # (B, T, D)

        # Lazy init memory when we first see D
        if self.memory is None:
            self._init_memory(D, x.device)

        alpha = F.softplus(self.log_alpha_raw)
        beta  = F.softplus(self.log_beta) + 1.0   # temperature ≥ 1

        # Normalise memory prototypes to unit sphere (prevents collapse)
        M_norm = F.normalize(self.memory, dim=-1)           # (K, D)

        # Normalise outputs for cosine retrieval
        out_norm = F.normalize(out_raw, dim=-1)             # (B, T, D)

        # Soft attention over prototypes
        # scores: (B, T, K)
        scores  = torch.einsum('btd,kd->btk', out_norm, M_norm) * beta
        weights = torch.softmax(scores, dim=-1)             # (B, T, K)

        # Retrieved pattern (in original scale – use unnormalised memory)
        retrieved = torch.einsum('btk,kd->btd', weights, self.memory)  # (B, T, D)

        # Prediction error amplification
        error  = out_raw - retrieved                        # (B, T, D)
        output = out_raw + alpha * error                    # amplify surprise

        return output
