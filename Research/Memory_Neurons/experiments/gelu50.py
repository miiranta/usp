"""GELU50 – Orthogonal Subspace Suppression.

ALL PREVIOUS EXPERIMENTS use a SCALAR gate applied uniformly to all D channels:
    output = GELU(x) * scalar(familiarity)

This is a blunt instrument: if position t is "familiar", ALL D channels are
suppressed equally, even the ones carrying genuinely novel information.

GELU50 does full VECTOR-SPACE signal separation:

1.  Maintain K EMA prototype vectors  {p_k}  in R^D.
2.  Gram-Schmidt → K orthonormal basis vectors  {e_k}  spanning "familiar subspace".
3.  Decompose each activation:
        x_fam = sum_k  (x · e_k) * e_k        ← component IN familiar subspace
        x_nov = x - x_fam                      ← component ORTHOGONAL to it
4.  Apply learned suppression scale λ to the familiar component only:
        x_sep = x_nov  +  λ * x_fam            λ = sigmoid(logit_lambda) ∈ (0,1)
5.  output = GELU(x_sep)

WHAT THIS ACHIEVES:
    Each token is split into "what the model has seen before" (x_fam) and
    "what is new" (x_nov).  Only x_fam is suppressed; x_nov passes untouched.
    This is the LINEAR ALGEBRAIC ideal of the familiarity-suppression hypothesis.

    gelu2_k1 approximates this with K=1 via a scalar cosine gate.
    GELU50 does it exactly with K=8 prototypes using actual projection.

GS STABILITY:
    Gram-Schmidt is run on the detached EMA prototypes once per forward pass.
    Very cheap: K small (K=8), D=1024 → K·D = 8192 flops per GS step.
    GS degeneration guarded: any basis vector with norm < 1e-4 after subtraction
    is replaced with a random unit vector (prototype hasn't diverged far enough).

PROTOTYPE UPDATE:
    Competitive update: the NEAREST prototype to the current batch mean is updated.
    All others remain frozen each step → stable set of diverse prototypical directions.

Params: logit_lambda (1 scalar) + logit_decay (1 scalar) = 2 total.
State:  K × D EMA prototype matrix (non-trainable).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU50(nn.Module):
    """K-prototype orthogonal subspace suppression."""

    def __init__(self, n_prototypes: int = 8, ema_decay: float = 0.95):
        super().__init__()
        self._K = n_prototypes
        self._protos: torch.Tensor = None   # (K, D)
        self._ready = False

        # λ: how much of the familiar subspace survives (0 = full suppression, 1 = no effect)
        self.logit_lambda = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))   # init λ ≈ 0.3
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))

    def reset_state(self):
        self._protos = None
        self._ready  = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )

    @staticmethod
    def _gram_schmidt(vecs: torch.Tensor) -> torch.Tensor:
        """Orthonormalize rows of vecs (K, D) → orthonormal basis (K, D).
        Degenerate rows replaced with random unit vectors."""
        K, D = vecs.shape
        basis = []
        for k in range(K):
            v = vecs[k].clone()
            for e in basis:
                v = v - (v @ e) * e
            norm = v.norm()
            if norm < 1e-4:
                # degenerate: use random unit vector
                v = F.normalize(torch.randn(D, device=vecs.device, dtype=vecs.dtype), dim=0)
            else:
                v = v / norm
            basis.append(v)
        return torch.stack(basis, dim=0)   # (K, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        d   = torch.sigmoid(self.logit_decay)
        lam = torch.sigmoid(self.logit_lambda)   # ∈ (0, 1)

        x_flat = x.detach().flatten(0, 1)        # (B*T, D)
        x_mean = x_flat.mean(0)                  # (D,)

        # ── Initialise prototypes on first call ───────────────────────────────
        if not self._ready:
            base = x_mean.unsqueeze(0).expand(self._K, -1).clone()
            base = base + torch.randn_like(base) * 1e-2
            self._protos = F.normalize(base, dim=-1)
            self._ready  = True
            return self._gelu(x)   # warm-up: no separation yet

        # ── Competitive prototype update (detached) ──────────────────────────
        d_val = d.detach().item()
        with torch.no_grad():
            x_norm = F.normalize(x_mean.unsqueeze(0), dim=-1)       # (1, D)
            sims   = (F.normalize(self._protos, dim=-1) * x_norm).sum(-1)  # (K,)
            k_star = sims.argmax().item()
            self._protos[k_star] = d_val * self._protos[k_star] + (1 - d_val) * x_mean

        # ── Gram-Schmidt: build orthonormal familiar basis ────────────────────
        basis = self._gram_schmidt(self._protos.detach())            # (K, D)

        # ── Project x onto familiar subspace ─────────────────────────────────
        # coefs[b, t, k] = x[b, t, :] · basis[k, :]
        coefs = torch.einsum('btd,kd->btk', x, basis)              # (B, T, K)
        # x_fam[b, t, :] = sum_k coefs[b,t,k] * basis[k,:]
        x_fam = torch.einsum('btk,kd->btd', coefs, basis)          # (B, T, D)
        x_nov = x - x_fam                                           # (B, T, D)

        # ── Suppress familiar, keep novel ─────────────────────────────────────
        x_sep = x_nov + lam * x_fam                                 # (B, T, D)

        return self._gelu(x_sep)
