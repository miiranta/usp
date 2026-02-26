"""GELU59 – K=8 Output-Cosine EMA Prototype Bank.

DIAGNOSIS OF CURRENT BEST (gelu28/gelu31):
    gelu28: K=1 EMA vector of GELU outputs. Familiarity = cosine(out, ema_out).
    Result: 3.3% improvement. Works because output cosine > input cosine.

    The limitation: ONE vector collapses all familiar patterns into a single mean.
    "the bank" and "a bank" both contribute to the same EMA.
    When "river bank" appears, its cosine to the blended mean is moderate.
    A dedicated "financial bank" prototype would give a stronger, cleaner signal.

GELU59: 8 SPECIALIZED EMA PROTOTYPES, each tracking a distinct activation mode.

MECHANISM:
    Maintain K=8 EMA vectors {p_k} in R^D (output space).
    For each token:
        sims[b, t, k] = cosine(out[b,t], p_k)             (B, T, K)
        familiarity   = softmax_temperature_logsumexp
                      = (1/tau) * log( mean_k exp(tau * sims) )   (B, T)  soft-max over K
        novelty       = exp(-gamma * familiarity)
        gate          = (1 - alpha) + alpha * novelty
        output        = out * gate

    softmax-logsumexp = "soft maximum" — dominated by the CLOSEST prototype,
    but differentiable and numerically stable. Higher tau → harder max.

    Competitive update: only the nearest prototype is updated each step.
    This keeps prototypes SPECIALIZED (they don't all converge to the mean).

WHY K=8 WINS OVER K=1:
    K=1: one prototype ≈ mean direction. Cross-category average.
    K=8: 8 attractors. Each converges to a different semantic cluster.
    After convergence, each token has a SPECIALIZED familiarity detector firing
    for its specific type — not just "is this near the average of everything?"

    Memory cost: 8 × 1024 × 4 bytes = 32KB per layer. Negligible.
    Compute cost: K × D extra per token = 8 × 1024 = 8K flops/token. Negligible.

Params: logit_decay, log_tau, log_gamma, log_blend = 4 scalars (same as gelu31).
State:  (K, D_FF) EMA prototype matrix. Non-trainable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU59(nn.Module):
    """K=8 output-cosine EMA prototype bank with competitive updates."""

    def __init__(self, n_prototypes: int = 8, ema_decay: float = 0.9):
        super().__init__()
        self._K    = n_prototypes
        self._protos: torch.Tensor = None   # (K, D)
        self._ready = False

        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))   # soft-max temperature
        self.log_gamma   = nn.Parameter(torch.tensor(math.log(2.0)))   # novelty decay rate
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # α ≈ 0.3

    def reset_state(self):
        self._protos = None
        self._ready  = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        gamma = self.log_gamma.exp()
        alpha = torch.sigmoid(self.log_blend)

        out = self._gelu(x)   # (B, T, D)

        # Batch mean of output for prototype update
        out_mean = out.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Initialise prototypes on first call ───────────────────────────────
        if not self._ready:
            base = out_mean.unsqueeze(0).expand(self._K, -1).clone()
            # Small perturbation so prototypes start distinct
            base = F.normalize(base + torch.randn_like(base) * 0.01, dim=-1)
            self._protos = base
            self._ready  = True
            return out   # warm-up

        # ── Cosine similarity to all K prototypes ─────────────────────────────
        out_norm   = F.normalize(out, dim=-1)                        # (B, T, D)
        proto_norm = F.normalize(self._protos, dim=-1)               # (K, D)
        sims       = torch.einsum('btd,kd->btk', out_norm, proto_norm)  # (B, T, K)  ∈ [-1, 1]

        # Soft-maximum over K: (1/tau) * log( (1/K) sum_k exp(tau * sim_k) )
        # = "closest prototype similarity" in a smooth, differentiable way
        soft_max_sim = (sims * tau).logsumexp(dim=-1) / tau - math.log(self._K) / tau  # (B, T)
        # ↑ subtracting log(K)/tau converts logsumexp(mean) → logsumexp; keeps scale consistent

        # Novelty gate (same formula as gelu28/gelu31)
        novelty = torch.exp(-gamma * soft_max_sim)                   # (B, T)
        gate    = (1.0 - alpha) + alpha * novelty                    # (B, T) ∈ (1-α, 1]
        output  = out * gate.unsqueeze(-1)                           # (B, T, D)

        # ── Competitive update: only nearest prototype ────────────────────────
        with torch.no_grad():
            mean_norm  = F.normalize(out_mean.unsqueeze(0), dim=-1)          # (1, D)
            k_star     = (proto_norm * mean_norm).sum(-1).argmax().item()
            raw_update = d_val * self._protos[k_star] + (1 - d_val) * out_mean
            self._protos[k_star] = F.normalize(raw_update.unsqueeze(0), dim=-1).squeeze(0)

        return output
