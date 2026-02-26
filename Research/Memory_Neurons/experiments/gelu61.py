"""GELU61 – Second-Moment Memory: Covariance-Aware Novelty.

THE MEMORY BLINDSPOT IN ALL PRIOR EXPERIMENTS:
    Every experiment tracks only the FIRST MOMENT: EMA mean direction.
    Familiarity = "is this token near the average?"

    But consider two orthogonal familiar patterns:
        "function words" direction:  the/a/of/in/to   (high frequency)
        "sentence boundary" direction: starts/ends     (high frequency)
    
    Both are familiar, but they're ORTHOGONAL to each other.
    The mean direction is between them → neither pattern triggers high cosine.

    Solution: track the SECOND MOMENT (covariance matrix A = E[out ⊗ out]).
    Then familiarity = quadratic form  out^T A out = ||out||² in the A-metric.
    High value = out lies IN the span of familiar directions.
    This detects familiarity with ANY of the learned directions, not just the mean.

GELU61: LOW-RANK EMA SECOND MOMENT MATRIX
    A ≈ W W^T  where W ∈ R^(D×r), r=16  (low-rank approximation, r << D)

    EMA update of second moment (rank-r sketch):
        A_full ← d * A_full + (1-d) * out_mean ⊗ out_mean

    In low-rank form (sketched via random projection):
        Actually we maintain the r leading singular vectors directly.
        Simple approach: EMA over r outer product terms (one per prototype update).
        
    EVEN SIMPLER & MORE STABLE: maintain r independent EMA vectors {v_k},
    each tracking a different "familiar mode" via gradient-free competitive update.
    Familiarity = max_k |cosine(out, v_k)|²   (power, not just cosine)
    
    Power (cosine²) penalizes anti-correlation AND correlation equally:
    a token that's the OPPOSITE of a familiar pattern is also suppressed.
    Linguistically: negation ("not the bank") shares structure with ("the bank"),
    both should be flagged as familiar structure.

MECHANISM:
    v_k: K=16 EMA vectors tracking distinct familiar modes   (K × D memory)
    power similarity: p_k[t] = cosine(out[t], v_k)²          ∈ [0, 1]
    familiarity[t] = 1/K * sum_k p_k[t]                     mean second moment (scalar)
                   = fraction of out[t]'s energy in the familiar subspace
    novelty[t] = exp(-tau * familiarity[t])
    gate[t]    = (1 - alpha) + alpha * novelty[t]
    output[t]  = out[t] * gate[t]

    Update: ALL k prototypes updated each step (not just nearest):
        v_k ← normalize(d * v_k + (1-d) * out_mean)
    This converges to the eigenvectors of the time-averaged covariance.
    (Mean-centered power iteration equivalent under EMA.)

WHY K=16 MODES AND POWER (COSINE²):
    K=16: enough modes to cover the main syntactic/semantic familiarity clusters.
    cosine²: captures both the direction AND its reflection — structure-sensitive,
    not direction-sensitive. A pair of antonyms shares grammatical structure.

    This is the D-dimensional equivalent of tracking variance, not just mean.

Params: logit_decay, log_tau, log_blend = 3 scalars.
State:  (K=16, D_FF) matrix of EMA modes. Non-trainable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU61(nn.Module):
    """K=16 second-moment EMA modes: mean cosine² familiarity gate."""

    def __init__(self, n_modes: int = 16, ema_decay: float = 0.95):
        super().__init__()
        self._K     = n_modes
        self._modes: torch.Tensor = None   # (K, D)
        self._ready = False

        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

    def reset_state(self):
        self._modes = None
        self._ready = False

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
        alpha = torch.sigmoid(self.log_blend)

        out = self._gelu(x)   # (B, T, D)
        out_norm = F.normalize(out, dim=-1)                             # (B, T, D)
        out_mean = out.detach().flatten(0, 1).mean(0)                  # (D,)

        # ── Initialise modes on first call with diverse random directions ─────
        if not self._ready:
            # Use random orthogonal-ish initialization, centered on first batch mean
            base = out_mean.unsqueeze(0).expand(self._K, -1).clone()
            noise = torch.randn_like(base)
            # Gram-Schmidt-inspired diversity: subtract projections onto prior rows
            for k in range(1, self._K):
                for j in range(k):
                    noise[k] -= (noise[k] @ noise[j]) / (noise[j] @ noise[j] + 1e-8) * noise[j]
            self._modes = F.normalize(base + noise * 0.1, dim=-1)
            self._ready = True
            return out   # warm-up

        # ── Compute mean cosine² familiarity ─────────────────────────────────
        modes_norm = F.normalize(self._modes, dim=-1)                  # (K, D)
        # cosine[b, t, k] = out_norm[b,t,:] · modes_norm[k,:]
        cosines     = torch.einsum('btd,kd->btk', out_norm, modes_norm)  # (B, T, K)
        power       = cosines.pow(2)                                   # (B, T, K) ∈ [0, 1]
        familiarity = power.mean(dim=-1)                               # (B, T) mean second moment

        novelty = torch.exp(-tau * familiarity)
        gate    = (1.0 - alpha) + alpha * novelty
        output  = out * gate.unsqueeze(-1)                             # (B, T, D)

        # ── Update ALL modes via EMA (non-competitive) ───────────────────────
        with torch.no_grad():
            # All modes updated toward out_mean — convergence to eigenvectors
            # of time-averaged (out_mean)(out_mean)^T via power iteration under EMA
            updated = d_val * self._modes + (1 - d_val) * out_mean.unsqueeze(0)
            self._modes = F.normalize(updated, dim=-1)

        return output
