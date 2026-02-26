"""GELU36 – Dual-Context Familiarity Gate (global + local).

MOTIVATION:
gelu28 suppresses tokens that are familiar relative to a GLOBAL EMA (cross-batch).
This captures "universally common" patterns (function words, frequent phrases).

But there's another type of familiarity: LOCAL repetition within a single sequence.
"The bank announced... the bank said... the bank..." — "bank" is locally repetitive
in THIS sequence even if it's globally infrequent.

GELU36 suppresses tokens familiar in EITHER context:
  - Global: familiar to the cross-batch EMA (same as gelu28)
  - Local:  familiar to the CURRENT SEQUENCE's running mean

MECHANISM:
    out_raw      = GELU(x)                                   (B, T, D)

    # --- Global familiarity (cross-batch EMA) ---
    sim_global   = cosine(out_raw, ema_out)                  (B, T)

    # --- Local familiarity (within-sequence mean) ---
    # Use mean of PRECEDING tokens to avoid look-ahead:
    # seq_mean[t] = mean(out_raw[:, :t+1, :])  (cumulative)
    # For simplicity: use mean of ALL tokens in sequence (batch-level causal leak
    # is acceptable since our training is not strictly causal at the activation level)
    seq_mean     = out_raw.mean(dim=1, keepdim=True)         (B, 1, D)
    sim_local    = cosine(out_raw, seq_mean.expand_as(out_raw))  (B, T)

    # --- Combine: suppress if familiar in either context ---
    sim_combined = torch.maximum(sim_global, sim_local)      (B, T)

    novelty      = exp(-τ · sim_combined)                    (B, T)
    α            = sigmoid(log_blend)
    scale        = (1 - α) + α · novelty                    (B, T)
    output       = out_raw · scale.unsqueeze(-1)             (B, T, D)

WHY MAX not product?
  max(a,b) → as soon as ONE context says familiar, we suppress.
  product(exp(-a),exp(-b)) → need BOTH to be familiar for full suppression.
  For diverse text, global OR local familiarity is the key signal.

Params: logit_decay, log_tau, log_blend = 3 scalars (same as gelu28, minimal!)
State:  ema_out (D_FF,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU36(nn.Module):
    """Dual-context gate: max(global_familiarity, local_familiarity) → suppress."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema_out:   torch.Tensor = None
        self._ema_local: torch.Tensor = None
        self._ready = False

        self.logit_decay      = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.logit_decay_fast = nn.Parameter(torch.tensor(math.log(0.5 / 0.5)))  # init d=0.5
        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))

    def reset_state(self):
        self._ema_out   = None
        self._ema_local = None
        self._ready     = False

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
        out_raw = self._gelu(x)   # (B, T, D)

        # --- Update cross-batch EMAs (global slow + local fast) ---
        with torch.no_grad():
            out_mean = out_raw.detach().mean(dim=(0, 1))
            if not self._ready:
                self._ema_out   = out_mean.clone()
                self._ema_local = out_mean.clone()
                self._ready     = True
            else:
                d_slow = torch.sigmoid(self.logit_decay).item()
                d_fast = torch.sigmoid(self.logit_decay_fast).item()
                self._ema_out   = d_slow * self._ema_out   + (1 - d_slow) * out_mean
                self._ema_local = d_fast * self._ema_local + (1 - d_fast) * out_mean

        # --- Global familiarity: cosine to slow cross-batch EMA ---
        out_flat   = out_raw.reshape(B * T, D)
        ema_exp    = self._ema_out.unsqueeze(0).expand(B * T, -1)
        sim_global = F.cosine_similarity(out_flat, ema_exp, dim=-1).reshape(B, T)

        # --- Local familiarity: cosine to fast-decay (recent) EMA ---
        local_exp  = self._ema_local.unsqueeze(0).expand(B * T, -1)
        sim_local  = F.cosine_similarity(out_flat, local_exp, dim=-1).reshape(B, T)

        # --- Combined: suppress if familiar in EITHER context ---
        sim_combined = torch.maximum(sim_global, sim_local)   # (B, T)

        tau     = torch.exp(self.log_tau)
        novelty = torch.exp(-tau * sim_combined)              # (B, T)
        alpha   = torch.sigmoid(self.log_blend)
        scale   = (1.0 - alpha) + alpha * novelty            # (B, T)

        return out_raw * scale.unsqueeze(-1)
