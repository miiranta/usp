"""GELU70 – Dual-Timescale Cosine Gate (Within-Sequence × Cross-Batch).

THE GAP IN ALL PRIOR EXPERIMENTS:
    gelu28/31 use CROSS-BATCH EMA only → suppresses tokens that are familiar
    relative to the ENTIRE TRAINING HISTORY.
    
    gelu16 used WITHIN-SEQUENCE temporal differences → suppresses tokens 
    similar to the IMMEDIATELY PRECEDING position.
    
    Neither experiment tried COMBINING BOTH TIMESCALES simultaneously.

THE BIOLOGICAL RATIONALE — TWO FORMS OF ADAPTATION:
    1. FAST adaptation (milliseconds → within-sequence):
       Rapid synaptic depression after repetitive firing (gelu56 timescale).
       "I just saw this pattern 3 positions ago, attenuate it."
       
    2. SLOW adaptation (minutes/hours → cross-batch):
       Homeostatic plasticity. Long-term familiarity tracking.
       "This type of input appears in every sentence I process."
    
    Combining both: most discriminative signal.
    A token can be long-term FAMILIAR but locally NOVEL (just re-appeared after absence).
    A token can be long-term NOVEL but locally REPETITIVE (first encounter, but repeated).
    
    TRUE NOVELTY = novel both within-sequence AND relative to long-term history.
    gelu70 requires BOTH conditions for full gate opening.

IMPLEMENTATION:
    ── SLOW axis (same as gelu28): cross-batch EMA ──
    ema_slow[d]: updated after each batch, carries across sequences
    gate_slow[t] = exp(-τ_s * cos(out[t], ema_slow))   ∈ (0,1]
    Familiar long-term → low gate_slow. Novel long-term → high gate_slow.
    
    ── FAST axis: within-sequence causal EMA ──
    ema_fast[b,d]: starts at ema_slow[d] (warm-initialized), updated causally
    ema_fast[b, t+1] = α_f * ema_fast[b, t] + (1-α_f) * out[b, t]
    gate_fast[b, t] = exp(-τ_f * cos(out[b,t], ema_fast[b, t-1]))
    Recently repeated → low gate_fast. Recent absence → high gate_fast.
    
    ── Combined gate ──
    gate[b, t] = gate_slow[b, t] * gate_fast[b, t]   ∈ (0,1]
    output[b, t] = out[b, t] * gate[b, t]
    
    CONTRAST NORMALIZE for energy preservation (from gelu69 insight):
    gate_norm[b, t] = gate[b, t] / mean_{b,t}(gate + eps)
    output = out * gate_norm

SIGNAL ANALYSIS OF THE FOUR CASES:
    fast=1, slow=1 (novel everywhere)       → gate≈1.0, then amplified by normalization
    fast=1, slow=0 (new locally, familiar globally) → medium gate
    fast=0, slow=1 (repeated locally, new overall)  → medium gate  
    fast=0, slow=0 (repeated everywhere)   → gate≈0, suppressed

    Only tokens novel at BOTH timescales get the full amplification boost.
    This is a stricter novelty filter than either alone.

Params: logit_decay_slow, logit_decay_fast, log_tau_slow, log_tau_fast = 4 scalars.
State: _ema_slow (D,) cross-batch; ema_fast rebuilt per-forward from ema_slow.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU70(nn.Module):
    """Dual-timescale cosine gate: within-sequence fast EMA × cross-batch slow EMA,
    with contrast normalization for novelty amplification."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Slow (cross-batch) EMA decay ≈ 0.9
        self.logit_decay_slow = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))
        # Fast (within-seq) EMA decay ≈ 0.7 — reacts quickly to local repetition
        self.logit_decay_fast = nn.Parameter(torch.tensor(math.log(0.7 / 0.3)))
        # Gate sharpness
        self.log_tau_slow = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_tau_fast = nn.Parameter(torch.tensor(math.log(2.0)))

        self._ema_slow: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_slow = None
        self._ready    = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_slow  = torch.sigmoid(self.logit_decay_slow).detach().item()
        d_fast  = torch.sigmoid(self.logit_decay_fast).detach().item()
        tau_s   = self.log_tau_slow.exp()
        tau_f   = self.log_tau_fast.exp()

        out = self._gelu(x)   # (B, T, D)

        # ── Init ─────────────────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._ema_slow = F.normalize(out.detach().flatten(0, 1).mean(0), dim=0)
                self._ready    = True
            return out

        # ── Slow axis: cross-batch (same as gelu28) ───────────────────────────
        ema_slow_n = F.normalize(self._ema_slow, dim=0).view(1, 1, D)   # (1,1,D)
        out_n      = F.normalize(out.detach(), dim=-1)                    # (B,T,D)
        cos_slow   = (out_n * ema_slow_n).sum(-1).clamp(-1, 1)           # (B,T)
        gate_slow  = torch.exp(-tau_s * cos_slow)                         # (B,T)

        # ── Fast axis: causal within-sequence EMA ────────────────────────────
        # Initialize fast EMA per-sample from ema_slow (warm start)
        ema_fast = self._ema_slow.unsqueeze(0).expand(B, D).clone()      # (B, D)
        gate_fast_list = []

        with torch.no_grad():
            for t in range(T):
                ema_fast_n = F.normalize(ema_fast, dim=-1)               # (B, D)
                cos_f = (out_n[:, t, :] * ema_fast_n).sum(-1).clamp(-1, 1)  # (B,)
                gate_fast_list.append(torch.exp(-tau_f.detach() * cos_f))    # (B,)
                # Update ema_fast causally (this step's output goes into next step)
                ema_fast = d_fast * ema_fast + (1.0 - d_fast) * out.detach()[:, t, :]

        gate_fast = torch.stack(gate_fast_list, dim=1)                   # (B,T)

        # ── Combined gate ─────────────────────────────────────────────────────
        gate_raw = gate_slow * gate_fast                                   # (B,T)

        # Contrast normalization (mean over B×T = 1)
        with torch.no_grad():
            gate_mean = gate_raw.mean().clamp(min=self.eps)
        gate_norm = gate_raw / gate_mean                                  # (B,T) mean=1

        output = out * gate_norm.unsqueeze(-1)                            # (B,T,D)

        # ── Update slow EMA ───────────────────────────────────────────────────
        with torch.no_grad():
            bm = F.normalize(out.detach().flatten(0, 1).mean(0), dim=0)
            self._ema_slow = d_slow * self._ema_slow + (1.0 - d_slow) * bm

        return output
