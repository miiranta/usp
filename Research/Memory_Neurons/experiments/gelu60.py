"""GELU60 – Within-Sequence Causal Replay Gate.

THE NEW MEMORY AXIS: the current sequence itself.

ALL prior experiments store memory ACROSS batches (EMA of past batches).
They answer: "is this token similar to the average of all past tokens?"

GELU60 stores the CURRENT SEQUENCE as a causal buffer and answers:
"has this EXACT token appeared earlier in THIS sequence?"

This captures a completely different familiarity signal:
    WITHIN-SEQUENCE REPETITION — "the bank ... the bank" in the same passage.

MECHANISM:
    For each position t, the sequence buffer contains:
        seq_buf[0..t-1, d] = out[0..t-1, d]   (all past activations in this sequence)
    
    Familiarity at t = cosine(out[t], seq_buf[s]) for the MOST SIMILAR past position s:
        sims[t, s] = cosine(out[t], out[s])   for s < t   (causal: only past)
        max_sim[t] = max_{s < t} sims[t, s]               (best match in sequence)
    
    Novelty = exp(-tau * max_sim[t])
    gate    = (1 - alpha) + alpha * novelty
    output  = out * gate

    For position 0 (no past): fall back to cross-batch EMA cosine (like gelu28).

CAUSAL GUARANTEE:
    sims[t, s] only uses s < t → strictly causal within sequence.
    No future information ever accessed.

COMPUTATIONAL COST:
    O(T²·D) per layer per batch. For T=64, D=1024: 64×64×1024 = 4M ops.
    Same order as gelu46's similarity matrix — shown to be fast enough.

DUAL MEMORY: within-sequence + cross-batch EMA
    max_sim[t] comes from within-sequence for t>0 (exact episodic recall)
    max_sim[0] comes from cross-batch EMA (statistical baseline)
    Blend is learned: logit_blend_seq controls how much sequence memory vs EMA.

Params: logit_decay (EMA), log_tau, log_blend, logit_blend_seq = 4 scalars.
State:  ema_out (D,) cross-batch running mean. No within-sequence persistent state.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU60(nn.Module):
    """Within-sequence causal max-similarity gate with cross-batch EMA fallback."""

    def __init__(self, ema_decay: float = 0.9):
        super().__init__()
        self._ema_out: torch.Tensor = None
        self._ready = False

        self.logit_decay     = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau         = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend       = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # gate strength
        # How much to weight sequence memory vs EMA at t>0 (init: 50/50)
        self.logit_blend_seq = nn.Parameter(torch.tensor(0.0))

    def reset_state(self):
        self._ema_out = None
        self._ready   = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        d_val    = torch.sigmoid(self.logit_decay).detach().item()
        tau      = self.log_tau.exp()
        alpha    = torch.sigmoid(self.log_blend)
        w_seq    = torch.sigmoid(self.logit_blend_seq)  # weight on sequence memory

        out = self._gelu(x)   # (B, T, D)

        # ── Initialise EMA on first call ──────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._ema_out = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
                self._ready   = True
            return out

        # ── Cross-batch EMA cosine similarity (baseline) ──────────────────────
        out_norm  = F.normalize(out, dim=-1)                               # (B, T, D)
        ema_norm  = F.normalize(self._ema_out, dim=0).view(1, 1, D)       # (1, 1, D)
        ema_sim   = (out_norm * ema_norm).sum(-1)                          # (B, T) ∈ [-1,1]

        # ── Within-sequence causal max similarity ─────────────────────────────
        # All-pairs cosine (B, T, T) — then take strict-causal max
        # sim_mat[b, t, s] = cosine(out[b,t], out[b,s])
        sim_mat = torch.bmm(out_norm, out_norm.transpose(1, 2))            # (B, T, T)

        # Mask out t=s and s>t (future): only s < t contributes
        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1
        )  # True where s < t  (strict past only)

        # For positions with no past (t=0), max over empty set → use EMA sim
        sim_past = sim_mat.masked_fill(~causal_mask.unsqueeze(0), -2.0)    # (B, T, T)
        seq_sim, _ = sim_past.max(dim=-1)                                  # (B, T)  max over s

        # At t=0, seq_sim = -2 (no past) → replace with EMA sim
        has_past = causal_mask.any(dim=-1).view(1, T)                      # (1, T) bool
        seq_sim  = torch.where(has_past.expand(B, T), seq_sim, ema_sim)    # (B, T)

        # ── Blend sequence and EMA similarity ────────────────────────────────
        # w_seq=1 → pure sequence memory; w_seq=0 → pure EMA (= gelu28)
        familiarity = w_seq * seq_sim + (1.0 - w_seq) * ema_sim           # (B, T)

        # ── Novelty gate ──────────────────────────────────────────────────────
        novelty = torch.exp(-tau * familiarity)
        gate    = (1.0 - alpha) + alpha * novelty
        output  = out * gate.unsqueeze(-1)                                 # (B, T, D)

        # ── Update EMA (no grad) ──────────────────────────────────────────────
        with torch.no_grad():
            bm = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
            self._ema_out = d_val * self._ema_out + (1 - d_val) * bm

        return output
