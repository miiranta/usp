"""GELU54 – Ring Buffer Episodic Recall Gate.

THE PROBLEM WITH EMA:
    All EMA-based experiments collapse the entire training history into a single
    vector via exponential decay. This loses EPISODE IDENTITY:

    Sequence A: "the bank of the river"     → mean activation m_A
    Sequence B: "the bank issued a loan"    → mean activation m_B
    EMA after both: 0.9*m_A + 0.1*m_B      → blended blur, loses m_B distinction

    When sequence C arrives and is similar to m_A, the EMA incorrectly flags m_B
    features as familiar. This is a fundamental limitation of single-prototype memory.

GELU54: FIXED-SIZE EPISODIC BUFFER:
    Maintain a ring buffer of the N most recent distinct activation vectors
    (one per batch step). For each new token:

        out     = GELU(x)                              (B, T, D)
        m_curr  = mean_BT(out)                         (D,)  current batch mean

        # Find closest episode in buffer (cosine similarity):
        sims    = cosine(m_curr, buffer[i])  ∀ i      (N,)
        max_sim = max(sims)                            scalar
        novelty = exp(-τ * max_sim)                    scalar

        # Gate output by per-token similarity to nearest episode:
        tok_sim = cosine(out, buffer[nearest])         (B, T)
        gate    = (1 - α) + α * exp(-τ * tok_sim)     (B, T, 1)
        output  = out * gate

    Buffer update: write m_curr into the next ring buffer slot (FIFO).

WHY BETTER THAN EMA:
    - The buffer preserves N SEPARATE episodes, not their average.
    - "I saw a bank-river pattern just 5 steps ago" is detected exactly.
    - Buffer diversity = covers multiple semantic contexts simultaneously.
    - At buffer fill time (N entries), older episodes expire naturally (FIFO).

BUFFER SIZE vs MEMORY:
    N=32, D=1024: 32 * 1024 * 4 bytes = 128KB per layer — negligible.
    N=32 steps × 32 batch × 64 seq_len = 66K tokens of effective episodic memory.

CAUSAL GUARANTEE:
    The buffer entry m_curr is computed from the CURRENT batch mean and stored
    AFTER the forward pass. The buffer used during forward contains only PAST batches.
    No future token information is ever retrievable.

Params: log_tau, log_blend, logit_decay (for buffer → EMA fallback warmup) = 3 scalars.
State:  ring buffer (N, D), write pointer (int).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU54(nn.Module):
    """Ring buffer episodic recall gate — exact multi-episode memory."""

    def __init__(self, buffer_size: int = 32, ema_decay: float = 0.95):
        super().__init__()
        self._N     = buffer_size
        self._buf:  torch.Tensor = None   # (N, D)
        self._mask: torch.Tensor = None   # (N,) bool — slot filled?
        self._ptr   = 0
        self._ready = False

        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))   # suppression sharpness
        self.log_blend   = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # α ≈ 0.3

    def reset_state(self):
        self._buf   = None
        self._mask  = None
        self._ptr   = 0
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
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)

        out = self._gelu(x)   # (B, T, D)

        # Batch mean of GELU output (for buffer update)
        m_curr = out.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Initialise buffer on first call ───────────────────────────────────
        if not self._ready:
            self._buf  = torch.zeros(self._N, D, device=x.device, dtype=out.dtype)
            self._mask = torch.zeros(self._N, dtype=torch.bool, device=x.device)
            self._buf[0] = F.normalize(m_curr, dim=0)
            self._mask[0] = True
            self._ptr  = 1
            self._ready = True
            return out   # warm-up

        # ── Find nearest episode in buffer ────────────────────────────────────
        filled     = self._mask                                  # (N,) bool
        n_filled   = filled.sum().item()

        m_norm     = F.normalize(m_curr.unsqueeze(0), dim=-1)   # (1, D)
        buf_norm   = F.normalize(self._buf, dim=-1)             # (N, D)

        sims_all   = (buf_norm * m_norm).sum(-1)                # (N,) cosine

        # Use only filled slots
        sims_filled = sims_all.masked_fill(~filled, -1.0)       # unfilled → -1
        nearest_idx = sims_filled.argmax()
        max_sim     = sims_filled[nearest_idx]                  # scalar, detached from grad

        # ── Per-token similarity to nearest episode ───────────────────────────
        nearest_vec = self._buf[nearest_idx].detach()           # (D,)
        out_norm    = F.normalize(out, dim=-1)                  # (B, T, D)
        nv_norm     = F.normalize(nearest_vec.view(1, 1, D), dim=-1)
        tok_sim     = (out_norm * nv_norm).sum(-1)              # (B, T)

        novelty     = torch.exp(-tau * tok_sim)                 # (B, T) ∈ (0, 1]
        gate        = (1.0 - alpha) + alpha * novelty           # (B, T)
        output      = out * gate.unsqueeze(-1)                  # (B, T, D)

        # ── Update ring buffer (no grad) ─────────────────────────────────────
        with torch.no_grad():
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
