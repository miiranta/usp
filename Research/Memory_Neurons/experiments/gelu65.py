"""GELU65 – Calcium-Dependent Afterhyperpolarization (AHP).

THE BIOLOGICAL MECHANISM:
    When a neuron fires repeatedly, intracellular Ca²⁺ accumulates through
    voltage-gated calcium channels. High [Ca²⁺] activates SK/BK potassium channels,
    producing an afterhyperpolarization (AHP) current that:
      - Reduces the cell's excitability proportional to recent firing history
      - Recovers as Ca²⁺ is pumped out (τ ≈ 100-500ms)
      - Is PER-CHANNEL: each dimensional feature has its own calcium pool

    This is the PUREST "neurotransmitter budget" mechanism:
    fire a lot → Ca builds → can't fire as strongly → Ca recovers → fire again.

IMPLEMENTATION:
    For each position t along the sequence (causal):
        Ca[t+1] = d_ca * Ca[t] + (1 - d_ca) * |out[b, t, d]|   per channel
        gate[t] = 1 / (1 + beta * Ca[t])     ← AHP suppression (high Ca → low gate)

    Then CONTRAST-NORMALIZE the gate (energy-preserving, as in gelu62):
        gate_norm[t] = gate[t] / mean_D(gate[t])   → mean gate = 1.0
        output[t] = out[t] * gate_norm[t]

    After sequence: EMA of final Ca level → next sequence starts partially loaded.
    This gives cross-sequence long-term adaptation for chronically active channels.

KEY ADVANTAGES OVER GELU56:
    gelu56 depletion: r depleted by raw |GELU(x)| → large activations catastrophically deplete.
    GELU65 calcium: Ca tracks a bounded EMA (|out| is soft-bounded by GELU ≥ 0).
                    Gate is 1/(1+β*Ca) — never goes to 0, always recovers smoothly.
    
    No "used.clamp(max=r*0.99)" hack needed — the 1/(1+Ca) form guarantees stability.

    Plus contrast normalization: active channels suppressed RELATIVELY, novel channels
    are amplified — not just a global energy drain.

SELF-REGULATION:
    Channel d firing strongly → Ca[d] rises → gate[d] falls → output[d] falls
    → contributes less to embedding → model learns to use it for rare events only.
    Quiet channel d → Ca[d] ≈ 0 → gate[d] ≈ 1 / mean_gate → amplified relative to busy channels.

Params: logit_beta (AHP strength), log_d_ca_raw (Ca decay rate), logit_decay (cross-seq EMA).
        3 scalars. No D-dimensional parameters.
State:  _ema_ca (D,) cross-sequence calcium EMA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU65(nn.Module):
    """Calcium-AHP gate: per-channel causal calcium buildup → contrast-normalized AHP."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # β: AHP strength. gate = 1/(1+β*Ca). init β≈1.0
        self.logit_beta    = nn.Parameter(torch.tensor(0.0))  # softplus→ ≈0.69, then we use exp
        self.log_beta_raw  = nn.Parameter(torch.tensor(0.0))  # softplus init ≈ 0.69
        # Ca decay: sigmoid → d_ca ≈ 0.9 (Ca decays slowly per step)
        self.logit_d_ca    = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))
        # Cross-sequence EMA decay
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(0.95 / 0.05)))

        self._ema_ca: torch.Tensor = None   # (D,) cross-batch calcium level
        self._ready = False

    def reset_state(self):
        self._ema_ca = None
        self._ready  = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        beta   = F.softplus(self.log_beta_raw)                    # AHP strength ≥ 0
        d_ca   = torch.sigmoid(self.logit_d_ca).detach().item()   # Ca decay ∈ (0,1)
        d_val  = torch.sigmoid(self.logit_decay).detach().item()  # EMA decay

        out = self._gelu(x)   # (B, T, D)

        # ── Init: Ca starts at cross-seq EMA level (or zero on cold start) ────
        if not self._ready:
            with torch.no_grad():
                self._ema_ca = torch.zeros(D, device=x.device, dtype=out.dtype)
                self._ready  = True
            return out   # first call: no calcium yet, full transmission

        # ── Causal calcium accumulation & AHP gate ────────────────────────────
        # Start each sequence with Ca = ema_ca (cross-sequence memory)
        Ca = self._ema_ca.unsqueeze(0).expand(B, D).clone()   # (B, D)
        gate_trace = []

        with torch.no_grad():
            for t in range(T):
                # Gate at position t based on Ca BEFORE this step fires
                gate_t = 1.0 / (1.0 + beta.detach() * Ca)    # (B, D) ∈ (0, 1]
                gate_trace.append(gate_t)

                # Ca update: accumulate firing magnitude, decay toward 0
                firing = out[:, t, :].detach().abs()           # (B, D) ≥ 0
                Ca = d_ca * Ca + (1.0 - d_ca) * firing        # (B, D) EMA

        gates = torch.stack(gate_trace, dim=1)                 # (B, T, D)

        # ── Contrast normalization (energy-preserving) ────────────────────────
        gates_norm = gates / (gates.mean(dim=-1, keepdim=True) + self.eps)  # mean over D = 1
        output = out * gates_norm                              # (B, T, D)

        # ── Update cross-sequence EMA of end-of-sequence Ca ──────────────────
        with torch.no_grad():
            self._ema_ca = d_val * self._ema_ca + (1.0 - d_val) * Ca.mean(0)

        return output
