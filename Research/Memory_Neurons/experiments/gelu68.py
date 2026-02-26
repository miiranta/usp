"""GELU68 – Familiarity-Sensitive Vesicle Depletion.

THE PROBLEM WITH GELU56:
    Depletion rate = U * r * |GELU(x)|   — proportional to raw firing MAGNITUDE.
    
    BUT: a large activation can be either FAMILIAR (common pattern, should deplete)
    or NOVEL (rare pattern, should NOT deplete — we want it to pass through).
    
    gelu56 depletes both equally, wasting vesicles on novel signals.

THE BIOLOGICAL REALITY:
    Synaptic vesicle release is MODULATED by presynaptic inputs.
    When familiar patterns activate the presynaptic terminal, neuromodulators
    (e.g., endocannabinoids from the postsynaptic cell via retrograde signaling)
    REDUCE neurotransmitter release specifically for those inputs.
    
    Novel patterns → no retrograde signal → full vesicle release.
    Familiar patterns → retrograde inhibition → reduced vesicle release.

GELU68: FAMILIARITY-GATED DEPLETION
    familiarity[t] = cosine(out[t], ema_out)    ∈ (-1, 1)   ← directional familiarity EMA
    depletion_rate = U * r * familiarity_pos    ← only deplete when FAMILIAR
    recovery_rate  = (1 - r) * (1 - exp(-1/τ)) ← standard recovery
    
    where familiarity_pos = relu(familiarity)   (only positive = truly familiar)
    
    THEN: contrast-normalize the gate (energy-preserving, from GELU62).

WHY THIS BEATS GELU56:
    Novel token (low cosine to EMA) → familiarity ≈ 0 → depletion ≈ 0 → r stays full
    → full vesicle release → full transmission. Novel signals preserved.
    
    Familiar token (high cosine to EMA) → familiarity ≈ 1 → depletion ∝ U*r
    → r depleted → suppressed output. Familiar patterns suppressed.
    
    gelu56: depletes on |out| (magnitude-based) — wrong metric
    GELU68: depletes on familiarity (direction-based) — correct biological metric
    
    This fuses two proven mechanisms:
    - EMA cosine familiarity (the signal that made gelu28/31 work)
    - Vesicle depletion dynamics (temporal sequence memory from gelu56)

DUAL MEMORY:
    ema_out: tracks "what's usually produced" (familiar baseline)
    r[b,t,d]: tracks "which channels have been depleted THIS sequence"
    
    Cross-sequence: ema_resource carries depletion level across batches.
    Long-familiar channels start partially depleted even at sequence start.

CAUSAL GUARANTEE:
    familiarity[t] = cosine(out[t], ema_from_past_batches) — past batches only.
    r[t] updated from depletion at t-1 — strictly causal.
    No future information used.

Params: logit_U (utilization), log_tau_rec (recovery), logit_decay_res (resource EMA),
        logit_decay_cos (cosine EMA), log_tau_fam (familiarity sharpness) = 5 scalars.
State:  _ema_resource (D,), _ema_out (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU68(nn.Module):
    """Familiarity-sensitive depletion: only familiar-direction firing depletes vesicles."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # Depletion parameters
        self.logit_U         = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))   # U ≈ 0.3
        self.log_tau_rec     = nn.Parameter(torch.tensor(math.log(math.exp(4.0) - 1.0)))  # τ ≈ 4
        self.logit_decay_res = nn.Parameter(torch.tensor(math.log(0.95 / 0.05)))

        # Familiarity signal (EMA cosine, same as gelu28)
        self.logit_decay_cos = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))
        self.log_tau_fam     = nn.Parameter(torch.tensor(math.log(2.0)))   # softens cosine

        self._ema_resource: torch.Tensor = None
        self._ema_out:      torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_resource = None
        self._ema_out      = None
        self._ready        = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        U_val   = torch.sigmoid(self.logit_U).detach().item()
        tau_val = F.softplus(self.log_tau_rec).clamp(min=0.5).detach().item()
        d_res   = torch.sigmoid(self.logit_decay_res).detach().item()
        d_cos   = torch.sigmoid(self.logit_decay_cos).detach().item()
        tau_fam = self.log_tau_fam.exp().detach()
        rec_rate = 1.0 - math.exp(-1.0 / tau_val)

        out = self._gelu(x)   # (B, T, D)

        # ── Init both EMA states ───────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._ema_resource = torch.ones(D, device=x.device, dtype=out.dtype)
                self._ema_out = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
                self._ready = True
            return out

        # ── Familiarity signal: cosine to EMA output direction ────────────────
        # Per-token scalar familiarity ∈ (-1, 1)
        out_norm  = F.normalize(out.detach(), dim=-1)                 # (B, T, D)
        ema_norm  = F.normalize(self._ema_out, dim=0).view(1, 1, D)  # (1, 1, D)
        cos_sim   = (out_norm * ema_norm).sum(-1).clamp(-1, 1)       # (B, T)
        # familiarity ∈ [0, 1]: positive cosine → familiar; negative → novel
        familiarity = torch.relu(cos_sim)                             # (B, T)

        # ── Familiarity-gated vesicle depletion ───────────────────────────────
        r = self._ema_resource.unsqueeze(0).expand(B, D).clone()   # (B, D)
        r_trace = []

        with torch.no_grad():
            for t in range(T):
                r_trace.append(r.clone())
                # Depletion ∝ familiarity (scalar) * r (per-channel available)
                fam_t = familiarity[:, t].unsqueeze(-1)              # (B, 1)
                used  = (U_val * r * fam_t).clamp(max=r * 0.99)     # (B, D)
                rec   = (1.0 - r) * rec_rate
                r     = (r - used + rec).clamp(0.01, 1.0)

        r_gates = torch.stack(r_trace, dim=1)                         # (B, T, D)

        # ── Contrast normalization (energy-preserving) ────────────────────────
        r_norm  = r_gates / (r_gates.mean(dim=-1, keepdim=True) + self.eps)
        output  = out * r_norm                                        # (B, T, D)

        # ── Update both EMA states ─────────────────────────────────────────────
        with torch.no_grad():
            self._ema_resource = d_res * self._ema_resource + (1 - d_res) * r.mean(0)
            bm = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
            self._ema_out = d_cos * self._ema_out + (1 - d_cos) * bm

        return output
