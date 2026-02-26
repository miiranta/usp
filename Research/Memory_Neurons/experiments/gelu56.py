"""GELU56 – Biological Synaptic Depression (Tsodyks-Markram model).

THE ACTUAL BIOLOGICAL MECHANISM OF REPETITION SUPPRESSION:
    Neuroscience has a precise model of why neurons suppress repeated stimuli:
    SYNAPTIC DEPRESSION. Each synapse has a pool of neurotransmitter vesicles
    (resources). When the synapse fires, it depletes some resources. Resources
    recover with time constant tau_rec.

    Tsodyks & Markram (1997):
        R  = fraction of resources available  ∈ (0, 1)
        U  = utilization rate per activation  (learned)
        dR/dt = (1 - R) / tau_rec  -  U * R * firing_rate

    This is SELF-REGULATING:
        - Novel input: R ≈ 1 → full transmission
        - Repeated strong input: R depletes → suppressed transmission
        - Silence: R recovers toward 1 automatically (no manual reset needed)

IMPLEMENTATION (vectorized over batch, causal over sequence):
    r[b, 0, d] = 1.0                           initial full resources
    effective_out[b, t, d] = GELU(x[b,t,d]) * r[b, t, d]   gated output
    
    Depletion at t:
        used[t]   = U * r[t] * |GELU(x[t])|    (proportional to firing strength)
    Recovery at t→t+1:
        recovery  = (1 - r[t]) * (1 - exp(-1/tau_rec))
    r[t+1] = clamp(r[t] - used[t] + recovery, 0.01, 1.0)

    r is updated under no_grad (causal resource tracking, not learned path).
    Gradient flows only through: GELU(x[t]) * r[t] where r[t] is treated as constant.

BETWEEN-SEQUENCE CARRY:
    After each batch, the mean residual resource level is tracked in an EMA.
    The next sequence starts with resources = ema_resource (not always 1.0).
    This means channels that were systematically overused across many sequences
    start PARTIALLY depleted — cross-sequence memory of chronic overuse.

WHY THIS IS FUNDAMENTALLY DIFFERENT FROM ALL PRIOR EXPERIMENTS:
    - All EMA experiments: suppression based on cosine to a PROTOTYPE VECTOR.
      Two tokens can be very different but equally suppressed if both are "normal".
    - GELU56: suppression based on RECENT ACTIVATION HISTORY OF EACH CHANNEL.
      A channel that fired at t-1 through t-5 is depleted regardless of direction.
      A channel that hasn't fired in the last 10 positions starts fresh.
    
    This is pure TEMPORAL FATIGUE, not directional familiarity.

Params: logit_U (utilization, 1 scalar), log_tau_rec (recovery time, 1 scalar),
        logit_decay (EMA decay for cross-seq resource init, 1 scalar) = 3 scalars.
State:  _ema_resource (D,) — running mean resource level across batches.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU56(nn.Module):
    """Synaptic depression: per-channel temporal resource depletion & recovery."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # U: utilization rate per activation. sigmoid(logit_U) ∈ (0,1). Init ~0.2
        self.logit_U      = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))
        # tau_rec: recovery time constant in sequence steps. softplus → positive. Init ~4
        self.log_tau_rec  = nn.Parameter(torch.tensor(math.log(math.exp(4.0) - 1.0)))
        # EMA decay for cross-sequence resource initialization
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(0.95 / 0.05)))

        self._ema_resource: torch.Tensor = None   # (D,)
        self._ready = False

    def reset_state(self):
        self._ema_resource = None
        self._ready        = False

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

        U       = torch.sigmoid(self.logit_U)
        tau_rec = F.softplus(self.log_tau_rec).clamp(min=0.5)   # ≥ 0.5 steps
        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        U_val   = U.detach().item()
        tau_val = tau_rec.detach().item()
        recovery_rate = 1.0 - math.exp(-1.0 / tau_val)          # fraction recovered per step

        out = self._gelu(x)   # (B, T, D)

        # ── Initialise on first call ──────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._ema_resource = torch.ones(D, device=x.device, dtype=out.dtype)
                self._ready = True
            return out   # warm-up: full resources, no gating

        # ── Run synaptic depression along T dimension ─────────────────────────
        # Start each sequence with resource = ema_resource (cross-seq memory)
        r = self._ema_resource.unsqueeze(0).expand(B, D).clone()  # (B, D) ∈ (0,1)

        gated_steps = []
        r_trace = []

        with torch.no_grad():  # resource tracking: no gradient path through r
            for t in range(T):
                r_trace.append(r.detach().clone())  # save r[t] for gating

                # Depletion: used = U * r * |GELU(x[t])|, capped so r stays ≥ 0
                firing = out[:, t, :].detach().abs()        # (B, D)
                used   = U_val * r * firing                 # (B, D)
                used   = used.clamp(max=r * 0.99)          # can't exceed available
                # Recovery toward 1
                rec = (1.0 - r) * recovery_rate             # (B, D)
                r   = (r - used + rec).clamp(0.01, 1.0)    # (B, D)

        # Stack saved r values → (B, T, D), use as gate on GELU output
        r_gates = torch.stack(r_trace, dim=1)               # (B, T, D)
        output  = out * r_gates                             # (B, T, D)

        # ── Update EMA resource level (no grad) ───────────────────────────────
        with torch.no_grad():
            mean_r = r.mean(0)   # (D,) — resource level at end of sequence
            self._ema_resource = d_val * self._ema_resource + (1 - d_val) * mean_r

        return output
