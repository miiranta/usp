"""GELU71 – Surprise-Amplified Cosine Gate (Prediction Error Modulation).

BIOLOGICAL BASIS — DOPAMINERGIC PREDICTION ERROR:
    Dopamine neurons in VTA/SNc FIRE when events are unexpected (positive surprise).
    They PAUSE when expected events are OMITTED (negative surprise).
    This dopaminergic prediction error signal AMPLIFIES synaptic transmission
    for surprising inputs and DAMPENS it for expected ones.
    
    GELU71 implements this at the single-neuron level:
    - "Prediction" = what the EMA history says should come next
    - "Surprise"   = ||x[t] - ema_x|| / ema_norm  (how different from expected)
    - High surprise → AMPLIFY gate → more transmission
    - Low surprise + familiar direction → gate reduced → suppression

THE ORTHOGONAL COMBINATION:
    gelu28/31's cosine gate captures DIRECTIONAL familiarity:
        "This pattern is in the usual direction" → suppress
    
    gelu71 adds an INPUT DEVIATION signal:
        "This input value deviates strongly from what I expected" → boost
    
    These two signals are ORTHOGONAL:
    - You can have familiar DIRECTION but unusual MAGNITUDE  
      (e.g., usual topic word but unexpectedly cap-locked)
    - You can have unfamiliar DIRECTION but expected MAGNITUDE
      (e.g., correct format but new vocabulary)
    
    Combined: gate = cos_gate * (1 + w * surprise)
    where:
      cos_gate = exp(-τ * cos(out, ema_out))      ← direction familiarity
      surprise  = tanh(σ * ||x - ema_x|| / ema_norm)  ← input deviation magnitude ∈ (0,1)
      w = learned weight for surprise boost

CAUSAL GUARANTEE:
    ema_x and ema_out are updated AFTER forward pass, using PAST batches only.
    No future information.

INITIALIZATION:
    ema_x initialized from first batch mean of x → surprise is non-zero from batch 2.
    cos_gate starts at 1/(1 + exp decay) ≈ 0.4 for typical cos ≈ 0 → mild suppression.
    Combined gate starts mild, warms up quickly.

SELF-REGULATION:
    Chronically surprising input (novel token type) → surprise ≈ 1 → gate boosted
    → ema_x tracks the new level → surprise decreases → gate normalizes
    
    Familiar input (expected direction + small deviation) → cos_gate low + surprise≈0
    → gate ≈ cos_gate ≈ 0 → fully suppressed

    The model routes attention: suppress well-predicted tokens, amplify surprising ones.

Params: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars.
State: _ema_x (D,), _ema_out (D,) — both as unit vectors.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU71(nn.Module):
    """Surprise × cosine gate: input-deviation surprise boosts directional familiarity gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))   # cosine sharpness
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))  # σ≈1
        self.log_w_raw    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))   # w≈1

        self._ema_x:   torch.Tensor = None   # (D,) raw (not normalized) for deviation
        self._ema_out: torch.Tensor = None   # (D,) normalized for cosine
        self._ema_x_norm: float = 1.0        # scalar norm of ema_x for bounding deviation
        self._ready = False

    def reset_state(self):
        self._ema_x    = None
        self._ema_out  = None
        self._ready    = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma_raw)    # sharpness of surprise response
        w     = F.softplus(self.log_w_raw)        # surprise boost weight

        out = self._gelu(x)   # (B, T, D)

        # ── Init ─────────────────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                x_mean = x.detach().flatten(0, 1).mean(0)
                self._ema_x      = x_mean.clone()
                self._ema_x_norm = x_mean.norm().item() + self.eps
                self._ema_out    = F.normalize(out.detach().flatten(0, 1).mean(0), dim=0)
                self._ready      = True
            return out

        # ── Cosine familiarity gate (directional, as gelu28) ──────────────────
        out_n   = F.normalize(out.detach(), dim=-1)                       # (B,T,D)
        ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)                   # (B,T)
        gate_cos = torch.exp(-tau * cos_sim)                               # (B,T) ∈ (0,1]

        # ── Surprise signal (input deviation magnitude) ───────────────────────
        ema_x = self._ema_x.view(1, 1, D)                                 # (1,1,D)
        delta = (x.detach() - ema_x).norm(dim=-1)                         # (B,T)
        # Normalize by historical input norm so it's scale-independent
        surprise = torch.tanh(sigma * delta / (self._ema_x_norm + self.eps))  # (B,T) ∈ (0,1)

        # ── Combined gate: familiar direction suppressed UNLESS surprising ────
        gate = gate_cos * (1.0 + w * surprise)                             # (B,T) ≥ 0
        output = out * gate.unsqueeze(-1)                                  # (B,T,D)

        # ── Update EMA states ──────────────────────────────────────────────────
        with torch.no_grad():
            x_mean     = x.detach().flatten(0, 1).mean(0)
            self._ema_x = d_val * self._ema_x + (1.0 - d_val) * x_mean
            self._ema_x_norm = self._ema_x.norm().item() + self.eps
            bm_out = F.normalize(out.detach().flatten(0, 1).mean(0), dim=0)
            self._ema_out = d_val * self._ema_out + (1.0 - d_val) * bm_out

        return output
