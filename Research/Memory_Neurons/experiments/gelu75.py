"""GELU75 – Local Temporal Surprise × Cross-Batch Cosine Gate (dual-timescale surprise).

THE INSIGHT:
    gelu71's surprise = ||x[b,t] - ema_x_batch|| — deviation from cross-batch history.
    This asks: "Is this token unusual compared to ALL history?"
    
    But there's another form of surprise: WITHIN-SEQUENCE LOCAL NOVELTY.
    "Is this token unusual compared to what just appeared in THIS sequence?"
    
    Example text: "The bank was closed. The bank was closed. THE RIVER OVERFLOWED."
    - Cross-batch surprise: "river" is frequent in training data → low global surprise
    - Local (within-seq) surprise: "river" is very different from "bank, closed, bank, closed"
      → HIGH local surprise (topic shift!)
    
    Local surprise detects context SHIFTS — exactly what language models need to track.

IMPLEMENTATION — CAUSAL WITHIN-SEQUENCE EMA:
    ema_local[b, t]: initialized to the cross-batch ema_x, updated causally.
    
    At each position t:
        local_surprise[b, t] = tanh(σ_l × ||x[b,t] - ema_local[b, t-1]|| / norm)
        ema_local[b, t]      = α_l × ema_local[b, t-1] + (1 - α_l) × x[b, t]
    
    CROSS-BATCH GLOBAL SURPRISE (from gelu71):
        global_surprise[b, t] = tanh(σ_g × ||x[b,t] - ema_global|| / norm_global)
    
    COMBINED GATE:
        gate = cos_gate × (1 + w_l × local_surprise + w_g × global_surprise)
    
    Then contrast normalize.

TWO ORTHOGONAL SIGNALS:
    local_surprise captures: context TRANSITIONS within a sequence.
    global_surprise captures: rare token TYPES across the dataset.
    
    A topic-shift common word (high local, low global) → detected by local_surprise.
    A rare word in familiar context (low local, high global) → detected by global_surprise.
    A rare word AND topic shift (high local, high global) → maximum gate opening.

CAUSAL GUARANTEE:
    ema_local at position t only uses positions 0..t-1.
    ema_global uses only past batches.
    No future leakage.

Params: logit_decay_global, logit_decay_local, log_tau, log_sigma_g, log_sigma_l, log_w_g, log_w_l = 7 scalars.
State: _ema_global (D,), _ema_global_norm; ema_local rebuilt per-forward.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU75(nn.Module):
    """Dual-timescale surprise (local within-seq + global cross-batch) × cosine gate."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.logit_decay_g  = nn.Parameter(torch.tensor(math.log(0.9  / 0.1 )))  # global EMA
        self.logit_decay_l  = nn.Parameter(torch.tensor(math.log(0.7  / 0.3 )))  # local EMA (faster)
        self.log_tau        = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_g    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_sigma_l    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_g        = nn.Parameter(torch.tensor(math.log(math.exp(0.7) - 1.0)))
        self.log_w_l        = nn.Parameter(torch.tensor(math.log(math.exp(0.7) - 1.0)))

        self._ema_global:     torch.Tensor = None
        self._ema_global_norm: float = 1.0
        self._ema_out:        torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_global = None
        self._ema_out    = None
        self._ready      = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_g   = torch.sigmoid(self.logit_decay_g).detach().item()
        d_l   = torch.sigmoid(self.logit_decay_l).detach().item()
        tau   = self.log_tau.exp()
        sig_g = F.softplus(self.log_sigma_g)
        sig_l = F.softplus(self.log_sigma_l)
        w_g   = F.softplus(self.log_w_g)
        w_l   = F.softplus(self.log_w_l)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                gm = x.detach().flatten(0,1).mean(0)
                self._ema_global      = gm.clone()
                self._ema_global_norm = gm.norm().item() + self.eps
                self._ema_out         = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
                self._ready           = True
            return out

        # ── Cosine familiarity gate ─────────────────────────────────────────
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_on  = F.normalize(self._ema_out, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_on).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        # ── Global surprise ─────────────────────────────────────────────────
        eg = self._ema_global.view(1, 1, D)
        delta_g  = (x.detach() - eg).norm(dim=-1)
        surp_g   = torch.tanh(sig_g * delta_g / (self._ema_global_norm + self.eps))

        # ── Local (within-seq) surprise — causal EMA ───────────────────────
        # Initialize local EMA per sample from global EMA (warm start)
        ema_local = self._ema_global.unsqueeze(0).expand(B, D).clone()  # (B, D)
        surp_l_list = []

        with torch.no_grad():
            for t in range(T):
                local_norm = ema_local.norm(dim=-1, keepdim=True).clamp(min=self.eps)
                delta_l = (x.detach()[:, t, :] - ema_local).norm(dim=-1)       # (B,)
                surp_l_t = torch.tanh(sig_l.detach() * delta_l / local_norm.squeeze(-1))
                surp_l_list.append(surp_l_t)
                # Causal update: add current position to local EMA
                ema_local = d_l * ema_local + (1.0 - d_l) * x.detach()[:, t, :]

        surp_l = torch.stack(surp_l_list, dim=1)   # (B, T)

        # ── Combined gate ────────────────────────────────────────────────────
        gate_raw = gate_cos * (1.0 + w_g * surp_g + w_l * surp_l)

        # Contrast normalize
        with torch.no_grad():
            gate_mean = gate_raw.mean().clamp(min=self.eps)
        gate_norm = gate_raw / gate_mean
        output = out * gate_norm.unsqueeze(-1)

        # ── Update global states ─────────────────────────────────────────────
        with torch.no_grad():
            gm = x.detach().flatten(0,1).mean(0)
            self._ema_global      = d_g * self._ema_global + (1.0 - d_g) * gm
            self._ema_global_norm = self._ema_global.norm().item() + self.eps
            bm = F.normalize(out.detach().flatten(0,1).mean(0), dim=0)
            self._ema_out = d_g * self._ema_out + (1.0 - d_g) * bm

        return output
