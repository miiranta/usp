"""GELU43 – Per-Channel Sequence Contrast (vectorised alpha).

gelu39 uses a single scalar alpha to amplify ALL channels' deviation from the sequence mean:
    output = GELU(x) + alpha * (GELU(x) - mu)

But different channels in D_FF carry different information:
- Some channels encode syntax (high-frequency, low-surprise → benefit from strong contrast)
- Some encode rare semantics (already distinctive → benefit from weak contrast)

GELU43 replaces the scalar alpha with a full D-dimensional learnable vector:
    output_d = GELU(x)_d + alpha_d * (GELU(x)_d - mu_d)
             = (1 + alpha_d) * GELU(x)_d  -  alpha_d * mu_d

Each channel learns its OWN optimal contrast strength.  This is a diagonal
linear layer applied to the deviation — maximally parameter-efficient while
giving the model full per-channel control.

RELATIONSHIP TO GELU39:
    gelu39 = GELU43 where all alpha_d are tied to a single scalar.
    GELU43 strictly generalises gelu39.

Params: alpha (D,) — per-channel contrast strengths, init to 0.3.
         For D_FF=1024: 1024 extra params = 0.01% overhead.
State:   None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU43(nn.Module):
    """Per-channel EMA contrast: vectorised per-channel alpha over D_FF dimension."""

    def __init__(self):
        super().__init__()
        # Lazy init — D unknown until first forward
        self._alpha_raw: nn.Parameter = None
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))  # d=0.9
        self._ema: torch.Tensor = None
        self._ready = False

    def _init_alpha(self, D: int, device):
        init_val = math.log(math.exp(0.3) - 1.0)
        self._alpha_raw = nn.Parameter(torch.full((D,), init_val, device=device))

    def reset_state(self):
        self._ema = None; self._ready = False

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
        if self._alpha_raw is None:
            self._init_alpha(D, x.device)

        out   = self._gelu(x)                              # (B, T, D)
        alpha = F.softplus(self._alpha_raw)                # (D,)
        d_val = torch.sigmoid(self.logit_decay).detach().item()

        # Cross-batch EMA — causally valid
        with torch.no_grad():
            bm = out.detach().mean(dim=(0, 1))             # (D,)
            if not self._ready:
                self._ema = bm.clone(); self._ready = True
                return out
            self._ema = d_val * self._ema + (1 - d_val) * bm

        mu     = self._ema.unsqueeze(0).unsqueeze(0)       # (1, 1, D)
        dev    = out - mu                                  # (B, T, D)
        output = out + alpha.unsqueeze(0).unsqueeze(0) * dev
        return output
