"""GELU45 – Double Sequence Contrast (pre + post nonlinearity).

Combines GELU44 (pre-activation contrast) and GELU39 (post-activation contrast)
into a single module.  Each stage has its own learnable alpha.

MECHANISM:
    Stage 1 (pre-GELU): contrast x in activation space
        x_c = x + alpha1 * (x - mean_T(x))           (B, T, D)

    Apply nonlinearity:
        out = GELU(x_c)                               (B, T, D)

    Stage 2 (post-GELU): contrast GELU output
        mu  = mean_T(out)                             (B, 1, D)
        output = out + alpha2 * (out - mu)            (B, T, D)

WHY TWO STAGES COULD BE BETTER THAN ONE:
    Stage 1 (pre) pushes dominant activations further into high-GELU-gain regime
    AND pushes weak activations toward zero/suppression.  This is a nonlinear gate.

    Stage 2 (post) then amplifies the resulting deviations in output space,
    ensuring that whatever distinctive features the nonlinearity emphasised are
    further spread apart.

    The two stages interact multiplicatively: Stage 1 changes WHICH features
    survive the nonlinearity; Stage 2 amplifies HOW DISTINCT those features are.
    Together they create a cascade of distinctiveness — like two layers of
    lateral inhibition operating at different representational levels.

Params: log_alpha1_raw, log_alpha2_raw (2 scalars).
State:  None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU45(nn.Module):
    """Double EMA contrast: pre-GELU EMA contrast + post-GELU EMA contrast."""

    def __init__(self):
        super().__init__()
        init = math.log(math.exp(0.3) - 1.0)
        self.log_alpha1_raw  = nn.Parameter(torch.tensor(init))
        self.log_alpha2_raw  = nn.Parameter(torch.tensor(init))
        self.logit_decay_pre  = nn.Parameter(torch.tensor(math.log(9.0)))  # d=0.9 pre
        self.logit_decay_post = nn.Parameter(torch.tensor(math.log(9.0)))  # d=0.9 post
        self._ema_pre:  torch.Tensor = None
        self._ema_post: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_pre = None; self._ema_post = None; self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha1 = F.softplus(self.log_alpha1_raw)
        alpha2 = F.softplus(self.log_alpha2_raw)
        d_pre  = torch.sigmoid(self.logit_decay_pre).detach().item()
        d_post = torch.sigmoid(self.logit_decay_post).detach().item()

        # Cross-batch EMA for pre-activation
        first_step = False
        with torch.no_grad():
            bm_pre = x.detach().mean(dim=(0, 1))
            if not self._ready:
                self._ema_pre  = bm_pre.clone()
                self._ema_post = bm_pre.clone()
                self._ready    = True
                first_step     = True
            else:
                self._ema_pre = d_pre * self._ema_pre + (1 - d_pre) * bm_pre

        if first_step:
            return self._gelu(x)  # outside no_grad: gradients preserved

        mu_x = self._ema_pre.unsqueeze(0).unsqueeze(0)
        x_c  = x + alpha1 * (x - mu_x)
        out  = self._gelu(x_c)

        # Cross-batch EMA for post-activation
        with torch.no_grad():
            bm_post = out.detach().mean(dim=(0, 1))
            self._ema_post = d_post * self._ema_post + (1 - d_post) * bm_post

        mu_out = self._ema_post.unsqueeze(0).unsqueeze(0)
        output = out + alpha2 * (out - mu_out)
        return output
