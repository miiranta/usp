"""GELU4 – Learned linear gate on top of GELU (GLU-family).

GeGLU (Noam Shazeer, 2020) showed that gating GELU with a learned linear
projection gives consistent perplexity improvements over plain GELU.
True GeGLU doubles D_FF; here we stay within the activation API by using
a lazy (D→D) projection as the gate, so no architecture change is needed.

    gate   = sigmoid(x @ W.T + b)        # (B, T, D) learned gate
    output = GELU(x) * gate              # element-wise

Initialization: W=0, b=4.0  →  gate≈0.98  →  starts as plain GELU.
The network then learns to selectively close gates for unimportant activations.

The lazy Linear is assigned inside forward() so it self-registers with the
parent nn.Module via PyTorch's __setattr__, and is therefore included in
model.parameters() after the dummy forward pass in run_experiment().
"""

import math
import torch
import torch.nn as nn


class GELU4(nn.Module):
    """Gated GELU: GELU(x) · sigmoid(Wx + b), lazily initialized."""

    def __init__(self):
        super().__init__()
        self._gate_proj: nn.Linear = None   # lazily created on first forward

    def reset_state(self):
        pass   # stateless (no EMA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lazy init – runs once; PyTorch __setattr__ auto-registers the module
        if self._gate_proj is None:
            D = x.shape[-1]
            proj = nn.Linear(D, D, bias=True).to(x.device)
            nn.init.zeros_(proj.weight)
            nn.init.constant_(proj.bias, 4.0)   # sigmoid(4) ≈ 0.982 → open gate
            self._gate_proj = proj               # registers as submodule

        gate = torch.sigmoid(self._gate_proj(x))
        gelu = (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        )
        return gelu * gate
