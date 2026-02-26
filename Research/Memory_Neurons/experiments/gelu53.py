"""GELU53 – FFT Spectral Whitening.

SIGNAL PROCESSING INSIGHT:
    Think of the D feature channels as a discrete signal of length D.
    Apply rfft across the D dimension → complex frequency spectrum.

    Channels that share a dominant "oscillation mode" across many tokens form
    the LOW-ENERGY FAMILIAR BACKGROUND — the model has learned to produce them
    reliably. Other modes that fire rarely are HIGH-SPECIFICITY NOVEL SIGNALS.

MECHANISM:
    out     = GELU(x)                             (B, T, D)
    F       = rfft(out, dim=-1)                   (B, T, D//2+1)  complex
    power   = |F|²                                (B, T, D//2+1)  real

    Track EMA of average power per frequency bin:
        ema_pow[f] ← d * ema_pow[f] + (1-d) * mean_BT(power[:,:,f])

    Build suppression gate (spectral whitening):
        gate[f] = 1 / (ema_pow[f]^alpha + eps)   high power → more suppressed
        gate    = gate / mean(gate)               normalize so total energy preserved

    Apply gate in frequency domain → ifft → real output.

    output = beta * ifft(F * gate_re) + (1 - beta) * out
    where beta is a learned blend weight so network can interpolate.

WHY THIS WORKS:
    Familiar patterns produce predictable activation modes across channels;
    these modes accumulate high power in ema_pow. The whitening gate divides
    them away, leaving only the RARE, HIGH-FREQUENCY (in channel-frequency space)
    components that carry unique information.

    This is the exact analogue of spectral whitening in audio processing:
    suppress the 50Hz hum (familiar baseline), reveal the speech signal (novel content).

CAUSAL GUARANTEE:
    FFT is over the D dimension (channels), NOT over T (time).
    output[t] depends only on x[t] → zero causal violation.
    EMA updated post-forward under no_grad.

Params: logit_decay, log_alpha_raw, log_beta_raw — 3 scalars.
State:  ema_pow (D//2+1,) — non-trainable EMA of per-frequency power.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU53(nn.Module):
    """FFT spectral whitening: suppress overrepresented channel-frequency modes."""

    def __init__(self, ema_decay: float = 0.95, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        # alpha: exponent on power; 0 = no suppression, 1 = full 1/power whitening
        self.log_alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # ≈ 0.5
        # beta: blend weight between whitened and original
        self.log_beta_raw  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # ≈ 0.3

        self._ema_pow: torch.Tensor = None   # (D//2+1,)
        self._ready = False

    def reset_state(self):
        self._ema_pow = None
        self._ready   = False

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
        F_bins = D // 2 + 1

        out   = self._gelu(x)    # (B, T, D)
        alpha = F.softplus(self.log_alpha_raw)
        beta  = F.softplus(self.log_beta_raw).clamp(max=1.0)
        d_val = torch.sigmoid(self.logit_decay).detach().item()

        # ── Initialise EMA power on first call ────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                spec  = torch.fft.rfft(out.detach(), dim=-1)       # (B, T, F_bins)
                power = spec.abs().pow(2).mean(dim=(0, 1))         # (F_bins,)
                self._ema_pow = power.clamp(min=self.eps).clone()
                self._ready   = True
            return out   # warm-up: no whitening yet

        # ── FFT of GELU output ────────────────────────────────────────────────
        spec  = torch.fft.rfft(out, dim=-1)                        # (B, T, F_bins)

        # ── Build spectral gate from EMA power ───────────────────────────────
        with torch.no_grad():
            gate_raw  = 1.0 / (self._ema_pow.pow(alpha) + self.eps)   # (F_bins,)
            gate_norm = gate_raw / gate_raw.mean()                   # preserve mean energy
            gate_norm = gate_norm.clamp(max=10.0)                    # stability cap

        # gate is real-valued; apply to complex spectrum
        gate_bc = gate_norm.view(1, 1, F_bins)                    # broadcast
        spec_whitened = spec * gate_bc                             # (B, T, F_bins)

        # ── Inverse FFT → real ────────────────────────────────────────────────
        out_whitened = torch.fft.irfft(spec_whitened, n=D, dim=-1) # (B, T, D)

        # ── Blend whitened with original ─────────────────────────────────────
        output = beta * out_whitened + (1.0 - beta) * out         # (B, T, D)

        # ── Update EMA power (no grad) ────────────────────────────────────────
        with torch.no_grad():
            power_batch = spec.detach().abs().pow(2).mean(dim=(0, 1))  # (F_bins,)
            self._ema_pow = d_val * self._ema_pow + (1 - d_val) * power_batch.clamp(min=self.eps)

        return output
