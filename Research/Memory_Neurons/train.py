import os
import csv
import math
import time
import atexit
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

class Config:
    # Paths (relative to this file)
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(BASE_DIR, "dataset", "wikitext-2")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    TRAIN_FILE = os.path.join(DATA_DIR, "wiki.train.tokens")
    VALID_FILE = os.path.join(DATA_DIR, "wiki.valid.tokens")
    TEST_FILE  = os.path.join(DATA_DIR, "wiki.test.tokens")

    # Model
    D_MODEL     = 256    # embedding / hidden dim
    N_HEADS     = 4      # attention heads
    N_LAYERS    = 4      # transformer encoder layers
    D_FF        = 1024   # feed-forward inner dim
    DROPOUT     = 0.1
    SEQ_LEN     = 128    # context window (tokens)

    # Training
    BATCH_SIZE  = 64
    EPOCHS      = 20
    LR          = 3e-4
    GRAD_CLIP   = 1.0
    LOG_EVERY   = 200    # steps between log lines

    # Checkpoint
    CHECKPOINT  = os.path.join(OUTPUT_DIR, "best_model.pt")

    # GELU2 EMA habituation defaults
    GELU2_EMA_DECAY = 0.9   # base EMA decay (overridden per experiment)
    GELU2_TEMP      = 2.0   # initial τ (learnable, this is the starting value)
    GELU2_BLEND     = 0.3   # initial α blend (learnable, this is the starting value)

    @staticmethod
    def get_device():

        if not torch.cuda.is_available():
            print("No CUDA devices available. Using CPU.")
            return torch.device('cpu')

        n_gpus = torch.cuda.device_count()
        print(f"\nScanning {n_gpus} GPUs for availability (checking {Config.OUTPUT_DIR}/.gpu_lock_*):")

        # 1. Get memory info for all
        gpu_stats = []
        for i in range(n_gpus):
            try:
                free, total = torch.cuda.mem_get_info(i)
                gpu_stats.append({'id': i, 'free': free})
            except:
                gpu_stats.append({'id': i, 'free': 0})

        # Sort by free memory desc
        gpu_stats.sort(key=lambda x: x['free'], reverse=True)

        selected_gpu = -1

        # Ensure output dir exists for locks
        if not os.path.exists(Config.OUTPUT_DIR):
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # 2. Try to lock
        for stat in gpu_stats:
            gpu_id = stat['id']
            lock_file = os.path.join(Config.OUTPUT_DIR, f".gpu_lock_{gpu_id}")

            # Check existing lock
            if os.path.exists(lock_file):
                try:
                    with open(lock_file, 'r') as f:
                        lock_pid = int(f.read().strip())

                    # Check if process still exists (Windows compatible)
                    try:
                        import psutil
                        if psutil.pid_exists(lock_pid):
                            print(f"  GPU {gpu_id}: LOCKED by PID {lock_pid} ({stat['free']/1024**3:.2f}GB free). Skipping.")
                            continue
                        else:
                            print(f"  GPU {gpu_id}: Stale lock (PID {lock_pid} not running). Removing.")
                            os.remove(lock_file)
                    except ImportError:
                        # Fallback without psutil
                        print(f"  GPU {gpu_id}: LOCKED by PID {lock_pid} ({stat['free']/1024**3:.2f}GB free). Skipping.")
                        continue
                except:
                    pass

            # Try acquire
            try:
                fd = os.open(lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(fd, 'w') as f:
                    f.write(str(os.getpid()))
                selected_gpu = gpu_id
                print(f"  GPU {gpu_id}: AVAILABLE ({stat['free']/1024**3:.2f}GB free). Locked.")
                break
            except OSError:
                print(f"  GPU {gpu_id}: Failed to lock. Skipping.")

        # 3. Fallback
        if selected_gpu == -1:
            selected_gpu = gpu_stats[0]['id']
            print(f"WARNING: No free GPUs found. Falling back to GPU {selected_gpu} (Max Free RAM).")

        # Register cleanup
        def cleanup_lock(path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

        lock_path = os.path.join(Config.OUTPUT_DIR, f".gpu_lock_{selected_gpu}")

        # Only register cleanup if we actually own it
        try:
            if os.path.exists(lock_path):
                with open(lock_path, 'r') as f:
                    lock_pid = int(f.read().strip())
                    if lock_pid == os.getpid():
                        atexit.register(cleanup_lock, lock_path)
        except:
            pass

        return torch.device(f'cuda:{selected_gpu}')


# ─────────────────────────────────────────────
#  Vocabulary
# ─────────────────────────────────────────────

class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def build(self, files):
        special = [self.PAD, self.UNK]
        counts = {}
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    for tok in line.split():
                        counts[tok] = counts.get(tok, 0) + 1
        for s in special:
            self._add(s)
        for tok in sorted(counts, key=lambda x: -counts[x]):
            self._add(tok)
        print(f"Vocabulary size: {len(self)}")

    def _add(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = len(self.idx2token)
            self.idx2token.append(token)

    def encode(self, tokens):
        unk = self.token2idx[self.UNK]
        return [self.token2idx.get(t, unk) for t in tokens]

    def __len__(self):
        return len(self.idx2token)


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, path, vocab, seq_len):
        with open(path, encoding="utf-8") as f:
            tokens = f.read().split()
        ids = vocab.encode(tokens)
        # chunk into (seq_len + 1) blocks: input = ids[:-1], target = ids[1:]
        total = (len(ids) - 1) // seq_len * seq_len
        self.data = torch.tensor(ids[:total + 1], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start     : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

class GELU(nn.Module):
    """Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016).

    GELU(x) = x * Φ(x)  where Φ is the standard-normal CDF.
    Approximated as:
        0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )


class GELU2(nn.Module):
    """EMA-based habituation GELU with K learned prototypes and per-token novelty.

    Maintains K EMA prototype vectors. For each token, computes cosine similarity
    against all K prototypes; the maximum (most familiar) drives suppression:

        sim_k   = cosine(x_token, prototype_k)     # (B, T, K)
        max_sim = sim_k.max(dim=-1)                # (B, T)
        novelty = exp(-τ * max_sim)               # (B, T)
        scale   = (1 - α*d) + α*d * novelty      # (B, T, 1) per-token gate
        output  = GELU(x * scale)

    The nearest prototype is updated each step (competitive EMA learning).
    All three hyperparameters are learnable per-layer:
        logit_decay → d ∈ (0,1) : prototype memory length
        log_tau     → τ > 0    : suppression sharpness
        log_blend   → α ∈ (0,1): max suppression strength
    """

    def __init__(self, n_prototypes: int = 1, ema_decay: float = None):
        super().__init__()
        init_decay  = ema_decay if ema_decay is not None else Config.GELU2_EMA_DECAY
        self._K     = n_prototypes
        self._emas: torch.Tensor = None   # (K, D)
        self._ready = False

        self.log_tau     = nn.Parameter(torch.tensor(math.log(Config.GELU2_TEMP)))
        self.log_blend   = nn.Parameter(
            torch.tensor(math.log(Config.GELU2_BLEND / (1.0 - Config.GELU2_BLEND)))
        )
        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(init_decay / (1.0 - init_decay)))
        )

    def reset_state(self):
        self._emas = None;  self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)
        d     = torch.sigmoid(self.logit_decay)          # in-graph for gradient
        d_val = d.detach().item()                        # plain float for EMA update

        x_mean = x.detach().flatten(0, -2).mean(0)      # (D,)

        if not self._ready:
            base = x_mean.unsqueeze(0).expand(self._K, -1).clone()
            if self._K > 1:
                base = base + torch.randn_like(base) * 1e-3
            self._emas  = base
            self._ready = True
            return self._gelu(x)

        # ── Per-token cosine sim against all K prototypes ──────────────
        x_norm = F.normalize(x, dim=-1)                        # (B, T, D)
        p_norm = F.normalize(self._emas, dim=-1)               # (K, D)
        sim    = torch.einsum('btd,kd->btk', x_norm, p_norm)  # (B, T, K)

        max_sim, _ = sim.max(dim=-1)                           # (B, T)
        novelty    = torch.exp(-tau * max_sim)                 # (B, T)

        effective_scale = (1.0 - alpha * d) + alpha * d * novelty.unsqueeze(-1)  # (B, T, 1)
        blended = x * effective_scale

        # ── Competitive EMA: update nearest prototype only ─────────────
        x_mean_norm = F.normalize(x_mean.unsqueeze(0), dim=-1)          # (1, D)
        k_star      = (F.normalize(self._emas, dim=-1) * x_mean_norm).sum(-1).argmax().item()
        self._emas[k_star] = d_val * self._emas[k_star] + (1 - d_val) * x_mean

        return self._gelu(blended)


class GELU2Soft(nn.Module):
    """GELU2 with **soft** prototype assignment.

    Replaces winner-takes-all competitive update with a softmax-weighted
    update across all K prototypes.  Every prototype moves toward x_mean
    proportionally to its cosine similarity with x_mean:

        weights_k = softmax(τ_s * cosine(x_mean, prototype_k))   # (K,)
        prototype_k ← d * prototype_k + (1-d) * weights_k * (x_mean - prot_k)
                     + d * prototype_k                           (vectorised)

    This ensures *all* prototypes remain alive and each specialises
    gradually — fixing the dead-prototype problem that makes hard
    winner-takes-all degrade above K≈4.

    Extra learnable scalar:
        log_tau_soft → τ_s > 0 : sharpness of soft assignment
    """

    def __init__(self, n_prototypes: int = 1, ema_decay: float = None):
        super().__init__()
        init_decay  = ema_decay if ema_decay is not None else Config.GELU2_EMA_DECAY
        self._K     = n_prototypes
        self._emas: torch.Tensor = None
        self._ready = False

        self.log_tau      = nn.Parameter(torch.tensor(math.log(Config.GELU2_TEMP)))
        self.log_blend    = nn.Parameter(
            torch.tensor(math.log(Config.GELU2_BLEND / (1.0 - Config.GELU2_BLEND)))
        )
        self.logit_decay  = nn.Parameter(
            torch.tensor(math.log(init_decay / (1.0 - init_decay)))
        )
        # Sharpness of the soft assignment (separate from novelty τ)
        self.log_tau_soft = nn.Parameter(torch.tensor(0.0))  # starts at τ=1

    def reset_state(self):
        self._emas = None;  self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau       = self.log_tau.exp()
        tau_soft  = self.log_tau_soft.exp()
        alpha     = torch.sigmoid(self.log_blend)
        d         = torch.sigmoid(self.logit_decay)
        d_val     = d.detach().item()

        x_mean = x.detach().flatten(0, -2).mean(0)   # (D,)

        if not self._ready:
            base = x_mean.unsqueeze(0).expand(self._K, -1).clone()
            if self._K > 1:
                base = base + torch.randn_like(base) * 1e-3
            self._emas  = base
            self._ready = True
            return self._gelu(x)

        # ── Per-token novelty (same as GELU2) ──────────────────────────
        # Detach _emas so stale grad_fn from a previous soft-update never
        # leaks into the current forward graph (fixes retain_graph error).
        x_norm  = F.normalize(x, dim=-1)                                # (B, T, D)
        p_norm  = F.normalize(self._emas.detach(), dim=-1)              # (K, D)
        sim     = torch.einsum('btd,kd->btk', x_norm, p_norm)          # (B, T, K)

        max_sim, _ = sim.max(dim=-1)                            # (B, T)
        novelty    = torch.exp(-tau * max_sim)                  # (B, T)

        effective_scale = (1.0 - alpha * d) + alpha * d * novelty.unsqueeze(-1)
        blended = x * effective_scale

        # ── Soft update: outside autograd — _emas must stay grad_fn-free
        with torch.no_grad():
            x_mean_norm = F.normalize(x_mean.unsqueeze(0), dim=-1)      # (1, D)
            proto_sims  = (p_norm * x_mean_norm).sum(-1)                # (K,) cosine sims
            weights     = F.softmax(tau_soft * proto_sims, dim=0)       # (K,) ∑=1
            delta       = x_mean.unsqueeze(0) - self._emas              # (K, D)
            self._emas  = self._emas + (1 - d_val) * weights.unsqueeze(1) * delta

        return self._gelu(blended)


class GELU2Selective(nn.Module):
    """GELU2 with **input-dependent** (selective) decay, inspired by Mamba.

    Instead of a fixed scalar decay *d*, the decay is computed as a linear
    function of the current batch mean activation:

        d = sigmoid(W_decay · x_mean + b_decay)

    Initialised so W≈0 gives the same behaviour as the scalar baseline.
    As training proceeds:
        • Novel / surprising input  → W·x pulls d toward 0 → fast update
        • Familiar / routine input  → W·x pulls d toward 1 → long memory

    This lets the memory length adapt on-the-fly rather than being frozen.
    """

    def __init__(self, n_prototypes: int = 1, ema_decay: float = None):
        super().__init__()
        init_decay   = ema_decay if ema_decay is not None else Config.GELU2_EMA_DECAY
        self._K      = n_prototypes
        self._emas: torch.Tensor = None
        self._ready  = False
        self._logit0 = math.log(init_decay / (1.0 - init_decay))

        self.log_tau   = nn.Parameter(torch.tensor(math.log(Config.GELU2_TEMP)))
        self.log_blend = nn.Parameter(
            torch.tensor(math.log(Config.GELU2_BLEND / (1.0 - Config.GELU2_BLEND)))
        )

        # LazyLinear: input dim inferred on first call → works for any D_FF / D_MODEL
        self.W_decay = nn.LazyLinear(1, bias=True)

    def _maybe_init_wdecay(self):
        """After LazyLinear materialises weights, apply the desired initialisation."""
        if isinstance(self.W_decay.weight, nn.UninitializedParameter):
            return
        if getattr(self, '_wdecay_init_done', False):
            return
        nn.init.zeros_(self.W_decay.weight)
        nn.init.constant_(self.W_decay.bias, self._logit0)
        self._wdecay_init_done = True

    def reset_state(self):
        self._emas = None;  self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)

        x_mean = x.detach().flatten(0, -2).mean(0)   # (D,)

        # Input-dependent decay: scalar derived from x_mean
        d     = torch.sigmoid(self.W_decay(x_mean))   # (1,) — in graph
        self._maybe_init_wdecay()                      # fix init after first materialise
        d_val = d.detach().item()

        if not self._ready:
            base = x_mean.unsqueeze(0).expand(self._K, -1).clone()
            if self._K > 1:
                base = base + torch.randn_like(base) * 1e-3
            self._emas  = base
            self._ready = True
            return self._gelu(x)

        # ── Per-token novelty ───────────────────────────────────────────
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self._emas, dim=-1)
        sim    = torch.einsum('btd,kd->btk', x_norm, p_norm)

        max_sim, _ = sim.max(dim=-1)
        novelty    = torch.exp(-tau * max_sim)

        effective_scale = (1.0 - alpha * d) + alpha * d * novelty.unsqueeze(-1)
        blended = x * effective_scale

        # ── Competitive (hard) EMA update ──────────────────────────────
        x_mean_norm = F.normalize(x_mean.unsqueeze(0), dim=-1)
        k_star      = (p_norm * x_mean_norm).sum(-1).argmax().item()
        self._emas[k_star] = d_val * self._emas[k_star] + (1 - d_val) * x_mean

        return self._gelu(blended)


class GELU2SelectiveSoft(nn.Module):
    """GELU2 with **both** input-dependent decay (Mamba-inspired) and
    **soft** prototype assignment — the full combination.

    Combines:
    • d = sigmoid(W_decay · x_mean) — content-dependent memory length
    • weights = softmax(τ_s · cosine(x_mean, prototypes)) — no dead prototypes
    """

    def __init__(self, n_prototypes: int = 1, ema_decay: float = None):
        super().__init__()
        init_decay   = ema_decay if ema_decay is not None else Config.GELU2_EMA_DECAY
        self._K      = n_prototypes
        self._emas: torch.Tensor = None
        self._ready  = False
        self._logit0 = math.log(init_decay / (1.0 - init_decay))

        self.log_tau      = nn.Parameter(torch.tensor(math.log(Config.GELU2_TEMP)))
        self.log_blend    = nn.Parameter(
            torch.tensor(math.log(Config.GELU2_BLEND / (1.0 - Config.GELU2_BLEND)))
        )
        self.log_tau_soft = nn.Parameter(torch.tensor(0.0))

        # LazyLinear: input dim inferred on first call → works for any D_FF / D_MODEL
        self.W_decay = nn.LazyLinear(1, bias=True)

    def _maybe_init_wdecay(self):
        if isinstance(self.W_decay.weight, nn.UninitializedParameter):
            return
        if getattr(self, '_wdecay_init_done', False):
            return
        nn.init.zeros_(self.W_decay.weight)
        nn.init.constant_(self.W_decay.bias, self._logit0)
        self._wdecay_init_done = True

    def reset_state(self):
        self._emas = None;  self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau      = self.log_tau.exp()
        tau_soft = self.log_tau_soft.exp()
        alpha    = torch.sigmoid(self.log_blend)

        x_mean = x.detach().flatten(0, -2).mean(0)   # (D,)

        d     = torch.sigmoid(self.W_decay(x_mean))   # (1,) — in graph
        self._maybe_init_wdecay()                      # fix init after first materialise
        d_val = d.detach().item()

        if not self._ready:
            base = x_mean.unsqueeze(0).expand(self._K, -1).clone()
            if self._K > 1:
                base = base + torch.randn_like(base) * 1e-3
            self._emas  = base
            self._ready = True
            return self._gelu(x)

        # ── Per-token novelty ───────────────────────────────────────────
        # Detach _emas to prevent stale grad_fn from propagating (retain_graph fix).
        x_norm  = F.normalize(x, dim=-1)
        p_norm  = F.normalize(self._emas.detach(), dim=-1)
        sim     = torch.einsum('btd,kd->btk', x_norm, p_norm)

        max_sim, _ = sim.max(dim=-1)
        novelty    = torch.exp(-tau * max_sim)

        effective_scale = (1.0 - alpha * d) + alpha * d * novelty.unsqueeze(-1)
        blended = x * effective_scale

        # ── Soft update — outside autograd so _emas stays grad_fn-free ─
        with torch.no_grad():
            x_mean_norm = F.normalize(x_mean.unsqueeze(0), dim=-1)
            proto_sims  = (p_norm * x_mean_norm).sum(-1)             # (K,)
            weights     = F.softmax(tau_soft * proto_sims, dim=0)    # (K,)
            delta       = x_mean.unsqueeze(0) - self._emas           # (K, D)
            self._emas  = self._emas + (1 - d_val) * weights.unsqueeze(1) * delta

        return self._gelu(blended)


class HabituatedEncoderLayer(nn.Module):
    """TransformerEncoderLayer with EMA-based attention habituation.

    Maintains a per-head EMA of attention weight matrices. A habituation
    bias is subtracted from attention logits before softmax:

        attn_bias = -τ_a * mean_over_heads(ema_attn)   # (T, T)
        output    = softmax(QKᵀ/√d + causal_mask + attn_bias) · V

    Frequently attended positions accumulate → suppressed over time.
    τ_a and d_a are learnable per-layer.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, activation_cls):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.linear1  = nn.Linear(d_model, d_ff)
        self.linear2  = nn.Linear(d_ff, d_model)
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)
        self.activation = activation_cls()
        self._n_heads   = n_heads

        self.log_tau_attn     = nn.Parameter(torch.tensor(0.0))
        self.logit_decay_attn = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))
        self._ema_attn: torch.Tensor = None   # (H, T, T)

    def reset_state(self):
        self._ema_attn = None
        if hasattr(self.activation, 'reset_state'):
            self.activation.reset_state()

    def forward(self, x: torch.Tensor,
                causal_mask: torch.Tensor = None) -> torch.Tensor:
        tau_a   = self.log_tau_attn.exp()
        d_a_val = torch.sigmoid(self.logit_decay_attn).detach().item()
        B, T, D = x.shape

        if self._ema_attn is not None and self._ema_attn.shape[-1] == T:
            habit_bias = -(tau_a * self._ema_attn).mean(0)        # (T, T)
            attn_mask  = (causal_mask + habit_bias
                          if causal_mask is not None else habit_bias)
        else:
            attn_mask = causal_mask

        attn_out, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,   # (B, H, T, T)
        )

        w = attn_weights.detach().mean(0)   # (H, T, T)
        if self._ema_attn is None or self._ema_attn.shape != w.shape:
            self._ema_attn = w.clone()
        else:
            self._ema_attn = d_a_val * self._ema_attn + (1 - d_a_val) * w

        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(
            self.linear2(self.drop(self.activation(self.linear1(x))))
        ))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, cfg: Config, activation_cls=GELU,
                 with_attn_habituation: bool = False):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, cfg.D_MODEL)
        self.pos_enc = PositionalEncoding(cfg.D_MODEL, cfg.DROPOUT)
        self._with_attn = with_attn_habituation

        if with_attn_habituation:
            # Custom layers with per-head EMA attention bias
            self.layers = nn.ModuleList([
                HabituatedEncoderLayer(
                    cfg.D_MODEL, cfg.N_HEADS, cfg.D_FF, cfg.DROPOUT, activation_cls
                )
                for _ in range(cfg.N_LAYERS)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.D_MODEL, nhead=cfg.N_HEADS,
                dim_feedforward=cfg.D_FF, dropout=cfg.DROPOUT,
                activation=activation_cls(), batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=cfg.N_LAYERS
            )

        self.head = nn.Linear(cfg.D_MODEL, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if isinstance(p, nn.parameter.UninitializedParameter):
                continue   # LazyLinear params – not yet materialised
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        seq_len     = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )
        out = self.pos_enc(self.embed(x))
        if self._with_attn:
            for layer in self.layers:
                out = layer(out, causal_mask=causal_mask)
        else:
            out = self.transformer(out, mask=causal_mask, is_causal=True)
        return self.head(out)


# ─────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, cfg, train=True):
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()

    phase = "train" if train else "eval "
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(enumerate(loader, 1), total=len(loader), desc=f"  {phase}", leave=False)
        for step, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)                          # (B, T, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

            total_loss   += loss.item() * y.numel()
            total_tokens += y.numel()

            avg = total_loss / total_tokens
            ppl = math.exp(min(avg, 20))
            pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{ppl:.2f}")

            if train and step % cfg.LOG_EVERY == 0:
                elapsed = time.time() - t0
                tqdm.write(f"  step {step:5d} | loss {avg:.4f} | ppl {ppl:8.2f} | {elapsed:.1f}s")

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(min(avg_loss, 20))


# ─────────────────────────────────────────────
#  Experiments
# ─────────────────────────────────────────────

K_VALS     = [1, 2, 4, 8, 16]
K_VALS_EXT = [1, 2, 4, 8, 16, 32]   # extended range to stress-test K scaling

# ── Round 1: original experiments (results already cached → auto-skipped) ──
EXPERIMENTS = (
    [("control",            GELU,                                     False)]
    + [(f"gelu2_k{k}",      functools.partial(GELU2, n_prototypes=k), False) for k in K_VALS]
    + [(f"gelu2_k{k}_attn", functools.partial(GELU2, n_prototypes=k), True)  for k in K_VALS]
)

# ── Round 2: feature ablations ─────────────────────────────────────────────
#  Soft assignment only — does PPL improve *monotonically* with K now?
_soft     = [(f"gelu2_soft_k{k}",
              functools.partial(GELU2Soft, n_prototypes=k), False)
             for k in K_VALS_EXT]

#  Input-dependent (selective) decay only
_sel      = [(f"gelu2_sel_k{k}",
              functools.partial(GELU2Selective, n_prototypes=k), False)
             for k in K_VALS]

#  Both combined (full model)
_selsoft  = [(f"gelu2_selsoft_k{k}",
              functools.partial(GELU2SelectiveSoft, n_prototypes=k), False)
             for k in K_VALS_EXT]

#  Best combos with attention habituation
_soft_attn    = [(f"gelu2_soft_k{k}_attn",
                  functools.partial(GELU2Soft, n_prototypes=k), True)
                 for k in [4, 8, 16, 32]]
_selsoft_attn = [(f"gelu2_selsoft_k{k}_attn",
                  functools.partial(GELU2SelectiveSoft, n_prototypes=k), True)
                 for k in [4, 8, 16, 32]]

NEW_EXPERIMENTS = _soft + _sel + _selsoft + _soft_attn + _selsoft_attn

ALL_EXPERIMENTS = EXPERIMENTS + NEW_EXPERIMENTS


# ─────────────────────────────────────────────
#  Single-experiment runner
# ─────────────────────────────────────────────

def run_experiment(exp_name, activation_cls, vocab, device, with_attn_habituation: bool = False):
    output_dir = os.path.join(Config.OUTPUT_DIR, exp_name)
    checkpoint  = os.path.join(output_dir, "best_model.pt")

    print(f"\n{'='*55}")
    print(f"  Experiment : {exp_name}")
    act_name = getattr(activation_cls, "__name__", None) or getattr(activation_cls, "func", activation_cls).__name__
    print(f"  Activation : {act_name}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*55}")

    # ── Skip if already done ──────────────────────────────
    if os.path.exists(output_dir):
        print(f"  Output folder already exists. Skipping.\n")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Datasets / loaders
    train_ds = TokenDataset(Config.TRAIN_FILE, vocab, Config.SEQ_LEN)
    valid_ds  = TokenDataset(Config.VALID_FILE, vocab, Config.SEQ_LEN)
    test_ds   = TokenDataset(Config.TEST_FILE,  vocab, Config.SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,  drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=Config.BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE, shuffle=False, drop_last=False)

    # Model
    model = TransformerLM(
        len(vocab), Config, activation_cls,
        with_attn_habituation=with_attn_habituation
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}\n")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx[Vocab.PAD])
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_val_loss = float('inf')
    metrics_path = os.path.join(output_dir, "metrics.csv")

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_ppl", "val_loss", "val_ppl"])

    # Training loop
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"── Epoch {epoch}/{Config.EPOCHS} ──")
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, Config, train=True)
        vl_loss, vl_ppl = run_epoch(model, valid_loader, criterion, optimizer, device, Config, train=False)
        scheduler.step()

        print(f"  train loss {tr_loss:.4f} | ppl {tr_ppl:.2f}")
        print(f"  valid loss {vl_loss:.4f} | ppl {vl_ppl:.2f}")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_ppl, vl_loss, vl_ppl])

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            # torch.save(model.state_dict(), checkpoint)
            print(f"  → new best model (val ppl {vl_ppl:.2f})")
        print()

    # Test
    print("── Test ──")
    # model.load_state_dict(torch.load(checkpoint, map_location=device))
    te_loss, te_ppl = run_epoch(model, test_loader, criterion, optimizer, device, Config, train=False)
    print(f"  test loss {te_loss:.4f} | ppl {te_ppl:.2f}\n")

    with open(os.path.join(output_dir, "test_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_loss", "test_ppl"])
        writer.writerow([te_loss, te_ppl])


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    device = Config.get_device()
    print(f"Using device: {device}\n")

    # Build vocabulary once, shared across experiments
    vocab = Vocab()
    vocab.build([Config.TRAIN_FILE, Config.VALID_FILE, Config.TEST_FILE])

    for exp_name, activation_cls, with_attn in ALL_EXPERIMENTS:
        run_experiment(exp_name, activation_cls, vocab, device,
                       with_attn_habituation=with_attn)


if __name__ == "__main__":
    main()
