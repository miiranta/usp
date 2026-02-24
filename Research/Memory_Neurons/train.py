import os
import csv
import math
import time
import atexit

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

    # GELU2 / GELU3 EMA habituation defaults (all learnable)
    GELU2_EMA_DECAY = 0.9   # base EMA decay
    GELU2_TEMP      = 2.0   # initial τ
    GELU2_BLEND     = 0.3   # initial α blend

    # GELU3 additional defaults
    GELU3_ACT_DECAY         = 0.99  # EMA decay for per-prototype activation scoring
    GELU3_SENSITIVITY_BIRTH =  2.0  # logit init for birth threshold  (sigmoid ≈ 0.88)
    GELU3_SENSITIVITY_DEATH = -3.0  # logit init for death threshold  (sigmoid ≈ 0.05)
    GELU3_TOP_K_UPDATE       = 2    # how many nearest prototypes receive EMA updates

    # GELU4 additional defaults (soft-gate death)
    GELU4_GATE_SHARPNESS    = 10.0  # initial gate sharpness — higher → harder cutoff
    GELU4_PRUNE_EPSILON     = 0.01  # remove prototype from memory when gate < this

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


class GELU3(nn.Module):
    """Per-neuron learnable EMA GELU with a fully dynamic prototype registry.

    All five hyperparameters are (D,) vectors — one value per feature dimension:
        logit_decay          → d ∈ (0,1)  prototype memory speed
        log_tau              → τ > 0      suppression sharpness
        log_blend            → α ∈ (0,1)  max suppression strength
        log_sensitivity_birth→ θ_b ∈ (0,1) novelty threshold for spawning
        log_sensitivity_death→ θ_d ∈ (0,1) activity threshold for pruning

    Parameters are lazily created on the first forward pass (D unknown at
    __init__ time).  Build the optimizer *after* the dummy forward that
    run_experiment already performs.
    """

    def __init__(self, ema_decay: float = None):
        super().__init__()
        self._init_decay = ema_decay if ema_decay is not None else Config.GELU2_EMA_DECAY

        # Per-neuron learnable parameters – lazily created in _init_params
        self.logit_decay           : nn.Parameter = None  # (D,)
        self.log_tau               : nn.Parameter = None  # (D,)
        self.log_blend             : nn.Parameter = None  # (D,)
        self.log_sensitivity_birth : nn.Parameter = None  # (D,)
        self.log_sensitivity_death : nn.Parameter = None  # (D,)
        self._D = None

        # Dynamic prototype state
        self._emas      : torch.Tensor = None   # (K, D)
        self._act_score : list         = None   # [float], length K
        self._proto_log : list         = []     # K count sampled every training forward
        self._ready = False

    # ── Lazy parameter initialisation ─────────────────────────────────
    def _init_params(self, D: int, device: torch.device, dtype: torch.dtype):
        self._D = D
        ld = math.log(self._init_decay / (1.0 - self._init_decay))
        self.logit_decay = nn.Parameter(
            torch.full((D,), ld, device=device, dtype=dtype))
        self.log_tau = nn.Parameter(
            torch.full((D,), math.log(Config.GELU2_TEMP), device=device, dtype=dtype))
        self.log_blend = nn.Parameter(
            torch.full((D,), math.log(Config.GELU2_BLEND / (1.0 - Config.GELU2_BLEND)),
                       device=device, dtype=dtype))
        self.log_sensitivity_birth = nn.Parameter(
            torch.full((D,), Config.GELU3_SENSITIVITY_BIRTH, device=device, dtype=dtype))
        self.log_sensitivity_death = nn.Parameter(
            torch.full((D,), Config.GELU3_SENSITIVITY_DEATH, device=device, dtype=dtype))

    def reset_state(self):
        self._emas      = None
        self._act_score = None
        self._proto_log = []
        self._ready     = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )

    # ── Prototype management ───────────────────────────────────────────
    def _spawn(self, x_mean: torch.Tensor):
        self._emas      = torch.cat([self._emas, x_mean.unsqueeze(0)], dim=0)
        self._act_score.append(1.0)

    def _prune(self, death_thresh: float):
        """Remove prototypes below death_thresh; keep ≥1."""
        K = self._emas.shape[0]
        if K <= 1:
            return
        alive = [k for k in range(K) if self._act_score[k] >= death_thresh]
        if not alive:
            alive = [max(range(K), key=lambda k: self._act_score[k])]
        if len(alive) < K:
            self._emas      = self._emas[alive]
            self._act_score = [self._act_score[k] for k in alive]

    # ── Forward ────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if self._D is None:
            self._init_params(D, x.device, x.dtype)

        tau           = self.log_tau.exp()                            # (D,)
        alpha         = torch.sigmoid(self.log_blend)                 # (D,)
        d             = torch.sigmoid(self.logit_decay)               # (D,) in-graph
        d_val         = d.detach()                                    # (D,) for EMA
        thresh_birth  = torch.sigmoid(self.log_sensitivity_birth).detach().mean().item()
        thresh_death  = torch.sigmoid(self.log_sensitivity_death).detach().mean().item()

        x_mean = x.detach().flatten(0, -2).mean(0)    # (D,)

        # ── Bootstrap first call ──────────────────────────────────────
        if not self._ready:
            self._emas      = x_mean.unsqueeze(0).clone()   # (1, D)
            self._act_score = [1.0]
            self._ready     = True
            return self._gelu(x)

        # ── Cosine similarity: (B, T, K) ──────────────────────────────
        x_norm = F.normalize(x,          dim=-1)   # (B, T, D)
        p_norm = F.normalize(self._emas, dim=-1)   # (K, D)
        sim    = torch.einsum('btd,kd->btk', x_norm, p_norm)   # (B, T, K)

        max_sim, _ = sim.max(dim=-1)   # (B, T)  — best-matching prototype per token

        # ── Per-neuron novelty & gate: (B, T, D) ──────────────────────
        novelty = torch.exp(
            -tau.view(1, 1, D) * max_sim.unsqueeze(-1)
        )                                                            # (B, T, D)
        a  = alpha.view(1, 1, D)
        dg = d.view(1, 1, D)
        effective_scale = (1.0 - a * dg) + a * dg * novelty         # (B, T, D)
        blended = x * effective_scale

        # ── Competitive EMA: update top-K nearest prototypes ──────────
        x_mean_norm = F.normalize(x_mean.unsqueeze(0), dim=-1)       # (1, D)
        sims_mean   = (F.normalize(self._emas, dim=-1) @ x_mean_norm.T).squeeze(-1)  # (K,)
        K           = self._emas.shape[0]
        top_k       = min(Config.GELU3_TOP_K_UPDATE, K)
        top_indices = sims_mean.topk(top_k).indices.tolist()
        k_star      = top_indices[0]   # primary winner (for act_score)
        for rank, k in enumerate(top_indices):
            w = 1.0 / (rank + 1)       # winner gets full weight, 2nd gets 1/2, etc.
            self._emas[k] = d_val * self._emas[k] + (1.0 - d_val) * w * x_mean

        # ── Activation EMA bookkeeping ─────────────────────────────────
        a_decay = Config.GELU3_ACT_DECAY
        for k in range(K):
            target = 1.0 if k == k_star else 0.0
            self._act_score[k] = a_decay * self._act_score[k] + (1.0 - a_decay) * target

        # ── Birth ──────────────────────────────────────────────────────
        if max_sim.mean().item() < thresh_birth:
            self._spawn(x_mean)

        # ── Death ──────────────────────────────────────────────────────
        self._prune(thresh_death)

        # ── Log prototype count (training only — grad mode is a good proxy) ──
        if torch.is_grad_enabled():
            self._proto_log.append(self._emas.shape[0])

        return self._gelu(blended)


class GELU4(nn.Module):
    """Same as GELU3 but with soft differentiable death gate.

    log_sensitivity_death is a scalar IN the computation graph:
        act   = tensor(_act_score)                          # (K,) detached floats
        gate  = sigmoid((act - thresh_death) * sharpness)  # (K,) IN graph
        max_sim = (sim * gate).max(dim=-1)                 # (B, T)

    Gradient path: loss → novelty → max_sim → gated_sim → gate
                   → thresh_death → log_sensitivity_death  ✓
                   → sharpness    → log_gate_sharpness      ✓

    All scalar params (birth/death/sharpness) created at __init__.
    Per-neuron (D,) params created lazily.
    """

    def __init__(self, ema_decay: float = None):
        super().__init__()
        self._init_decay = ema_decay if ema_decay is not None else Config.GELU2_EMA_DECAY

        # All scalar params — no D dependency, created now
        self.log_sensitivity_birth = nn.Parameter(
            torch.tensor(Config.GELU3_SENSITIVITY_BIRTH))           # detached in forward
        self.log_sensitivity_death = nn.Parameter(
            torch.tensor(Config.GELU3_SENSITIVITY_DEATH))           # IN graph
        self.log_gate_sharpness    = nn.Parameter(
            torch.tensor(math.log(Config.GELU4_GATE_SHARPNESS)))   # IN graph

        # Per-neuron params — lazily created in _init_params
        self.logit_decay : nn.Parameter = None  # (D,)
        self.log_tau     : nn.Parameter = None  # (D,)
        self.log_blend   : nn.Parameter = None  # (D,)
        self._D = None

        # Dynamic prototype state
        self._emas      : torch.Tensor = None   # (K, D)
        self._act_score : list         = None   # [float], length K
        self._proto_log : list         = []     # K count sampled per training step
        self._ready = False

    # ── Lazy parameter initialisation ─────────────────────────────────
    def _init_params(self, D: int, device: torch.device, dtype: torch.dtype):
        self._D = D
        ld = math.log(self._init_decay / (1.0 - self._init_decay))
        self.logit_decay = nn.Parameter(
            torch.full((D,), ld, device=device, dtype=dtype))
        self.log_tau = nn.Parameter(
            torch.full((D,), math.log(Config.GELU2_TEMP), device=device, dtype=dtype))
        self.log_blend = nn.Parameter(
            torch.full((D,), math.log(Config.GELU2_BLEND / (1.0 - Config.GELU2_BLEND)),
                       device=device, dtype=dtype))

    def reset_state(self):
        self._emas      = None
        self._act_score = None
        self._proto_log = []
        self._ready     = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )

    # ── Prototype management ───────────────────────────────────────────
    def _spawn(self, x_mean: torch.Tensor):
        self._emas      = torch.cat([self._emas, x_mean.unsqueeze(0)], dim=0)
        self._act_score.append(1.0)

    def _cleanup(self, gate_vals: list):
        """Memory-only prune: drop prototypes whose gate collapsed. No gradient impact.
        gate_vals corresponds to prototypes present at gate-computation time;
        any newly spawned prototypes (appended after) are always kept.
        """
        K_gated = len(gate_vals)
        if K_gated <= 1:
            return
        eps     = Config.GELU4_PRUNE_EPSILON
        K_total = self._emas.shape[0]
        alive   = [k for k in range(K_gated) if gate_vals[k] >= eps]
        if not alive:
            alive = [max(range(K_gated), key=lambda k: gate_vals[k])]
        alive  += list(range(K_gated, K_total))   # keep newly born
        if len(alive) < K_total:
            self._emas      = self._emas[alive]
            self._act_score = [self._act_score[k] for k in alive]

    # ── Forward ────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if self._D is None:
            self._init_params(D, x.device, x.dtype)

        tau          = self.log_tau.exp()                                            # (D,)
        alpha        = torch.sigmoid(self.log_blend)                                 # (D,)
        d            = torch.sigmoid(self.logit_decay)                               # (D,) in-graph
        d_val        = d.detach()                                                    # (D,) for EMA
        sharpness    = self.log_gate_sharpness.exp()                                 # scalar, IN graph
        thresh_death = torch.sigmoid(self.log_sensitivity_death)                     # scalar, IN graph
        thresh_birth = torch.sigmoid(self.log_sensitivity_birth).detach().item()     # scalar, detached

        x_mean = x.detach().flatten(0, -2).mean(0)    # (D,)

        # ── Bootstrap ─────────────────────────────────────────────────
        if not self._ready:
            self._emas      = x_mean.unsqueeze(0).clone()
            self._act_score = [1.0]
            self._ready     = True
            return self._gelu(x)

        # ── Cosine similarity: (B, T, K) ──────────────────────────────
        x_norm = F.normalize(x,          dim=-1)
        p_norm = F.normalize(self._emas, dim=-1)
        sim    = torch.einsum('btd,kd->btk', x_norm, p_norm)

        # ── Soft death gate — IN graph ─────────────────────────────────
        act        = torch.tensor(self._act_score, device=x.device, dtype=x.dtype)  # (K,)
        gate       = torch.sigmoid((act - thresh_death) * sharpness)                # (K,) IN graph
        max_sim, _ = (sim * gate.view(1, 1, -1)).max(dim=-1)                        # (B, T)

        # ── Per-neuron novelty & scale: (B, T, D) ─────────────────────
        novelty = torch.exp(-tau.view(1, 1, D) * max_sim.unsqueeze(-1))
        a  = alpha.view(1, 1, D)
        dg = d.view(1, 1, D)
        effective_scale = (1.0 - a * dg) + a * dg * novelty
        blended = x * effective_scale

        # ── Competitive EMA: top-2 update ─────────────────────────────
        x_mean_norm = F.normalize(x_mean.unsqueeze(0), dim=-1)
        sims_mean   = (F.normalize(self._emas, dim=-1) @ x_mean_norm.T).squeeze(-1)
        K           = self._emas.shape[0]
        top_indices = sims_mean.topk(min(2, K)).indices.tolist()
        k_star      = top_indices[0]
        for rank, k in enumerate(top_indices):
            w = 1.0 / (rank + 1)
            self._emas[k] = d_val * self._emas[k] + (1.0 - d_val) * w * x_mean

        # ── Activation EMA bookkeeping ─────────────────────────────────
        a_decay = Config.GELU3_ACT_DECAY
        for k in range(K):
            target = 1.0 if k == k_star else 0.0
            self._act_score[k] = a_decay * self._act_score[k] + (1.0 - a_decay) * target

        # ── Birth (capped at K_MAX, uses detached thresh_birth) ────────
        if max_sim.detach().mean().item() < thresh_birth:
            self._spawn(x_mean)

        # ── Memory cleanup (post-forward, no gradient impact) ──────────
        self._cleanup(gate.detach().tolist())

        if torch.is_grad_enabled():
            self._proto_log.append(self._emas.shape[0])

        return self._gelu(blended)


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
    def __init__(self, vocab_size, cfg: Config, activation_cls=GELU):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, cfg.D_MODEL)
        self.pos_enc = PositionalEncoding(cfg.D_MODEL, cfg.DROPOUT)

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

ALL_EXPERIMENTS = [
    ("control",  GELU),
    ("gelu2_k1", GELU2),
    ("gelu3_kn", GELU3),
    ("gelu4_kn", GELU4),
]


# ─────────────────────────────────────────────
#  Single-experiment runner
# ─────────────────────────────────────────────

def run_experiment(exp_name, activation_cls, vocab, device):
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
        len(vocab), Config, activation_cls
    ).to(device)

    # Materialise any LazyLinear parameters (e.g. GELU2Selective.W_decay)
    # by running a single dummy forward pass before inspecting parameters.
    with torch.no_grad():
        dummy = torch.zeros(1, Config.SEQ_LEN, dtype=torch.long, device=device)
        model(dummy)
    # Reset EMA state so the dummy pass doesn't pollute training
    for m in model.modules():
        if hasattr(m, 'reset_state'):
            m.reset_state()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}\n")

    # Save model info so plots can compare parameter counts across experiments
    with open(os.path.join(output_dir, "model_info.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["n_params"], [n_params]])

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

        # ── GELU3 / GELU4: print prototype-count frequency for this epoch ─
        proto_mods = [m for m in model.modules() if isinstance(m, (GELU3, GELU4))]
        if proto_mods:
            from collections import Counter
            epoch_log = []
            for m in proto_mods:
                epoch_log.extend(m._proto_log)
                m._proto_log.clear()
            if epoch_log:
                cls_name = type(proto_mods[0]).__name__
                counts   = Counter(epoch_log)
                total    = len(epoch_log)
                max_freq = max(counts.values())
                print(f"  Prototype count distribution ({cls_name}, {len(proto_mods)} layer(s), {total} samples):")
                for k in sorted(counts):
                    bar = "█" * max(1, round(counts[k] / max_freq * 30))
                    print(f"    K={k:3d}: {counts[k]:6d} ({100*counts[k]/total:5.1f}%) {bar}")
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

    for exp_name, activation_cls in ALL_EXPERIMENTS:
        run_experiment(exp_name, activation_cls, vocab, device)


if __name__ == "__main__":
    main()
