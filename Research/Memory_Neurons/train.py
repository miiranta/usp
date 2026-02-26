import os
import csv
import math
import time
import atexit
import functools
import sys

# Allow importing from experiments/ subfolder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.gelu5 import GELU5
from experiments.gelu6 import GELU6
from experiments.gelu7 import GELU7
from experiments.gelu8 import GELU8
from experiments.gelu10 import GELU10
from experiments.gelu11 import GELU11
from experiments.gelu12 import GELU12
from experiments.gelu13 import GELU13
from experiments.gelu16 import GELU16
from experiments.gelu17 import GELU17
from experiments.gelu18 import GELU18
from experiments.gelu19 import GELU19
from experiments.gelu20 import GELU20
from experiments.gelu21 import GELU21
from experiments.gelu22 import GELU22
from experiments.gelu23 import GELU23
from experiments.gelu24 import GELU24
from experiments.gelu25 import GELU25
from experiments.gelu26 import GELU26
from experiments.gelu27 import GELU27
from experiments.gelu28 import GELU28
from experiments.gelu29 import GELU29
from experiments.gelu30 import GELU30
from experiments.gelu31 import GELU31
from experiments.gelu32 import GELU32
from experiments.gelu33 import GELU33
from experiments.gelu34 import GELU34
from experiments.gelu35 import GELU35
from experiments.gelu36 import GELU36
from experiments.gelu37 import GELU37
from experiments.gelu38 import GELU38
from experiments.gelu39 import GELU39
from experiments.gelu40 import GELU40
from experiments.gelu41 import GELU41
from experiments.gelu42 import GELU42
from experiments.gelu43 import GELU43
from experiments.gelu44 import GELU44
from experiments.gelu45 import GELU45
from experiments.gelu46 import GELU46
from experiments.gelu47 import GELU47
from experiments.gelu48 import GELU48
from experiments.gelu49 import GELU49
from experiments.gelu50 import GELU50
from experiments.gelu51 import GELU51
from experiments.gelu52 import GELU52
from experiments.gelu53 import GELU53
from experiments.gelu54 import GELU54
from experiments.gelu55 import GELU55
from experiments.gelu56 import GELU56
from experiments.gelu57 import GELU57
from experiments.gelu58 import GELU58
from experiments.gelu59 import GELU59
from experiments.gelu60 import GELU60
from experiments.gelu61 import GELU61
from experiments.gelu62 import GELU62
from experiments.gelu63 import GELU63
from experiments.gelu64 import GELU64
from experiments.gelu65 import GELU65
from experiments.gelu66 import GELU66
from experiments.gelu67 import GELU67
from experiments.gelu68 import GELU68
from experiments.gelu69 import GELU69
from experiments.gelu70 import GELU70
from experiments.gelu71 import GELU71
from experiments.gelu72 import GELU72

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
    D_MODEL     = 128    # embedding / hidden dim
    N_HEADS     = 4      # attention heads
    N_LAYERS    = 4      # transformer encoder layers
    D_FF        = 1024   # feed-forward inner dim
    DROPOUT     = 0.1
    SEQ_LEN     = 64    # context window (tokens)

    # Training
    BATCH_SIZE  = 32
    EPOCHS      = 15
    LR          = 3e-4
    GRAD_CLIP   = 1.0
    LOG_EVERY   = 200    # steps between log lines

    # Checkpoint
    CHECKPOINT  = os.path.join(OUTPUT_DIR, "best_model.pt")

    # GELU2
    GELU2_EMA_DECAY = 0.9   # base EMA decay
    GELU2_TEMP      = 2.0   # initial τ
    GELU2_BLEND     = 0.3   # initial α blend

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

# Each entry: (name, activation_cls, early_stop)
# early_stop = (check_epoch, max_val_ppl) - aborts if val_ppl exceeds threshold at check_epoch
# Control epoch-5 val_ppl = 239.14; +10% slack = 263
ALL_EXPERIMENTS = [
    ("control",   GELU,                                     None),
    ("gelu2_k1",  functools.partial(GELU2, n_prototypes=1), None),
    ("gelu5",     GELU5,                                    None),
    ("gelu6",     GELU6,                                    None),
    ("gelu7",     GELU7,                                    None),
    ("gelu8",     functools.partial(GELU8, n_prototypes=4), None),
    ("gelu10",    GELU10,                                   None),  # sign-agreement (re-run)
    ("gelu11",    GELU11,                                   None),  # deviation amplification (done)
    ("gelu12",    functools.partial(GELU12, n_prototypes=8), None),  # WTA prototype memory
    ("gelu13",    functools.partial(GELU13, n_prototypes=8), None),  # momentum-updated memory
    ("gelu16",    GELU16,                                    None),  # causal temporal-difference surprise
    ("gelu17",    GELU17,                                    None),  # per-channel predictive coding gate
    ("gelu18",    GELU18,                                    None),  # within-sequence episodic familiarity
    ("gelu19",    GELU19,                                    None),  # divisive normalization (homeostatic gain)
    ("gelu20",    GELU20,                                    None),  # orthogonal familiarity subtraction
    ("gelu21",    GELU21,                                    None),  # cross-position channel uniqueness
    ("gelu22",    GELU22,                                    None),  # dual-timescale EMA (fast+slow)
    ("gelu23",    GELU23,                                    None),  # EMA threshold shift (adaptive spike threshold)
    ("gelu24",    GELU24,                                    None),  # Z-score calibrated familiarity suppression
    ("gelu25",    GELU25,                                    None),  # output-side EMA gate (suppress habitual GELU outputs)
    ("gelu26",    GELU26,                                    None),  # diagonal-Gaussian NLL surprise (Mahalanobis)
    ("gelu27",    GELU27,                                    None),  # backprop-trained familiarity prototype (no EMA)
    ("gelu28",    GELU28,                                    None),  # output-cosine EMA gate (cosine on GELU output)
    ("gelu29",    GELU29,                                    None),  # contrastive dual-EMA (recent minus global)
    ("gelu30",    GELU30,                                    None),  # binary median-split hard suppression
    ("gelu31",    GELU31,                                    None),  # double gate: input cosine + output cosine combined
    ("gelu32",    GELU32,                                    None),  # projection decomp: suppress familiar direction only
    ("gelu33",    GELU33,                                    None),  # adaptive sigmoid suppression curve (learned shape)
    ("gelu34",    GELU34,                                    None),  # multi-head familiarity: per-group cosine EMA
    ("gelu35",    GELU35,                                    None),  # position-progressive: later positions suppressed more
    ("gelu36",    GELU36,                                    None),  # dual-context: max(global EMA, local sequence) familiarity
    ("gelu37",    GELU37,                                    None),  # predictive coding: amplify prediction error (subtractive EMA)
    ("gelu38",    GELU38,                                    None),  # per-channel frequency habituation (D-dim gate, not scalar)
    ("gelu39",    GELU39,                                    None),  # stateless within-sequence instance contrast (no EMA)
    ("gelu40",    functools.partial(GELU40, n_prototypes=8), None),  # gradient-trained Hopfield memory + residual amplification
    ("gelu41",    GELU41,                                    None),  # causal cumulative-mean contrast (causal within-seq coding)
    # ── gelu39 derivatives: pushing toward 50% PPL reduction ──
    ("gelu42",    GELU42,                                    None),  # full seq instance norm (mean+variance, learned affine 2D params)
    ("gelu43",    GELU43,                                    None),  # per-channel alpha vector (D params) — gelu39 generalised
    ("gelu44",    GELU44,                                    None),  # pre-activation seq contrast (contrast x before GELU)
    ("gelu45",    GELU45,                                    None),  # double contrast: pre-GELU + post-GELU (2 scalars)
    ("gelu46",    GELU46,                                    None),  # self-similarity weighted contrast (free attention inside FF)
    ("gelu47",    GELU47,                                    None),  # causal online instance norm: normalize pos-t against mean/std of past 0..t-1
    ("gelu48",    GELU48,                                    None),  # exponentially-weighted causal mean subtraction (smooth causal gelu41)
    ("gelu49",    GELU49,                                    None),  # batch-dimension contrast: subtract cross-batch mean (safe axis, not time)
    # ── signal separation experiments ──
    ("gelu50",    GELU50,                                    None),  # K-prototype orthogonal subspace suppression (geometric projection, K=8)
    ("gelu51",    GELU51,                                    None),  # per-channel EMA z-score gate: D-dimensional novelty selector
    ("gelu52",    GELU52,                                    None),  # learned low-rank interference cancellation (rank-4, task-driven)
    # ── signal processing experiments ──
    ("gelu53",    GELU53,                                    None),  # FFT spectral whitening: suppress overrepresented channel-frequency modes
    ("gelu54",    GELU54,                                    None),  # ring buffer episodic recall: exact N-episode memory, nearest-episode gate
    ("gelu55",    GELU55,                                    None),  # 5-timescale filter bank: multi-pole IIR residual amplification
    # ── self-regulating neuro-inspired experiments ──
    ("gelu56",    GELU56,                                    None),  # synaptic depression: per-channel resource depletion & recovery (Tsodyks-Markram)
    ("gelu57",    GELU57,                                    None),  # homeostatic plasticity: per-channel activity normalization toward target rate
    ("gelu58",    GELU58,                                    None),  # Oja online PCA: suppress first principal component (dominant variance direction)
    # ── richer memory experiments ──
    ("gelu59",    GELU59,                                    None),  # K=8 output-cosine EMA bank: competitive prototypes, soft-max familiarity
    ("gelu60",    GELU60,                                    None),  # within-sequence causal max-similarity + cross-batch EMA dual memory
    ("gelu61",    GELU61,                                    None),  # K=16 second-moment EMA modes: mean cosine² familiarity (power-based)
    # ── gelu56 variations: fixing energy loss + dual-axis ──
    ("gelu62",    GELU62,                                    None),  # contrast-normalized depression: tanh firing + r/mean(r) energy preservation
    ("gelu63",    GELU63,                                    None),  # depression × EMA cosine: orthogonal dual-axis suppression
    ("gelu64",    GELU64,                                    None),  # two-pool depression: fast (burst) + slow (sustained) resource pools
    # ── biology-faithful mechanisms: calcium, shunting, threshold, familiarity ──
    ("gelu65",    GELU65,                                    None),  # Ca-AHP: per-channel causal calcium → AHP gate, contrast-normalized
    ("gelu66",    GELU66,                                    None),  # shunting inhibition: divisive conductance g[d]/mean(g), relative normalization
    ("gelu67",    GELU67,                                    None),  # adaptive firing threshold: GELU(x - EMA_x), pre-GELU threshold shift
    ("gelu68",    GELU68,                                    None),  # familiarity-sensitive depletion: cosine-gated vesicle release
    # ── breaking the 4.4% ceiling: contrast-norm, dual-timescale, surprise, opponent-process ──
    ("gelu69",    GELU69,                                    None),  # contrast-norm double cosine: novelty amplified (gate/mean), not just suppressed
    ("gelu70",    GELU70,                                    None),  # dual-timescale: within-seq fast EMA × cross-batch slow EMA, contrast-norm
    ("gelu71",    GELU71,                                    None),  # surprise × cosine: input-deviation surprise boosts familiar-direction gate
    ("gelu72",    GELU72,                                    None),  # opponent-process: output = GELU(x) + alpha*(GELU(x) - ema_out) deviation ampl
]


# ─────────────────────────────────────────────
#  Single-experiment runner
# ─────────────────────────────────────────────

def run_experiment(exp_name, activation_cls, vocab, device, early_stop=None):
    """
    Trains a single experiment and compares against the best model (gelu2_k1).
    Early stops if validation PPL is worse than gelu2_k1 for 5 consecutive epochs.
    """
    output_dir = os.path.join(Config.OUTPUT_DIR, exp_name)
    checkpoint  = os.path.join(output_dir, "best_model.pt")

    print(f"\n{'='*55}")
    print(f"  Experiment : {exp_name}")
    act_name = getattr(activation_cls, "__name__", None) or getattr(activation_cls, "func", activation_cls).__name__
    print(f"  Activation : {act_name}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*55}")

    # ── Load best model metrics for comparison ──────────────
    best_model_metrics = {}
    best_model_path = os.path.join(Config.OUTPUT_DIR, "gelu2_k1", "metrics.csv")
    if os.path.exists(best_model_path):
        with open(best_model_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row["epoch"])
                best_model_metrics[epoch] = float(row["val_ppl"])
        print(f"  Reference model: gelu2_k1 (loaded {len(best_model_metrics)} epochs)\n")
    else:
        print(f"  No reference model found, proceeding without baseline comparison\n")

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
    consecutive_epochs_below_baseline = 0
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

        # ── Early stopping: 5 consecutive epochs below baseline ──
        if best_model_metrics and epoch in best_model_metrics:
            baseline_ppl = best_model_metrics[epoch]
            if vl_ppl > baseline_ppl:
                consecutive_epochs_below_baseline += 1
                print(f"  ⚠ Below baseline (gelu2_k1 epoch {epoch}: {baseline_ppl:.2f}) "
                      f"— {consecutive_epochs_below_baseline}/5 strikes")
            else:
                consecutive_epochs_below_baseline = 0
                print(f"  ✓ Above baseline (gelu2_k1 epoch {epoch}: {baseline_ppl:.2f})")

            if consecutive_epochs_below_baseline >= 5:
                print(f"\n  ✗ Early stop: 5 consecutive epochs below baseline. Moving to next experiment.")
                return

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

    for exp_name, activation_cls, early_stop in ALL_EXPERIMENTS:
        run_experiment(exp_name, activation_cls, vocab, device, early_stop=early_stop)


if __name__ == "__main__":
    main()
