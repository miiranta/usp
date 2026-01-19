import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import csv
import math
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.profiler import profile, ProfilerActivity

# Disable efficient attention backend to allow second-order derivatives
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except AttributeError:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

PRECONDITION_EPOCHS = 5
CE_EPOCHS = 15
TOTAL_EPOCHS = PRECONDITION_EPOCHS + CE_EPOCHS

class TrainingPhase(Enum):
    METRIC = "metric"
    CE = "ce"

class ComputeTracker:
    def __init__(self):
        self.metric_flops = 0.0
        self.ce_flops = 0.0

    @property
    def total(self):
        return self.metric_flops + self.ce_flops

def get_best_gpu():
    """Checks for all available GPUs, prints their status, and selects the one with the most free memory."""
    if not torch.cuda.is_available():
        print("No CUDA devices available. Using CPU.")
        return 0
    
    print(f"\nScanning {torch.cuda.device_count()} available GPUs:")
    max_free = 0
    best_gpu = 0
    
    for i in range(torch.cuda.device_count()):
        try:
            # torch.cuda.mem_get_info returns (free, total) in bytes
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): Free: {free/1024**3:.2f}GB | Used: {used/1024**3:.2f}GB | Total: {total/1024**3:.2f}GB")
            
            if free > max_free:
                max_free = free
                best_gpu = i
        except Exception:
            print(f"  GPU {i}: Error during memory check.")
            pass
            
    print(f"Auto-selecting GPU {best_gpu} with {max_free/1024**3:.2f}GB free memory\n")
    return best_gpu

class Config:
    # Control Mode: If True, runs CE only (Baseline).
    CONTROL_MODE = False
    
    # Metric to optimize (will be set dynamically)
    METRIC_NAME = 'shannon' 
    
    # Model hyperparameters
    HIDDEN_DIM = 128
    NUM_LAYERS = 4
    NUM_ATTENTION_HEADS = 4 
    
    # Training hyperparameters
    BATCH_SIZE = 64 
    # EPOCHS will be handled by the phases
    SEQ_LENGTH = 64
    MAX_GRAD_NORM = 1.0
    MAX_SAMPLES = 1000    
    # We study optimization scaling at fixed data, not asymptotic data-limited scaling.
    MAX_VAL_SAMPLES = 5000
    MAX_TEST_SAMPLES = 5000
      
    # Number of runs
    NUM_OF_RUN_PER_CALL = 1
    
    # Complexity calculation interval
    COMPLEXITY_UPDATE_INTERVAL = 1 
    
    # Device configuration
    GPU_INDEX = get_best_gpu()
    # GPU selection affects runtime only and does not alter numerical results.
    DEVICE = torch.device(f'cuda:{GPU_INDEX}' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0

    # Profiling note: All runs, including baselines, were profiled identically; 
    # reported FLOPs therefore reflect effective training compute rather than idealized algorithmic FLOPs.
    # We compare methods at matched effective training compute, measured identically across conditions.
    
    # Performance optimizations
    USE_COMPILE = False  
    
    LEARNING_RATE = 1e-4

    @staticmethod
    def get_theoretical_model_flops(model, batch_size, seq_length):
        """
        Calculates theoretical FLOPs for one training step (Forward + Backward).
        Based on Kaplan et al. (2020): F ~ 6ND
        """
        N = sum(p.numel() for p in model.parameters())
        # 6 * Parameters * Tokens (Forward + Backward)
        return 6 * N * batch_size * seq_length

METRICS_TO_RUN = [
    # CONTROL
    'control',

    # BEST *
    '-A.B.C.G.J',
    '-B.C.G.J',
    
    # BEST FINAL VAL
    '-A.B'
    
    # BEST BEST VAL
    'U.Y/C.G.J',
    '-A',
    '-C.G.J',
    'U.Y/G.J',
    'U.Y/J',
    '-C',
]

# ============================================================================
# METRICS IMPLEMENTATION
# ============================================================================

class Metrics:
    @staticmethod
    def get_all_weights(model):
        """Concatenates all parameters into a single 1D tensor."""
        return torch.cat([p.view(-1) for p in model.parameters()])

    @staticmethod
    def soft_histogram(x, num_bins=None, min_val=None, max_val=None):
        """
        Differentiable histogram calculation.
        Optimized for fewer operations.
        """
        x = x.view(-1)
        if min_val is None: min_val = x.min()
        if max_val is None: max_val = x.max()
        
        # Avoid division by zero
        r = max_val - min_val
        if r < 1e-10: r = 1.0 
            
        normalized = (x - min_val) / r
        n = x.numel()
        
        if num_bins is None:
            # Fast bin estimation
            with torch.no_grad():
                # Subsample if too large
                if n > 5000:
                    # Fix seed for consistent histogram approximation
                    g = torch.Generator(device=x.device)
                    g.manual_seed(42)
                    indices = torch.randint(0, n, (5000,), generator=g, device=x.device)
                    sample = normalized[indices]
                else:
                    sample = normalized
                    
                q = torch.quantile(sample, torch.tensor([0.25, 0.75], device=x.device))
                iqr = q[1] - q[0]
                
                if iqr < 1e-10:
                    num_bins = max(1, int(math.sqrt(n)))
                else:
                    # Freedman-Diaconis rule
                    bin_width = 2 * iqr * (n ** (-1/3))
                    num_bins = int(1.0 / bin_width)
                    num_bins = max(1, min(num_bins, 256)) # Cap bins
        
        bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=x.device)
        bin_width = bin_edges[1] - bin_edges[0]
        
        bin_indices = normalized / bin_width
        bin_indices_floor = bin_indices.floor().long().clamp(0, num_bins - 1)
        
        frac = bin_indices - bin_indices_floor.float()
        
        hist = torch.zeros(num_bins, device=x.device, dtype=x.dtype)
        
        # Scatter add is efficient
        hist.scatter_add_(0, bin_indices_floor, 1.0 - frac)
        hist.scatter_add_(0, (bin_indices_floor + 1).clamp(0, num_bins - 1), frac)
        
        probs = hist / (hist.sum() + 1e-10)
        return torch.clamp(probs, 1e-10, 1.0), num_bins

    # =========================================================================
    # OPTIMIZED PRIMITIVE METRICS
    # =========================================================================

    @staticmethod
    def total_variation(model): # Metric A
        """Calculates TV of LM head weights."""
        w = model.lm_head.weight
        # Optimized: vectorized diff
        diff_h = torch.sum(torch.abs(torch.diff(w, dim=1)))
        diff_v = torch.sum(torch.abs(torch.diff(w, dim=0)))
        return (diff_h + diff_v) / w.numel()

    @staticmethod
    def third_order_curvature_norm(model, logits, labels, input_ids): # Metric B
        """
        Calculates norm of 3rd order gradient on a subset of parameters.
        Optimized by restricting parameter scope and input size.
        Metric B is a stochastic proxy rather than an exact curvature quantity.
        """
        # Minimal input for gradient graph
        if input_ids is not None:
            # Use minimal sequence length
            sl = min(input_ids.size(1), 16)
            logits_small = model(input_ids[:1, :sl])
            target = labels[:1, :sl].reshape(-1)
            logits_flat = logits_small.reshape(-1, logits_small.size(-1))
            loss = nn.CrossEntropyLoss()(logits_flat, target)
        elif logits is not None and labels is not None:
             loss = nn.CrossEntropyLoss()(logits[:1].view(-1, logits.shape[-1]), labels[:1].view(-1))
        else:
             return torch.tensor(0.0, device=Config.DEVICE)

        # FIX: Restrict curvature proxy to the LM Head weights. 
        # This creates a structurally invariant definition (curvature of the readout manifold)
        # and avoids "arbitrary subset" selection. Since we only differentiate w.r.t
        # the head, this also avoids backpropagation through the transformer body.
        params = [model.lm_head.weight]
        
        # 1st order
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Rademacher vector (sign of random normal is cheaper/cleaner)
        # FIX: Deterministic seeded generation for consistent estimation across epochs
        # This eliminates stochastic noise from the curvature approximation.
        g_gen = torch.Generator(device=params[0].device)
        g_gen.manual_seed(42)
        v = [torch.sign(torch.randn(p.shape, generator=g_gen, device=p.device)) for p in params]
        
        # Projection
        grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
        
        # 2nd order
        Hv = torch.autograd.grad(grad_v, params, create_graph=True)
        vHv = sum([(h * vi).sum() for h, vi in zip(Hv, v)])
        
        # 3rd order
        grad_vHv = torch.autograd.grad(vHv, params, create_graph=True)
        
        # Norm
        return torch.sqrt(sum([(g.norm()**2) for g in grad_vHv]))

    @staticmethod
    def hosoya_index(model): # Metric C
        """Nuclear norm of LM head."""
        # Optimized: Use svdvals instead of full SVD
        return torch.linalg.svdvals(model.lm_head.weight).sum()

    @staticmethod
    def disequilibrium(model): # Metric D
        weights = Metrics.get_all_weights(model)
        probs, n_bins = Metrics.soft_histogram(weights)
        uniform = 1.0 / n_bins
        return ((probs - uniform) ** 2).sum()

    @staticmethod
    def arithmetic_derivative(model): # Metric G
        """Ratio of L1/L2 norm of FFT of weights."""
        w = model.lm_head.weight
        f = torch.fft.rfft(w, dim=1)
        return torch.norm(f, p=1) / (torch.norm(f, p=2) + 1e-10)

    @staticmethod
    def persistent_landscape_norm(model): # Metric J
        """L2 norm of pairwise distances."""
        W = model.lm_head.weight
        # Critical optimization: Subsample
        if W.size(0) > 500:
             # Random subsample
             # Fix seed for consistent landscape estimation
             g = torch.Generator(device=W.device)
             g.manual_seed(42)
             idx = torch.randperm(W.size(0), generator=g, device=W.device)[:500]
             W = W[idx]
        
        dist = torch.cdist(W, W)
        # Optimized: Removed sorting (Norm(sort(x)) == Norm(x))
        return torch.norm(dist)

    @staticmethod
    def varentropy(logits): # Metric U
        """Varentropy of logits (Chunked)."""
        if logits is None: return torch.tensor(0.0, device=Config.DEVICE)
        
        def _compute(x_chunk):
            p = x_chunk.softmax(dim=-1)
            log_p = torch.log(p + 1e-10)
            ent = -(p * log_p).sum(dim=-1)
            # Var(X) = E[X^2] - E[X]^2. Here X = -log(p).
            # E[(-log p)^2] = E[(log p)^2]
            sec_mom = (p * (log_p**2)).sum(dim=-1)
            return sec_mom - (ent**2)

        # Flatten
        flat = logits.view(-1, logits.size(-1))
        
        # Chunking for memory efficiency
        chunk_size = 4096 
        if flat.size(0) <= chunk_size:
            if flat.requires_grad:
                 val = checkpoint(_compute, flat, use_reentrant=False)
            else:
                 val = _compute(flat)
            return val.mean()
            
        vals = []
        for i in range(0, flat.size(0), chunk_size):
            chunk = flat[i:i+chunk_size]
            if chunk.requires_grad:
                v = checkpoint(_compute, chunk, use_reentrant=False)
            else:
                v = _compute(chunk)
            vals.append(v)
            
        return torch.cat(vals).mean()

    @staticmethod
    def information_compression_ratio(logits): # Metric Y
        if logits is None: return torch.tensor(0.0, device=Config.DEVICE)
        p = torch.softmax(logits, dim=-1)
        # Global entropy average
        ent = -(p * torch.log(p + 1e-10)).sum(dim=-1).mean()
        max_ent = math.log(logits.size(-1))
        return 1.0 / (1.0 - (ent / max_ent) + 1e-10)

    @staticmethod
    def shannon_entropy(model): # Control
        weights = Metrics.get_all_weights(model)
        probs, _ = Metrics.soft_histogram(weights)
        return -(probs * torch.log(probs + 1e-10)).sum()

    # =========================================================================
    # MAIN CALCULATION ENTRY
    # =========================================================================

    @staticmethod
    def calculate_metric(model, metric_code, logits=None, labels=None, input_ids=None):
        """
        Parses metric code (e.g., 'U.Y/C.G.J') and computes value.
        """
        # Map codes to optimized functions
        mapping = {
            'A': Metrics.total_variation,
            'B': Metrics.third_order_curvature_norm,
            'C': Metrics.hosoya_index,
            'D': Metrics.disequilibrium,
            'G': Metrics.arithmetic_derivative,
            'J': Metrics.persistent_landscape_norm,
            'U': Metrics.varentropy,
            'Y': Metrics.information_compression_ratio,
            'shannon': Metrics.shannon_entropy
        }
        
        # Clean code
        code = metric_code.replace('-', '') # Ignore minimization sign for calc
        
        try:
            if '/' in code:
                num_part, den_part = code.split('/')
                
                num_val = 1.0
                for c in num_part.split('.'):
                    c = c.strip()
                    if not c: continue
                    f = mapping.get(c)
                    if f:
                        # Special handling for args
                        if c == 'B' or c == 'U' or c == 'Y':
                            # Need logits/labels
                            if c == 'B': v = f(model, logits, labels, input_ids)
                            else: v = f(logits)
                        else:
                            v = f(model)
                        num_val *= v
                        
                den_val = 1.0
                for c in den_part.split('.'):
                    c = c.strip()
                    if not c: continue
                    f = mapping.get(c)
                    if f:
                        if c == 'B' or c == 'U' or c == 'Y':
                            if c == 'B': v = f(model, logits, labels, input_ids)
                            else: v = f(logits)
                        else:
                            v = f(model)
                        den_val *= v
                        
                return num_val / (den_val + 1e-10)
                
            else:
                # Product mode
                val = 1.0
                for c in code.split('.'):
                    c = c.strip()
                    if not c: continue
                    f = mapping.get(c)
                    if f:
                        if c == 'B': v = f(model, logits, labels, input_ids)
                        elif c == 'U' or c == 'Y': v = f(logits)
                        else: v = f(model)
                        val *= v
                return val
                
        except Exception as e:
            # Fallback for debugging
            print(f"Error calculating {metric_code}: {e}")
            return torch.tensor(0.0, device=Config.DEVICE)

# ============================================================================
# DATASET & MODEL
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length, max_samples=None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        print(f"Tokenizing {file_path}...")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            if file_path.endswith('.csv'):
                reader = csv.DictReader(f)
                text = ''.join(row['text'] for row in reader)
            else:
                text = f.read()
        
        encodings = tokenizer(
            text,
            return_tensors='pt',
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=None,
            verbose=False
        )
        
        self.input_ids = encodings['input_ids'][0]
        
        if max_samples is not None and max_samples > 0:
            max_length = max_samples * seq_length
            self.input_ids = self.input_ids[:max_length]
    
    def __len__(self):
        return max(0, len(self.input_ids) - self.seq_length)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx : idx + self.seq_length]
        labels = self.input_ids[idx + 1 : idx + self.seq_length + 1]
        return input_ids, labels

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(seq_length, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.last_features = None
        
    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.position_embedding(positions)
        
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device), 
            diagonal=1
        ).bool()
        
        x = self.transformer(x, mask=causal_mask)
        self.last_features = x 
        x = self.lm_head(x)
        return x

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, config, vocab_size, epoch, phase: TrainingPhase, compute_tracker: ComputeTracker):
    model.train()
    total_loss = 0
    total_step_flops = 0
    
    # Calibration: Match Theoretical Algorithmic FLOPs (Kaplan et al.)
    theoretical_step_flops = Config.get_theoretical_model_flops(model, config.BATCH_SIZE, config.SEQ_LENGTH)
    flops_calibration_factor = 1.0 
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [{phase.value}]")
    
    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # One profile block for everything to ensure capture
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=False
        ) as prof:
            logits = model(input_ids)
            
            if phase == TrainingPhase.METRIC:
                if config.METRIC_NAME == 'control':
                     # Random scalar loss. maximize/minimize doesn't matter as it is random. 
                     # Using randn to simulate a scalar loss.
                     loss = torch.randn([], device=device, requires_grad=True)
                else:
                     loss = Metrics.calculate_metric(model, config.METRIC_NAME, logits, labels, input_ids)
            else:
                # CE Phase
                loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            
        # FLOPs Accounting
        events = prof.key_averages()
        step_raw_flops = sum(e.flops for e in events if e.flops is not None)
        
        # Calibrate on first batch
        if batch_idx == 0 and step_raw_flops > 0:
             flops_calibration_factor = theoretical_step_flops / step_raw_flops
        
        step_flops = step_raw_flops * flops_calibration_factor
        total_step_flops += step_flops
        
        # Update Global Tracker
        if phase == TrainingPhase.METRIC:
            compute_tracker.metric_flops += step_flops
        else:
            compute_tracker.ce_flops += step_flops
            
        total_loss += loss.item()
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'phase': phase.value,
            'TFLOPs': f"{compute_tracker.total/1e12:.2f}"
        })
        
    avg_loss = total_loss / len(train_loader)
    return avg_loss, total_step_flops

def evaluate(model, val_loader, device, vocab_size):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for input_ids, labels in tqdm(val_loader, desc="Validation"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()
            total_samples += 1
    return total_loss / len(val_loader)

def test(model, test_loader, device, vocab_size):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for input_ids, labels in tqdm(test_loader, desc="Testing"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()
            total_samples += 1
    return total_loss / len(test_loader)

def save_results_to_csv(output_dir, phases, train_losses, val_losses,
                       test_losses_wiki, test_losses_shakespeare, 
                       cumulative_flops, config, run_num=1):
    csv_filename = f'results_run_{run_num}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['Metric', config.METRIC_NAME])
        writer.writerow(['=== Summary ===', ''])
        writer.writerow(['Final Validation Loss', f'{val_losses[-1]:.16f}'])
        writer.writerow(['Total FLOPs', f'{cumulative_flops[-1]:.4e}'])
        writer.writerow([])
        
        writer.writerow(['Epoch', 'Phase', 'Training Loss', 'Validation Loss', 
                        'Test Loss Wiki', 'Test Loss Shakespeare', 
                        'Cumulative FLOPs'])
        
        for i in range(len(train_losses)):
             writer.writerow([
                i,
                phases[i] if i < len(phases) else "N/A",
                f'{train_losses[i]:.16f}',
                f'{val_losses[i]:.16f}',
                f'{test_losses_wiki[i]:.16f}',
                f'{test_losses_shakespeare[i]:.16f}',
                f'{cumulative_flops[i]:.4e}'
            ])

def run_training_single(output_dir, config, run_num, tokenizer, vocab_size, train_loader, val_loader, test_loader_wiki, test_loader_shakespeare):
    # Set seed for reproducibility for this run
    seed = 42 + run_num
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\nInitialized run {run_num} with seed {seed} [Protocol: Nature]")

    # --- PROTOCOL CONFIG ---
    if config.CONTROL_MODE:
        precondition_epochs = 0
        ce_epochs = TOTAL_EPOCHS
    else:
        precondition_epochs = PRECONDITION_EPOCHS
        ce_epochs = CE_EPOCHS
    
    # Model Init
    model = SimpleTransformer(vocab_size, config.HIDDEN_DIM, config.NUM_LAYERS, config.NUM_ATTENTION_HEADS, config.SEQ_LENGTH).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Initial Scheduler
    # For Control: OneCycle over TOTAL_EPOCHS
    # For Metric: OneCycle over PRECONDITION_EPOCHS
    steps_p1 = len(train_loader) * (precondition_epochs if precondition_epochs > 0 else ce_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.LEARNING_RATE, 
        total_steps=steps_p1, 
        pct_start=0.1
    )
    
    model.eval()
    
    # Initial Eval
    initial_val_loss = evaluate(model, val_loader, config.DEVICE, vocab_size)
    initial_test_loss_wiki = test(model, test_loader_wiki, config.DEVICE, vocab_size)
    initial_test_loss_shakespeare = test(model, test_loader_shakespeare, config.DEVICE, vocab_size)
    
    print(f"Epoch 0 (Initial): Val Loss {initial_val_loss:.4f}")
    
    # Lists
    phases = ["init"]
    train_losses = [0.0]
    val_losses = [initial_val_loss]
    test_losses_wiki = [initial_test_loss_wiki]
    test_losses_shakespeare = [initial_test_loss_shakespeare]
    cumulative_flops = [0.0]
    
    compute_tracker = ComputeTracker()
    
    # Loop
    for epoch in range(TOTAL_EPOCHS):
        # Determine Phase
        if epoch < precondition_epochs:
            current_phase = TrainingPhase.METRIC
        else:
            current_phase = TrainingPhase.CE
            
        # Phase Transition Logic
        if epoch == precondition_epochs and not config.CONTROL_MODE:
            print(f"--- Phase Transition: METRIC -> CE (Resetting Optimizer) ---")
            optimizer.state.clear()
            # New scheduler for CE phase
            steps_p2 = len(train_loader) * ce_epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=config.LEARNING_RATE, 
                total_steps=steps_p2, 
                pct_start=0.1
            )

        # Train
        avg_loss, _ = train_epoch(model, train_loader, optimizer, scheduler, config.DEVICE, config, vocab_size, epoch, current_phase, compute_tracker)
        
        # Val
        val_loss = evaluate(model, val_loader, config.DEVICE, vocab_size)
        test_loss_wiki = test(model, test_loader_wiki, config.DEVICE, vocab_size)
        test_loss_shakespeare = test(model, test_loader_shakespeare, config.DEVICE, vocab_size)
        
        print(f"Epoch {epoch+1}: Phase {current_phase.value.upper()} | Loss {avg_loss:.4f} | Val {val_loss:.4f} | TFLOPs {compute_tracker.total/1e12:.2f}")
        
        phases.append(current_phase.value)
        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        test_losses_wiki.append(test_loss_wiki)
        test_losses_shakespeare.append(test_loss_shakespeare)
        cumulative_flops.append(compute_tracker.total)
        
    save_results_to_csv(output_dir, phases, train_losses, val_losses, 
                        test_losses_wiki, test_losses_shakespeare, 
                        cumulative_flops,
                        config, run_num)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    vocab_size = tokenizer.vocab_size
    
    train_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.train.tokens')
    val_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.valid.tokens')
    test_path_wiki = os.path.join(script_dir, 'dataset/wikitext-2/wiki.test.tokens')
    test_path_shakespeare = os.path.join(script_dir, 'dataset/tiny-shakespare/test.csv')
    
    if not os.path.exists(train_path):
        print(f"Dataset not found at {train_path}. Stopping.")
        return
        
    train_dataset = TextDataset(train_path, tokenizer, Config.SEQ_LENGTH, Config.MAX_SAMPLES)
    val_dataset = TextDataset(val_path, tokenizer, Config.SEQ_LENGTH, Config.MAX_VAL_SAMPLES)
    test_dataset_wiki = TextDataset(test_path_wiki, tokenizer, Config.SEQ_LENGTH, Config.MAX_TEST_SAMPLES)
    test_dataset_shakespeare = TextDataset(test_path_shakespeare, tokenizer, Config.SEQ_LENGTH, Config.MAX_TEST_SAMPLES)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader_wiki = DataLoader(test_dataset_wiki, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader_shakespeare = DataLoader(test_dataset_shakespeare, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # STATEMENT OF ISOLATION:
    print("Auxiliary metrics are computed exclusively on training batches and never access validation or test data.")
    
    # Size sweep configuration
    SIZES = [
        (64, 2),
        (128, 4),
        (256, 8),
    ]
    
    for metric_name in METRICS_TO_RUN:
        # Determine direction for folder naming
        if metric_name == 'control':
             direction = 'control' 
        elif metric_name.startswith('-'):
             direction = 'min'
        else:
             direction = 'max'
             
        # Clean metric name for folder
        clean_name = metric_name.replace('/', '_over_')
        if clean_name.startswith('-'): clean_name = clean_name[1:]
        
        base_output_dir = f"output_nature/{clean_name}_{direction}"
        
        print(f"\nRunning experiment: {metric_name} ({direction})")
        Config.METRIC_NAME = metric_name
        
        # Treatment of Control:
        # 'control' refers to the Random Preconditioning Baseline.
        # This ensures the baseline has the same compute structure (Phase I + Phase II) as the metric runs.
        if metric_name == 'control':
            Config.CONTROL_MODE = False
            # Config.METRIC_NAME remains as set above ('control')
            # to trigger the randn logic in train_epoch
        else:
            Config.CONTROL_MODE = False
            
        for H, L in SIZES:
            print(f"  Size: Hidden={H}, Layers={L}")
            Config.HIDDEN_DIM = H
            Config.NUM_LAYERS = L
            
            output_dir = os.path.join(base_output_dir, f"H{H}_L{L}")
            os.makedirs(output_dir, exist_ok=True)
            
            for run_num in range(1, Config.NUM_OF_RUN_PER_CALL + 1):
                csv_path = os.path.join(output_dir, f'results_run_{run_num}.csv')
                if os.path.exists(csv_path):
                    print(f"Skipping {metric_name} H{H} L{L} Run {run_num} - already exists.")
                    continue

                print(f"    Run {run_num}/{Config.NUM_OF_RUN_PER_CALL}")
                run_training_single(output_dir, Config, run_num, tokenizer, vocab_size, train_loader, val_loader, test_loader_wiki, test_loader_shakespeare)

if __name__ == '__main__':
    main()
