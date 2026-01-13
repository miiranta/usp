import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import csv
import math
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

class Config:
    # Control Mode: If True, runs CE only (Baseline). If False, runs CE + Metric
    CONTROL_MODE = False  
    
    # Metric to optimize (will be set dynamically)
    METRIC_NAME = 'shannon' 
    
    # Model hyperparameters
    HIDDEN_DIM = 128
    NUM_LAYERS = 4
    NUM_ATTENTION_HEADS = 4 
    
    # Training hyperparameters
    BATCH_SIZE = 64 
    EPOCHS = 20
    SEQ_LENGTH = 64
    MAX_GRAD_NORM = 1.0
    MAX_SAMPLES = 100    
    MAX_VAL_SAMPLES = 500
    MAX_TEST_SAMPLES = 500    
    
    # Number of runs
    NUM_OF_RUN_PER_CALL = 2
    
    # Complexity calculation interval
    COMPLEXITY_UPDATE_INTERVAL = 1 
    
    # Device configuration
    GPU_INDEX = 1
    DEVICE = torch.device(f'cuda:{GPU_INDEX}' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0
    
    # Performance optimizations
    USE_COMPILE = False  
    
    # DONT CHANGE
    LMC_WEIGHT = 0.0         
    LEARNING_RATE = 1e-4

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
                    indices = torch.randint(0, n, (5000,), device=x.device)
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

        # Only use last 5 parameters that require grad to save massive compute
        params = [p for p in model.parameters() if p.requires_grad][-5:]
        
        # 1st order
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Rademacher vector (sign of random normal is cheaper/cleaner)
        v = [torch.sign(torch.randn_like(p)) for p in params]
        
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
             idx = torch.randperm(W.size(0), device=W.device)[:500]
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

def train_epoch(model, train_loader, optimizer, scheduler, device, config, vocab_size, metric_start, ce_start, epoch):
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_metric_loss = 0
    total_flops = 0
    total_model_flops = 0
    total_metric_flops = 0
    total_opt_flops = 0
    
    # 5-epoch schedule (Optimization Phase)
    if epoch < 5:
        lmc_weight = 1.0
    else:
        lmc_weight = 0.0
        
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
    
    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=False
        ) as prof:
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids)
            
            # CE Loss
            loss_fn = nn.CrossEntropyLoss()
            ce_loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            
            # Metric Loss
            # CRITICAL: Do not calculate metric if weight is 0 to save FLOPs for scaling law
            if config.CONTROL_MODE or lmc_weight == 0:
                metric_val = torch.tensor(0.0, device=device)
            else:
                metric_val = Metrics.calculate_metric(model, config.METRIC_NAME, logits, labels, input_ids)
            
            if config.METRIC_NAME.startswith('-'):
                metric_loss_normalized = (metric_val / (metric_start + 1e-10)) * ce_start
            else:
                metric_loss_normalized = (metric_start / (metric_val + 1e-10)) * ce_start
                
            if config.CONTROL_MODE:
                combined_loss = ce_loss
            else:
                combined_loss = (1.0 - lmc_weight) * ce_loss + lmc_weight * metric_loss_normalized
            
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
        
        # Calculate FLOPs
        events = prof.key_averages()
        step_flops = sum(e.flops for e in events if e.flops is not None)
        total_flops += step_flops
        
        step_model_flops = 0
        step_metric_flops = 0
        step_opt_flops = 0
        
        for e in events:
            if e.flops is None:
                continue
            name = e.key.lower()
            if "softmax" in name or "linear" in name or "matmul" in name:
                step_model_flops += e.flops
            elif "svd" in name or "fft" in name or "eig" in name or "checkpoint" in name:
                step_metric_flops += e.flops
            else:
                step_opt_flops += e.flops
                
        total_model_flops += step_model_flops
        total_metric_flops += step_metric_flops
        total_opt_flops += step_opt_flops
        
        total_loss += combined_loss.item()
        total_ce_loss += ce_loss.item()
        total_metric_loss += metric_val.item()
        
        progress_bar.set_postfix({
            'loss': f"{combined_loss.item():.4f}",
            'ce': f"{ce_loss.item():.4f}",
            'met': f"{metric_val.item():.4f}",
            'w': f"{lmc_weight:.1f}",
            'TFLOPs': f"{total_flops/1e12:.2f}"
        })
        
    return total_loss / len(train_loader), total_ce_loss / len(train_loader), total_metric_loss / len(train_loader), total_flops, total_model_flops, total_metric_flops, total_opt_flops

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

def save_results_to_csv(output_dir, train_losses, val_losses, metric_values, lmc_weight_values,
                       test_losses_wiki, test_losses_shakespeare, 
                       train_flops, train_model_flops, train_metric_flops, train_opt_flops,
                       config, run_num=1):
    csv_filename = f'results_run_{run_num}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header similar to train_min.py
        writer.writerow(['Metric', config.METRIC_NAME])
        writer.writerow(['=== WikiText-2 Test Results ===', ''])
        writer.writerow(['Test Loss (WikiText-2)', f'{test_losses_wiki[-1]:.16f}'])
        writer.writerow([])
        writer.writerow(['=== Tiny-Shakespeare Test Results ===', ''])
        writer.writerow(['Test Loss (Tiny-Shakespeare)', f'{test_losses_shakespeare[-1]:.16f}'])
        writer.writerow([])
        writer.writerow(['=== Training Summary ===', ''])
        writer.writerow(['Final Training Loss', f'{train_losses[-1]:.16f}'])
        writer.writerow(['Final Validation Loss', f'{val_losses[-1]:.16f}'])
        writer.writerow(['Final Metric Value', f'{metric_values[-1]:.16f}'])
        writer.writerow(['Total FLOPs', f'{sum(train_flops):.4e}'])
        writer.writerow(['Run Number', f'{run_num}'])
        writer.writerow([])
        
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 
                        'Test Loss Wiki', 'Test Loss Shakespeare', 
                        'Metric Value', 'Weight',
                        'Step FLOPs', 'Model FLOPs', 'Metric FLOPs', 'Opt FLOPs'])
        for epoch in range(len(train_losses)):
            # Handle potential length mismatch if FLOPs arrays are shorter (e.g. initial epoch 0 is fake)
            tf = train_flops[epoch] if epoch < len(train_flops) else 0.0
            tmod = train_model_flops[epoch] if epoch < len(train_model_flops) else 0.0
            tmet = train_metric_flops[epoch] if epoch < len(train_metric_flops) else 0.0
            topt = train_opt_flops[epoch] if epoch < len(train_opt_flops) else 0.0

            writer.writerow([
                epoch,
                f'{train_losses[epoch]:.16f}',
                f'{val_losses[epoch]:.16f}',
                f'{test_losses_wiki[epoch]:.16f}',
                f'{test_losses_shakespeare[epoch]:.16f}',
                f'{metric_values[epoch]:.16f}',
                f'{lmc_weight_values[epoch]:.3f}',
                f'{tf:.4e}',
                f'{tmod:.4e}',
                f'{tmet:.4e}',
                f'{topt:.4e}'
            ])

def run_training_single(output_dir, config, run_num, tokenizer, vocab_size, train_loader, val_loader, test_loader_wiki, test_loader_shakespeare):
    # Set seed for reproducibility for this run
    seed = 42 + run_num
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\nInitialized run {run_num} with seed {seed}")
    
    model = SimpleTransformer(vocab_size, config.HIDDEN_DIM, config.NUM_LAYERS, config.NUM_ATTENTION_HEADS, config.SEQ_LENGTH).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE, total_steps=total_steps, pct_start=0.1)
    
    model.eval()
    
    input_ids, labels = next(iter(train_loader))
    input_ids, labels = input_ids.to(config.DEVICE), labels.to(config.DEVICE)
    
    # Calculate metric start (enable grad as some metrics require it)
    if config.CONTROL_MODE:
        ce_start = 10.0 # Placeholder
        metric_start = 1.0 # Placeholder
        with torch.no_grad():
            logits = model(input_ids)
            ce_start = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), labels.view(-1)).item()
    else:
        with torch.enable_grad():
            logits = model(input_ids)
            metric_start = Metrics.calculate_metric(model, config.METRIC_NAME, logits, labels, input_ids).item()
            
        # Calculate CE start
        with torch.no_grad():
            ce_start = nn.CrossEntropyLoss()(logits.detach().view(-1, vocab_size), labels.view(-1)).item()
    
    print(f"Start CE: {ce_start:.4f}, Start Metric: {metric_start:.4f}")
    
    # Evaluate initial model (epoch 0)
    initial_val_loss = evaluate(model, val_loader, config.DEVICE, vocab_size)
    initial_test_loss_wiki = test(model, test_loader_wiki, config.DEVICE, vocab_size)
    initial_test_loss_shakespeare = test(model, test_loader_shakespeare, config.DEVICE, vocab_size)
    
    print(f"Epoch 0 (Initial): Train Loss {ce_start:.4f}, Val Loss {initial_val_loss:.4f}, Metric {metric_start:.4f}, Wiki Test {initial_test_loss_wiki:.4f}, Shake Test {initial_test_loss_shakespeare:.4f}")
    
    # Initialize lists with epoch 0 values
    train_losses = [ce_start]
    val_losses = [initial_val_loss]
    metric_values = [metric_start]
    lmc_weight_values = [1.0]  # No training weight at epoch 0
    test_losses_wiki = [initial_test_loss_wiki]
    test_losses_shakespeare = [initial_test_loss_shakespeare]
    
    train_flops = [0.0]
    train_model_flops = [0.0]
    train_metric_flops = [0.0]
    train_opt_flops = [0.0]
    
    for epoch in range(config.EPOCHS):
        train_loss, train_ce, train_metric, t_flops, t_model, t_metric, t_opt = train_epoch(model, train_loader, optimizer, scheduler, config.DEVICE, config, vocab_size, metric_start, ce_start, epoch)
        val_loss = evaluate(model, val_loader, config.DEVICE, vocab_size)
        
        # Run tests
        test_loss_wiki = test(model, test_loader_wiki, config.DEVICE, vocab_size)
        test_loss_shakespeare = test(model, test_loader_shakespeare, config.DEVICE, vocab_size)
        
        # Values for logging
        # (Note: train_epoch here hardcodes weight schedule, so we replicate it for logging)
        if epoch < 5:
            lmc_weight = 1.0
        else:
            lmc_weight = 0.0
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Metric {train_metric:.4f}, Wiki Test {test_loss_wiki:.4f}, Shake Test {test_loss_shakespeare:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metric_values.append(train_metric)
        lmc_weight_values.append(lmc_weight)
        test_losses_wiki.append(test_loss_wiki)
        test_losses_shakespeare.append(test_loss_shakespeare)

        train_flops.append(t_flops)
        train_model_flops.append(t_model)
        train_metric_flops.append(t_metric)
        train_opt_flops.append(t_opt)
        
    save_results_to_csv(output_dir, train_losses, val_losses, metric_values, lmc_weight_values, 
                        test_losses_wiki, test_losses_shakespeare, 
                        train_flops, train_model_flops, train_metric_flops, train_opt_flops,
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
    
    # Size sweep configuration
    SIZES = [
        (64, 2),
        (128, 4),
        (256, 8),
    ]
    
    for metric_name in METRICS_TO_RUN:
        # Determine direction for folder naming
        if metric_name == 'control':
             direction = 'min' 
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
        
        if metric_name == 'control':
            Config.CONTROL_MODE = True
            Config.METRIC_NAME = 'shannon' 
        else:
            Config.CONTROL_MODE = False
            
        for H, L in SIZES:
            print(f"  Size: Hidden={H}, Layers={L}")
            Config.HIDDEN_DIM = H
            Config.NUM_LAYERS = L
            
            output_dir = os.path.join(base_output_dir, f"H{H}_L{L}")
            if os.path.exists(output_dir):
                print(f"Skipping {metric_name} ({direction}) H{H} L{L} - already exists.")
                continue
            
            os.makedirs(output_dir, exist_ok=True)
            
            for run_num in range(1, Config.NUM_OF_RUN_PER_CALL + 1):
                print(f"    Run {run_num}/{Config.NUM_OF_RUN_PER_CALL}")
                run_training_single(output_dir, Config, run_num, tokenizer, vocab_size, train_loader, val_loader, test_loader_wiki, test_loader_shakespeare)

if __name__ == '__main__':
    main()
