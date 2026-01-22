"""
Language Model Training with Metric-Based Preconditioning

This code trains transformer models using a two-phase approach:
1. Metric optimization phase (optional preconditioning)
2. Cross-entropy training phase

Null hypothesis: Metric preconditioning does not change the intercept or slope of CE scaling laws at fixed compute.
"""

import os
import csv
import math
import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader

# Configure PyTorch memory and attention
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except AttributeError:
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Central configuration for all training parameters"""
    
    # Training phases
    PRECONDITION_EPOCHS: int = 5
    CE_EPOCHS: int = 15
    
    # Model architecture
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 4
    NUM_ATTENTION_HEADS: int = 4
    SEQ_LENGTH: int = 64
    
    # Training hyperparameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-4
    MAX_GRAD_NORM: float = 1.0
    
    # Data limits
    MAX_SAMPLES: int = 1000
    MAX_VAL_SAMPLES: int = 5000
    MAX_TEST_SAMPLES: int = 5000
    
    # Experiment settings
    NUM_RUNS: int = 3  # Nature requirement: >= 3 seeds for variance
    METRIC_NAME: str = 'shannon'
    CONTROL_MODE: bool = False
    
    # Hard Protocol Requirements
    TARGET_FLOPS: float = float('inf')  # Limit for compute-matched controls
    
    # Hardware
    NUM_WORKERS: int = 0
    
    @property
    def TOTAL_EPOCHS(self) -> int:
        return self.PRECONDITION_EPOCHS + self.CE_EPOCHS
    
    @property
    def DEVICE(self) -> torch.device:
        gpu_idx = self._get_best_gpu()
        return torch.device(f'cuda:{gpu_idx}' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def _get_best_gpu() -> int:
        """Select GPU with most free memory"""
        if not torch.cuda.is_available():
            print("No CUDA devices available. Using CPU.")
            return 0
        
        print(f"\nScanning {torch.cuda.device_count()} available GPUs:")
        best_gpu, max_free = 0, 0
        
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): "
                      f"Free: {free/1024**3:.2f}GB | "
                      f"Used: {used/1024**3:.2f}GB | "
                      f"Total: {total/1024**3:.2f}GB")
                
                if free > max_free:
                    max_free, best_gpu = free, i
            except Exception:
                print(f"  GPU {i}: Error during memory check.")
        
        print(f"Auto-selecting GPU {best_gpu} with {max_free/1024**3:.2f}GB free\n")
        return best_gpu
    
    @staticmethod
    def get_theoretical_flops(model: nn.Module, batch_size: int, seq_length: int) -> float:
        """Calculate theoretical FLOPs (Kaplan et al. 2020): F ~ 6ND"""
        num_params = sum(p.numel() for p in model.parameters())
        return 6 * num_params * batch_size * seq_length


class TrainingPhase(Enum):
    """Training phase enumeration"""
    METRIC = "metric"
    CE = "ce"


class ComputeTracker:
    """Track computational cost across training phases"""
    
    def __init__(self):
        self.metric_flops = 0.0
        self.ce_flops = 0.0
    
    @property
    def total(self) -> float:
        return self.metric_flops + self.ce_flops


# ============================================================================
# METRICS
# ============================================================================

class Metrics:
    """Collection of differentiable complexity metrics"""
    
    @staticmethod
    def get_all_weights(model: nn.Module) -> torch.Tensor:
        """Concatenate all model parameters into single tensor"""
        return torch.cat([p.view(-1) for p in model.parameters()])
    
    @staticmethod
    def soft_histogram(x: torch.Tensor, 
                      num_bins: Optional[int] = None,
                      min_val: Optional[float] = None,
                      max_val: Optional[float] = None) -> Tuple[torch.Tensor, int]:
        """Differentiable histogram using linear interpolation"""
        x = x.view(-1)
        n = x.numel()
        
        if min_val is None:
            min_val = x.min()
        if max_val is None:
            max_val = x.max()
        
        # Normalize to [0, 1]
        range_val = max_val - min_val
        if range_val < 1e-10:
            range_val = 1.0
        normalized = (x - min_val) / range_val
        
        # Auto-determine bins using Freedman-Diaconis rule
        if num_bins is None:
            with torch.no_grad():
                # Subsample for efficiency
                sample = normalized
                if n > 5000:
                    g = torch.Generator(device=x.device).manual_seed(42)
                    idx = torch.randint(0, n, (5000,), generator=g, device=x.device)
                    sample = normalized[idx]
                
                q = torch.quantile(sample, torch.tensor([0.25, 0.75], device=x.device))
                iqr = q[1] - q[0]
                
                if iqr < 1e-10:
                    num_bins = max(1, int(math.sqrt(n)))
                else:
                    bin_width = 2 * iqr * (n ** (-1/3))
                    num_bins = max(1, min(int(1.0 / bin_width), 256))
        
        # Create bins and compute soft assignment
        bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=x.device)
        bin_width = bin_edges[1] - bin_edges[0]
        
        bin_indices = normalized / bin_width
        bin_floor = bin_indices.floor().long().clamp(0, num_bins - 1)
        frac = bin_indices - bin_floor.float()
        
        # Soft histogram via scatter_add
        hist = torch.zeros(num_bins, device=x.device, dtype=x.dtype)
        hist.scatter_add_(0, bin_floor, 1.0 - frac)
        hist.scatter_add_(0, (bin_floor + 1).clamp(0, num_bins - 1), frac)
        
        # Normalize to probabilities
        probs = hist / (hist.sum() + 1e-10)
        return torch.clamp(probs, 1e-10, 1.0), num_bins
    
    # Primitive Metrics
    
    @staticmethod
    def total_variation(model: nn.Module) -> torch.Tensor:
        """Total variation of LM head weights (Metric A)"""
        w = model.lm_head.weight
        diff_h = torch.sum(torch.abs(torch.diff(w, dim=1)))
        diff_v = torch.sum(torch.abs(torch.diff(w, dim=0)))
        return (diff_h + diff_v) / w.numel()
    
    @staticmethod
    def third_order_curvature_norm(model: nn.Module,
                                   logits: Optional[torch.Tensor],
                                   labels: Optional[torch.Tensor],
                                   input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Third-order curvature proxy (Metric B)"""
        # Compute loss on minimal input
        if input_ids is not None:
            # Protocol 3: Use random tokens to prevent input leakage
            sl = min(input_ids.size(1), 16)
            vocab_size = model.lm_head.out_features
            rand_input_ids = torch.randint(0, vocab_size, (1, sl), device=input_ids.device)
            logits_small = model(rand_input_ids)
            
            # A3: Use random labels to prevent leakage
            target = torch.randint(0, vocab_size, (sl,), device=logits_small.device)
            loss = nn.CrossEntropyLoss()(logits_small.reshape(-1, vocab_size), target)
        elif logits is not None and labels is not None:
            # A3: Use random labels to prevent leakage
            vocab_size = logits.size(-1)
            target = torch.randint(0, vocab_size, labels[:1].shape, device=logits.device).view(-1)
            loss = nn.CrossEntropyLoss()(logits[:1].view(-1, logits.shape[-1]), 
                                        target)
        else:
            return torch.tensor(0.0, device=model.lm_head.weight.device)
        
        # Compute curvature on LM head only
        params = [model.lm_head.weight]
        
        # First-order gradient
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Random projection vector (seeded for consistency)
        g = torch.Generator(device=params[0].device).manual_seed(42)
        v = [torch.sign(torch.randn(p.shape, generator=g, device=p.device)) 
             for p in params]
        
        # Second-order: Hessian-vector product
        grad_v = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(grad_v, params, create_graph=True)
        vHv = sum((h * vi).sum() for h, vi in zip(Hv, v))
        
        # Third-order gradient
        grad_vHv = torch.autograd.grad(vHv, params, create_graph=True)
        
        return torch.sqrt(sum(g.norm()**2 for g in grad_vHv))
    
    @staticmethod
    def hosoya_index(model: nn.Module) -> torch.Tensor:
        """Nuclear norm of LM head (Metric C)"""
        return torch.linalg.svdvals(model.lm_head.weight).sum()
    
    @staticmethod
    def disequilibrium(model: nn.Module) -> torch.Tensor:
        """Statistical disequilibrium (Metric D)"""
        weights = Metrics.get_all_weights(model)
        probs, n_bins = Metrics.soft_histogram(weights)
        uniform = 1.0 / n_bins
        return ((probs - uniform) ** 2).sum()
    
    @staticmethod
    def arithmetic_derivative(model: nn.Module) -> torch.Tensor:
        """FFT-based arithmetic derivative (Metric G)"""
        w = model.lm_head.weight
        f = torch.fft.rfft(w, dim=1)
        return torch.norm(f, p=1) / (torch.norm(f, p=2) + 1e-10)
    
    @staticmethod
    def persistent_landscape_norm(model: nn.Module) -> torch.Tensor:
        """Topological persistence approximation (Metric J)"""
        W = model.lm_head.weight
        
        # Subsample for efficiency
        if W.size(0) > 500:
            g = torch.Generator(device=W.device).manual_seed(42)
            idx = torch.randperm(W.size(0), generator=g, device=W.device)[:500]
            W = W[idx]
        
        dist = torch.cdist(W, W)
        return torch.norm(dist)
    
    @staticmethod
    def varentropy(logits: torch.Tensor) -> torch.Tensor:
        """Variance of entropy (Metric U)"""
        if logits is None:
            return torch.tensor(0.0)
        
        def compute_chunk(x):
            p = x.softmax(dim=-1)
            log_p = torch.log(p + 1e-10)
            ent = -(p * log_p).sum(dim=-1)
            sec_moment = (p * (log_p**2)).sum(dim=-1)
            return sec_moment - (ent**2)
        
        flat = logits.view(-1, logits.size(-1))
        chunk_size = 4096
        
        if flat.size(0) <= chunk_size:
            return compute_chunk(flat).mean()
        
        # Process in chunks
        vals = []
        for i in range(0, flat.size(0), chunk_size):
            chunk = flat[i:i+chunk_size]
            vals.append(compute_chunk(chunk))
        
        return torch.cat(vals).mean()
    
    @staticmethod
    def information_compression_ratio(logits: torch.Tensor) -> torch.Tensor:
        """Information compression ratio (Metric Y)"""
        if logits is None:
            return torch.tensor(0.0)
        
        p = torch.softmax(logits, dim=-1)
        ent = -(p * torch.log(p + 1e-10)).sum(dim=-1).mean()
        max_ent = math.log(logits.size(-1))
        return 1.0 / (1.0 - (ent / max_ent) + 1e-10)
    
    @staticmethod
    def shannon_entropy(model: nn.Module) -> torch.Tensor:
        """Shannon entropy of weight distribution"""
        weights = Metrics.get_all_weights(model)
        probs, _ = Metrics.soft_histogram(weights)
        return -(probs * torch.log(probs + 1e-10)).sum()
    
    @classmethod
    def calculate_metric(cls, model: nn.Module, 
                        metric_code: str,
                        logits: Optional[torch.Tensor] = None,
                        labels: Optional[torch.Tensor] = None,
                        input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Parse and compute metric from code (e.g., 'U.Y/C.G.J')"""
        
        metric_map = {
            'A': cls.total_variation,
            'B': cls.third_order_curvature_norm,
            'C': cls.hosoya_index,
            'D': cls.disequilibrium,
            'G': cls.arithmetic_derivative,
            'J': cls.persistent_landscape_norm,
            'U': cls.varentropy,
            'Y': cls.information_compression_ratio,
            'shannon': cls.shannon_entropy
        }
        
        # Determine optimization direction: '-' prefix means minimize
        sign = -1.0 if metric_code.startswith('-') else 1.0
        code = metric_code.replace('-', '')
        
        def eval_metric(metric_id: str) -> torch.Tensor:
            """Helper to evaluate single metric"""
            func = metric_map.get(metric_id.strip())
            if not func:
                return torch.tensor(1.0)
            
            # Metrics requiring logits/labels
            if metric_id in ['B']:
                return func(model, logits, labels, input_ids)
            elif metric_id in ['U', 'Y']:
                return func(logits)
            else:
                return func(model)
        
        try:
            # Handle division
            if '/' in code:
                num_part, den_part = code.split('/')
                
                num_val = 1.0
                for metric_id in num_part.split('.'):
                    if metric_id.strip():
                        num_val *= eval_metric(metric_id)
                
                den_val = 1.0
                for metric_id in den_part.split('.'):
                    if metric_id.strip():
                        den_val *= eval_metric(metric_id)
                
                return sign * (num_val / (den_val + 1e-10))
            
            # Handle product
            else:
                val = 1.0
                for metric_id in code.split('.'):
                    if metric_id.strip():
                        val *= eval_metric(metric_id)
                return sign * val
        
        except Exception as e:
            print(f"Error calculating {metric_code}: {e}")
            return torch.tensor(0.0)


# ============================================================================
# DATASET & MODEL
# ============================================================================

class TextDataset(Dataset):
    """Token-based language modeling dataset"""
    
    def __init__(self, file_path: str, tokenizer, seq_length: int, 
                 max_samples: Optional[int] = None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        print(f"Tokenizing {file_path}...")
        
        # Read text
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            if file_path.endswith('.csv'):
                reader = csv.DictReader(f)
                text = ''.join(row['text'] for row in reader)
            else:
                text = f.read()
        
        # Tokenize
        encodings = tokenizer(
            text,
            return_tensors='pt',
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False
        )
        
        self.input_ids = encodings['input_ids'][0]
        
        # Limit samples
        if max_samples is not None and max_samples > 0:
            max_length = max_samples * seq_length
            self.input_ids = self.input_ids[:max_length]
    
    def __len__(self) -> int:
        return max(0, len(self.input_ids) - self.seq_length)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.input_ids[idx : idx + self.seq_length]
        labels = self.input_ids[idx + 1 : idx + self.seq_length + 1]
        return input_ids, labels


class SimpleTransformer(nn.Module):
    """Causal transformer language model"""
    
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, 
                 num_heads: int, seq_length: int):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        
        # Embed tokens and positions
        x = self.embedding(x) + self.position_embedding(positions)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device),
            diagonal=1
        ).bool()
        
        # Transform and project
        x = self.transformer(x, mask=causal_mask)
        self.last_features = x
        return self.lm_head(x)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler,
                device: torch.device,
                config: TrainingConfig,
                vocab_size: int,
                epoch: int,
                phase: TrainingPhase,
                compute_tracker: ComputeTracker,
                ema_tracker: Optional[dict] = None,
                control_tensor: Optional[torch.Tensor] = None) -> Tuple[float, float, bool]:
    """Train for one epoch. Returns (avg_loss, epoch_flops, should_stop)"""
    
    model.train()
    total_loss = 0.0
    total_flops = 0.0
    
    progress_bar = tqdm(train_loader, 
                       desc=f"Epoch {epoch+1}/{config.TOTAL_EPOCHS} [{phase.value}]")
    
    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        # 1. Check FLOP limit (Hard Requirement A.1)
        if config.TARGET_FLOPS != float('inf') and compute_tracker.total >= config.TARGET_FLOPS:
            # Return current stats and signal stop
            return total_loss / max(1, batch_idx), total_flops, True

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # A2: Analytic FLOPs (Nature standard)
        step_flops = config.get_theoretical_flops(
            model, input_ids.size(0), input_ids.size(1)
        )
        
        optimizer.zero_grad()
        
        logits = model(input_ids)
        
        # Compute loss based on phase
        if phase == TrainingPhase.METRIC:
            if config.METRIC_NAME == 'control':
                # B3: Frozen random projection
                if control_tensor is None:
                    loss = torch.randn([], device=device, requires_grad=True)
                else:
                    # Bug 3: Ensure control tensor is on correct device
                    control_tensor = control_tensor.to(device)
                    loss = (model.lm_head.weight * control_tensor).sum()
            elif config.METRIC_NAME == 'control_reset':
                # CE-based Preconditioning (Optimization Reset Control)
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, vocab_size),
                    labels.view(-1)
                )
            else:
                raw_loss = Metrics.calculate_metric(
                    model, config.METRIC_NAME, logits, labels, input_ids
                )
                # A4: EMA Normalization
                if ema_tracker is None:
                     loss = raw_loss / (raw_loss.detach().abs() + 1e-6)
                else:
                     val = raw_loss.detach().abs()
                     if 'metric' not in ema_tracker:
                         ema_tracker['metric'] = val
                     
                     ema_tracker['metric'] = 0.99 * ema_tracker['metric'] + 0.01 * val
                     loss = raw_loss / (ema_tracker['metric'] + 1e-6)
        else:
            # Standard cross-entropy
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, vocab_size),
                labels.view(-1)
            )
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
    
        total_flops += step_flops
        
        # Update tracker
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
    
    return total_loss / len(train_loader), total_flops, False


def evaluate(model: nn.Module,
            data_loader: DataLoader,
            device: torch.device,
            vocab_size: int,
            desc: str = "Evaluation") -> float:
    """Evaluate model on dataset"""
    
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for input_ids, labels in tqdm(data_loader, desc=desc):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids)
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, vocab_size),
                labels.view(-1)
            )
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


def save_results(output_dir: str,
                phases: List[str],
                train_losses: List[float],
                val_losses: List[float],
                test_losses_wiki: List[float],
                test_losses_shakespeare: List[float],
                compute_stats: dict,
                config: TrainingConfig,
                run_num: int):
    """Save training results to CSV (Standard Format)"""
    
    csv_path = os.path.join(output_dir, f'results_run_{run_num}.csv')
    
    cumulative_flops = compute_stats['total']
    ce_flops = compute_stats['ce']
    metric_flops = compute_stats['metric']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Protocol 2: Flag scaling axis
        writer.writerow(['Scaling Axis', 'CE_FLOPs'])
        
        # Summary
        writer.writerow(['Metric', config.METRIC_NAME])
        writer.writerow(['=== Summary ===', ''])
        writer.writerow(['Final Validation Loss', f'{val_losses[-1]:.16f}'])
        writer.writerow(['Total FLOPs', f'{cumulative_flops[-1]:.4e}'])
        writer.writerow([])
        
        # Headers (Expanded for Nature Analysis)
        writer.writerow(['Epoch', 'Phase', 'Training Loss', 'Validation Loss',
                        'Test Loss Wiki', 'Test Loss Shakespeare', 
                        'Cumulative FLOPs', 'CE FLOPs', 'Metric FLOPs'])
        
        # Data
        for i in range(len(train_losses)):
            writer.writerow([
                i,
                phases[i] if i < len(phases) else "N/A",
                f'{train_losses[i]:.16f}',
                f'{val_losses[i]:.16f}',
                f'{test_losses_wiki[i]:.16f}',
                f'{test_losses_shakespeare[i]:.16f}',
                f'{cumulative_flops[i]:.4e}',
                f'{ce_flops[i]:.4e}',
                f'{metric_flops[i]:.4e}'
            ])


def run_single_experiment(output_dir: str,
                         config: TrainingConfig,
                         run_num: int,
                         tokenizer,
                         vocab_size: int,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         test_loader_wiki: DataLoader,
                         test_loader_shakespeare: DataLoader) -> float:
    """Execute single training run. Returns total FLOPs used."""
    
    # Set seed for reproducibility
    seed = 42 + run_num
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\nRun {run_num} initialized with seed {seed}")
    
    # Determine phase lengths
    if config.CONTROL_MODE:
        precondition_epochs = 0
        ce_epochs = config.TOTAL_EPOCHS
        
        # Extended run for compute-matched control
        if config.TARGET_FLOPS != float('inf'):
             ce_epochs = 100 # Ensure we have enough epochs to hit the FLOP limit
    else:
        precondition_epochs = config.PRECONDITION_EPOCHS
        ce_epochs = config.CE_EPOCHS
    
    # Override for control_reset: Use explicit phase structure but with CE loss
    if config.METRIC_NAME == 'control_reset':
        precondition_epochs = 5
        ce_epochs = 15
    
    # Initialize model
    model = SimpleTransformer(
        vocab_size,
        config.HIDDEN_DIM,
        config.NUM_LAYERS,
        config.NUM_ATTENTION_HEADS,
        config.SEQ_LENGTH
    ).to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Protocol 1: Use ConstantLR everywhere for clean scaling comparison (Nature-safe)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    # B3: Frozen control tensor
    control_tensor = None
    if config.METRIC_NAME == 'control':
        g = torch.Generator(device=config.DEVICE).manual_seed(seed)
        control_tensor = torch.randn(model.lm_head.weight.shape, generator=g, device=config.DEVICE)
    
    # Initial evaluation
    model.eval()
    initial_val = evaluate(model, val_loader, config.DEVICE, vocab_size, "Initial Val")
    initial_wiki = evaluate(model, test_loader_wiki, config.DEVICE, vocab_size, "Initial Wiki")
    initial_shakespeare = evaluate(model, test_loader_shakespeare, config.DEVICE, vocab_size, "Initial Shakespeare")
    
    print(f"Epoch 0 (Initial): Val Loss {initial_val:.4f}")
    
    # Tracking lists
    phases = ["init"]
    train_losses = [0.0]
    val_losses = [initial_val]
    test_losses_wiki = [initial_wiki]
    test_losses_shakespeare = [initial_shakespeare]
    
    # Precise FLOP tracking for scaling laws
    cumulative_flops = [0.0]
    ce_flops_list = [0.0]
    metric_flops_list = [0.0]
    
    compute_tracker = ComputeTracker()
    ema_tracker = {}
    
    # Training loop
    # If targeting FLOPs (matched control), loop can go longer than Total Epochs
    epoch_limit = 100 if config.TARGET_FLOPS != float('inf') else config.TOTAL_EPOCHS
    
    for epoch in range(epoch_limit):
        # Determine current phase
        if epoch < precondition_epochs:
            current_phase = TrainingPhase.METRIC
        else:
            current_phase = TrainingPhase.CE
        
        # Handle phase transition (Hard Requirement A.2)
        # Reset if it's the transition epoch, OR if it's the specific "control_reset" baseline
        if epoch == precondition_epochs:
            is_transition = not config.CONTROL_MODE 
            is_reset_baseline = (config.METRIC_NAME == 'control_reset')
            
            if is_transition or is_reset_baseline:
                print("--- Phase Transition: Resetting Optimizer ---")
                optimizer.state.clear()
                
                # Protocol 1: Maintain ConstantLR in CE phase
                scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        
        # Train
        # Bug 1: Correct argument passing
        avg_loss, _, should_stop = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            config=config,
            vocab_size=vocab_size,
            epoch=epoch,
            phase=current_phase,
            compute_tracker=compute_tracker,
            ema_tracker=ema_tracker,
            control_tensor=control_tensor
        )

        # Evaluate
        val_loss = evaluate(model, val_loader, config.DEVICE, vocab_size, "Validation")
        wiki_loss = evaluate(model, test_loader_wiki, config.DEVICE, vocab_size, "Test Wiki")
        shakespeare_loss = evaluate(model, test_loader_shakespeare, config.DEVICE, vocab_size, "Test Shakespeare")
        
        print(f"Epoch {epoch+1}: Phase {current_phase.value.upper()} | "
              f"Loss {avg_loss:.4f} | Val {val_loss:.4f} | "
              f"TFLOPs {compute_tracker.total/1e12:.2f}")

        phases.append(current_phase.value)
        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        test_losses_wiki.append(wiki_loss)
        test_losses_shakespeare.append(shakespeare_loss)
        
        cumulative_flops.append(compute_tracker.total)
        ce_flops_list.append(compute_tracker.ce_flops)
        metric_flops_list.append(compute_tracker.metric_flops)
        
        if should_stop:
            print(f"Stopping training: Target FLOPs ({config.TARGET_FLOPS:.2e}) reached.")
            break
    
    # Verify CE FLOPs were recorded for scaling law analysis
    assert compute_tracker.ce_flops > 0, "No CE FLOPs recorded - scaling law analysis requires CE phase"
    
    # Save results
    compute_stats = {
        'total': cumulative_flops,
        'ce': ce_flops_list,
        'metric': metric_flops_list
    }
    
    save_results(
        output_dir, phases, train_losses, val_losses,
        test_losses_wiki, test_losses_shakespeare,
        compute_stats, config, run_num
    )
    
    return compute_tracker.total


# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

def main():
    """Main experimental pipeline"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    vocab_size = tokenizer.vocab_size
    
    # Dataset paths
    train_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.train.tokens')
    val_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.valid.tokens')
    test_wiki_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.test.tokens')
    test_shakespeare_path = os.path.join(script_dir, 'dataset/tiny-shakespare/test.csv')
    
    if not os.path.exists(train_path):
        print(f"Dataset not found at {train_path}")
        return
    
    # Create base config
    base_config = TrainingConfig()
    
    # Metrics list fixed prior to experiments (no post-hoc pruning)
    # Create datasets
    train_dataset = TextDataset(train_path, tokenizer, base_config.SEQ_LENGTH, 
                                base_config.MAX_SAMPLES)
    val_dataset = TextDataset(val_path, tokenizer, base_config.SEQ_LENGTH,
                              base_config.MAX_VAL_SAMPLES)
    test_dataset_wiki = TextDataset(test_wiki_path, tokenizer, base_config.SEQ_LENGTH,
                                    base_config.MAX_TEST_SAMPLES)
    test_dataset_shakespeare = TextDataset(test_shakespeare_path, tokenizer, 
                                          base_config.SEQ_LENGTH,
                                          base_config.MAX_TEST_SAMPLES)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=base_config.BATCH_SIZE,
                             shuffle=True, num_workers=base_config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=base_config.BATCH_SIZE,
                           shuffle=False, num_workers=base_config.NUM_WORKERS)
    test_loader_wiki = DataLoader(test_dataset_wiki, batch_size=base_config.BATCH_SIZE,
                                 shuffle=False, num_workers=base_config.NUM_WORKERS)
    test_loader_shakespeare = DataLoader(test_dataset_shakespeare, 
                                        batch_size=base_config.BATCH_SIZE,
                                        shuffle=False, num_workers=base_config.NUM_WORKERS)
    
    print("Auxiliary metrics computed exclusively on training batches only.")
    
    # Valid metric configs
    metrics_to_run = [
        '-A.B.C.G.J',
        '-B.C.G.J',
        '-A.B',
        'U.Y/C.G.J',
        '-A',
        '-C.G.J',
        'U.Y/G.J',
        'U.Y/J',
        '-C',
    ]
    
    # Model size configurations
    model_sizes = [
        (64, 2),   # (hidden_dim, num_layers)
        (128, 4),
        (256, 8),
    ]
    
    # Run experiments
    for metric_name in metrics_to_run:
        # Determine optimization direction
        if metric_name == 'control':
            continue # Control is now handled as an automated baseline for each metric
        elif metric_name.startswith('-'):
            direction = 'min'
        else:
            direction = 'max'
        
        # Clean metric name for directory
        clean_name = metric_name.replace('/', '_over_')
        if clean_name.startswith('-'):
            clean_name = clean_name[1:]
        
        base_output_dir = f"output_nature/{clean_name}_{direction}"
        
        print(f"\n{'='*60}")
        print(f"Experiment: {metric_name} ({direction})")
        print(f"{'='*60}")
        
        # Configure experiment
        config = TrainingConfig()
        config.METRIC_NAME = metric_name
        config.CONTROL_MODE = False
        
        # Run across model sizes
        for hidden_dim, num_layers in model_sizes:
            print(f"\n  Model: Hidden={hidden_dim}, Layers={num_layers}")
            
            config.HIDDEN_DIM = hidden_dim
            config.NUM_LAYERS = num_layers
            
            # Setup directories
            output_dir_metric = os.path.join(base_output_dir, f"H{hidden_dim}_L{num_layers}")
            os.makedirs(output_dir_metric, exist_ok=True)
            
            output_dir_control_matched = os.path.join(f"{base_output_dir}_control_matched", f"H{hidden_dim}_L{num_layers}")
            os.makedirs(output_dir_control_matched, exist_ok=True)
            
            output_dir_control_reset = os.path.join(f"{base_output_dir}_control_reset", f"H{hidden_dim}_L{num_layers}")
            os.makedirs(output_dir_control_reset, exist_ok=True)
            
            # Run multiple trials
            for run_num in range(1, config.NUM_RUNS + 1):
                # 1. Metric Run
                # ----------------
                print(f"    [Metric] Run {run_num}/{config.NUM_RUNS}")
                metric_flops = run_single_experiment(
                    output_dir_metric, config, run_num, tokenizer, vocab_size,
                    train_loader, val_loader, test_loader_wiki, 
                    test_loader_shakespeare
                )
                
                # 2. Matched-Compute Control
                # --------------------------
                # CE-only training until same FLOP count
                print(f"    [Control: Matched FLOPs] Run {run_num}")
                config_matched = copy.deepcopy(config)
                config_matched.CONTROL_MODE = True
                config_matched.TARGET_FLOPS = metric_flops
                config_matched.METRIC_NAME = 'control'
                
                run_single_experiment(
                    output_dir_control_matched, config_matched, run_num, tokenizer, vocab_size,
                    train_loader, val_loader, test_loader_wiki, 
                    test_loader_shakespeare
                )
                
                # 3. Optimizer-Reset Control
                # --------------------------
                # CE-only, but with 5/15 split and reset
                print(f"    [Control: Reset] Run {run_num}")
                config_reset = copy.deepcopy(config)
                config_reset.CONTROL_MODE = False
                config_reset.METRIC_NAME = 'control_reset'
                
                run_single_experiment(
                    output_dir_control_reset, config_reset, run_num, tokenizer, vocab_size,
                    train_loader, val_loader, test_loader_wiki, 
                    test_loader_shakespeare
                )


if __name__ == '__main__':
    main()