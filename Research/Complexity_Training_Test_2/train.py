import os
import csv
import math
import copy
import time
import glob
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torch.utils.flop_counter import FlopCounterMode
import pandas as pd

# Disable efficient attention backend to allow second-order derivatives if needed
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
    # Experiment Settings
    EXPERIMENT_NAME = "nature_precond_efficiency"
    OUTPUT_DIR = "output"
    
    # Preconditioning
    # Test these durations of preconditioning (in epochs)
    PRECOND_EPOCHS_TO_TEST = [1, 2, 3, 4, 5] 
    
    # Metrics to Run (from original script)
    METRICS_TO_RUN = [
        'control',
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

    # Training Loop
    MAX_TRAIN_EPOCHS = 100 
    EARLY_STOPPING_PATIENCE = 3
    
    # Model
    HIDDEN_DIM = 128
    NUM_LAYERS = 4
    NUM_ATTENTION_HEADS = 4 
    SEQ_LENGTH = 64
    
    # Optimization
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    MAX_GRAD_NORM = 1.0
    
    # Data limits
    MAX_TRAIN_SAMPLES = 1000 # 0 means all data
    MAX_VAL_SAMPLES = 1000
    MAX_TEST_SAMPLES = 1000
    
    # System
    NUM_WORKERS = 0
    NUM_RUNS = 2 

    # Data
    DATASET_ROOT = "dataset" 

    @staticmethod
    def get_device():
        """Select GPU with most free memory, respecting locks from other instances"""
        import atexit
        
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
                        pid = int(f.read().strip())
                    
                    # Check if alive
                    is_dead = False
                    try:
                        os.kill(pid, 0)
                    except OSError as e:
                        # ProcessLookupError / ESRCH (errno 3) -> Process Dead
                        if isinstance(e, ProcessLookupError) or getattr(e, 'errno', 0) == 3:
                             is_dead = True
                        # PermissionError / EPERM (errno 1) -> Process Alive (other user)
                    
                    if not is_dead:
                         print(f"  GPU {gpu_id}: Locked by Active PID {pid} ({stat['free']/1024**3:.2f}GB free). Skipping.")
                         continue 
                    else:
                        print(f"  GPU {gpu_id}: Stale lock (PID {pid} dead). Reclaiming.")
                        try: os.remove(lock_file)
                        except: pass
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
                try: os.remove(path)
                except: pass
                
        lock_path = os.path.join(Config.OUTPUT_DIR, f".gpu_lock_{selected_gpu}")
        
        # Only register cleanup if we actually own it
        try:
            if os.path.exists(lock_path):
                with open(lock_path, 'r') as f:
                    content = f.read().strip()
                    if content and int(content) == os.getpid():
                        atexit.register(cleanup_lock, lock_path)
        except:
            pass

        return torch.device(f'cuda:{selected_gpu}')
        
    # Lazy initialization of device
    _DEVICE = None
    @classmethod
    def DEVICE(cls):
        if cls._DEVICE is None:
            cls._DEVICE = cls.get_device()
        return cls._DEVICE



# ============================================================================
# METRICS IMPLEMENTATION
# ============================================================================

class Metrics:
    @staticmethod
    def get_all_weights(model):
        all_weights = []
        for param in model.parameters():
            all_weights.append(param.view(-1))
        return torch.cat(all_weights)

    @staticmethod
    def get_layer_weights(model):
        layers = []
        layers.append(model.embedding.weight.view(-1))
        for layer in model.transformer.layers:
            layer_params = []
            for p in layer.parameters():
                layer_params.append(p.view(-1))
            layers.append(torch.cat(layer_params))
        layers.append(model.lm_head.weight.view(-1))
        return layers

    @staticmethod
    def soft_histogram(x, num_bins=None, min_val=None, max_val=None):
        x = x.view(-1)
        if min_val is None: min_val = x.min()
        if max_val is None: max_val = x.max()
        normalized = (x - min_val) / (max_val - min_val + 1e-10)
        n = x.numel()
        if num_bins is None:
            with torch.no_grad():
                sample_size = min(10000, n)
                if n > sample_size:
                    indices = torch.randint(0, n, (sample_size,), device=x.device)
                    sample = normalized[indices]
                else:
                    sample = normalized
                q1 = torch.quantile(sample, 0.25)
                q3 = torch.quantile(sample, 0.75)
                iqr = q3 - q1
                if iqr == 0:
                    num_bins = max(1, int(np.ceil(float(np.sqrt(n)))))
                else:
                    bin_width = float(2 * iqr.item() * (n ** (-1/3)))
                    num_bins = max(1, int(np.ceil(1.0 / bin_width)))
                    num_bins = max(1, min(num_bins, 500))
        
        bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=x.device)
        bin_width_tensor = bin_edges[1] - bin_edges[0]
        bin_indices_float = normalized / bin_width_tensor
        bin_indices_floor = torch.floor(bin_indices_float)
        bin_indices_left = bin_indices_floor.long().clamp(0, num_bins - 1)
        bin_indices_right = (bin_indices_floor + 1).long().clamp(0, num_bins - 1)
        frac = bin_indices_float - bin_indices_floor
        weight_left = 1.0 - frac
        weight_right = frac
        hist = torch.zeros(num_bins, device=x.device, dtype=x.dtype)
        hist.scatter_add_(0, bin_indices_left, weight_left)
        hist.scatter_add_(0, bin_indices_right, weight_right)
        probs = hist / (hist.sum() + 1e-10)
        probs = torch.clamp(probs, 1e-10, 1.0)
        probs = probs / probs.sum()
        return probs, num_bins

    @staticmethod
    def _varentropy_func(logits_chunk):
        probs = torch.softmax(logits_chunk, dim=-1)
        log_p = torch.log(probs + 1e-10)
        entropy = -(probs * log_p).sum(dim=-1, keepdim=True)
        term1 = (probs * log_p**2).sum(dim=-1)
        v_chunk = term1 - entropy.squeeze(-1)**2
        return v_chunk

    @staticmethod
    def calculate_metric(model, metric_name, logits=None, labels=None, input_ids=None):
        mapping = {
            'A': 'total_variation',
            'B': 'third_order_curvature_norm',
            'C': 'hosoya_index',
            'D': 'disequilibrium',
            'G': 'arithmetic_derivative',
            'J': 'persistent_landscape_norm',
            'U': 'varentropy',
            'Y': 'information_compression_ratio'
        }
        
        if metric_name == 'control':
            return torch.tensor(0.0, device=Config.DEVICE)
            
        if metric_name.startswith('-'):
            clean_name = metric_name[1:]
        else:
            clean_name = metric_name
            
        try:
            if '/' in clean_name:
                parts = clean_name.split('/')
                if len(parts) != 2: raise ValueError("Only one division allowed")
                num_str, den_str = parts
                
                num_val = 1.0
                for code in num_str.split('.'):
                    code = code.strip()
                    if not code: continue
                    real_name = mapping.get(code, code)
                    val = Metrics._calculate_primitive_metric(model, real_name, logits, labels, input_ids)
                    num_val = num_val * val
                    
                den_val = 1.0
                for code in den_str.split('.'):
                    code = code.strip()
                    if not code: continue
                    real_name = mapping.get(code, code)
                    val = Metrics._calculate_primitive_metric(model, real_name, logits, labels, input_ids)
                    den_val = den_val * val
                    
                final_val = num_val / (den_val + 1e-10)
            else:
                val = 1.0
                for code in clean_name.split('.'):
                    code = code.strip()
                    if not code: continue
                    real_name = mapping.get(code, code)
                    v = Metrics._calculate_primitive_metric(model, real_name, logits, labels, input_ids)
                    val = val * v
                final_val = val
                
            return final_val
            
        except Exception as e:
            print(f"Error calculating metric {metric_name}: {e}")
            return torch.tensor(0.0, device=Config.DEVICE)

    @staticmethod
    def _calculate_primitive_metric(model, metric_name, logits=None, labels=None, input_ids=None):
        weights = Metrics.get_all_weights(model)
        device = weights.device
        
        if metric_name == 'shannon':
            probs, _ = Metrics.soft_histogram(weights)
            return -(probs * torch.log(probs)).sum()
            
        elif metric_name == 'total_variation':
            w = model.lm_head.weight
            diff_h = torch.abs(w[:, 1:] - w[:, :-1]).sum()
            diff_v = torch.abs(w[1:, :] - w[:-1, :]).sum()
            return (diff_h + diff_v) / w.numel()

        elif metric_name == 'third_order_curvature_norm':
            if labels is None: return torch.tensor(0.0, device=device)
            # Use smaller batch for expensive curvature
            if input_ids is not None:
                b_size = 1
                seq_len = min(input_ids.size(1), 32) # Reduced for efficiency
                input_ids_small = input_ids[:b_size, :seq_len]
                labels_small = labels[:b_size, :seq_len]
                logits_small = model(input_ids_small)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_small.view(-1, logits_small.size(-1)), labels_small.view(-1))
            else:
               return torch.tensor(0.0, device=device)

            all_params = [p for p in model.parameters() if p.requires_grad]
            # Optimization: only check last few layers or simplified subset
            if len(all_params) > 5:
                # Taking last 5 params (likely head and last layer)
                params_subset = all_params[-5:]
            else:
                params_subset = all_params
                
            grads = torch.autograd.grad(loss, params_subset, create_graph=True)
            v = [torch.randint_like(p, high=2) * 2 - 1 for p in params_subset]
            grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
            Hv = torch.autograd.grad(grad_v, params_subset, create_graph=True)
            vHv = sum([(h * vi).sum() for h, vi in zip(Hv, v)])
            grad_vHv = torch.autograd.grad(vHv, params_subset, create_graph=True)
            norm_grad_vHv = torch.sqrt(sum([(g**2).sum() for g in grad_vHv]))
            return norm_grad_vHv

        elif metric_name == 'varentropy':
            if logits is None: return torch.tensor(0.0, device=device)
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            # Checkpoint for memory efficiency
            chunk_size = 1024 
            chunks = torch.split(logits_flat, chunk_size, dim=0)
            vals = []
            for chunk in chunks:
                if chunk.requires_grad:
                    v_chunk = checkpoint(Metrics._varentropy_func, chunk, use_reentrant=False)
                else:
                    v_chunk = Metrics._varentropy_func(chunk)
                vals.append(v_chunk)
            return torch.cat(vals).mean()

        elif metric_name == 'arithmetic_derivative':
            W = model.lm_head.weight
            f = torch.fft.rfft(W, dim=1)
            return torch.norm(f, p=1) / (torch.norm(f, p=2) + 1e-10)

        elif metric_name == 'hosoya_index':
            W = model.lm_head.weight
            return torch.norm(W, p='nuc')

        elif metric_name == 'persistent_landscape_norm':
            W = model.lm_head.weight
            if W.size(0) > 200: # Reduced sampling 
                indices = torch.randperm(W.size(0))[:200]
                W = W[indices]
            dist = torch.cdist(W, W)
            vals = torch.sort(dist.view(-1))[0]
            return torch.norm(vals, p=2)

        elif metric_name == 'disequilibrium':
            probs, _ = Metrics.soft_histogram(weights)
            n_bins = probs.size(0)
            uniform_prob = 1.0 / n_bins
            return ((probs - uniform_prob) ** 2).sum()

        elif metric_name == 'information_compression_ratio':
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            max_entropy = math.log(probs.size(-1))
            excess_entropy = 1.0 - (entropy / (max_entropy + 1e-10))
            return 1.0 / (excess_entropy + 1e-10)
        
        return torch.tensor(0.0, device=device)


# ============================================================================
# DATASET & MODEL
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length, max_samples=None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"CRITICAL: Dataset file not found at {file_path}. Please check the path.")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            if file_path.endswith('.csv'):
                reader = csv.DictReader(f)
                text = ''.join(row['text'] for row in reader)
            else:
                text = f.read()
        
        # Suppress tokenization length warning
        # (We handle slicing manually in __getitem__)
        _prev_max_len = tokenizer.model_max_length
        tokenizer.model_max_length = int(1e9)
        
        encodings = tokenizer(
            text,
            return_tensors='pt',
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False
        )
        
        tokenizer.model_max_length = _prev_max_len # Restore
        self.input_ids = encodings['input_ids'][0]
        
        # Apply token limit if configured
        if max_samples is not None and max_samples > 0:
            max_tokens = max_samples * seq_length
            if len(self.input_ids) > max_tokens:
                self.input_ids = self.input_ids[:max_tokens]
    
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
        
    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.position_embedding(positions)
        
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device), 
            diagonal=1
        ).bool()
        
        x = self.transformer(x, mask=causal_mask)
        x = self.lm_head(x)
        return x

# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, run_id, precond_epochs, metric_name, seed):
        self.run_id = run_id
        self.precond_epochs = precond_epochs
        self.metric_name = metric_name
        self.seed = seed
        self.device = Config.DEVICE()
        
        self._set_seed()
        self._init_data()
        self._init_model()
        
        self.results = []
        self.total_flops = 0.0
        
        # Initialize Scaling Factors
        self.ce_start = 1.0
        self.metric_start = 1.0
        self._initialize_scalers()
        
        print(f"[Run {run_id}] Metric: {metric_name} | Precond: {precond_epochs} eps")

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
    
    def _init_data(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_root = os.path.join(script_dir, Config.DATASET_ROOT)
        
        # Paths
        train_path = os.path.join(dataset_root, 'wikitext-2/wiki.train.tokens')
        val_path = os.path.join(dataset_root, 'wikitext-2/wiki.valid.tokens')
        test_path_wiki = os.path.join(dataset_root, 'wikitext-2/wiki.test.tokens')
        
        # Datasets
        self.train_ds = TextDataset(train_path, self.tokenizer, Config.SEQ_LENGTH, max_samples=Config.MAX_TRAIN_SAMPLES)
        self.val_ds = TextDataset(val_path, self.tokenizer, Config.SEQ_LENGTH, max_samples=Config.MAX_VAL_SAMPLES)
        self.test_ds = TextDataset(test_path_wiki, self.tokenizer, Config.SEQ_LENGTH, max_samples=Config.MAX_TEST_SAMPLES)
        
        # Loaders
        self.train_loader = DataLoader(self.train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
        self.val_loader = DataLoader(self.val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
        self.test_loader = DataLoader(self.test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    def _init_model(self):
        self.model = SimpleTransformer(
            self.tokenizer.vocab_size,
            Config.HIDDEN_DIM,
            Config.NUM_LAYERS,
            Config.NUM_ATTENTION_HEADS,
            Config.SEQ_LENGTH
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

    def _initialize_scalers(self):
        if self.metric_name == 'control': return
        
        self.model.eval()
        input_ids, labels = next(iter(self.train_loader))
        input_ids, labels = input_ids.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
             logits = self.model(input_ids)
             ce = nn.CrossEntropyLoss()(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))
             self.ce_start = ce.item()
        
        with torch.enable_grad():
             # Forward again for graph if needed
             logits = self.model(input_ids)
             metric = Metrics.calculate_metric(self.model, self.metric_name, logits, labels, input_ids)
             self.metric_start = metric.item()
             
        # Guard against zero
        if self.metric_start == 0: self.metric_start = 1.0
        
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        min_model_state = None
        
        # 1. Evaluate Initial State
        initial_val_loss = self.evaluate(self.val_loader)
        self.log_epoch(0, 0.0, initial_val_loss, 0.0, 0.0, mode='init')
        best_val_loss = initial_val_loss

        # 2. Main Loop
        for epoch in range(1, Config.MAX_TRAIN_EPOCHS + 1):
            is_precond = epoch <= self.precond_epochs
            
            flops_before = self.total_flops

            # --- Train Epoch ---
            train_metrics = self.run_epoch(epoch, is_precond)
            
            flops_epoch = self.total_flops - flops_before

            # --- Validation ---
            val_loss = self.evaluate(self.val_loader, desc=f"Val Ep {epoch}")
            
            # --- Logging ---
            self.log_epoch(epoch, train_metrics['loss'], val_loss, self.total_flops, train_metrics['metric'], mode='precond' if is_precond else 'train')
            
            # Print Summary
            print(f"    Ep {epoch}: Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_loss:.4f} | FLOPs: {flops_epoch:.2e}")
            
            # --- Early Stopping Logic ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                min_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"  > Stopping early at epoch {epoch}. Best Val: {best_val_loss:.4f}")
                break
                
        return self.results

    def run_epoch(self, epoch, is_precond):
        self.model.train()
        total_loss = 0
        total_ce = 0
        total_metric = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Ep {epoch} (Pre={is_precond})", leave=True)
        
        for batch_idx, (input_ids, labels) in enumerate(progress_bar):
            input_ids, labels = input_ids.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            # Count exact FLOPs using PyTorch's FlopCounterMode
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                logits = self.model(input_ids)
                
                # CE Loss
                ce_loss = nn.CrossEntropyLoss()(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))
                
                # Metric Calculation
                if self.metric_name != 'control':
                    metric_val = Metrics.calculate_metric(self.model, self.metric_name, logits, labels, input_ids)
                else:
                    metric_val = torch.tensor(0.0, device=self.device)
                
                # Determine Loss
                if is_precond and self.metric_name != 'control':
                    # Normalization
                    if self.metric_name.startswith('-'): # Minimizing metric
                         # metric / start * ce_start
                         norm_metric = (metric_val / (abs(self.metric_start) + 1e-10)) * self.ce_start
                    else: # Maximizing metric -> Minimizing 1/metric
                         # start / metric * ce_start
                         norm_metric = (abs(self.metric_start) / (metric_val + 1e-10)) * self.ce_start
                         
                    loss = norm_metric
                else:
                    loss = ce_loss
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.MAX_GRAD_NORM)
                self.optimizer.step()
            
            # Update Counters with Measured FLOPs
            self.total_flops += flop_counter.get_total_flops()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_metric += metric_val.item()
            
            # Live Loss Update
            progress_bar.set_postfix({'loss': f"{total_loss / (batch_idx + 1):.4f}"})
            
        self.scheduler.step()
        
        return {
            'loss': total_loss / len(self.train_loader),
            'ce': total_ce / len(self.train_loader),
            'metric': total_metric / len(self.train_loader)
        }

    def evaluate(self, loader, desc="Validation"):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for input_ids, labels in tqdm(loader, desc=desc, leave=False):
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)
                logits = self.model(input_ids)
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))
                total_loss += loss.item()
        return total_loss / len(loader)
        
    def log_epoch(self, epoch, train_loss, val_loss, flops, metric, mode):
        entry = {
            'metric_name': self.metric_name,
            'run_id': self.run_id,
            'precond_epochs': self.precond_epochs,
            'epoch': epoch,
            'mode': mode,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metric_val': metric,
            'cumulative_flops': flops
        }
        self.results.append(entry)

def main():
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
        
    print(f"System Check: Using {Config.DEVICE()} for computation.")
    
    # 1. Run Experiments Loop
    for metric in Config.METRICS_TO_RUN:
        
        # Determine strict epochs to test
        if metric == 'control':
            epochs_to_test = [0]
        else:
            epochs_to_test = Config.PRECOND_EPOCHS_TO_TEST
        
        for precond_eps in epochs_to_test:
            
            # Organize output by metric and precond configuration (Nature-style organization)
            clean_metric_name = metric.replace('/', '_over_').replace('.', '_')
            if clean_metric_name.startswith('-'): clean_metric_name = clean_metric_name[1:]
            
            config_dir_name = f"{clean_metric_name}_precond_{precond_eps}"
            config_output_dir = os.path.join(Config.OUTPUT_DIR, config_dir_name)
            
            # Skip if the folder is already created (as requested)
            if os.path.exists(config_output_dir):
                print(f"Skipping {metric} ({precond_eps} eps) - Folder '{config_dir_name}' exists.")
                continue
                
            print(f"\n=== STARTING CONFIGURATION: {metric} | Precond: {precond_eps} ===")
            os.makedirs(config_output_dir, exist_ok=True)
            
            config_results = []
            
            for run_num in range(1, Config.NUM_RUNS + 1):
                seed = 42 + run_num
                print(f"  > Run {run_num}/{Config.NUM_RUNS}")
                
                trainer = Trainer(run_id=run_num, precond_epochs=precond_eps, metric_name=metric, seed=seed)
                run_results = trainer.train()
                config_results.extend(run_results)
                
                # Save individual run results
                run_csv_path = os.path.join(config_output_dir, f'results_run_{run_num}.csv')
                pd.DataFrame(run_results).to_csv(run_csv_path, index=False)
            
            # Save aggregated results for this config
            pd.DataFrame(config_results).to_csv(os.path.join(config_output_dir, 'aggregated_results.csv'), index=False)

    # 2. Analysis (Aggregating all subfolders)
    print("\nTraining Complete. Generating Detailed Analysis...")
    
    # Collect all CSVs from subfolders
    all_results = []
    # glob pattern to find all results_run_*.csv in subfolders
    pattern = os.path.join(Config.OUTPUT_DIR, "*", "results_run_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print("No result files found for analysis.")
        return

    print(f"Found {len(csv_files)} result files. Aggregating...")
    for f in csv_files:
        try:
            df_chunk = pd.read_csv(f)
            all_results.append(df_chunk)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_results:
        print("No valid data loaded.")
        return
        
    df = pd.concat(all_results, ignore_index=True)
    
    # Ensure numeric columns
    cols = ['cumulative_flops', 'val_loss', 'precond_epochs', 'run_id']
    for c in cols: 
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # 2a. Determine Baseline (Control) Performance
    baseline_flops_list = []
    # Identify control runs
    control_df = df[df['metric_name'] == 'control']
    
    if not control_df.empty:
        # Find global min val loss for control runs to establish baseline target
        for r_id in control_df['run_id'].unique():
            run_data = control_df[control_df['run_id'] == r_id]
            if run_data.empty: continue
            
            # Find the epoch where validation loss was minimized
            best_idx = run_data['val_loss'].idxmin()
            best_row = run_data.loc[best_idx]
            baseline_flops_list.append(best_row['cumulative_flops'])
            
        avg_baseline_flops = np.mean(baseline_flops_list) if baseline_flops_list else 1.0
        print(f"Baseline (Control) Mean FLOPs to Convergence: {avg_baseline_flops:.4e}")
    else:
        print("Warning: No Control runs found in results. Setting baseline to 1.0 for relative calculation.")
        avg_baseline_flops = 1.0
        
    # 2b. Comparative Analysis
    analysis_data = []
    
    # Group by Configuration
    configs = df[['metric_name', 'precond_epochs']].drop_duplicates()
    
    for _, config in configs.iterrows():
        m_name = config['metric_name']
        p_eps = config['precond_epochs']
        
        # Get all runs for this config
        sdf = df[(df['metric_name'] == m_name) & (df['precond_epochs'] == p_eps)]
        
        flops_list = []
        min_val_list = []
        epochs_list = []
        
        for r_id in sdf['run_id'].unique():
            rdf = sdf[sdf['run_id'] == r_id]
            if rdf.empty: continue
            
            # Find the row with Minimum Validation Loss
            min_idx = rdf['val_loss'].idxmin()
            min_row = rdf.loc[min_idx]
            
            flops_list.append(min_row['cumulative_flops'])
            min_val_list.append(min_row['val_loss'])
            epochs_list.append(min_row['epoch'])
        
        if not flops_list: continue

        mean_flops = np.mean(flops_list)
        # Speedup > 1.0 means fewer FLOPs than control
        speedup = avg_baseline_flops / mean_flops if mean_flops > 0 and avg_baseline_flops > 0 else 0.0
            
        analysis_data.append({
            'metric': m_name,
            'precond_epochs': int(p_eps),
            'n_runs': len(flops_list),
            'mean_flops_to_min': mean_flops,
            'std_flops_to_min': np.std(flops_list),
            'speedup_vs_control': speedup,
            'mean_min_val_loss': np.mean(min_val_list),
            'ste_min_val_loss': stats.sem(min_val_list) if len(min_val_list) > 1 else 0.0,
            'mean_convergence_epoch': np.mean(epochs_list)
        })
        
    analysis_df = pd.DataFrame(analysis_data)
    
    if not analysis_df.empty:
        # Sort by efficiency (speedup)
        analysis_df = analysis_df.sort_values('speedup_vs_control', ascending=False)
        
        output_path = os.path.join(Config.OUTPUT_DIR, 'final_flops_analysis.csv')
        analysis_df.to_csv(output_path, index=False)
        print(f"Analysis saved to {output_path}")
        print("\nTop 5 Configurations by Efficiency (Speedup vs Control):")
        print(analysis_df[['metric', 'precond_epochs', 'speedup_vs_control', 'mean_min_val_loss']].head(5).to_string())
    else:
        print("No analysis data generated.")

if __name__ == '__main__':
    main()
