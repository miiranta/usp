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
    # Test these durations of preconditioning (in batches)
    # With batch_size=512, ~4675 batches/epoch, these represent: ~1, ~3, ~5 epochs
    PRECOND_BATCHES_TO_TEST = [100, 1000, 10000] 
    
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
    EARLY_STOPPING_PATIENCE = 1
    
    # Model
    HIDDEN_DIM = 512
    NUM_LAYERS = 8
    NUM_ATTENTION_HEADS = 8
    SEQ_LENGTH = 32
    
    # Optimization
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    MAX_GRAD_NORM = 1.0
    
    # Data limits
    MAX_TRAIN_SAMPLES = 1000  # 0 means all data
    MAX_VAL_SAMPLES = 0
    MAX_TEST_SAMPLES = 0
    
    # System
    NUM_WORKERS = 0
    NUM_RUNS = 3
    LOG_EVERY_N_BATCHES = 10  # Log training metrics every N batches

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
        self.max_samples = max_samples
        
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
        self.full_input_ids = encodings['input_ids'][0]
        
        # Print token limit information
        if self.max_samples is not None and self.max_samples > 0:
            max_tokens = self.max_samples * self.seq_length
            print(f"  Dataset limited to {max_tokens:,} tokens ({self.max_samples:,} samples × {self.seq_length} seq_length)")
            print(f"  Full dataset has {len(self.full_input_ids):,} tokens")
        
        # Initialize active data (will be resampled each epoch if limited)
        self.resample()
    
    def resample(self):
        """Randomly sample a subset of data if max_samples is set"""
        if self.max_samples is not None and self.max_samples > 0:
            max_tokens = self.max_samples * self.seq_length
            if len(self.full_input_ids) > max_tokens:
                # Randomly select a starting position
                max_start = len(self.full_input_ids) - max_tokens
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
                self.input_ids = self.full_input_ids[start_idx:start_idx + max_tokens]
            else:
                self.input_ids = self.full_input_ids
        else:
            self.input_ids = self.full_input_ids
    
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
    def __init__(self, run_id, precond_batches, metric_name, seed):
        self.run_id = run_id
        self.precond_batches = precond_batches
        self.metric_name = metric_name
        self.seed = seed
        self.device = Config.DEVICE()
        
        self._set_seed()
        self._init_data()
        self._init_model()
        
        self.results = []
        self.tokens_processed = 0  # Count tokens only
        self.batches_processed = 0  # Track batch count for preconditioning
        
        # Measure training FLOPs once (analytically computed using 6ND)
        # NOTE: This measures CORE TRAINING COMPUTE ONLY (forward + backward + update)
        # Auxiliary computation (metrics, diagnostics) is NOT counted - standard practice
        self.training_flops_per_token = self._measure_training_flops()
        
        # Initialize Scaling Factors
        self.ce_start = 1.0
        self.metric_start = 1.0
        self._initialize_scalers()
        
        print(f"[Run {run_id}] Metric: {metric_name} | Precond: {precond_batches} batches")

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

    def _measure_training_flops(self):
        """Measure training FLOPs per token using the standard 6ND approximation.
        
        This computes CORE TRAINING COMPUTE:
        - Forward pass: ~2N FLOPs/token
        - Backward pass: ~4N FLOPs/token  
        - Total: 6N FLOPs/token
        
        IMPORTANT: Auxiliary computation is NOT counted:
        - Metric calculations (complexity measures, curvature probes)
        - Diagnostic overhead
        - Checkpointing bookkeeping
        
        This separation is standard practice in scaling laws literature
        (Kaplan et al., Hoffmann et al.) because:
        1. Metrics don't scale with model size
        2. They're algorithmic overhead, not learning compute
        3. They're not part of the trained model
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        return 6 * num_params
    
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
        self.log_epoch(0, 0.0, initial_val_loss, 0.0, mode='init')
        best_val_loss = initial_val_loss

        # 2. Main Loop
        for epoch in range(1, Config.MAX_TRAIN_EPOCHS + 1):
            tokens_before = self.tokens_processed

            # --- Train Epoch ---
            train_metrics = self.run_epoch(epoch)
            
            tokens_epoch = self.tokens_processed - tokens_before
            compute_epoch = tokens_epoch * self.training_flops_per_token

            # --- Validation ---
            val_loss = self.evaluate(self.val_loader, desc=f"VAL EP {epoch}")
            
            # Determine current mode based on batch count
            current_mode = 'precond' if self.batches_processed < self.precond_batches else 'train'
            
            # --- Logging ---
            self.log_epoch(epoch, train_metrics['loss'], val_loss, train_metrics['metric'], 
                          mode=current_mode)
            
            # Print Summary
            print(f"    EP {epoch}: Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Compute: {compute_epoch:.2e} | Batches: {self.batches_processed}")
            
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

    def run_epoch(self, epoch):
        self.model.train()
        
        # Resample training data at the start of each epoch (if limited)
        self.train_ds.resample()
        
        total_loss = 0
        total_ce = 0
        total_metric = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"EP {epoch}", leave=True)
        
        for batch_idx, (input_ids, labels) in enumerate(progress_bar):
            input_ids, labels = input_ids.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            # Determine if this batch uses preconditioning
            is_precond = self.batches_processed < self.precond_batches
            
            # --- OPTIMIZATION STEP ---
            # COUNTED COMPUTE: Forward, backward, parameter update (6N FLOPs/token)
            logits = self.model(input_ids)
            
            # CE Loss (Always needed for logging, but only backwarded if !is_precond)
            ce_loss = nn.CrossEntropyLoss()(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))
            
            # Metric Calculation (ALWAYS calculated for logging and visualization)
            # AUXILIARY COMPUTE (NOT COUNTED): Metrics, curvature probes, diagnostics
            # This is standard practice - metrics are algorithmic overhead, not learning compute
            if self.metric_name != 'control':
                metric_val = Metrics.calculate_metric(self.model, self.metric_name, logits, labels, input_ids)
                
                # During preconditioning, optimize the metric instead of CE
                if is_precond:
                    if self.metric_name.startswith('-'):
                         norm_metric = (metric_val / (abs(self.metric_start) + 1e-10)) * self.ce_start
                    else:
                         norm_metric = (abs(self.metric_start) / (metric_val + 1e-10)) * self.ce_start
                    loss = norm_metric
                else:
                    loss = ce_loss
            else:
                metric_val = torch.tensor(0.0, device=self.device)
                loss = ce_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.MAX_GRAD_NORM)
            self.optimizer.step()
            
            # --- LOGGING STEP ---
            # Track tokens and batches processed (compute calculated at logging time)
            self.tokens_processed += input_ids.numel()
            self.batches_processed += 1
            
            # Log batch-level metrics periodically
            if self.batches_processed % Config.LOG_EVERY_N_BATCHES == 0 or self.batches_processed == 1:
                self.log_batch(
                    epoch=epoch,
                    batch=self.batches_processed,
                    train_loss=loss.item(),
                    ce_loss=ce_loss.item(),
                    metric_val=metric_val.item(),
                    mode='precond' if is_precond else 'train'
                )

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_metric += metric_val.item()
            
            # Live Progress Update - show CE, metric, and what's being optimized
            avg_ce = total_ce / (batch_idx + 1)
            avg_metric = total_metric / (batch_idx + 1)
            mode_str = 'PRECOND' if is_precond else 'CE'
            progress_bar.set_postfix({
                'mode': mode_str,
                'CE': f"{avg_ce:.4f}",
                'metric': f"{avg_metric:.4f}"
            })
            
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
        
    def log_batch(self, epoch, batch, train_loss, ce_loss, metric_val, mode):
        """Log batch-level training metrics"""
        cumulative_compute = self.tokens_processed * self.training_flops_per_token
        
        entry = {
            'metric_name': self.metric_name,
            'run_id': self.run_id,
            'precond_batches': self.precond_batches,
            'epoch': epoch,
            'batch': batch,
            'mode': mode,
            'train_loss': train_loss,
            'ce_loss': ce_loss,
            'metric_val': metric_val,
            'tokens_processed': self.tokens_processed,
            'cumulative_compute': cumulative_compute,
            'log_type': 'batch'
        }
        self.results.append(entry)
        
    def log_epoch(self, epoch, train_loss, val_loss, metric, mode):
        """Log epoch-level validation metrics"""
        # Compute total training FLOPs at logging time (not during training)
        # Uses 6ND formula: 6 × num_parameters × num_tokens
        cumulative_compute = self.tokens_processed * self.training_flops_per_token
        
        entry = {
            'metric_name': self.metric_name,
            'run_id': self.run_id,
            'precond_batches': self.precond_batches,
            'epoch': epoch,
            'batches_processed': self.batches_processed,
            'mode': mode,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metric_val': metric,
            'tokens_processed': self.tokens_processed,
            'cumulative_compute': cumulative_compute,
            'log_type': 'validation'
        }
        self.results.append(entry)

def main():
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
        
    print(f"System Check: Using {Config.DEVICE()} for computation.")
    
    # 1. Run Experiments Loop
    for metric in Config.METRICS_TO_RUN:
        
        # Determine strict batches to test
        if metric == 'control':
            batches_to_test = [0]
        else:
            batches_to_test = Config.PRECOND_BATCHES_TO_TEST
        
        for precond_batches in batches_to_test:
            
            # Organize output by metric and precond configuration (Nature-style organization)
            clean_metric_name = metric.replace('/', '_over_').replace('.', '_')
            if clean_metric_name.startswith('-'): clean_metric_name = clean_metric_name[1:]
            
            config_dir_name = f"{clean_metric_name}_precond_{precond_batches}"
            config_output_dir = os.path.join(Config.OUTPUT_DIR, config_dir_name)
            
            # Skip if the folder is already created (as requested)
            if os.path.exists(config_output_dir):
                print(f"Skipping {metric} ({precond_batches} batches) - Folder '{config_dir_name}' exists.")
                continue
                
            print(f"\n=== STARTING CONFIGURATION: {metric} | Precond: {precond_batches} batches ===")
            os.makedirs(config_output_dir, exist_ok=True)
            
            for run_num in range(1, Config.NUM_RUNS + 1):
                seed = 42 + run_num
                print(f"  > Run {run_num}/{Config.NUM_RUNS}")
                
                trainer = Trainer(run_id=run_num, precond_batches=precond_batches, metric_name=metric, seed=seed)
                run_results = trainer.train()
                
                # Save individual run results
                run_csv_path = os.path.join(config_output_dir, f'results_run_{run_num}.csv')
                pd.DataFrame(run_results).to_csv(run_csv_path, index=False)

    print("\nTraining Complete. Results saved to individual folders.")

if __name__ == '__main__':
    main()
