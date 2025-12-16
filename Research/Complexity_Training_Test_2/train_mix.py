import os
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

# Disable efficient attention backend to allow second-order derivatives
# This is necessary for metrics that require double backward (e.g. gradient_entropy)
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
    
    # Metric to optimize (if CONTROL_MODE is False)
    METRIC_NAME = 'shannon' 
    
    # Model hyperparameters
    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    NUM_ATTENTION_HEADS = 4 
    
    # Training hyperparameters
    BATCH_SIZE = 256 
    EPOCHS = 20
    SEQ_LENGTH = 32
    MAX_GRAD_NORM = 1.0
    MAX_SAMPLES = 1000
    
    # Number of runs
    NUM_OF_RUN_PER_CALL = 1
    
    # Complexity calculation interval
    COMPLEXITY_UPDATE_INTERVAL = 1 
    
    # Device configuration
    GPU_INDEX = 0
    DEVICE = torch.device(f'cuda:{GPU_INDEX}' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0
    
    # Performance optimizations
    USE_COMPILE = False  
    
    # DONT CHANGE
    LMC_WEIGHT = 0.0         
    LEARNING_RATE = 1e-4

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
        # Extract weights per layer (simplified for Transformer)
        layers = []
        # Embeddings
        layers.append(model.embedding.weight.view(-1))
        # Transformer layers
        for layer in model.transformer.layers:
            # Concat all weights in a layer
            layer_params = []
            for p in layer.parameters():
                layer_params.append(p.view(-1))
            layers.append(torch.cat(layer_params))
        # Head
        layers.append(model.lm_head.weight.view(-1))
        return layers

    @staticmethod
    def soft_histogram(x, num_bins=None, min_val=None, max_val=None):
        x = x.view(-1)
        if min_val is None: min_val = x.min()
        if max_val is None: max_val = x.max()
        
        # Normalize
        normalized = (x - min_val) / (max_val - min_val + 1e-10)
        
        n = x.numel()
        if num_bins is None:
            # Freedman-Diaconis rule approximation
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
                    num_bins = min(num_bins, 500) # Cap bins
        
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
    def _calculate_primitive_metric(model, metric_name, logits=None, labels=None, input_ids=None):
        weights = Metrics.get_all_weights(model)
        device = weights.device
        
        if metric_name == 'shannon':
            probs, _ = Metrics.soft_histogram(weights)
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'hessian_spectral_radius':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            
            # Subsample batch to reduce memory usage
            batch_size = logits.size(0)
            max_samples = 2
            if batch_size > max_samples:
                indices = torch.randperm(batch_size, device=device)[:max_samples]
                logits_sub = logits[indices]
                labels_sub = labels[indices]
            else:
                logits_sub = logits
                labels_sub = labels

            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_sub.view(-1, logits_sub.size(-1)), labels_sub.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            # Random v
            v = [torch.randn_like(p) for p in params]
            # Normalize v
            v_norm = torch.sqrt(sum([(vi**2).sum() for vi in v]))
            v = [vi / (v_norm + 1e-10) for vi in v]
            
            # Power iteration (1 step for speed, differentiable)
            grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
            Hv = torch.autograd.grad(grad_v, params, create_graph=True)
            
            # Rayleigh quotient
            vHv = sum([(h * vi).sum() for h, vi in zip(Hv, v)])
            return torch.abs(vHv)

        elif metric_name == 'lipschitz_variance':
            # Variance of local Lipschitz constants
            # Proxy: Variance of spectral norms of layers
            norms = []
            for layer in model.transformer.layers:
                w = layer.linear1.weight
                try:
                    s = torch.linalg.svdvals(w)[0]
                    norms.append(s)
                except: pass
            if not norms: return torch.tensor(0.0, device=device)
            return torch.var(torch.stack(norms))

        elif metric_name == 'activation_kurtosis_variance':
            # Layerwise tail instability
            # Proxy: Variance of kurtosis across layers
            kurts = []
            for layer in model.transformer.layers:
                w = layer.linear1.weight.view(-1)
                mu = w.mean()
                sigma = w.std()
                k = ((w - mu)**4).mean() / (sigma**4 + 1e-10)
                kurts.append(k)
            if not kurts: return torch.tensor(0.0, device=device)
            return torch.var(torch.stack(kurts))

        elif metric_name == 'disequilibrium':
            probs, _ = Metrics.soft_histogram(weights)
            n_bins = probs.size(0)
            uniform_prob = 1.0 / n_bins
            return ((probs - uniform_prob) ** 2).sum()

        elif metric_name == 'gradient_direction_entropy':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            all_grads = torch.cat([g.view(-1) for g in grads])
            # Soft histogram of gradient values
            probs, _ = Metrics.soft_histogram(torch.abs(all_grads))
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'gradient_covariance_trace':
            # ||g||^2 as proxy for trace of covariance if mean g is small
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_norm_sq = sum([(g**2).sum() for g in grads])
            return grad_norm_sq

        return torch.tensor(0.0, device=device)

    @staticmethod
    def calculate_metric(model, metric_name, logits=None, labels=None, input_ids=None):
        mapping = {
            'H': 'shannon',
            'A': 'hessian_spectral_radius',
            'B': 'lipschitz_variance',
            'C': 'activation_kurtosis_variance',
            'D': 'disequilibrium',
            'E': 'gradient_direction_entropy',
            'F': 'gradient_covariance_trace'
        }
        
        if metric_name.startswith('-'):
            metric_name = metric_name[1:]
        
        try:
            if '/' in metric_name:
                parts = metric_name.split('/')
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
                    
                return num_val / (den_val + 1e-10)
            else:
                val = 1.0
                for code in metric_name.split('.'):
                    code = code.strip()
                    if not code: continue
                    real_name = mapping.get(code, code)
                    v = Metrics._calculate_primitive_metric(model, real_name, logits, labels, input_ids)
                    val = val * v
                return val
        except Exception as e:
            print(f"Error calculating metric {metric_name}: {e}")
            return torch.tensor(0.0, device=Config.DEVICE)

# ============================================================================
# DEVICE INITIALIZATION
# ============================================================================

def initialize_device():
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_index = Config.GPU_INDEX
        print(f"GPU: {torch.cuda.get_device_name(gpu_index)}")
        print(f"Memory: {torch.cuda.get_device_properties(gpu_index).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        torch.cuda.set_device(gpu_index)
        torch.cuda.empty_cache()
        
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: CUDA is not available! Using CPU instead.")
    
    print()
    return device

def check_efficient_attention():
    if not torch.cuda.is_available():
        return False, "pytorch"
    
    try:
        import xformers
        return True, "xformers"
    except ImportError:
        return False, "pytorch"

# ============================================================================
# DATASET
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
        
        print(f"  File size: {len(text) / 1024 / 1024:.2f} MB")
        
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
        
        if max_samples is not None:
            max_length = max_samples * seq_length
            original_len = len(self.input_ids)
            self.input_ids = self.input_ids[:max_length]
            print(f"  Tokens before limit: {original_len:,}")
            print(f"  Tokens after limit:  {len(self.input_ids):,} (max_samples={max_samples})")
        else:
            print(f"  Tokens loaded: {len(self.input_ids):,} (no limit)")
    
    def __len__(self):
        return max(0, len(self.input_ids) - self.seq_length)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx:idx + self.seq_length]
        target_ids = self.input_ids[idx + 1:idx + self.seq_length + 1]
        
        return {'input_ids': input_ids, 'labels': target_ids}

# ============================================================================
# MODEL
# ============================================================================

class TransformerLLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_attention_heads, seq_length, 
                 enable_efficient_attention=False, attention_backend="pytorch"):
        super(TransformerLLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(seq_length, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        
        if enable_efficient_attention and attention_backend in ["xformers"]:
            try:
                encoder_layer.self_attn = nn.MultiheadAttention(
                    hidden_dim, num_attention_heads, dropout=0.1, batch_first=True
                )
            except Exception as e:
                print(f"Warning: Could not enable {attention_backend} attention: {e}")
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.efficient_attention_enabled = enable_efficient_attention
        self.attention_backend = attention_backend
    
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        embeddings = self.embedding(input_ids) + self.position_embedding(positions)
        
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=input_ids.device), 
            diagonal=1
        ).bool()
    
        transformer_out = self.transformer(
            embeddings,
            mask=causal_mask
        )
        
        logits = self.lm_head(transformer_out)
        return logits

# ============================================================================
# SLOPE NORMALIZATION
# ============================================================================

def normalize_slope_arctan(m):
    return math.atan(m) / (math.pi / 2)

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, config, vocab_size, metric_start, ce_start, val_error_slope=0.0, lmc_weight=0.0):
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    total_combined_loss = 0.0
    total_samples = 0
    
    metric_value = None
    
    progress_bar = tqdm(train_loader, desc=f"Train")
    
    for batch_idx, batch in enumerate(progress_bar):
        if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
            torch.compiler.cudagraph_mark_step_begin()
        
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        logits = model(input_ids)
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
        
        if metric_value is None or batch_idx % config.COMPLEXITY_UPDATE_INTERVAL == 0:
            metric_tensor = Metrics.calculate_metric(model, config.METRIC_NAME, logits, labels, input_ids)
            metric_value = metric_tensor 
            metric_value_scalar = metric_tensor.item()
 
        # New Loss Formula: (x_start / (x_value + 1e-10)) * ce_start
        # We want to MAXIMIZE x_value (disequilibrium).
        if config.METRIC_NAME.startswith('-'):
            metric_loss_normalized = (metric_value / (metric_start + 1e-10)) * ce_start
        else:
            metric_loss_normalized = (metric_start / (metric_value + 1e-10)) * ce_start
        
        ce_weight = 1.0 - lmc_weight
        lmc_weight_actual = lmc_weight
        
        if config.CONTROL_MODE == True:
            combined_loss = ce_loss
        else:
            combined_loss = ce_weight * ce_loss + lmc_weight_actual * metric_loss_normalized
        
        combined_loss.backward()
        
        if config.MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += ce_loss.detach().item()
        total_metric += metric_value_scalar
        total_combined_loss += combined_loss.detach().item()
        total_samples += 1 
    
        progress_bar.set_postfix({
            'loss': f'{total_loss / total_samples:.4f}',
            'met': f'{total_metric / total_samples:.4f}',
            'w': f'{lmc_weight:.3f}',
            'comb': f'{combined_loss.detach().item():.4f}'
        })
    
    avg_loss = total_loss / total_samples
    avg_metric = total_metric / total_samples
    avg_combined = total_combined_loss / total_samples
    
    return avg_loss, avg_metric, avg_combined

def validate(model, val_loader, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            logits = model(input_ids)
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
        
            total_loss += loss.item()
            total_samples += 1
    
    return total_loss / total_samples if total_samples > 0 else 0.0

def test(model, test_loader, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            logits = model(input_ids)
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
            
            total_loss += loss.item()
            total_samples += 1
    
    return total_loss / total_samples if total_samples > 0 else 0.0

# ============================================================================
# LOGGING AND VISUALIZATION
# ============================================================================

def save_results_to_csv(output_dir, train_losses, val_losses, metric_values, slope_values, lmc_weight_values,
                       test_losses_wiki, test_losses_shakespeare, config, run_num=1):
    csv_filename = f'results_{config.METRIC_NAME.replace("/", "_")}_{run_num}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['Metric', 'Value'])
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
        writer.writerow(['Final Val Error Slope', f'{slope_values[-1]:.16f}'])
        writer.writerow(['Final LMC Weight', f'{lmc_weight_values[-1]:.3f}'])
        writer.writerow(['Run Number', f'{run_num}'])
        writer.writerow([])
        
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Test Loss Wiki', 'Test Loss Shakespeare', 'Metric Value', 'Slope', 'Weight'])
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                f'{train_losses[epoch]:.16f}',
                f'{val_losses[epoch]:.16f}',
                f'{test_losses_wiki[epoch]:.16f}',
                f'{test_losses_shakespeare[epoch]:.16f}',
                f'{metric_values[epoch]:.16f}',
                f'{slope_values[epoch]:.16f}',
                f'{lmc_weight_values[epoch]:.3f}'
            ])
    
    print(f"Results saved to '{csv_path}'")

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def run_training_single(output_dir, config, run_num):
    device = initialize_device()
    
    enable_efficient_attention, attention_backend = check_efficient_attention()
    
    print("Initializing RoBERTa tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.train.tokens')
    val_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.valid.tokens')
    test_path_wiki = os.path.join(script_dir, 'dataset/wikitext-2/wiki.test.tokens')
    test_path_shakespeare = os.path.join(script_dir, 'dataset/tiny-shakespare/test.csv')
    
    for path in [train_path, val_path, test_path_wiki, test_path_shakespeare]:
        if not os.path.exists(path):
            print(f"Error: Dataset file '{path}' not found!")
            return
    
    train_dataset = TextDataset(train_path, tokenizer, config.SEQ_LENGTH, config.MAX_SAMPLES)
    val_dataset = TextDataset(val_path, tokenizer, config.SEQ_LENGTH, None) 
    test_dataset_wiki = TextDataset(test_path_wiki, tokenizer, config.SEQ_LENGTH, None)
    test_dataset_shakespeare = TextDataset(test_path_shakespeare, tokenizer, config.SEQ_LENGTH, None) 
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader_wiki = DataLoader(test_dataset_wiki, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader_shakespeare = DataLoader(test_dataset_shakespeare, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    print("\nInitializing Transformer model...")
    vocab_size = len(tokenizer)
    model = TransformerLLM(
        vocab_size=vocab_size,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        seq_length=config.SEQ_LENGTH,
        enable_efficient_attention=enable_efficient_attention,
        attention_backend=attention_backend
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE, total_steps=total_steps, pct_start=0.1)
    
    train_losses = []
    val_losses = []
    metric_values = []
    slope_values = []
    lmc_weight_values = []
    test_losses_wiki = []
    test_losses_shakespeare = []
    
    print("\nCalculating initial CE loss and Metric...")
    model.eval()
    
    metric_input_ids = None
    metric_labels = None
    
    with torch.no_grad():
        initial_ce_losses = []
        
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            logits = model(input_ids)
            
            if i == 0:
                metric_input_ids = input_ids
                metric_labels = labels
            
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
            initial_ce_losses.append(ce_loss.item())
            if len(initial_ce_losses) > 10: break # Estimate from first few batches
        start_ce = sum(initial_ce_losses) / len(initial_ce_losses)
        
    # Initial Metric Calculation
    # Re-compute logits with gradients enabled for metrics that require it (e.g. gradient_entropy)
    with torch.enable_grad():
        metric_logits = model(metric_input_ids)
        start_metric = Metrics.calculate_metric(model, config.METRIC_NAME, metric_logits, metric_labels, metric_input_ids).item()
    print(f"Initial CE loss: {start_ce:.16f}")
    print(f"Initial Metric ({config.METRIC_NAME}): {start_metric:.16f}")
    model.train()
    
    prev_val_loss = None
    current_slope = 0.0
    lmc_weight = 0.0
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        
        train_loss, train_metric, train_combined = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, vocab_size, 
            metric_start=start_metric, ce_start=start_ce, val_error_slope=current_slope, lmc_weight=lmc_weight
        )
        val_loss = validate(model, val_loader, device, vocab_size)
        
        if prev_val_loss is not None:
            raw_slope = val_loss - prev_val_loss
            current_slope = normalize_slope_arctan(raw_slope)
            slope_magnitude = abs(current_slope)
            if current_slope < 0:
                lmc_weight = max(0.0, lmc_weight - slope_magnitude)
            else:
                lmc_weight = min(1.0, lmc_weight + slope_magnitude)
        else:
            current_slope = 0.0
            lmc_weight = 0.0
        
        prev_val_loss = val_loss
        
        test_loss_wiki_epoch = test(model, test_loader_wiki, device, vocab_size)
        test_loss_shakespeare_epoch = test(model, test_loader_shakespeare, device, vocab_size)
        
        metric_val = train_metric
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metric_values.append(metric_val)
        slope_values.append(current_slope)
        lmc_weight_values.append(lmc_weight)
        test_losses_wiki.append(test_loss_wiki_epoch)
        test_losses_shakespeare.append(test_loss_shakespeare_epoch)
        
        print(f"Val Loss: {val_loss:.6f}, Metric: {metric_val:.6f}, Weight: {lmc_weight:.4f}")

    save_results_to_csv(
        output_dir, train_losses, val_losses, metric_values, slope_values, lmc_weight_values,
        test_losses_wiki, test_losses_shakespeare, config, run_num
    )

def run_training(output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}\n")
    
    for run_num in range(1, config.NUM_OF_RUN_PER_CALL + 1):
        print(f"\n{'='*80}")
        print(f"Run {run_num}/{config.NUM_OF_RUN_PER_CALL}")
        print(f"{'='*80}\n")
        
        seed = 42 + run_num
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        run_training_single(output_dir, config, run_num)

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = Config()
    
    # List of experiments
    experiments_list = [
        "Control",
        "-H", "-H.A", "-H.A.B", "-H.A.B.C",
        "-A", "-A.B", "-A.B.C",
        "-B", "-B.C",
        "-C",
        "D", "D.E", "D.E.F",
        "E", "E.F",
        "F",
        "D.H", "D.H.A", "D.H.A.B", "D.H.A.B.C",
        "D.A", "D.A.B", "D.A.B.C",
        "D.B", "D.B.C",
        "D.C",
        "D.E.H", "D.E.H.A", "D.E.H.A.B", "D.E.H.A.B.C",
        "D.E.A", "D.E.A.B", "D.E.A.B.C",
        "D.E.B", "D.E.B.C",
        "D.E.C",
        "D.E.F.H", "D.E.F.H.A", "D.E.F.H.A.B", "D.E.F.H.A.B.C",
        "D.E.F.A", "D.E.F.A.B", "D.E.F.A.B.C",
        "D.E.F.B", "D.E.F.B.C",
        "D.E.F.C",
        "E.H", "E.H.A", "E.H.A.B", "E.H.A.B.C",
        "E.A", "E.A.B", "E.A.B.C",
        "E.B", "E.B.C",
        "E.C",
        "E.F.H", "E.F.H.A", "E.F.H.A.B", "E.F.H.A.B.C",
        "E.F.A", "E.F.A.B", "E.F.A.B.C",
        "E.F.B", "E.F.B.C",
        "E.F.C",
        "F.H", "F.H.A", "F.H.A.B", "F.H.A.B.C",
        "F.A", "F.A.B", "F.A.B.C",
        "F.B", "F.B.C",
        "F.C",
        "D/H", "D/H.A", "D/H.A.B", "D/H.A.B.C",
        "D/A", "D/A.B", "D/A.B.C",
        "D/B", "D/B.C",
        "D/C",
        "D.E/H", "D.E/H.A", "D.E/H.A.B", "D.E/H.A.B.C",
        "D.E/A", "D.E/A.B", "D.E/A.B.C",
        "D.E/B", "D.E/B.C",
        "D.E/C",
        "D.E.F/H", "D.E.F/H.A", "D.E.F/H.A.B", "D.E.F/H.A.B.C",
        "D.E.F/A", "D.E.F/A.B", "D.E.F/A.B.C",
        "D.E.F/B", "D.E.F/B.C",
        "D.E.F/C",
        "E/H", "E/H.A", "E/H.A.B", "E/H.A.B.C",
        "E/A", "E/A.B", "E/A.B.C",
        "E/B", "E/B.C",
        "E/C",
        "E.F/H", "E.F/H.A", "E.F/H.A.B", "E.F/H.A.B.C",
        "E.F/A", "E.F/A.B", "E.F/A.B.C",
        "E.F/B", "E.F/B.C",
        "E.F/C",
        "F/H", "F/H.A", "F/H.A.B", "F/H.A.B.C",
        "F/A", "F/A.B", "F/A.B.C",
        "F/B", "F/B.C",
        "F/C"
    ]
    
    experiments = []
    for exp_name in experiments_list:
        if exp_name == "Control":
            experiments.append((True, 'shannon', 'control'))
        else:
            # Sanitize folder name
            folder_name = exp_name.replace('/', '_over_')
            experiments.append((False, exp_name, folder_name))
    
    for control_mode, metric_name, folder_name in experiments:
        output_dir = os.path.join(script_dir, f'output_mix/{folder_name}')
        
        if os.path.exists(output_dir):
            print(f"Skipping {folder_name} (already exists)")
            continue
            
        print(f"{'='*80}")
        print(f"Running Experiment: {folder_name}")
        print(f"Control Mode: {control_mode}")
        print(f"Metric: {metric_name}")
        print(f"{'='*80}\n")
        
        config.CONTROL_MODE = control_mode
        config.METRIC_NAME = metric_name
        
        run_training(output_dir, config)

if __name__ == '__main__':
    main()
