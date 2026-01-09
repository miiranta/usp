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
    EPOCHS = 30 
    SEQ_LENGTH = 64
    MAX_GRAD_NORM = 1.0
    MAX_SAMPLES = 0 
    
    # Number of runs
    NUM_OF_RUN_PER_CALL = 3
    
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
                    num_bins = min(num_bins, 500)
        
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
    def calculate_metric(model, metric_name, logits=None, labels=None, input_ids=None, features=None, hidden_states=None):
        mapping = {
            'A': 'total_variation', # min
            'B': 'third_order_curvature_norm', # min
            'C': 'hosoya_index', # min
            'D': 'disequilibrium', # max
            #'E': 'dyn_topological_pressure',
            #'F': 'multifractal_spectrum_width',
            'G': 'arithmetic_derivative', # min
            #'H': 'shannon',
            #'I': 'renyi_entropy',
            'J': 'persistent_landscape_norm', # min
            #'K': 'schmidt_rank_proxy',
            #'L': 'empirical_covering_number',
            #'M': 'born_infeld_action',
            #'N': 'graph_spectral_gap_normalized',
            #'O': 'tsallis_divergence_q',
            #'P': 'multi_scale_entropy',
            #'Q': 'amari_alpha_divergence',
            #'R': 'weight_distribution_entropy',
            #'S': 'tsallis_entropy',
            #'T': 'output_bimodality_coefficient',
            'U': 'varentropy', # max
            #'V': 'fisher_info_entropy',
            #'W': 'hessian_eigenvalue_entropy',
            #'X': 'spectral_entropy',
            'Y': 'information_compression_ratio' # max
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
                    val = Metrics._calculate_primitive_metric(model, real_name, logits, labels, input_ids, features, hidden_states)
                    num_val = num_val * val
                    
                den_val = 1.0
                for code in den_str.split('.'):
                    code = code.strip()
                    if not code: continue
                    real_name = mapping.get(code, code)
                    val = Metrics._calculate_primitive_metric(model, real_name, logits, labels, input_ids, features, hidden_states)
                    den_val = den_val * val
                    
                return num_val / (den_val + 1e-10)
            else:
                val = 1.0
                for code in metric_name.split('.'):
                    code = code.strip()
                    if not code: continue
                    real_name = mapping.get(code, code)
                    v = Metrics._calculate_primitive_metric(model, real_name, logits, labels, input_ids, features, hidden_states)
                    val = val * v
                return val
        except Exception as e:
            print(f"Error calculating metric {metric_name}: {e}")
            return torch.tensor(0.0, device=Config.DEVICE)

    @staticmethod
    def _calculate_primitive_metric(model, metric_name, logits=None, labels=None, input_ids=None, features=None, hidden_states=None):
        weights = Metrics.get_all_weights(model)
        device = weights.device
        
        if metric_name == 'shannon':
            probs, _ = Metrics.soft_histogram(weights)
            return -(probs * torch.log(probs)).sum()
            
        elif metric_name == 'renyi_entropy':
            probs, _ = Metrics.soft_histogram(weights)
            alpha = 2.0
            return (1.0 / (1.0 - alpha)) * torch.log((probs ** alpha).sum())

        elif metric_name == 'spectral_entropy':
            n = len(weights)
            if n > 65536:
                indices = torch.randperm(n, device=weights.device)[:65536]
                w_sample = weights[indices]
            else:
                w_sample = weights
            fft_mag = torch.abs(torch.fft.rfft(w_sample))
            psd = fft_mag ** 2
            psd_norm = psd / (psd.sum() + 1e-10)
            psd_norm = torch.clamp(psd_norm, 1e-10, 1.0)
            return -(psd_norm * torch.log(psd_norm)).sum()

        elif metric_name == 'tsallis_entropy':
            probs, _ = Metrics.soft_histogram(weights)
            q = 2.0
            return (1.0 / (q - 1.0)) * (1.0 - (probs ** q).sum())

        elif metric_name == 'weight_distribution_entropy':
            probs, _ = Metrics.soft_histogram(weights)
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'gradient_entropy':
            if input_ids is None or labels is None:
                probs, _ = Metrics.soft_histogram(weights)
                return -(probs * torch.log(probs)).sum()
            
            # Use fresh forward pass to ensure graph connectivity
            # Subsample batch to reduce memory usage
            batch_size = input_ids.size(0)
            max_samples = 2
            if batch_size > max_samples:
                indices = torch.randperm(batch_size, device=device)[:max_samples]
                input_ids_sub = input_ids[indices]
                labels_sub = labels[indices]
            else:
                input_ids_sub = input_ids
                labels_sub = labels

            logits_sub = model(input_ids_sub)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_sub.view(-1, logits_sub.size(-1)), labels_sub.view(-1))
            
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            all_grads = torch.cat([g.view(-1) for g in grads])
            probs, _ = Metrics.soft_histogram(torch.abs(all_grads))
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'fisher_info_entropy':
            if logits is None or labels is None:
                return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            fisher_diag = torch.cat([g.view(-1)**2 for g in grads])
            probs = fisher_diag / (fisher_diag.sum() + 1e-10)
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'multi_scale_entropy':
            n = len(weights)
            scales = [1, 2, 4, 8]
            total_mse = 0.0
            for s in scales:
                if n // s < 10: continue
                w_coarse = weights[:(n//s)*s].view(-1, s).mean(dim=1)
                probs, _ = Metrics.soft_histogram(w_coarse)
                total_mse += -(probs * torch.log(probs)).sum()
            return total_mse / len(scales)

        elif metric_name == 'structural_entropy':
            norms = []
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    norms.append(param.norm())
            norms = torch.stack(norms)
            probs = norms / (norms.sum() + 1e-10)
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'hessian_eigenvalue_entropy':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            fisher_diag = torch.cat([g.view(-1)**2 for g in grads])
            probs = fisher_diag / (fisher_diag.sum() + 1e-10)
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'third_order_curvature_norm':
            if labels is None: return torch.tensor(0.0, device=device)
            if input_ids is not None:
                b_size = 1
                seq_len = min(input_ids.size(1), 64)
                input_ids_small = input_ids[:b_size, :seq_len]
                labels_small = labels[:b_size, :seq_len]
                logits_small = model(input_ids_small)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_small.view(-1, logits_small.size(-1)), labels_small.view(-1))
            else:
                if logits is None: return torch.tensor(0.0, device=device)
                batch_size = logits.shape[0]
                max_samples = 1
                if batch_size > max_samples:
                    logits_subset = logits[:max_samples]
                    labels_subset = labels[:max_samples]
                else:
                    logits_subset = logits
                    labels_subset = labels
                loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_subset.view(-1, logits_subset.size(-1)), labels_subset.view(-1))
            all_params = [p for p in model.parameters() if p.requires_grad]
            if len(all_params) > 5:
                params = all_params[-5:]
            else:
                params = all_params
            grads = torch.autograd.grad(loss, params, create_graph=True)
            v = [torch.randint_like(p, high=2) * 2 - 1 for p in params]
            grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
            Hv = torch.autograd.grad(grad_v, params, create_graph=True)
            vHv = sum([(h * vi).sum() for h, vi in zip(Hv, v)])
            grad_vHv = torch.autograd.grad(vHv, params, create_graph=True)
            norm_grad_vHv = torch.sqrt(sum([(g**2).sum() for g in grad_vHv]))
            return norm_grad_vHv

        elif metric_name == 'empirical_covering_number':
            ent = Metrics.calculate_metric(model, 'shannon', logits, labels, input_ids)
            return torch.exp(ent)

        elif metric_name == 'output_bimodality_coefficient':
            if logits is None: return torch.tensor(0.0, device=device)
            l = logits.view(-1)
            mu = l.mean()
            sigma = l.std()
            skew = ((l - mu)**3).mean() / (sigma**3 + 1e-10)
            kurt = ((l - mu)**4).mean() / (sigma**4 + 1e-10)
            return (skew**2 + 1) / (kurt + 1e-10)

        elif metric_name == 'varentropy':
            if logits is None: return torch.tensor(0.0, device=device)
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
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

        elif metric_name == 'total_variation':
            w = model.lm_head.weight
            diff_h = torch.abs(w[:, 1:] - w[:, :-1]).sum()
            diff_v = torch.abs(w[1:, :] - w[:-1, :]).sum()
            return (diff_h + diff_v) / w.numel()

        elif metric_name == 'controllability_gramian_trace':
            if not hasattr(model, 'last_features'): return torch.tensor(0.0, device=device)
            z = model.last_features
            return torch.norm(z)

        elif metric_name == 'multifractal_spectrum_width':
            weights = model.lm_head.weight
            weights_abs = torch.abs(weights) + 1e-10
            weights_norm = weights_abs / weights_abs.sum()
            q_vals = torch.tensor([-2.0, 2.0], device=weights.device)
            h_q1 = (1.0 / (1.0 - q_vals[0])) * torch.log((weights_norm ** q_vals[0]).sum())
            h_q2 = (1.0 / (1.0 - q_vals[1])) * torch.log((weights_norm ** q_vals[1]).sum())
            return torch.abs(h_q1 - h_q2)

        elif metric_name == 'arithmetic_derivative':
            W = model.lm_head.weight
            f = torch.fft.rfft(W, dim=1)
            return torch.norm(f, p=1) / (torch.norm(f, p=2) + 1e-10)

        elif metric_name == 'hosoya_index':
            W = model.lm_head.weight
            return torch.norm(W, p='nuc')

        elif metric_name == 'born_infeld_action':
            W = model.lm_head.weight
            alpha = 1e-4
            s = torch.linalg.svdvals(W)
            return torch.sum(torch.log(1 + alpha * s**2))

        elif metric_name == 'persistent_landscape_norm':
            W = model.lm_head.weight
            if W.size(0) > 500:
                indices = torch.randperm(W.size(0))[:500]
                W = W[indices]
            dist = torch.cdist(W, W)
            vals = torch.sort(dist.view(-1))[0]
            return torch.norm(vals, p=2)

        elif metric_name == 'amari_alpha_divergence':
            alpha = 0.5
            if logits is None: return torch.tensor(0.0, device=device)
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / probs.size(-1)
            term = (probs ** ((1-alpha)/2)) * (uniform ** ((1+alpha)/2))
            return (4 / (1 - alpha**2)) * (1 - term.sum(dim=-1)).mean()

        elif metric_name == 'tsallis_divergence_q':
            q = 2.0
            if logits is None: return torch.tensor(0.0, device=device)
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / probs.size(-1)
            term = (probs ** q) * (uniform ** (1-q))
            return (1.0 / (q - 1.0)) * (term.sum(dim=-1) - 1.0).mean()

        elif metric_name == 'schmidt_rank_proxy':
            W = model.lm_head.weight
            s = torch.linalg.svdvals(W)
            s_norm = s / (s.norm() + 1e-10)
            return 1.0 / (torch.sum(s_norm ** 4) + 1e-10)

        elif metric_name == 'graph_cheeger_constant_proxy':
            W = model.lm_head.weight
            if W.size(0) > 500:
                indices = torch.randperm(W.size(0))[:500]
                W = W[indices]
            A = torch.abs(W @ W.t())
            mask = torch.eye(A.size(0), device=A.device).bool()
            A.masked_fill_(mask, 0.0)
            deg = A.sum(dim=1)
            d_inv_sqrt = torch.pow(deg + 1e-10, -0.5)
            L = torch.eye(A.size(0), device=A.device) - d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)
            try:
                eigs = torch.linalg.eigvalsh(L)
                if eigs.size(0) > 1:
                    return eigs[1]
            except: pass
            return torch.tensor(0.0, device=device)

        elif metric_name == 'graph_spectral_gap_normalized':
            W = model.lm_head.weight
            s = torch.linalg.svdvals(W)
            if s.size(0) > 1:
                return (s[0] - s[1]) / (s[0] + 1e-10)
            return torch.tensor(0.0, device=device)

        elif metric_name == 'dyn_topological_pressure':
            W = model.lm_head.weight
            s = torch.linalg.svdvals(W)
            return torch.sum(F.relu(torch.log(s + 1e-10)))

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
    
    # 5-epoch schedule
    if epoch < 10:
        lmc_weight = 1.0
    else:
        lmc_weight = 0.0
        
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
    
    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids)
        
        # CE Loss
        loss_fn = nn.CrossEntropyLoss()
        ce_loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        
        # Metric Loss
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
        
        total_loss += combined_loss.item()
        total_ce_loss += ce_loss.item()
        total_metric_loss += metric_val.item()
        
        progress_bar.set_postfix({
            'loss': f"{combined_loss.item():.4f}",
            'ce': f"{ce_loss.item():.4f}",
            'met': f"{metric_val.item():.4f}",
            'w': f"{lmc_weight:.1f}"
        })
        
    return total_loss / len(train_loader), total_ce_loss / len(train_loader), total_metric_loss / len(train_loader)

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
                       test_losses_wiki, test_losses_shakespeare, config, run_num=1):
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
        writer.writerow(['Run Number', f'{run_num}'])
        writer.writerow([])
        
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Test Loss Wiki', 'Test Loss Shakespeare', 'Metric Value', 'Weight'])
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch,
                f'{train_losses[epoch]:.16f}',
                f'{val_losses[epoch]:.16f}',
                f'{test_losses_wiki[epoch]:.16f}',
                f'{test_losses_shakespeare[epoch]:.16f}',
                f'{metric_values[epoch]:.16f}',
                f'{lmc_weight_values[epoch]:.3f}'
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
    
    for epoch in range(config.EPOCHS):
        train_loss, train_ce, train_metric = train_epoch(model, train_loader, optimizer, scheduler, config.DEVICE, config, vocab_size, metric_start, ce_start, epoch)
        val_loss = evaluate(model, val_loader, config.DEVICE, vocab_size)
        
        # Run tests
        test_loss_wiki = test(model, test_loader_wiki, config.DEVICE, vocab_size)
        test_loss_shakespeare = test(model, test_loader_shakespeare, config.DEVICE, vocab_size)
        
        # Values for logging
        # (Note: train_epoch here hardcodes weight schedule, so we replicate it for logging)
        if epoch < 10:
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
        
    save_results_to_csv(output_dir, train_losses, val_losses, metric_values, lmc_weight_values, test_losses_wiki, test_losses_shakespeare, config, run_num)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    vocab_size = tokenizer.vocab_size
    
    train_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.train.tokens')
    val_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.valid.tokens')
    test_path_wiki = os.path.join(script_dir, 'dataset/wikitext-2/wiki.test.tokens')
    test_path_shakespeare = os.path.join(script_dir, 'dataset/tiny-shakespare/test.csv')
    
    if not os.path.exists(train_path):
        print(f"Dataset not found at {train_path}. Creating dummy dataset.")
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        with open(train_path, 'w') as f: f.write("Hello world " * 1000)
        with open(val_path, 'w') as f: f.write("Hello world " * 100)
        with open(test_path_wiki, 'w') as f: f.write("Hello world " * 50)
        with open(test_path_shakespeare, 'w') as f: f.write("text\n" + "Hello world " * 50)
            
    train_dataset = TextDataset(train_path, tokenizer, Config.SEQ_LENGTH, Config.MAX_SAMPLES)
    val_dataset = TextDataset(val_path, tokenizer, Config.SEQ_LENGTH, None)
    test_dataset_wiki = TextDataset(test_path_wiki, tokenizer, Config.SEQ_LENGTH, None)
    test_dataset_shakespeare = TextDataset(test_path_shakespeare, tokenizer, Config.SEQ_LENGTH, None)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader_wiki = DataLoader(test_dataset_wiki, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader_shakespeare = DataLoader(test_dataset_shakespeare, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
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
        
        output_dir = f"output/{clean_name}_{direction}"
        if os.path.exists(output_dir):
            print(f"Skipping {metric_name} ({direction}) - already exists.")
            continue
        
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nRunning experiment: {metric_name} ({direction})")
        Config.METRIC_NAME = metric_name
        
        if metric_name == 'control':
            Config.CONTROL_MODE = True
            # For control, we can use any metric name as placeholder, e.g. 'shannon'
            # but calculate_metric will still be called.
            # To avoid errors if 'control' is passed to calculate_metric (which returns 0.0),
            # we can set METRIC_NAME to 'shannon' but CONTROL_MODE=True ensures it's not used in loss.
            Config.METRIC_NAME = 'shannon' 
        else:
            Config.CONTROL_MODE = False
            
        for run_num in range(1, Config.NUM_OF_RUN_PER_CALL + 1):
            print(f"  Run {run_num}/{Config.NUM_OF_RUN_PER_CALL}")
            run_training_single(output_dir, Config, run_num, tokenizer, vocab_size, train_loader, val_loader, test_loader_wiki, test_loader_shakespeare)

if __name__ == '__main__':
    main()
