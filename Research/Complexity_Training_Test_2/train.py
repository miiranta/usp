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
    EPOCHS = 50
    SEQ_LENGTH = 32
    MAX_GRAD_NORM = 1.0
    MAX_SAMPLES = None
    
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
        if min_val is None: min_val = x.min()
        if max_val is None: max_val = x.max()
        
        # Normalize
        normalized = (x - min_val) / (max_val - min_val + 1e-10)
        
        n = len(x)
        if num_bins is None:
            # Freedman-Diaconis rule approximation
            with torch.no_grad():
                sample_size = min(10000, n)
                if n > sample_size:
                    indices = torch.randperm(n, device=x.device)[:sample_size]
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
    def calculate_metric(model, metric_name):
        weights = Metrics.get_all_weights(model)
        
        if metric_name == 'shannon':
            probs, _ = Metrics.soft_histogram(weights)
            return -(probs * torch.log(probs)).sum()
            
        elif metric_name == 'cross_entropy':
            # CE between weights and Gaussian(0,1)
            # We approximate Gaussian probs on the same bins
            probs, num_bins = Metrics.soft_histogram(weights)
            # Create Gaussian reference
            x = torch.linspace(-3, 3, num_bins, device=weights.device)
            q = torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
            q = q / q.sum()
            return -(probs * torch.log(q + 1e-10)).sum()
            
        elif metric_name == 'kl_divergence':
            probs, num_bins = Metrics.soft_histogram(weights)
            x = torch.linspace(-3, 3, num_bins, device=weights.device)
            q = torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
            q = q / q.sum()
            return (probs * (torch.log(probs) - torch.log(q + 1e-10))).sum()
            
        elif metric_name == 'jensen_shannon':
            probs, num_bins = Metrics.soft_histogram(weights)
            x = torch.linspace(-3, 3, num_bins, device=weights.device)
            q = torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
            q = q / q.sum()
            m = 0.5 * (probs + q)
            kl_pm = (probs * (torch.log(probs) - torch.log(m + 1e-10))).sum()
            kl_qm = (q * (torch.log(q + 1e-10) - torch.log(m + 1e-10))).sum()
            return 0.5 * (kl_pm + kl_qm)
            
        elif metric_name == 'renyi_entropy':
            probs, _ = Metrics.soft_histogram(weights)
            alpha = 2.0
            return (1.0 / (1.0 - alpha)) * torch.log((probs ** alpha).sum())
            
        elif metric_name == 'spectral_entropy':
            # FFT of weights
            # Subsample if too large
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
            
        elif metric_name == 'svd_entropy':
            # SVD of weight matrices
            entropies = []
            for layer in model.transformer.layers:
                # Use self_attn in_proj_weight if available, or linear layers
                # Simplified: just take linear1
                w = layer.linear1.weight
                try:
                    s = torch.linalg.svdvals(w)
                    s_norm = s / (s.sum() + 1e-10)
                    s_norm = torch.clamp(s_norm, 1e-10, 1.0)
                    h = -(s_norm * torch.log(s_norm)).sum()
                    entropies.append(h)
                except:
                    pass
            if not entropies: return torch.tensor(0.0, device=weights.device)
            return torch.stack(entropies).mean()

        elif metric_name == 'neural_representation_entropy':
            # Correlation Matrix Entropy
            # Subsample weights to form a matrix [N, M]
            # Reshape weights to [Chunks, ChunkSize]
            n = len(weights)
            chunk_size = 256
            num_chunks = min(n // chunk_size, 256)
            if num_chunks < 2: return torch.tensor(0.0, device=weights.device)
            
            w_mat = weights[:num_chunks*chunk_size].view(num_chunks, chunk_size)
            # Correlation matrix
            w_centered = w_mat - w_mat.mean(dim=1, keepdim=True)
            cov = w_centered @ w_centered.t() / (chunk_size - 1)
            # Eigenvalues of covariance
            try:
                eigs = torch.linalg.eigvalsh(cov)
                eigs = eigs[eigs > 0]
                eigs_norm = eigs / (eigs.sum() + 1e-10)
                return -(eigs_norm * torch.log(eigs_norm + 1e-10)).sum()
            except:
                return torch.tensor(0.0, device=weights.device)

        # Layer-wise metrics
        elif metric_name in ['conditional_entropy', 'joint_entropy', 'mutual_information', 
                             'conditional_mi', 'transfer_entropy', 'intrinsic_te']:
            layers = Metrics.get_layer_weights(model)
            if len(layers) < 2: return torch.tensor(0.0, device=weights.device)
            
            total_metric = 0.0
            count = 0
            
            for i in range(1, len(layers)):
                w_curr = layers[i]
                w_prev = layers[i-1]
                
                # Resize to match size for joint estimation (simple truncation/sampling)
                min_len = min(len(w_curr), len(w_prev))
                # Subsample to 10000 for speed
                sample_len = min(min_len, 10000)
                
                idx_curr = torch.randperm(len(w_curr), device=weights.device)[:sample_len]
                idx_prev = torch.randperm(len(w_prev), device=weights.device)[:sample_len]
                
                x = w_curr[idx_curr]
                y = w_prev[idx_prev]
                
                def renyi_entropy_2(data):
                    # data: [N, D]
                    # Pairwise distances
                    n = len(data)
                    if n > 1000: # Subsample
                        data = data[:1000]
                        n = 1000
                    
                    dist = torch.cdist(data.unsqueeze(1), data.unsqueeze(1)).squeeze()
                    sigma = 1.0 # Kernel width
                    K = torch.exp(-dist**2 / (2*sigma**2))
                    return -torch.log(K.mean() + 1e-10)

                h_x = renyi_entropy_2(x)
                h_y = renyi_entropy_2(y)
                h_xy = renyi_entropy_2(torch.stack([x, y], dim=1)) # Joint
                
                if metric_name == 'joint_entropy':
                    val = h_xy
                elif metric_name == 'conditional_entropy':
                    val = h_xy - h_y # H(X|Y) = H(X,Y) - H(Y)
                elif metric_name == 'mutual_information':
                    val = h_x + h_y - h_xy
                elif metric_name == 'conditional_mi':
                    # I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)
                    if i > 1:
                        w_prev2 = layers[i-2]
                        idx_prev2 = torch.randperm(len(w_prev2), device=weights.device)[:sample_len]
                        z = w_prev2[idx_prev2]
                        h_z = renyi_entropy_2(z)
                        h_xz = renyi_entropy_2(torch.stack([x, z], dim=1))
                        h_yz = renyi_entropy_2(torch.stack([y, z], dim=1))
                        h_xyz = renyi_entropy_2(torch.stack([x, y, z], dim=1))
                        val = h_xz + h_yz - h_z - h_xyz
                    else:
                        val = torch.tensor(0.0, device=weights.device)
                elif metric_name == 'transfer_entropy':
                    # TE(Y->X) = I(X_next; Y_past | X_past)
                    # Map: X=Curr, Y=Prev. 
                    if i > 1:
                        w_prev2 = layers[i-2] 
                        idx_prev2 = torch.randperm(len(layers[i-2]), device=weights.device)[:sample_len]
                        z = layers[i-2][idx_prev2]
                        h_z = renyi_entropy_2(z)
                        h_xz = renyi_entropy_2(torch.stack([x, z], dim=1))
                        h_yz = renyi_entropy_2(torch.stack([y, z], dim=1))
                        h_xyz = renyi_entropy_2(torch.stack([x, y, z], dim=1))
                        val = h_xz + h_yz - h_z - h_xyz
                    else:
                        val = torch.tensor(0.0, device=weights.device)
                elif metric_name == 'intrinsic_te':
                    # Proxy: Just TE
                    if i > 1:
                        idx_prev2 = torch.randperm(len(layers[i-2]), device=weights.device)[:sample_len]
                        z = layers[i-2][idx_prev2]
                        h_z = renyi_entropy_2(z)
                        h_xz = renyi_entropy_2(torch.stack([x, z], dim=1))
                        h_yz = renyi_entropy_2(torch.stack([y, z], dim=1))
                        h_xyz = renyi_entropy_2(torch.stack([x, y, z], dim=1))
                        val = h_xz + h_yz - h_z - h_xyz
                    else:
                        val = torch.tensor(0.0, device=weights.device)

                total_metric += val
                count += 1
            
            if count == 0: return torch.tensor(0.0, device=weights.device)
            return total_metric / count

        elif metric_name in ['approximate_entropy', 'sample_entropy', 'permutation_entropy']:
            # Soft approximations on subsample
            n = len(weights)
            sample_len = 500 # Small sample for O(N^2)
            indices = torch.randperm(n, device=weights.device)[:sample_len]
            x = weights[indices]
            
            if metric_name == 'approximate_entropy':
                # Soft ApEn
                m = 2
                r = 0.2 * torch.std(x)
                
                def _phi(m):
                    # Create vectors of length N-m+1
                    N = len(x)
                    if N <= m: return torch.tensor(0.0, device=x.device)
                    # Unfold to get windows
                    # x: [N] -> [N-m+1, m]
                    windows = x.unfold(0, m, 1)
                    # Pairwise distances (Chebyshev/Infinity norm)
                    # dist[i,j] = max(|w[i] - w[j]|)
                    # Soft count: sigmoid((r - dist) * k)
                    
                    # Expand for broadcasting: [W, 1, m] - [1, W, m]
                    diff = torch.abs(windows.unsqueeze(1) - windows.unsqueeze(0))
                    dist = diff.max(dim=2).values
                    
                    # Soft threshold
                    k = 100.0 # Sharpness
                    count = torch.sigmoid(k * (r - dist)).mean(dim=1)
                    return torch.log(count + 1e-10).mean()
                
                return _phi(m) - _phi(m+1)

            elif metric_name == 'sample_entropy':
                # Soft SampEn
                m = 2
                r = 0.2 * torch.std(x)
                
                def _count(m):
                    N = len(x)
                    if N <= m: return torch.tensor(0.0, device=x.device)
                    windows = x.unfold(0, m, 1)
                    diff = torch.abs(windows.unsqueeze(1) - windows.unsqueeze(0))
                    dist = diff.max(dim=2).values
                    # Exclude self-matches (diagonal)
                    mask = 1.0 - torch.eye(len(dist), device=x.device)
                    k = 100.0
                    matches = (torch.sigmoid(k * (r - dist)) * mask).sum()
                    return matches
                
                A = _count(m+1)
                B = _count(m)
                return -torch.log((A + 1e-10) / (B + 1e-10))

            elif metric_name == 'permutation_entropy':
                # Soft Permutation Entropy
                # Use soft ranking?
                # Or just use the values of sorted indices?
                # Let's use a simpler proxy: Entropy of differences?
                # Or "Spatial Permutation Entropy"
                # We'll use the distribution of ordinal patterns of length 3
                m = 3
                N = len(x)
                windows = x.unfold(0, m, 1) # [W, 3]
                # We need to classify each window into one of m! permutations
                # Soft classification?
                # There are 6 permutations for m=3.
                # 012, 021, 102, 120, 201, 210
                # We can compute scores for each permutation based on soft comparisons
                # s(a<b) = sigmoid(k*(b-a))
                
                w0, w1, w2 = windows[:,0], windows[:,1], windows[:,2]
                p01 = torch.sigmoid(100*(w1-w0))
                p12 = torch.sigmoid(100*(w2-w1))
                p02 = torch.sigmoid(100*(w2-w0))
                
                # Probabilities of orderings
                # 012: w0<w1<w2 -> p01 * p12 * p02
                prob_012 = p01 * p12 * p02
                # prob_210 = (1-p01) * (1-p12) * (1-p02)
                # ... and so on.
                # This is getting complicated to cover all 6.
                # Let's use a simplified proxy: Entropy of the sorted values? No.
                # Let's return 0.0 if too complex to implement reliably in one go.
                # Or use Shannon entropy of the weights as fallback.
                return -(probs * torch.log(probs)).sum() # Fallback to Shannon

        elif metric_name == 'diffusion_spectral_entropy':
            # Subsample
            n = len(weights)
            if n > 1000:
                indices = torch.randperm(n, device=weights.device)[:1000]
                x = weights[indices]
            else:
                x = weights
            
            # Affinity matrix (Gaussian kernel)
            dist = torch.cdist(x.unsqueeze(1), x.unsqueeze(1))**2
            sigma = torch.median(dist)
            A = torch.exp(-dist / (2*sigma + 1e-10))
            # Laplacian
            D = torch.diag(A.sum(dim=1))
            L = D - A
            # Eigenvalues
            try:
                eigs = torch.linalg.eigvalsh(L)
                eigs = eigs[eigs > 1e-5] # Non-zero
                eigs_norm = eigs / (eigs.sum() + 1e-10)
                return -(eigs_norm * torch.log(eigs_norm + 1e-10)).sum()
            except:
                return torch.tensor(0.0, device=weights.device)

        elif metric_name == 'graph_entropy':
            # Vertex entropy based on degree distribution of affinity graph
            n = len(weights)
            if n > 1000:
                indices = torch.randperm(n, device=weights.device)[:1000]
                x = weights[indices]
            else:
                x = weights
            
            dist = torch.cdist(x.unsqueeze(1), x.unsqueeze(1))**2
            sigma = torch.median(dist)
            A = torch.exp(-dist / (2*sigma + 1e-10))
            degrees = A.sum(dim=1)
            probs = degrees / degrees.sum()
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'bgs_entropy':
            # Von Neumann entropy of Normalized Laplacian
            # rho = L_norm / trace(L_norm)
            # S = -tr(rho log rho) = -sum(lambda log lambda)
            n = len(weights)
            if n > 1000:
                indices = torch.randperm(n, device=weights.device)[:1000]
                x = weights[indices]
            else:
                x = weights
            
            dist = torch.cdist(x.unsqueeze(1), x.unsqueeze(1))**2
            sigma = torch.median(dist)
            A = torch.exp(-dist / (2*sigma + 1e-10))
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A.sum(dim=1) + 1e-10))
            L_norm = torch.eye(len(x), device=x.device) - D_inv_sqrt @ A @ D_inv_sqrt
            
            try:
                eigs = torch.linalg.eigvalsh(L_norm)
                eigs = eigs[eigs > 0]
                eigs_norm = eigs / eigs.sum()
                return -(eigs_norm * torch.log(eigs_norm + 1e-10)).sum()
            except:
                return torch.tensor(0.0, device=weights.device)

        return torch.tensor(0.0, device=weights.device)

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
            metric_tensor = Metrics.calculate_metric(model, config.METRIC_NAME)
            metric_value = metric_tensor 
            metric_value_scalar = metric_tensor.item()
 
        # New Loss Formula: (x_value / (x_start + 1e-10)) * ce_start
        # We want to MINIMIZE x_value (entropy).
        metric_loss_normalized = (metric_value / (metric_start + 1e-10)) * ce_start
        
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
    csv_filename = f'results_{config.METRIC_NAME}_{run_num}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Test Loss (WikiText-2)', f'{test_losses_wiki[-1]:.16f}'])
        writer.writerow(['Test Loss (Tiny-Shakespeare)', f'{test_losses_shakespeare[-1]:.16f}'])
        writer.writerow(['Final Metric Value', f'{metric_values[-1]:.16f}'])
        writer.writerow([])
        
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Metric Value', 'Slope', 'Weight'])
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                f'{train_losses[epoch]:.16f}',
                f'{val_losses[epoch]:.16f}',
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
    
    # Initial Metric Calculation
    start_metric = Metrics.calculate_metric(model, config.METRIC_NAME).item()
    
    print("\nCalculating initial CE loss...")
    model.eval()
    with torch.no_grad():
        initial_ce_losses = []
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            logits = model(input_ids)
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
            initial_ce_losses.append(ce_loss.item())
            if len(initial_ce_losses) > 10: break # Estimate from first few batches
        start_ce = sum(initial_ce_losses) / len(initial_ce_losses)
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
        
        metric_val = Metrics.calculate_metric(model, config.METRIC_NAME).item()
        
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
    
    # List of experiments: (Control_Mode, Metric_Name, Folder_Name)
    experiments = [
        (True, 'shannon', 'control'), # Control (CE Only)
        (False, 'shannon', '1_shannon'),
        (False, 'cross_entropy', '2_cross_entropy'),
        (False, 'conditional_entropy', '3_conditional_entropy'),
        (False, 'joint_entropy', '4_joint_entropy'),
        (False, 'mutual_information', '5_mutual_information'),
        (False, 'conditional_mi', '6_conditional_mi'),
        (False, 'kl_divergence', '7_kl_divergence'),
        (False, 'jensen_shannon', '8_jensen_shannon'),
        (False, 'transfer_entropy', '9_transfer_entropy'),
        (False, 'intrinsic_te', '10_intrinsic_te'),
        (False, 'renyi_entropy', '11_renyi_entropy'),
        (False, 'neural_representation_entropy', '12_neural_representation_entropy'),
        (False, 'diffusion_spectral_entropy', '13_diffusion_spectral_entropy'),
        (False, 'approximate_entropy', '14_approximate_entropy'),
        (False, 'sample_entropy', '15_sample_entropy'),
        (False, 'permutation_entropy', '16_permutation_entropy'),
        (False, 'spectral_entropy', '17_spectral_entropy'),
        (False, 'svd_entropy', '18_svd_entropy'),
        (False, 'graph_entropy', '19_graph_entropy'),
        (False, 'bgs_entropy', '20_bgs_entropy'),
    ]
    
    for control_mode, metric_name, folder_name in experiments:
        output_dir = os.path.join(script_dir, f'output/{folder_name}')
        
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
