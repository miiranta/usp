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
    GPU_INDEX = 1
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
    def calculate_metric(model, metric_name, logits=None, labels=None, input_ids=None):
        weights = Metrics.get_all_weights(model)
        device = weights.device
        
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
                # We need a contiguous sequence to measure order patterns
                n = len(weights)
                limit = min(n, 10000)
                x_seq = weights[:limit]
                
                m = 3
                if len(x_seq) < m: return torch.tensor(0.0, device=weights.device)
                
                windows = x_seq.unfold(0, m, 1) # [W, 3]
                w0, w1, w2 = windows[:,0], windows[:,1], windows[:,2]
                
                k = 100.0
                p01 = torch.sigmoid(k * (w1 - w0))
                p12 = torch.sigmoid(k * (w2 - w1))
                p02 = torch.sigmoid(k * (w2 - w0))
                
                # Complements
                q01 = 1.0 - p01
                q12 = 1.0 - p12
                q02 = 1.0 - p02
                
                # 6 Permutations probabilities (approximate)
                # 012: 0<1<2
                s_012 = p01 * p12
                # 021: 0<2<1
                s_021 = p02 * q12
                # 102: 1<0<2
                s_102 = q01 * p02
                # 120: 1<2<0
                s_120 = p12 * q02
                # 201: 2<0<1
                s_201 = q02 * p01
                # 210: 2<1<0
                s_210 = q12 * q01
                
                scores = torch.stack([s_012, s_021, s_102, s_120, s_201, s_210], dim=1)
                probs_window = scores / (scores.sum(dim=1, keepdim=True) + 1e-10)
                global_counts = probs_window.sum(dim=0)
                global_probs = global_counts / (global_counts.sum() + 1e-10)
                
                return -(global_probs * torch.log(global_probs + 1e-10)).sum()

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

        elif metric_name == 'tsallis_entropy':
            # 21. Tsallis Entropy (q=2)
            probs, _ = Metrics.soft_histogram(weights)
            q = 2.0
            return (1.0 / (q - 1.0)) * (1.0 - (probs ** q).sum())

        elif metric_name == 'von_neumann_entropy':
            # 22. von Neumann Entropy (of weight correlation matrix)
            n = len(weights)
            chunk_size = 256
            num_chunks = min(n // chunk_size, 256)
            if num_chunks < 2: return torch.tensor(0.0, device=device)
            w_mat = weights[:num_chunks*chunk_size].view(num_chunks, chunk_size)
            # Density matrix rho = W W^T / tr(W W^T)
            rho = w_mat @ w_mat.t()
            rho = rho / (torch.trace(rho) + 1e-10)
            try:
                eigs = torch.linalg.eigvalsh(rho)
                eigs = eigs[eigs > 0]
                return -(eigs * torch.log(eigs + 1e-10)).sum()
            except: return torch.tensor(0.0, device=device)

        elif metric_name == 'path_entropy':
            # 23. Path Entropy (Proxy: Entropy of product of layer norms)
            norms = []
            for layer in model.transformer.layers:
                norms.append(torch.norm(layer.linear1.weight))
                norms.append(torch.norm(layer.linear2.weight))
            norms = torch.stack(norms)
            probs = norms / (norms.sum() + 1e-10)
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'layer_activation_entropy':
            # 24. Layer-wise Activation Entropy (Proxy: Embedding Entropy)
            w = model.embedding.weight
            probs, _ = Metrics.soft_histogram(w.view(-1))
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'weight_distribution_entropy':
            # 25. Weight Distribution Entropy (Same as Shannon but explicit)
            probs, _ = Metrics.soft_histogram(weights)
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'gradient_entropy':
            # 26. Gradient Entropy (Proxy: Entropy of gradient magnitudes)
            if logits is None or labels is None:
                # Fallback to weight entropy
                probs, _ = Metrics.soft_histogram(weights)
                return -(probs * torch.log(probs)).sum()
            
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            all_grads = torch.cat([g.view(-1) for g in grads])
            probs, _ = Metrics.soft_histogram(torch.abs(all_grads))
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'fisher_info_entropy':
            # 27. Fisher Information Entropy (Diagonal approximation)
            if logits is None or labels is None:
                return torch.tensor(0.0, device=device)
            
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            fisher_diag = torch.cat([g.view(-1)**2 for g in grads])
            probs = fisher_diag / (fisher_diag.sum() + 1e-10)
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'prediction_entropy':
            # 28. Prediction Entropy (Entropy of output probabilities)
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            return entropy

        elif metric_name == 'bayesian_entropy':
            # 29. Bayesian Entropy (Proxy: Entropy of weights + Entropy of predictions)
            # H(w) ~ sum log(|w| * 0.1)
            h_w = torch.log(torch.abs(weights) * 0.1 + 1e-10).mean()
            
            if logits is not None:
                probs = torch.softmax(logits, dim=-1)
                h_y = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                return h_w + h_y
            return h_w

        elif metric_name == 'attention_entropy':
            # 30. Attention Entropy (Entropy of attention parameters)
            entropies = []
            for layer in model.transformer.layers:
                # Try to access self_attn weights if available, else linear1
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'in_proj_weight'):
                    w = layer.self_attn.in_proj_weight
                elif hasattr(layer, 'linear1'):
                    w = layer.linear1.weight
                else:
                    continue
                
                probs, _ = Metrics.soft_histogram(w.view(-1))
                entropies.append(-(probs * torch.log(probs)).sum())
            
            if not entropies: return torch.tensor(0.0, device=device)
            return torch.stack(entropies).mean()

        elif metric_name == 'token_entropy':
            # 31. Token Entropy (Same as Prediction Entropy)
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        elif metric_name == 'entropy_rate':
            # 32. Entropy Rate (Conditional entropy of next token given previous)
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        elif metric_name == 'multi_scale_entropy':
            # 33. Multi-Scale Entropy (Coarse-grained weights)
            n = len(weights)
            scales = [1, 2, 4, 8]
            total_mse = 0.0
            for s in scales:
                if n // s < 10: continue
                w_coarse = weights[:(n//s)*s].view(-1, s).mean(dim=1)
                probs, _ = Metrics.soft_histogram(w_coarse)
                total_mse += -(probs * torch.log(probs)).sum()
            return total_mse / len(scales)

        elif metric_name == 'conditional_output_entropy':
            # 34. Conditional Output Entropy (H(Y|X))
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        elif metric_name == 'effective_dimensionality_entropy':
            # 35. Effective Dimensionality Entropy (SVD of weights)
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                s_norm = s / (s.sum() + 1e-10)
                return -(s_norm * torch.log(s_norm + 1e-10)).sum()
            except: return torch.tensor(0.0, device=device)

        elif metric_name == 'latent_entropy':
            # 36. Latent Entropy (Entropy of embedding space)
            w = model.embedding.weight
            probs, _ = Metrics.soft_histogram(w.view(-1))
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'structural_entropy':
            # 38. Structural Entropy (Graph entropy of layer connections)
            norms = []
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    norms.append(param.norm())
            norms = torch.stack(norms)
            probs = norms / (norms.sum() + 1e-10)
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'topological_entropy':
            # 39. Topological Entropy (Spectral radius proxy)
            max_eigs = []
            for layer in model.transformer.layers:
                w = layer.linear1.weight
                try:
                    v = torch.randn(w.size(1), 1, device=device)
                    v = v / v.norm()
                    for _ in range(5):
                        v = w.t() @ (w @ v)
                        v = v / v.norm()
                    spectral_radius = torch.norm(w @ v)
                    max_eigs.append(spectral_radius)
                except: pass
            if not max_eigs: return torch.tensor(0.0, device=device)
            eigs = torch.stack(max_eigs)
            probs = eigs / (eigs.sum() + 1e-10)
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'differential_entropy':
            # 41. Differential Entropy (Gaussian assumption)
            var = torch.var(weights)
            return 0.5 * torch.log(2 * math.pi * math.e * var + 1e-10)

        elif metric_name == 'log_determinant_entropy':
            # 42. Log-Determinant Entropy
            n = len(weights)
            chunk_size = 256
            num_chunks = min(n // chunk_size, 256)
            if num_chunks < 2: return torch.tensor(0.0, device=device)
            w_mat = weights[:num_chunks*chunk_size].view(num_chunks, chunk_size)
            w_centered = w_mat - w_mat.mean(dim=1, keepdim=True)
            cov = w_centered @ w_centered.t() / (chunk_size - 1)
            try:
                eigs = torch.linalg.eigvalsh(cov)
                eigs = eigs[eigs > 1e-6]
                return 0.5 * torch.log(eigs).sum()
            except: return torch.tensor(0.0, device=device)

        elif metric_name == 'gaussian_entropy':
            # 43. Gaussian Entropy (Same as Differential)
            var = torch.var(weights)
            return 0.5 * torch.log(2 * math.pi * math.e * var + 1e-10)

        elif metric_name == 'kde_entropy':
            # 46. KDE Entropy (Resubstitution entropy)
            n = len(weights)
            if n > 1000:
                indices = torch.randperm(n, device=device)[:1000]
                x = weights[indices]
            else: x = weights
            dist = torch.cdist(x.unsqueeze(1), x.unsqueeze(1))**2
            sigma = torch.std(x) * (n**(-0.2))
            p = torch.exp(-dist / (2*sigma**2 + 1e-10)).mean(dim=1) / (sigma * math.sqrt(2*math.pi) + 1e-10)
            return -torch.log(p + 1e-10).mean()

        elif metric_name == 'copula_entropy':
            # 47. Copula Entropy (Proxy: Sum of marginal entropies - Joint entropy)
            w = model.embedding.weight
            n = min(len(w), 500)
            idx = torch.randperm(len(w), device=device)[:n]
            x = w[idx] # [N, D]
            
            # H(X) via log det cov
            cov = torch.cov(x.t())
            h_joint = 0.5 * torch.logdet(cov + 1e-6*torch.eye(cov.size(0), device=device))
            
            # sum H(X_i) via variances
            h_marginals = 0.5 * torch.log(torch.diag(cov) + 1e-6).sum()
            
            return h_marginals - h_joint

        elif metric_name == 'cumulative_residual_entropy':
            # 48. Cumulative Residual Entropy
            probs, num_bins = Metrics.soft_histogram(weights)
            cdf = torch.cumsum(probs, dim=0)
            survival = 1.0 - cdf
            survival = torch.clamp(survival, 1e-10, 1.0)
            return -(survival * torch.log(survival)).sum()

        elif metric_name == 'quadratic_entropy':
            # 49. Quadratic Entropy (Rao's Quadratic Entropy)
            probs, num_bins = Metrics.soft_histogram(weights)
            indices = torch.arange(num_bins, device=device).float()
            dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
            return (probs.unsqueeze(0) * probs.unsqueeze(1) * dist).sum()

        elif metric_name == 'energy_entropy':
            # 50. Energy Entropy (LogSumExp of logits)
            if logits is None: return torch.tensor(0.0, device=device)
            energies = -torch.logsumexp(logits, dim=-1)
            probs, _ = Metrics.soft_histogram(energies)
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'logit_distribution_entropy':
            # 51. Logit Distribution Entropy
            if logits is None: return torch.tensor(0.0, device=device)
            probs, _ = Metrics.soft_histogram(logits.view(-1))
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'softmax_temperature_entropy':
            # 52. Softmax Temperature-Scaled Entropy (High Temp)
            if logits is None: return torch.tensor(0.0, device=device)
            T = 2.0
            probs = torch.softmax(logits / T, dim=-1)
            return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        elif metric_name == 'renyi_divergence':
            # 53. Renyi Divergence
            probs, num_bins = Metrics.soft_histogram(weights)
            x = torch.linspace(-3, 3, num_bins, device=device)
            q = torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
            q = q / q.sum()
            alpha = 2.0
            return (1.0 / (alpha - 1.0)) * torch.log((probs**alpha * q**(1-alpha)).sum())

        elif metric_name == 'wasserstein_entropy':
            # 55. Wasserstein Entropy (Wasserstein distance to Gaussian)
            probs, num_bins = Metrics.soft_histogram(weights)
            cdf_p = torch.cumsum(probs, dim=0)
            x = torch.linspace(-3, 3, num_bins, device=device)
            q = torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
            q = q / q.sum()
            cdf_q = torch.cumsum(q, dim=0)
            return torch.abs(cdf_p - cdf_q).sum()

        elif metric_name == 'flow_entropy':
            # 57. Flow Entropy (Proxy: Smoothness of weights across layers)
            layers = Metrics.get_layer_weights(model)
            val = 0.0
            count = 0
            for i in range(1, len(layers)):
                w1 = layers[i-1]
                w2 = layers[i]
                n = min(len(w1), len(w2))
                diff = w1[:n] - w2[:n]
                val += (diff**2).mean()
                count += 1
            if count == 0: return torch.tensor(0.0, device=device)
            return val / count

        elif metric_name == 'spike_entropy':
            # 58. Spike Entropy (Entropy of soft spikes)
            mu = weights.mean()
            sigma = weights.std()
            threshold = mu + 2 * sigma
            spikes = torch.sigmoid(100.0 * (weights - threshold)) # Soft spike indicator
            prob_spike = spikes.mean()
            # Binary entropy
            p = prob_spike
            return -(p * torch.log(p + 1e-10) + (1-p) * torch.log(1-p + 1e-10))

        elif metric_name == 'class_conditional_entropy':
            # 59. Class-Conditional Distribution Entropy
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            mask = labels != -100
            valid_logits = logits.view(-1, logits.size(-1))[mask.view(-1)]
            valid_labels = labels.view(-1)[mask.view(-1)]
            if len(valid_labels) == 0: return torch.tensor(0.0, device=device)
            classes = torch.unique(valid_labels)
            total_entropy = 0.0
            for c in classes:
                class_logits = valid_logits[valid_labels == c]
                probs = torch.softmax(class_logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                total_entropy += entropy
            return total_entropy / len(classes)

        elif metric_name == 'hessian_trace':
            # Hutchinson estimator: Tr(H) = E[v^T H v]
            # v ~ Rademacher (+-1)
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            
            # Subsample batch to save memory for 3rd order derivatives
            batch_size = logits.shape[0]
            max_samples = 2
            if batch_size > max_samples:
                logits_subset = logits[:max_samples]
                labels_subset = labels[:max_samples]
            else:
                logits_subset = logits
                labels_subset = labels
                
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_subset.view(-1, logits_subset.size(-1)), labels_subset.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            # Generate random vector v
            v = [torch.randint_like(p, high=2) * 2 - 1 for p in params]
            
            # Compute Hv = grad(grads * v)
            # grads * v is dot product
            grad_v = sum([(g * v_i).sum() for g, v_i in zip(grads, v)])
            Hv = torch.autograd.grad(grad_v, params, create_graph=True)
            
            # v^T H v
            trace_est = sum([(h * v_i).sum() for h, v_i in zip(Hv, v)])
            return trace_est

        elif metric_name == 'hessian_spectral_radius':
            # Power iteration
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            
            # Subsample batch to save memory
            batch_size = logits.shape[0]
            max_samples = 2
            if batch_size > max_samples:
                logits_subset = logits[:max_samples]
                labels_subset = labels[:max_samples]
            else:
                logits_subset = logits
                labels_subset = labels
                
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_subset.view(-1, logits_subset.size(-1)), labels_subset.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            v = [torch.randn_like(p) for p in params]
            # Normalize v
            v_norm = torch.sqrt(sum([(vi**2).sum() for vi in v]))
            v = [vi / (v_norm + 1e-10) for vi in v]
            
            for _ in range(5): # Few iterations
                grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
                Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
                # Normalize
                v_norm = torch.sqrt(sum([(hvi**2).sum() for hvi in Hv]))
                v = [hvi / (v_norm + 1e-10) for hvi in Hv]
                
            # Rayleigh quotient: v^T H v / v^T v (v is normalized)
            grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
            Hv = torch.autograd.grad(grad_v, params, create_graph=True)
            spectral_radius = sum([(hvi * vi).sum() for hvi, vi in zip(Hv, v)])
            return torch.abs(spectral_radius)

        elif metric_name == 'fisher_info_condition_number':
            # Fisher Information Matrix Condition Number
            # F = E[g g^T]
            # We can approximate F diagonal or block diagonal.
            # Let's use diagonal approximation.
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            # Diagonal Fisher: g^2
            fisher_diag = torch.cat([g.view(-1)**2 for g in grads])
            f_max = fisher_diag.max()
            f_min = fisher_diag.min()
            return f_max / (f_min + 1e-10)

        elif metric_name == 'sharpness_perturbation':
            # Maximize loss in epsilon ball
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            
            # Subsample batch to save memory
            batch_size = logits.shape[0]
            max_samples = 2
            if batch_size > max_samples:
                logits_subset = logits[:max_samples]
                labels_subset = labels[:max_samples]
            else:
                logits_subset = logits
                labels_subset = labels
                
            epsilon = 0.01
            loss_orig = nn.CrossEntropyLoss(ignore_index=-100)(logits_subset.view(-1, logits_subset.size(-1)), labels_subset.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss_orig, params, create_graph=True)
            
            # Perturbation direction = sign(grad) for L_inf or grad/norm for L_2
            # Use L_2
            grad_norm = torch.sqrt(sum([(g**2).sum() for g in grads]))
            perturbation = [epsilon * g / (grad_norm + 1e-10) for g in grads]
            
            # We can't easily change weights in-place and backprop through it in this framework without functional call
            # But we can approximate L(theta+delta) ~ L(theta) + g^T delta + 0.5 delta^T H delta
            # S(eps) ~ epsilon * |g| + 0.5 * epsilon^2 * (g^T H g) / |g|^2
            # Let's use the first order approximation + second order term
            
            # g^T delta = g^T (eps * g / |g|) = eps * |g|
            term1 = epsilon * grad_norm
            
            # delta^T H delta = (eps * g / |g|)^T H (eps * g / |g|) = eps^2 / |g|^2 * (g^T H g)
            # Compute H g
            grad_v = sum([(g * gi).sum() for g, gi in zip(grads, grads)]) # g^T g
            Hg = torch.autograd.grad(grad_v, params, create_graph=True) # H g * 2 ?? No.
            # grad(g^T v) = H v. Here v=g.
            # But grad(g^T g) = 2 H g.
            Hg = [0.5 * h for h in Hg]
            
            gHg = sum([(hg * g).sum() for hg, g in zip(Hg, grads)])
            term2 = 0.5 * (epsilon**2) * gHg / (grad_norm**2 + 1e-10)
            
            return term1 + term2

        elif metric_name == 'pac_bayes_flatness':
            # E[L(theta+delta)] - L(theta)
            # delta ~ N(0, sigma^2 I)
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            sigma = 0.01
            # Second order approximation: 0.5 * Tr(H) * sigma^2
            trace = Metrics.calculate_metric(model, 'hessian_trace', logits, labels, input_ids)
            return 0.5 * trace * (sigma**2)

        elif metric_name == 'gradient_covariance_trace':
            # Tr(Cov(g))
            # Requires per-sample gradients.
            # Proxy: Variance of gradients across batch?
            # If we can't get per-sample, we can't compute covariance properly.
            # But we can use the "Gradient Noise" proxy.
            # S_noise = ||g_batch||^2 - 1/B sum ||g_i||^2 ... hard without per-sample.
            # Let's return 0.0 or use a proxy.
            # Proxy: ||g_batch||^2
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_norm_sq = sum([(g**2).sum() for g in grads])
            return grad_norm_sq

        elif metric_name == 'gradient_direction_entropy':
            # Entropy of gradient direction
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            all_grads = torch.cat([g.view(-1) for g in grads])
            abs_grads = torch.abs(all_grads)
            probs = abs_grads / (abs_grads.sum() + 1e-10)
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'gradient_cosine_drift':
            # 1 - cos(g_t, g_{t-1})
            # We don't have g_{t-1}.
            # Proxy: 1 - cos(g_batch1, g_batch2) using split batch?
            # Split batch in half.
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            batch_size = logits.size(0)
            if batch_size < 2: return torch.tensor(0.0, device=device)
            
            half = batch_size // 2
            l1 = logits[:half]
            y1 = labels[:half]
            l2 = logits[half:]
            y2 = labels[half:]
            
            loss1 = nn.CrossEntropyLoss(ignore_index=-100)(l1.view(-1, l1.size(-1)), y1.view(-1))
            loss2 = nn.CrossEntropyLoss(ignore_index=-100)(l2.view(-1, l2.size(-1)), y2.view(-1))
            
            params = [p for p in model.parameters() if p.requires_grad]
            # Both gradient computations need create_graph=True to be differentiable
            grads1 = torch.autograd.grad(loss1, params, create_graph=True)
            grads2 = torch.autograd.grad(loss2, params, create_graph=True)
            
            g1 = torch.cat([g.view(-1) for g in grads1])
            g2 = torch.cat([g.view(-1) for g in grads2])
            
            cos = nn.CosineSimilarity(dim=0)(g1, g2)
            return 1.0 - cos

        elif metric_name == 'activation_jacobian_frobenius_norm':
            # ||J_x||_F
            # Proxy: Product of Frobenius norms of weights
            val = 1.0
            for layer in model.transformer.layers:
                val *= torch.norm(layer.linear1.weight)
            return val

        elif metric_name == 'layerwise_lipschitz':
            # Product of spectral norms of weights
            prod = 1.0
            for layer in model.transformer.layers:
                w1 = layer.linear1.weight
                try:
                    s1 = torch.linalg.svdvals(w1)[0]
                    prod *= s1
                except: pass
            return torch.tensor(prod, device=device)

        elif metric_name == 'effective_rank_activations':
            # Entropy of singular values of activations
            # Need activations.
            # We can hook the model to get activations.
            # Or just use embeddings as proxy.
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                s = s / (s.sum() + 1e-10)
                return -(s * torch.log(s + 1e-10)).sum()
            except: return torch.tensor(0.0, device=device)

        elif metric_name == 'log_det_activation_covariance':
            # log det (Cov(a))
            # Use embeddings as proxy
            w = model.embedding.weight
            n = w.size(0)
            cov = w.t() @ w / n
            try:
                return torch.logdet(cov + 1e-6 * torch.eye(cov.size(0), device=device))
            except: return torch.tensor(0.0, device=device)

        elif metric_name == 'class_conditional_overlap':
            # Trace(Sigma_c Sigma_c')
            # Proxy: Overlap of class means
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            mask = labels != -100
            z = logits.view(-1, logits.size(-1))[mask.view(-1)]
            y = labels.view(-1)[mask.view(-1)]
            classes = torch.unique(y)
            if len(classes) < 2: return torch.tensor(0.0, device=device)
            
            means = []
            for c in classes:
                zc = z[y==c]
                if len(zc) > 0: means.append(zc.mean(dim=0))
            
            if not means: return torch.tensor(0.0, device=device)
            means = torch.stack(means)
            # Overlap = mean dot product of means
            overlap = (means @ means.t()).mean()
            return overlap

        elif metric_name == 'information_compression_ratio':
            # I(X;T) / I(T;Y)
            # Proxy: H(Input) / I(Input; Output)
            # H(Input) is constant.
            # Just 1 / MI
            mi = Metrics.calculate_metric(model, 'mutual_information', logits, labels, input_ids)
            return 1.0 / (mi + 1e-10)

        elif metric_name == 'trajectory_length':
            # Length of weight trajectory
            # We don't have history.
            # Proxy: Norm of gradient (velocity)
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_norm = torch.sqrt(sum([(g**2).sum() for g in grads]))
            return grad_norm

        elif metric_name == 'stochastic_loss_variance':
            # Var(L_i)
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            losses = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return torch.var(losses)

        elif metric_name == 'local_linearization_error':
            # |L(theta+d) - (L(theta) + g^T d)|
            # Proxy: Second order term 0.5 d^T H d
            # Use Hessian Trace * eps^2
            trace = Metrics.calculate_metric(model, 'hessian_trace', logits, labels, input_ids)
            return 0.5 * trace * 0.0001

        elif metric_name == 'output_jacobian_condition_number':
            # Cond(J_y)
            # Proxy: Cond(Cov(logits))
            if logits is None: return torch.tensor(0.0, device=device)
            l = logits.view(-1, logits.size(-1))
            
            # Subsample vocabulary if too large to avoid OOM
            if l.size(1) > 4000:
                indices = torch.randperm(l.size(1), device=device)[:4000]
                l = l[:, indices]
            
            cov = l.t() @ l
            return torch.linalg.cond(cov + 1e-6*torch.eye(cov.size(0), device=device))

        elif metric_name == 'mdl_surrogate':
            # Sum log det (W^T W + eps I)
            val = 0.0
            for layer in model.transformer.layers:
                w = layer.linear1.weight
                try:
                    val += torch.logdet(w.t() @ w + 1e-6 * torch.eye(w.size(1), device=device))
                except: pass
            return torch.tensor(val, device=device)

        elif metric_name == 'kolmogorov_complexity_proxy':
            # Sum H(sigma(W))
            val = 0.0
            for layer in model.transformer.layers:
                w = layer.linear1.weight
                try:
                    s = torch.linalg.svdvals(w)
                    s = s / (s.sum() + 1e-10)
                    val += -(s * torch.log(s + 1e-10)).sum()
                except: pass
            return torch.tensor(val, device=device)

        elif metric_name == 'fractal_dimension':
            # Correlation dimension of embeddings
            w = model.embedding.weight
            n = min(len(w), 1000)
            idx = torch.randperm(len(w), device=device)[:n]
            x = w[idx]
            dist = torch.cdist(x, x)
            # Correlation integral C(r)
            r_vals = torch.logspace(-1, 1, 10, device=device)
            c_vals = []
            for r in r_vals:
                # Soft count: sigmoid(k * (r - dist))
                # k controls sharpness.
                k = 10.0
                c = torch.sigmoid(k * (r - dist)).mean()
                c_vals.append(c)
            c_vals = torch.stack(c_vals)
            mask = c_vals > 1e-5 # Avoid log(0)
            if mask.sum() < 2: return torch.tensor(0.0, device=device)
            log_c = torch.log(c_vals[mask])
            log_r = torch.log(r_vals[mask])
            # Slope
            mean_x = log_r.mean()
            mean_y = log_c.mean()
            num = ((log_r - mean_x) * (log_c - mean_y)).sum()
            den = ((log_r - mean_x)**2).sum()
            return num / (den + 1e-10)

        elif metric_name == 'ntk_condition_number':
            # Proxy: Condition number of embedding covariance
            w = model.embedding.weight
            cov = w.t() @ w
            return torch.linalg.cond(cov + 1e-6*torch.eye(cov.size(0), device=device))

        elif metric_name == 'ntk_trace':
            # Trace of NTK = sum ||g(x)||^2
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            norm_sq = sum([(g**2).sum() for g in grads])
            return norm_sq

        elif metric_name == 'rademacher_complexity':
            # E[sup 1/n sum sigma_i f(x_i)]
            # Random signs sigma
            if logits is None: return torch.tensor(0.0, device=device)
            sigma = torch.randint(0, 2, (logits.size(0),), device=device) * 2 - 1
            # We want to maximize sum sigma_i f(x_i) w.r.t theta? No, R_hat is fixed for theta.
            # It measures the correlation with random noise.
            # Just compute sum sigma_i * logits_i
            # logits: [B, V]
            # We sum over batch.
            # For each class? Or max over classes?
            # Usually max over classes.
            # Flatten logits to handle [B, S, V] case by taking max over all outputs for a sample
            val = (logits.view(logits.size(0), -1).max(dim=1).values * sigma).mean()
            return val

        elif metric_name == 'pac_bayes_kl':
            # KL(q || p)
            # q = N(theta, sigma^2), p = N(0, lambda^2)
            # KL = log(lambda/sigma) + (sigma^2 + theta^2)/(2 lambda^2) - 0.5
            # Sum over all params
            sigma = 0.01
            prior_sigma = 1.0
            weights_sq = sum([(p**2).sum() for p in model.parameters()])
            n_params = sum([p.numel() for p in model.parameters()])
            kl = n_params * (math.log(prior_sigma/sigma) + 0.5 * (sigma**2 / prior_sigma**2) - 0.5) + weights_sq / (2 * prior_sigma**2)
            return kl

        elif metric_name == 'activation_total_correlation':
            # TC(Z) = sum H(Z_i) - H(Z)
            # Use embeddings
            w = model.embedding.weight
            # H(Z) approx by log det cov
            cov = torch.cov(w.t())
            h_z = 0.5 * torch.logdet(cov + 1e-6*torch.eye(cov.size(0), device=device))
            # sum H(Z_i) approx by sum log var
            h_zi = 0.5 * torch.log(torch.diag(cov) + 1e-6).sum()
            return h_zi - h_z

        elif metric_name == 'class_conditional_activation_kl':
            # KL(p(z|c) || p(z))
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            # Use logits as proxy for z
            mask = labels != -100
            z = logits.view(-1, logits.size(-1))[mask.view(-1)]
            y = labels.view(-1)[mask.view(-1)]
            if len(y) == 0: return torch.tensor(0.0, device=device)
            
            # Global mean/cov
            mu = z.mean(dim=0)
            # Use diagonal covariance to save memory
            var = z.var(dim=0, unbiased=True) + 1e-6
            
            kl_sum = 0.0
            classes = torch.unique(y)
            for c in classes:
                zc = z[y==c]
                if len(zc) < 2: continue
                mu_c = zc.mean(dim=0)
                var_c = zc.var(dim=0, unbiased=True) + 1e-6
                
                # KL(N_c || N) for diagonal Gaussians
                term1 = torch.log(var).sum() - torch.log(var_c).sum()
                term2 = (var_c / var).sum()
                term3 = ((mu_c - mu)**2 / var).sum()
                kl = 0.5 * (term1 - z.size(1) + term2 + term3)
                kl_sum += kl
            return kl_sum / len(classes)

        elif metric_name == 'energy_distance_class':
            # Energy distance between classes
            # Proxy: Distance between class means
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            mask = labels != -100
            z = logits.view(-1, logits.size(-1))[mask.view(-1)]
            y = labels.view(-1)[mask.view(-1)]
            classes = torch.unique(y)
            if len(classes) < 2: return torch.tensor(0.0, device=device)
            
            means = []
            for c in classes:
                zc = z[y==c]
                if len(zc) > 0: means.append(zc.mean(dim=0))
            
            if not means: return torch.tensor(0.0, device=device)
            means = torch.stack(means)
            dist = torch.cdist(means, means)
            return dist.mean()

        elif metric_name == 'wasserstein_barycenter_dispersion':
            # Variance of Wasserstein distances to barycenter
            # Proxy: Variance of class means
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            mask = labels != -100
            z = logits.view(-1, logits.size(-1))[mask.view(-1)]
            y = labels.view(-1)[mask.view(-1)]
            classes = torch.unique(y)
            if len(classes) < 2: return torch.tensor(0.0, device=device)
            
            means = []
            for c in classes:
                zc = z[y==c]
                if len(zc) > 0: means.append(zc.mean(dim=0))
            
            if not means: return torch.tensor(0.0, device=device)
            means = torch.stack(means)
            center = means.mean(dim=0)
            dispersion = ((means - center)**2).sum(dim=1).mean()
            return dispersion

        elif metric_name == 'entropy_rate_activations':
            # H(Z_t | Z_{t-1})
            # Use embeddings sequence
            if input_ids is None: return torch.tensor(0.0, device=device)
            z = model.embedding(input_ids) # [B, S, H]
            # Linear prediction error entropy?
            # Or just MI(Z_t; Z_{t-1})
            # Let's use cosine similarity entropy
            z_curr = z[:, 1:, :]
            z_prev = z[:, :-1, :]
            cos = nn.CosineSimilarity(dim=-1)(z_curr, z_prev)
            probs, _ = Metrics.soft_histogram(cos.view(-1))
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'lyapunov_exponent':
            # Log of largest singular value of Jacobian product
            # Proxy: Sum of log spectral norms of layers
            val = 0.0
            for layer in model.transformer.layers:
                w = layer.linear1.weight
                try:
                    s = torch.linalg.svdvals(w)[0]
                    val += torch.log(s + 1e-10)
                except: pass
            return val

        elif metric_name == 'topological_persistence_entropy':
            # Entropy of persistence diagram
            # Proxy: Entropy of pairwise distances of embeddings
            w = model.embedding.weight
            n = min(len(w), 500)
            idx = torch.randperm(len(w), device=device)[:n]
            x = w[idx]
            dist = torch.cdist(x, x).view(-1)
            probs, _ = Metrics.soft_histogram(dist)
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'info_geometric_volume':
            # Volume of Fisher Information Metric
            # Proxy: LogDet of Fisher (diagonal)
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            fisher_diag = torch.cat([g.view(-1)**2 for g in grads])
            return 0.5 * torch.log(fisher_diag + 1e-10).sum()

        elif metric_name == 'vc_dimension_proxy':
            # R^2 / M^2 (Radius / Margin)
            # Radius of inputs, Margin of classifier
            if logits is None or input_ids is None: return torch.tensor(0.0, device=device)
            z = model.embedding(input_ids).view(-1, Config.HIDDEN_DIM)
            radius = torch.norm(z, dim=1).max()
            # Margin: min_i (y_i * f(x_i)) ... hard for multiclass
            # Proxy: Average confidence margin (prob_correct - prob_runner_up)
            probs = torch.softmax(logits, dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margin = (top2[:,:,0] - top2[:,:,1]).mean()
            return (radius**2) / (margin**2 + 1e-10)

        elif metric_name == 'spectral_decay_rate':
            # alpha s.t. sigma_k ~ k^-alpha
            # Fit log(sigma_k) = -alpha * log(k) + c
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                k = torch.arange(1, len(s)+1, device=device).float()
                # Linear regression
                log_s = torch.log(s + 1e-10)
                log_k = torch.log(k)
                # slope = cov(x,y)/var(x)
                mean_x = log_k.mean()
                mean_y = log_s.mean()
                num = ((log_k - mean_x) * (log_s - mean_y)).sum()
                den = ((log_k - mean_x)**2).sum()
                alpha = -num / (den + 1e-10)
                return alpha
            except: return torch.tensor(0.0, device=device)

        elif metric_name == 'distributional_flatness':
            # Flatness of weight distribution (Kurtosis)
            w = weights
            mu = w.mean()
            sigma = w.std()
            kurt = ((w - mu)**4).mean() / (sigma**4 + 1e-10)
            return kurt

        elif metric_name == 'hessian_log_determinant':
            # Proxy: Sum of log of diagonal of Hessian (approx)
            # Use Hutchinson for trace of log H? No.
            # Use diagonal approximation of Fisher as proxy for Hessian
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            fisher_diag = torch.cat([g.view(-1)**2 for g in grads])
            return torch.log(fisher_diag + 1e-10).sum()

        elif metric_name == 'hessian_eigenvalue_entropy':
            # Entropy of Hessian eigenvalues
            # Proxy: Entropy of Fisher eigenvalues (approx by diagonal)
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            fisher_diag = torch.cat([g.view(-1)**2 for g in grads])
            probs = fisher_diag / (fisher_diag.sum() + 1e-10)
            return -(probs * torch.log(probs + 1e-10)).sum()

        elif metric_name == 'gradient_subspace_dimension':
            # Dimensionality of gradient subspace
            # Proxy: Effective rank of gradients (requires multiple samples)
            # Use gradients of different layers as "samples"?
            # Or just return 0.0 if single sample batch.
            # Let's use the rank of the embedding gradients (w.r.t. input)
            # Proxy: Rank of embedding matrix
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                s = s / s.sum()
                return torch.exp(-(s * torch.log(s + 1e-10)).sum())
            except: return torch.tensor(0.0, device=device)

        elif metric_name == 'jacobian_mutual_coherence':
            # Mutual coherence of Jacobian columns
            # Proxy: Coherence of embedding matrix columns
            w = model.embedding.weight
            # Normalize columns
            w_norm = w / (w.norm(dim=0, keepdim=True) + 1e-10)
            gram = w_norm.t() @ w_norm
            # Max off-diagonal
            mask = 1.0 - torch.eye(gram.size(0), device=device)
            return (torch.abs(gram) * mask).max()

        elif metric_name == 'activation_covariance_condition':
            # Condition number of activation covariance
            # Use embeddings
            w = model.embedding.weight
            cov = w.t() @ w
            return torch.linalg.cond(cov + 1e-6*torch.eye(cov.size(0), device=device))

        elif metric_name == 'spectral_participation_ratio':
            # (sum lambda)^2 / sum lambda^2
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                lam = s**2 # Eigenvalues of Cov
                pr = (lam.sum())**2 / ((lam**2).sum() + 1e-10)
                return pr
            except: return torch.tensor(0.0, device=device)

        elif metric_name == 'energy_landscape_ruggedness':
            # Proxy: Gradient norm variance or similar
            # Or "Roughness" of loss surface
            # Use gradient norm
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_norm = torch.sqrt(sum([(g**2).sum() for g in grads]))
            return grad_norm

        elif metric_name == 'gradient_sign_entropy':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            all_grads = torch.cat([g.view(-1) for g in grads])
            
            # Soft sign approximation: tanh(k * g)
            # Values close to -1, 0, 1
            k = 100.0
            soft_signs = torch.tanh(k * all_grads)
            
            # Soft binning
            # Pos: close to 1. Neg: close to -1. Zero: close to 0.
            # We can use Gaussian kernels centered at -1, 0, 1
            
            # p_pos ~ exp(-(s-1)^2 / sigma)
            sigma = 0.1
            w_pos = torch.exp(-(soft_signs - 1.0)**2 / sigma)
            w_neg = torch.exp(-(soft_signs + 1.0)**2 / sigma)
            w_zero = torch.exp(-(soft_signs)**2 / sigma)
            
            total = w_pos.sum() + w_neg.sum() + w_zero.sum() + 1e-10
            p_pos = w_pos.sum() / total
            p_neg = w_neg.sum() / total
            p_zero = w_zero.sum() / total
            
            probs = torch.stack([p_pos, p_neg, p_zero])
            probs = probs + 1e-10 # Avoid log(0)
            probs = probs / probs.sum()
            
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'activation_skewness':
            # E[(a - mu)^3 / sigma^3]
            w = model.embedding.weight.view(-1)
            mu = w.mean()
            sigma = w.std()
            skew = ((w - mu)**3).mean() / (sigma**3 + 1e-10)
            return torch.abs(skew)

        elif metric_name == 'singular_vector_alignment_entropy':
            # Alignment between singular vectors of adjacent layers
            # Proxy: Cosine similarity between top singular vectors of linear layers
            val = 0.0
            count = 0
            layers = model.transformer.layers
            for i in range(len(layers)-1):
                w1 = layers[i].linear2.weight
                w2 = layers[i+1].linear1.weight
                try:
                    # Top singular vector
                    _, _, v1 = torch.linalg.svd(w1)
                    u2, _, _ = torch.linalg.svd(w2)
                    # Alignment: (v1[0] @ u2[:,0])**2
                    align = (v1[0] @ u2[:,0])**2
                    val += align
                    count += 1
                except: pass
            if count == 0: return torch.tensor(0.0, device=device)
            return val / count

        elif metric_name == 'log_volume_convex_hull':
            # Log volume of convex hull of activations
            # Proxy: LogDet of Covariance of embeddings
            w = model.embedding.weight
            cov = w.t() @ w / w.size(0)
            return torch.logdet(cov + 1e-6*torch.eye(cov.size(0), device=device))

        elif metric_name == 'activation_manifold_curvature':
            # Curvature of activation manifold
            # Proxy: 1 / (Spectral Gap)
            gap = Metrics.calculate_metric(model, 'spectral_gap_activation', logits, labels, input_ids)
            return 1.0 / (gap + 1e-10)

        elif metric_name == 'spectral_gap_activation':
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                if len(s) > 1:
                    return s[0] - s[1]
                return torch.tensor(0.0, device=device)
            except: return torch.tensor(0.0, device=device)

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

        elif metric_name == 'third_order_curvature_norm':
            # ||grad^3 L||_F
            # Proxy: Norm of gradient of Hessian Trace
            # Or gradient of gradient norm squared
            if labels is None: return torch.tensor(0.0, device=device)
            
            # Use a fresh forward pass with small batch/seq to save memory
            # The graph for 3rd order derivatives is HUGE.
            if input_ids is not None:
                # Slice input to minimal size
                # Batch size 1, Sequence length min(64, current)
                b_size = 1
                seq_len = min(input_ids.size(1), 64)
                
                input_ids_small = input_ids[:b_size, :seq_len]
                labels_small = labels[:b_size, :seq_len]
                
                # New forward pass
                logits_small = model(input_ids_small)
                
                loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_small.view(-1, logits_small.size(-1)), labels_small.view(-1))
            else:
                if logits is None: return torch.tensor(0.0, device=device)
                # Subsample batch to save memory for 3rd order derivatives
                batch_size = logits.shape[0]
                max_samples = 1 # Very small batch for 3rd order
                if batch_size > max_samples:
                    logits_subset = logits[:max_samples]
                    labels_subset = labels[:max_samples]
                else:
                    logits_subset = logits
                    labels_subset = labels

                loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_subset.view(-1, logits_subset.size(-1)), labels_subset.view(-1))
            
            # Restrict parameters to save memory (3rd order is very expensive)
            all_params = [p for p in model.parameters() if p.requires_grad]
            # Use only the last few parameters (e.g. last layer) to avoid OOM
            if len(all_params) > 5:
                params = all_params[-5:]
            else:
                params = all_params
                
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            # Grad of grad norm sq is 2 * H * g
            # We want 3rd order.
            # Let's take grad of (g^T H g) or similar.
            # Proxy: Grad of Hessian Trace
            # Trace H approx by Hutchinson
            v = [torch.randint_like(p, high=2) * 2 - 1 for p in params]
            grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
            Hv = torch.autograd.grad(grad_v, params, create_graph=True)
            vHv = sum([(h * vi).sum() for h, vi in zip(Hv, v)])
            # Grad of vHv
            grad_vHv = torch.autograd.grad(vHv, params, create_graph=True)
            norm_grad_vHv = torch.sqrt(sum([(g**2).sum() for g in grad_vHv]))
            return norm_grad_vHv

        elif metric_name == 'curvature_skewness':
            # E[(lambda - mean)^3]
            # Proxy: Skewness of Fisher diagonal
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            fisher_diag = torch.cat([g.view(-1)**2 for g in grads])
            mu = fisher_diag.mean()
            sigma = fisher_diag.std()
            skew = ((fisher_diag - mu)**3).mean() / (sigma**3 + 1e-10)
            return torch.abs(skew)

        elif metric_name == 'curvature_kurtosis':
            # E[(lambda - mean)^4]
            # Proxy: Kurtosis of Fisher diagonal
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            fisher_diag = torch.cat([g.view(-1)**2 for g in grads])
            mu = fisher_diag.mean()
            sigma = fisher_diag.std()
            kurt = ((fisher_diag - mu)**4).mean() / (sigma**4 + 1e-10)
            return kurt

        elif metric_name == 'loss_surface_convexity_ratio':
            # Fraction of lambda > 0
            # Proxy: Soft fraction of positive curvature directions
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            pos_score = 0.0
            n_probes = 5
            for _ in range(n_probes):
                v = [torch.randn_like(p) for p in params]
                grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
                Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
                vHv = sum([(h * vi).sum() for h, vi in zip(Hv, v)])
                # Soft indicator: sigmoid(vHv)
                pos_score += torch.sigmoid(vHv)
            return pos_score / n_probes

        elif metric_name == 'negative_curvature_mass':
            # Sum |lambda| for lambda < 0
            # Proxy: Sum |v^T H v| for v^T H v < 0
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            neg_mass = torch.tensor(0.0, device=device)
            n_probes = 5
            for _ in range(n_probes):
                v = [torch.randn_like(p) for p in params]
                grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
                Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
                vHv = sum([(h * vi).sum() for h, vi in zip(Hv, v)])
                if vHv < 0: neg_mass += torch.abs(vHv)
            return neg_mass / n_probes

        elif metric_name == 'random_direction_curvature_variance':
            # Var(v^T H v)
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            curvatures = []
            n_probes = 5
            for _ in range(n_probes):
                v = [torch.randn_like(p) for p in params]
                # Normalize v
                v_norm = torch.sqrt(sum([(vi**2).sum() for vi in v]))
                v = [vi / (v_norm + 1e-10) for vi in v]
                
                grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
                Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
                vHv = sum([(h * vi).sum() for h, vi in zip(Hv, v)])
                curvatures.append(vHv)
            
            if not curvatures: return torch.tensor(0.0, device=device)
            return torch.var(torch.stack(curvatures))

        elif metric_name == 'loss_basin_connectivity_index':
            # Prob random perturbation remains low loss
            # Proxy: 1 / (1 + Sharpness)
            sharpness = Metrics.calculate_metric(model, 'sharpness_perturbation', logits, labels, input_ids)
            return 1.0 / (1.0 + sharpness)

        elif metric_name == 'curvature_gradient_alignment':
            # cos(g, Hg)
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            
            # Subsample batch to save memory
            batch_size = logits.shape[0]
            max_samples = 2
            if batch_size > max_samples:
                logits_subset = logits[:max_samples]
                labels_subset = labels[:max_samples]
            else:
                logits_subset = logits
                labels_subset = labels

            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_subset.view(-1, logits_subset.size(-1)), labels_subset.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            # Hg
            grad_norm_sq = sum([(g**2).sum() for g in grads]) # g^T g
            # grad(g^T g) = 2 Hg
            Hg_2 = torch.autograd.grad(grad_norm_sq, params, create_graph=True)
            Hg = [0.5 * h for h in Hg_2]
            
            # Cosine similarity
            dot = sum([(g * h).sum() for g, h in zip(grads, Hg)])
            norm_g = torch.sqrt(grad_norm_sq)
            norm_Hg = torch.sqrt(sum([(h**2).sum() for h in Hg]))
            
            return dot / (norm_g * norm_Hg + 1e-10)

        elif metric_name == 'local_loss_lipschitz_constant':
            # Max gradient norm in neighborhood
            # Proxy: Gradient norm + Hessian norm * eps
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_norm = torch.sqrt(sum([(g**2).sum() for g in grads]))
            # Add Hessian spectral radius * eps
            hess = Metrics.calculate_metric(model, 'hessian_spectral_radius', logits, labels, input_ids)
            return grad_norm + hess * 0.01

        elif metric_name == 'activation_description_length':
            # MDL of activations
            # Proxy: Sum of log(|a| + eps) + entropy
            # Use embeddings
            w = model.embedding.weight
            # Shannon entropy of quantized activations?
            # Proxy: Shannon entropy
            probs, _ = Metrics.soft_histogram(w.view(-1))
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'kolmogorov_structure_index':
            # Entropy of differences between adjacent weights
            w = Metrics.get_all_weights(model)
            diff = w[1:] - w[:-1]
            probs, _ = Metrics.soft_histogram(diff)
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'empirical_covering_number':
            # Covering number
            # Proxy: exp(Entropy)
            ent = Metrics.calculate_metric(model, 'shannon', logits, labels, input_ids)
            return torch.exp(ent)

        elif metric_name == 'cross_layer_rank_coupling':
            # Rank correlation of singular spectra
            # Proxy: Correlation of singular values of adjacent layers
            val = 0.0
            count = 0
            layers = model.transformer.layers
            for i in range(len(layers)-1):
                w1 = layers[i].linear1.weight
                w2 = layers[i+1].linear1.weight
                try:
                    s1 = torch.linalg.svdvals(w1)
                    s2 = torch.linalg.svdvals(w2)
                    # Resize
                    n = min(len(s1), len(s2))
                    s1 = s1[:n]
                    s2 = s2[:n]
                    # Correlation
                    corr = torch.corrcoef(torch.stack([s1, s2]))[0,1]
                    val += corr
                    count += 1
                except: pass
            if count == 0: return torch.tensor(0.0, device=device)
            return val / count

        elif metric_name == 'ntk_entropy':
            # Entropy of NTK eigenvalues
            # Proxy: Entropy of embedding covariance eigenvalues
            w = model.embedding.weight
            cov = w.t() @ w
            try:
                eigs = torch.linalg.eigvalsh(cov)
                eigs = eigs[eigs > 0]
                probs = eigs / eigs.sum()
                return -(probs * torch.log(probs + 1e-10)).sum()
            except: return torch.tensor(0.0, device=device)

        elif metric_name == 'capacity_utilization_ratio':
            # Effective rank / parameter count
            # Proxy: Spectral participation ratio / Hidden dim
            spr = Metrics.calculate_metric(model, 'spectral_participation_ratio', logits, labels, input_ids)
            return spr / Config.HIDDEN_DIM

        elif metric_name == 'vc_margin_ratio':
            # Margin / VC
            # Proxy: 1 / VC_proxy
            vc = Metrics.calculate_metric(model, 'vc_dimension_proxy', logits, labels, input_ids)
            return 1.0 / (vc + 1e-10)

        elif metric_name == 'higher_order_cumulant_energy':
            # Sum of squared cumulants (3, 4)
            # Proxy: Skew^2 + Kurt^2
            skew = Metrics.calculate_metric(model, 'activation_skewness', logits, labels, input_ids)
            kurt = Metrics.calculate_metric(model, 'distributional_flatness', logits, labels, input_ids)
            return skew**2 + kurt**2

        elif metric_name == 'logit_gap_entropy':
            # Entropy of differences between top-k logits
            if logits is None: return torch.tensor(0.0, device=device)
            topk = logits.topk(min(5, logits.size(-1)), dim=-1).values
            gaps = topk[:, :-1] - topk[:, 1:]
            probs, _ = Metrics.soft_histogram(gaps.view(-1))
            return -(probs * torch.log(probs)).sum()

        elif metric_name == 'output_bimodality_coefficient':
            # Skew^2 + 1 / Kurt
            if logits is None: return torch.tensor(0.0, device=device)
            l = logits.view(-1)
            mu = l.mean()
            sigma = l.std()
            skew = ((l - mu)**3).mean() / (sigma**3 + 1e-10)
            kurt = ((l - mu)**4).mean() / (sigma**4 + 1e-10)
            return (skew**2 + 1) / (kurt + 1e-10)

        elif metric_name == 'population_sparsity_gini':
            # Gini coefficient of neuron activations
            # Use embeddings
            w = model.embedding.weight
            # Mean activation per neuron (column)
            act = w.abs().mean(dim=0)
            # Gini
            act_sorted = torch.sort(act).values
            n = len(act)
            index = torch.arange(1, n+1, device=device)
            return (2 * (index * act_sorted).sum() / (n * act_sorted.sum()) - (n + 1) / n)

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

        elif metric_name == 'lyapunov_spectrum_width':
            # Range of exponents
            # Proxy: Max - Min singular value (log)
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                return torch.log(s[0] + 1e-10) - torch.log(s[-1] + 1e-10)
            except: return torch.tensor(0.0, device=device)

        elif metric_name == 'update_direction_autocorrelation':
            # Memory
            # Proxy: 1 - Gradient cosine drift
            drift = Metrics.calculate_metric(model, 'gradient_cosine_drift', logits, labels, input_ids)
            return 1.0 - drift

        elif metric_name == 'stochastic_stability_radius':
            # Max noise
            # Proxy: 1 / Hessian spectral radius
            hsr = Metrics.calculate_metric(model, 'hessian_spectral_radius', logits, labels, input_ids)
            return 1.0 / (hsr + 1e-10)

        elif metric_name == 'functional_inertia':
            # Resistance to change
            # Proxy: 1 / Gradient norm
            gn = Metrics.calculate_metric(model, 'trajectory_length', logits, labels, input_ids)
            return 1.0 / (gn + 1e-10)

        elif metric_name == 'implicit_bias_alignment':
            # Correlation with min-norm
            # Proxy: Norm of weights (L2)
            w = Metrics.get_all_weights(model)
            return torch.norm(w)

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
            metric_tensor = Metrics.calculate_metric(model, config.METRIC_NAME, logits, labels, input_ids)
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
    
    # List of experiments: (Control_Mode, Metric_Name, Folder_Name)
    experiments = [
        (True, 'shannon', 'control'),
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
        (False, 'tsallis_entropy', '21_tsallis_entropy'),
        (False, 'von_neumann_entropy', '22_von_neumann_entropy'),
        (False, 'path_entropy', '23_path_entropy'),
        (False, 'layer_activation_entropy', '24_layer_activation_entropy'),
        (False, 'weight_distribution_entropy', '25_weight_distribution_entropy'),
        (False, 'gradient_entropy', '26_gradient_entropy'),
        (False, 'fisher_info_entropy', '27_fisher_info_entropy'),
        (False, 'prediction_entropy', '28_prediction_entropy'),
        (False, 'bayesian_entropy', '29_bayesian_entropy'),
        (False, 'attention_entropy', '30_attention_entropy'),
        (False, 'token_entropy', '31_token_entropy'),
        (False, 'entropy_rate', '32_entropy_rate'),
        (False, 'multi_scale_entropy', '33_multi_scale_entropy'),
        (False, 'conditional_output_entropy', '34_conditional_output_entropy'),
        (False, 'effective_dimensionality_entropy', '35_effective_dimensionality_entropy'),
        (False, 'latent_entropy', '36_latent_entropy'),
        (False, 'structural_entropy', '37_structural_entropy'),
        (False, 'topological_entropy', '38_topological_entropy'),
        (False, 'differential_entropy', '39_differential_entropy'),
        (False, 'log_determinant_entropy', '40_log_determinant_entropy'),
        (False, 'gaussian_entropy', '41_gaussian_entropy'),
        (False, 'kde_entropy', '42_kde_entropy'),
        (False, 'copula_entropy', '43_copula_entropy'),
        (False, 'cumulative_residual_entropy', '44_cumulative_residual_entropy'),
        (False, 'quadratic_entropy', '45_quadratic_entropy'),
        (False, 'energy_entropy', '46_energy_entropy'),
        (False, 'logit_distribution_entropy', '47_logit_distribution_entropy'),
        (False, 'softmax_temperature_entropy', '48_softmax_temperature_entropy'),
        (False, 'renyi_divergence', '49_renyi_divergence'),
        (False, 'wasserstein_entropy', '50_wasserstein_entropy'),
        (False, 'flow_entropy', '51_flow_entropy'),
        (False, 'spike_entropy', '52_spike_entropy'),
        (False, 'class_conditional_entropy', '53_class_conditional_entropy'),
        (False, 'population_coding_entropy', '54_population_coding_entropy'),
        (False, 'hessian_trace', '55_hessian_trace'),
        (False, 'hessian_spectral_radius', '56_hessian_spectral_radius'),
        (False, 'fisher_info_condition_number', '57_fisher_info_condition_number'),
        (False, 'sharpness_perturbation', '58_sharpness_perturbation'),
        (False, 'pac_bayes_flatness', '59_pac_bayes_flatness'),
        (False, 'gradient_covariance_trace', '60_gradient_covariance_trace'),
        (False, 'gradient_direction_entropy', '61_gradient_direction_entropy'),
        (False, 'gradient_cosine_drift', '62_gradient_cosine_drift'),
        (False, 'activation_jacobian_frobenius_norm', '63_activation_jacobian_frobenius_norm'),
        (False, 'layerwise_lipschitz', '64_layerwise_lipschitz'),
        (False, 'effective_rank_activations', '65_effective_rank_activations'),
        (False, 'log_det_activation_covariance', '66_log_det_activation_covariance'),
        (False, 'class_conditional_overlap', '67_class_conditional_overlap'),
        (False, 'information_compression_ratio', '68_information_compression_ratio'),
        (False, 'trajectory_length', '69_trajectory_length'),
        (False, 'stochastic_loss_variance', '70_stochastic_loss_variance'),
        (False, 'local_linearization_error', '71_local_linearization_error'),
        (False, 'output_jacobian_condition_number', '72_output_jacobian_condition_number'),
        (False, 'mdl_surrogate', '73_mdl_surrogate'),
        (False, 'kolmogorov_complexity_proxy', '74_kolmogorov_complexity_proxy'),
        (False, 'fractal_dimension', '75_fractal_dimension'),
        (False, 'ntk_condition_number', '76_ntk_condition_number'),
        (False, 'ntk_trace', '77_ntk_trace'),
        (False, 'rademacher_complexity', '78_rademacher_complexity'),
        (False, 'pac_bayes_kl', '79_pac_bayes_kl'),
        (False, 'activation_total_correlation', '80_activation_total_correlation'),
        (False, 'class_conditional_activation_kl', '81_class_conditional_activation_kl'),
        (False, 'energy_distance_class', '82_energy_distance_class'),
        (False, 'wasserstein_barycenter_dispersion', '83_wasserstein_barycenter_dispersion'),
        (False, 'entropy_rate_activations', '84_entropy_rate_activations'),
        (False, 'lyapunov_exponent', '85_lyapunov_exponent'),
        (False, 'topological_persistence_entropy', '86_topological_persistence_entropy'),
        (False, 'info_geometric_volume', '87_info_geometric_volume'),
        (False, 'vc_dimension_proxy', '88_vc_dimension_proxy'),
        (False, 'spectral_decay_rate', '89_spectral_decay_rate'),
        (False, 'distributional_flatness', '90_distributional_flatness'),
        (False, 'hessian_log_determinant', '91_hessian_log_determinant'),
        (False, 'hessian_eigenvalue_entropy', '92_hessian_eigenvalue_entropy'),
        (False, 'gradient_subspace_dimension', '93_gradient_subspace_dimension'),
        (False, 'jacobian_mutual_coherence', '94_jacobian_mutual_coherence'),
        (False, 'activation_covariance_condition', '95_activation_covariance_condition'),
        (False, 'spectral_participation_ratio', '96_spectral_participation_ratio'),
        (False, 'energy_landscape_ruggedness', '97_energy_landscape_ruggedness'),
        (False, 'gradient_sign_entropy', '98_gradient_sign_entropy'),
        (False, 'activation_skewness', '99_activation_skewness'),
        (False, 'singular_vector_alignment_entropy', '100_singular_vector_alignment_entropy'),
        (False, 'log_volume_convex_hull', '101_log_volume_convex_hull'),
        (False, 'activation_manifold_curvature', '102_activation_manifold_curvature'),
        (False, 'spectral_gap_activation', '103_spectral_gap_activation'),
        (False, 'lipschitz_variance', '104_lipschitz_variance'),
        (False, 'third_order_curvature_norm', '105_third_order_curvature_norm'),
        (False, 'curvature_skewness', '106_curvature_skewness'),
        (False, 'curvature_kurtosis', '107_curvature_kurtosis'),
        (False, 'loss_surface_convexity_ratio', '108_loss_surface_convexity_ratio'),
        (False, 'negative_curvature_mass', '109_negative_curvature_mass'),
        (False, 'random_direction_curvature_variance', '110_random_direction_curvature_variance'),
        (False, 'loss_basin_connectivity_index', '111_loss_basin_connectivity_index'),
        (False, 'curvature_gradient_alignment', '112_curvature_gradient_alignment'),
        (False, 'local_loss_lipschitz_constant', '114_local_loss_lipschitz_constant'),
        (False, 'activation_description_length', '115_activation_description_length'),
        (False, 'kolmogorov_structure_index', '116_kolmogorov_structure_index'),
        (False, 'empirical_covering_number', '117_empirical_covering_number'),
        (False, 'cross_layer_rank_coupling', '118_cross_layer_rank_coupling'),
        (False, 'ntk_entropy', '119_ntk_entropy'),
        (False, 'capacity_utilization_ratio', '120_capacity_utilization_ratio'),
        (False, 'vc_margin_ratio', '121_vc_margin_ratio'),
        (False, 'higher_order_cumulant_energy', '122_higher_order_cumulant_energy'),
        (False, 'logit_gap_entropy', '123_logit_gap_entropy'),
        (False, 'output_bimodality_coefficient', '124_output_bimodality_coefficient'),
        (False, 'population_sparsity_gini', '125_population_sparsity_gini'),
        (False, 'activation_kurtosis_variance', '126_activation_kurtosis_variance'),
        (False, 'lyapunov_spectrum_width', '127_lyapunov_spectrum_width'),
        (False, 'update_direction_autocorrelation', '128_update_direction_autocorrelation'),
        (False, 'stochastic_stability_radius', '130_stochastic_stability_radius'),
        (False, 'functional_inertia', '131_functional_inertia'),
        (False, 'implicit_bias_alignment', '132_implicit_bias_alignment'),
    ]
    
    for control_mode, metric_name, folder_name in experiments:
        output_dir = os.path.join(script_dir, f'output_min/{folder_name}')
        
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
