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
from torch.utils.checkpoint import checkpoint

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
    EPOCHS = 50
    SEQ_LENGTH = 32
    MAX_GRAD_NORM = 1.0
    MAX_SAMPLES = 2000
    
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
    def _varentropy_func(logits_chunk):
        probs = torch.softmax(logits_chunk, dim=-1)
        log_p = torch.log(probs + 1e-10)
        entropy = -(probs * log_p).sum(dim=-1, keepdim=True)
        # V(p) = sum p (log p)^2 - H^2
        term1 = (probs * log_p**2).sum(dim=-1)
        v_chunk = term1 - entropy.squeeze(-1)**2
        return v_chunk

    @staticmethod
    def calculate_metric(model, metric_name, logits=None, labels=None, input_ids=None, features=None, hidden_states=None):
        weights = Metrics.get_all_weights(model)
        device = weights.device
        
        if metric_name == 'shannon':
            probs, _ = Metrics.soft_histogram(weights)
            return -(probs * torch.log(probs)).sum()

        # 0. Disequilibrium
        elif metric_name == 'disequilibrium':
            probs, _ = Metrics.soft_histogram(weights)
            n_bins = probs.size(0)
            uniform_prob = 1.0 / n_bins
            return ((probs - uniform_prob) ** 2).sum()

        # 1. Symmetry-Breaking Order Parameters
        elif metric_name == 'symmetry_breaking':
            # max(p) - 1/N
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            max_p = probs.max(dim=-1).values.mean()
            return max_p

        # 2. Predictability / Excess Entropy
        elif metric_name == 'excess_entropy':
            # 1 - H(p)/log(N) (Negentropy)
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            max_entropy = math.log(probs.size(-1))
            return 1.0 - (entropy / (max_entropy + 1e-10))

        # 3. Energy-Based Distance
        elif metric_name == 'energy_distance':
            # KL(p || uniform)
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n_classes = probs.size(-1)
            # KL = -H(p) + log(N)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            return math.log(n_classes) - entropy

        # 4. Flatness Avoidance / Curvature
        elif metric_name == 'flatness_avoidance':
            # Sum p^2 (Renyi 2)
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            return (probs ** 2).sum(dim=-1).mean()

        # 5. Multimodality
        elif metric_name == 'multimodality':
            # Variance of probabilities
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            return torch.var(probs, dim=-1).mean()

        # 6. Broken Ergodicity
        elif metric_name == 'broken_ergodicity':
            # KL(mean(p) || uniform)
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            mean_probs = probs.mean(dim=0) # Average over batch
            n_classes = probs.size(-1)
            entropy_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum()
            return math.log(n_classes) - entropy_mean

        # 7. Participation Ratio
        elif metric_name == 'participation_ratio':
            # N - 1/sum(p^2)
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n_classes = probs.size(-1)
            inverse_pr = (probs ** 2).sum(dim=-1) # 1/N_eff
            n_eff = 1.0 / (inverse_pr + 1e-10)
            return float(n_classes) - n_eff.mean()

        # 8. Representation Commitment
        elif metric_name == 'decision_sharpness':
            # Margin between top 1 and top 2
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            return margin.mean()

        # 9. Fisher-Rao Distance (Geometry)
        elif metric_name == 'fisher_rao_distance':
            # d(p, u) = 2 * arccos( sum( sqrt(p * u) ) )
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n_classes = probs.size(-1)
            uniform = torch.full_like(probs, 1.0 / n_classes)
            # sum(sqrt(p*u)) = sum(sqrt(p) * sqrt(1/N)) = (1/sqrt(N)) * sum(sqrt(p))
            inner_prod = (torch.sqrt(probs)).sum(dim=-1) / math.sqrt(n_classes)
            # Clamp for numerical stability of arccos
            inner_prod = torch.clamp(inner_prod, -1.0 + 1e-7, 1.0 - 1e-7)
            dist = 2.0 * torch.acos(inner_prod)
            return dist.mean()

        # 10. Wasserstein Distance to Uniform (1D Approx)
        elif metric_name == 'wasserstein_uniform':
            # EMD between sorted p and sorted uniform
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n_classes = probs.size(-1)
            # Sort probabilities
            p_sorted, _ = torch.sort(probs, dim=-1)
            # Uniform CDF is linear, but PDF is constant.
            # For 1D Wasserstein between distributions on the same support (indices),
            # we usually compare CDFs.
            # Here we treat the probability vector itself as a distribution over indices?
            # Or the values?
            # "Wasserstein distance between clustered components" was the prompt.
            # Let's use the "Earth Mover's Distance" to the uniform distribution
            # assuming the ground metric is distance between indices (0..N-1).
            # CDF of p
            cdf_p = torch.cumsum(probs, dim=-1)
            cdf_u = torch.linspace(1.0/n_classes, 1.0, n_classes, device=device)
            # W1 = sum |CDF_p - CDF_u|
            w1 = torch.abs(cdf_p - cdf_u).sum(dim=-1)
            return w1.mean()

        # 11. Gini Index (Inequality)
        elif metric_name == 'gini_index':
            # G = sum (2i - n - 1) x_i / (n sum x_i)
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n = probs.size(-1)
            # Sort probs
            probs_sorted, _ = torch.sort(probs, dim=-1)
            index = torch.arange(1, n + 1, device=device).float()
            gini = (2 * index - n - 1) * probs_sorted
            return gini.sum(dim=-1).mean() / n

        # 12. Semantic Diversity (Rao's Q with Embeddings)
        elif metric_name == 'semantic_diversity':
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            # Top-k for efficiency
            k = 5
            topk_probs, topk_indices = probs.topk(k, dim=-1)
            # Normalize top-k probs
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            
            # Get embeddings
            E = model.embedding.weight
            vecs = E[topk_indices] # [B, S, k, D]
            
            # Pairwise distances [B, S, k, k]
            # Flatten B, S -> N
            vecs = vecs.view(-1, k, vecs.size(-1))
            topk_probs = topk_probs.view(-1, k)
            
            dists = torch.cdist(vecs, vecs)
            
            # Weighted sum: sum_ij p_i p_j d_ij
            weights = topk_probs.unsqueeze(2) * topk_probs.unsqueeze(1)
            diversity = (weights * dists).sum(dim=(1, 2))
            return diversity.mean()

        # 13. Inverse Spectral Flatness
        elif metric_name == 'inverse_spectral_flatness':
            # ISF = Arithmetic Mean / Geometric Mean
            # AM = 1/N (since sum p = 1)
            # GM = exp( mean(log p) )
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n = probs.size(-1)
            am = 1.0 / n
            log_gm = torch.log(probs + 1e-10).mean(dim=-1)
            gm = torch.exp(log_gm)
            return (am / (gm + 1e-10)).mean()

        # 14. Hoyer Sparsity
        elif metric_name == 'hoyer_sparsity':
            if features is None: return torch.tensor(0.0, device=device)
            # features: [B, S, D]
            x = features.view(-1, features.size(-1))
            n = x.size(1)
            sqrt_n = math.sqrt(n)
            
            l1 = x.norm(p=1, dim=1)
            l2 = x.norm(p=2, dim=1)
            
            # (sqrt(n) - L1/L2) / (sqrt(n) - 1)
            sparsity = (sqrt_n - l1 / (l2 + 1e-10)) / (sqrt_n - 1 + 1e-10)
            return sparsity.mean()

        # 15. Varentropy (Variance of Information)
        elif metric_name == 'varentropy':
            if logits is None: return torch.tensor(0.0, device=device)
            
            # Memory efficient implementation with checkpointing
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            
            # Chunk size for processing
            chunk_size = 1024 
            chunks = torch.split(logits_flat, chunk_size, dim=0)
            
            vals = []
            for chunk in chunks:
                # Use checkpoint to save memory during backward
                if chunk.requires_grad:
                    v_chunk = checkpoint(Metrics._varentropy_func, chunk, use_reentrant=False)
                else:
                    v_chunk = Metrics._varentropy_func(chunk)
                vals.append(v_chunk)
            
            return torch.cat(vals).mean()

        # 16. Jensen-Shannon Divergence to Uniform
        elif metric_name == 'jensen_shannon':
            # JSD(p||u) = 0.5 KL(p||m) + 0.5 KL(u||m), m = 0.5(p+u)
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n = probs.size(-1)
            u = torch.full_like(probs, 1.0/n)
            m = 0.5 * (probs + u)
            kl_pm = (probs * (torch.log(probs + 1e-10) - torch.log(m + 1e-10))).sum(dim=-1)
            kl_um = (u * (torch.log(u + 1e-10) - torch.log(m + 1e-10))).sum(dim=-1)
            return 0.5 * (kl_pm + kl_um).mean()

        # 17. Hellinger Distance to Uniform
        elif metric_name == 'hellinger':
            # H(p,u) = 1/sqrt(2) * sqrt( sum (sqrt(p) - sqrt(u))^2 )
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n = probs.size(-1)
            sqrt_p = torch.sqrt(probs)
            sqrt_u = math.sqrt(1.0/n)
            dist = torch.sqrt( ((sqrt_p - sqrt_u)**2).sum(dim=-1) ) / math.sqrt(2.0)
            return dist.mean()

        # 18. Total Variation Distance to Uniform
        elif metric_name == 'total_variation':
            # TV(p,u) = 0.5 * sum |p - u|
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n = probs.size(-1)
            dist = 0.5 * torch.abs(probs - 1.0/n).sum(dim=-1)
            return dist.mean()

        # 19. Weight Anisotropy (1 - Spectral Entropy)
        elif metric_name == 'weight_anisotropy':
            # High anisotropy = Low spectral entropy
            # We want to MAXIMIZE anisotropy.
            # SVD of embedding or layers
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                s_norm = s / (s.sum() + 1e-10)
                entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
                max_ent = math.log(len(s))
                return 1.0 - (entropy / max_ent)
            except: return torch.tensor(0.0, device=device)

        # 20. Weight Gini Index (Sparsity)
        elif metric_name == 'weight_gini':
            w = Metrics.get_all_weights(model).abs()
            # Subsample if too large
            if len(w) > 10000:
                idx = torch.randperm(len(w), device=device)[:10000]
                w = w[idx]
            w_sorted, _ = torch.sort(w)
            n = len(w)
            index = torch.arange(1, n + 1, device=device).float()
            gini = (2 * index - n - 1) * w_sorted
            return gini.sum() / (n * w_sorted.sum() + 1e-10)

        # 21. Logit Effective Rank (Batch Diversity)
        elif metric_name == 'logit_rank':
            # exp(Entropy of singular values of logits)
            if logits is None: return torch.tensor(0.0, device=device)
            # logits: [B, V]
            # If B < V, rank is limited by B.
            try:
                s = torch.linalg.svdvals(torch.softmax(logits, dim=-1))
                s_norm = s / (s.sum() + 1e-10)
                entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
                return torch.exp(entropy)
            except: return torch.tensor(0.0, device=device)

        # 22. Logit Variance (Simple Diversity)
        elif metric_name == 'logit_variance':
            if logits is None: return torch.tensor(0.0, device=device)
            return torch.var(logits)

        # 23. Margin Entropy
        elif metric_name == 'margin_entropy':
            # Entropy of the margin distribution
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
            hist_probs, _ = Metrics.soft_histogram(margins)
            return -(hist_probs * torch.log(hist_probs + 1e-10)).sum()

        # 24. Cluster Distance (Phase Separation)
        elif metric_name == 'cluster_distance':
            # Ratio of mean distance between class means to mean distance within classes
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            mask = labels != -100
            z = logits.view(-1, logits.size(-1))[mask.view(-1)]
            y = labels.view(-1)[mask.view(-1)]
            classes = torch.unique(y)
            if len(classes) < 2: return torch.tensor(0.0, device=device)
            
            means = []
            within_dist = 0.0
            count = 0
            for c in classes:
                zc = z[y==c]
                if len(zc) > 1: 
                    center = zc.mean(dim=0)
                    means.append(center)
                    # Mean dist to center
                    d = torch.norm(zc - center, dim=1).mean()
                    within_dist += d
                    count += 1
            
            if len(means) < 2: return torch.tensor(0.0, device=device)
            means = torch.stack(means)
            # Mean pairwise dist between means
            between_dist = torch.cdist(means, means).mean()
            
            return between_dist / (within_dist / count + 1e-10)

        # 25. Feature Rank (Embedding Dimension)
        elif metric_name == 'feature_rank':
            # Effective rank of embeddings
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                s_norm = s / (s.sum() + 1e-10)
                entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
                return torch.exp(entropy)
            except: return torch.tensor(0.0, device=device)

        # 26. Fractal Dimension (Correlation Dimension Proxy)
        elif metric_name == 'fractal_dim':
            # Slope of log C(r) vs log r
            # Use embeddings
            w = model.embedding.weight
            n = min(len(w), 500)
            idx = torch.randperm(len(w), device=device)[:n]
            x = w[idx]
            dist = torch.cdist(x, x) + 1e-10
            # Correlation integral
            r_vals = torch.logspace(-1, 1, 8, device=device)
            c_vals = []
            temperature = 50.0 # Soft threshold for differentiability
            for r in r_vals:
                # Soft count: sigmoid((r - dist) * temp)
                # If dist < r, r - dist > 0, sigmoid > 0.5 -> 1
                # If dist > r, r - dist < 0, sigmoid < 0.5 -> 0
                count = torch.sigmoid((r - dist) * temperature).mean()
                c_vals.append(count)
            c_vals = torch.stack(c_vals)
            # Linear fit
            log_r = torch.log(r_vals)
            log_c = torch.log(c_vals + 1e-10)
            # Slope
            mean_x = log_r.mean()
            mean_y = log_c.mean()
            num = ((log_r - mean_x) * (log_c - mean_y)).sum()
            den = ((log_r - mean_x)**2).sum()
            return num / (den + 1e-10)

        # 27. PAC-Bayes Bound Proxy
        elif metric_name == 'pac_bayes':
            # ||W||^2 / Margin^2
            # We want to MAXIMIZE disequilibrium, so maybe minimize this?
            # Or maximize Margin / ||W||.
            # Let's return Margin / ||W||
            w_norm = Metrics.get_all_weights(model).norm()
            sharpness = Metrics.calculate_metric(model, 'decision_sharpness', logits, labels, input_ids)
            return sharpness / (w_norm + 1e-10)

        # 28. Path Norm
        elif metric_name == 'path_norm':
            # Product of spectral norms
            prod = 1.0
            for layer in model.transformer.layers:
                # Linear1
                try:
                    s = torch.linalg.svdvals(layer.linear1.weight)[0]
                    prod *= s
                except: pass
            return torch.tensor(prod, device=device)

        # 29. Hessian Trace (Hutchinson Estimator)
        elif metric_name == 'hessian_trace':
            # Tr(H) approx E[v^T H v]
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            
            # Subsample batch to reduce memory usage (Hessian is expensive)
            # Use a small subset of the batch for the trace estimation
            batch_size = logits.size(0)
            max_samples = 2
            if batch_size > max_samples:
                indices = torch.randperm(batch_size, device=device)[:max_samples]
                logits_sub = logits[indices]
                labels_sub = labels[indices]
            else:
                logits_sub = logits
                labels_sub = labels

            # Need create_graph=True for double backprop
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_sub.view(-1, logits_sub.size(-1)), labels_sub.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            # Random vector v (Rademacher)
            v = [torch.randint_like(p, high=2) * 2 - 1 for p in params]
            
            # Hv product = grad(grad^T v)
            grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
            Hv = torch.autograd.grad(grad_v, params, create_graph=True)
            
            # v^T H v
            trace_est = sum([(h * vi).sum() for h, vi in zip(Hv, v)])
            return trace_est

        # 30. Hessian Spectral Radius (Power Iteration)
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

        # 31. Fisher Information Condition Number (Diagonal Proxy)
        elif metric_name == 'fisher_info_condition_number':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            # Diagonal Fisher ~ g^2
            fisher_diag = torch.cat([g.view(-1)**2 for g in grads])
            f_max = fisher_diag.max()
            f_min = fisher_diag.min() + 1e-10
            return f_max / f_min

        # 32. Sharpness (Perturbation)
        elif metric_name == 'sharpness_perturbation':
            # L(w + eps*g) - L(w)
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
            
            grad_norm = torch.sqrt(sum([(g**2).sum() for g in grads]))
            epsilon = 0.01
            
            # Taylor expansion approx: eps * ||g|| + 0.5 * eps^2 * (g^T H g / ||g||^2)
            # First order term
            term1 = epsilon * grad_norm
            
            # Second order term (g^T H g)
            grad_v = sum([(g * gi).sum() for g, gi in zip(grads, grads)]) # g^T g
            # grad(g^T g) = 2 H g
            Hg_2 = torch.autograd.grad(grad_v, params, create_graph=True)
            Hg = [0.5 * h for h in Hg_2]
            gHg = sum([(hg * g).sum() for hg, g in zip(Hg, grads)])
            
            term2 = 0.5 * (epsilon**2) * gHg / (grad_norm**2 + 1e-10)
            return term1 + term2

        # 33. Curvature Energy (Hessian Frobenius Norm)
        elif metric_name == 'curvature_energy':
            # Tr(H^2) approx E[||Hv||^2]
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            
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
            v = [torch.randint_like(p, high=2) * 2 - 1 for p in params]
            
            # Hv
            grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])
            Hv = torch.autograd.grad(grad_v, params, create_graph=True)
            
            # ||Hv||^2
            norm_sq = sum([(h**2).sum() for h in Hv])
            return norm_sq

        # 34. Gradient Covariance Trace
        elif metric_name == 'gradient_covariance_trace':
            # ||g||^2 as proxy for trace of covariance if mean g is small
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_norm_sq = sum([(g**2).sum() for g in grads])
            return grad_norm_sq

        # 35. Gradient Direction Entropy
        elif metric_name == 'gradient_direction_entropy':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            all_grads = torch.cat([g.view(-1) for g in grads])
            # Soft histogram of gradient values
            probs, _ = Metrics.soft_histogram(torch.abs(all_grads))
            return -(probs * torch.log(probs + 1e-10)).sum()

        # 36. Gradient Cosine Drift
        elif metric_name == 'gradient_cosine_drift':
            # 1 - Cos(g1, g2) from split batch
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            batch_size = logits.size(0)
            if batch_size < 2: return torch.tensor(0.0, device=device)
            half = batch_size // 2
            
            l1, y1 = logits[:half], labels[:half]
            l2, y2 = logits[half:], labels[half:]
            
            loss1 = nn.CrossEntropyLoss(ignore_index=-100)(l1.view(-1, l1.size(-1)), y1.view(-1))
            loss2 = nn.CrossEntropyLoss(ignore_index=-100)(l2.view(-1, l2.size(-1)), y2.view(-1))
            
            params = [p for p in model.parameters() if p.requires_grad]
            grads1 = torch.autograd.grad(loss1, params, create_graph=True)
            grads2 = torch.autograd.grad(loss2, params, create_graph=True)
            
            g1 = torch.cat([g.view(-1) for g in grads1])
            g2 = torch.cat([g.view(-1) for g in grads2])
            
            cos = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).mean()
            return 1.0 - cos

        # 37. Activation Jacobian Frobenius Norm
        elif metric_name == 'activation_jacobian_frobenius_norm':
            # Proxy: Product of weight norms
            val = 1.0
            for layer in model.transformer.layers:
                val *= layer.linear1.weight.norm()
            return val

        # 38. Layerwise Lipschitz
        elif metric_name == 'layerwise_lipschitz':
            # Product of spectral norms
            prod = 1.0
            for layer in model.transformer.layers:
                try:
                    s = torch.linalg.svdvals(layer.linear1.weight)[0]
                    prod *= s
                except: pass
            return torch.tensor(prod, device=device)

        # 39. Effective Rank of Activations
        elif metric_name == 'effective_rank_activations':
            # Entropy of singular values of embeddings
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                s_norm = s / (s.sum() + 1e-10)
                return torch.exp(-(s_norm * torch.log(s_norm + 1e-10)).sum())
            except: return torch.tensor(0.0, device=device)

        # 40. Log Det Activation Covariance
        elif metric_name == 'log_det_activation_covariance':
            w = model.embedding.weight
            cov = w.t() @ w / w.size(0)
            return torch.logdet(cov + 1e-6 * torch.eye(cov.size(0), device=device))

        # 41. Class Conditional Overlap
        elif metric_name == 'class_conditional_overlap':
            # Mean dot product of class means
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
            overlap = (means @ means.t()).mean()
            return overlap

        # 42. Information Compression Ratio
        elif metric_name == 'information_compression_ratio':
            # 1 / Mutual Information
            # Proxy: 1 / (Entropy of logits - Conditional Entropy)
            # Just use 1 / Excess Entropy
            ee = Metrics.calculate_metric(model, 'excess_entropy', logits, labels, input_ids)
            return 1.0 / (ee + 1e-10)

        # 43. Trajectory Length (Gradient Norm)
        elif metric_name == 'trajectory_length':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            return torch.sqrt(sum([(g**2).sum() for g in grads]))

        # 44. Stochastic Loss Variance
        elif metric_name == 'stochastic_loss_variance':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            losses = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return torch.var(losses)

        # 45. Local Linearization Error
        elif metric_name == 'local_linearization_error':
            # Proxy: Hessian Trace * eps^2
            trace = Metrics.calculate_metric(model, 'hessian_trace', logits, labels, input_ids)
            return trace * 1e-4

        # 46. Output Jacobian Condition Number
        elif metric_name == 'output_jacobian_condition_number':
            # Cond(Cov(logits))
            if logits is None: return torch.tensor(0.0, device=device)
            l = logits.view(-1, logits.size(-1))
            n = l.size(0)
            v = l.size(1)
            
            if n < v:
                # Use Gram matrix G = L @ L.T which has same non-zero eigenvalues as L.T @ L
                # Size [N, N] (e.g. 8192x8192) vs [V, V] (50000x50000)
                cov = l @ l.t()
                # Eigenvalues of (L.T @ L + eps I) are (lambda_i + eps) and (eps)
                # Max is max(lambda) + eps
                # Min is eps (since rank <= N < V, there are zero eigenvalues)
                lambda_max = torch.linalg.eigvalsh(cov)[-1]
                return (lambda_max + 1e-6) / 1e-6
            else:
                cov = l.t() @ l
                return torch.linalg.cond(cov + 1e-6*torch.eye(cov.size(0), device=device))

        # 47. MDL Surrogate
        elif metric_name == 'mdl_surrogate':
            # LogDet(W^T W)
            val = 0.0
            for layer in model.transformer.layers:
                w = layer.linear1.weight
                try:
                    val += torch.logdet(w.t() @ w + 1e-6 * torch.eye(w.size(1), device=device))
                except: pass
            return torch.tensor(val, device=device)

        # 48. Kolmogorov Complexity Proxy
        elif metric_name == 'kolmogorov_complexity_proxy':
            # Entropy of singular values (sum over layers)
            val = 0.0
            for layer in model.transformer.layers:
                w = layer.linear1.weight
                try:
                    s = torch.linalg.svdvals(w)
                    s = s / (s.sum() + 1e-10)
                    val += -(s * torch.log(s + 1e-10)).sum()
                except: pass
            return torch.tensor(val, device=device)

        # 49. Kernel Target Alignment (KTA)
        elif metric_name == 'kernel_target_alignment':
            if features is None or labels is None: return torch.tensor(0.0, device=device)
            
            mask = labels != -100
            z = features.view(-1, features.size(-1))[mask.view(-1)]
            y = labels.view(-1)[mask.view(-1)]
            
            if len(y) > 2000: # Subsample if too large
                idx = torch.randperm(len(y))[:2000]
                z = z[idx]
                y = y[idx]
            
            z = F.normalize(z, p=2, dim=1)
            K = z @ z.t()
            
            # Y matrix
            y_mat = (y.unsqueeze(0) == y.unsqueeze(1)).float()
            
            num = (K * y_mat).sum()
            den = torch.norm(K) * torch.norm(y_mat)
            return num / (den + 1e-10)

        # 50. Classifier-Feature Alignment (NC4)
        elif metric_name == 'classifier_feature_alignment':
            if features is None or labels is None: return torch.tensor(0.0, device=device)
            mask = labels != -100
            z = features.view(-1, features.size(-1))[mask.view(-1)]
            y = labels.view(-1)[mask.view(-1)]
            
            W = model.lm_head.weight # [Vocab, H]
            
            classes = torch.unique(y)
            if len(classes) < 1: return torch.tensor(0.0, device=device)
            
            sim_sum = 0.0
            count = 0
            for c in classes:
                zc = z[y==c]
                if len(zc) > 0:
                    mu_c = zc.mean(dim=0)
                    w_c = W[c]
                    sim = F.cosine_similarity(mu_c.unsqueeze(0), w_c.unsqueeze(0))
                    sim_sum += sim
                    count += 1
            return sim_sum / (count + 1e-10)

        # 51. Simplex Equiangularity (NC2)
        elif metric_name == 'simplex_equiangularity':
            if features is None or labels is None: return torch.tensor(0.0, device=device)
            mask = labels != -100
            z = features.view(-1, features.size(-1))[mask.view(-1)]
            y = labels.view(-1)[mask.view(-1)]
            
            classes = torch.unique(y)
            if len(classes) < 2: return torch.tensor(0.0, device=device)
            
            means = []
            for c in classes:
                zc = z[y==c]
                if len(zc) > 0:
                    means.append(zc.mean(dim=0))
            
            if len(means) < 2: return torch.tensor(0.0, device=device)
            means = torch.stack(means)
            means = F.normalize(means, p=2, dim=1)
            
            G = means @ means.t()
            K = len(means)
            target = -1.0 / (K - 1.0)
            
            mask_off = ~torch.eye(K, dtype=torch.bool, device=device)
            off_diag = G[mask_off]
            
            mse = ((off_diag - target) ** 2).mean()
            return 1.0 / (mse + 1e-10)

        # 52. Minimum Margin
        elif metric_name == 'minimum_margin':
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margins = top2[:, :, 0] - top2[:, :, 1] # [B, S]
            return margins.min()

        # 53. Thermodynamic Susceptibility (Fisher Info Trace)
        elif metric_name == 'thermodynamic_susceptibility':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            # Re-compute loss with create_graph=True
            # Subsample for speed
            batch_size = logits.size(0)
            max_samples = 4
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
            susceptibility = sum([(g**2).sum() for g in grads])
            return susceptibility

        # 54. Algebraic Connectivity (Fiedler Value)
        elif metric_name == 'algebraic_connectivity':
            # Use the largest weight matrix (usually embedding or first layer)
            # Or average over layers. Let's use the first transformer layer's linear1
            w = model.transformer.layers[0].linear1.weight
            # Construct bipartite adjacency: [0, |W|; |W|^T, 0]
            # Size: (Out+In) x (Out+In)
            w_abs = w.abs()
            n_out, n_in = w_abs.shape
            size = n_out + n_in
            
            # Block matrix construction
            # A = torch.zeros(size, size, device=device)
            # A[:n_out, n_out:] = w_abs
            # A[n_out:, :n_out] = w_abs.t()
            
            # Degree matrix D is diagonal. D_ii = sum(A_ij)
            # Row sums:
            # For first n_out rows: sum(|W|, dim=1)
            # For last n_in rows: sum(|W|^T, dim=1) = sum(|W|, dim=0)
            d_out = w_abs.sum(dim=1)
            d_in = w_abs.sum(dim=0)
            d = torch.cat([d_out, d_in])
            
            # Laplacian L = D - A
            # We want 2nd smallest eigenvalue.
            # L is symmetric.
            # L = [diag(d_out), -|W|; -|W|^T, diag(d_in)]
            
            # To save memory, we can try to compute it for a smaller matrix if possible,
            # but 768+3072 = 3840, 3840^2 is ~14M elements, feasible.
            
            # Construct L explicitly
            L = torch.diag(d)
            L[:n_out, n_out:] -= w_abs
            L[n_out:, :n_out] -= w_abs.t()
            
            try:
                # eigvalsh for symmetric
                eigvals = torch.linalg.eigvalsh(L)
                # Sorted ascending. 0 is first.
                return eigvals[1]
            except:
                return torch.tensor(0.0, device=device)

        # 55. Persistent Entropy (Proxy)
        elif metric_name == 'persistent_entropy':
            if features is None: return torch.tensor(0.0, device=device)
            # Use features from the batch
            # Subsample
            z = features.view(-1, features.size(-1))
            if z.size(0) > 500:
                idx = torch.randperm(z.size(0))[:500]
                z = z[idx]
            
            # Pairwise distances
            dists = torch.cdist(z, z)
            # Normalize to prob
            probs = dists / (dists.sum() + 1e-10)
            # Entropy
            # Filter zeros
            mask = probs > 1e-10
            p = probs[mask]
            entropy = -(p * torch.log(p)).sum()
            return entropy

        # 56. Level Spacing Ratio
        elif metric_name == 'level_spacing_ratio':
            # Use embedding weight
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                s, _ = torch.sort(s)
                deltas = s[1:] - s[:-1]
                ratios = torch.minimum(deltas[1:], deltas[:-1]) / (torch.maximum(deltas[1:], deltas[:-1]) + 1e-10)
                return ratios.mean()
            except:
                return torch.tensor(0.0, device=device)

        # 57. Effective Information
        elif metric_name == 'effective_information':
            # Use first layer
            w = model.transformer.layers[0].linear1.weight # [Out, In]
            # Softmax over output (dim 0) to get P(out|in)
            # W_norm[i, j] = P(out_i | in_j)
            w_norm = torch.softmax(w, dim=0)
            
            # 1. Entropy of average output
            # P(out) = sum_j P(out|in_j) P(in_j)
            # Assume P(in_j) = 1/N
            avg_out = w_norm.mean(dim=1)
            h_avg = -(avg_out * torch.log(avg_out + 1e-10)).sum()
            
            # 2. Average entropy of outputs
            # H(out|in_j) = -sum_i P(out_i|in_j) log P(out_i|in_j)
            h_cond = -(w_norm * torch.log(w_norm + 1e-10)).sum(dim=0)
            avg_h_cond = h_cond.mean()
            
            return h_avg - avg_h_cond

        # 58. Ollivier-Ricci Curvature (Proxy)
        elif metric_name == 'ollivier_ricci_curvature':
            if features is None: return torch.tensor(0.0, device=device)
            # Subsample
            z = features.view(-1, features.size(-1))
            if z.size(0) > 200:
                idx = torch.randperm(z.size(0))[:200]
                z = z[idx]
            
            # 1. Adjacency (Correlation)
            # Normalize z
            z_norm = F.normalize(z, p=2, dim=1)
            adj = torch.mm(z_norm, z_norm.t()).abs() # [N, N]
            # Remove self-loops
            adj.fill_diagonal_(0)
            
            # 2. Distributions (Softmax of adjacency)
            m = F.softmax(adj * 10, dim=1) # Sharpen
            
            # 3. Geodesic distance (1 - adj)
            d = 1.0 - adj
            
            # 4. Wasserstein Proxy (Overlap)
            # W1(m_i, m_j) is hard.
            # Proxy: 1 - Overlap(m_i, m_j)
            # ORC(i, j) = 1 - W1 / d
            # We want to MAXIMIZE ORC.
            # Let's use a simpler proxy: Average Clustering Coefficient of the weighted graph
            # C_i = sum(w_ij * w_jk * w_ki) / ...
            # Or just maximize the "Triangle Density"
            
            # Let's implement the actual ORC proxy:
            # W1 approx ||m_i - m_j||_1 / 2 (for disjoint support)
            # But here they overlap.
            # Let's use Total Variation Distance as lower bound for W1
            # W1 >= TV = 0.5 * ||m_i - m_j||_1
            
            # Compute pairwise TV
            # m: [N, N]
            # m_i: [1, N], m_j: [N, 1] -> [N, N, N] too big.
            
            # Let's use a very simple proxy for "Positive Curvature":
            # High clustering.
            # Trace(A^3) / Trace(A^2) ?
            
            # Let's use Trace(A^3) (Number of triangles)
            # Normalize A
            A = adj / (adj.sum(dim=1, keepdim=True) + 1e-10)
            A3 = torch.mm(torch.mm(A, A), A)
            return torch.trace(A3)

        # 59. Von Neumann Entropy (Spectral Entropy)
        elif metric_name == 'von_neumann_entropy':
            if features is None: return torch.tensor(0.0, device=device)
            z = features.view(-1, features.size(-1))
            # Center
            z = z - z.mean(dim=0)
            # Covariance
            cov = (z.t() @ z) / (z.size(0) - 1)
            # Normalize to density matrix
            rho = cov / (torch.trace(cov) + 1e-10)
            # Eigenvalues
            try:
                e = torch.linalg.eigvalsh(rho)
                e = torch.clamp(e, min=1e-10)
                return -(e * torch.log(e)).sum()
            except:
                return torch.tensor(0.0, device=device)

        # 60. Interaction Information (Synergy Proxy)
        elif metric_name == 'interaction_information':
            # We want to MAXIMIZE Synergy (which is usually negative O-info).
            # Or minimize Redundancy.
            # Let's maximize: H(Joint) - Sum(H(Marginals)) (This is -Total Correlation)
            # Maximizing this means minimizing Total Correlation (Independence).
            # Wait, Synergy is different.
            # Synergy = Info(Whole) > Sum Info(Parts).
            # Let's use "O-Information" formula.
            # Omega = (N-2)H(X) + Sum[H(Xi) - H(X-i)]
            # If Omega < 0, Synergy dominates.
            # So we want to MINIMIZE Omega.
            # Since the code maximizes the metric, we return -Omega.
            
            if features is None: return torch.tensor(0.0, device=device)
            z = features.view(-1, features.size(-1))
            
            # Use Gaussian Entropy proxy
            # H(X) = 0.5 * logdet(Cov) + const
            
            # Subsample neurons (features) if too many
            if z.size(1) > 100:
                idx = torch.randperm(z.size(1))[:100]
                z = z[:, idx]
            
            N = z.size(1)
            cov = torch.cov(z.t())
            sign, logdet = torch.linalg.slogdet(cov + 1e-6 * torch.eye(N, device=device))
            if sign <= 0: return torch.tensor(0.0, device=device)
            h_joint = 0.5 * logdet
            
            # Sum of marginal entropies (diagonal)
            h_marginals = 0.5 * torch.log(torch.diag(cov) + 1e-6).sum()
            
            # Total Correlation = Sum(H_marginals) - H_joint
            # We want to minimize TC (maximize independence) OR maximize Synergy?
            # Let's return -Total Correlation (Maximize Independence)
            # This is a good proxy for "Disentanglement"
            return -(h_marginals - h_joint)

        # 61. Local Lyapunov Exponent (Expansion Rate)
        elif metric_name == 'local_lyapunov_exponent':
            # Proxy: Mean Log Singular Value of Layer Weights
            # If > 0, expansion. If < 0, contraction.
            # We want to maximize it (up to a point, usually 0).
            # But here we just return the value.
            val = 0.0
            count = 0
            for layer in model.transformer.layers:
                w = layer.linear1.weight
                try:
                    s = torch.linalg.svdvals(w)
                    val += torch.log(s.max() + 1e-10)
                    count += 1
                except: pass
            return torch.tensor(val / count, device=device)

        # 62. Multifractal Spectrum Width
        elif metric_name == 'multifractal_spectrum_width':
            if features is None: return torch.tensor(0.0, device=device)
            z = features.view(-1, features.size(-1))
            # Normalize
            z = (z - z.mean()) / (z.std() + 1e-8)
            
            # Coarse graining at scales 1, 2, 4
            scales = [1, 2, 4]
            moments = torch.tensor([0.5, 1.0, 2.0, 4.0], device=device)
            partition_functions = []
            
            for s in scales:
                if s == 1:
                    coarse = torch.abs(z)
                else:
                    # Avg pool 1d on flattened features
                    # Reshape to [B, L] -> [1, 1, B*L]
                    flat = torch.abs(z).view(1, 1, -1)
                    coarse = F.avg_pool1d(flat, kernel_size=s, stride=s).view(-1)
                
                # Z(q) = sum |x|^q
                Z_q = torch.stack([(coarse ** q).sum() for q in moments])
                partition_functions.append(torch.log(Z_q + 1e-8))
            
            # Tau(q) slope
            # log2(4) - log2(1) = 2.0
            tau_q = (partition_functions[-1] - partition_functions[0]) / 2.0
            
            # Alpha(q) = dTau/dq
            alpha = (tau_q[1:] - tau_q[:-1]) / (moments[1:] - moments[:-1])
            
            return alpha.max() - alpha.min()

        # 63. Predictive V-Information (Linear Probe Utility)
        elif metric_name == 'predictive_v_information':
            if features is None or labels is None: return torch.tensor(0.0, device=device)
            mask = labels != -100
            H = features.view(-1, features.size(-1))[mask.view(-1)]
            y = labels.view(-1)[mask.view(-1)]
            
            # Subsample
            if H.size(0) > 1000:
                idx = torch.randperm(H.size(0))[:1000]
                H = H[idx]
                y = y[idx]
            
            # One-hot targets
            num_classes = model.lm_head.out_features
            Y = F.one_hot(y, num_classes=num_classes).float()
            
            # Center
            H = H - H.mean(dim=0, keepdim=True)
            Y = Y - Y.mean(dim=0, keepdim=True)
            
            # Ridge Regression
            lambda_reg = 1e-3
            cov_H = H.t() @ H / H.size(0)
            cov_HY = H.t() @ Y / H.size(0)
            
            identity = torch.eye(H.size(1), device=device)
            try:
                W_opt = torch.linalg.solve(cov_H + lambda_reg * identity, cov_HY)
                Y_pred = H @ W_opt
                mse = ((Y - Y_pred) ** 2).mean()
                return -mse # Maximize negative MSE
            except:
                return torch.tensor(0.0, device=device)

        # 64. Log-Det Controllability Gramian
        elif metric_name == 'log_det_controllability':
            # Use first layer weight
            w = model.transformer.layers[0].linear1.weight
            G = w @ w.t()
            G = G / (G.norm() + 1e-8)
            G = G + 1e-6 * torch.eye(G.size(0), device=device)
            return torch.logdet(G)

        # 65. Trajectory Complexity (Logical Depth)
        elif metric_name == 'trajectory_complexity':
            if hidden_states is None: return torch.tensor(0.0, device=device)
            
            # hidden_states is a list of tensors [Batch, Seq, Dim]
            # We want to measure the path length in the activation space.
            
            # Stack layers: [Layers, Batch, Seq, Dim]
            # We can average over Batch and Seq, or compute per sample.
            
            # Let's compute per token and average.
            # Flatten Batch and Seq: [Layers, N, Dim]
            layers_stack = torch.stack(hidden_states)
            L, B, S, D = layers_stack.shape
            layers_flat = layers_stack.view(L, -1, D)
            
            # Subsample if too large
            if layers_flat.size(1) > 1000:
                idx = torch.randperm(layers_flat.size(1))[:1000]
                layers_flat = layers_flat[:, idx, :]
            
            # 1. Step lengths: ||h_{l+1} - h_l||
            steps = layers_flat[1:] - layers_flat[:-1]
            step_lengths = torch.norm(steps, dim=2).sum(dim=0) # Sum over layers -> [N]
            
            # 2. Displacement: ||h_L - h_0||
            displacement = layers_flat[-1] - layers_flat[0]
            disp_length = torch.norm(displacement, dim=1) # [N]
            
            # 3. Ratio
            ratio = step_lengths / (disp_length + 1e-6)
            
            # We want to MAXIMIZE this (more complex trajectory)
            return ratio.mean()

        # 66. Lempel-Ziv Complexity (Binarized)
        elif metric_name == 'lempel_ziv_complexity':
            if features is None: return torch.tensor(0.0, device=device)
            # Binarize features
            z = features.view(-1, features.size(-1))
            # Subsample
            if z.size(0) > 500:
                idx = torch.randperm(z.size(0))[:500]
                z = z[idx]
            
            threshold = z.mean()
            binary = (z > threshold).float()
            
            # Differentiable proxy: Entropy of soft binarization
            # sigmoid(beta * (z - threshold))
            beta = 10.0
            probs = torch.sigmoid(beta * (z - threshold))
            # Entropy of Bernoulli
            entropy = -(probs * torch.log(probs + 1e-10) + (1-probs) * torch.log(1-probs + 1e-10))
            return entropy.mean()

        # 67. Gradient Diversity
        elif metric_name == 'gradient_diversity':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            # Requires per-sample gradients.
            # Very expensive without functorch.
            # Let's use a "Batch Split" proxy.
            # Split batch into 4 mini-batches, compute gradients, measure diversity.
            
            batch_size = logits.size(0)
            if batch_size < 4: return torch.tensor(0.0, device=device)
            
            splits = 4
            split_size = batch_size // splits
            
            grads_list = []
            params = [p for p in model.parameters() if p.requires_grad]
            
            for i in range(splits):
                l_sub = logits[i*split_size : (i+1)*split_size]
                y_sub = labels[i*split_size : (i+1)*split_size]
                loss = nn.CrossEntropyLoss(ignore_index=-100)(l_sub.view(-1, l_sub.size(-1)), y_sub.view(-1))
                
                gs = torch.autograd.grad(loss, params, retain_graph=True)
                # Flatten
                g_flat = torch.cat([g.view(-1) for g in gs])
                grads_list.append(g_flat)
            
            grads_tensor = torch.stack(grads_list) # [4, P]
            
            sum_sq_norms = (grads_tensor.norm(dim=1) ** 2).sum()
            norm_sum_sq = grads_tensor.sum(dim=0).norm() ** 2
            
            return sum_sq_norms / (norm_sum_sq + 1e-10)

        # 68. Bi-Lipschitz Lower Bound (Min Singular Value)
        elif metric_name == 'bi_lipschitz_lower_bound':
            val = 0.0
            count = 0
            for layer in model.transformer.layers:
                w = layer.linear1.weight
                try:
                    # SVD is slow, use small proxy or just compute for one layer
                    # Or use power iteration for min SV?
                    # Let's use full SVD on just one layer to be safe on time
                    if count > 0: break 
                    s = torch.linalg.svdvals(w)
                    val += s.min()
                    count += 1
                except: pass
            return torch.tensor(val, device=device)

        # 69. Graph Modularity (Spectral)
        elif metric_name == 'graph_modularity':
            # Use first layer weights
            w = model.transformer.layers[0].linear1.weight
            # Symmetrize: W^T W
            adj = torch.abs(w.t() @ w)
            k = adj.sum(dim=1, keepdim=True)
            m = k.sum()
            if m < 1e-6: return torch.tensor(0.0, device=device)
            
            # Modularity Matrix B = A - k k^T / m
            # We want leading eigenvalue of B
            # Power iteration
            v = torch.randn(adj.size(0), 1, device=device)
            v = v / v.norm()
            
            # B v = A v - k (k^T v) / m
            for _ in range(5):
                Av = adj @ v
                kTv = (k.t() @ v).item()
                Bv = Av - k * (kTv / m)
                v = Bv / (Bv.norm() + 1e-10)
            
            # Rayleigh quotient: v^T B v
            Av = adj @ v
            kTv = (k.t() @ v).item()
            Bv = Av - k * (kTv / m)
            score = (v.t() @ Bv).squeeze()
            return score / m

        # 70. Graph Assortativity
        elif metric_name == 'graph_assortativity':
            w = model.transformer.layers[0].linear1.weight
            adj = torch.abs(w.t() @ w)
            k = adj.sum(dim=1)
            M = adj.sum()
            if M < 1e-6: return torch.tensor(0.0, device=device)
            
            # Pearson correlation of degrees of connected nodes
            # Numerator: sum_ij A_ij (k_i - mu)(k_j - mu)
            # Denom: sum_ij A_ij (k_i - mu)^2
            
            # Expected degree (weighted)
            mu = (k * k).sum() / M
            
            # Centered degrees
            k_centered = k - mu
            
            # Num = k_c^T A k_c
            num = k_centered @ adj @ k_centered
            
            # Denom = sum_i k_i * k_c_i^2 ? No.
            # Standard formula for weighted assortativity is complex.
            # Simplified: Correlation of k_source and k_target over edges.
            # Let's use the implementation from research:
            # term1 = sum(A * k_out * k_in) / M
            term1 = (adj * (k.unsqueeze(1) @ k.unsqueeze(0))).sum() / M
            term2 = (k * k).sum() / M # E[k] ? No, this is sum(k^2)/M
            # Actually term2 should be (sum(A * k) / M)^2 ?
            # Let's use the simplified Pearson form:
            # r = (Tr(K A K) - Tr(A K)^2 / M) / ...
            
            # Let's use a simpler proxy: Variance of degrees / Mean degree (Dispersion)
            # High assortativity often correlates with high variance?
            # No, let's stick to the correlation.
            
            # E[XY]
            exy = term1
            # E[X]
            ex = (adj @ k).sum() / M
            # E[X^2]
            ex2 = (adj @ (k**2)).sum() / M
            
            cov = exy - ex**2
            var = ex2 - ex**2
            return cov / (var + 1e-10)

        # 71. Small-Worldness (Sigma)
        elif metric_name == 'small_worldness':
            w = model.transformer.layers[0].linear1.weight
            adj = torch.abs(w.t() @ w)
            # Normalize
            adj = adj / (adj.max() + 1e-10)
            
            # Clustering Coefficient C
            # Onnela formula: Trace(A^(1/3)^3) / ...
            adj_cuberoot = torch.pow(adj, 1.0/3.0)
            # Trace(A^3)
            # A @ A @ A
            # Too expensive for 768x768? 768^3 is 450M ops, fine.
            A3 = adj_cuberoot @ adj_cuberoot @ adj_cuberoot
            triangles = torch.diagonal(A3).sum()
            
            k = adj.sum(dim=1)
            denom = (k * (k - 1)).sum()
            C = triangles / (denom + 1e-10)
            
            # Path length L proxy: 1 / Global Efficiency
            # E = 1/N(N-1) sum 1/d_ij
            # For dense graphs, L ~ 1.
            # So Sigma ~ C.
            return C

        # 72. Hurst Exponent (Rescaled Range)
        elif metric_name == 'hurst_exponent':
            if features is None: return torch.tensor(0.0, device=device)
            # features: [B, S, D]
            # Treat as time series of length S
            x = features # [B, S, D]
            # Mean over batch and dim? Or compute per series.
            # Let's compute for a subset of neurons
            if x.size(2) > 50:
                x = x[:, :, :50]
            
            # R/S Analysis
            # Y = cumsum(X - mean)
            # R = max(Y) - min(Y)
            # S = std(X)
            
            # We need to do this for different lags (window sizes)
            # But S is fixed (Seq Length).
            # Let's just compute R/S for the full sequence and return log(R/S) / log(S)
            # This is a point estimate of H.
            
            mean = x.mean(dim=1, keepdim=True)
            y = torch.cumsum(x - mean, dim=1)
            r_val = y.max(dim=1).values - y.min(dim=1).values
            s_val = x.std(dim=1)
            
            rs = r_val / (s_val + 1e-10)
            # H ~ log(RS) / log(N)
            n = x.size(1)
            h = torch.log(rs + 1e-10) / math.log(n)
            return h.mean()

        # 73. Approximate Entropy (ApEn)
        elif metric_name == 'approximate_entropy':
            if features is None: return torch.tensor(0.0, device=device)
            # [B, S, D] -> [B, S] (Mean over D)
            seq = features.mean(dim=2)
            # Subsample batch
            if seq.size(0) > 10: seq = seq[:10]
            
            m = 2
            r = 0.2 * seq.std(dim=1, keepdim=True)
            
            def phi(u, m, r):
                # u: [B, S]
                B, S = u.shape
                # Unfold to [B, S-m+1, m]
                if S < m: return torch.zeros(B, device=device)
                x = u.unfold(1, m, 1)
                # Pairwise distances (Chebyshev)
                # [B, N, 1, m] - [B, 1, N, m]
                dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).max(dim=-1).values
                # Count < r
                count = (dist < r.unsqueeze(2)).float().mean(dim=2)
                return torch.log(count + 1e-10).mean(dim=1)
            
            p2 = phi(seq, m, r)
            p3 = phi(seq, m+1, r)
            return (p2 - p3).mean()

        # 74. Binding Information (Total Correlation)
        elif metric_name == 'binding_information':
            if features is None: return torch.tensor(0.0, device=device)
            z = features.view(-1, features.size(-1))
            if z.size(0) > 1000:
                idx = torch.randperm(z.size(0))[:1000]
                z = z[idx]
            
            # TC = Sum H(Xi) - H(X)
            # Gaussian proxy
            cov = torch.cov(z.t())
            cov = cov + 1e-6 * torch.eye(cov.size(0), device=device)
            
            # H(X)
            sign, logdet = torch.linalg.slogdet(cov)
            h_joint = 0.5 * logdet
            
            # Sum H(Xi)
            h_marginals = 0.5 * torch.log(torch.diag(cov)).sum()
            
            return h_marginals - h_joint

        # 75. Weight Quantum Discord (Spectral Entropy of Correlation)
        elif metric_name == 'weight_quantum_discord':
            w = model.embedding.weight
            c = w @ w.t()
            # Normalize to density matrix
            rho = c / (torch.trace(c) + 1e-10)
            # Eigenvalues
            try:
                e = torch.linalg.eigvalsh(rho)
                e = torch.clamp(e, min=1e-12)
                return -(e * torch.log(e)).sum()
            except: return torch.tensor(0.0, device=device)

        # 76. Specific Heat (Spectrum Variance)
        elif metric_name == 'specific_heat':
            w = model.embedding.weight
            try:
                s = torch.linalg.svdvals(w)
                # Normalize
                s = s / s.sum()
                # Variance
                var = torch.var(s)
                return var
            except: return torch.tensor(0.0, device=device)

        # 77. Synaptic Homeostasis
        elif metric_name == 'synaptic_homeostasis':
            # Negative variance of weight norms
            w = model.transformer.layers[0].linear1.weight
            norms = w.norm(dim=1)
            return -torch.var(norms)

        # 78. Topological Entropy (Spectral Radius)
        elif metric_name == 'topological_entropy':
            # Log of spectral radius of transition matrix
            w = model.transformer.layers[0].linear1.weight
            # Make square if not
            if w.size(0) != w.size(1):
                m = min(w.size(0), w.size(1))
                w = w[:m, :m]
            
            try:
                # Power iteration for spectral radius
                v = torch.randn(w.size(0), 1, device=device)
                v = v / v.norm()
                for _ in range(5):
                    v = w @ v
                    v = v / v.norm()
                
                rho = (v.t() @ (w @ v)).abs()
                return torch.log(rho + 1e-10).squeeze()
            except: return torch.tensor(0.0, device=device)

        # 79. Network Motif Count (Triangles)
        elif metric_name == 'motif_count':
            w = model.transformer.layers[0].linear1.weight
            adj = torch.abs(w.t() @ w)
            # Normalize
            adj = adj / (adj.max() + 1e-10)
            # Threshold to make sparse? Or weighted count.
            # Weighted: Trace(A^3)
            # Subsample if too large
            if adj.size(0) > 500:
                adj = adj[:500, :500]
            
            A3 = torch.mm(adj, torch.mm(adj, adj))
            triangles = torch.trace(A3) / 6.0
            return triangles

        # 80. Rich-Club Coefficient
        elif metric_name == 'rich_club_coefficient':
            w = model.transformer.layers[0].linear1.weight
            adj = torch.abs(w.t() @ w)
            k = adj.sum(dim=1)
            
            # Define rich nodes as top 10% degree
            threshold = torch.quantile(k, 0.9)
            rich_mask = k > threshold
            
            if rich_mask.sum() < 2: return torch.tensor(0.0, device=device)
            
            # Subgraph of rich nodes
            rich_adj = adj[rich_mask][:, rich_mask]
            phi = rich_adj.mean() # Density of rich club
            
            # Normalize by overall density
            rho = adj.mean()
            return phi / (rho + 1e-10)

        # 81. Percolation Threshold (Molloy-Reed)
        elif metric_name == 'percolation_threshold':
            w = model.transformer.layers[0].linear1.weight
            adj = torch.abs(w.t() @ w)
            # Binarize for standard definition? Or weighted degrees.
            # Let's use weighted degrees.
            k = adj.sum(dim=1)
            
            k_mean = k.mean()
            k2_mean = (k**2).mean()
            
            kappa = k2_mean / (k_mean + 1e-10)
            # Robustness ~ kappa.
            return kappa

        # 82. Kuramoto Order Parameter (Synchronization)
        elif metric_name == 'kuramoto_order':
            if features is None: return torch.tensor(0.0, device=device)
            # Map activations to phases [0, 2pi]
            z = features.view(-1, features.size(-1))
            if z.size(0) > 1000: z = z[:1000]
            
            min_val = z.min(dim=1, keepdim=True)[0]
            max_val = z.max(dim=1, keepdim=True)[0]
            phases = 2 * math.pi * (z - min_val) / (max_val - min_val + 1e-10)
            
            # Order parameter r = |mean(e^i theta)|
            real = torch.cos(phases).mean(dim=1)
            imag = torch.sin(phases).mean(dim=1)
            r = torch.sqrt(real**2 + imag**2)
            return r.mean()

        # 83. Avalanche Size Distribution (Activity Bursts)
        elif metric_name == 'avalanche_size':
            if features is None: return torch.tensor(0.0, device=device)
            # features: [B, S, D]
            # Activity: sum of activations per time step
            activity = features.abs().mean(dim=2) # [B, S]
            
            # Define avalanche as continuous region above threshold
            threshold = activity.mean()
            active = (activity > threshold).float()
            
            # We want distribution of avalanche sizes (durations)
            # This is hard to vectorize perfectly.
            # Proxy: Variance of activity (Burstiness)
            return torch.var(activity)

        # 84. Sample Entropy (SampEn)
        elif metric_name == 'sample_entropy':
            if features is None: return torch.tensor(0.0, device=device)
            seq = features.mean(dim=2) # [B, S]
            if seq.size(0) > 10: seq = seq[:10]
            
            m = 2
            r = 0.2 * seq.std(dim=1, keepdim=True)
            
            def get_matches(u, m, r):
                B, S = u.shape
                if S < m: return torch.zeros(B, device=device)
                x = u.unfold(1, m, 1)
                # Pairwise dists
                dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).max(dim=-1).values
                # Count matches (excluding self if we were strict, but here we include)
                count = (dist < r.unsqueeze(2)).float().sum(dim=2) - 1.0 # Exclude self
                return count.mean(dim=1) # Average count per template
            
            B_count = get_matches(seq, m, r)
            A_count = get_matches(seq, m+1, r)
            
            return -torch.log((A_count + 1e-10) / (B_count + 1e-10)).mean()

        # 85. Permutation Entropy
        elif metric_name == 'permutation_entropy':
            if features is None: return torch.tensor(0.0, device=device)
            seq = features.mean(dim=2) # [B, S]
            m = 3
            if seq.size(1) < m: return torch.tensor(0.0, device=device)
            
            windows = seq.unfold(1, m, 1) # [B, N, m]
            # Argsort to get permutations
            perms = torch.argsort(windows, dim=2)
            
            # Hash permutations: sum(idx * m^k)
            scale = torch.tensor([m**i for i in range(m)], device=device)
            hashes = (perms * scale).sum(dim=2) # [B, N]
            
            # Entropy per batch item
            ent_sum = 0.0
            for i in range(hashes.size(0)):
                u, c = torch.unique(hashes[i], return_counts=True)
                p = c.float() / c.sum()
                ent_sum += -(p * torch.log(p + 1e-10)).sum()
            
            return ent_sum / hashes.size(0)

        # 86. Erasure Entropy (ReLU Information Loss)
        elif metric_name == 'erasure_entropy':
            if features is None: return torch.tensor(0.0, device=device)
            # Fraction of dead neurons (<= 0)
            # Assuming ReLU-like activation
            dead = (features <= 0).float()
            p_dead = dead.mean(dim=0) # Prob of being dead per neuron
            
            # Entropy of binary state
            h = -p_dead * torch.log(p_dead + 1e-10) - (1-p_dead) * torch.log(1-p_dead + 1e-10)
            return h.mean()

        # 87. Global Efficiency (Inverse Path Length)
        elif metric_name == 'global_efficiency':
            w = model.transformer.layers[0].linear1.weight
            adj = torch.abs(w.t() @ w)
            # Subsample
            if adj.size(0) > 200: adj = adj[:200, :200]
            
            # Invert weights to get "distances" (strong weight = short distance)
            dist = 1.0 / (adj + 1e-6)
            
            # Floyd-Warshall is O(N^3), too slow.
            # Use simple proxy: Mean of 1/dist (which is just Mean of adj)
            # Real Global Efficiency requires shortest paths.
            # Let's use "Communicability" (Exp(A))
            # G_eff ~ sum(exp(A)_ij)
            
            # Let's use Communicability
            # expA = U exp(S) U^T
            try:
                s = torch.linalg.svdvals(adj)
                comm = torch.exp(s).sum()
                return torch.log(comm)
            except: return adj.mean()

        # 88. Causal Emergence (Psi)
        elif metric_name == 'causal_emergence':
            w = model.transformer.layers[0].linear1.weight
            # Normalize to transition matrix
            P = torch.softmax(w, dim=1)
            
            # Micro EI
            avg_out = P.mean(dim=0)
            h_avg = -(avg_out * torch.log(avg_out + 1e-10)).sum()
            row_ent = -(P * torch.log(P + 1e-10)).sum(dim=1).mean()
            ei_micro = h_avg - row_ent
            
            # Macro: Coarse grain 2x2
            # Reshape [Out, In] -> [Out/2, 2, In/2, 2]
            if P.size(0) % 2 == 0 and P.size(1) % 2 == 0:
                P_macro = P.view(P.size(0)//2, 2, P.size(1)//2, 2).sum(dim=(1, 3))
                # Renormalize
                P_macro = P_macro / (P_macro.sum(dim=1, keepdim=True) + 1e-10)
                
                avg_out_m = P_macro.mean(dim=0)
                h_avg_m = -(avg_out_m * torch.log(avg_out_m + 1e-10)).sum()
                row_ent_m = -(P_macro * torch.log(P_macro + 1e-10)).sum(dim=1).mean()
                ei_macro = h_avg_m - row_ent_m
                
                return ei_macro - ei_micro
            else:
                return torch.tensor(0.0, device=device)

        # 89. Higuchi Fractal Dimension
        elif metric_name == 'higuchi_fractal_dimension':
            if features is None: return torch.tensor(0.0, device=device)
            # [B, S, D] -> [B, S]
            x = features.mean(dim=2)
            if x.size(0) > 10: x = x[:10] # Subsample batch
            
            N = x.size(1)
            k_max = 10
            L_k = []
            k_values = []
            
            for k in range(1, k_max + 1):
                Lk_m = []
                for m in range(k):
                    # Construct series: x[m], x[m+k], ...
                    idxs = torch.arange(m, N, k, device=device)
                    if len(idxs) < 2: continue
                    
                    series = x[:, idxs]
                    # Sum of absolute differences
                    diffs = torch.abs(series[:, 1:] - series[:, :-1]).sum(dim=1)
                    norm = (N - 1) / (len(idxs) - 1) / k
                    Lk_m.append(diffs * norm)
                
                if len(Lk_m) > 0:
                    L_k.append(torch.stack(Lk_m).mean(dim=0)) # Mean over m, keep batch
                    k_values.append(k)
            
            if len(L_k) < 2: return torch.tensor(0.0, device=device)
            
            L_k = torch.stack(L_k).mean(dim=1) # Mean over batch -> [K]
            log_L = torch.log(L_k + 1e-10)
            log_k = torch.log(torch.tensor(k_values, device=device).float())
            log_1_k = -log_k
            
            # Slope of log_L vs log_1_k
            mean_x = log_1_k.mean()
            mean_y = log_L.mean()
            num = ((log_1_k - mean_x) * (log_L - mean_y)).sum()
            den = ((log_1_k - mean_x)**2).sum()
            return num / (den + 1e-10)

        # 90. Petrosian Fractal Dimension
        elif metric_name == 'petrosian_fractal_dimension':
            if features is None: return torch.tensor(0.0, device=device)
            x = features.mean(dim=2) # [B, S]
            if x.size(0) > 50: x = x[:50]
            
            # Binarize based on diff
            diff = x[:, 1:] - x[:, :-1]
            # Sign changes
            sign_changes = (diff[:, 1:] * diff[:, :-1] < 0).float().sum(dim=1)
            
            N = x.size(1)
            D = torch.log10(torch.tensor(N, device=device).float()) / (torch.log10(torch.tensor(N, device=device).float()) + torch.log10(N / (N + 0.4 * sign_changes + 1e-10)))
            return D.mean()

        # 91. Estrada Index (Graph Folding)
        elif metric_name == 'estrada_index':
            w = model.transformer.layers[0].linear1.weight
            adj = torch.abs(w.t() @ w)
            # Subsample
            if adj.size(0) > 500: adj = adj[:500, :500]
            
            # EE = Trace(exp(A))
            # exp(A) = U exp(D) U^T
            try:
                s = torch.linalg.eigvalsh(adj)
                ee = torch.exp(s).sum()
                return torch.log(ee) # Return log to keep scale reasonable
            except: return torch.tensor(0.0, device=device)

        # 92. Weight Stable Rank
        elif metric_name == 'weight_stable_rank':
            w = model.embedding.weight
            # ||W||_F^2 / ||W||_2^2
            frob = (w**2).sum()
            try:
                # Spectral norm (max singular value)
                # Power iteration for speed
                u = torch.randn(w.size(0), device=device)
                u = u / u.norm()
                v = torch.randn(w.size(1), device=device)
                v = v / v.norm()
                for _ in range(5):
                    v = (u @ w) / (u @ w).norm()
                    u = (w @ v) / (w @ v).norm()
                spectral = (u @ w @ v)
                return frob / (spectral**2 + 1e-10)
            except: return torch.tensor(0.0, device=device)

        # 93. Local Intrinsic Dimension (LID)
        elif metric_name == 'local_intrinsic_dimension':
            if features is None: return torch.tensor(0.0, device=device)
            z = features.view(-1, features.size(-1))
            if z.size(0) > 1000:
                idx = torch.randperm(z.size(0))[:1000]
                z = z[idx]
            
            # k-NN
            k = 20
            dists = torch.cdist(z, z)
            # Top k+1 (including self)
            topk = dists.topk(k+1, largest=False).values
            
            # MLE estimator: LID = -1 / ( 1/k * sum log(d_i / d_k) )
            # d_k is the k-th neighbor distance (last col of topk)
            d_k = topk[:, -1:]
            d_i = topk[:, 1:-1] # Exclude self (0) and k-th
            
            ratio = d_i / (d_k + 1e-10)
            log_ratio = torch.log(ratio + 1e-10)
            lid = -1.0 / (log_ratio.mean(dim=1) + 1e-10)
            return lid.mean()

        # 94. Quantum Coherence (L1)
        elif metric_name == 'quantum_coherence':
            if features is None: return torch.tensor(0.0, device=device)
            z = features.view(-1, features.size(-1))
            if z.size(0) > 200: z = z[:200]
            
            # Density matrix
            z = z - z.mean(dim=0)
            rho = z @ z.t()
            rho = rho / (torch.trace(rho) + 1e-10)
            
            # Sum of off-diagonal magnitudes
            off_diag = torch.abs(rho).sum() - torch.trace(torch.abs(rho))
            return off_diag

        # 95. Spectral Gap
        elif metric_name == 'spectral_gap':
            w = model.transformer.layers[0].linear1.weight
            adj = torch.abs(w.t() @ w)
            if adj.size(0) > 500: adj = adj[:500, :500]
            
            # Laplacian
            d = adj.sum(dim=1)
            L = torch.diag(d) - adj
            
            try:
                e = torch.linalg.eigvalsh(L)
                # Sorted: 0, lambda_2, ...
                if len(e) > 1:
                    return e[1] # Fiedler value is the gap from 0
                return torch.tensor(0.0, device=device)
            except: return torch.tensor(0.0, device=device)

        # 96. Inverse Participation Ratio (IPR)
        elif metric_name == 'inverse_participation_ratio':
            if features is None: return torch.tensor(0.0, device=device)
            # Localization of activity
            z = features.view(-1, features.size(-1))
            # Normalize per neuron (column)
            z_norm = F.normalize(z, p=2, dim=0)
            ipr = (z_norm ** 4).sum(dim=0)
            return ipr.mean()

        # 97. Gradient Signal-to-Noise Ratio
        elif metric_name == 'gradient_snr':
            if logits is None or labels is None: return torch.tensor(0.0, device=device)
            # Compute per-sample gradients (expensive) or batch split
            # Let's use batch split (4 splits)
            batch_size = logits.size(0)
            if batch_size < 4: return torch.tensor(0.0, device=device)
            
            splits = 4
            split_size = batch_size // splits
            grads_list = []
            params = [p for p in model.parameters() if p.requires_grad]
            
            for i in range(splits):
                l_sub = logits[i*split_size : (i+1)*split_size]
                y_sub = labels[i*split_size : (i+1)*split_size]
                loss = nn.CrossEntropyLoss(ignore_index=-100)(l_sub.view(-1, l_sub.size(-1)), y_sub.view(-1))
                gs = torch.autograd.grad(loss, params, retain_graph=True)
                g_flat = torch.cat([g.view(-1) for g in gs])
                grads_list.append(g_flat)
            
            grads = torch.stack(grads_list) # [4, P]
            mean_grad = grads.mean(dim=0)
            std_grad = grads.std(dim=0)
            
            # SNR = mean / std
            # Average SNR across parameters
            snr = mean_grad.abs() / (std_grad + 1e-10)
            return snr.mean()

        # 98. Cross-Layer Entropy
        elif metric_name == 'cross_layer_entropy':
            if hidden_states is None: return torch.tensor(0.0, device=device)
            # Joint entropy of first and last layer
            h0 = hidden_states[0].view(-1, hidden_states[0].size(-1))
            hL = hidden_states[-1].view(-1, hidden_states[-1].size(-1))
            
            if h0.size(0) > 500:
                idx = torch.randperm(h0.size(0))[:500]
                h0 = h0[idx]
                hL = hL[idx]
            
            # Concat
            joint = torch.cat([h0, hL], dim=1)
            
            def get_entropy(x):
                cov = torch.cov(x.t())
                sign, logdet = torch.linalg.slogdet(cov + 1e-6 * torch.eye(cov.size(0), device=device))
                return 0.5 * logdet
            
            return get_entropy(joint)

        # 99. Weight Skewness
        elif metric_name == 'weight_skewness':
            w = model.embedding.weight.view(-1)
            mean = w.mean()
            std = w.std()
            skew = ((w - mean)**3).mean() / (std**3 + 1e-10)
            return torch.abs(skew)

        # 100. Weight Kurtosis
        elif metric_name == 'weight_kurtosis':
            w = model.embedding.weight.view(-1)
            mean = w.mean()
            std = w.std()
            kurt = ((w - mean)**4).mean() / (std**4 + 1e-10)
            return kurt

        return torch.tensor(0.0, device=device)

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
    
    def forward(self, input_ids, return_features=False, return_all_layers=False):
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        embeddings = self.embedding(input_ids) + self.position_embedding(positions)
        
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=input_ids.device), 
            diagonal=1
        ).bool()
    
        if return_all_layers:
            hidden_states = [embeddings]
            out = embeddings
            for layer in self.transformer.layers:
                out = layer(out, src_mask=causal_mask)
                hidden_states.append(out)
            transformer_out = out
        else:
            transformer_out = self.transformer(
                embeddings,
                mask=causal_mask
            )
        
        logits = self.lm_head(transformer_out)
        
        if return_all_layers:
            return logits, hidden_states
            
        if return_features:
            return logits, transformer_out
            
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
        
        hidden_states = None
        if config.METRIC_NAME == 'trajectory_complexity':
            logits, hidden_states = model(input_ids, return_all_layers=True)
            features = hidden_states[-1]
        else:
            logits, features = model(input_ids, return_features=True)
        
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
        
        if metric_value is None or batch_idx % config.COMPLEXITY_UPDATE_INTERVAL == 0:
            metric_tensor = Metrics.calculate_metric(model, config.METRIC_NAME, logits, labels, input_ids, features, hidden_states=hidden_states)
            metric_value = metric_tensor 
            metric_value_scalar = metric_tensor.item()
 
        # New Loss Formula: (x_start / (x_value + 1e-10)) * ce_start
        # We want to MAXIMIZE x_value (disequilibrium).
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
        hidden_states = None
        if config.METRIC_NAME == 'trajectory_complexity':
            metric_logits, hidden_states = model(metric_input_ids, return_all_layers=True)
            metric_features = hidden_states[-1]
        else:
            metric_logits, metric_features = model(metric_input_ids, return_features=True)
            
        start_metric = Metrics.calculate_metric(model, config.METRIC_NAME, metric_logits, metric_labels, metric_input_ids, metric_features, hidden_states=hidden_states).item()
    print(f"Initial CE loss: {start_ce:.16f}")
    print(f"Initial Metric ({config.METRIC_NAME}): {start_metric:.16f}")
    model.train()
    
    prev_val_loss = None
    current_slope = 0.0
    lmc_weight = 1.0
    
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
            lmc_weight = 1.0
        
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
        (False, 'disequilibrium', '0_disequilibrium'),
        (False, 'symmetry_breaking', '1_symmetry_breaking'),
        (False, 'excess_entropy', '2_excess_entropy'),
        (False, 'energy_distance', '3_energy_distance'),
        (False, 'flatness_avoidance', '4_flatness_avoidance'),
        (False, 'multimodality', '5_multimodality'),
        (False, 'broken_ergodicity', '6_broken_ergodicity'),
        (False, 'participation_ratio', '7_participation_ratio'),
        (False, 'decision_sharpness', '8_decision_sharpness'),
        (False, 'fisher_rao_distance', '9_fisher_rao_distance'),
        (False, 'wasserstein_uniform', '10_wasserstein_uniform'),
        (False, 'gini_index', '11_gini_index'),
        (False, 'semantic_diversity', '12_semantic_diversity'),
        (False, 'inverse_spectral_flatness', '13_inverse_spectral_flatness'),
        (False, 'hoyer_sparsity', '14_hoyer_sparsity'),
        (False, 'varentropy', '15_varentropy'),
        (False, 'jensen_shannon', '16_jensen_shannon'),
        (False, 'hellinger', '17_hellinger'),
        (False, 'total_variation', '18_total_variation'),
        (False, 'weight_anisotropy', '19_weight_anisotropy'),
        (False, 'weight_gini', '20_weight_gini'),
        (False, 'logit_rank', '21_logit_rank'),
        (False, 'logit_variance', '22_logit_variance'),
        (False, 'margin_entropy', '23_margin_entropy'),
        (False, 'cluster_distance', '24_cluster_distance'),
        (False, 'feature_rank', '25_feature_rank'),
        (False, 'fractal_dim', '26_fractal_dim'),
        (False, 'pac_bayes', '27_pac_bayes'),
        (False, 'path_norm', '28_path_norm'),
        (False, 'hessian_trace', '29_hessian_trace'),
        (False, 'hessian_spectral_radius', '30_hessian_spectral_radius'),
        (False, 'fisher_info_condition_number', '31_fisher_info_condition_number'),
        (False, 'sharpness_perturbation', '32_sharpness_perturbation'),
        (False, 'curvature_energy', '33_curvature_energy'),
        (False, 'gradient_covariance_trace', '34_gradient_covariance_trace'),
        (False, 'gradient_direction_entropy', '35_gradient_direction_entropy'),
        (False, 'gradient_cosine_drift', '36_gradient_cosine_drift'),
        (False, 'activation_jacobian_frobenius_norm', '37_activation_jacobian_frobenius_norm'),
        (False, 'layerwise_lipschitz', '38_layerwise_lipschitz'),
        (False, 'effective_rank_activations', '39_effective_rank_activations'),
        (False, 'log_det_activation_covariance', '40_log_det_activation_covariance'),
        (False, 'class_conditional_overlap', '41_class_conditional_overlap'),
        (False, 'information_compression_ratio', '42_information_compression_ratio'),
        (False, 'trajectory_length', '43_trajectory_length'),
        (False, 'stochastic_loss_variance', '44_stochastic_loss_variance'),
        (False, 'local_linearization_error', '45_local_linearization_error'),
        (False, 'output_jacobian_condition_number', '46_output_jacobian_condition_number'),
        (False, 'mdl_surrogate', '47_mdl_surrogate'),
        (False, 'kolmogorov_complexity_proxy', '48_kolmogorov_complexity_proxy'),
        (False, 'kernel_target_alignment', '49_kernel_target_alignment'),
        (False, 'classifier_feature_alignment', '50_classifier_feature_alignment'),
        (False, 'simplex_equiangularity', '51_simplex_equiangularity'),
        (False, 'minimum_margin', '52_minimum_margin'),
        (False, 'thermodynamic_susceptibility', '53_thermodynamic_susceptibility'),
        (False, 'algebraic_connectivity', '54_algebraic_connectivity'),
        (False, 'persistent_entropy', '55_persistent_entropy'),
        (False, 'level_spacing_ratio', '56_level_spacing_ratio'),
        (False, 'effective_information', '57_effective_information'),
        (False, 'ollivier_ricci_curvature', '58_ollivier_ricci_curvature'),
        (False, 'von_neumann_entropy', '59_von_neumann_entropy'),
        (False, 'interaction_information', '60_interaction_information'),
        (False, 'local_lyapunov_exponent', '61_local_lyapunov_exponent'),
        (False, 'multifractal_spectrum_width', '62_multifractal_spectrum_width'),
        (False, 'predictive_v_information', '63_predictive_v_information'),
        (False, 'log_det_controllability', '64_log_det_controllability'),
        (False, 'trajectory_complexity', '65_trajectory_complexity'),
        (False, 'lempel_ziv_complexity', '66_lempel_ziv_complexity'),
        (False, 'gradient_diversity', '67_gradient_diversity'),
        (False, 'bi_lipschitz_lower_bound', '68_bi_lipschitz_lower_bound'),
        (False, 'graph_modularity', '69_graph_modularity'),
        (False, 'graph_assortativity', '70_graph_assortativity'),
        (False, 'small_worldness', '71_small_worldness'),
        (False, 'hurst_exponent', '72_hurst_exponent'),
        (False, 'approximate_entropy', '73_approximate_entropy'),
        (False, 'binding_information', '74_binding_information'),
        (False, 'weight_quantum_discord', '75_weight_quantum_discord'),
        (False, 'specific_heat', '76_specific_heat'),
        (False, 'synaptic_homeostasis', '77_synaptic_homeostasis'),
        (False, 'topological_entropy', '78_topological_entropy'),
        (False, 'motif_count', '79_motif_count'),
        (False, 'rich_club_coefficient', '80_rich_club_coefficient'),
        (False, 'percolation_threshold', '81_percolation_threshold'),
        (False, 'kuramoto_order', '82_kuramoto_order'),
        (False, 'avalanche_size', '83_avalanche_size'),
        (False, 'sample_entropy', '84_sample_entropy'),
        (False, 'permutation_entropy', '85_permutation_entropy'),
        (False, 'erasure_entropy', '86_erasure_entropy'),
        (False, 'global_efficiency', '87_global_efficiency'),
        (False, 'causal_emergence', '88_causal_emergence'),
        (False, 'higuchi_fractal_dimension', '89_higuchi_fractal_dimension'),
        (False, 'petrosian_fractal_dimension', '90_petrosian_fractal_dimension'),
        (False, 'estrada_index', '91_estrada_index'),
        (False, 'weight_stable_rank', '92_weight_stable_rank'),
        (False, 'local_intrinsic_dimension', '93_local_intrinsic_dimension'),
        (False, 'quantum_coherence', '94_quantum_coherence'),
        (False, 'spectral_gap', '95_spectral_gap'),
        (False, 'inverse_participation_ratio', '96_inverse_participation_ratio'),
        (False, 'gradient_snr', '97_gradient_snr'),
        (False, 'cross_layer_entropy', '98_cross_layer_entropy'),
        (False, 'weight_skewness', '99_weight_skewness'),
        (False, 'weight_kurtosis', '100_weight_kurtosis'),
    ]
    
    for control_mode, metric_name, folder_name in experiments:
        output_dir = os.path.join(script_dir, f'output_max/{folder_name}')
        
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
