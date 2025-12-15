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
    def calculate_metric(model, metric_name, logits=None, labels=None, input_ids=None):
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

        # 12. Theil Index (Inequality)
        elif metric_name == 'theil_index':
            # T = (1/N) sum (x_i / mu) log (x_i / mu)
            # For probs, mu = 1/N.
            # T = (1/N) sum (N p_i) log (N p_i)
            #   = sum p_i (log p_i + log N)
            #   = -H(p) + log N
            # This is exactly KL(p || u) or Energy Distance.
            # Let's implement it explicitly as Theil.
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n = probs.size(-1)
            theil = (probs * (torch.log(probs + 1e-10) + math.log(n))).sum(dim=-1)
            return theil.mean()

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

        # 14. Renyi Divergence (Alpha=2)
        elif metric_name == 'renyi_divergence':
            # D_a(p||u) = 1/(a-1) log( sum p^a / u^(a-1) )
            # alpha = 2
            # log( sum p^2 / (1/N) ) = log( N * sum p^2 )
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n = probs.size(-1)
            sum_sq = (probs ** 2).sum(dim=-1)
            return torch.log(n * sum_sq + 1e-10).mean()

        # 15. Tsallis Divergence (Alpha=2)
        elif metric_name == 'tsallis_divergence':
            # D_q(p||u) = 1/(q-1) (1 - sum p^q u^(1-q)) ? No, standard definition:
            # 1/(q-1) ( sum(p^q / u^(q-1)) - 1 )
            # q=2: sum(p^2 / (1/N)) - 1 = N * sum(p^2) - 1
            if logits is None: return torch.tensor(0.0, device=device)
            probs = torch.softmax(logits, dim=-1)
            n = probs.size(-1)
            sum_sq = (probs ** 2).sum(dim=-1)
            return (n * sum_sq - 1.0).mean()

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
            # Need create_graph=True for double backprop
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
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
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
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
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
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

        # 33. PAC-Bayes Flatness
        elif metric_name == 'pac_bayes_flatness':
            # Trace(H) * sigma^2
            trace = Metrics.calculate_metric(model, 'hessian_trace', logits, labels, input_ids)
            return trace * 0.001

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
        (False, 'theil_index', '12_theil_index'),
        (False, 'inverse_spectral_flatness', '13_inverse_spectral_flatness'),
        (False, 'renyi_divergence', '14_renyi_divergence'),
        (False, 'tsallis_divergence', '15_tsallis_divergence'),
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
        (False, 'pac_bayes_flatness', '33_pac_bayes_flatness'),
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
