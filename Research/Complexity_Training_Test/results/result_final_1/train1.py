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

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    CONTROL_MODE = False  # False = CE + LMC optimization | True = CE only
    
    # Model hyperparameters
    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    NUM_ATTENTION_HEADS = 4 # Standard ratio (hidden_dim / num_heads = 64)
    
    # Training hyperparameters
    BATCH_SIZE = 256 
    EPOCHS = 50
    SEQ_LENGTH = 32
    MAX_GRAD_NORM = 1.0
    MAX_SAMPLES = None
    
    # Number of runs
    NUM_OF_RUN_PER_CALL = 3
    
    # Complexity calculation interval | Calculate LMC every X batches (1 = every batch)
    COMPLEXITY_UPDATE_INTERVAL = 1 
    
    # Device configuration
    GPU_INDEX = 1  # Which GPU to use (0, 1, 2, etc.)
    DEVICE = torch.device(f'cuda:{GPU_INDEX}' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 8  # DataLoader workers
    
    # Performance optimizations
    USE_COMPILE = False  
    
    # DONT CHANGE
    LMC_WEIGHT = 0.0         
    LEARNING_RATE = 1e-4

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
        
        # Enable efficient attention backend if available
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
        
        # Causal mask (prevent attention to future tokens)
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
# LMC COMPLEXITY CALCULATION
# ============================================================================

def calculate_lmc_from_weights(model):
    all_weights = []
    for param in model.parameters():
        all_weights.append(param.view(-1))
    
    weights = torch.cat(all_weights)
    
    weights_min = weights.min()
    weights_max = weights.max()
    normalized_weights = (weights - weights_min) / (weights_max - weights_min + 1e-10)
    
    n = len(weights)
    sample_size_iqr = min(10000, len(normalized_weights))
    
    with torch.no_grad():
        if len(normalized_weights) > sample_size_iqr:
            sample_indices = torch.randperm(len(normalized_weights), device=normalized_weights.device)[:sample_size_iqr]
            sample = normalized_weights[sample_indices]
        else:
            sample = normalized_weights
        
        q1 = torch.quantile(sample, 0.25)
        q3 = torch.quantile(sample, 0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            num_bins = max(1, int(np.ceil(float(np.sqrt(n)))))
        else:
            bin_width = float(2 * iqr.item() * (n ** (-1/3)))
            data_range = float((weights_max - weights_min).item())
            num_bins = max(1, int(np.ceil(data_range / bin_width)))
    
    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=normalized_weights.device)
    bin_width_tensor = bin_edges[1] - bin_edges[0]
    
    bin_indices_float = normalized_weights / bin_width_tensor
    bin_indices_floor = torch.floor(bin_indices_float)
    bin_indices_left = bin_indices_floor.long().clamp(0, num_bins - 1)
    bin_indices_right = (bin_indices_floor + 1).long().clamp(0, num_bins - 1)
    
    frac = bin_indices_float - bin_indices_floor
    weight_left = 1.0 - frac
    weight_right = frac
    
    hist = torch.zeros(num_bins, device=normalized_weights.device, dtype=normalized_weights.dtype)
    hist.scatter_add_(0, bin_indices_left, weight_left)
    hist.scatter_add_(0, bin_indices_right, weight_right)
    
    probs = hist / hist.sum()

    eps = 1e-10
    probs = torch.clamp(probs, eps, 1.0)
    probs = probs / probs.sum()  # Renormalize

    shannon_entropy = -(probs * torch.log(probs)).sum()
    
    N = len(probs)
    uniform_prob = 1.0 / N
    disequilibrium = ((probs - uniform_prob) ** 2).sum()
    
    lmc = shannon_entropy * disequilibrium
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    return lmc, lmc.item(), shannon_entropy.item(), disequilibrium.item(), num_bins, probs.detach(), bin_centers.detach()

# ============================================================================
# SLOPE NORMALIZATION
# ============================================================================

def normalize_slope_arctan(m):
    return math.atan(m) / (math.pi / 2)

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, config, vocab_size, lmc_start, ce_start, val_error_slope=0.0, lmc_weight=0.0):
    model.train()
    total_loss = 0.0
    total_lmc = 0.0
    total_combined_loss = 0.0
    total_samples = 0
    
    lmc_value = None
    
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
        
        if lmc_value is None or batch_idx % config.COMPLEXITY_UPDATE_INTERVAL == 0:
            lmc_tensor, lmc_scalar, _, _, _, _, _ = calculate_lmc_from_weights(model)
            lmc_value = lmc_tensor 
            lmc_value_scalar = lmc_scalar
 
        lmc_loss_normalized = (lmc_start / (lmc_value + 1e-10)) * ce_start
        
        ce_weight = 1.0 - lmc_weight
        lmc_weight_actual = lmc_weight
        
        if config.CONTROL_MODE == True:
            combined_loss = ce_loss
        else:
            combined_loss = ce_weight * ce_loss + lmc_weight_actual * lmc_loss_normalized
        
        combined_loss.backward()
        
        # Optimization step
        if config.MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += ce_loss.detach().item()
        total_lmc += lmc_value_scalar
        total_combined_loss += combined_loss.detach().item()
        total_samples += 1  # Count batches, not samples
    
        progress_bar.set_postfix({
            'loss': f'{total_loss / total_samples:.4f}',
            'lmc': f'{total_lmc / total_samples:.4f}',
            'lmc_w': f'{lmc_weight:.3f}',
            'ce_w': f'{ce_weight:.3f}',
            'combined': f'{combined_loss.detach().item():.4f}'
        })
    
    avg_loss = total_loss / total_samples
    avg_lmc = total_lmc / total_samples
    avg_combined = total_combined_loss / total_samples
    
    return avg_loss, avg_lmc, avg_combined

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

def plot_weight_distribution(probs, bin_centers, epoch, output_dir, run_num):
    dist_dir = os.path.join(output_dir, 'distributions')
    os.makedirs(dist_dir, exist_ok=True)
    
    probs_np = probs.cpu().numpy()
    bin_centers_np = bin_centers.cpu().numpy()
    
    epoch_str = f'{epoch:03d}' if isinstance(epoch, int) else epoch
    csv_path = os.path.join(dist_dir, f'distribution_epoch_{epoch_str}_run_{run_num}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Bin_Center', 'Probability'])
        for center, prob in zip(bin_centers_np, probs_np):
            writer.writerow([f'{center:.16f}', f'{prob:.16f}'])
    
    max_plot_bins = 1000
    if len(probs_np) > max_plot_bins:
        bin_factor = int(np.ceil(len(probs_np) / max_plot_bins))
        plot_probs = []
        plot_centers = []
        for i in range(0, len(probs_np), bin_factor):
            chunk_probs = probs_np[i:i+bin_factor]
            chunk_centers = bin_centers_np[i:i+bin_factor]
            plot_probs.append(chunk_probs.sum())
            plot_centers.append(chunk_centers.mean())
        plot_probs = np.array(plot_probs)
        plot_centers = np.array(plot_centers)
    else:
        plot_probs = probs_np
        plot_centers = bin_centers_np
    
    plt.figure(figsize=(10, 6))
    plt.bar(plot_centers, plot_probs, width=(plot_centers[1] - plot_centers[0]) * 0.8 if len(plot_centers) > 1 else 0.01,
            alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Normalized Weight Value', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title(f'Weight Distribution - Epoch {epoch} (Run {run_num})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    entropy = -(probs_np * np.log(probs_np + 1e-10)).sum()
    uniform_prob = 1.0 / len(probs_np)
    disequilibrium = ((probs_np - uniform_prob) ** 2).sum()
    lmc = entropy * disequilibrium
    
    stats_text = f'H (Entropy): {entropy:.4f}\nD (Disequilibrium): {disequilibrium:.4f}\nC (LMC): {lmc:.4f}\nBins (full): {len(probs_np)}\nBins (plot): {len(plot_probs)}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    epoch_str = f'{epoch:03d}' if isinstance(epoch, int) else epoch
    plot_path = os.path.join(dist_dir, f'distribution_epoch_{epoch_str}_run_{run_num}.png')
    plt.savefig(plot_path, dpi=100)
    plt.close()

def save_results_to_csv(output_dir, train_losses, val_losses, lmc_values, slope_values, lmc_weight_values,
                       test_losses_wiki_per_epoch, test_losses_shakespeare_per_epoch,
                       weights_entropy_values, weights_disequilibrium_values, 
                       num_bins_values, test_loss_wiki, test_metrics_wiki, 
                       test_loss_shakespeare, test_metrics_shakespeare, config, run_num=1):
    csv_filename = f'z_loss_test_results_transformers-{run_num}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['=== WikiText-2 Test Results ===', ''])
        writer.writerow(['Test Loss (WikiText-2)', f'{test_loss_wiki:.16f}'])
        writer.writerow(['Test LMC Complexity (WikiText-2)', f'{test_metrics_wiki[0]:.16f}'])
        writer.writerow(['Test Weights Entropy (WikiText-2)', f'{test_metrics_wiki[1]:.16f}'])
        writer.writerow(['Test Weights Disequilibrium (WikiText-2)', f'{test_metrics_wiki[2]:.16f}'])
        writer.writerow(['Test Weights Bins (WikiText-2)', f'{test_metrics_wiki[3]}'])
        writer.writerow([])
        writer.writerow(['=== Tiny-Shakespeare Test Results ===', ''])
        writer.writerow(['Test Loss (Tiny-Shakespeare)', f'{test_loss_shakespeare:.16f}'])
        writer.writerow(['Test LMC Complexity (Tiny-Shakespeare)', f'{test_metrics_shakespeare[0]:.16f}'])
        writer.writerow(['Test Weights Entropy (Tiny-Shakespeare)', f'{test_metrics_shakespeare[1]:.16f}'])
        writer.writerow(['Test Weights Disequilibrium (Tiny-Shakespeare)', f'{test_metrics_shakespeare[2]:.16f}'])
        writer.writerow(['Test Weights Bins (Tiny-Shakespeare)', f'{test_metrics_shakespeare[3]}'])
        writer.writerow([])
        writer.writerow(['=== Training Summary ===', ''])
        writer.writerow(['Final Training Loss', f'{train_losses[-1]:.16f}'])
        writer.writerow(['Final Validation Loss', f'{val_losses[-1]:.16f}'])
        writer.writerow(['Final Model LMC', f'{lmc_values[-1]:.16f}'])
        writer.writerow(['Final Val Error Slope', f'{slope_values[-1]:.16f}'])
        writer.writerow(['Final LMC Weight', f'{lmc_weight_values[-1]:.3f}'])
        writer.writerow(['Optimization Strategy', 'Slope-based: If d(Val)/dx < 0 → CE only (LMC_weight -= |slope_norm|), else CE*(1-LMC_weight) + LMC*LMC_weight (LMC_weight += |slope_norm|), where slope_norm = arctan(d(Val)/dx)/(π/2), clamped to [0,1]'])
        writer.writerow(['Run Number', f'{run_num}'])
        writer.writerow([])
        
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Test Loss Wiki', 'Test Loss Shakespeare',
                        'Model LMC', 'Val Error Slope', 'LMC Weight', 'Weights Entropy', 'Weights Disequilibrium', 'Num Bins'])
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                f'{train_losses[epoch]:.16f}',
                f'{val_losses[epoch]:.16f}',
                f'{test_losses_wiki_per_epoch[epoch]:.16f}',
                f'{test_losses_shakespeare_per_epoch[epoch]:.16f}',
                f'{lmc_values[epoch]:.16f}',
                f'{slope_values[epoch]:.16f}',
                f'{lmc_weight_values[epoch]:.3f}',
                f'{weights_entropy_values[epoch]:.16f}',
                f'{weights_disequilibrium_values[epoch]:.16f}',
                f'{num_bins_values[epoch]}'
            ])
    
    print(f"Results saved to '{csv_path}'")

def plot_results(output_dir, train_losses, val_losses, lmc_values, test_losses_wiki_per_epoch, test_losses_shakespeare_per_epoch, test_loss_wiki, test_loss_shakespeare, config, run_num=1):
    plot_filename = f'z_loss_training_losses_transformers-{run_num}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.close('all')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
    ax1.plot(epochs, val_losses, label='Validation Loss', marker='s', color='orange')
    ax1.plot(epochs, test_losses_wiki_per_epoch, label='Test Loss WikiText-2 (per epoch)', marker='^', color='red', alpha=0.7)
    ax1.plot(epochs, test_losses_shakespeare_per_epoch, label='Test Loss Tiny-Shakespeare (per epoch)', marker='v', color='purple', alpha=0.7)
    ax1.axhline(y=test_loss_wiki, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Final Test WikiText-2 ({test_loss_wiki:.4f})')
    ax1.axhline(y=test_loss_shakespeare, color='purple', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Final Test Shakespeare ({test_loss_shakespeare:.4f})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, lmc_values, label='Model LMC', marker='D', color='green', linewidth=2)
    ax2.set_ylabel('LMC Complexity (C = H × D)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    title = f'Transformer Model Training: Loss and LMC Complexity (Run {run_num})'
    plt.title(title, fontsize=14, pad=20)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close(fig)  # Close the figure after saving to free memory
    print(f"Plot saved to '{plot_path}'")

def aggregate_results_csv(output_dir, config):
    aggregate_data = {}
    test_data = {
        'test_loss_wiki': [],
        'test_lmc_wiki': [],
        'test_loss_shakespeare': [],
        'test_lmc_shakespeare': []
    }

    for run_num in range(1, config.NUM_OF_RUN_PER_CALL + 1):
        csv_filename = f'z_loss_test_results_transformers-{run_num}.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping run {run_num}")
            continue
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            for i, row in enumerate(rows):
                if len(row) >= 2:
                    if row[0] == 'Test Loss (WikiText-2)':
                        test_data['test_loss_wiki'].append(float(row[1]))
                    elif row[0] == 'Test LMC Complexity (WikiText-2)':
                        test_data['test_lmc_wiki'].append(float(row[1]))
                    elif row[0] == 'Test Loss (Tiny-Shakespeare)':
                        test_data['test_loss_shakespeare'].append(float(row[1]))
                    elif row[0] == 'Test LMC Complexity (Tiny-Shakespeare)':
                        test_data['test_lmc_shakespeare'].append(float(row[1]))
            
            epoch_start = None
            for i, row in enumerate(rows):
                if row and row[0] == 'Epoch':
                    epoch_start = i + 1
                    break
            
            if epoch_start is None:
                continue
            
            for i in range(epoch_start, len(rows)):
                row = rows[i]
                if not row or not row[0].isdigit():
                    continue
                
                epoch = int(row[0])
                if epoch not in aggregate_data:
                    aggregate_data[epoch] = {
                        'train_loss': [],
                        'val_loss': [],
                        'test_loss_wiki': [],
                        'test_loss_shakespeare': [],
                        'lmc': [],
                        'slope': [],
                        'lmc_weight': [],
                        'entropy': [],
                        'diseq': [],
                        'bins': []
                    }
                
                aggregate_data[epoch]['train_loss'].append(float(row[1]))
                aggregate_data[epoch]['val_loss'].append(float(row[2]))
                aggregate_data[epoch]['test_loss_wiki'].append(float(row[3]))
                aggregate_data[epoch]['test_loss_shakespeare'].append(float(row[4]))
                aggregate_data[epoch]['lmc'].append(float(row[5]))
                aggregate_data[epoch]['slope'].append(float(row[6]))
                aggregate_data[epoch]['lmc_weight'].append(float(row[7]))
                aggregate_data[epoch]['entropy'].append(float(row[8]))
                aggregate_data[epoch]['diseq'].append(float(row[9]))
                aggregate_data[epoch]['bins'].append(float(row[10]))
    
    aggregate_stats = {}
    for epoch in sorted(aggregate_data.keys()):
        stats_dict = {}
        for metric in ['train_loss', 'val_loss', 'test_loss_wiki', 'test_loss_shakespeare', 'lmc', 'slope', 'lmc_weight', 'entropy', 'diseq', 'bins']:
            values = np.array(aggregate_data[epoch][metric])
            stats_dict[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        aggregate_stats[epoch] = stats_dict
    
    test_stats = {}
    for key, values in test_data.items():
        if len(values) > 0:
            test_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    csv_filename = 'z_loss_test_results_transformers-AGGREGATE.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['AGGREGATE RESULTS FROM', f'{config.NUM_OF_RUN_PER_CALL} RUNS'])
        writer.writerow(['Optimization Strategy', 'Slope-based: If d(Val)/dx < 0 → CE only (LMC_weight -= |slope_norm|), else CE*(1-LMC_weight) + LMC*LMC_weight (LMC_weight += |slope_norm|), where slope_norm = arctan(d(Val)/dx)/(π/2), clamped to [0,1]'])
        writer.writerow([])
        
        writer.writerow(['=== WikiText-2 Test Results (Aggregated) ===', ''])
        if 'test_loss_wiki' in test_stats:
            writer.writerow(['Test Loss Mean', f"{test_stats['test_loss_wiki']['mean']:.16f}"])
            writer.writerow(['Test Loss Std', f"{test_stats['test_loss_wiki']['std']:.16f}"])
            writer.writerow(['Test Loss Min', f"{test_stats['test_loss_wiki']['min']:.16f}"])
            writer.writerow(['Test Loss Max', f"{test_stats['test_loss_wiki']['max']:.16f}"])
        if 'test_lmc_wiki' in test_stats:
            writer.writerow(['Test LMC Mean', f"{test_stats['test_lmc_wiki']['mean']:.16f}"])
            writer.writerow(['Test LMC Std', f"{test_stats['test_lmc_wiki']['std']:.16f}"])
        writer.writerow([])
        
        writer.writerow(['=== Tiny-Shakespeare Test Results (Aggregated) ===', ''])
        if 'test_loss_shakespeare' in test_stats:
            writer.writerow(['Test Loss Mean', f"{test_stats['test_loss_shakespeare']['mean']:.16f}"])
            writer.writerow(['Test Loss Std', f"{test_stats['test_loss_shakespeare']['std']:.16f}"])
            writer.writerow(['Test Loss Min', f"{test_stats['test_loss_shakespeare']['min']:.16f}"])
            writer.writerow(['Test Loss Max', f"{test_stats['test_loss_shakespeare']['max']:.16f}"])
        if 'test_lmc_shakespeare' in test_stats:
            writer.writerow(['Test LMC Mean', f"{test_stats['test_lmc_shakespeare']['mean']:.16f}"])
            writer.writerow(['Test LMC Std', f"{test_stats['test_lmc_shakespeare']['std']:.16f}"])
        writer.writerow([])
    
        header = ['Epoch', 
                 'Train_Loss_Mean', 'Train_Loss_Std', 'Train_Loss_Min', 'Train_Loss_Max',
                 'Val_Loss_Mean', 'Val_Loss_Std', 'Val_Loss_Min', 'Val_Loss_Max',
                 'Test_Wiki_Mean', 'Test_Wiki_Std', 'Test_Wiki_Min', 'Test_Wiki_Max',
                 'Test_Shakespeare_Mean', 'Test_Shakespeare_Std', 'Test_Shakespeare_Min', 'Test_Shakespeare_Max',
                 'LMC_Mean', 'LMC_Std', 'LMC_Min', 'LMC_Max',
                 'Slope_Mean', 'Slope_Std', 'Slope_Min', 'Slope_Max',
                 'LMC_Weight_Mean', 'LMC_Weight_Std', 'LMC_Weight_Min', 'LMC_Weight_Max',
                 'Entropy_Mean', 'Entropy_Std', 'Entropy_Min', 'Entropy_Max',
                 'Diseq_Mean', 'Diseq_Std', 'Diseq_Min', 'Diseq_Max',
                 'Bins_Mean', 'Bins_Std', 'Bins_Min', 'Bins_Max']
        writer.writerow(header)
        
        for epoch in sorted(aggregate_stats.keys()):
            row = [epoch]
            for metric in ['train_loss', 'val_loss', 'test_loss_wiki', 'test_loss_shakespeare', 'lmc', 'slope', 'lmc_weight', 'entropy', 'diseq', 'bins']:
                row.extend([
                    f'{aggregate_stats[epoch][metric]["mean"]:.16f}',
                    f'{aggregate_stats[epoch][metric]["std"]:.16f}',
                    f'{aggregate_stats[epoch][metric]["min"]:.16f}',
                    f'{aggregate_stats[epoch][metric]["max"]:.16f}'
                ])
            writer.writerow(row)
    
    print(f"Aggregate results saved to '{csv_path}'")
    return aggregate_stats, test_stats

def plot_aggregate_results(output_dir, config, aggregate_stats, test_stats):
    if not aggregate_stats:
        print("No aggregate stats to plot")
        return
    
    plot_filename = 'z_loss_training_losses_transformers-AGGREGATE.png'
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.close('all')
    
    epochs = sorted(aggregate_stats.keys())

    train_loss_mean = np.array([aggregate_stats[e]['train_loss']['mean'] for e in epochs])
    train_loss_std = np.array([aggregate_stats[e]['train_loss']['std'] for e in epochs])
    
    val_loss_mean = np.array([aggregate_stats[e]['val_loss']['mean'] for e in epochs])
    val_loss_std = np.array([aggregate_stats[e]['val_loss']['std'] for e in epochs])
    
    test_wiki_mean_per_epoch = np.array([aggregate_stats[e]['test_loss_wiki']['mean'] for e in epochs])
    test_wiki_std_per_epoch = np.array([aggregate_stats[e]['test_loss_wiki']['std'] for e in epochs])
    
    test_shakespeare_mean_per_epoch = np.array([aggregate_stats[e]['test_loss_shakespeare']['mean'] for e in epochs])
    test_shakespeare_std_per_epoch = np.array([aggregate_stats[e]['test_loss_shakespeare']['std'] for e in epochs])
    
    lmc_mean = np.array([aggregate_stats[e]['lmc']['mean'] for e in epochs])
    lmc_std = np.array([aggregate_stats[e]['lmc']['std'] for e in epochs])
    
    # 95% confidence interval (1.96 * std for normal distribution)
    ci_factor = 1.96
    train_loss_ci = ci_factor * train_loss_std
    val_loss_ci = ci_factor * val_loss_std
    test_wiki_ci_per_epoch = ci_factor * test_wiki_std_per_epoch
    test_shakespeare_ci_per_epoch = ci_factor * test_shakespeare_std_per_epoch
    lmc_ci = ci_factor * lmc_std
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.plot(epochs, train_loss_mean, label='Training Loss (Mean)', marker='o', color='blue', linewidth=2)
    ax1.fill_between(epochs, train_loss_mean - train_loss_ci, train_loss_mean + train_loss_ci, 
                     alpha=0.2, color='blue', label='Training Loss (95% CI)')
    
    ax1.plot(epochs, val_loss_mean, label='Validation Loss (Mean)', marker='s', color='orange', linewidth=2)
    ax1.fill_between(epochs, val_loss_mean - val_loss_ci, val_loss_mean + val_loss_ci, 
                     alpha=0.2, color='orange', label='Validation Loss (95% CI)')
    
    ax1.plot(epochs, test_wiki_mean_per_epoch, label='Test WikiText-2 (Mean, per epoch)', marker='^', color='red', linewidth=1.5, alpha=0.8)
    ax1.fill_between(epochs, test_wiki_mean_per_epoch - test_wiki_ci_per_epoch, test_wiki_mean_per_epoch + test_wiki_ci_per_epoch,
                     alpha=0.15, color='red')
    
    ax1.plot(epochs, test_shakespeare_mean_per_epoch, label='Test Tiny-Shakespeare (Mean, per epoch)', marker='v', color='purple', linewidth=1.5, alpha=0.8)
    ax1.fill_between(epochs, test_shakespeare_mean_per_epoch - test_shakespeare_ci_per_epoch, test_shakespeare_mean_per_epoch + test_shakespeare_ci_per_epoch,
                     alpha=0.15, color='purple')
    
    if 'test_loss_wiki' in test_stats:
        test_wiki_mean = test_stats['test_loss_wiki']['mean']
        test_wiki_std = test_stats['test_loss_wiki']['std']
        test_wiki_ci = ci_factor * test_wiki_std
        ax1.axhline(y=test_wiki_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Test Loss WikiText-2 ({test_wiki_mean:.4f}±{test_wiki_std:.4f})')
        ax1.axhspan(test_wiki_mean - test_wiki_ci, test_wiki_mean + test_wiki_ci, 
                   alpha=0.1, color='red')
    
    if 'test_loss_shakespeare' in test_stats:
        test_shakes_mean = test_stats['test_loss_shakespeare']['mean']
        test_shakes_std = test_stats['test_loss_shakespeare']['std']
        test_shakes_ci = ci_factor * test_shakes_std
        ax1.axhline(y=test_shakes_mean, color='purple', linestyle=':', linewidth=2,
                   label=f'Test Loss Tiny-Shakespeare ({test_shakes_mean:.4f}±{test_shakes_std:.4f})')
        ax1.axhspan(test_shakes_mean - test_shakes_ci, test_shakes_mean + test_shakes_ci, 
                   alpha=0.1, color='purple')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, lmc_mean, label='Model LMC (Mean)', marker='D', color='green', linewidth=2.5)
    ax2.fill_between(epochs, lmc_mean - lmc_ci, lmc_mean + lmc_ci, 
                     alpha=0.2, color='green', label='Model LMC (95% CI)')
    
    ax2.set_ylabel('LMC Complexity (C = H × D)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    title = f'Aggregate Results: {config.NUM_OF_RUN_PER_CALL} Runs (Slope-based Optimization)'
    plt.title(title, fontsize=14, pad=20, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"Aggregate plot saved to '{plot_path}'")

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def run_training_single(output_dir, config, run_num):
    device = initialize_device()
    
    enable_efficient_attention, attention_backend = check_efficient_attention()
    print()
    
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
    
    print("Loading datasets...")
    train_dataset = TextDataset(train_path, tokenizer, config.SEQ_LENGTH, config.MAX_SAMPLES)
    val_dataset = TextDataset(val_path, tokenizer, config.SEQ_LENGTH, None) 
    test_dataset_wiki = TextDataset(test_path_wiki, tokenizer, config.SEQ_LENGTH, None)
    test_dataset_shakespeare = TextDataset(test_path_shakespeare, tokenizer, config.SEQ_LENGTH, None) 
    
    # Print dataset token summary (only for first run)
    if run_num == 1:
        total_tokens = len(train_dataset.input_ids) + len(val_dataset.input_ids) + len(test_dataset_wiki.input_ids) + len(test_dataset_shakespeare.input_ids)
        print(f"\n{'='*70}")
        print(f"DATASET TOKEN SUMMARY")
        print(f"{'='*70}")
        print(f"Training tokens:            {len(train_dataset.input_ids):>20,}")
        print(f"Validation tokens:          {len(val_dataset.input_ids):>20,}")
        print(f"Test tokens (WikiText-2):   {len(test_dataset_wiki.input_ids):>20,}")
        print(f"Test tokens (Tiny-Shakes.): {len(test_dataset_shakespeare.input_ids):>20,}")
        print(f"{'─'*70}")
        print(f"Total tokens:               {total_tokens:>20,}")
        print(f"{'='*70}\n")

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    test_loader_wiki = DataLoader(
        test_dataset_wiki, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    test_loader_shakespeare = DataLoader(
        test_dataset_shakespeare, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if model.efficient_attention_enabled:
        print(f"Efficient attention enabled: {model.attention_backend}")
    else:
        print(f"(!) Using standard PyTorch attention")
    
    vocab_size = model.vocab_size
    
    if config.USE_COMPILE and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile (first epoch will be slower)...")
            model = torch.compile(model, mode='default')
            print("Model compiled successfully")
        except Exception as e:
            print(f"(!) Could not compile model: {e}")
    
    #ADAMW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # OneCycleLR scheduler
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95
    )
    
    print(f"Training on {device}")
    print(f"LMC weight: {config.LMC_WEIGHT:.16f} ({config.LMC_WEIGHT*100:.16f}% LMC, "
          f"{(1-config.LMC_WEIGHT)*100:.16f}% loss)\n")
    
    train_losses = []
    val_losses = []
    lmc_values = []
    slope_values = []
    lmc_weight_values = []
    test_losses_wiki = []
    test_losses_shakespeare = []
    weights_entropy_values = []
    weights_disequilibrium_values = []
    num_bins_values = []
    
    _, start_weights_lmc, weights_entropy, weights_diseq, num_bins, _, _ = calculate_lmc_from_weights(model)
    
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
        start_ce = sum(initial_ce_losses) / len(initial_ce_losses)
    print(f"Initial CE loss: {start_ce:.16f}")
    print(f"Initial LMC: {start_weights_lmc:.16f}")
    model.train()
    
    prev_val_loss = None
    current_slope = 0.0
    lmc_weight = 0.0
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        
        train_loss, train_lmc, train_combined = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, vocab_size, 
            lmc_start=start_weights_lmc, ce_start=start_ce, val_error_slope=current_slope, lmc_weight=lmc_weight
        )
        val_loss = validate(model, val_loader, device, vocab_size)
        
        # Calculate validation error slope d(Val)/dx and update lmc_weight
        if prev_val_loss is not None:
            raw_slope = val_loss - prev_val_loss
            current_slope = normalize_slope_arctan(raw_slope)

            slope_magnitude = abs(current_slope)
            if current_slope < 0:
                # Validation loss decreased - decrement lmc_weight by slope magnitude
                lmc_weight = max(0.0, lmc_weight - slope_magnitude)
                print(f"Val loss decreased ({prev_val_loss:.4f} → {val_loss:.4f}): norm_slope={current_slope:.6f}, LMC_weight={lmc_weight:.3f} (decreased by {slope_magnitude:.3f})")
            else:
                # Validation loss increased/flat - increment lmc_weight by slope magnitude
                lmc_weight = min(1.0, lmc_weight + slope_magnitude)
                print(f"Val loss increased/flat ({prev_val_loss:.4f} → {val_loss:.4f}): norm_slope={current_slope:.6f}, LMC_weight={lmc_weight:.3f} (increased by {slope_magnitude:.3f})")
        else:
            current_slope = 0.0
            lmc_weight = 0.0
            print(f"First epoch: slope=0.0, LMC_weight=0.0 (no previous val loss)")
        
        prev_val_loss = val_loss
        
        # Test on both datasets after each epoch
        test_loss_wiki_epoch = test(model, test_loader_wiki, device, vocab_size)
        print(f"  Test Loss (WikiText-2): {test_loss_wiki_epoch:.16f}")
    
        test_loss_shakespeare_epoch = test(model, test_loader_shakespeare, device, vocab_size)
        print(f"  Test Loss (Tiny-Shakespeare): {test_loss_shakespeare_epoch:.16f}")
        
        _, weights_lmc, weights_entropy, weights_diseq, num_bins, probs, bin_centers = calculate_lmc_from_weights(model)
    
        plot_weight_distribution(probs, bin_centers, epoch + 1, output_dir, run_num)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lmc_values.append(weights_lmc)
        slope_values.append(current_slope)
        lmc_weight_values.append(lmc_weight)
        test_losses_wiki.append(test_loss_wiki_epoch)
        test_losses_shakespeare.append(test_loss_shakespeare_epoch)
        weights_entropy_values.append(weights_entropy)
        weights_disequilibrium_values.append(weights_diseq)
        num_bins_values.append(num_bins)
        
        print(f"Training Loss: {train_loss:.16f}, Combined: {train_combined:.16f}")
        print(f"Validation Loss: {val_loss:.16f}")
        print(f"Model LMC (per-epoch weights): {weights_lmc:.16f}")
        print(f"Val Error Slope: {current_slope:.16f}")
    
    # Test on both datasets
    print("\nEvaluating on test sets...")
    print("Testing on WikiText-2...")
    test_loss_wiki = test(model, test_loader_wiki, device, vocab_size)
    _, test_lmc_wiki, test_entropy_wiki, test_diseq_wiki, test_bins_wiki, test_probs_wiki, test_centers_wiki = calculate_lmc_from_weights(model)
    test_metrics_wiki = (test_lmc_wiki, test_entropy_wiki, test_diseq_wiki, test_bins_wiki)
    print(f"Test Loss (WikiText-2): {test_loss_wiki:.16f}")
    
    print("\nTesting on Tiny-Shakespeare...")
    test_loss_shakespeare = test(model, test_loader_shakespeare, device, vocab_size)
    _, test_lmc_shakespeare, test_entropy_shakespeare, test_diseq_shakespeare, test_bins_shakespeare, test_probs_shakespeare, test_centers_shakespeare = calculate_lmc_from_weights(model)
    test_metrics_shakespeare = (test_lmc_shakespeare, test_entropy_shakespeare, test_diseq_shakespeare, test_bins_shakespeare)
    print(f"Test Loss (Tiny-Shakespeare): {test_loss_shakespeare:.16f}")
    
    plot_weight_distribution(test_probs_wiki, test_centers_wiki, 'final_test_wiki', output_dir, run_num)
    plot_weight_distribution(test_probs_shakespeare, test_centers_shakespeare, 'final_test_shakespeare', output_dir, run_num)

    save_results_to_csv(
        output_dir, train_losses, val_losses, lmc_values, slope_values, lmc_weight_values,
        test_losses_wiki, test_losses_shakespeare,
        weights_entropy_values, weights_disequilibrium_values, num_bins_values,
        test_loss_wiki, test_metrics_wiki, test_loss_shakespeare, test_metrics_shakespeare, config, run_num
    )
    
    plot_results(output_dir, train_losses, val_losses, lmc_values, test_losses_wiki, test_losses_shakespeare, test_loss_wiki, test_loss_shakespeare, config, run_num)

def run_training(output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}\n")
    
    # Run training NUM_OF_RUN_PER_CALL times
    for run_num in range(1, config.NUM_OF_RUN_PER_CALL + 1):
        print(f"\n{'='*80}")
        print(f"Run {run_num}/{config.NUM_OF_RUN_PER_CALL}")
        print(f"{'='*80}\n")
        
        # Set seed for reproducibility per run
        seed = 42 + run_num
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        run_training_single(output_dir, config, run_num)
    
    if config.NUM_OF_RUN_PER_CALL > 1:
        print(f"\n{'='*80}")
        print(f"AGGREGATING RESULTS FROM {config.NUM_OF_RUN_PER_CALL} RUNS")
        print(f"{'='*80}\n")
        
        aggregate_stats, test_stats = aggregate_results_csv(output_dir, config)
        plot_aggregate_results(output_dir, config, aggregate_stats, test_stats)
        
# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = Config()
    mode = ""

    if config.CONTROL_MODE == True:
        mode = "control (ce only)"
        output_dir = os.path.join(script_dir, 'output/output_0.0')
    else:
        mode = "experimental (lmc + ce)"
        output_dir = os.path.join(script_dir, 'output/output_1.0')
        
    print(f"{'='*80}")
    print(f"Transformer LMC Training - Mode: {mode.upper()}")
    print(f"Runs per configuration: {config.NUM_OF_RUN_PER_CALL}")
    print(f"{'='*80}\n")
    
    run_training(output_dir, config)

if __name__ == '__main__':
    main()
