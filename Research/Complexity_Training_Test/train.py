import os
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model hyperparameters
    HIDDEN_DIM = 100
    NUM_LAYERS = 2
    NUM_ATTENTION_HEADS = 4
    
    # Training hyperparameters
    BATCH_SIZE = 512 
    GRADIENT_ACCUMULATION_STEPS = 2
    EPOCHS = 20
    LEARNING_RATE = 3e-4
    SEQ_LENGTH = 4
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    MAX_SAMPLES = 1000
    
    # LMC Complexity weight sweep configuration
    LMC_WEIGHT_START = 0.0   # Starting value
    LMC_WEIGHT_END = 37.0     # Ending value (inclusive)
    LMC_WEIGHT_STEP = 1.0   # Step size (e.g., 0.01 gives 0.0, 0.01, 0.02, ..., 1.0)
    
    LMC_WEIGHT = 0.0         # DONT CHANGE
    
    # Number of runs per configuration call
    NUM_OF_RUN_PER_CALL = 2
    
    # LMC weight sampling configuration
    # Number of weights to sample for LMC calculation (0 = use all weights)
    LMC_SAMPLE_SIZE = 0
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 6  # DataLoader workers


# ============================================================================
# DEVICE INITIALIZATION
# ============================================================================

def initialize_device():
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: CUDA is not available! Using CPU instead.")
    
    print()
    return device


def check_efficient_attention():
    """Check for efficient attention libraries (xFormers/flash_attn)"""
    if not torch.cuda.is_available():
        return False, "pytorch"
    
    try:
        import xformers
        print("✓ xFormers available: enabling efficient attention (2–4× speedup)")
        return True, "xformers"
    except ImportError:
        try:
            import flash_attn
            print("✓ flash_attn available: enabling efficient attention (2–4× speedup)")
            return True, "flash_attn"
        except ImportError:
            print("ℹ xFormers/flash_attn not available. Using standard PyTorch attention.")
            print("  To install xFormers: pip install xformers")
            print("  To install flash_attn: pip install flash-attn (requires build)")
            return False, "pytorch"


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length, max_samples=None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        # Read and tokenize text
        print(f"Tokenizing {file_path}...")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
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
        
        # Only limit if max_samples is explicitly set
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
        
        # Pad if necessary
        if len(input_ids) < self.seq_length:
            padding = self.seq_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(padding, dtype=torch.long)])
            target_ids = torch.cat([target_ids, torch.full((padding,), -100, dtype=torch.long)])
        
        return {'input_ids': input_ids, 'labels': target_ids}


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attention_mask = (input_ids == 0)  # True for padding tokens
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


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
            dropout=0.3,
            activation='gelu'
        )
        
        # Enable efficient attention backend if available
        if enable_efficient_attention and attention_backend in ["xformers", "flash_attn"]:
            try:
                encoder_layer.self_attn = nn.MultiheadAttention(
                    hidden_dim, num_attention_heads, dropout=0.1, batch_first=True
                )
                # xFormers/flash_attn will optimize this automatically during forward pass
            except Exception as e:
                print(f"Warning: Could not enable {attention_backend} attention: {e}")
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.efficient_attention_enabled = enable_efficient_attention
        self.attention_backend = attention_backend
    
    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        embeddings = self.embedding(input_ids) + self.position_embedding(positions)
        
        # Causal mask (prevent attention to future tokens)
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=input_ids.device), 
            diagonal=1
        ).bool()
        
        # Transformer
        transformer_out = self.transformer(
            embeddings,
            mask=causal_mask,
            src_key_padding_mask=attention_mask
        )
        
        # Language model head
        logits = self.lm_head(transformer_out)
        return logits


# ============================================================================
# LMC COMPLEXITY CALCULATION
# ============================================================================

def calculate_lmc_from_weights(model, sample_size=0):
    # Collect ALL weights from the model
    all_weights = []
    for param in model.parameters():
        all_weights.append(param.data.view(-1))
    
    # Flatten all weights into a single tensor
    weights = torch.cat(all_weights)
    
    # Apply sampling if sample_size > 0
    if sample_size > 0 and len(weights) > sample_size:
        sample_indices = torch.randperm(len(weights))[:sample_size]
        weights = weights[sample_indices]
    
    # Move to CPU for histogram calculation (if on GPU)
    if weights.is_cuda:
        weights = weights.cpu()
    
    # Normalize weights to [0, 1] range
    weights_min = weights.min()
    weights_max = weights.max()
    normalized_weights = (weights - weights_min) / (weights_max - weights_min + 1e-10)
    
    # Calculate number of bins using Freedman-Diaconis rule
    n = len(weights)
    
    # Sample 10,000 random values to calculate IQR (memory efficient)
    sample_size_iqr = min(10000, len(normalized_weights))
    if len(normalized_weights) > sample_size_iqr:
        sample_indices = torch.randperm(len(normalized_weights))[:sample_size_iqr]
        sample = normalized_weights[sample_indices]
    else:
        sample = normalized_weights
    
    q1 = torch.quantile(sample, 0.25)
    q3 = torch.quantile(sample, 0.75)
    iqr = q3 - q1
    
    if iqr == 0:
        num_bins = max(1, int(np.ceil(float(np.sqrt(n)))))
    else:
        bin_width = float(2 * iqr * (n ** (-1/3)))
        data_range = float(weights_max - weights_min)
        num_bins = max(1, int(np.ceil(data_range / bin_width)))
    
    # Create histogram
    hist, _ = torch.histogram(normalized_weights, bins=num_bins, range=(0.0, 1.0))
    
    # Convert to probability distribution
    probs = hist.float() / hist.sum().float()
    
    # Ensure valid probabilities (clip to avoid log(0))
    eps = 1e-10
    probs = torch.clamp(probs, eps, 1.0)
    probs = probs / probs.sum()  # Renormalize
    
    # Calculate Shannon entropy: H = -sum(p_i * log(p_i))
    shannon_entropy = -(probs * torch.log(probs)).sum()
    
    # Calculate disequilibrium: D = sum((p_i - 1/N)^2)
    N = len(probs)
    uniform_prob = 1.0 / N
    disequilibrium = ((probs - uniform_prob) ** 2).sum()
    
    # LMC complexity: C = H × D
    lmc = shannon_entropy * disequilibrium
    
    return lmc.item(), shannon_entropy.item(), disequilibrium.item(), num_bins


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, config, scaler):
    model.train()
    total_loss = 0.0
    total_lmc = 0.0
    total_combined_loss = 0.0
    batch_count = 0
    
    # Calculate loss/LMC weights (must sum to 1.0)
    loss_weight = 1.0 - config.LMC_WEIGHT
    lmc_weight = config.LMC_WEIGHT
    
    # Get primary device (handles both single and multi-GPU)
    primary_device = next(model.parameters()).device
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(primary_device)
        labels = batch['labels'].to(primary_device)
        attention_mask = batch['attention_mask'].to(primary_device)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            logits = model(input_ids, attention_mask=attention_mask)
            # Handle both DataParallel and regular models
            vocab_size = model.module.vocab_size if hasattr(model, 'module') else model.vocab_size
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
        
        # Calculate LMC complexity from model weights (per batch)
        lmc_value, _, _, _ = calculate_lmc_from_weights(model, sample_size=config.LMC_SAMPLE_SIZE)
        lmc_tensor = torch.tensor(lmc_value, dtype=torch.float32, device=device)
        
        # Combined objective using different formulations based on lmc_weight
        # Use lmc_weight to select which loss formulation to use:
        # 0: loss
        # 1: loss/2
        # 2: loss/lmc
        # 3: -log(loss)/log(lmc)
        # 4: log(loss)/lmc
        # 5: -loss/log(lmc)
        # 6: log(loss*lmc)
        # 7: log(loss/lmc)
        # 8: log(loss)-log(lmc)
        # 9: log(loss)+log(lmc)
        # 10: loss-log(lmc)
        # 11: loss+log(lmc)
        # 12: log(loss)-lmc
        # 13: log(loss)+lmc
        # 14: -log(loss)*log(lmc)
        # 15: -loss*log(lmc)
        # 16: log(loss)*lmc
        # 17: exp(loss)*lmc
        # 18: loss*exp(lmc)
        # 19: exp(loss)*exp(lmc)
        # 20: exp(loss+lmc)
        # 21: exp(loss-lmc)
        # 22: exp(loss)*exp(-lmc)
        # 23: -exp(-loss)*exp(lmc)
        # 24: loss^lmc
        # 25: lmc^loss
        # 26: loss^(1/lmc)
        # 27: lmc^(1/loss)
        # 28: -loss^(-lmc)
        # 29: lmc^(-loss)
        # 30: loss^(-1/lmc)
        # 31: -lmc^(-1/loss)
        # 32: loss * 0.9 - lmc * 0.1
        # 33: exp(loss)*lmc
        # 34: loss*exp(lmc)
        # 35: exp(loss*lmc)
        # 36: exp(loss/lmc)
        # 37: -exp(lmc/loss)
        
        eps = 1e-8
        
        if lmc_weight == 0:
            combined_loss = ce_loss
        elif lmc_weight == 1:
            combined_loss = ce_loss / 2
        elif lmc_weight == 2:
            combined_loss = ce_loss / (lmc_tensor + eps)
        elif lmc_weight == 3:
            combined_loss = -torch.log(ce_loss + eps) / (torch.log(lmc_tensor + eps) + eps)
        elif lmc_weight == 4:
            combined_loss = torch.log(ce_loss + eps) / (lmc_tensor + eps)
        elif lmc_weight == 5:
            combined_loss = -ce_loss / (torch.log(lmc_tensor + eps) + eps)
        elif lmc_weight == 6:
            combined_loss = torch.log(ce_loss * lmc_tensor + eps)
        elif lmc_weight == 7:
            combined_loss = torch.log((ce_loss / (lmc_tensor + eps)) + eps)
        elif lmc_weight == 8:
            combined_loss = torch.log(ce_loss + eps) - torch.log(lmc_tensor + eps)
        elif lmc_weight == 9:
            combined_loss = torch.log(ce_loss + eps) + torch.log(lmc_tensor + eps)
        elif lmc_weight == 10:
            combined_loss = ce_loss - torch.log(lmc_tensor + eps)
        elif lmc_weight == 11:
            combined_loss = ce_loss + torch.log(lmc_tensor + eps)
        elif lmc_weight == 12:
            combined_loss = torch.log(ce_loss + eps) - lmc_tensor
        elif lmc_weight == 13:
            combined_loss = torch.log(ce_loss + eps) + lmc_tensor
        elif lmc_weight == 14:
            combined_loss = -torch.log(ce_loss + eps) * torch.log(lmc_tensor + eps)
        elif lmc_weight == 15:
            combined_loss = -ce_loss * torch.log(lmc_tensor + eps)
        elif lmc_weight == 16:
            combined_loss = torch.log(ce_loss + eps) * lmc_tensor
        elif lmc_weight == 17:
            combined_loss = torch.exp(ce_loss) * lmc_tensor
        elif lmc_weight == 18:
            combined_loss = ce_loss * torch.exp(lmc_tensor)
        elif lmc_weight == 19:
            combined_loss = torch.exp(ce_loss) * torch.exp(lmc_tensor)
        elif lmc_weight == 20:
            combined_loss = torch.exp(ce_loss + lmc_tensor)
        elif lmc_weight == 21:
            combined_loss = torch.exp(ce_loss - lmc_tensor)
        elif lmc_weight == 22:
            combined_loss = torch.exp(ce_loss) * torch.exp(-lmc_tensor)
        elif lmc_weight == 23:
            combined_loss = -torch.exp(-ce_loss) * torch.exp(lmc_tensor)
        elif lmc_weight == 24:
            combined_loss = torch.pow(ce_loss + eps, lmc_tensor + eps)
        elif lmc_weight == 25:
            combined_loss = torch.pow(lmc_tensor + eps, ce_loss + eps)
        elif lmc_weight == 26:
            combined_loss = torch.pow(ce_loss + eps, 1.0 / (lmc_tensor + eps))
        elif lmc_weight == 27:
            combined_loss = torch.pow(lmc_tensor + eps, 1.0 / (ce_loss + eps))
        elif lmc_weight == 28:
            combined_loss = -torch.pow(ce_loss + eps, -(lmc_tensor + eps))
        elif lmc_weight == 29:
            combined_loss = torch.pow(lmc_tensor + eps, -(ce_loss + eps))
        elif lmc_weight == 30:
            combined_loss = torch.pow(ce_loss + eps, -1.0 / (lmc_tensor + eps))
        elif lmc_weight == 31:
            combined_loss = -torch.pow(lmc_tensor + eps, -1.0 / (ce_loss + eps))
        elif lmc_weight == 32:
            combined_loss = ce_loss * 0.9 - lmc_tensor * 0.1
        elif lmc_weight == 33:
            combined_loss = torch.exp(ce_loss) * lmc_tensor
        elif lmc_weight == 34:
            combined_loss = ce_loss * torch.exp(lmc_tensor)
        elif lmc_weight == 35:
            combined_loss = torch.exp(ce_loss * lmc_tensor)
        elif lmc_weight == 36:
            combined_loss = torch.exp(ce_loss / (lmc_tensor + eps))
        elif lmc_weight == 37:
            combined_loss = -torch.exp(lmc_tensor / (ce_loss + eps))
        
        combined_loss = combined_loss / config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        scaler.scale(combined_loss).backward()
        
        # Accumulate metrics
        total_loss += ce_loss.detach().item()
        total_lmc += lmc_value
        total_combined_loss += combined_loss.detach().item() * config.GRADIENT_ACCUMULATION_STEPS
        batch_count += 1
        
        # Optimization step
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{total_loss / batch_count:.4f}',
            'lmc': f'{total_lmc / batch_count:.8f}',
            'combined': f'{total_combined_loss / batch_count:.4f}'
        })
    
    avg_loss = total_loss / batch_count
    avg_lmc = total_lmc / batch_count
    avg_combined = total_combined_loss / batch_count
    
    return avg_loss, avg_lmc, avg_combined


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    
    # Get primary device (handles both single and multi-GPU)
    primary_device = next(model.parameters()).device
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(primary_device)
            labels = batch['labels'].to(primary_device)
            attention_mask = batch['attention_mask'].to(primary_device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            # Handle both DataParallel and regular models
            vocab_size = model.module.vocab_size if hasattr(model, 'module') else model.vocab_size
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def test(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    
    # Get primary device (handles both single and multi-GPU)
    primary_device = next(model.parameters()).device
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(primary_device)
            labels = batch['labels'].to(primary_device)
            attention_mask = batch['attention_mask'].to(primary_device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            # Handle both DataParallel and regular models
            vocab_size = model.module.vocab_size if hasattr(model, 'module') else model.vocab_size
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


# ============================================================================
# LOGGING AND VISUALIZATION
# ============================================================================

def save_results_to_csv(output_dir, train_losses, val_losses, lmc_values, 
                       weights_entropy_values, weights_disequilibrium_values, 
                       num_bins_values, test_loss, test_metrics, config, run_num=1):
    """Save training results to CSV file"""
    csv_filename = f'z_loss_test_results_transformers-{run_num}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Summary metrics
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Test Loss', f'{test_loss:.16f}'])
        writer.writerow(['Test LMC Complexity', f'{test_metrics[0]:.16f}'])
        writer.writerow(['Test Weights Entropy', f'{test_metrics[1]:.16f}'])
        writer.writerow(['Test Weights Disequilibrium', f'{test_metrics[2]:.16f}'])
        writer.writerow(['Test Weights Bins (Freedman-Diaconis)', f'{test_metrics[3]}'])
        writer.writerow(['Final Training Loss', f'{train_losses[-1]:.16f}'])
        writer.writerow(['Final Validation Loss', f'{val_losses[-1]:.16f}'])
        writer.writerow(['Final Model LMC', f'{lmc_values[-1]:.16f}'])
        writer.writerow(['LMC Weight', f'{config.LMC_WEIGHT:.16f}'])
        writer.writerow(['Loss Weight', f'{1.0 - config.LMC_WEIGHT:.16f}'])
        writer.writerow(['Run Number', f'{run_num}'])
        writer.writerow([])
        
        # Epoch-by-epoch data
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Model LMC', 
                        'Weights Entropy', 'Weights Disequilibrium', 'Num Bins'])
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                f'{train_losses[epoch]:.16f}',
                f'{val_losses[epoch]:.16f}',
                f'{lmc_values[epoch]:.16f}',
                f'{weights_entropy_values[epoch]:.16f}',
                f'{weights_disequilibrium_values[epoch]:.16f}',
                f'{num_bins_values[epoch]}'
            ])
    
    print(f"Results saved to '{csv_path}'")


def plot_results(output_dir, train_losses, val_losses, lmc_values, test_loss, config, run_num=1):
    plot_filename = f'z_loss_training_losses_transformers-{run_num}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    
    # Close any existing figures to prevent data carryover
    plt.close('all')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    # Primary y-axis: Losses
    ax1.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
    ax1.plot(epochs, val_losses, label='Validation Loss', marker='s', color='orange')
    ax1.axhline(y=test_loss, color='red', linestyle='--', label=f'Test Loss ({test_loss:.4f})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Secondary y-axis: LMC Complexity
    ax2 = ax1.twinx()
    ax2.plot(epochs, lmc_values, label='Model LMC', marker='D', color='green', linewidth=2)
    ax2.set_ylabel('LMC Complexity (C = H × D)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Title and legend
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
    """Aggregate results from all runs and create aggregate CSV"""
    aggregate_data = {}
    
    # Read all run results
    for run_num in range(1, config.NUM_OF_RUN_PER_CALL + 1):
        csv_filename = f'z_loss_test_results_transformers-{run_num}.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping run {run_num}")
            continue
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Find the epoch-by-epoch data section
            epoch_start = None
            for i, row in enumerate(rows):
                if row and row[0] == 'Epoch':
                    epoch_start = i + 1
                    break
            
            if epoch_start is None:
                continue
            
            # Parse epoch data
            for i in range(epoch_start, len(rows)):
                row = rows[i]
                if not row or not row[0].isdigit():
                    continue
                
                epoch = int(row[0])
                if epoch not in aggregate_data:
                    aggregate_data[epoch] = {
                        'train_loss': [],
                        'val_loss': [],
                        'lmc': [],
                        'entropy': [],
                        'diseq': [],
                        'bins': []
                    }
                
                aggregate_data[epoch]['train_loss'].append(float(row[1]))
                aggregate_data[epoch]['val_loss'].append(float(row[2]))
                aggregate_data[epoch]['lmc'].append(float(row[3]))
                aggregate_data[epoch]['entropy'].append(float(row[4]))
                aggregate_data[epoch]['diseq'].append(float(row[5]))
                aggregate_data[epoch]['bins'].append(float(row[6]))
    
    # Calculate statistics
    aggregate_stats = {}
    for epoch in sorted(aggregate_data.keys()):
        stats_dict = {}
        for metric in ['train_loss', 'val_loss', 'lmc', 'entropy', 'diseq', 'bins']:
            values = np.array(aggregate_data[epoch][metric])
            stats_dict[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        aggregate_stats[epoch] = stats_dict
    
    # Save aggregate CSV
    csv_filename = 'z_loss_test_results_transformers-AGGREGATE.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Summary
        writer.writerow(['AGGREGATE RESULTS FROM', f'{config.NUM_OF_RUN_PER_CALL} RUNS'])
        writer.writerow(['LMC Weight', f'{config.LMC_WEIGHT:.16f}'])
        writer.writerow([])
        
        # Epoch-by-epoch data with statistics
        header = ['Epoch', 
                 'Train_Loss_Mean', 'Train_Loss_Std', 'Train_Loss_Min', 'Train_Loss_Max',
                 'Val_Loss_Mean', 'Val_Loss_Std', 'Val_Loss_Min', 'Val_Loss_Max',
                 'LMC_Mean', 'LMC_Std', 'LMC_Min', 'LMC_Max',
                 'Entropy_Mean', 'Entropy_Std', 'Entropy_Min', 'Entropy_Max',
                 'Diseq_Mean', 'Diseq_Std', 'Diseq_Min', 'Diseq_Max',
                 'Bins_Mean', 'Bins_Std', 'Bins_Min', 'Bins_Max']
        writer.writerow(header)
        
        for epoch in sorted(aggregate_stats.keys()):
            row = [epoch]
            for metric in ['train_loss', 'val_loss', 'lmc', 'entropy', 'diseq', 'bins']:
                row.extend([
                    f'{aggregate_stats[epoch][metric]["mean"]:.16f}',
                    f'{aggregate_stats[epoch][metric]["std"]:.16f}',
                    f'{aggregate_stats[epoch][metric]["min"]:.16f}',
                    f'{aggregate_stats[epoch][metric]["max"]:.16f}'
                ])
            writer.writerow(row)
    
    print(f"Aggregate results saved to '{csv_path}'")
    return aggregate_stats


def plot_aggregate_results(output_dir, config, aggregate_stats):
    """Plot aggregate results with 95% confidence intervals"""
    if not aggregate_stats:
        print("No aggregate stats to plot")
        return
    
    plot_filename = 'z_loss_training_losses_transformers-AGGREGATE.png'
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.close('all')
    
    epochs = sorted(aggregate_stats.keys())
    
    # Extract statistics
    train_loss_mean = np.array([aggregate_stats[e]['train_loss']['mean'] for e in epochs])
    train_loss_std = np.array([aggregate_stats[e]['train_loss']['std'] for e in epochs])
    
    val_loss_mean = np.array([aggregate_stats[e]['val_loss']['mean'] for e in epochs])
    val_loss_std = np.array([aggregate_stats[e]['val_loss']['std'] for e in epochs])
    
    lmc_mean = np.array([aggregate_stats[e]['lmc']['mean'] for e in epochs])
    lmc_std = np.array([aggregate_stats[e]['lmc']['std'] for e in epochs])
    
    # 95% confidence interval (1.96 * std for normal distribution)
    ci_factor = 1.96
    train_loss_ci = ci_factor * train_loss_std
    val_loss_ci = ci_factor * val_loss_std
    lmc_ci = ci_factor * lmc_std
    
    # Create figure with dual axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Primary y-axis: Losses with confidence intervals
    ax1.plot(epochs, train_loss_mean, label='Training Loss (Mean)', marker='o', color='blue', linewidth=2)
    ax1.fill_between(epochs, train_loss_mean - train_loss_ci, train_loss_mean + train_loss_ci, 
                     alpha=0.2, color='blue', label='Training Loss (95% CI)')
    
    ax1.plot(epochs, val_loss_mean, label='Validation Loss (Mean)', marker='s', color='orange', linewidth=2)
    ax1.fill_between(epochs, val_loss_mean - val_loss_ci, val_loss_mean + val_loss_ci, 
                     alpha=0.2, color='orange', label='Validation Loss (95% CI)')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Secondary y-axis: LMC Complexity with confidence interval
    ax2 = ax1.twinx()
    ax2.plot(epochs, lmc_mean, label='Model LMC (Mean)', marker='D', color='green', linewidth=2.5)
    ax2.fill_between(epochs, lmc_mean - lmc_ci, lmc_mean + lmc_ci, 
                     alpha=0.2, color='green', label='Model LMC (95% CI)')
    
    ax2.set_ylabel('LMC Complexity (C = H × D)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Title and legend
    title = f'Aggregate Results: {config.NUM_OF_RUN_PER_CALL} Runs (LMC_WEIGHT={config.LMC_WEIGHT:.2f})'
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
    """Run a single training iteration"""
    device = initialize_device()
    enable_efficient_attention, attention_backend = check_efficient_attention()
    print()
    
    # Initialize tokenizer
    print("Initializing RoBERTa tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.train.tokens')
    val_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.valid.tokens')
    test_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.test.tokens')
    
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Error: Dataset file '{path}' not found!")
            return
    
    print("Loading datasets...")
    train_dataset = TextDataset(train_path, tokenizer, config.SEQ_LENGTH, config.MAX_SAMPLES)
    val_dataset = TextDataset(val_path, tokenizer, config.SEQ_LENGTH, config.MAX_SAMPLES)
    test_dataset = TextDataset(test_path, tokenizer, config.SEQ_LENGTH, config.MAX_SAMPLES)
    
    # Print dataset token summary (only for first run)
    if run_num == 1:
        total_tokens = len(train_dataset.input_ids) + len(val_dataset.input_ids) + len(test_dataset.input_ids)
        print(f"\n{'='*70}")
        print(f"DATASET TOKEN SUMMARY")
        print(f"{'='*70}")
        print(f"Training tokens:   {len(train_dataset.input_ids):>20,}")
        print(f"Validation tokens: {len(val_dataset.input_ids):>20,}")
        print(f"Test tokens:       {len(test_dataset.input_ids):>20,}")
        print(f"{'─'*70}")
        print(f"Total tokens:      {total_tokens:>20,}")
        print(f"{'='*70}\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
    )
    
    # Initialize model
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
    
    # Apply DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"\nUsing {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if hasattr(model, 'module'):
        print(f"Efficient attention enabled: {model.module.efficient_attention_enabled}")
    else:
        if model.efficient_attention_enabled:
            print(f"✓ Efficient attention enabled: {model.attention_backend}")
        else:
            print(f"ℹ Using standard PyTorch attention")
    
    # Optimizer and scheduler
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(config.WARMUP_RATIO * total_steps)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Training on {device}")
    print(f"Total steps: {total_steps}, Warmup steps: {num_warmup_steps}")
    print(f"LMC weight: {config.LMC_WEIGHT} ({config.LMC_WEIGHT*100:.0f}% LMC, "
          f"{(1-config.LMC_WEIGHT)*100:.0f}% loss)\n")
    
    # Training loop
    train_losses = []
    val_losses = []
    lmc_values = []
    weights_entropy_values = []
    weights_disequilibrium_values = []
    num_bins_values = []
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        
        train_loss, train_lmc, train_combined = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, scaler
        )
        val_loss = validate(model, val_loader, device)
        
        # Calculate detailed weight statistics for CSV and plotting
        weights_lmc, weights_entropy, weights_diseq, num_bins = calculate_lmc_from_weights(model, sample_size=config.LMC_SAMPLE_SIZE)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lmc_values.append(weights_lmc)  # Use per-epoch weight LMC instead of batch avg
        weights_entropy_values.append(weights_entropy)
        weights_disequilibrium_values.append(weights_diseq)
        num_bins_values.append(num_bins)
        
        print(f"Training Loss: {train_loss:.4f}, Combined: {train_combined:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Model LMC (per-epoch weights): {weights_lmc:.8f}")
    
    # Test
    print("\nEvaluating on test set...")
    test_loss = test(model, test_loader, device)
    test_metrics = calculate_lmc_from_weights(model, sample_size=config.LMC_SAMPLE_SIZE)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test LMC: {test_metrics[0]:.8f} (bins={test_metrics[3]})\n")
    
    # Save results with run number
    save_results_to_csv(
        output_dir, train_losses, val_losses, lmc_values,
        weights_entropy_values, weights_disequilibrium_values, num_bins_values,
        test_loss, test_metrics, config, run_num
    )
    plot_results(output_dir, train_losses, val_losses, lmc_values, test_loss, config, run_num)


def run_training(output_dir, config):
    """Run training NUM_OF_RUN_PER_CALL times"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}\n")
    
    # Run training NUM_OF_RUN_PER_CALL times
    for run_num in range(1, config.NUM_OF_RUN_PER_CALL + 1):
        print(f"\n{'='*80}")
        print(f"Run {run_num}/{config.NUM_OF_RUN_PER_CALL}")
        print(f"{'='*80}\n")
        
        # Set seed for reproducibility per run (different seed each run)
        seed = 42 + run_num
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        run_training_single(output_dir, config, run_num)
    
    # After all runs, aggregate results
    if config.NUM_OF_RUN_PER_CALL > 1:
        print(f"\n{'='*80}")
        print(f"AGGREGATING RESULTS FROM {config.NUM_OF_RUN_PER_CALL} RUNS")
        print(f"{'='*80}\n")
        
        aggregate_stats = aggregate_results_csv(output_dir, config)
        plot_aggregate_results(output_dir, config, aggregate_stats)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = Config()
    
    # Generate LMC weight sweep using START, END, and STEP
    num_steps = int(round((config.LMC_WEIGHT_END - config.LMC_WEIGHT_START) / config.LMC_WEIGHT_STEP)) + 1
    lmc_weights = np.linspace(config.LMC_WEIGHT_START, config.LMC_WEIGHT_END, num_steps)
    
    print(f"\n{'='*80}")
    print(f"LMC WEIGHT SWEEP CONFIGURATION")
    print(f"{'='*80}")
    print(f"START: {config.LMC_WEIGHT_START:.2f}")
    print(f"END:   {config.LMC_WEIGHT_END:.2f}")
    print(f"STEP:  {config.LMC_WEIGHT_STEP:.4f}")
    print(f"Total configurations: {len(lmc_weights)}")
    print(f"Weights: {[f'{w:.2f}' for w in lmc_weights]}")
    print(f"{'='*80}\n")
    
    # Iterate through LMC weight values
    for lmc_weight in lmc_weights:
        output_dir = os.path.join(script_dir, f'output/output_LMC_{lmc_weight:.2f}')
        
        if os.path.exists(output_dir):
            print(f"Skipping LMC_WEIGHT={lmc_weight:.2f}, output already exists.\n")
            continue
        
        # Create config with current LMC weight
        config = Config()
        config.LMC_WEIGHT = lmc_weight
        
        print(f"\n{'='*80}")
        print(f"Starting training with LMC_WEIGHT = {lmc_weight:.2f}")
        print(f"{'='*80}\n")
        
        run_training(output_dir, config)


if __name__ == '__main__':
    main()
