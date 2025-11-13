import os
import csv
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
    HIDDEN_DIM = 768
    NUM_LAYERS = 8
    NUM_ATTENTION_HEADS = 12 # Standard ratio (hidden_dim / num_heads = 64)
    
    # Training hyperparameters
    BATCH_SIZE = 32 
    GRADIENT_ACCUMULATION_STEPS = 16 
    EPOCHS = 50
    LEARNING_RATE = 3e-4
    SEQ_LENGTH = 512
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    MAX_SAMPLES = None
    
    # LMC Complexity weight sweep configuration
    LMC_WEIGHT_START = 0.0   # Starting value
    LMC_WEIGHT_END = 1.0     # Ending value (inclusive)
    LMC_WEIGHT_STEP = 1.0   # Step size (e.g., 0.01 gives 0.0, 0.01, 0.02, ..., 1.0)
    
    # Number of runs per configuration call
    NUM_OF_RUN_PER_CALL = 10
    
    # LMC weight sampling configuration
    LMC_SAMPLE_SIZE = 0
    
    # Device configuration
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 8  # DataLoader workers
    
    # Debug configuration
    DEBUG = False  # Enable detailed debug output
    
    LMC_WEIGHT = 0.0         # DONT CHANGE


# ============================================================================
# DEVICE INITIALIZATION
# ============================================================================

def find_free_port():
    """Find a free port for distributed training"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed(rank, world_size, master_port):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training environment"""
    dist.destroy_process_group()


def initialize_device(rank=None, world_size=None):
    """Initialize device(s) for training"""
    if world_size is not None and world_size > 1:
        # Multi-GPU mode
        device = torch.device(f'cuda:{rank}')
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"MULTI-GPU TRAINING MODE")
            print(f"{'='*70}")
            print(f"Number of GPUs: {world_size}")
            for i in range(world_size):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"{'='*70}\n")
    else:
        # Single GPU or CPU mode
        device = Config.DEVICE
        print(f"Using device: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            torch.cuda.set_device(0)
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

def calculate_lmc_from_weights(model, sample_size=0, debug=False):
    if debug:
        print(f"[DEBUG LMC] Starting LMC calculation, sample_size={sample_size}")
    
    # Handle DDP wrapper
    if isinstance(model, DDP):
        model_params = model.module.parameters()
    else:
        model_params = model.parameters()
    
    # Collect ALL weights from the model
    all_weights = []
    for param in model_params:
        all_weights.append(param.data.view(-1))
    
    if debug:
        print(f"[DEBUG LMC] Collected {len(all_weights)} parameter tensors")
    
    # Flatten all weights into a single tensor
    weights = torch.cat(all_weights)
    
    # Apply sampling if sample_size > 0
    if sample_size > 0 and len(weights) > sample_size:
        # Create random indices on the same device as weights to avoid CPU-GPU deadlock
        sample_indices = torch.randperm(len(weights), device=weights.device)[:sample_size]
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

def train_epoch(model, train_loader, optimizer, scheduler, device, config, scaler, rank=0, world_size=1):
    model.train()
    total_loss = 0.0
    total_lmc = 0.0
    total_combined_loss = 0.0
    batch_count = 0
    
    if config.DEBUG and rank == 0:
        print("[DEBUG] Starting train_epoch")
        print(f"[DEBUG] Device: {device}")
        print(f"[DEBUG] Model is DDP: {isinstance(model, DDP)}")
    
    # Calculate loss/LMC weights (must sum to 1.0)
    loss_weight = 1.0 - config.LMC_WEIGHT
    lmc_weight = config.LMC_WEIGHT
    
    # Only show progress bar on rank 0
    progress_bar = tqdm(train_loader, desc="Training", disable=(config.DEBUG or rank != 0))
    
    for batch_idx, batch in enumerate(progress_bar):
        if config.DEBUG and rank == 0:
            print(f"\n[DEBUG] Batch {batch_idx + 1}")
            print(f"[DEBUG] Step 1: Moving data to device {device}")
        
        # Use the explicit device parameter
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        if config.DEBUG and rank == 0:
            print(f"[DEBUG] Step 2: Data moved. Shapes - input_ids: {input_ids.shape}, labels: {labels.shape}")
            print(f"[DEBUG] Step 3: Starting forward pass with mixed precision")
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            if config.DEBUG and rank == 0:
                print(f"[DEBUG] Step 3a: Calling model forward...")
            logits = model(input_ids, attention_mask=attention_mask)
            if config.DEBUG and rank == 0:
                print(f"[DEBUG] Step 3b: Model forward complete. Logits shape: {logits.shape}")
            
            # Get vocab_size from the model (handle DDP wrapper)
            if isinstance(model, DDP):
                vocab_size = model.module.vocab_size
            else:
                vocab_size = model.vocab_size
            
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            if config.DEBUG and rank == 0:
                print(f"[DEBUG] Step 3c: Computing CE loss...")
            ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
            if config.DEBUG and rank == 0:
                print(f"[DEBUG] Step 3d: CE loss computed: {ce_loss.item():.16f}")
        
        if config.DEBUG and rank == 0:
            print(f"[DEBUG] Step 4: Calculating LMC from weights...")
        
        # Only calculate LMC every 100 batches to save memory and time
        # For LMC_WEIGHT=0, we don't need it at all during training
        if lmc_weight > 0 and batch_idx % 100 == 0:
            lmc_value, _, _, _ = calculate_lmc_from_weights(model, sample_size=config.LMC_SAMPLE_SIZE, debug=config.DEBUG)
            if config.DEBUG and rank == 0:
                print(f"[DEBUG] Step 4a: LMC calculated: {lmc_value:.16f}")
        else:
            # Use last calculated LMC value (or default for first batches)
            if not hasattr(train_epoch, 'last_lmc'):
                train_epoch.last_lmc = 0.001  # Default initial value
            lmc_value = train_epoch.last_lmc
        
        # Store for reuse
        if lmc_weight > 0 and batch_idx % 100 == 0:
            train_epoch.last_lmc = lmc_value
        
        lmc_tensor = torch.tensor(lmc_value, dtype=torch.float32, device=device)
        
        # Combined objective using different formulations based on lmc_weight
        if lmc_weight == 0:
            combined_loss = ce_loss
        elif lmc_weight == 1:
            combined_loss = ce_loss / lmc_tensor
        
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
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            progress_bar.set_postfix({
                'loss': f'{total_loss / batch_count:.16f}',
                'lmc': f'{total_lmc / batch_count:.16f}',
                'combined': f'{total_combined_loss / batch_count:.16f}'
            })
    
    # Synchronize metrics across all GPUs
    if world_size > 1:
        # Convert to tensors for all_reduce
        loss_tensor = torch.tensor(total_loss, device=device)
        lmc_tensor = torch.tensor(total_lmc, device=device)
        combined_tensor = torch.tensor(total_combined_loss, device=device)
        count_tensor = torch.tensor(batch_count, device=device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(lmc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(combined_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = loss_tensor.item()
        total_lmc = lmc_tensor.item()
        total_combined_loss = combined_tensor.item()
        batch_count = int(count_tensor.item())
    
    avg_loss = total_loss / batch_count
    avg_lmc = total_lmc / batch_count
    avg_combined = total_combined_loss / batch_count
    
    return avg_loss, avg_lmc, avg_combined


def validate(model, val_loader, device, rank=0, world_size=1):
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", disable=(rank != 0))
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            
            # Get vocab_size from the model (handle DDP wrapper)
            if isinstance(model, DDP):
                vocab_size = model.module.vocab_size
            else:
                vocab_size = model.vocab_size
            
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
            
            total_loss += loss.item()
            batch_count += 1
    
    # Synchronize metrics across all GPUs
    if world_size > 1:
        loss_tensor = torch.tensor(total_loss, device=device)
        count_tensor = torch.tensor(batch_count, device=device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = loss_tensor.item()
        batch_count = int(count_tensor.item())
    
    return total_loss / batch_count


def test(model, test_loader, device, rank=0, world_size=1):
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", disable=(rank != 0))
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            
            # Get vocab_size from the model (handle DDP wrapper)
            if isinstance(model, DDP):
                vocab_size = model.module.vocab_size
            else:
                vocab_size = model.vocab_size
            
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
            
            total_loss += loss.item()
            batch_count += 1
    
    # Synchronize metrics across all GPUs
    if world_size > 1:
        loss_tensor = torch.tensor(total_loss, device=device)
        count_tensor = torch.tensor(batch_count, device=device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = loss_tensor.item()
        batch_count = int(count_tensor.item())
    
    return total_loss / batch_count


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
    ax1.axhline(y=test_loss, color='red', linestyle='--', label=f'Test Loss ({test_loss:.16f})')
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
    title = f'Aggregate Results: {config.NUM_OF_RUN_PER_CALL} Runs (LMC_WEIGHT={config.LMC_WEIGHT:.16f})'
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

def run_training_single(output_dir, config, run_num, rank=0, world_size=1):
    # Only print initialization messages on rank 0
    if rank == 0:
        # Initialize device
        device = initialize_device(rank, world_size)
        
        enable_efficient_attention, attention_backend = check_efficient_attention()
        print()
        
        # Initialize tokenizer
        print("Initializing RoBERTa tokenizer...")
    else:
        device = torch.device(f'cuda:{rank}')
        enable_efficient_attention, attention_backend = check_efficient_attention()
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.train.tokens')
    val_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.valid.tokens')
    test_path = os.path.join(script_dir, 'dataset/wikitext-2/wiki.test.tokens')
    
    if rank == 0:
        for path in [train_path, val_path, test_path]:
            if not os.path.exists(path):
                print(f"Error: Dataset file '{path}' not found!")
                return
        
        print("Loading datasets...")
    
    train_dataset = TextDataset(train_path, tokenizer, config.SEQ_LENGTH, config.MAX_SAMPLES)
    val_dataset = TextDataset(val_path, tokenizer, config.SEQ_LENGTH, config.MAX_SAMPLES)
    test_dataset = TextDataset(test_path, tokenizer, config.SEQ_LENGTH, config.MAX_SAMPLES)
    
    # Synchronize after dataset loading in multi-GPU mode
    if world_size > 1:
        dist.barrier()
    
    # Print dataset token summary (only for first run on rank 0)
    if run_num == 1 and rank == 0:
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
    
    # Create data loaders with DistributedSampler for multi-GPU
    if world_size > 1:
        # Use fewer workers in distributed mode to avoid deadlocks
        num_workers = min(2, config.NUM_WORKERS)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, sampler=val_sampler,
            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, sampler=test_sampler,
            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
        )
    else:
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
    
    # Synchronize all processes after dataloader creation
    if world_size > 1:
        dist.barrier()
    
    # Initialize model
    if rank == 0:
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
    
    # Wrap model with DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        # Synchronize all processes after DDP initialization
        dist.barrier()
    
    if config.DEBUG and rank == 0:
        print(f"[DEBUG] Model device: {next(model.parameters()).device}")
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        if enable_efficient_attention:
            print(f"✓ Efficient attention enabled: {attention_backend}")
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
    
    if rank == 0:
        print(f"Training on {device}")
        print(f"Total steps: {total_steps}, Warmup steps: {num_warmup_steps}")
        print(f"LMC weight: {config.LMC_WEIGHT:.16f} ({config.LMC_WEIGHT*100:.16f}% LMC, "
              f"{(1-config.LMC_WEIGHT)*100:.16f}% loss)\n")
    
    # Training loop
    train_losses = []
    val_losses = []
    lmc_values = []
    weights_entropy_values = []
    weights_disequilibrium_values = []
    num_bins_values = []
    
    for epoch in range(config.EPOCHS):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        
        # Set epoch for DistributedSampler (ensures different shuffling each epoch)
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_lmc, train_combined = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, scaler, rank, world_size
        )
        val_loss = validate(model, val_loader, device, rank, world_size)
        
        # Calculate detailed weight statistics for CSV and plotting (only on rank 0)
        if rank == 0:
            weights_lmc, weights_entropy, weights_diseq, num_bins = calculate_lmc_from_weights(model, sample_size=config.LMC_SAMPLE_SIZE)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            lmc_values.append(weights_lmc)  # Use per-epoch weight LMC instead of batch avg
            weights_entropy_values.append(weights_entropy)
            weights_disequilibrium_values.append(weights_diseq)
            num_bins_values.append(num_bins)
            
            print(f"Training Loss: {train_loss:.16f}, Combined: {train_combined:.16f}")
            print(f"Validation Loss: {val_loss:.16f}")
            print(f"Model LMC (per-epoch weights): {weights_lmc:.16f}")
    
    # Test (only on rank 0 for saving results)
    if rank == 0:
        print("\nEvaluating on test set...")
    
    test_loss = test(model, test_loader, device, rank, world_size)
    
    if rank == 0:
        test_metrics = calculate_lmc_from_weights(model, sample_size=config.LMC_SAMPLE_SIZE)
        print(f"Test Loss: {test_loss:.16f}")
        print(f"Test LMC: {test_metrics[0]:.16f} (bins={test_metrics[3]})\n")
        
        # Save results with run number
        save_results_to_csv(
            output_dir, train_losses, val_losses, lmc_values,
            weights_entropy_values, weights_disequilibrium_values, num_bins_values,
            test_loss, test_metrics, config, run_num
        )
        plot_results(output_dir, train_losses, val_losses, lmc_values, test_loss, config, run_num)


def run_training_distributed(rank, world_size, master_port, output_dir, config, run_num):
    """Distributed training wrapper for a single run"""
    setup_distributed(rank, world_size, master_port)
    
    try:
        run_training_single(output_dir, config, run_num, rank, world_size)
    finally:
        cleanup_distributed()


def run_training(output_dir, config):
    """Run training NUM_OF_RUN_PER_CALL times"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}\n")
    
    # Detect number of GPUs
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Run training NUM_OF_RUN_PER_CALL times
    for run_num in range(1, config.NUM_OF_RUN_PER_CALL + 1):
        print(f"\n{'='*80}")
        print(f"Run {run_num}/{config.NUM_OF_RUN_PER_CALL}")
        print(f"{'='*80}\n")
        
        # Set seed for reproducibility per run (different seed each run)
        seed = 42 + run_num
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if world_size > 1:
            # Find a free port for this run
            master_port = find_free_port()
            
            # Multi-GPU training with DistributedDataParallel
            mp.spawn(
                run_training_distributed,
                args=(world_size, master_port, output_dir, config, run_num),
                nprocs=world_size,
                join=True
            )
        else:
            # Single GPU or CPU training
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
    print(f"START: {config.LMC_WEIGHT_START:.16f}")
    print(f"END:   {config.LMC_WEIGHT_END:.16f}")
    print(f"STEP:  {config.LMC_WEIGHT_STEP:.16f}")
    print(f"Total configurations: {len(lmc_weights)}")
    print(f"Weights: {[f'{w:.16f}' for w in lmc_weights]}")
    print(f"{'='*80}\n")
    
    # Iterate through LMC weight values
    for lmc_weight in lmc_weights:
        output_dir = os.path.join(script_dir, f'output/output_LMC_{lmc_weight:.16f}')
        
        if os.path.exists(output_dir):
            print(f"Skipping LMC_WEIGHT={lmc_weight:.16f}, output already exists.\n")
            continue
        
        # Create config with current LMC weight
        config = Config()
        config.LMC_WEIGHT = lmc_weight
        
        print(f"\n{'='*80}")
        print(f"Starting training with LMC_WEIGHT = {lmc_weight:.16f}")
        print(f"{'='*80}\n")
        
        run_training(output_dir, config)


if __name__ == '__main__':
    main()
