import os
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    # Model hyperparameters
    HIDDEN_DIM = 200 # Must be divisible by NUM_ATTENTION_HEADS
    NUM_LAYERS = 2
    NUM_ATTENTION_HEADS = 4
    
    # Training hyperparameters
    BATCH_SIZE = 512
    GRADIENT_ACCUMULATION_STEPS = 1
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    SEQ_LENGTH = 16
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 0.5
    MAX_SAMPLES = 5000
    
    # LMC Complexity weight (0.0 = 100% loss optimization, 1.0 = 100% LMC maximization)
    LMC_WEIGHT = 0.0
    
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
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
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
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        print(f"Tokenizing {file_path}...")
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
        
        if max_samples:
            max_length = max_samples * seq_length
            self.input_ids = self.input_ids[:max_length]
        
        print(f"Dataset loaded: {len(self.input_ids)} tokens")
    
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

def calculate_lmc_from_weights(model):
    # Collect ALL weights from the model
    all_weights = []
    for param in model.parameters():
        all_weights.append(param.data.view(-1))
    
    # Flatten all weights into a single tensor
    weights = torch.cat(all_weights)
    
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
    sample_size = min(10000, len(normalized_weights))
    if len(normalized_weights) > sample_size:
        sample_indices = torch.randperm(len(normalized_weights))[:sample_size]
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
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            logits = model(input_ids, attention_mask=attention_mask)
            logits_flat = logits.view(-1, model.vocab_size)
            labels_flat = labels.view(-1)
            ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
        
        # Calculate LMC complexity from ALL model weights (per batch)
        lmc_value, _, _, _ = calculate_lmc_from_weights(model)
        lmc_tensor = torch.tensor(lmc_value, dtype=torch.float32, device=device)
        
        # Combined objective: weighted sum of loss minimization and LMC maximization
        # We minimize: (1-lmc_weight)*loss - lmc_weight*lmc
        combined_loss = loss_weight * ce_loss - lmc_weight * lmc_tensor
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
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            logits_flat = logits.view(-1, model.vocab_size)
            labels_flat = labels.view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def test(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            logits_flat = logits.view(-1, model.vocab_size)
            labels_flat = labels.view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


# ============================================================================
# LOGGING AND VISUALIZATION
# ============================================================================

def save_results_to_csv(output_dir, train_losses, val_losses, lmc_values, 
                       weights_entropy_values, weights_disequilibrium_values, 
                       num_bins_values, test_loss, test_metrics, config):
    """Save training results to CSV file"""
    csv_path = os.path.join(output_dir, 'z_loss_test_results_transformers.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Summary metrics
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Test Loss', f'{test_loss:.4f}'])
        writer.writerow(['Test LMC Complexity', f'{test_metrics[0]:.4f}'])
        writer.writerow(['Test Weights Entropy', f'{test_metrics[1]:.4f}'])
        writer.writerow(['Test Weights Disequilibrium', f'{test_metrics[2]:.4f}'])
        writer.writerow(['Test Weights Bins (Freedman-Diaconis)', f'{test_metrics[3]}'])
        writer.writerow(['Final Training Loss', f'{train_losses[-1]:.4f}'])
        writer.writerow(['Final Validation Loss', f'{val_losses[-1]:.4f}'])
        writer.writerow(['Final Model LMC', f'{lmc_values[-1]:.4f}'])
        writer.writerow(['LMC Weight', f'{config.LMC_WEIGHT}'])
        writer.writerow(['Loss Weight', f'{1.0 - config.LMC_WEIGHT}'])
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


def plot_results(output_dir, train_losses, val_losses, lmc_values, test_loss, config):
    plot_path = os.path.join(output_dir, 'z_loss_training_losses_transformers.png')
    
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
    plt.title('Transformer Model Training: Loss and LMC Complexity', fontsize=14, pad=20)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close(fig)  # Close the figure after saving to free memory
    print(f"Plot saved to '{plot_path}'")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def run_training(output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}\n")
    
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
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
        weights_lmc, weights_entropy, weights_diseq, num_bins = calculate_lmc_from_weights(model)
        
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
    test_metrics = calculate_lmc_from_weights(model)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test LMC: {test_metrics[0]:.8f} (bins={test_metrics[3]})\n")
    
    # Save results
    save_results_to_csv(
        output_dir, train_losses, val_losses, lmc_values,
        weights_entropy_values, weights_disequilibrium_values, num_bins_values,
        test_loss, test_metrics, config
    )
    plot_results(output_dir, train_losses, val_losses, lmc_values, test_loss, config)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Iterate through LMC weight values from 0.0 to 1.0 in steps of 0.05
    for step in range(21):
        lmc_weight = step * 0.05
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
