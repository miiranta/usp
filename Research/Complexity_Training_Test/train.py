import os
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    # Enable memory efficient features
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
else:
    print("WARNING: CUDA is not available! Using CPU instead.\n")

VOCAB_SIZE = 10000
HIDDEN_DIM = 400   # Reasonable size for P5000
NUM_LAYERS = 2     # Good depth for small dataset
NUM_ATTENTION_HEADS = 4  # Proper multi-head attention
BATCH_SIZE = 20    # Better batch size for stable gradients
GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation needed
EPOCHS = 40
LEARNING_RATE = 5e-4  # Higher LR for better convergence
SEQ_LENGTH = 256   # Longer context window
WARMUP_STEPS = 500
MAX_GRAD_NORM = 0.5
MAX_SAMPLES = None  # Use full dataset to prevent overfitting

# LMC Complexity weight (0.0 = 100% loss optimization, 1.0 = 100% LMC maximization)
LMC_WEIGHT = 0.0  # e.g., 0.2 = 20% LMC maximization + 80% loss minimization

class TextDataset(Dataset):
    """Optimized text dataset for transformers"""
    def __init__(self, file_path, tokenizer, seq_length=SEQ_LENGTH, max_samples=None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        # Read the text file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Tokenize the entire text
        print(f"Tokenizing {file_path}...")
        encodings = tokenizer(
            text,
            return_tensors='pt',
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False
        )
        
        self.input_ids = encodings['input_ids'][0]
        
        if max_samples:
            max_length = max_samples * seq_length
            self.input_ids = self.input_ids[:max_length]
        
        print(f"Dataset loaded: {len(self.input_ids)} tokens")
    
    def __len__(self):
        return max(0, len(self.input_ids) - self.seq_length)
    
    def __getitem__(self, idx):
        # Get sequence and target
        input_ids = self.input_ids[idx:idx + self.seq_length]
        target_ids = self.input_ids[idx + 1:idx + self.seq_length + 1]
        
        # Ensure correct length
        if len(input_ids) < self.seq_length:
            padding = self.seq_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(padding, dtype=torch.long)])
            target_ids = torch.cat([target_ids, torch.full((padding,), -100, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'labels': target_ids
        }

class TransformerLLM(nn.Module):
    """Transformer-based Language Model"""
    def __init__(self, vocab_size, hidden_dim, num_layers, num_attention_heads, seq_length):
        super(TransformerLLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(seq_length, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.3,  # Increased dropout for regularization
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
    
    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        embeddings = self.embedding(input_ids) + self.position_embedding(positions)
        
        # Convert attention mask to proper format (True = mask out, False = attend)
        # TransformerEncoder expects True where we want to mask
        transformer_out = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        # Language model head
        logits = self.lm_head(transformer_out)
        
        return logits

def collate_fn(batch):
    """Collate function for data loader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Create attention mask (True for padding tokens to be masked out)
    attention_mask = (input_ids == 0)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }

def calculate_lmc_complexity(logits, temperature=1.0):
    """
    Calculate LMC complexity from model predictions.
    LMC complexity C = H * D where:
    - H is Shannon entropy (information)
    - D is disequilibrium (distance from uniform distribution)
    
    Args:
        logits: Model output logits (batch_size, seq_length, vocab_size)
        temperature: Temperature for softmax (default=1.0)
    
    Returns:
        lmc: LMC complexity value (scalar)
        shannon_entropy: Shannon entropy H
        disequilibrium: Disequilibrium D
    """
    # Apply softmax to get probability distribution
    probs = torch.softmax(logits / temperature, dim=-1)
    
    # Average probabilities across batch and sequence length
    # Shape: (vocab_size,)
    avg_probs = probs.mean(dim=0).mean(dim=0)
    
    # Ensure probabilities are valid (clip to avoid log(0))
    eps = 1e-10
    avg_probs = torch.clamp(avg_probs, eps, 1.0)
    avg_probs = avg_probs / avg_probs.sum()  # Renormalize
    
    # Calculate Shannon entropy H = -sum(p_i * log(p_i))
    shannon_entropy = -(avg_probs * torch.log(avg_probs)).sum()
    
    # Calculate disequilibrium D = sum((p_i - 1/N)^2)
    N = avg_probs.size(0)
    uniform_prob = 1.0 / N
    disequilibrium = ((avg_probs - uniform_prob) ** 2).sum()
    
    # LMC complexity C = H * D
    lmc = shannon_entropy * disequilibrium
    
    return lmc, shannon_entropy, disequilibrium

def train_epoch(model, train_loader, optimizer, scheduler, device, gradient_accumulation_steps, lmc_weight=0.2):
    """
    Train for one epoch with gradient accumulation and LMC complexity optimization.
    
    Args:
        lmc_weight: Weight for LMC complexity (0.0-1.0)
                   0.0 = 100% loss minimization, 0% LMC maximization
                   0.5 = 50% loss minimization, 50% LMC maximization
                   1.0 = 0% loss minimization, 100% LMC maximization
    """
    model.train()
    total_loss = 0
    total_lmc = 0
    total_combined_loss = 0
    batch_count = 0
    
    # Calculate weights: loss_weight + lmc_weight = 1.0
    loss_weight = 1.0 - lmc_weight
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, model.vocab_size)
        labels_flat = labels.view(-1)
        
        # Cross-entropy loss (ignore padding tokens marked with -100)
        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_flat, labels_flat)
        
        # Calculate LMC complexity (for optimization only)
        lmc, _, _ = calculate_lmc_complexity(logits)
        
        # Combined objective: weighted sum of loss minimization and LMC maximization
        # Minimize: (1-lmc_weight)*loss - lmc_weight*lmc
        combined_loss = loss_weight * ce_loss - lmc_weight * lmc
        
        # Normalize by gradient accumulation steps
        combined_loss = combined_loss / gradient_accumulation_steps
        
        # Backward pass
        combined_loss.backward()
        
        total_loss += ce_loss.detach().item()
        total_lmc += lmc.detach().item()
        total_combined_loss += combined_loss.detach().item() * gradient_accumulation_steps
        batch_count += 1
        
        # Gradient accumulation step
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            progress_bar.set_postfix({
                'loss': total_loss / batch_count,
                'lmc': total_lmc / batch_count,
                'combined': total_combined_loss / batch_count
            })
    
    return total_loss / batch_count, total_combined_loss / batch_count

def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
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
    """Test the model"""
    model.eval()
    total_loss = 0
    
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

def measure_model_lmc(model, sample_loader, device):
    """
    Measure the LMC complexity of the model's weights.
    This should be called once per epoch to measure the model's current state.
    """
    model.eval()
    
    with torch.no_grad():
        # Get one batch to measure model's predictions
        batch = next(iter(sample_loader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        logits = model(input_ids, attention_mask=attention_mask)
        lmc, _, _ = calculate_lmc_complexity(logits)
    
    return lmc.item()

def run(output_dir='output'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}\n")
    
    # Initialize tokenizer
    print("Initializing GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_path = 'dataset/wikitext-2/wiki.train.tokens'
    val_path = 'dataset/wikitext-2/wiki.valid.tokens'
    test_path = 'dataset/wikitext-2/wiki.test.tokens'
    
    # Check if all files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Error: Dataset file '{path}' not found!")
            return
    
    print("Loading training dataset...")
    train_dataset = TextDataset(train_path, tokenizer, seq_length=SEQ_LENGTH, max_samples=MAX_SAMPLES)
    
    print("Loading validation dataset...")
    val_dataset = TextDataset(val_path, tokenizer, seq_length=SEQ_LENGTH, max_samples=MAX_SAMPLES)
    
    print("Loading test dataset...")
    test_dataset = TextDataset(test_path, tokenizer, seq_length=SEQ_LENGTH, max_samples=MAX_SAMPLES)
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("\nInitializing Transformer model...")
    vocab_size = len(tokenizer)
    model = TransformerLLM(
        vocab_size=vocab_size,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        seq_length=SEQ_LENGTH
    ).to(device)
    
    # Move model to GPU and use mixed precision if available
    print(f"Model device: {next(model.parameters()).device}")
    
    # Optimizer and scheduler
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {device}")
    print(f"Total training steps: {total_steps}")
    print(f"LMC weight: {LMC_WEIGHT} ({LMC_WEIGHT*100:.0f}% LMC maximization, {(1-LMC_WEIGHT)*100:.0f}% loss minimization)\n")
    
    # Lists to track losses and LMC complexity
    train_losses = []
    val_losses = []
    lmc_values = []  # Single LMC measure per epoch
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss, train_combined = train_epoch(
            model, train_loader, optimizer, scheduler, device, 
            GRADIENT_ACCUMULATION_STEPS, LMC_WEIGHT
        )
        val_loss = validate(model, val_loader, device)
        
        # Measure LMC complexity of the model (once per epoch)
        model_lmc = measure_model_lmc(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lmc_values.append(model_lmc)
        
        print(f"Training Loss: {train_loss:.4f}, Combined: {train_combined:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Model LMC Complexity: {model_lmc:.4f}")
    
    # Test the model
    print("\nEvaluating on test set...")
    test_loss = test(model, test_loader, device)
    test_lmc = measure_model_lmc(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test LMC: {test_lmc:.4f}\n")
    
    # Save test loss to CSV
    csv_path = os.path.join(output_dir, 'z_loss_test_results_transformers.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Summary metrics
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Test Loss', f'{test_loss:.4f}'])
        writer.writerow(['Test LMC Complexity', f'{test_lmc:.4f}'])
        writer.writerow(['Final Training Loss', f'{train_losses[-1]:.4f}'])
        writer.writerow(['Final Validation Loss', f'{val_losses[-1]:.4f}'])
        writer.writerow(['Final Model LMC', f'{lmc_values[-1]:.4f}'])
        writer.writerow(['LMC Weight', f'{LMC_WEIGHT}'])
        writer.writerow(['Loss Weight', f'{1.0 - LMC_WEIGHT}'])
        
        # Empty row separator
        writer.writerow([])
        
        # Epoch-by-epoch data
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Model LMC'])
        for epoch in range(EPOCHS):
            writer.writerow([
                epoch + 1,
                f'{train_losses[epoch]:.4f}',
                f'{val_losses[epoch]:.4f}',
                f'{lmc_values[epoch]:.4f}'
            ])
    
    print(f"Test results saved to '{csv_path}'")
    
    # Plot losses and LMC complexity on the same graph with dual y-axes
    plot_path = os.path.join(output_dir, 'z_loss_training_losses_transformers.png')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot losses on primary y-axis
    ax1.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss', marker='o', color='blue')
    ax1.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss', marker='s', color='orange')
    ax1.axhline(y=test_loss, color='red', linestyle='--', label=f'Test Loss ({test_loss:.4f})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for LMC complexity
    ax2 = ax1.twinx()
    ax2.plot(range(1, EPOCHS + 1), lmc_values, label='Model LMC', marker='D', color='green', linewidth=2)
    ax2.axhline(y=test_lmc, color='darkgreen', linestyle='--', label=f'Test LMC ({test_lmc:.4f})')
    ax2.set_ylabel('LMC Complexity (C = H × D)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Title
    plt.title('Transformer Model Training: Loss and LMC Complexity', fontsize=14, pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    print(f"Loss and LMC plot saved to '{plot_path}'")
    #plt.show()
    
    # Save the model
    # model_path = os.path.join(output_dir, 'z_loss_model_transformers.pt')
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'vocab_size': vocab_size,
    #     'hidden_dim': HIDDEN_DIM,
    #     'num_layers': NUM_LAYERS,
    #     'num_attention_heads': NUM_ATTENTION_HEADS,
    #     'seq_length': SEQ_LENGTH
    # }, model_path)
    
    # print(f"Model saved to '{model_path}'")

def main():
    global LMC_WEIGHT
    global EPOCHS
    
    EPOCHS = 50
    LMC_WEIGHT = 0.0
    # Iterate from 0.0 to 1.0 in steps of 0.05, run(output-LMC_WEIGHT)
    # Ignore if folder already exists
    for step in range(21):
        LMC_WEIGHT = step * 0.05
        output_dir = f'output/output_LMC_{LMC_WEIGHT:.2f}'
        if not os.path.exists(output_dir):
            run(output_dir)
        else:
            print(f"Skipping LMC_WEIGHT={LMC_WEIGHT:.2f}, output directory '{output_dir}' already exists.")

    EPOCHS = 50
    LMC_WEIGHT = 0.0
    # Final run with EPOCHS=50 and LMC_WEIGHT=0
    output_dir = f'output/final_run_LMC_{LMC_WEIGHT:.2f}_epochs_{EPOCHS}'
    if not os.path.exists(output_dir):
        run(output_dir)

if __name__ == '__main__':
    main()
