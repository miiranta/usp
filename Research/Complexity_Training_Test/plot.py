import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Define the output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(SCRIPT_DIR, "output")

# Lists to store data
lmc_weights = []
test_losses = []
test_lmc_complexities = []

# Dictionary to store training curves for each LMC weight
training_data = {}

# Get all folders in the output directory
folders = sorted([f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))])

# Extract data from each folder
for folder in folders:
    # Extract LMC weight from folder name (e.g., "output_LMC_0.00" -> 0.00)
    lmc_weight = float(folder.split('_')[-1])
    
    # Path to the CSV file
    csv_path = os.path.join(output_dir, folder, "z_loss_test_results_transformers.csv")
    
    if os.path.exists(csv_path):
        # Read the file manually to handle the mixed format
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Extract Test Loss and Test LMC Complexity from the first section
        test_loss = None
        test_lmc = None
        
        for line in lines[:10]:  # Only check first 10 lines
            if line.startswith('Test Loss,'):
                test_loss = float(line.split(',')[1].strip())
            elif line.startswith('Test LMC Complexity,'):
                test_lmc = float(line.split(',')[1].strip())
        
        if test_loss is not None and test_lmc is not None:
            lmc_weights.append(lmc_weight)
            test_losses.append(test_loss)
            test_lmc_complexities.append(test_lmc)
            
            print(f"LMC Weight: {lmc_weight:.2f}, Test Loss: {test_loss:.4f}, Test LMC: {test_lmc:.4f}")
        
        # Read the training data (second section of the CSV)
        # Find the line where the epoch data starts
        epoch_start_idx = None
        for i, line in enumerate(lines):
            if line.startswith('Epoch,Training Loss'):
                epoch_start_idx = i
                break
        
        if epoch_start_idx is not None:
            # Read the epoch data
            epoch_data = []
            for line in lines[epoch_start_idx + 1:]:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split(',')
                    if len(parts) >= 4:  # Now has 7 columns: Epoch, Training Loss, Validation Loss, Model LMC, Weights Entropy, Weights Disequilibrium, Num Bins
                        epoch_data.append({
                            'epoch': int(parts[0]),
                            'training_loss': float(parts[1]),
                            'validation_loss': float(parts[2]),
                            'model_lmc': float(parts[3])
                        })
            
            if epoch_data:
                training_data[lmc_weight] = epoch_data

# Create the plot with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Test Loss on the left y-axis
color = 'tab:blue'
ax1.set_xlabel('LMC Weight', fontsize=12)
ax1.set_ylabel('Test Loss', color=color, fontsize=12)
ax1.plot(lmc_weights, test_losses, marker='o', linestyle='-', linewidth=2, markersize=8, color=color, label='Test Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Create a second y-axis for Test LMC
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Test LMC Complexity', color=color, fontsize=12)
ax2.plot(lmc_weights, test_lmc_complexities, marker='s', linestyle='-', linewidth=2, markersize=8, color=color, label='Test LMC Complexity')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and legends
plt.title('Test Loss and Test LMC Complexity vs LMC Weight', fontsize=14, fontweight='bold')
fig.tight_layout()

# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Save the plot
plt.savefig('plot_test_loss_and_lmc_plot.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'test_loss_and_lmc_plot.png'")

# Show the plot
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"LMC Weight Range: {min(lmc_weights):.2f} to {max(lmc_weights):.2f}")
print(f"Test Loss Range: {min(test_losses):.4f} to {max(test_losses):.4f}")
print(f"Test LMC Range: {min(test_lmc_complexities):.4f} to {max(test_lmc_complexities):.4f}")

# ============================================================================
# NEW PLOT: Training curves for all LMC weights
# ============================================================================

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Generate colors for each LMC weight
colors = plt.cm.viridis(np.linspace(0, 1, len(training_data)))

# Find the LMC weights with lowest final values for each metric
final_training_losses = {}
final_validation_losses = {}
final_model_lmc = {}

for lmc_weight, data in training_data.items():
    final_training_losses[lmc_weight] = data[-1]['training_loss']
    final_validation_losses[lmc_weight] = data[-1]['validation_loss']
    final_model_lmc[lmc_weight] = data[-1]['model_lmc']

best_training_lmc = min(final_training_losses.items(), key=lambda x: x[1])[0]
best_validation_lmc = min(final_validation_losses.items(), key=lambda x: x[1])[0]
best_model_lmc = min(final_model_lmc.items(), key=lambda x: x[1])[0]

print(f"\nBest final Training Loss: LMC {best_training_lmc:.2f} with loss {final_training_losses[best_training_lmc]:.4f}")
print(f"Best final Validation Loss: LMC {best_validation_lmc:.2f} with loss {final_validation_losses[best_validation_lmc]:.4f}")
print(f"Best final Model LMC: LMC {best_model_lmc:.2f} with complexity {final_model_lmc[best_model_lmc]:.4f}")

# Plot 1: Training Loss
ax = axes[0]
for i, (lmc_weight, data) in enumerate(sorted(training_data.items())):
    epochs = [d['epoch'] for d in data]
    training_losses = [d['training_loss'] for d in data]
    
    # Highlight the best curve
    if lmc_weight == best_training_lmc:
        ax.plot(epochs, training_losses, marker='o', linestyle='-', linewidth=4, 
                markersize=8, color='red', label=f'LMC {lmc_weight:.2f} ★', zorder=10)
    else:
        ax.plot(epochs, training_losses, marker='o', linestyle='-', linewidth=2, 
                markersize=5, color=colors[i], label=f'LMC {lmc_weight:.2f}', alpha=0.7)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 2: Validation Loss
ax = axes[1]
for i, (lmc_weight, data) in enumerate(sorted(training_data.items())):
    epochs = [d['epoch'] for d in data]
    validation_losses = [d['validation_loss'] for d in data]
    
    # Highlight the best curve
    if lmc_weight == best_validation_lmc:
        ax.plot(epochs, validation_losses, marker='s', linestyle='-', linewidth=4, 
                markersize=8, color='red', label=f'LMC {lmc_weight:.2f} ★', zorder=10)
    else:
        ax.plot(epochs, validation_losses, marker='s', linestyle='-', linewidth=2, 
                markersize=5, color=colors[i], label=f'LMC {lmc_weight:.2f}', alpha=0.7)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Validation Loss vs Epoch', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 3: Model LMC
ax = axes[2]
for i, (lmc_weight, data) in enumerate(sorted(training_data.items())):
    epochs = [d['epoch'] for d in data]
    model_lmc = [d['model_lmc'] for d in data]
    
    # Highlight the best curve
    if lmc_weight == best_model_lmc:
        ax.plot(epochs, model_lmc, marker='^', linestyle='-', linewidth=4, 
                markersize=8, color='red', label=f'LMC {lmc_weight:.2f} ★', zorder=10)
    else:
        ax.plot(epochs, model_lmc, marker='^', linestyle='-', linewidth=2, 
                markersize=5, color=colors[i], label=f'LMC {lmc_weight:.2f}', alpha=0.7)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Model LMC', fontsize=12)
ax.set_title('Model LMC vs Epoch', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_training_curves_all_lmc_weights.png', dpi=300, bbox_inches='tight')
print("\nTraining curves plot saved as 'training_curves_all_lmc_weights.png'")
plt.show()

# ============================================================================
# 3D PLOT: Training curves using seaborn with 3D visualization
# ============================================================================

# Prepare data for 3D plots
training_data_3d = []

for lmc_weight, data in training_data.items():
    for entry in data:
        training_data_3d.append({
            'LMC Weight': lmc_weight,
            'Epoch': entry['epoch'],
            'Training Loss': entry['training_loss'],
            'Validation Loss': entry['validation_loss'],
            'Model LMC': entry['model_lmc']
        })

df_3d = pd.DataFrame(training_data_3d)

# Create 3D plots for each metric with better visibility
fig = plt.figure(figsize=(20, 6))

# Generate colors for each LMC weight
unique_lmc_weights = sorted(df_3d['LMC Weight'].unique())
colors_map = plt.cm.viridis(np.linspace(0, 1, len(unique_lmc_weights)))

# 3D Plot 1: Training Loss
ax1 = fig.add_subplot(131, projection='3d')
for i, lmc_weight in enumerate(unique_lmc_weights):
    subset = df_3d[df_3d['LMC Weight'] == lmc_weight]
    ax1.plot(subset['Epoch'], [lmc_weight]*len(subset), subset['Training Loss'], 
             marker='o', color=colors_map[i], label=f'LMC {lmc_weight:.2f}', linewidth=2.5, markersize=6)

ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('LMC Weight', fontsize=11, fontweight='bold')
ax1.set_zlabel('Training Loss', fontsize=11, fontweight='bold')
ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
ax1.view_init(elev=25, azim=120)
ax1.grid(True, alpha=0.3)

# 3D Plot 2: Validation Loss
ax2 = fig.add_subplot(132, projection='3d')
for i, lmc_weight in enumerate(unique_lmc_weights):
    subset = df_3d[df_3d['LMC Weight'] == lmc_weight]
    ax2.plot(subset['Epoch'], [lmc_weight]*len(subset), subset['Validation Loss'], 
             marker='s', color=colors_map[i], label=f'LMC {lmc_weight:.2f}', linewidth=2.5, markersize=6)

ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('LMC Weight', fontsize=11, fontweight='bold')
ax2.set_zlabel('Validation Loss', fontsize=11, fontweight='bold')
ax2.set_title('Validation Loss', fontsize=13, fontweight='bold')
ax2.view_init(elev=25, azim=120)
ax2.grid(True, alpha=0.3)

# 3D Plot 3: Model LMC
ax3 = fig.add_subplot(133, projection='3d')
for i, lmc_weight in enumerate(unique_lmc_weights):
    subset = df_3d[df_3d['LMC Weight'] == lmc_weight]
    ax3.plot(subset['Epoch'], [lmc_weight]*len(subset), subset['Model LMC'], 
             marker='^', color=colors_map[i], label=f'LMC {lmc_weight:.2f}', linewidth=2.5, markersize=6)

ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('LMC Weight', fontsize=11, fontweight='bold')
ax3.set_zlabel('Model LMC', fontsize=11, fontweight='bold')
ax3.set_title('Model LMC', fontsize=13, fontweight='bold')
ax3.view_init(elev=25, azim=120)
ax3.grid(True, alpha=0.3)

fig.suptitle('3D Training Metrics: Epoch vs LMC Weight vs Metric Value', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('plot_3d_training_curves_all_lmc_weights.png', dpi=300, bbox_inches='tight')
print("\n3D Training curves plot saved as 'plot_3d_training_curves_all_lmc_weights.png'")
plt.show()
