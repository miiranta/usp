import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Define the output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(SCRIPT_DIR, "output")

# Lists to store data
lmc_weights = []
test_losses_wiki_mean = []
test_losses_wiki_std = []
test_lmc_wiki_mean = []
test_lmc_wiki_std = []
test_losses_shakespeare_mean = []
test_losses_shakespeare_std = []
test_lmc_shakespeare_mean = []
test_lmc_shakespeare_std = []

# Dictionary to store training curves for each LMC weight
training_data = {}

# Get all folders in the output directory
folders = sorted([f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))])

# Extract data from each folder (using AGGREGATE CSVs only)
for folder in folders:
    # Extract LMC weight from folder name (e.g., "output_LMC_0.00" -> 0.00)
    # Skip folders that don't match the expected pattern (e.g., output_AUTO_LAMBDA)
    try:
        lmc_weight = float(folder.split('_')[-1])
    except ValueError:
        print(f"Skipping folder (not a numeric LMC weight): {folder}")
        continue
    
    # Path to the AGGREGATE CSV file
    csv_path = os.path.join(output_dir, folder, "z_loss_test_results_transformers-AGGREGATE.csv")
    
    if os.path.exists(csv_path):
        print(f"Reading aggregate CSV: {csv_path}")
        
        # Read the file manually to extract summary metrics
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Extract test metrics from summary section
        test_loss_wiki = None
        test_loss_wiki_std = None
        test_lmc_wiki = None
        test_lmc_wiki_std = None
        test_loss_shakespeare = None
        test_loss_shakespeare_std = None
        test_lmc_shakespeare = None
        test_lmc_shakespeare_std = None
        
        for i, line in enumerate(lines):
            if 'Test Loss Mean' in line and 'WikiText-2' in ''.join(lines[max(0,i-5):i]):
                test_loss_wiki = float(line.split(',')[1].strip())
            elif 'Test Loss Std' in line and 'WikiText-2' in ''.join(lines[max(0,i-5):i]):
                test_loss_wiki_std = float(line.split(',')[1].strip())
            elif 'Test LMC Mean' in line and 'WikiText-2' in ''.join(lines[max(0,i-5):i]):
                test_lmc_wiki = float(line.split(',')[1].strip())
            elif 'Test LMC Std' in line and 'WikiText-2' in ''.join(lines[max(0,i-5):i]):
                test_lmc_wiki_std = float(line.split(',')[1].strip())
            elif 'Test Loss Mean' in line and 'Shakespeare' in ''.join(lines[max(0,i-5):i]):
                test_loss_shakespeare = float(line.split(',')[1].strip())
            elif 'Test Loss Std' in line and 'Shakespeare' in ''.join(lines[max(0,i-5):i]):
                test_loss_shakespeare_std = float(line.split(',')[1].strip())
            elif 'Test LMC Mean' in line and 'Shakespeare' in ''.join(lines[max(0,i-5):i]):
                test_lmc_shakespeare = float(line.split(',')[1].strip())
            elif 'Test LMC Std' in line and 'Shakespeare' in ''.join(lines[max(0,i-5):i]):
                test_lmc_shakespeare_std = float(line.split(',')[1].strip())
        
        # Find the epoch-by-epoch data section
        epoch_start_idx = None
        for i, line in enumerate(lines):
            if line.startswith('Epoch,'):
                epoch_start_idx = i
                break
        
        if epoch_start_idx is not None:
            # Read epoch data with statistics
            epoch_data = []
            for line in lines[epoch_start_idx + 1:]:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split(',')
                    # Expected columns: Epoch, Train_Loss_Mean, Train_Loss_Std, Train_Loss_Min, Train_Loss_Max,
                    #                   Val_Loss_Mean, Val_Loss_Std, Val_Loss_Min, Val_Loss_Max,
                    #                   LMC_Mean, LMC_Std, LMC_Min, LMC_Max, ...
                    if len(parts) >= 13:
                        epoch_data.append({
                            'epoch': int(parts[0]),
                            'train_loss_mean': float(parts[1]),
                            'train_loss_std': float(parts[2]),
                            'val_loss_mean': float(parts[5]),
                            'val_loss_std': float(parts[6]),
                            'lmc_mean': float(parts[9]),
                            'lmc_std': float(parts[10])
                        })
            
            if epoch_data:
                training_data[lmc_weight] = epoch_data
                
                # Store test metrics if available
                if all(v is not None for v in [test_loss_wiki, test_loss_wiki_std, test_lmc_wiki, test_lmc_wiki_std,
                                                test_loss_shakespeare, test_loss_shakespeare_std, 
                                                test_lmc_shakespeare, test_lmc_shakespeare_std]):
                    lmc_weights.append(lmc_weight)
                    test_losses_wiki_mean.append(test_loss_wiki)
                    test_losses_wiki_std.append(test_loss_wiki_std)
                    test_lmc_wiki_mean.append(test_lmc_wiki)
                    test_lmc_wiki_std.append(test_lmc_wiki_std)
                    test_losses_shakespeare_mean.append(test_loss_shakespeare)
                    test_losses_shakespeare_std.append(test_loss_shakespeare_std)
                    test_lmc_shakespeare_mean.append(test_lmc_shakespeare)
                    test_lmc_shakespeare_std.append(test_lmc_shakespeare_std)
                    
                    print(f"LMC Weight: {lmc_weight:.2f}")
                    print(f"  WikiText-2 - Test Loss: {test_loss_wiki:.4f}±{test_loss_wiki_std:.4f}, "
                          f"Test LMC: {test_lmc_wiki:.4f}±{test_lmc_wiki_std:.4f}")
                    print(f"  Tiny-Shakespeare - Test Loss: {test_loss_shakespeare:.4f}±{test_loss_shakespeare_std:.4f}, "
                          f"Test LMC: {test_lmc_shakespeare:.4f}±{test_lmc_shakespeare_std:.4f}")
    else:
        print(f"Aggregate CSV not found for {folder}")

if not lmc_weights:
    print("No aggregate CSV files found!")
    exit(1)

# ============================================================================
# PLOT 1: Test Loss and Test LMC vs LMC Weight with 95% CI (WikiText-2 and Tiny-Shakespeare)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Convert to numpy arrays
lmc_weights_array = np.array(lmc_weights)
test_losses_wiki_mean = np.array(test_losses_wiki_mean)
test_losses_wiki_std = np.array(test_losses_wiki_std)
test_lmc_wiki_mean = np.array(test_lmc_wiki_mean)
test_lmc_wiki_std = np.array(test_lmc_wiki_std)
test_losses_shakespeare_mean = np.array(test_losses_shakespeare_mean)
test_losses_shakespeare_std = np.array(test_losses_shakespeare_std)
test_lmc_shakespeare_mean = np.array(test_lmc_shakespeare_mean)
test_lmc_shakespeare_std = np.array(test_lmc_shakespeare_std)

# Calculate 95% CI (1.96 * std)
ci_factor = 1.96

# --- WikiText-2 Plots ---

# Plot 1.1: WikiText-2 Test Loss
ax1 = axes[0, 0]
test_loss_wiki_ci = ci_factor * test_losses_wiki_std
color = 'tab:blue'
ax1.set_xlabel('LMC Weight', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Loss (WikiText-2)', color=color, fontsize=12, fontweight='bold')
ax1.plot(lmc_weights_array, test_losses_wiki_mean, marker='o', linestyle='-', linewidth=2.5, 
         markersize=10, color=color, label='Test Loss (Mean)', zorder=3)
ax1.fill_between(lmc_weights_array, test_losses_wiki_mean - test_loss_wiki_ci, 
                 test_losses_wiki_mean + test_loss_wiki_ci,
                 alpha=0.2, color=color, label='Test Loss (95% CI)', zorder=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')
ax1.set_title('WikiText-2: Test Loss vs LMC Weight', fontsize=13, fontweight='bold')

# Plot 1.2: WikiText-2 Test LMC
ax2 = axes[0, 1]
test_lmc_wiki_ci = ci_factor * test_lmc_wiki_std
color = 'tab:red'
ax2.set_xlabel('LMC Weight', fontsize=12, fontweight='bold')
ax2.set_ylabel('Test LMC (WikiText-2)', color=color, fontsize=12, fontweight='bold')
ax2.plot(lmc_weights_array, test_lmc_wiki_mean, marker='s', linestyle='-', linewidth=2.5, 
         markersize=10, color=color, label='Test LMC (Mean)', zorder=3)
ax2.fill_between(lmc_weights_array, test_lmc_wiki_mean - test_lmc_wiki_ci, 
                 test_lmc_wiki_mean + test_lmc_wiki_ci,
                 alpha=0.2, color=color, label='Test LMC (95% CI)', zorder=2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')
ax2.set_title('WikiText-2: Test LMC vs LMC Weight', fontsize=13, fontweight='bold')

# --- Tiny-Shakespeare Plots ---

# Plot 1.3: Tiny-Shakespeare Test Loss
ax3 = axes[1, 0]
test_loss_shakespeare_ci = ci_factor * test_losses_shakespeare_std
color = 'tab:purple'
ax3.set_xlabel('LMC Weight', fontsize=12, fontweight='bold')
ax3.set_ylabel('Test Loss (Tiny-Shakespeare)', color=color, fontsize=12, fontweight='bold')
ax3.plot(lmc_weights_array, test_losses_shakespeare_mean, marker='o', linestyle='-', linewidth=2.5, 
         markersize=10, color=color, label='Test Loss (Mean)', zorder=3)
ax3.fill_between(lmc_weights_array, test_losses_shakespeare_mean - test_loss_shakespeare_ci, 
                 test_losses_shakespeare_mean + test_loss_shakespeare_ci,
                 alpha=0.2, color=color, label='Test Loss (95% CI)', zorder=2)
ax3.tick_params(axis='y', labelcolor=color)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best')
ax3.set_title('Tiny-Shakespeare: Test Loss vs LMC Weight', fontsize=13, fontweight='bold')

# Plot 1.4: Tiny-Shakespeare Test LMC
ax4 = axes[1, 1]
test_lmc_shakespeare_ci = ci_factor * test_lmc_shakespeare_std
color = 'tab:green'
ax4.set_xlabel('LMC Weight', fontsize=12, fontweight='bold')
ax4.set_ylabel('Test LMC (Tiny-Shakespeare)', color=color, fontsize=12, fontweight='bold')
ax4.plot(lmc_weights_array, test_lmc_shakespeare_mean, marker='s', linestyle='-', linewidth=2.5, 
         markersize=10, color=color, label='Test LMC (Mean)', zorder=3)
ax4.fill_between(lmc_weights_array, test_lmc_shakespeare_mean - test_lmc_shakespeare_ci, 
                 test_lmc_shakespeare_mean + test_lmc_shakespeare_ci,
                 alpha=0.2, color=color, label='Test LMC (95% CI)', zorder=2)
ax4.tick_params(axis='y', labelcolor=color)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best')
ax4.set_title('Tiny-Shakespeare: Test LMC vs LMC Weight', fontsize=13, fontweight='bold')

fig.suptitle('Test Metrics vs LMC Weight (with 95% CI)', fontsize=15, fontweight='bold', y=0.995)
fig.tight_layout()
output_plot_path = os.path.join(SCRIPT_DIR, 'plot_test_loss_and_lmc_aggregate.png')
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"\nTest metrics plot saved as '{output_plot_path}'")
plt.show()

# ============================================================================
# PLOT 2: Training curves for all LMC weights with 95% CI
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Generate colors for each LMC weight
colors_grad = plt.cm.viridis(np.linspace(0, 1, len(training_data)))

# Find best performing configurations
final_train_losses = {}
final_val_losses = {}
final_model_lmc = {}

for lmc_weight, data in training_data.items():
    final_train_losses[lmc_weight] = data[-1]['train_loss_mean']
    final_val_losses[lmc_weight] = data[-1]['val_loss_mean']
    final_model_lmc[lmc_weight] = data[-1]['lmc_mean']

best_training = min(final_train_losses.items(), key=lambda x: x[1])[0]
best_validation = min(final_val_losses.items(), key=lambda x: x[1])[0]
best_lmc = min(final_model_lmc.items(), key=lambda x: x[1])[0]

print(f"\nBest final Training Loss: LMC {best_training:.4f}")
print(f"Best final Validation Loss: LMC {best_validation:.4f}")
print(f"Best final Model LMC: LMC {best_lmc:.4f}")

# Plot 1: Training Loss with 95% CI
ax = axes[0]
for i, (lmc_weight, data) in enumerate(sorted(training_data.items())):
    epochs = np.array([d['epoch'] for d in data])
    train_loss_mean = np.array([d['train_loss_mean'] for d in data])
    train_loss_std = np.array([d['train_loss_std'] for d in data])
    train_loss_ci = ci_factor * train_loss_std
    
    if lmc_weight == best_training:
        ax.plot(epochs, train_loss_mean, marker='o', linestyle='-', linewidth=3.5, 
                markersize=10, color='red', label=f'LMC {lmc_weight:.4f} ★', zorder=10)
        ax.fill_between(epochs, train_loss_mean - train_loss_ci, train_loss_mean + train_loss_ci,
                       alpha=0.25, color='red', zorder=9)
    else:
        ax.plot(epochs, train_loss_mean, marker='o', linestyle='-', linewidth=2, 
                markersize=6, color=colors_grad[i], label=f'LMC {lmc_weight:.4f}', alpha=0.8, zorder=5-i*0.1)
        ax.fill_between(epochs, train_loss_mean - train_loss_ci, train_loss_mean + train_loss_ci,
                       alpha=0.1, color=colors_grad[i], zorder=4-i*0.1)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
ax.set_title('Training Loss vs Epoch (with 95% CI)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 2: Validation Loss with 95% CI
ax = axes[1]
for i, (lmc_weight, data) in enumerate(sorted(training_data.items())):
    epochs = np.array([d['epoch'] for d in data])
    val_loss_mean = np.array([d['val_loss_mean'] for d in data])
    val_loss_std = np.array([d['val_loss_std'] for d in data])
    val_loss_ci = ci_factor * val_loss_std
    
    if lmc_weight == best_validation:
        ax.plot(epochs, val_loss_mean, marker='s', linestyle='-', linewidth=3.5, 
                markersize=10, color='red', label=f'LMC {lmc_weight:.4f} ★', zorder=10)
        ax.fill_between(epochs, val_loss_mean - val_loss_ci, val_loss_mean + val_loss_ci,
                       alpha=0.25, color='red', zorder=9)
    else:
        ax.plot(epochs, val_loss_mean, marker='s', linestyle='-', linewidth=2, 
                markersize=6, color=colors_grad[i], label=f'LMC {lmc_weight:.4f}', alpha=0.8, zorder=5-i*0.1)
        ax.fill_between(epochs, val_loss_mean - val_loss_ci, val_loss_mean + val_loss_ci,
                       alpha=0.1, color=colors_grad[i], zorder=4-i*0.1)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
ax.set_title('Validation Loss vs Epoch (with 95% CI)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 3: Model LMC with 95% CI
ax = axes[2]
for i, (lmc_weight, data) in enumerate(sorted(training_data.items())):
    epochs = np.array([d['epoch'] for d in data])
    lmc_mean = np.array([d['lmc_mean'] for d in data])
    lmc_std = np.array([d['lmc_std'] for d in data])
    lmc_ci = ci_factor * lmc_std
    
    if lmc_weight == best_lmc:
        ax.plot(epochs, lmc_mean, marker='^', linestyle='-', linewidth=3.5, 
                markersize=10, color='red', label=f'LMC {lmc_weight:.4f} ★', zorder=10)
        ax.fill_between(epochs, lmc_mean - lmc_ci, lmc_mean + lmc_ci,
                       alpha=0.25, color='red', zorder=9)
    else:
        ax.plot(epochs, lmc_mean, marker='^', linestyle='-', linewidth=2, 
                markersize=6, color=colors_grad[i], label=f'LMC {lmc_weight:.4f}', alpha=0.8, zorder=5-i*0.1)
        ax.fill_between(epochs, lmc_mean - lmc_ci, lmc_mean + lmc_ci,
                       alpha=0.1, color=colors_grad[i], zorder=4-i*0.1)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Model LMC', fontsize=12, fontweight='bold')
ax.set_title('Model LMC vs Epoch (with 95% CI)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

fig.suptitle('Training Curves with 95% Confidence Intervals', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
output_training_path = os.path.join(SCRIPT_DIR, 'plot_training_curves_aggregate.png')
plt.savefig(output_training_path, dpi=300, bbox_inches='tight')
print(f"Training curves plot saved as '{output_training_path}'")
plt.show()

# ============================================================================
# PLOT 3: 3D Surface plots from aggregate data
# ============================================================================

print("\nGenerating 3D surface plots...")

fig = plt.figure(figsize=(20, 6))

# Prepare data for 3D surface
training_data_3d = []
for lmc_weight, data in training_data.items():
    for entry in data:
        training_data_3d.append({
            'LMC Weight': lmc_weight,
            'Epoch': entry['epoch'],
            'Training Loss': entry['train_loss_mean'],
            'Validation Loss': entry['val_loss_mean'],
            'Model LMC': entry['lmc_mean']
        })

df_3d = pd.DataFrame(training_data_3d)

# Get unique values for interpolation
unique_lmc_weights = sorted(df_3d['LMC Weight'].unique())
unique_epochs = sorted(df_3d['Epoch'].unique())

# Function to create interpolated surface
def create_surface_plot(ax, metric_name, title):
    # Create Z matrix for the metric
    Z = np.zeros((len(unique_lmc_weights), len(unique_epochs)))
    
    for i, lmc_weight in enumerate(unique_lmc_weights):
        for j, epoch in enumerate(unique_epochs):
            value = df_3d[(df_3d['LMC Weight'] == lmc_weight) & (df_3d['Epoch'] == epoch)][metric_name]
            if not value.empty:
                Z[i, j] = value.values[0]
            else:
                Z[i, j] = np.nan
    
    # Create meshgrid for original data
    X, Y = np.meshgrid(unique_epochs, unique_lmc_weights)
    
    # Create a fine grid for smooth visualization
    epochs_fine = np.linspace(unique_epochs[0], unique_epochs[-1], len(unique_epochs) * 3)
    lmc_weights_fine = np.linspace(unique_lmc_weights[0], unique_lmc_weights[-1], len(unique_lmc_weights) * 3)
    X_fine, Y_fine = np.meshgrid(epochs_fine, lmc_weights_fine)
    
    # Prepare points for interpolation
    points = np.array([X[~np.isnan(Z)].flatten(), Y[~np.isnan(Z)].flatten()]).T
    values = Z[~np.isnan(Z)].flatten()
    
    # Interpolate on fine grid
    points_fine = np.array([X_fine.flatten(), Y_fine.flatten()]).T
    Z_fine = griddata(points, values, points_fine, method='cubic')
    Z_fine = Z_fine.reshape(X_fine.shape)
    
    # Plot surface
    surf = ax.plot_surface(X_fine, Y_fine, Z_fine, cmap='viridis', 
                          alpha=0.9, edgecolor='none', shade=True, 
                          antialiased=True, rstride=1, cstride=1)
    
    # Add contour at base
    ax.contour(X_fine, Y_fine, Z_fine, zdir='z', offset=np.nanmin(Z_fine), 
              cmap='viridis', alpha=0.3, linewidths=1)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('LMC Weight', fontsize=11, fontweight='bold')
    ax.set_zlabel(title, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.view_init(elev=25, azim=120)
    ax.grid(True, alpha=0.3)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return surf

# 3D Plot 1: Training Loss
ax1 = fig.add_subplot(131, projection='3d')
create_surface_plot(ax1, 'Training Loss', 'Training Loss')

# 3D Plot 2: Validation Loss
ax2 = fig.add_subplot(132, projection='3d')
create_surface_plot(ax2, 'Validation Loss', 'Validation Loss')

# 3D Plot 3: Model LMC
ax3 = fig.add_subplot(133, projection='3d')
create_surface_plot(ax3, 'Model LMC', 'Model LMC')

fig.suptitle('3D Surface Plots: Metrics vs Epoch vs LMC Weight (Aggregate Data)', 
            fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
output_3d_path = os.path.join(SCRIPT_DIR, 'plot_3d_training_aggregate.png')
plt.savefig(output_3d_path, dpi=300, bbox_inches='tight')
print(f"3D plots saved as '{output_3d_path}'")
plt.show()

# ============================================================================
# PLOT 4: Bar charts comparing test losses for each dataset
# ============================================================================

print("\nGenerating bar chart comparison plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Calculate 95% CI (1.96 * std)
ci_factor = 1.96

# Prepare data for bar charts
x_positions = np.arange(len(lmc_weights_array))
bar_width = 0.35

# --- WikiText-2 Bar Chart ---
ax1 = axes[0]
wiki_ci = ci_factor * test_losses_wiki_std

bars1 = ax1.bar(x_positions, test_losses_wiki_mean, bar_width, 
                yerr=wiki_ci, capsize=5, alpha=0.8, color='steelblue',
                edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2, 'ecolor': 'darkred'})

ax1.set_xlabel('LMC Weight', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
ax1.set_title('WikiText-2: Test Loss by LMC Weight (with 95% CI)', fontsize=13, fontweight='bold')
ax1.set_xticks(x_positions)
ax1.set_xticklabels([f'{w:.2f}' for w in lmc_weights_array], rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val, err) in enumerate(zip(bars1, test_losses_wiki_mean, wiki_ci)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
             f'{val:.3f}±{err:.3f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Tiny-Shakespeare Bar Chart ---
ax2 = axes[1]
shakespeare_ci = ci_factor * test_losses_shakespeare_std

bars2 = ax2.bar(x_positions, test_losses_shakespeare_mean, bar_width,
                yerr=shakespeare_ci, capsize=5, alpha=0.8, color='coral',
                edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2, 'ecolor': 'darkred'})

ax2.set_xlabel('LMC Weight', fontsize=12, fontweight='bold')
ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
ax2.set_title('Tiny-Shakespeare: Test Loss by LMC Weight (with 95% CI)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_positions)
ax2.set_xticklabels([f'{w:.2f}' for w in lmc_weights_array], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val, err) in enumerate(zip(bars2, test_losses_shakespeare_mean, shakespeare_ci)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
             f'{val:.3f}±{err:.3f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.suptitle('Test Loss Comparison Across LMC Weights', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
output_bar_path = os.path.join(SCRIPT_DIR, 'plot_test_loss_bars.png')
plt.savefig(output_bar_path, dpi=300, bbox_inches='tight')
print(f"Bar chart comparison saved as '{output_bar_path}'")
plt.show()

print("="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"LMC Weight Range: {min(lmc_weights):.4f} to {max(lmc_weights):.4f}")
print(f"\nWikiText-2:")
print(f"  Test Loss Range: {np.min(test_losses_wiki_mean):.4f} to {np.max(test_losses_wiki_mean):.4f}")
print(f"  Test Loss Std: {np.mean(test_losses_wiki_std):.4f} (mean)")
print(f"  Test LMC Range: {np.min(test_lmc_wiki_mean):.4f} to {np.max(test_lmc_wiki_mean):.4f}")
print(f"  Test LMC Std: {np.mean(test_lmc_wiki_std):.4f} (mean)")
print(f"\nTiny-Shakespeare:")
print(f"  Test Loss Range: {np.min(test_losses_shakespeare_mean):.4f} to {np.max(test_losses_shakespeare_mean):.4f}")
print(f"  Test Loss Std: {np.mean(test_losses_shakespeare_std):.4f} (mean)")
print(f"  Test LMC Range: {np.min(test_lmc_shakespeare_mean):.4f} to {np.max(test_lmc_shakespeare_mean):.4f}")
print(f"  Test LMC Std: {np.mean(test_lmc_shakespeare_std):.4f} (mean)")
print("="*70)
