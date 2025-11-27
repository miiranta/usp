import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import re

# ==========================================
# Configuration
# ==========================================

# Define the base output directory relative to this script
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

# Set Seaborn Theme
# "paper" context makes elements smaller/finer. "whitegrid" gives a clean look.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['grid.alpha'] = 0.4  # Make grid lines very faint
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.9

# List of tuples: (label, folder_path)
# Configure which folders to use for plotting here.
SOURCES = [
    #('Control', os.path.join(BASE_OUTPUT_DIR, 'output_0.0')),
    ('Control', os.path.join(BASE_OUTPUT_DIR, 'output_0.0_test')),
    #('Optimized', os.path.join(BASE_OUTPUT_DIR, 'output_1.0')),
    ('Optimized', os.path.join(BASE_OUTPUT_DIR, 'output_1.0_test')),
]

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# Data Loading and Aggregation
# ==========================================

def parse_run_csv(file_path):
    """
    Parses a single run CSV file.
    Skips the summary header and reads the epoch data table.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the line number where the epoch data starts
        header_line_idx = -1
        for i, line in enumerate(lines):
            if line.startswith('Epoch,Training Loss'):
                header_line_idx = i
                break
        
        if header_line_idx == -1:
            print(f"Warning: Could not find epoch data header in {file_path}")
            return None
            
        # Read the data section using pandas
        # We pass the lines starting from the header
        from io import StringIO
        data_str = ''.join(lines[header_line_idx:])
        df = pd.read_csv(StringIO(data_str))
        return df
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def load_source_data(folder_path, label):
    """
    Reads all run CSVs in the folder (ignoring AGGREGATE),
    and returns a concatenated DataFrame with a 'Source' column.
    """
    csv_files = glob.glob(os.path.join(folder_path, 'z_loss_test_results_transformers-*.csv'))
    run_dfs = []
    
    for i, csv_file in enumerate(csv_files):
        if 'AGGREGATE' in csv_file:
            continue
            
        df = parse_run_csv(csv_file)
        if df is not None and not df.empty:
            # Add a unique run identifier for this source
            # We use the index i to distinguish runs within the same source
            df['Run_ID'] = f"{label}_Run_{i}"
            run_dfs.append(df)
    
    if not run_dfs:
        print(f"No valid run CSVs found in {folder_path}")
        return None
        
    combined = pd.concat(run_dfs)
    combined['Source'] = label
    return combined

def load_distribution_file(file_path):
    """Loads a single distribution CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading distribution file {file_path}: {e}")
        return None

# ==========================================
# Plotting Functions
# ==========================================

def load_all_data(sources):
    """Loads data from all sources into a single DataFrame."""
    all_data = []
    for label, folder_path in sources:
        print(f"Loading data for: {label}")
        df = load_source_data(folder_path, label)
        if df is not None:
            all_data.append(df)
            
    if not all_data:
        print("No data found.")
        return None

    return pd.concat(all_data)

def plot_all_metrics(master_df):
    """
    Generates plots for each metric comparing all sources using Seaborn.
    """
    if master_df is None: return

    # Identify all metrics (columns except Epoch and Source)
    # Also exclude non-numeric columns just in case
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    metrics = [c for c in numeric_cols if c not in ['Epoch']]
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        sns.lineplot(
            data=master_df,
            x='Epoch',
            y=metric,
            hue='Source',
            style='Source',
            dashes=False,
            errorbar=('se', 1.96),
            palette='muted',
            linewidth=1.5,
            alpha=0.85
        )
        
        plt.title(f'{metric} Comparison', fontweight='normal', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        sns.despine(left=True, bottom=True)
        
        safe_metric_name = re.sub(r'[^\w\s-]', '', metric).strip().replace(' ', '_')
        output_path = os.path.join(PLOTS_DIR, f'metric_{safe_metric_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")

def plot_generalization_gap(master_df):
    """Plots (Validation Loss - Training Loss) over epochs."""
    if master_df is None: return
    
    if 'Validation Loss' in master_df.columns and 'Training Loss' in master_df.columns:
        df = master_df.copy()
        df['Generalization Gap'] = df['Validation Loss'] - df['Training Loss']
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=df,
            x='Epoch',
            y='Generalization Gap',
            hue='Source',
            style='Source',
            dashes=False,
            errorbar=('se', 1.96),
            palette='muted',
            linewidth=1.5,
            alpha=0.85
        )
        plt.title('Generalization Gap (Val - Train Loss)', fontweight='normal', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Generalization Gap', fontsize=12)
        plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        sns.despine(left=True, bottom=True)
        
        output_path = os.path.join(PLOTS_DIR, 'metric_Generalization_Gap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")

def plot_scatter_phase_space(master_df, x_metric, y_metric):
    """Plots a scatter plot of two metrics against each other."""
    if master_df is None: return
    if x_metric not in master_df.columns or y_metric not in master_df.columns: return

    plt.figure(figsize=(12, 8))
    
    # Plot all points with low alpha to show density/trajectory
    sns.scatterplot(
        data=master_df,
        x=x_metric,
        y=y_metric,
        hue='Source',
        style='Source',
        alpha=0.5,
        palette='muted',
        s=40,
        edgecolor=None
    )
    
    plt.title(f'{y_metric} vs {x_metric}', fontweight='normal', fontsize=16)
    plt.xlabel(x_metric, fontsize=12)
    plt.ylabel(y_metric, fontsize=12)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    safe_x = re.sub(r'[^\w\s-]', '', x_metric).strip().replace(' ', '_')
    safe_y = re.sub(r'[^\w\s-]', '', y_metric).strip().replace(' ', '_')
    output_path = os.path.join(PLOTS_DIR, f'scatter_{safe_y}_vs_{safe_x}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")
    
def plot_final_metrics_boxplot(master_df):
    """Box plots of metrics at the final epoch of each run."""
    if master_df is None: return
    
    # Get the last epoch for each run
    # We group by Run_ID and take the last row (assuming sorted by epoch, which they usually are)
    # Or better, find max epoch per Run_ID
    final_epochs_df = master_df.loc[master_df.groupby('Run_ID')['Epoch'].idxmax()]
    
    numeric_cols = final_epochs_df.select_dtypes(include=[np.number]).columns
    metrics = [c for c in numeric_cols if c not in ['Epoch']]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=final_epochs_df,
            x='Source',
            y=metric,
            palette='muted',
            linewidth=1.2,
            fliersize=3
        )
        sns.stripplot(
            data=final_epochs_df,
            x='Source',
            y=metric,
            color='black',
            alpha=0.3,
            jitter=True,
            size=4
        )
        
        plt.title(f'Final {metric} Distribution', fontweight='normal', fontsize=16)
        plt.ylabel(metric, fontsize=12)
        sns.despine(left=True, bottom=True)
        
        safe_metric_name = re.sub(r'[^\w\s-]', '', metric).strip().replace(' ', '_')
        output_path = os.path.join(PLOTS_DIR, f'boxplot_final_{safe_metric_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")

def plot_correlation_heatmaps(master_df):
    """Plots correlation heatmaps for each source."""
    if master_df is None: return
    
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    cols_to_corr = [c for c in numeric_cols if c not in ['Epoch']]
    
    for source in master_df['Source'].unique():
        source_df = master_df[master_df['Source'] == source]
        corr = source_df[cols_to_corr].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix - {source}', fontweight='bold')
        
        safe_source = re.sub(r'[^\w\s-]', '', source).strip().replace(' ', '_')
        output_path = os.path.join(PLOTS_DIR, f'heatmap_correlation_{safe_source}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")
        
def get_epochs_for_folder(folder_path):
    dist_dir = os.path.join(folder_path, 'distributions')
    if not os.path.isdir(dist_dir):
        return []
    files = glob.glob(os.path.join(dist_dir, 'distribution_epoch_*_run_*.csv'))
    epochs = set()
    for f in files:
        match = re.search(r'distribution_epoch_(\d+|final_test_[a-z]+)_run_', os.path.basename(f))
        if match:
            ep_str = match.group(1)
            if ep_str.isdigit():
                epochs.add(int(ep_str))
    return sorted(list(epochs))

def plot_distributions_overlay(sources):
    """
    Plots overlay histograms for the first and last epochs of each source.
    """
    print("Plotting distributions...")
    
    # Plot Start (First Epoch)
    _plot_dynamic_epoch_distribution(sources, lambda epochs: epochs[0] if epochs else None, "Distribution - Start", "distribution_start")
    
    # Plot End (Last Epoch)
    _plot_dynamic_epoch_distribution(sources, lambda epochs: epochs[-1] if epochs else None, "Distribution - End", "distribution_end")

def _plot_dynamic_epoch_distribution(sources, epoch_selector, title, filename_suffix):
    plt.figure(figsize=(14, 8))
    
    # Use standard tab10 colors
    palette = sns.color_palette('muted', n_colors=len(sources))
    
    for idx, (label, folder_path) in enumerate(sources):
        epochs = get_epochs_for_folder(folder_path)
        target_epoch = epoch_selector(epochs)
        
        if target_epoch is None:
            print(f"No epochs found for {label}")
            continue
            
        dist_dir = os.path.join(folder_path, 'distributions')
        # Pattern for numeric epoch
        pattern = os.path.join(dist_dir, f'distribution_epoch_{target_epoch:03d}_run_*.csv')
        files = glob.glob(pattern)
        
        if not files:
            print(f"No distribution files found for {label} epoch {target_epoch}")
            continue
            
        color = palette[idx]
        
        # Plot each run
        for i, fpath in enumerate(files):
            df = load_distribution_file(fpath)
            if df is not None:
                df = df.sort_values('Bin_Center')
                # Shift x-axis to center at 0 (assuming [0, 1] range)
                df['Bin_Center'] = df['Bin_Center'] - 0.5
                # Plot line
                lbl = f"{label} (Ep {target_epoch})" if i == 0 else None
                
                # Use matplotlib directly for fine-grained control over fill and line
                # but use seaborn's color
                plt.plot(df['Bin_Center'], df['Probability'], color=color, alpha=0.8, linewidth=1.5, label=lbl)
                plt.fill_between(df['Bin_Center'], df['Probability'], color=color, alpha=0.2)
                
    plt.title(title, fontweight='normal', fontsize=16)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, f'{filename_suffix}_overlay.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_train_vs_val_loss(master_df):
    """Plots Training vs Validation Loss for all sources."""
    if master_df is None: return
    
    # Melt the dataframe to have a 'Loss Type' column
    # Check if columns exist first
    if 'Training Loss' not in master_df.columns or 'Validation Loss' not in master_df.columns:
        return

    df_melted = master_df.melt(
        id_vars=['Epoch', 'Source', 'Run_ID'], 
        value_vars=['Training Loss', 'Validation Loss'],
        var_name='Loss Type', 
        value_name='Loss'
    )
    
    plt.figure(figsize=(12, 8))
    
    # Define styles manually to match legend
    loss_types = df_melted['Loss Type'].unique()
    sources = df_melted['Source'].unique()
    
    # Create palette and dashes
    palette = sns.color_palette('muted', n_colors=len(sources))
    source_colors = {source: palette[i] for i, source in enumerate(sources)}
    
    dashes_map = {'Training Loss': (2, 2), 'Validation Loss': ""}
    
    sns.lineplot(
        data=df_melted,
        x='Epoch',
        y='Loss',
        hue='Source',
        style='Loss Type',
        dashes=dashes_map,
        palette='muted',
        linewidth=2,
        alpha=0.9,
        legend=False # Disable default legend
    )
    
    plt.title('Training vs Validation Loss', fontweight='normal', fontsize=16)
    plt.ylabel('Loss', fontsize=12)
    
    # Custom Legend
    legend_elements = []
    
    # Source Section (Color)
    legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Source}$'))
    for source in sources:
        legend_elements.append(Line2D([0], [0], color=source_colors[source], lw=2, label=source))
    
    # Spacer
    legend_elements.append(Line2D([0], [0], color='none', label=''))

    # Loss Type Section (Style)
    legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Loss\ Type}$'))
    for loss_type in loss_types:
        # Create a line with the correct dash pattern
        # Line2D linestyle: '-' or '--' or '-.' or ':' or (offset, (on_off_seq))
        # dashes_map has (2, 2) which is a tuple. Line2D expects linestyle string or tuple.
        # For solid line, dashes_map has "". Line2D expects '-'.
        ls = '-' if dashes_map[loss_type] == "" else (0, dashes_map[loss_type])
        legend_elements.append(Line2D([0], [0], color='gray', linestyle=ls, lw=2, label=loss_type))

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'comparison_train_vs_val_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")
    
def plot_test_loss_barplot(master_df):
    """Bar plot comparing final Test Losses on different datasets."""
    if master_df is None: return
    
    # Get final epoch data
    final_epochs_df = master_df.loc[master_df.groupby('Run_ID')['Epoch'].idxmax()]
    
    # Check if columns exist
    cols = ['Test Loss Wiki', 'Test Loss Shakespeare']
    available_cols = [c for c in cols if c in final_epochs_df.columns]
    
    if not available_cols: return
    
    df_melted = final_epochs_df.melt(
        id_vars=['Source', 'Run_ID'],
        value_vars=available_cols,
        var_name='Dataset',
        value_name='Loss'
    )
    
    # Clean up dataset names
    df_melted['Dataset'] = df_melted['Dataset'].str.replace('Test Loss ', '')
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_melted,
        x='Source',
        y='Loss',
        hue='Dataset',
        palette='pastel',
        edgecolor='.2'
    )
    
    # Add numeric labels
    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height) and height != 0:
            ax.annotate(f'{height:.4f}', 
                        (p.get_x() + p.get_width() / 2., height / 2), 
                        ha='center', va='center', 
                        xytext=(0, 0), 
                        textcoords='offset points',
                        fontsize=10,
                        color='black')
    
    plt.title('Final Test Loss Comparison', fontweight='normal', fontsize=16)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'barplot_test_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_3d_trajectory(master_df):
    """3D Trajectory of Optimization: LMC vs Entropy vs Val Loss."""
    if master_df is None: return
    
    required = ['Model LMC', 'Weights Entropy', 'Validation Loss']
    if not all(c in master_df.columns for c in required): return
    
    # Aggregate mean per epoch per source for the trajectory line
    agg_df = master_df.groupby(['Source', 'Epoch'])[required].mean().reset_index()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    sources = agg_df['Source'].unique()
    palette = sns.color_palette('muted', n_colors=len(sources))
    
    for i, source in enumerate(sources):
        subset = agg_df[agg_df['Source'] == source].sort_values('Epoch')
        
        # Plot line
        ax.plot(
            subset['Model LMC'], 
            subset['Weights Entropy'], 
            subset['Validation Loss'], 
            label=source,
            color=palette[i],
            linewidth=2,
            alpha=0.8
        )
        
        # Add start and end markers
        ax.scatter(
            subset.iloc[0]['Model LMC'], subset.iloc[0]['Weights Entropy'], subset.iloc[0]['Validation Loss'],
            color=palette[i], marker='o', s=50, alpha=0.6
        )
        ax.scatter(
            subset.iloc[-1]['Model LMC'], subset.iloc[-1]['Weights Entropy'], subset.iloc[-1]['Validation Loss'],
            color=palette[i], marker='^', s=80, alpha=1.0
        )

    ax.set_xlabel('Model LMC')
    ax.set_ylabel('Weights Entropy')
    ax.set_zlabel('Validation Loss')
    ax.set_title('Optimization Trajectory (3D)', fontweight='normal', fontsize=16)
    ax.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Improve 3D look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    output_path = os.path.join(PLOTS_DIR, '3d_trajectory_optimization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_complexity_performance_path(master_df):
    """2D Path of Complexity vs Performance."""
    if master_df is None: return
    
    if 'Model LMC' not in master_df.columns or 'Validation Loss' not in master_df.columns: return

    # Aggregate mean per epoch per source
    agg_df = master_df.groupby(['Source', 'Epoch'])[['Model LMC', 'Validation Loss']].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=agg_df,
        x='Model LMC',
        y='Validation Loss',
        hue='Source',
        sort=False, # Keep epoch order
        palette='muted',
        linewidth=2,
        alpha=0.8,
        marker='o',
        markevery=10 # Mark every 10th epoch to show direction/speed
    )
    
    plt.title('Complexity-Performance Path', fontweight='normal', fontsize=16)
    plt.xlabel('Model LMC (Complexity)', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'path_complexity_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_global_best_metrics_barplot(master_df):
    """
    Bar plot comparing the BEST value found (per run) for each metric for each source.
    Aggregates across runs (mean +/- error).
    Min for losses/errors/entropy/LMC, Max for accuracy.
    """
    if master_df is None: return

    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    metrics = [c for c in numeric_cols if c not in ['Epoch']]

    for metric in metrics:
        # Determine if we want min or max
        # Heuristic: if "Accuracy" in name -> Max, else Min
        is_accuracy = 'Accuracy' in metric
        
        # Calculate best value per run
        # Group by Source AND Run_ID
        # We want the min (or max) of the metric for each run
        if is_accuracy:
            best_per_run = master_df.groupby(['Source', 'Run_ID'])[metric].max().reset_index()
        else:
            best_per_run = master_df.groupby(['Source', 'Run_ID'])[metric].min().reset_index()
            
        best_per_run.rename(columns={metric: 'Best Value'}, inplace=True)
        
        plt.figure(figsize=(10, 6))
        
        # Use hue='Source' to ensure consistent coloring with other plots
        # dodge=False because x and hue are the same
        ax = sns.barplot(
            data=best_per_run,
            x='Source',
            y='Best Value',
            hue='Source',
            palette='pastel',
            edgecolor='.2',
            errorbar=('se', 1.96), # Match lineplot error bars
            dodge=False
        )
        
        # Add numeric labels
        for p in ax.patches:
            height = p.get_height()
            if np.isfinite(height) and height != 0:
                ax.annotate(f'{height:.4f}', 
                            (p.get_x() + p.get_width() / 2., height / 2), 
                            ha='center', va='center', 
                            xytext=(0, 0), 
                            textcoords='offset points',
                            fontsize=10,
                            color='black')
        
        direction = "Max" if is_accuracy else "Min"
        plt.title(f'Best {metric} per Run ({direction})', fontweight='normal', fontsize=16)
        plt.ylabel(f'Best {metric}', fontsize=12)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        sns.despine(left=True, bottom=True)
        
        safe_metric_name = re.sub(r'[^\w\s-]', '', metric).strip().replace(' ', '_')
        output_path = os.path.join(PLOTS_DIR, f'barplot_best_{safe_metric_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")

def plot_final_metrics_barplot(master_df):
    """
    Bar plot comparing the FINAL value (last epoch) for each metric for each source.
    Aggregates across runs (mean +/- error).
    """
    if master_df is None: return

    # Get final epoch data
    final_epochs_df = master_df.loc[master_df.groupby('Run_ID')['Epoch'].idxmax()]

    numeric_cols = final_epochs_df.select_dtypes(include=[np.number]).columns
    metrics = [c for c in numeric_cols if c not in ['Epoch']]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Use hue='Source' to ensure consistent coloring with other plots
        # dodge=False because x and hue are the same
        ax = sns.barplot(
            data=final_epochs_df,
            x='Source',
            y=metric,
            hue='Source',
            palette='pastel', # Use pastel as requested
            edgecolor='.2',
            errorbar=('se', 1.96), # Match lineplot error bars
            dodge=False
        )
        
        # Add numeric labels
        for p in ax.patches:
            height = p.get_height()
            if np.isfinite(height) and height != 0:
                ax.annotate(f'{height:.4f}', 
                            (p.get_x() + p.get_width() / 2., height / 2), 
                            ha='center', va='center', 
                            xytext=(0, 0), 
                            textcoords='offset points',
                            fontsize=10,
                            color='black')
        
        plt.title(f'Final {metric} per Run', fontweight='normal', fontsize=16)
        plt.ylabel(f'Final {metric}', fontsize=12)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        sns.despine(left=True, bottom=True)
        
        safe_metric_name = re.sub(r'[^\w\s-]', '', metric).strip().replace(' ', '_')
        output_path = os.path.join(PLOTS_DIR, f'barplot_final_{safe_metric_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")

def plot_complexity_metrics_comparison(master_df):
    """Plots Model LMC, Weights Entropy, and Weights Disequilibrium comparison with different scales."""
    if master_df is None: return
    
    cols = ['Model LMC', 'Weights Entropy', 'Weights Disequilibrium']
    available_cols = [c for c in cols if c in master_df.columns]
    
    if len(available_cols) < 2:
        return

    # Create figure and first axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Colors for metrics - use a palette that is distinct
    metric_colors = sns.color_palette('deep', n_colors=len(available_cols))
    
    # Line styles for sources
    sources = master_df['Source'].unique()
    # Map sources to line styles
    styles = ['-', '--', ':', '-.']
    source_styles = {source: styles[i % len(styles)] for i, source in enumerate(sources)}
    
    axes = [ax1]
    # Create additional axes
    for i in range(1, len(available_cols)):
        axes.append(ax1.twinx())
        
    # Offset the third+ axis
    if len(axes) > 2:
        for i in range(2, len(axes)):
            # Offset the right spine of the new axes
            axes[i].spines['right'].set_position(('outward', 60 * (i-1)))
            
    for i, metric in enumerate(available_cols):
        ax = axes[i]
        color = metric_colors[i]
        
        for source in sources:
            subset = master_df[master_df['Source'] == source]
            # Group by epoch to get mean (if multiple runs)
            subset_mean = subset.groupby('Epoch')[metric].mean()
            
            # Plot
            ax.plot(subset_mean.index, subset_mean.values, 
                    color=color, linestyle=source_styles.get(source, '-'),
                    linewidth=2, alpha=0.8)
            
        ax.set_ylabel(metric, color=color, fontsize=12)
        ax.tick_params(axis='y', labelcolor=color)
        
        # Color the spine to match
        if i == 0:
            ax.spines['left'].set_color(color)
            ax.spines['left'].set_linewidth(2)
        else:
            ax.spines['right'].set_color(color)
            ax.spines['right'].set_linewidth(2)

    ax1.set_xlabel('Epoch', fontsize=12)
    plt.title('Complexity Metrics Comparison', fontweight='normal', fontsize=16)
    
    # Custom Legend
    legend_elements = []
    
    # Metric Section
    # legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Metric}$'))
    for i, metric in enumerate(available_cols):
        legend_elements.append(Line2D([0], [0], color=metric_colors[i], lw=2, label=metric))
    
    # Spacer
    legend_elements.append(Line2D([0], [0], color='none', label=''))

    # Source Section
    legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Source}$'))
    for source in sources:
        legend_elements.append(Line2D([0], [0], color='gray', linestyle=source_styles.get(source, '-'), lw=2, label=source))

    # Combined legend
    ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05 + (0.05 * (len(axes)-1)), 1), loc='upper left', frameon=False)
    
    # Remove top spine
    ax1.spines['top'].set_visible(False)
    for ax in axes[1:]:
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False) # Hide left spine of secondary axes
        
    output_path = os.path.join(PLOTS_DIR, 'comparison_complexity_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_adaptive_mechanism(master_df):
    """
    Plots LMC Weight and Validation Error Slope to visualize the adaptive mechanism.
    """
    if master_df is None: return
    
    required = ['LMC Weight', 'Val Error Slope']
    if not all(c in master_df.columns for c in required): return

    # Create figure and first axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    sources = master_df['Source'].unique()
    
    # Define colors for metrics
    metric_colors = sns.color_palette('deep', n_colors=2)
    color_weight = metric_colors[0]
    color_slope = metric_colors[1]
    
    # Define styles for sources
    styles = ['-', '--', ':', '-.']
    source_styles = {source: styles[i % len(styles)] for i, source in enumerate(sources)}
    
    # Plot LMC Weight on left axis (ax1)
    for source in sources:
        subset = master_df[master_df['Source'] == source]
        subset_mean = subset.groupby('Epoch')['LMC Weight'].mean()
        ax1.plot(subset_mean.index, subset_mean.values, 
                 color=color_weight, linestyle=source_styles[source], linewidth=2, label=f"{source}")
                 
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('LMC Weight', color=color_weight, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_weight)
    ax1.spines['left'].set_color(color_weight)
    ax1.spines['left'].set_linewidth(2)
    
    # Plot Slope on right axis (ax2)
    ax2 = ax1.twinx()
    for source in sources:
        subset = master_df[master_df['Source'] == source]
        subset_mean = subset.groupby('Epoch')['Val Error Slope'].mean()
        ax2.plot(subset_mean.index, subset_mean.values, 
                 color=color_slope, linestyle=source_styles[source], linewidth=2, alpha=0.8)
                 
    ax2.set_ylabel('Val Error Slope', color=color_slope, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_slope)
    ax2.spines['right'].set_color(color_slope)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['right'].set_position(('outward', 10)) # Offset slightly
    
    # Add zero line for slope
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)

    plt.title('Adaptive Mechanism: LMC Weight vs Val Error Slope', fontweight='normal', fontsize=16)
    
    # Custom Legend
    legend_elements = []
    
    # Metric Section
    # legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Metric}$'))
    legend_elements.append(Line2D([0], [0], color=color_weight, lw=2, label='LMC Weight'))
    legend_elements.append(Line2D([0], [0], color=color_slope, lw=2, label='Val Error Slope'))
    
    # Spacer
    legend_elements.append(Line2D([0], [0], color='none', label=''))

    # Source Section
    legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Source}$'))
    for source in sources:
        legend_elements.append(Line2D([0], [0], color='gray', linestyle=source_styles[source], lw=2, label=source))

    ax1.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), loc='upper left', frameon=False)
    
    # Remove top spine
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    output_path = os.path.join(PLOTS_DIR, 'mechanism_adaptive_control.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_adaptive_mechanism_faceted(master_df):
    """
    Plots LMC Weight and Validation Error Slope in faceted subplots by Source.
    """
    if master_df is None: return
    
    required = ['LMC Weight', 'Val Error Slope']
    if not all(c in master_df.columns for c in required): return

    sources = master_df['Source'].unique()
    n_sources = len(sources)
    
    # Create subplots - 1 row, n_sources columns
    fig, axes = plt.subplots(1, n_sources, figsize=(6 * n_sources, 6), sharex=True)
    if n_sources == 1: axes = [axes]
    
    # Define colors for metrics
    metric_colors = sns.color_palette('deep', n_colors=2)
    color_weight = metric_colors[0]
    color_slope = metric_colors[1]
    
    for i, source in enumerate(sources):
        ax1 = axes[i]
        subset = master_df[master_df['Source'] == source]
        
        # Group by Epoch to get mean (if multiple runs)
        subset_mean = subset.groupby('Epoch')[required].mean()
        
        # Plot LMC Weight (Left Axis)
        ax1.plot(subset_mean.index, subset_mean['LMC Weight'], 
                 color=color_weight, linewidth=2, label='LMC Weight')
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('LMC Weight', color=color_weight, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color_weight)
        ax1.set_title(f'{source}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot Slope (Right Axis)
        ax2 = ax1.twinx()
        ax2.plot(subset_mean.index, subset_mean['Val Error Slope'], 
                 color=color_slope, linewidth=2, linestyle='--', alpha=0.8, label='Val Error Slope')
                 
        ax2.set_ylabel('Val Error Slope', color=color_slope, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color_slope)
        
        # Add zero line for slope
        ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        # Spines
        ax1.spines['left'].set_color(color_weight)
        ax1.spines['left'].set_linewidth(2)
        ax2.spines['right'].set_color(color_slope)
        ax2.spines['right'].set_linewidth(2)
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    # Common title
    fig.suptitle('Adaptive Mechanism by Source', fontsize=16)
    fig.subplots_adjust(top=0.85, wspace=0.5)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=color_weight, lw=2, label='LMC Weight'),
        Line2D([0], [0], color=color_slope, lw=2, linestyle='--', label='Val Error Slope')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False)

    output_path = os.path.join(PLOTS_DIR, 'mechanism_adaptive_control_faceted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_complexity_phase_space(master_df):
    """
    Plots Weights Entropy vs Weights Disequilibrium (The two components of LMC).
    """
    if master_df is None: return
    
    required = ['Weights Entropy', 'Weights Disequilibrium']
    if not all(c in master_df.columns for c in required): return

    plt.figure(figsize=(12, 8))
    
    # Aggregate mean per epoch per source
    agg_df = master_df.groupby(['Source', 'Epoch'])[required].mean().reset_index()
    
    sns.lineplot(
        data=agg_df,
        x='Weights Entropy',
        y='Weights Disequilibrium',
        hue='Source',
        sort=False,
        palette='muted',
        linewidth=2,
        alpha=0.8,
        marker='o',
        markevery=5
    )
    
    # Add start/end annotations
    for source in agg_df['Source'].unique():
        subset = agg_df[agg_df['Source'] == source]
        if not subset.empty:
            # Start
            plt.text(subset.iloc[0]['Weights Entropy'], subset.iloc[0]['Weights Disequilibrium'], 'Start', 
                     fontsize=9, ha='right', va='bottom')
            # End
            plt.text(subset.iloc[-1]['Weights Entropy'], subset.iloc[-1]['Weights Disequilibrium'], 'End', 
                     fontsize=9, ha='left', va='top')

    plt.title('Complexity Phase Space (Entropy vs Disequilibrium)', fontweight='normal', fontsize=16)
    plt.xlabel('Weights Entropy (H)', fontsize=12)
    plt.ylabel('Weights Disequilibrium (D)', fontsize=12)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'phase_space_entropy_diseq.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_overfitting_vs_complexity(master_df):
    """
    Plots Generalization Gap vs Model LMC to see if complexity correlates with overfitting.
    """
    if master_df is None: return
    
    if 'Validation Loss' not in master_df.columns or 'Training Loss' not in master_df.columns or 'Model LMC' not in master_df.columns:
        return
        
    df = master_df.copy()
    df['Generalization Gap'] = df['Validation Loss'] - df['Training Loss']
    
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(
        data=df,
        x='Model LMC',
        y='Generalization Gap',
        hue='Source',
        style='Source',
        alpha=0.6,
        palette='muted',
        s=50
    )
    
    # Add trend lines
    sources = df['Source'].unique()
    palette = sns.color_palette('muted', n_colors=len(sources))
    
    for i, source in enumerate(sources):
        subset = df[df['Source'] == source]
        if len(subset) > 1:
            # Simple linear regression for trend
            z = np.polyfit(subset['Model LMC'], subset['Generalization Gap'], 1)
            p = np.poly1d(z)
            
            x_range = np.linspace(subset['Model LMC'].min(), subset['Model LMC'].max(), 100)
            plt.plot(x_range, p(x_range), color=palette[i], linestyle='--', alpha=0.5, linewidth=1.5)

    plt.title('Overfitting vs Complexity (Gen Gap vs LMC)', fontweight='normal', fontsize=16)
    plt.xlabel('Model LMC', fontsize=12)
    plt.ylabel('Generalization Gap (Val - Train)', fontsize=12)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'scatter_overfitting_vs_complexity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_3d_trajectory_improved(master_df):
    """
    Improved 3D Trajectory: Training Loss vs Validation Loss vs Weights Entropy.
    Avoids LMC vs Entropy correlation.
    """
    if master_df is None: return
    
    # Check available columns
    x_col = 'Training Loss'
    y_col = 'Validation Loss'
    z_col = 'Weights Entropy'
    
    required = [x_col, y_col, z_col]
    if not all(c in master_df.columns for c in required):
        print(f"Missing columns for improved 3D plot. Required: {required}")
        return
    
    # Aggregate mean per epoch per source
    agg_df = master_df.groupby(['Source', 'Epoch'])[required].mean().reset_index()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    sources = agg_df['Source'].unique()
    palette = sns.color_palette('muted', n_colors=len(sources))
    
    for i, source in enumerate(sources):
        subset = agg_df[agg_df['Source'] == source].sort_values('Epoch')
        
        ax.plot(
            subset[x_col], 
            subset[y_col], 
            subset[z_col], 
            label=source,
            color=palette[i],
            linewidth=2,
            alpha=0.8
        )
        
        # Start
        ax.scatter(
            subset.iloc[0][x_col], subset.iloc[0][y_col], subset.iloc[0][z_col],
            color=palette[i], marker='o', s=50, alpha=0.6
        )
        # End
        ax.scatter(
            subset.iloc[-1][x_col], subset.iloc[-1][y_col], subset.iloc[-1][z_col],
            color=palette[i], marker='^', s=80, alpha=1.0
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title('Optimization Landscape Trajectory (Improved)', fontweight='normal', fontsize=16)
    ax.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    output_path = os.path.join(PLOTS_DIR, '3d_trajectory_improved.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_train_val_test_loss_comparison(master_df):
    """Plots Training vs Validation vs Test Loss for all sources."""
    if master_df is None: return
    
    # Identify loss columns
    loss_cols = ['Training Loss', 'Validation Loss']
    # Find test loss columns dynamically
    test_cols = [c for c in master_df.columns if 'Test Loss' in c]
    
    all_loss_cols = loss_cols + test_cols
    available_cols = [c for c in all_loss_cols if c in master_df.columns]
    
    if len(available_cols) < 2:
        return

    df_melted = master_df.melt(
        id_vars=['Epoch', 'Source', 'Run_ID'], 
        value_vars=available_cols,
        var_name='Loss Type', 
        value_name='Loss'
    )
    
    plt.figure(figsize=(12, 8))
    
    # Define styles manually to match legend
    loss_types = df_melted['Loss Type'].unique()
    sources = df_melted['Source'].unique()
    
    # Create palette
    palette = sns.color_palette('muted', n_colors=len(sources))
    source_colors = {source: palette[i] for i, source in enumerate(sources)}
    
    # Define line styles for loss types
    # We need enough styles for all loss types
    styles = ['-', '--', ':', '-.']
    loss_styles = {lt: styles[i % len(styles)] for i, lt in enumerate(loss_types)}
    
    # Map styles to dashes for seaborn if needed, or just let seaborn handle it but we need to know what it did.
    # Better to pass the dict to style_order or dashes?
    # sns.lineplot uses 'dashes' parameter which takes a list or dict.
    # But 'dashes' in seaborn expects boolean or tuple.
    # For solid line, dashes_map has "". Line2D expects '-'.
    mpl_to_sns_dashes = {
        '-': "",
        '--': (2, 2),
        ':': (1, 1),
        '-.': (3, 1, 1, 1)
    }
    sns_dashes = {lt: mpl_to_sns_dashes[loss_styles[lt]] for lt in loss_types}

    sns.lineplot(
        data=df_melted,
        x='Epoch',
        y='Loss',
        hue='Source',
        style='Loss Type',
        dashes=sns_dashes,
        palette='muted',
        linewidth=1.5,
        alpha=0.9,
        legend=False
    )
    
    plt.title('Training vs Validation vs Test Loss', fontweight='normal', fontsize=16)
    plt.ylabel('Loss', fontsize=12)
    
    # Custom Legend
    legend_elements = []
    
    # Source Section (Color)
    legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Source}$'))
    for source in sources:
        legend_elements.append(Line2D([0], [0], color=source_colors[source], lw=2, label=source))
    
    # Spacer
    legend_elements.append(Line2D([0], [0], color='none', label=''))

    # Loss Type Section (Style)
    # legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Loss\ Type}$'))
    for loss_type in loss_types:
        legend_elements.append(Line2D([0], [0], color='gray', linestyle=loss_styles[loss_type], lw=2, label=loss_type))

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'comparison_train_val_test_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_train_val_test_loss_faceted(master_df):
    """Plots Training vs Validation vs Test Loss in faceted subplots."""
    if master_df is None: return
    
    # Identify loss columns
    loss_cols = ['Training Loss', 'Validation Loss']
    test_cols = [c for c in master_df.columns if 'Test Loss' in c]
    all_loss_cols = loss_cols + test_cols
    available_cols = [c for c in all_loss_cols if c in master_df.columns]
    
    if len(available_cols) < 2: return

    df_melted = master_df.melt(
        id_vars=['Epoch', 'Source', 'Run_ID'], 
        value_vars=available_cols,
        var_name='Loss Type', 
        value_name='Loss'
    )
    
    # Facet by Loss Type
    g = sns.FacetGrid(df_melted, col="Loss Type", hue="Source", height=5, aspect=1.2, sharey=False)
    g.map(sns.lineplot, "Epoch", "Loss", errorbar=('se', 1.96))
    g.add_legend()
    g.fig.suptitle('Loss Comparison by Type', fontsize=16)
    g.fig.subplots_adjust(top=0.85)
    
    output_path = os.path.join(PLOTS_DIR, 'comparison_train_val_test_loss_faceted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_complexity_metrics_faceted(master_df):
    """Plots complexity metrics in faceted subplots."""
    if master_df is None: return
    
    cols = ['Model LMC', 'Weights Entropy', 'Weights Disequilibrium']
    available_cols = [c for c in cols if c in master_df.columns]
    
    if not available_cols: return

    df_melted = master_df.melt(
        id_vars=['Epoch', 'Source', 'Run_ID'], 
        value_vars=available_cols,
        var_name='Metric', 
        value_name='Value'
    )
    
    g = sns.FacetGrid(df_melted, col="Metric", hue="Source", height=5, aspect=1.2, sharey=False)
    g.map(sns.lineplot, "Epoch", "Value", errorbar=('se', 1.96))
    g.add_legend()
    g.fig.suptitle('Complexity Metrics Comparison', fontsize=16)
    g.fig.subplots_adjust(top=0.85)
    
    output_path = os.path.join(PLOTS_DIR, 'comparison_complexity_metrics_faceted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    print(f"Starting plot generation...")
    print(f"Sources: {[s[0] for s in SOURCES]}")
    
    if not SOURCES:
        print("No sources configured. Exiting.")
        return

    # Load all data once
    master_df = load_all_data(SOURCES)

    # 1. Plot Metrics
    plot_all_metrics(master_df)
    
    # 2. Plot Distributions
    plot_distributions_overlay(SOURCES)
    
    # 3. New Plots
    plot_generalization_gap(master_df)
    plot_scatter_phase_space(master_df, 'Model LMC', 'Validation Loss')
    plot_scatter_phase_space(master_df, 'Weights Entropy', 'Validation Loss')
    plot_scatter_phase_space(master_df, 'Weights Entropy', 'Model LMC')
    # plot_final_metrics_boxplot(master_df)
    plot_correlation_heatmaps(master_df)
    plot_train_vs_val_loss(master_df)
    plot_test_loss_barplot(master_df)
    plot_3d_trajectory(master_df)
    plot_complexity_performance_path(master_df)
    plot_global_best_metrics_barplot(master_df)
    plot_final_metrics_barplot(master_df)
    plot_3d_trajectory_improved(master_df)
    plot_train_val_test_loss_comparison(master_df)
    plot_complexity_metrics_comparison(master_df)
    plot_adaptive_mechanism(master_df)
    plot_complexity_phase_space(master_df)
    plot_overfitting_vs_complexity(master_df)
    
    # Faceted Plots
    plot_train_val_test_loss_faceted(master_df)
    plot_complexity_metrics_faceted(master_df)
    plot_adaptive_mechanism_faceted(master_df)
    
    print("Done.")

if __name__ == "__main__":
    main()
