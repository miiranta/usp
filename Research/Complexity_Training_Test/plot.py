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
# Switching to "talk" for larger, more readable plots as requested.
sns.set_theme(style="whitegrid", context="talk", font_scale=1.2) 
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 2.0 # Thicker axes
plt.rcParams['lines.linewidth'] = 3.5 # Thicker lines by default
plt.rcParams['font.weight'] = 'bold' # Make text bolder
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Standard Colors
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN = '#2ca02c'
RED = '#d62728'

# Consistent Palette for Sources
SOURCE_PALETTE = {'Control': BLUE, 'Optimized': ORANGE}
# Fallback list if dict doesn't work in some seaborn versions (though it should for hue)
SOURCE_COLORS_LIST = [BLUE, ORANGE]

# List of tuples: (label, folder_path)
# Configure which folders to use for plotting here.
SOURCES = [
    ('Control', os.path.join(BASE_OUTPUT_DIR, 'just_another_output_0.0')),
    #('Control', os.path.join(BASE_OUTPUT_DIR, 'output_0.0_test')),
    ('Optimized', os.path.join(BASE_OUTPUT_DIR, 'just_another_output_1.0')),
    #('Optimized', os.path.join(BASE_OUTPUT_DIR, 'output_1.0_test')),
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

    combined = pd.concat(all_data)
    
    # Rename metrics as requested
    combined.rename(columns={
        'Model LMC': 'LMC Complexity',
        'Weights Entropy': 'Shannon Entropy',
        'Weights Disequilibrium': 'Disequilibrium',
        'LMC Weight': 'λ',
        'Val Error Slope': '∆Lval',
        'Weight': 'λ',
        'Slope': '∆Lval'
    }, inplace=True)
    
    return combined

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
    
    hue_order = ['Control', 'Optimized']
    
    # Calculate stats for annotation including 95% CI
    stats = df_melted.groupby(['Dataset', 'Source'])['Loss'].agg(['mean', 'std', 'sem']).reset_index()
    stats['ci'] = stats['sem'] * 1.96  # 95% confidence interval
    
    ax = sns.barplot(
        data=df_melted,
        x='Dataset',
        y='Loss',
        hue='Source',
        hue_order=hue_order,
        palette=SOURCE_PALETTE, # Use consistent palette
        edgecolor='.2',
        errorbar=('se', 1.96),  # 95% confidence interval
        capsize=.1,
        err_kws={'linewidth': 3, 'color': 'black'}
    )
    
    # Add numeric labels with 95% CI
    for i, container in enumerate(ax.containers):
        if i >= len(hue_order): break
        source_label = hue_order[i]
        
        for j, bar in enumerate(container):
            # Get dataset label from x-axis (assuming sorted order if not available, but try getting from ax)
            # Seaborn sorts x-axis alphabetically by default for strings
            datasets = sorted(df_melted['Dataset'].unique())
            if j < len(datasets):
                dataset_label = datasets[j]
                
                try:
                    stat_row = stats[(stats['Dataset'] == dataset_label) & (stats['Source'] == source_label)]
                    if not stat_row.empty:
                        mean_val = stat_row['mean'].values[0]
                        ci_val = stat_row['ci'].values[0]
                        
                        height = bar.get_height()
                        if np.isfinite(height) and height != 0:
                            label = f'{mean_val:.4f}\n±{ci_val:.4f}'
                            
                            ax.annotate(label, 
                                        (bar.get_x() + bar.get_width() / 2., height / 2), 
                                        ha='center', va='center', 
                                        xytext=(0, 0), 
                                        textcoords='offset points',
                                        fontsize=14,
                                        color='black',
                                        fontweight='bold')
                except (KeyError, IndexError):
                    pass
    
    # plt.title('Final Test Loss Comparison', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontsize=18, fontweight='bold')
    plt.xlabel('Dataset', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=16, title_fontsize=18)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'barplot_test_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_3d_trajectory(master_df):
    """3D Trajectory of Optimization: LMC vs Shannon Entropy vs Val Loss."""
    if master_df is None: return
    
    required = ['LMC Complexity', 'Shannon Entropy', 'Validation Loss']
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
            subset['LMC Complexity'], 
            subset['Shannon Entropy'], 
            subset['Validation Loss'], 
            label=source,
            color=palette[i],
            linewidth=2,
            alpha=0.8
        )
        
        # Add start and end markers
        ax.scatter(
            subset.iloc[0]['LMC Complexity'], subset.iloc[0]['Shannon Entropy'], subset.iloc[0]['Validation Loss'],
            color=palette[i], marker='o', s=50, alpha=0.6
        )
        ax.scatter(
            subset.iloc[-1]['LMC Complexity'], subset.iloc[-1]['Shannon Entropy'], subset.iloc[-1]['Validation Loss'],
            color=palette[i], marker='^', s=80, alpha=1.0
        )

    ax.set_xlabel('LMC Complexity')
    ax.set_ylabel('Shannon Entropy')
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
    
    if 'LMC Complexity' not in master_df.columns or 'Validation Loss' not in master_df.columns: return

    # Aggregate mean per epoch per source
    agg_df = master_df.groupby(['Source', 'Epoch'])[['LMC Complexity', 'Validation Loss']].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=agg_df,
        x='LMC Complexity',
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
    plt.xlabel('LMC Complexity', fontsize=12)
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
        
        # Calculate stats for annotations
        stats = best_per_run.groupby(['Source'])['Best Value'].agg(['mean', 'sem']).reset_index()
        stats['ci'] = stats['sem'] * 1.96
        
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
            dodge=False,
            capsize=.1,
            err_kws={'linewidth': 2, 'color': 'black'}
        )
        
        # Add numeric labels
        for p in ax.patches:
            height = p.get_height()
            if np.isfinite(height) and height != 0:
                # Find corresponding CI
                closest_idx = (stats['mean'] - height).abs().idxmin()
                ci = stats.loc[closest_idx, 'ci']
                
                ax.annotate(f'{height:.4f}\n±{ci:.4f}', 
                            (p.get_x() + p.get_width() / 2., height / 2), 
                            ha='center', va='center', 
                            xytext=(0, 0), 
                            textcoords='offset points',
                            fontsize=10,
                            color='black',
                            fontweight='bold')
        
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
        
        # Calculate stats for annotations
        stats = final_epochs_df.groupby(['Source'])[metric].agg(['mean', 'sem']).reset_index()
        stats['ci'] = stats['sem'] * 1.96
        
        # Use hue='Source' to ensure consistent coloring with other plots
        # dodge=False because x and hue are the same
        ax = sns.barplot(
            data=final_epochs_df,
            x='Source',
            y=metric,
            hue='Source',
            palette=SOURCE_PALETTE, # Use consistent palette
            edgecolor='.2',
            errorbar=('se', 1.96), # Match lineplot error bars
            dodge=False,
            capsize=.1,
            err_kws={'linewidth': 3, 'color': 'black'}
        )
        
        # Add numeric labels
        for p in ax.patches:
            height = p.get_height()
            if np.isfinite(height) and height != 0:
                # Find corresponding CI
                closest_idx = (stats['mean'] - height).abs().idxmin()
                ci = stats.loc[closest_idx, 'ci']
                
                ax.annotate(f'{height:.4f}\n±{ci:.4f}', 
                            (p.get_x() + p.get_width() / 2., height / 2), 
                            ha='center', va='center', 
                            xytext=(0, 0), 
                            textcoords='offset points',
                            fontsize=16, # Increased font size
                            color='black',
                            fontweight='bold')
        
        # plt.title(f'Final {metric} per Run', fontweight='bold', fontsize=22)
        plt.ylabel(f'Final {metric}', fontsize=18, fontweight='bold')
        plt.xlabel('Source', fontsize=18, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=16)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        sns.despine(left=True, bottom=True)
        
        safe_metric_name = re.sub(r'[^\w\s-]', '', metric).strip().replace(' ', '_')
        output_path = os.path.join(PLOTS_DIR, f'barplot_final_{safe_metric_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")

def plot_complexity_metrics_comparison(master_df):
    """Plots LMC Complexity, Shannon Entropy, and Disequilibrium comparison with different scales."""
    if master_df is None: return
    
    cols = ['LMC Complexity', 'Shannon Entropy', 'Disequilibrium']
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
    Plots λ and ∆Lval to visualize the adaptive mechanism.
    """
    if master_df is None: return
    
    required = ['λ', '∆Lval']
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
    
    # Plot λ on left axis (ax1)
    for source in sources:
        subset = master_df[master_df['Source'] == source]
        
        # Calculate mean and CI
        grouped = subset.groupby('Epoch')['λ']
        mean = grouped.mean()
        sem = grouped.sem()
        ci = sem * 1.96
        
        ax1.plot(mean.index, mean.values, 
                 color=color_weight, linestyle=source_styles[source], linewidth=2, label=f"{source}")
        ax1.fill_between(mean.index, mean - ci, mean + ci, color=color_weight, alpha=0.2)
                 
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('λ', color=color_weight, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_weight)
    ax1.spines['left'].set_color(color_weight)
    ax1.spines['left'].set_linewidth(2)
    
    # Plot Slope on right axis (ax2)
    ax2 = ax1.twinx()
    for source in sources:
        subset = master_df[master_df['Source'] == source]
        
        # Calculate mean and CI
        grouped = subset.groupby('Epoch')['∆Lval']
        mean = grouped.mean()
        sem = grouped.sem()
        ci = sem * 1.96
        
        ax2.plot(mean.index, mean.values, 
                 color=color_slope, linestyle=source_styles[source], linewidth=2, alpha=0.8)
        ax2.fill_between(mean.index, mean - ci, mean + ci, color=color_slope, alpha=0.2)
                 
    ax2.set_ylabel('∆Lval', color=color_slope, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_slope)
    ax2.spines['right'].set_color(color_slope)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['right'].set_position(('outward', 10)) # Offset slightly
    
    # Add zero line for slope
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)

    plt.title('Adaptive Mechanism: λ vs ∆Lval', fontweight='normal', fontsize=16)
    
    # Custom Legend
    legend_elements = []
    
    # Metric Section
    # legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Metric}$'))
    legend_elements.append(Line2D([0], [0], color=color_weight, lw=2, label='λ'))
    legend_elements.append(Line2D([0], [0], color=color_slope, lw=2, label='∆Lval'))
    
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
    Plots λ and ∆Lval in faceted subplots by Source.
    """
    if master_df is None: return
    
    required = ['λ', '∆Lval']
    if not all(c in master_df.columns for c in required): return

    sources = master_df['Source'].unique()
    n_sources = len(sources)
    
    # Create subplots - n_sources rows, 1 column (stacked vertically)
    # Standardized to 14 inch width, height scales with number of sources
    fig, axes = plt.subplots(n_sources, 1, figsize=(14, 7 * n_sources), sharex=True)
    if n_sources == 1: axes = [axes]
    
    # Map sources to Control/Optimized colors
    source_colors = {'Control': BLUE, 'Optimized': ORANGE}
    
    for i, source in enumerate(sources):
        ax1 = axes[i]
        subset = master_df[master_df['Source'] == source]
        
        # Determine color for this graph based on source
        graph_color = source_colors.get(source, 'black')
        
        # Plot λ (Left Axis) - SOLID
        grouped_weight = subset.groupby('Epoch')['λ']
        mean_weight = grouped_weight.mean()
        sem_weight = grouped_weight.sem()
        ci_weight = sem_weight * 1.96
        
        ax1.plot(mean_weight.index, mean_weight.values, 
                 color=graph_color, linewidth=4.0, linestyle='-', label='λ')
        ax1.fill_between(mean_weight.index, mean_weight - ci_weight, mean_weight + ci_weight, 
                         color=graph_color, alpha=0.2)
        
        # Only show x-axis label on the last subplot
        if i == n_sources - 1:
            ax1.set_xlabel('Epoch', fontsize=24, fontweight='bold', labelpad=15)
        ax1.set_ylabel('λ', color=graph_color, fontsize=24, fontweight='bold', labelpad=15)
        ax1.tick_params(axis='y', labelcolor=graph_color, labelsize=22)
        ax1.tick_params(axis='x', labelsize=22)
        ax1.grid(True, alpha=0.3)
        
        # Plot Slope (Right Axis) - DASHED
        ax2 = ax1.twinx()
        
        grouped_slope = subset.groupby('Epoch')['∆Lval']
        mean_slope = grouped_slope.mean()
        sem_slope = grouped_slope.sem()
        ci_slope = sem_slope * 1.96
        
        # Dashed line
        ax2.plot(mean_slope.index, mean_slope.values, 
                 color=graph_color, linewidth=4.0, linestyle='--', alpha=1.0, label='∆Lval')
        ax2.fill_between(mean_slope.index, mean_slope - ci_slope, mean_slope + ci_slope,
                         color=graph_color, alpha=0.2)
                 
        ax2.set_ylabel('∆Lval', color=graph_color, fontsize=24, fontweight='bold', labelpad=15)
        ax2.tick_params(axis='y', labelcolor=graph_color, labelsize=22)
        
        # Add zero line for slope
        ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        # Spines
        ax1.spines['left'].set_color(graph_color)
        ax1.spines['left'].set_linewidth(3)
        ax1.spines['left'].set_linestyle('-')
        # Hide the right spine of ax1 to prevent black line overlap on the right
        ax1.spines['right'].set_visible(False)

        ax2.spines['right'].set_color(graph_color)
        ax2.spines['right'].set_linewidth(3)
        ax2.spines['right'].set_linestyle('--')
        # Hide the left spine of ax2 to prevent black line overlap on the left
        ax2.spines['left'].set_visible(False)

        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    # Align y-axis labels across subplots
    fig.align_ylabels()
    
    # Adjusted margins for vertical stacking - reduced hspace for closer plots
    fig.subplots_adjust(top=0.95, hspace=0.15, bottom=0.10, left=0.10, right=0.90)
    
    # No legend for consistency

    output_path = os.path.join(PLOTS_DIR, 'mechanism_adaptive_control_faceted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_complexity_phase_space(master_df):
    """
    Plots Shannon Entropy vs Disequilibrium (The two components of LMC).
    """
    if master_df is None: return
    
    required = ['Shannon Entropy', 'Disequilibrium']
    if not all(c in master_df.columns for c in required): return

    plt.figure(figsize=(12, 8))
    
    # Aggregate mean per epoch per source
    agg_df = master_df.groupby(['Source', 'Epoch'])[required].mean().reset_index()
    
    sns.lineplot(
        data=agg_df,
        x='Shannon Entropy',
        y='Disequilibrium',
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
            plt.text(subset.iloc[0]['Shannon Entropy'], subset.iloc[0]['Disequilibrium'], 'Start', 
                     fontsize=9, ha='right', va='bottom')
            # End
            plt.text(subset.iloc[-1]['Shannon Entropy'], subset.iloc[-1]['Disequilibrium'], 'End', 
                     fontsize=9, ha='left', va='top')

    plt.title('Complexity Phase Space (Shannon Entropy vs Disequilibrium)', fontweight='normal', fontsize=16)
    plt.xlabel('Shannon Entropy (H)', fontsize=12)
    plt.ylabel('Disequilibrium (D)', fontsize=12)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'phase_space_entropy_diseq.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_distributions_faceted(sources):
    """
    Plots weight distributions in a 2x2 faceted layout:
    Left column: Control (Start, End)
    Right column: Optimized (Start, End)
    """
    print("Plotting faceted distributions...")
    
    # We expect exactly 2 sources: Control and Optimized
    source_dict = {label: folder for label, folder in sources}
    
    if 'Control' not in source_dict or 'Optimized' not in source_dict:
        print("Warning: Both Control and Optimized sources required for faceted distribution plot.")
        return
    
    # Create 2x2 subplot grid - standardized to 14 inch width
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), sharex=True, sharey=True)
    
    # Get epochs for each source
    control_epochs = get_epochs_for_folder(source_dict['Control'])
    optimized_epochs = get_epochs_for_folder(source_dict['Optimized'])
    
    if not control_epochs or not optimized_epochs:
        print("Warning: No epochs found for distribution plotting.")
        plt.close()
        return
    
    # Define subplot positions
    # axes[0, 0] = Control Start
    # axes[0, 1] = Control End
    # axes[1, 0] = Optimized Start
    # axes[1, 1] = Optimized End
    
    configs = [
        (0, 0, 'Control', control_epochs[0], source_dict['Control'], 'End of Epoch 1'),
        (0, 1, 'Control', control_epochs[-1], source_dict['Control'], 'End of Epoch 50'),
        (1, 0, 'Optimized', optimized_epochs[0], source_dict['Optimized'], 'End of Epoch 1'),
        (1, 1, 'Optimized', optimized_epochs[-1], source_dict['Optimized'], 'End of Epoch 50'),
    ]
    
    # Store data for all subplots to determine common y-axis limits
    plot_data = {}
    
    for row, col, source_label, epoch, folder, time_label in configs:
        ax = axes[row, col]
        
        # Load distribution files for this epoch
        dist_dir = os.path.join(folder, 'distributions')
        files = glob.glob(os.path.join(dist_dir, f'distribution_epoch_{epoch:03d}_run_*.csv'))
        
        if not files:
            ax.text(0.5, 0.5, f'No data for {source_label} {time_label}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            plot_data[(row, col)] = None
            # Add Start/End titles only to first row
            if row == 0:
                ax.set_title(time_label, fontweight='bold', fontsize=18, pad=15)
            continue
        
        # Color selection
        color = BLUE if source_label == 'Control' else ORANGE
        
        # Aggregate distributions across all runs
        all_dfs = []
        for fpath in files:
            df = load_distribution_file(fpath)
            if df is not None and 'Bin_Center' in df.columns and 'Probability' in df.columns:
                all_dfs.append(df.sort_values('Bin_Center'))
        
        if all_dfs:
            # Check if all runs have the same number of bins
            bin_lengths = [len(df) for df in all_dfs]
            if len(set(bin_lengths)) == 1:
                # All runs have same bins - simple averaging
                avg_df = all_dfs[0][['Bin_Center']].copy()
                
                # Shift x-axis to center at 0 (weights are typically centered around 0)
                avg_df['Bin_Center'] = avg_df['Bin_Center'] - 0.5
                
                prob_values = np.array([df['Probability'].values for df in all_dfs])
                
                avg_df['Probability'] = np.mean(prob_values, axis=0)
                avg_df['Std'] = np.std(prob_values, axis=0, ddof=1)  # Sample std
                avg_df['SE'] = avg_df['Std'] / np.sqrt(len(all_dfs))  # Standard error
                avg_df['CI_Lower'] = avg_df['Probability'] - 1.96 * avg_df['SE']
                avg_df['CI_Upper'] = avg_df['Probability'] + 1.96 * avg_df['SE']
            else:
                # Bins don't align - use interpolation to common grid
                from scipy.interpolate import interp1d
                
                # Create common grid from min to max bin center across all runs
                all_bins = np.concatenate([df['Bin_Center'].values for df in all_dfs])
                common_bins = np.linspace(all_bins.min(), all_bins.max(), 200)
                
                # Interpolate each run to common grid
                interpolated_probs = []
                for df in all_dfs:
                    interp_func = interp1d(df['Bin_Center'].values, df['Probability'].values, 
                                          kind='linear', bounds_error=False, fill_value=0)
                    interpolated_probs.append(interp_func(common_bins))
                
                prob_values = np.array(interpolated_probs)
                
                avg_df = pd.DataFrame({
                    'Bin_Center': common_bins - 0.5,  # Shift to center at 0
                    'Probability': np.mean(prob_values, axis=0),
                    'Std': np.std(prob_values, axis=0, ddof=1),
                })
                avg_df['SE'] = avg_df['Std'] / np.sqrt(len(all_dfs))
                avg_df['CI_Lower'] = avg_df['Probability'] - 1.96 * avg_df['SE']
                avg_df['CI_Upper'] = avg_df['Probability'] + 1.96 * avg_df['SE']
            
            # Ensure CI doesn't go below 0
            avg_df['CI_Lower'] = avg_df['CI_Lower'].clip(lower=0)
            
            # Store data for later y-axis standardization
            plot_data[(row, col)] = avg_df
            
            # Plot the averaged distribution with 95% CI
            ax.plot(avg_df['Bin_Center'], avg_df['Probability'], 
                   color=color, alpha=0.9, linewidth=3.0, zorder=3)
            
            # Add 95% CI shaded region
            ax.fill_between(avg_df['Bin_Center'], 
                           avg_df['CI_Lower'], 
                           avg_df['CI_Upper'],
                           color=color, alpha=0.2, zorder=2, label='95% CI')
            
            # Add base fill under the mean line for visual appeal
            ax.fill_between(avg_df['Bin_Center'], avg_df['Probability'],
                           color=color, alpha=0.15, zorder=1)

            # Peak probability density line + label
            peak_idx = avg_df['Probability'].idxmax()
            peak_x   = avg_df.loc[peak_idx, 'Bin_Center']
            ax.axvline(peak_x, color=color, linestyle='--', linewidth=1.8,
                       alpha=0.85, zorder=5)
            # Offset label to the right of the line with a white background
            x_range = avg_df['Bin_Center'].max() - avg_df['Bin_Center'].min()
            ax.text(peak_x + x_range * 0.02, 0.88, f'Mo={peak_x:.3f}',
                    transform=ax.get_xaxis_transform(),
                    ha='left', va='top', fontsize=13,
                    fontweight='bold', color=color, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='none', alpha=0.75))
            # Weighted mean (avg weight) dotted vertical line + label
            total = avg_df['Probability'].sum()
            if total > 1e-12:
                avg_w = float((avg_df['Bin_Center'] * avg_df['Probability']).sum() / total)
                ax.axvline(avg_w, color=color, linestyle=':', linewidth=1.8,
                           alpha=0.80, zorder=5)
                ax.text(avg_w + x_range * 0.02, 0.70, f'\u03bc={avg_w:.3f}',
                        transform=ax.get_xaxis_transform(),
                        ha='left', va='top', fontsize=13,
                        fontweight='bold', color=color, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec='none', alpha=0.75))

        # Add Start/End column titles only to first row
        if row == 0:
            ax.set_title(time_label, fontweight='bold', fontsize=18, pad=15)
        
        # Styling
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Labels for edge subplots only
        if col == 0:  # Left column
            ax.set_ylabel('Probability Density', fontsize=18, fontweight='bold', labelpad=15)
        if row == 1:  # Bottom row
            ax.set_xlabel('Weight Value', fontsize=18, fontweight='bold', labelpad=12)
            
        ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Set common x-axis and y-axis limits across all subplots
    all_max_probs = []
    all_min_x = []
    all_max_x = []
    for data in plot_data.values():
        if data is not None:
            all_max_probs.append(data['CI_Upper'].max())
            all_min_x.append(data['Bin_Center'].min())
            all_max_x.append(data['Bin_Center'].max())
    
    if all_max_probs:
        global_y_max = max(all_max_probs)
        
        # Aggressive zoom - find where probability mass is concentrated
        # Calculate weighted center and spread for better zoom
        zoom_bins = []
        zoom_probs = []
        for data in plot_data.values():
            if data is not None:
                # Only consider bins with significant probability (> 0.1% of max for more data)
                significant = data[data['Probability'] > global_y_max * 0.001]
                if not significant.empty:
                    zoom_bins.extend(significant['Bin_Center'].values)
                    zoom_probs.extend(significant['Probability'].values)
        
        if zoom_bins:
            # Use wider percentile range to include more distribution tails
            zoom_x_min = np.percentile(zoom_bins, 1)
            zoom_x_max = np.percentile(zoom_bins, 99)
        else:
            # Fallback to tight range around 0
            zoom_x_min = -0.15
            zoom_x_max = 0.15
        
        for ax in axes.flat:
            # Crop y-axis to show more detail - only display up to 50% of max peak
            ax.set_ylim(0, global_y_max * 0.50)
            ax.set_xlim(zoom_x_min, zoom_x_max)
    
    # Adjust layout with reduced space between plots
    plt.subplots_adjust(hspace=0.08, wspace=0.08)
    
    output_path = os.path.join(PLOTS_DIR, 'distributions_faceted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_combined_loss_barplots(master_df):
    """Combined bar plot for all loss types (Training, Validation, Test Wiki, Test Shakespeare) in one plot."""
    if master_df is None: return
    
    # Get final epoch data
    final_epochs_df = master_df.loc[master_df.groupby('Run_ID')['Epoch'].idxmax()]
    
    # Define all loss columns we want to include
    loss_cols = ['Training Loss', 'Validation Loss']
    
    # Find test loss columns
    test_cols = [c for c in final_epochs_df.columns if 'Test Loss' in c]
    loss_cols.extend(test_cols)
    
    # Filter only available columns
    available_loss_cols = [c for c in loss_cols if c in final_epochs_df.columns]
    
    if not available_loss_cols:
        return
    
    # Melt the dataframe to long format
    df_melted = final_epochs_df.melt(
        id_vars=['Source', 'Run_ID'],
        value_vars=available_loss_cols,
        var_name='Loss Type',
        value_name='Cross-Entropy Loss'
    )
    
    # Clean up loss type names for better labels with full dataset names
    df_melted['Loss Type'] = df_melted['Loss Type'].str.replace('Training Loss', 'Training')
    df_melted['Loss Type'] = df_melted['Loss Type'].str.replace('Validation Loss', 'Validation')
    df_melted['Loss Type'] = df_melted['Loss Type'].str.replace('Test Loss Wiki', 'Test (WikiText-2)')
    df_melted['Loss Type'] = df_melted['Loss Type'].str.replace('Test Loss Shakespeare', 'Test (Shakespeare)')
    
    # Create the order for x-axis: Training, Validation, Test (WikiText-2), Test (Shakespeare)
    loss_order = ['Training', 'Validation']
    # Add test types in specific order: Wiki first, then Shakespeare
    if 'Test (WikiText-2)' in df_melted['Loss Type'].unique():
        loss_order.append('Test (WikiText-2)')
    if 'Test (Shakespeare)' in df_melted['Loss Type'].unique():
        loss_order.append('Test (Shakespeare)')
    
    # Create single plot - standardized to 14 inch width
    fig, ax = plt.subplots(figsize=(14, 8))
    
    hue_order = ['Control', 'Optimized']
    
    # Calculate stats for annotation including 95% CI
    stats = df_melted.groupby(['Loss Type', 'Source'])['Cross-Entropy Loss'].agg(['mean', 'std', 'sem']).reset_index()
    stats['ci'] = stats['sem'] * 1.96  # 95% confidence interval
    
    # Create barplot
    sns.barplot(
        data=df_melted,
        x='Loss Type',
        y='Cross-Entropy Loss',
        hue='Source',
        order=loss_order,
        hue_order=hue_order,
        palette=SOURCE_PALETTE,
        edgecolor='.2',
        errorbar=('se', 1.96),  # 95% confidence interval
        capsize=.1,
        err_kws={'linewidth': 3, 'color': 'black'},
        ax=ax,
        legend=False  # Remove legend
    )
    
    # Add value labels inside bars
    for i, container in enumerate(ax.containers):
        if i >= len(hue_order): break
        source_label = hue_order[i]
        
        for j, bar in enumerate(container):
            if j < len(loss_order):
                loss_type = loss_order[j]
                
                try:
                    stat_row = stats[(stats['Loss Type'] == loss_type) & (stats['Source'] == source_label)]
                    if not stat_row.empty:
                        mean_val = stat_row['mean'].values[0]
                        ci_val = stat_row['ci'].values[0]
                        
                        height = bar.get_height()
                        if np.isfinite(height) and height != 0:
                            # Add value label with CI inside the bar
                            ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                                   f'{mean_val:.4f}\n±{ci_val:.4f}',
                                   ha='center', va='center',
                                   fontsize=11,
                                   color='black',
                                   fontweight='bold')
                except (KeyError, IndexError):
                    pass
    
    # Reduce fontsize to compensate for global font_scale=1.2
    ax.set_ylabel('Cross-Entropy Loss', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_xlabel('', fontsize=20, fontweight='bold')  # No x-label needed
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    sns.despine(ax=ax, left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'barplot_combined_losses.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_overfitting_vs_complexity(master_df):
    """
    Plots Generalization Gap vs LMC Complexity to see if complexity correlates with overfitting.
    """
    if master_df is None: return
    
    if 'Validation Loss' not in master_df.columns or 'Training Loss' not in master_df.columns or 'LMC Complexity' not in master_df.columns:
        return
        
    df = master_df.copy()
    df['Generalization Gap'] = df['Validation Loss'] - df['Training Loss']
    
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(
        data=df,
        x='LMC Complexity',
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
            z = np.polyfit(subset['LMC Complexity'], subset['Generalization Gap'], 1)
            p = np.poly1d(z)
            
            x_range = np.linspace(subset['LMC Complexity'].min(), subset['LMC Complexity'].max(), 100)
            plt.plot(x_range, p(x_range), color=palette[i], linestyle='--', alpha=0.5, linewidth=1.5)

    plt.title('Overfitting vs Complexity (Gen Gap vs LMC)', fontweight='normal', fontsize=16)
    plt.xlabel('LMC Complexity', fontsize=12)
    plt.ylabel('Generalization Gap (Val - Train)', fontsize=12)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'scatter_overfitting_vs_complexity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_3d_trajectory_improved(master_df):
    """
    Improved 3D Trajectory: Training Loss vs Validation Loss vs Entropy.
    Avoids LMC vs Entropy correlation.
    """
    if master_df is None: return
    
    # Check available columns
    x_col = 'Training Loss'
    y_col = 'Validation Loss'
    z_col = 'Entropy'
    
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
    legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Loss\ Type}$'))
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
        value_name='Cross-Entropy Loss'
    )
    
    # Clean up loss type names to match barplot_combined_losses standard
    df_melted['Loss Type'] = df_melted['Loss Type'].str.replace('Training Loss', 'Training')
    df_melted['Loss Type'] = df_melted['Loss Type'].str.replace('Validation Loss', 'Validation')
    df_melted['Loss Type'] = df_melted['Loss Type'].str.replace('Test Loss Wiki', 'Test (WikiText-2)')
    df_melted['Loss Type'] = df_melted['Loss Type'].str.replace('Test Loss Shakespeare', 'Test (Shakespeare)')
    
    # Facet by Loss Type
    # Standardized to 14 inch width: height=5, aspect=1.4 gives ~7 inch width per subplot, 2 columns = 14 inches
    g = sns.FacetGrid(df_melted, col="Loss Type", hue="Source", col_wrap=2, 
                      height=5, aspect=1.4, sharey=False, palette=SOURCE_PALETTE)
    g.map(sns.lineplot, "Epoch", "Cross-Entropy Loss", errorbar=('se', 1.96), linewidth=4.0)
    # No legend
    g.set_titles("{col_name}", size=26, fontweight='bold')
    g.set_axis_labels("Epoch", "Cross-Entropy Loss", fontsize=24, fontweight='bold')
    
    # Add consistent labelpad
    for ax in g.axes.flat:
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
    
    # Align y-axis labels
    g.fig.align_ylabels()
    
    # Improve tick readability
    for ax in g.axes.flat:
        ax.tick_params(labelsize=22)

    g.fig.subplots_adjust(top=0.9)
    
    output_path = os.path.join(PLOTS_DIR, 'comparison_train_val_test_loss_faceted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_complexity_metrics_faceted(master_df):
    """Plots complexity metrics in faceted subplots stacked vertically."""
    if master_df is None: return
    
    cols = ['LMC Complexity', 'Shannon Entropy', 'Disequilibrium']
    available_cols = [c for c in cols if c in master_df.columns]
    
    if not available_cols: return

    df_melted = master_df.melt(
        id_vars=['Epoch', 'Source', 'Run_ID'], 
        value_vars=available_cols,
        var_name='Metric', 
        value_name='Value'
    )
    
    # Use row instead of col to stack vertically
    # Standardized to 14 inch width: height=5.6, aspect=2.5 gives 14 inch width
    g = sns.FacetGrid(df_melted, row="Metric", hue="Source", height=5.6, aspect=2.5, sharey=False, palette=SOURCE_PALETTE)
    g.map(sns.lineplot, "Epoch", "Value", errorbar=('se', 1.96), linewidth=4.0)
    # No legend
    # Set titles temporarily to extract row names
    g.set_titles("{row_name}")
    g.set_xlabels("Epoch", fontsize=24, fontweight='bold')
    
    # Move title to Y-axis label
    for ax in g.axes.flat:
        title = ax.get_title()
        ax.set_ylabel(title, fontsize=24, fontweight='bold')
        ax.set_title("")
        
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.tick_params(labelsize=22)
    
    # Align y-axis labels
    g.fig.align_ylabels()

    g.fig.subplots_adjust(top=0.95)
    
    output_path = os.path.join(PLOTS_DIR, 'comparison_complexity_metrics_faceted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

# ==========================================
# Distribution Evolution Plots (3D, GIF, Interactive)
# ==========================================

def _load_epoch_distributions(folder_path):
    """
    Helper: loads all numeric-epoch distribution CSVs from a folder, averaging
    across runs and computing a 95% normal confidence interval.

    Returns
    -------
    dict  {epoch_int: pd.DataFrame}
          Each DataFrame has columns:
              Bin_Center   – weight value shifted so mean≈0
              Probability  – mean probability density across runs
              CI_Lower     – lower bound of 95% CI (clipped at 0)
              CI_Upper     – upper bound of 95% CI
    """
    dist_dir = os.path.join(folder_path, 'distributions')
    if not os.path.isdir(dist_dir):
        return {}

    epochs = get_epochs_for_folder(folder_path)
    result = {}

    for epoch in epochs:
        files = glob.glob(os.path.join(dist_dir, f'distribution_epoch_{epoch:03d}_run_*.csv'))
        if not files:
            continue

        all_dfs = []
        for fpath in files:
            df = load_distribution_file(fpath)
            if df is not None and 'Bin_Center' in df.columns and 'Probability' in df.columns:
                all_dfs.append(df.sort_values('Bin_Center').reset_index(drop=True))

        if not all_dfs:
            continue

        n_runs = len(all_dfs)
        bin_lengths = [len(df) for df in all_dfs]

        if len(set(bin_lengths)) == 1:
            avg_df = all_dfs[0][['Bin_Center']].copy()
            avg_df['Bin_Center'] = avg_df['Bin_Center'] - 0.5
            prob_values = np.array([df['Probability'].values for df in all_dfs])
        else:
            from scipy.interpolate import interp1d
            all_bins = np.concatenate([df['Bin_Center'].values for df in all_dfs])
            common_bins = np.linspace(all_bins.min(), all_bins.max(), 200)
            interpolated = []
            for df in all_dfs:
                f = interp1d(df['Bin_Center'].values, df['Probability'].values,
                             kind='linear', bounds_error=False, fill_value=0)
                interpolated.append(f(common_bins))
            avg_df = pd.DataFrame({'Bin_Center': common_bins - 0.5})
            prob_values = np.array(interpolated)

        avg_df['Probability'] = np.mean(prob_values, axis=0)
        std = np.std(prob_values, axis=0, ddof=1) if n_runs > 1 else np.zeros(len(avg_df))
        se = std / np.sqrt(n_runs)
        avg_df['CI_Lower'] = (avg_df['Probability'] - 1.96 * se).clip(lower=0)
        avg_df['CI_Upper'] = avg_df['Probability'] + 1.96 * se

        result[epoch] = avg_df

    return result


def _dist_axis_limits(source_data, common_epochs_set):
    """Return (x_min, x_max, global_y_max) consistent with the faceted plot style."""
    all_probs, all_sig_bins = [], []
    for epoch_data in source_data.values():
        for epoch, df in epoch_data.items():
            if epoch in common_epochs_set:
                all_probs.append(df['CI_Upper'].max())
                sig = df[df['Probability'] > max(df['Probability'].max() * 0.001, 1e-8)]
                all_sig_bins.extend(sig['Bin_Center'].values)
    global_y_max = max(all_probs) * 0.5 if all_probs else 1.0
    if all_sig_bins:
        x_min = float(np.percentile(all_sig_bins, 1))
        x_max = float(np.percentile(all_sig_bins, 99))
    else:
        x_min, x_max = -0.15, 0.15
    return x_min, x_max, global_y_max


def _load_epoch_distribution_stats(folder_path):
    """
    For each epoch, computes per-run summary statistics of the weight distribution:
      - avg_density  : mean probability density across all bins
      - peak_density : maximum probability density (mode height)
      - peak_weight  : weight value (bin center) where probability is highest (mode position)
      - avg_weight   : weighted mean weight value (first moment / centre of mass)

    Returns
    -------
    dict  {epoch_int: {
               'avg_density':  (mean, ci_lower, ci_upper),
               'peak_density': (mean, ci_lower, ci_upper),
               'peak_weight':  (mean, ci_lower, ci_upper),
               'avg_weight':   (mean, ci_lower, ci_upper),
           }}
    """
    dist_dir = os.path.join(folder_path, 'distributions')
    if not os.path.isdir(dist_dir):
        return {}

    epochs = get_epochs_for_folder(folder_path)
    result = {}

    for epoch in epochs:
        files = glob.glob(
            os.path.join(dist_dir, f'distribution_epoch_{epoch:03d}_run_*.csv')
        )
        if not files:
            continue

        run_stats = {'avg_density': [], 'peak_density': [],
                     'peak_weight': [], 'avg_weight': []}

        for fpath in files:
            df = load_distribution_file(fpath)
            if df is None or 'Bin_Center' not in df.columns or 'Probability' not in df.columns:
                continue
            df   = df.sort_values('Bin_Center').reset_index(drop=True)
            bins = df['Bin_Center'].values - 0.5
            prob = df['Probability'].values

            run_stats['avg_density'].append(float(np.mean(prob)))
            run_stats['peak_density'].append(float(np.max(prob)))
            run_stats['peak_weight'].append(float(bins[np.argmax(prob)]))

            total = prob.sum()
            w = prob / total if total > 1e-12 else np.ones(len(prob)) / len(prob)
            run_stats['avg_weight'].append(float(np.sum(bins * w)))

        out = {}
        for key, vals in run_stats.items():
            if not vals:
                continue
            arr = np.array(vals)
            n   = len(arr)
            m   = float(arr.mean())
            se  = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            ci_lo = m - 1.96 * se
            # weight positions can be negative — don't clip to 0
            if key in ('avg_density', 'peak_density'):
                ci_lo = max(0.0, ci_lo)
            out[key] = (m, ci_lo, m + 1.96 * se)

        if out:
            result[epoch] = out

    return result


def plot_distribution_summary_per_epoch(sources):
    """
    1×N figure (one panel per source) mirroring the two vertical lines shown
    in the GIF per frame — both tracked over epochs on the same weight-value axis:
      --  Peak Weight Position  (where density peaks, purple)   ← GIF dashed line
      ··  Avg Weight Value / μ  (weighted mean of distribution, grey) ← GIF dotted line
    Shaded bands = 95% CI across runs.
    """
    print("Plotting distribution summary per epoch...")

    source_data = {}
    for label, folder in sources:
        stats = _load_epoch_distribution_stats(folder)
        if stats:
            source_data[label] = stats
    if not source_data:
        print("No distribution data for summary plot.")
        return

    sources_list = list(source_data.keys())
    n_cols       = len(sources_list)
    src_colors   = {'Control': BLUE, 'Optimized': ORANGE}

    fig, axes = plt.subplots(1, n_cols, figsize=(9 * n_cols, 6), sharey=False)
    if n_cols == 1:
        axes = [axes]
    plt.subplots_adjust(left=0.08, right=0.97, top=0.92,
                        bottom=0.12, wspace=0.45)

    for col_idx, label in enumerate(sources_list):
        ax           = axes[col_idx]
        ax_right     = ax.twinx()
        epoch_stats  = source_data[label]
        sorted_epochs = sorted(epoch_stats.keys())
        xs = sorted_epochs

        # Source colour: Control = BLUE, Optimized = ORANGE
        panel_color = src_colors.get(label, BLUE)

        def _vec(key, _stats=epoch_stats, _xs=sorted_epochs):
            means    = [_stats[e][key][0] for e in _xs]
            ci_lower = [_stats[e][key][1] for e in _xs]
            ci_upper = [_stats[e][key][2] for e in _xs]
            return np.array(means), np.array(ci_lower), np.array(ci_upper)

        # --- Mode (Mo)  (X of GIF dashed line) -- left axis ---
        m, lo, hi = _vec('peak_weight')
        ax.plot(xs, m, color=panel_color, linewidth=3.0, linestyle='--',
                label='Mode (Mo)')
        ax.fill_between(xs, lo, hi, color=panel_color, alpha=0.18)

        # --- Average (μ)  (X of GIF dotted line) -- left axis ---
        m, lo, hi = _vec('avg_weight')
        ax.plot(xs, m, color=panel_color, linewidth=2.5, linestyle=':',
                label='Average (μ)')
        ax.fill_between(xs, lo, hi, color=panel_color, alpha=0.15)

        # --- Highest probability density f(Mo)  (Y of GIF dashed line) -- right axis ---
        m, lo, hi = _vec('peak_density')
        ax_right.plot(xs, m, color=panel_color, linewidth=2.5, linestyle='-',
                      label='Highest Probability Density (f(Mo))')
        ax_right.fill_between(xs, lo, hi, color=panel_color, alpha=0.13)
        ax_right.set_ylabel('Highest Probability Density [f(Mo)]', fontsize=18,
                            fontweight='bold', labelpad=10, color='black')
        ax_right.tick_params(axis='y', labelcolor='black', labelsize=16)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_color('black')
        ax_right.spines['right'].set_linewidth(1.5)

        ax.set_xlabel('Epoch', fontsize=18, fontweight='bold', labelpad=10)
        ax.set_ylabel('Weight Values [μ, Mo]', fontsize=18, fontweight='bold', labelpad=10, color='black')
        ax.tick_params(axis='both', labelsize=16, labelcolor='black')
        ax.grid(True, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(1.5)

        # Combined legend from both axes
        h_l, lbl_l = ax.get_legend_handles_labels()
        h_r, lbl_r = ax_right.get_legend_handles_labels()
        ax.legend(h_l + h_r, lbl_l + lbl_r,
                  fontsize=11, loc='best', framealpha=0.85, edgecolor='#cccccc')

    output_path = os.path.join(PLOTS_DIR, 'distribution_summary_per_epoch.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_distributions_3d_evolution(sources):
    """
    3D waterfall (ribbon) plot of weight distribution evolution across all epochs.
    One subplot per source (side by side). No figure title.
    Axes: X = Weight Value, Y = Epoch, Z = Probability Density.
    Control uses the Blues colormap; Optimized uses the Oranges colormap.
    Mean line + 95% CI ribbon drawn for every epoch (average across runs).
    """
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    print("Plotting 3D distribution evolution...")

    source_data = {}
    for label, folder in sources:
        data = _load_epoch_distributions(folder)
        if data:
            source_data[label] = data

    if not source_data:
        print("No distribution data found for 3D plot.")
        return

    all_epochs = sorted({ep for ed in source_data.values() for ep in ed})

    # Robust axis limits for the 3D plot:
    # - z (probability): 90th-percentile of per-epoch probability peaks × 1.20.
    #   Using a percentile instead of the absolute max prevents late-epoch
    #   narrow spikes from collapsing all other epochs to a flat baseline.
    # - x (weight value): bins with probability > 1 % of the local epoch peak,
    #   then the 1st–99th percentile of those bins for a moderate zoom.
    per_epoch_maxes = []
    all_sig_bins = []
    for epoch_data_src in source_data.values():
        for df in epoch_data_src.values():
            ep_max = df['Probability'].max()
            per_epoch_maxes.append(ep_max)
            sig = df[df['Probability'] > ep_max * 0.01]
            all_sig_bins.extend(sig['Bin_Center'].values)

    global_y_max = float(np.percentile(per_epoch_maxes, 90)) * 1.20 if per_epoch_maxes else 1.0
    if all_sig_bins:
        x_min = float(np.percentile(all_sig_bins, 1))
        x_max = float(np.percentile(all_sig_bins, 99))
    else:
        x_min, x_max = -0.15, 0.15

    cmaps_map = {'Control': plt.cm.Blues, 'Optimized': plt.cm.Oranges}

    sources_list = list(source_data.keys())
    n_panels = len(sources_list)

    fig = plt.figure(figsize=(10 * n_panels, 10))

    for idx, label in enumerate(sources_list):
        ax = fig.add_subplot(1, n_panels, idx + 1, projection='3d')
        epoch_data = source_data[label]
        cmap = cmaps_map.get(label, plt.cm.Blues)
        epochs = sorted(epoch_data.keys())
        n_epochs = len(epochs)

        for i, epoch in enumerate(epochs):
            df = epoch_data[epoch]
            mask = (df['Bin_Center'] >= x_min) & (df['Bin_Center'] <= x_max)
            df_z = df[mask]
            if df_z.empty:
                continue

            x = df_z['Bin_Center'].values
            y_val = float(epoch)
            z_mean = df_z['Probability'].values
            z_lower = df_z['CI_Lower'].values
            z_upper = df_z['CI_Upper'].values

            # Colour: light (early epochs) → dark (late epochs)
            t = i / max(n_epochs - 1, 1)
            color = cmap(0.30 + 0.55 * t)

            # Mean line
            ax.plot(x, [y_val] * len(x), z_mean,
                    color=color, alpha=0.80, linewidth=1.2, zorder=3)

            # 95% CI ribbon
            verts_ci = (
                list(zip(x,        [y_val] * len(x), z_lower)) +
                list(zip(x[::-1],  [y_val] * len(x), z_upper[::-1]))
            )
            poly_ci = Poly3DCollection([verts_ci], alpha=0.07, zorder=2)
            poly_ci.set_facecolor(color)
            poly_ci.set_edgecolor('none')
            ax.add_collection3d(poly_ci)

            # Base fill under mean curve
            verts_base = (
                list(zip(x,       [y_val] * len(x), np.zeros_like(z_mean))) +
                list(zip(x[::-1], [y_val] * len(x), z_mean[::-1]))
            )
            poly_base = Poly3DCollection([verts_base], alpha=0.04, zorder=1)
            poly_base.set_facecolor(color)
            poly_base.set_edgecolor('none')
            ax.add_collection3d(poly_base)

        ax.set_xlabel('Weight Value', fontsize=16, fontweight='bold', labelpad=14)
        ax.set_ylabel('Epoch', fontsize=16, fontweight='bold', labelpad=14)
        ax.set_zlabel('Probability Density', fontsize=15, fontweight='bold', labelpad=30)
        ax.set_xlim(x_min, x_max)
        ax.set_zlim(0, global_y_max)
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.tick_params(axis='both', labelsize=14)
        ax.tick_params(axis='z', labelsize=13, pad=12)
        # Fewer ticks + compact format so labels don't reach the spine
        ax.set_zticks(np.linspace(0, global_y_max, 5))
        ax.zaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v:.1e}')
        )

    plt.subplots_adjust(left=0.05, right=0.82, top=0.88, bottom=0.06,
                        wspace=0.20)

    output_path = os.path.join(PLOTS_DIR, '3d_distribution_evolution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.6)
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_distributions_gif_evolution(sources):
    """
    Animated GIF of weight distribution evolution across epochs.
    Two side-by-side panels (one per source).
    Each frame = one epoch. Shows mean + 95% CI. Averages across all runs.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    print("Creating distribution evolution GIF (this may take a moment)...")

    source_data = {}
    for label, folder in sources:
        data = _load_epoch_distributions(folder)
        if data:
            source_data[label] = data

    if not source_data:
        print("No distribution data for GIF.")
        return

    all_epoch_sets = [set(d.keys()) for d in source_data.values()]
    common_epochs = sorted(
        set.intersection(*all_epoch_sets) if len(all_epoch_sets) > 1 else all_epoch_sets[0]
    )
    if not common_epochs:
        print("No common epochs found for GIF.")
        return

    x_min, x_max, global_y_max = _dist_axis_limits(source_data, set(common_epochs))

    sources_list = list(source_data.keys())
    n_sources = len(sources_list)
    src_colors = {'Control': BLUE, 'Optimized': ORANGE}

    fig, axes = plt.subplots(1, n_sources, figsize=(14, 6), sharey=True)
    if n_sources == 1:
        axes = [axes]
    plt.subplots_adjust(wspace=0.06, top=0.93, bottom=0.13, left=0.08, right=0.98)

    lines = []
    for i, label in enumerate(sources_list):
        ax = axes[i]
        color = src_colors.get(label, BLUE)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, global_y_max)
        ax.set_xlabel('Weight Value', fontsize=15, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Probability Density', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=13)
        line, = ax.plot([], [], color=color, linewidth=2.5, zorder=4)
        lines.append(line)

    epoch_text = fig.text(0.5, 0.97, '', ha='center', va='top',
                          fontsize=17, fontweight='bold')

    # Track per-panel peak indicators so they can be cleared each frame
    peak_vlines  = [None] * n_sources
    peak_texts   = [None] * n_sources
    avg_vlines   = [None] * n_sources
    avg_vtexts   = [None] * n_sources

    def init():
        for ln in lines:
            ln.set_data([], [])
        epoch_text.set_text('')
        return lines + [epoch_text]

    def update(frame_idx):
        epoch = common_epochs[frame_idx]
        epoch_text.set_text(f'Epoch {epoch}')
        for i, label in enumerate(sources_list):
            ax = axes[i]
            color = src_colors.get(label, BLUE)
            df = source_data[label].get(epoch)
            # Remove previous fill collections
            for coll in ax.collections[:]:
                coll.remove()
            # Remove previous peak indicators
            if peak_vlines[i] is not None:
                peak_vlines[i].remove()
                peak_vlines[i] = None
            if peak_texts[i] is not None:
                peak_texts[i].remove()
                peak_texts[i] = None
            # Remove previous avg-weight indicators
            if avg_vlines[i] is not None:
                avg_vlines[i].remove()
                avg_vlines[i] = None
            if avg_vtexts[i] is not None:
                avg_vtexts[i].remove()
                avg_vtexts[i] = None
            if df is not None:
                lines[i].set_data(df['Bin_Center'].values, df['Probability'].values)
                ax.fill_between(df['Bin_Center'], df['CI_Lower'], df['CI_Upper'],
                                color=color, alpha=0.25, zorder=2)
                ax.fill_between(df['Bin_Center'], df['Probability'],
                                color=color, alpha=0.10, zorder=1)
                # Peak vertical line + label
                peak_idx = df['Probability'].idxmax()
                peak_x   = df.loc[peak_idx, 'Bin_Center']
                vl = ax.axvline(peak_x, color=color, linestyle='--',
                                linewidth=1.8, alpha=0.85, zorder=5)
                peak_vlines[i] = vl
                x_range = x_max - x_min
                txt = ax.text(peak_x + x_range * 0.02, 0.88, f'Mo={peak_x:.3f}',
                              transform=ax.get_xaxis_transform(),
                              ha='left', va='top', fontsize=11,
                              fontweight='bold', color=color, zorder=6,
                              bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                        ec='none', alpha=0.75))
                peak_texts[i] = txt
                # Weighted mean (average weight value) vertical dotted line
                total = df['Probability'].sum()
                w = df['Probability'] / total if total > 1e-12 else None
                if w is not None:
                    avg_w = float((df['Bin_Center'] * w).sum())
                    avl = ax.axvline(avg_w, color=color, linestyle=':',
                                     linewidth=1.8, alpha=0.80, zorder=5)
                    avg_vlines[i] = avl
                    atxt = ax.text(avg_w + x_range * 0.02, 0.70, f'μ={avg_w:.3f}',
                                   transform=ax.get_xaxis_transform(),
                                   ha='left', va='top', fontsize=11,
                                   fontweight='bold', color=color, zorder=6,
                                   bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                             ec='none', alpha=0.75))
                    avg_vtexts[i] = atxt
        return lines + [epoch_text]

    anim = FuncAnimation(fig, update, frames=len(common_epochs),
                         init_func=init, blit=False, interval=200)

    output_path = os.path.join(PLOTS_DIR, 'distribution_evolution.gif')
    anim.save(output_path, writer=PillowWriter(fps=5), dpi=100)
    plt.close()
    print(f"Saved GIF: {output_path}")


def plot_distributions_interactive(sources):
    """
    Interactive HTML plot using pure HTML5 Canvas — no external libraries.
    Guaranteed zero-latency frame switching (plain JS setInterval + slider).
    Both sources overlaid. Shows mean line + 95% CI band + peak dashed line.
    """
    import json

    print("Creating interactive distribution plot (pure canvas)...")

    source_data = {}
    for label, folder in sources:
        data = _load_epoch_distributions(folder)
        if data:
            source_data[label] = data

    if not source_data:
        print("No data for interactive plot.")
        return

    all_epoch_sets = [set(d.keys()) for d in source_data.values()]
    epochs = sorted(
        set.intersection(*all_epoch_sets) if len(all_epoch_sets) > 1 else all_epoch_sets[0]
    )
    if not epochs:
        print("No common epochs for interactive plot.")
        return

    x_min, x_max, global_y_max = _dist_axis_limits(source_data, set(epochs))
    src_colors_hex = {'Control': '#1f77b4', 'Optimized': '#ff7f0e'}

    # Serialise all epoch data to JSON (only x range that fits the zoom window)
    all_data = {}
    for epoch in epochs:
        all_data[str(epoch)] = {}
        for label, epoch_data in source_data.items():
            df = epoch_data.get(epoch)
            if df is not None:
                mask = (df['Bin_Center'] >= x_min) & (df['Bin_Center'] <= x_max)
                df_z = df[mask]
                all_data[str(epoch)][label] = {
                    'x':         df_z['Bin_Center'].tolist(),
                    'mean':      df_z['Probability'].tolist(),
                    'ci_lower':  df_z['CI_Lower'].tolist(),
                    'ci_upper':  df_z['CI_Upper'].tolist(),
                    'avg_weight': float(
                        (df_z['Bin_Center'] * df_z['Probability']).sum() /
                        df_z['Probability'].sum()
                        if df_z['Probability'].sum() > 1e-12 else 0.0
                    ),
                }

    data_json   = json.dumps(all_data)
    epochs_json = json.dumps([str(e) for e in epochs])
    colors_json = json.dumps(src_colors_hex)
    n_epochs    = len(epochs)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Weight Distribution Evolution</title>
<style>
  body      {{ font-family: Arial, sans-serif; background:#fff; margin:20px; }}
  h2        {{ text-align:center; font-size:18px; margin-bottom:6px; }}
  #controls {{ text-align:center; margin:10px 0; }}
  #slider   {{ width:80%; cursor:pointer; }}
  #epoch-label {{ font-weight:bold; font-size:16px; margin:0 12px; }}
  button    {{ padding:6px 18px; font-size:15px; cursor:pointer; margin:0 4px; border:1px solid #aaa; border-radius:4px; background:#f5f5f5; }}
  button:hover {{ background:#e0e0e0; }}
  canvas    {{ display:block; margin:0 auto; border:1px solid #e8e8e8; }}
  #legend   {{ text-align:center; margin-top:10px; font-size:14px; }}
  .leg-item {{ display:inline-block; margin:0 14px; }}
  .leg-line {{ display:inline-block; width:30px; height:3px; vertical-align:middle; margin-right:5px; border-radius:2px; }}
</style>
</head>
<body>
<h2>Weight Distribution Evolution &mdash; Mean &plusmn; 95% CI (avg over runs)</h2>
<div id="controls">
  <button id="btnPlay">&#9654; Play</button>
  <button id="btnPause">&#9646;&#9646; Pause</button>
  <span id="epoch-label">Epoch {epochs[0]}</span>
  <br><br>
  <input type="range" id="slider" min="0" max="{n_epochs - 1}" value="0" step="1">
</div>
<canvas id="chart" width="1100" height="560"></canvas>
<div id="legend"></div>

<script>
const DATA   = {data_json};
const EPOCHS = {epochs_json};
const COLORS = {colors_json};
const X_MIN  = {float(x_min)};
const X_MAX  = {float(x_max)};
const Y_MAX  = {float(global_y_max)};

const canvas     = document.getElementById('chart');
const ctx        = canvas.getContext('2d');
const slider     = document.getElementById('slider');
const epochLabel = document.getElementById('epoch-label');

const PAD = {{left:82, right:20, top:20, bottom:58}};
const W   = canvas.width  - PAD.left - PAD.right;
const H   = canvas.height - PAD.top  - PAD.bottom;

function xToC(x) {{ return PAD.left + (x - X_MIN) / (X_MAX - X_MIN) * W; }}
function yToC(y) {{ return PAD.top  + H - Math.max(0, y / Y_MAX) * H;    }}

function hexToRgba(hex, a) {{
  const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

function drawBackground() {{
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Y grid + labels
  const nY = 5;
  for (let i = 0; i <= nY; i++) {{
    const cy  = PAD.top + (H / nY) * i;
    const val = Y_MAX * (1 - i / nY);
    ctx.save();
    ctx.strokeStyle = '#ebebeb'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD.left, cy); ctx.lineTo(PAD.left + W, cy); ctx.stroke();
    ctx.fillStyle = '#555'; ctx.font = '11px Arial'; ctx.textAlign = 'right';
    ctx.fillText(val.toExponential(2), PAD.left - 6, cy + 4);
    ctx.restore();
  }}

  // X grid + labels
  const nX = 6;
  for (let i = 0; i <= nX; i++) {{
    const cx  = PAD.left + (W / nX) * i;
    const val = X_MIN + (X_MAX - X_MIN) * i / nX;
    ctx.save();
    ctx.strokeStyle = '#ebebeb'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, PAD.top); ctx.lineTo(cx, PAD.top + H); ctx.stroke();
    ctx.fillStyle = '#555'; ctx.font = '11px Arial'; ctx.textAlign = 'center';
    ctx.fillText(val.toFixed(3), cx, PAD.top + H + 18);
    ctx.restore();
  }}

  // Axes frame
  ctx.save();
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1.5;
  ctx.strokeRect(PAD.left, PAD.top, W, H);
  ctx.restore();

  // Axis labels
  ctx.save();
  ctx.fillStyle = '#222'; ctx.font = 'bold 13px Arial'; ctx.textAlign = 'center';
  ctx.fillText('Weight Value', PAD.left + W / 2, canvas.height - 8);
  ctx.translate(16, PAD.top + H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Probability Density', 0, 0);
  ctx.restore();
}}

function drawFrame(epochIdx) {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawBackground();

  const epochKey = EPOCHS[epochIdx];
  const frameData = DATA[epochKey] || {{}};
  const labels = Object.keys(frameData);

  // Track peak positions to avoid label overlap between sources
  const peakPositions = [];

  labels.forEach(label => {{
    const d     = frameData[label];
    const color = COLORS[label] || '#1f77b4';
    const n     = d.x.length;

    // CI band
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < n; i++) {{
      const cx = xToC(d.x[i]), cy = yToC(d.ci_upper[i]);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }}
    for (let i = n - 1; i >= 0; i--) {{
      ctx.lineTo(xToC(d.x[i]), yToC(d.ci_lower[i]));
    }}
    ctx.closePath();
    ctx.fillStyle = hexToRgba(color, 0.18);
    ctx.fill();
    ctx.restore();

    // Mean line
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < n; i++) {{
      const cx = xToC(d.x[i]), cy = yToC(d.mean[i]);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }}
    ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.restore();

    // Peak dashed line
    let maxV = -Infinity, maxI = 0;
    d.mean.forEach((v, i) => {{ if (v > maxV) {{ maxV = v; maxI = i; }} }});
    const peakX  = d.x[maxI];
    const peakCX = xToC(peakX);

    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.moveTo(peakCX, PAD.top); ctx.lineTo(peakCX, PAD.top + H); ctx.stroke();
    ctx.restore();

    // Peak label — stack vertically if positions clash
    const lText = 'Mo=' + peakX.toFixed(3);
    ctx.font = 'bold 12px Arial';
    const tw  = ctx.measureText(lText).width;
    let lx = peakCX + 5;
    let ly = PAD.top + 20;
    // Shift down if another label is nearby
    peakPositions.forEach(p => {{ if (Math.abs(p.x - lx) < tw + 10) ly += 20; }});
    peakPositions.push({{x: lx, y: ly}});
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.80)';
    ctx.fillRect(lx - 2, ly - 13, tw + 6, 17);
    ctx.fillStyle = color;
    ctx.fillText(lText, lx, ly);
    ctx.restore();

    // Weighted mean (avg weight value) dotted vertical line + label
    const avgW  = d.avg_weight;
    const avgCX = xToC(avgW);
    ctx.save();
    ctx.setLineDash([3, 5]);
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.globalAlpha = 0.75;
    ctx.beginPath(); ctx.moveTo(avgCX, PAD.top); ctx.lineTo(avgCX, PAD.top + H); ctx.stroke();
    ctx.restore();
    const muText = '\u03bc=' + avgW.toFixed(3);
    ctx.font = 'bold 11px Arial';
    const mtw = ctx.measureText(muText).width;
    let mlx = avgCX + 5;
    let mly = PAD.top + 44;   // below the peak label row
    peakPositions.forEach(p => {{ if (Math.abs(p.x - mlx) < mtw + 10) mly += 20; }});
    peakPositions.push({{x: mlx, y: mly}});
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.80)';
    ctx.fillRect(mlx - 2, mly - 13, mtw + 6, 17);
    ctx.fillStyle = color;
    ctx.fillText(muText, mlx, mly);
    ctx.restore();
  }});
}}

// Legend
const legendDiv = document.getElementById('legend');
Object.entries(COLORS).forEach(([label, color]) => {{
  legendDiv.innerHTML +=
    `<span class="leg-item"><span class="leg-line" style="background:${{color}};display:inline-block;"></span>${{label}} (mean + 95% CI)</span>`;
}});

let currentIdx = 0;
let playTimer  = null;

function setEpoch(idx) {{
  currentIdx       = Math.max(0, Math.min(EPOCHS.length - 1, idx));
  slider.value     = currentIdx;
  epochLabel.textContent = 'Epoch ' + EPOCHS[currentIdx];
  drawFrame(currentIdx);
}}

slider.addEventListener('input', () => setEpoch(parseInt(slider.value)));

document.getElementById('btnPlay').addEventListener('click', () => {{
  if (playTimer) return;
  playTimer = setInterval(() => {{
    if (currentIdx >= EPOCHS.length - 1) {{ clearInterval(playTimer); playTimer = null; return; }}
    setEpoch(currentIdx + 1);
  }}, 150);
}});

document.getElementById('btnPause').addEventListener('click', () => {{
  clearInterval(playTimer); playTimer = null;
}});

setEpoch(0);
</script>
</body>
</html>"""

    output_path = os.path.join(PLOTS_DIR, 'distribution_evolution_interactive.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved interactive plot: {output_path}")


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
    plot_scatter_phase_space(master_df, 'LMC Complexity', 'Validation Loss')
    plot_scatter_phase_space(master_df, 'Shannon Entropy', 'Validation Loss')
    plot_scatter_phase_space(master_df, 'Shannon Entropy', 'LMC Complexity')
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
    plot_distributions_faceted(SOURCES)

    # Distribution Summary per Epoch
    plot_distribution_summary_per_epoch(SOURCES)

    # Distribution Evolution (3D, GIF, Interactive)
    plot_distributions_3d_evolution(SOURCES)
    plot_distributions_gif_evolution(SOURCES)
    plot_distributions_interactive(SOURCES)

    # Combined Loss Barplots
    plot_combined_loss_barplots(master_df)
    
    print("Done.")

if __name__ == "__main__":
    main()
