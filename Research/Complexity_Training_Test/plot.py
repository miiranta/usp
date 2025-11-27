import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
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
sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

# List of tuples: (label, folder_path)
# Configure which folders to use for plotting here.
SOURCES = [
    ('output_0.0', os.path.join(BASE_OUTPUT_DIR, 'output_0.0')),
    ('output_0.0_test', os.path.join(BASE_OUTPUT_DIR, 'output_0.0_test')),
    ('output_1.0', os.path.join(BASE_OUTPUT_DIR, 'output_1.0')),
    ('output_1.0_test', os.path.join(BASE_OUTPUT_DIR, 'output_1.0_test')),
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
    
    for csv_file in csv_files:
        if 'AGGREGATE' in csv_file:
            continue
            
        df = parse_run_csv(csv_file)
        if df is not None and not df.empty:
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

def plot_all_metrics(sources):
    """
    Generates plots for each metric comparing all sources using Seaborn.
    """
    all_data = []
    for label, folder_path in sources:
        print(f"Loading data for: {label}")
        df = load_source_data(folder_path, label)
        if df is not None:
            all_data.append(df)
            
    if not all_data:
        print("No data found to plot metrics.")
        return

    master_df = pd.concat(all_data)
    
    # Identify all metrics (columns except Epoch and Source)
    metrics = [c for c in master_df.columns if c not in ['Epoch', 'Source']]
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Plot using Seaborn
        # errorbar=('se', 1.96) corresponds to 95% CI for normal distribution
        sns.lineplot(
            data=master_df,
            x='Epoch',
            y=metric,
            hue='Source',
            style='Source',
            dashes=False,
            errorbar=('se', 1.96),
            palette='tab10',
            linewidth=2.5,
            alpha=0.9
        )
        
        plt.title(f'{metric} Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
        sns.despine()
        
        # Sanitize filename
        safe_metric_name = re.sub(r'[^\w\s-]', '', metric).strip().replace(' ', '_')
        output_path = os.path.join(PLOTS_DIR, f'metric_{safe_metric_name}.png')
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
    palette = sns.color_palette('tab10', n_colors=len(sources))
    
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
                # Plot line
                lbl = f"{label} (Ep {target_epoch})" if i == 0 else None
                
                # Use matplotlib directly for fine-grained control over fill and line
                # but use seaborn's color
                plt.plot(df['Bin_Center'], df['Probability'], color=color, alpha=1.0, linewidth=2.5, label=lbl)
                plt.fill_between(df['Bin_Center'], df['Probability'], color=color, alpha=0.3)
                
    plt.title(title, fontweight='bold')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    
    output_path = os.path.join(PLOTS_DIR, f'{filename_suffix}_overlay.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    print(f"Starting plot generation...")
    print(f"Sources: {[s[0] for s in SOURCES]}")
    
    if not SOURCES:
        print("No sources configured. Exiting.")
        return

    # 1. Plot Metrics
    plot_all_metrics(SOURCES)
    
    # 2. Plot Distributions
    plot_distributions_overlay(SOURCES)
    
    print("Done.")

if __name__ == "__main__":
    main()
