import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ==========================================
# Configuration
# ==========================================

# Define the base output directory relative to this script
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

# List of tuples: (label, folder_path)
# Configure which folders to use for plotting here.
SOURCES = [
    ('output_0.0', os.path.join(BASE_OUTPUT_DIR, 'output_0.0')),
    ('output_0.0_test', os.path.join(BASE_OUTPUT_DIR, 'output_0.0_test')),
    ('output_1.0', os.path.join(BASE_OUTPUT_DIR, 'output_1.0')),
    ('output_1.0_test', os.path.join(BASE_OUTPUT_DIR, 'output_1.0_test')),
]

# if os.path.exists(BASE_OUTPUT_DIR):
#     # Find all folders starting with output_
#     candidates = glob.glob(os.path.join(BASE_OUTPUT_DIR, 'output_*'))
#     for folder_path in candidates:
#         if os.path.isdir(folder_path):
#             folder_name = os.path.basename(folder_path)
#             # Check if it matches the pattern output_x.x (digits and dots)
#             # User mentioned "output_x.x", so we'll be inclusive of anything starting with output_
#             # We can use the suffix as the label
#             label = folder_name
#             SOURCES.append((label, folder_path))

# Sort sources by label to ensure consistent color assignment
SOURCES.sort(key=lambda x: x[0])

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

def aggregate_metrics(folder_path):
    """
    Reads all run CSVs in the folder (ignoring AGGREGATE),
    and calculates mean, std, and count for each metric per epoch.
    """
    # Find all csv files that look like z_loss_test_results_transformers-*.csv
    # and do NOT end with AGGREGATE.csv
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
        
    # Concatenate all runs
    all_runs = pd.concat(run_dfs)
    
    # Group by Epoch and calculate stats
    # We want mean, std, and count (to calculate confidence interval)
    aggregated = all_runs.groupby('Epoch').agg(['mean', 'std', 'count'])
    
    return aggregated

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
    Generates plots for each metric comparing all sources.
    """
    # Pre-load and aggregate data for all sources
    source_data = {}
    for label, folder_path in sources:
        print(f"Processing metrics for: {label}")
        agg_df = aggregate_metrics(folder_path)
        if agg_df is not None:
            source_data[label] = agg_df
            
    if not source_data:
        print("No data found to plot metrics.")
        return

    # Identify all metrics (columns at level 0 of the multi-index)
    # The columns are MultiIndex: (Metric, Stat)
    first_df = next(iter(source_data.values()))
    metrics = first_df.columns.levels[0]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for label, df in source_data.items():
            if metric not in df:
                continue
                
            stats = df[metric]
            epochs = stats.index
            mean = stats['mean']
            std = stats['std']
            count = stats['count']
            
            # Calculate 95% Confidence Interval
            # CI = 1.96 * (std / sqrt(n))
            # If count is 1, std is NaN, so fill with 0
            std = std.fillna(0)
            ci = 1.96 * (std / np.sqrt(count))
            
            # Use consistent colors
            # Find index of label in original sources list to ensure consistent coloring
            try:
                color_idx = [s[0] for s in sources].index(label)
            except ValueError:
                color_idx = 0
            color = plt.cm.tab10(color_idx % 10)

            line, = plt.plot(epochs, mean, label=label, color=color)
            plt.fill_between(epochs, mean - ci, mean + ci, color=color, alpha=0.2)
            
        plt.title(f'{metric} Comparison')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sanitize filename
        safe_metric_name = re.sub(r'[^\w\s-]', '', metric).strip().replace(' ', '_')
        output_path = os.path.join(PLOTS_DIR, f'metric_{safe_metric_name}.png')
        plt.savefig(output_path, dpi=300)
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
    plt.figure(figsize=(12, 7))
    
    # Use a color cycle
    # Use standard tab10 colors which are designed to be distinct
    colors = [plt.cm.tab10(i % 10) for i in range(len(sources))]
    
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
            
        color = colors[idx]
        
        # Plot each run
        for i, fpath in enumerate(files):
            df = load_distribution_file(fpath)
            if df is not None:
                # Plot line
                # Only add label for the first run to avoid legend clutter
                lbl = f"{label} (Ep {target_epoch})" if i == 0 else None
                # Use a thicker line for individual runs
                plt.plot(df['Bin_Center'], df['Probability'], color=color, alpha=1.0, linewidth=2.0, label=lbl)
                plt.fill_between(df['Bin_Center'], df['Probability'], color=color, alpha=0.4)
                
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(PLOTS_DIR, f'{filename_suffix}_overlay.png')
    plt.savefig(output_path, dpi=300)
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
