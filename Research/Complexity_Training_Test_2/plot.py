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

# Limits for plotting
# Set to 0 to disable limiting
TOP_X_LOWEST_VAL_LOSS = 30
TOP_X_LOWEST_TEST_LOSS_SHAKESPEARE = 0
EPOCH_OFFSET = 10 # Start epochs at 11 (1 + 10)
MAX_EPOCH = 30 # Maximum epoch to plot (set to 0 to disable)
CONTROL_KEYWORD = 'control min'

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
# We will generate this AFTER finding the sources to ensure maximum distinctness

# List of tuples: (label, folder_path)
# Configure which folders to use for plotting here.
SOURCES = []
if os.path.exists(BASE_OUTPUT_DIR):
    for folder_name in sorted(os.listdir(BASE_OUTPUT_DIR)):
        folder_path = os.path.join(BASE_OUTPUT_DIR, folder_name)
        if os.path.isdir(folder_path):
            # Create a label from the folder name
            # Replace _over_ with / and underscores with spaces
            label = folder_name.replace('_over_', '/').replace('_', ' ')
            SOURCES.append((label, folder_path))
else:
    print(f"Warning: Output directory {BASE_OUTPUT_DIR} does not exist.")

# Generate palette dynamically based on found sources
_source_order = [s[0] for s in SOURCES]

# Separate control and others to assign RED to control
_control_sources = [s for s in _source_order if CONTROL_KEYWORD in s.lower()]
_other_sources = [s for s in _source_order if s not in _control_sources]

n_others = len(_other_sources)

if n_others > 0:
    # Use husl to spread colors evenly across the spectrum for the number of sources we have
    # This ensures maximum distinctness between sources
    _colors = sns.color_palette('husl', n_others)
else:
    _colors = []

SOURCE_PALETTE = {name: color for name, color in zip(_other_sources, _colors)}

# Add control sources as RED
for cs in _control_sources:
    SOURCE_PALETTE[cs] = RED

# Fallback list if dict doesn't work in some seaborn versions (though it should for hue)
SOURCE_COLORS_LIST = list(SOURCE_PALETTE.values())

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# Utility Functions
# ==========================================

def calculate_ci_stats(df, group_cols, metric_col, ci_level=1.96):
    """
    Centralized function to calculate mean and confidence interval for a metric.
    
    Parameters:
    - df: DataFrame with the data
    - group_cols: Column(s) to group by (e.g., ['Source', 'Epoch'] or 'Epoch')
    - metric_col: The metric column to calculate statistics for
    - ci_level: Multiplier for SEM (1.96 for 95% CI, 2.576 for 99% CI)
    
    Returns:
    - DataFrame with mean, sem, and ci columns
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    grouped = df.groupby(group_cols)[metric_col]
    stats = pd.DataFrame({
        'mean': grouped.mean(),
        'sem': grouped.sem(),
    }).reset_index()
    
    stats['ci'] = stats['sem'] * ci_level
    # Replace NaN with 0 for single-run cases
    stats['ci'].fillna(0, inplace=True)
    stats['sem'].fillna(0, inplace=True)
    
    return stats

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
    csv_files = glob.glob(os.path.join(folder_path, 'results*.csv'))
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
    
    if EPOCH_OFFSET > 0 and CONTROL_KEYWORD in label.lower():
        combined['Epoch'] += EPOCH_OFFSET
        
    return combined



# ==========================================
# Plotting Functions
# ==========================================

def filter_data(df):
    """
    Filters the master dataframe based on configuration limits.
    """
    if df is None: return None
    
    # Filter by maximum epoch if set
    if MAX_EPOCH > 0:
        df = df[df['Epoch'] <= MAX_EPOCH]
        print(f"Filtering by Max Epoch: {MAX_EPOCH}")
    
    # If no filters are set, return original
    if TOP_X_LOWEST_VAL_LOSS == 0 and TOP_X_LOWEST_TEST_LOSS_SHAKESPEARE == 0:
        return df

    allowed_sources_sets = []
    
    # Filter by Validation Loss
    if TOP_X_LOWEST_VAL_LOSS > 0:
        if 'Validation Loss' in df.columns:
            # Get best (min) value per run
            best_per_run = df.groupby(['Source', 'Run_ID'])['Validation Loss'].min()
            # Group by Source and mean
            source_val_loss = best_per_run.groupby('Source').mean()
            # Get top X smallest
            top_val_sources = set(source_val_loss.nsmallest(TOP_X_LOWEST_VAL_LOSS).index)
            allowed_sources_sets.append(top_val_sources)
            print(f"Filtering by Top {TOP_X_LOWEST_VAL_LOSS} Val Loss. Kept: {top_val_sources}")
        else:
            print("Warning: 'Validation Loss' column not found. Skipping Val Loss filter.")

    # Filter by Test Loss Shakespeare
    if TOP_X_LOWEST_TEST_LOSS_SHAKESPEARE > 0:
        test_col = 'Test Loss Shakespeare'
        if test_col in df.columns:
            # Get best (min) value per run
            best_per_run = df.groupby(['Source', 'Run_ID'])[test_col].min()
            # Group by Source and mean
            source_test_loss = best_per_run.groupby('Source').mean()
            # Get top X smallest
            top_test_sources = set(source_test_loss.nsmallest(TOP_X_LOWEST_TEST_LOSS_SHAKESPEARE).index)
            allowed_sources_sets.append(top_test_sources)
            print(f"Filtering by Top {TOP_X_LOWEST_TEST_LOSS_SHAKESPEARE} Test Loss Shakespeare. Kept: {top_test_sources}")
        else:
            print(f"Warning: '{test_col}' column not found. Skipping Shakespeare filter.")

    if not allowed_sources_sets:
        print("Warning: No sources selected by filters (or columns missing). Returning all.")
        return df

    # Union of all sets
    final_allowed_sources = set().union(*allowed_sources_sets)

    # Always include 'control' if it exists in the original data
    control_sources = [s for s in df['Source'].unique() if CONTROL_KEYWORD in s.lower()]
    if control_sources:
        for cs in control_sources:
            final_allowed_sources.add(cs)
        print(f"Added control sources to plot: {control_sources}")

    if not final_allowed_sources:
        print("Warning: No sources left after filtering!")
        return None

    print(f"Final sources to plot: {final_allowed_sources}")
    return df[df['Source'].isin(final_allowed_sources)]

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
    
    # Reset index to avoid duplicates which cause issues in seaborn
    combined.reset_index(drop=True, inplace=True)

    # Rename metrics as requested
    combined.rename(columns={
        'Train Loss': 'Training Loss',
        'Val Loss': 'Validation Loss',
        'Weights Entropy': 'Entropy',
        'Weights Disequilibrium': 'Disequilibrium',
        'Weight': 'LMC Weight',
        'Metric Value': 'Model LMC' # Map generic metric value to Model LMC for compatibility
    }, inplace=True)
    
    return combined

def get_starred_df_and_palette(df, metric):
    """
    Returns a new dataframe with starred source names if they are consistently
    below the control source for the given metric, and a corresponding palette.
    """
    # Identify control source(s)
    sources = df['Source'].unique()
    control_sources = [s for s in sources if CONTROL_KEYWORD in s.lower()]
    
    if not control_sources:
        return df, SOURCE_PALETTE
        
    # Use the first control found for comparison
    control_source = control_sources[0]
    
    # Calculate means per epoch for control
    control_data = df[df['Source'] == control_source].groupby('Epoch')[metric].mean()
    
    rename_map = {}
    
    for source in sources:
        if source in control_sources:
            rename_map[source] = source
            continue
            
        source_data = df[df['Source'] == source].groupby('Epoch')[metric].mean()
        
        # Align indices (epochs)
        common_epochs = control_data.index.intersection(source_data.index)
        
        if len(common_epochs) == 0:
            rename_map[source] = source
            continue
            
        # Check condition: source < control for ALL common epochs
        # Using a small tolerance for float comparison if needed, but strict < requested
        is_below = (source_data.loc[common_epochs] < control_data.loc[common_epochs]).all()
        
        if is_below:
            rename_map[source] = f"{source} *"
        else:
            rename_map[source] = source
            
    # Create new DF
    new_df = df.copy()
    new_df['Source'] = new_df['Source'].map(rename_map)
    
    # Create new Palette
    new_palette = {}
    for old_name, new_name in rename_map.items():
        if old_name in SOURCE_PALETTE:
            new_palette[new_name] = SOURCE_PALETTE[old_name]
        else:
            # Fallback
            new_palette[new_name] = '#333333'
            
    return new_df, new_palette

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
        # Apply star logic
        plot_df, plot_palette = get_starred_df_and_palette(master_df, metric)
        
        plt.figure(figsize=(12, 8))
        
        sns.lineplot(
            data=plot_df,
            x='Epoch',
            y=metric,
            hue='Source',
            style='Source',
            dashes=False,
            errorbar=('se', 1.96),
            palette=plot_palette,
            linewidth=1.5,
            alpha=0.85
        )
        
        # Add preoptimization shaded region
        if EPOCH_OFFSET > 0:
            plt.axvspan(plot_df['Epoch'].min(), EPOCH_OFFSET, alpha=0.15, color='gray', label='Preoptimization')
        
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
        
        # Apply star logic
        plot_df, plot_palette = get_starred_df_and_palette(df, 'Generalization Gap')
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=plot_df,
            x='Epoch',
            y='Generalization Gap',
            hue='Source',
            style='Source',
            dashes=False,
            errorbar=('se', 1.96),
            palette=plot_palette,
            linewidth=1.5,
            alpha=0.85
        )
        # Add preoptimization shaded region
        if EPOCH_OFFSET > 0:
            plt.axvspan(plot_df['Epoch'].min(), EPOCH_OFFSET + 1, alpha=0.15, color='gray', label='Preoptimization')
        
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
        palette=SOURCE_PALETTE,
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
    # palette = sns.color_palette('muted', n_colors=len(sources))
    # source_colors = {source: palette[i] for i, source in enumerate(sources)}
    source_colors = SOURCE_PALETTE
    
    dashes_map = {'Training Loss': (2, 2), 'Validation Loss': ""}
    
    sns.lineplot(
        data=df_melted,
        x='Epoch',
        y='Loss',
        hue='Source',
        style='Loss Type',
        dashes=dashes_map,
        palette=SOURCE_PALETTE,
        linewidth=2,
        alpha=0.9,
        legend=False # Disable default legend
    )
    
    # Add preoptimization shaded region
    if EPOCH_OFFSET > 0:
        plt.axvspan(df_melted['Epoch'].min(), EPOCH_OFFSET, alpha=0.15, color='gray', zorder=0)
    
    plt.title('Training vs Validation Loss', fontweight='normal', fontsize=16)
    plt.ylabel('Loss', fontsize=12)
    
    # Custom Legend
    legend_elements = []
    
    # Preoptimization section
    if EPOCH_OFFSET > 0:
        legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Region}$'))
        from matplotlib.patches import Rectangle
        legend_elements.append(Rectangle((0, 0), 1, 1, fc='gray', alpha=0.15, label='Preoptimization'))
        legend_elements.append(Line2D([0], [0], color='none', label=''))
    
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
    
    # Calculate stats for annotations
    stats = df_melted.groupby(['Source', 'Dataset'])['Loss'].agg(['mean', 'sem']).reset_index()
    stats['ci'] = stats['sem'] * 1.96
    
    # Calculate order
    order = df_melted.groupby('Dataset')['Loss'].mean().sort_values().index.tolist()

    ax = sns.barplot(
        data=df_melted,
        x='Dataset',
        y='Loss',
        hue='Source',
        order=order,
        palette=SOURCE_PALETTE, # Use consistent palette
        edgecolor='.2',
        capsize=.1,
        err_kws={'linewidth': 3, 'color': 'black'}
    )
    
    # Add numeric labels
    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height) and height != 0:
            # Find corresponding CI based on height (mean)
            # We need to filter stats by the correct Source and Dataset
            # This is tricky with barplot patches.
            # Simplified approach: just show height
            
            ax.annotate(f'{height:.4f}', 
                        (p.get_x() + p.get_width() / 2., height / 2), 
                        ha='center', va='center', 
                        xytext=(0, 0), 
                        textcoords='offset points',
                        fontsize=16, # Increased font size
                        color='black',
                        fontweight='bold')
    
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



def plot_complexity_performance_path(master_df):
    """2D Path of Complexity vs Performance."""
    if master_df is None: return
    
    if 'Model LMC' not in master_df.columns or 'Validation Loss' not in master_df.columns: return

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=master_df,
        x='Model LMC',
        y='Validation Loss',
        hue='Source',
        sort=False, # Keep epoch order
        palette=SOURCE_PALETTE,
        linewidth=2,
        alpha=0.8,
        errorbar=('se', 1.96),
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
        
        plt.figure(figsize=(14, 8))
        
        # Calculate stats for annotations
        stats = best_per_run.groupby(['Source'])['Best Value'].agg(['mean', 'sem']).reset_index()
        stats['ci'] = stats['sem'] * 1.96
        
        # Calculate order
        order = best_per_run.groupby('Source')['Best Value'].mean().sort_values().index.tolist()

        # Use hue='Source' to ensure consistent coloring with other plots
        # dodge=False because x and hue are the same
        ax = sns.barplot(
            data=best_per_run,
            x='Source',
            y='Best Value',
            hue='Source',
            order=order,
            palette=SOURCE_PALETTE,
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
        plt.xticks(rotation=45, ha='right')
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
        plt.figure(figsize=(14, 8))
        
        # Calculate stats for annotations
        stats = final_epochs_df.groupby(['Source'])[metric].agg(['mean', 'sem']).reset_index()
        stats['ci'] = stats['sem'] * 1.96
        
        # Calculate order
        order = final_epochs_df.groupby('Source')[metric].mean().sort_values().index.tolist()

        # Use hue='Source' to ensure consistent coloring with other plots
        # dodge=False because x and hue are the same
        ax = sns.barplot(
            data=final_epochs_df,
            x='Source',
            y=metric,
            hue='Source',
            order=order,
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
        plt.xticks(rotation=45, ha='right')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        sns.despine(left=True, bottom=True)
        
        safe_metric_name = re.sub(r'[^\w\s-]', '', metric).strip().replace(' ', '_')
        output_path = os.path.join(PLOTS_DIR, f'barplot_final_{safe_metric_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")

def plot_complexity_metrics_comparison(master_df):
    """Plots Model LMC, Entropy, and Disequilibrium comparison with different scales."""
    if master_df is None: return
    
    cols = ['Model LMC', 'Entropy', 'Disequilibrium']
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
            # Calculate statistics using centralized function
            stats = calculate_ci_stats(subset, 'Epoch', metric)
            
            # Plot mean line
            ax.plot(stats['Epoch'], stats['mean'], 
                    color=color, linestyle=source_styles.get(source, '-'),
                    linewidth=2, alpha=0.8)
            
            # Add confidence interval shading
            ax.fill_between(stats['Epoch'], 
                          stats['mean'] - stats['ci'], 
                          stats['mean'] + stats['ci'],
                          color=color, alpha=0.15)
            
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
    
    # Add preoptimization shaded region
    if EPOCH_OFFSET > 0:
        ax1.axvspan(master_df['Epoch'].min(), EPOCH_OFFSET, alpha=0.15, color='gray', zorder=0)
    
    plt.title('Complexity Metrics Comparison', fontweight='normal', fontsize=16)
    
    # Custom Legend
    legend_elements = []
    
    # Preoptimization section
    if EPOCH_OFFSET > 0:
        from matplotlib.patches import Rectangle
        legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Region}$'))
        legend_elements.append(Rectangle((0, 0), 1, 1, fc='gray', alpha=0.15, label='Preoptimization'))
        legend_elements.append(Line2D([0], [0], color='none', label=''))
    
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
        
        # Calculate statistics using centralized function
        stats_weight = calculate_ci_stats(subset, 'Epoch', 'LMC Weight')
        
        ax1.plot(stats_weight['Epoch'], stats_weight['mean'], 
                 color=color_weight, linestyle=source_styles[source], linewidth=2, label=f"{source}")
        ax1.fill_between(stats_weight['Epoch'], 
                        stats_weight['mean'] - stats_weight['ci'], 
                        stats_weight['mean'] + stats_weight['ci'], 
                        color=color_weight, alpha=0.2)
                 
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('LMC Weight', color=color_weight, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_weight)
    ax1.spines['left'].set_color(color_weight)
    ax1.spines['left'].set_linewidth(2)
    
    # Plot Slope on right axis (ax2)
    ax2 = ax1.twinx()
    for source in sources:
        subset = master_df[master_df['Source'] == source]
        
        # Calculate statistics using centralized function
        stats_slope = calculate_ci_stats(subset, 'Epoch', 'Val Error Slope')
        
        ax2.plot(stats_slope['Epoch'], stats_slope['mean'], 
                 color=color_slope, linestyle=source_styles[source], linewidth=2, alpha=0.8)
        ax2.fill_between(stats_slope['Epoch'], 
                        stats_slope['mean'] - stats_slope['ci'], 
                        stats_slope['mean'] + stats_slope['ci'], 
                        color=color_slope, alpha=0.2)
                 
    ax2.set_ylabel('Val Error Slope', color=color_slope, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_slope)
    ax2.spines['right'].set_color(color_slope)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['right'].set_position(('outward', 10)) # Offset slightly
    
    # Add zero line for slope
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Add preoptimization shaded region
    if EPOCH_OFFSET > 0:
        ax1.axvspan(master_df['Epoch'].min(), EPOCH_OFFSET, alpha=0.15, color='gray', zorder=0)

    plt.title('Adaptive Mechanism: LMC Weight vs Val Error Slope', fontweight='normal', fontsize=16)
    
    # Custom Legend
    legend_elements = []
    
    # Preoptimization section
    if EPOCH_OFFSET > 0:
        from matplotlib.patches import Rectangle
        legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Region}$'))
        legend_elements.append(Rectangle((0, 0), 1, 1, fc='gray', alpha=0.15, label='Preoptimization'))
        legend_elements.append(Line2D([0], [0], color='none', label=''))
    
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
    # Increased figure size to avoid overlapping
    fig, axes = plt.subplots(1, n_sources, figsize=(8 * n_sources, 8), sharex=True)
    if n_sources == 1: axes = [axes]
    
    # Define colors for metrics - Use consistent Blue/Orange
    color_weight = BLUE
    color_slope = ORANGE
    
    for i, source in enumerate(sources):
        ax1 = axes[i]
        subset = master_df[master_df['Source'] == source]
        
        # Group by Epoch to get mean (if multiple runs)
        # subset_mean = subset.groupby('Epoch')[required].mean()
        
        # Plot LMC Weight (Left Axis)
        stats_weight = calculate_ci_stats(subset, 'Epoch', 'LMC Weight')
        
        ax1.plot(stats_weight['Epoch'], stats_weight['mean'], 
                 color=color_weight, linewidth=4.0, label='LMC Weight')
        ax1.fill_between(stats_weight['Epoch'], 
                         stats_weight['mean'] - stats_weight['ci'], 
                         stats_weight['mean'] + stats_weight['ci'], 
                         color=color_weight, alpha=0.2)
        
        ax1.set_xlabel('Epoch', fontsize=18, fontweight='bold')
        ax1.set_ylabel('LMC Weight', color=color_weight, fontsize=18, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_weight, labelsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.set_title(f'{source}', fontsize=20, fontweight='bold', pad=20) # Added padding
        ax1.grid(True, alpha=0.3)
        
        # Plot Slope (Right Axis)
        ax2 = ax1.twinx()
        
        stats_slope = calculate_ci_stats(subset, 'Epoch', 'Val Error Slope')
        
        ax2.plot(stats_slope['Epoch'], stats_slope['mean'], 
                 color=color_slope, linewidth=4.0, linestyle='--', alpha=1.0, label='Val Error Slope')
        ax2.fill_between(stats_slope['Epoch'], 
                         stats_slope['mean'] - stats_slope['ci'], 
                         stats_slope['mean'] + stats_slope['ci'],
                         color=color_slope, alpha=0.2)
                 
        ax2.set_ylabel('Val Error Slope', color=color_slope, fontsize=18, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_slope, labelsize=16)
        
        # Add zero line for slope
        ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        # Add preoptimization shaded region
        if EPOCH_OFFSET > 0:
            ax1.axvspan(subset['Epoch'].min(), EPOCH_OFFSET, alpha=0.15, color='gray', zorder=0)
        
        # Spines
        ax1.spines['left'].set_color(color_weight)
        ax1.spines['left'].set_linewidth(3)
        ax2.spines['right'].set_color(color_slope)
        ax2.spines['right'].set_linewidth(3)
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    # Common title
    # fig.suptitle('Adaptive Mechanism by Source', fontsize=24, fontweight='bold', y=0.98) # Adjusted y position
    fig.subplots_adjust(top=0.85, wspace=0.4, bottom=0.15, left=0.1, right=0.9) # Adjusted margins
    
    # Legend
    legend_elements = []
    if EPOCH_OFFSET > 0:
        from matplotlib.patches import Rectangle
        legend_elements.append(Rectangle((0, 0), 1, 1, fc='gray', alpha=0.15, label='Preoptimization'))
    legend_elements.extend([
        Line2D([0], [0], color=color_weight, lw=4.0, label='LMC Weight'),
        Line2D([0], [0], color=color_slope, lw=4.0, linestyle='--', label='Val Error Slope')
    ])
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False, fontsize=18)

    output_path = os.path.join(PLOTS_DIR, 'mechanism_adaptive_control_faceted.png')
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
        palette=SOURCE_PALETTE,
        s=50
    )
    
    # Add trend lines
    sources = df['Source'].unique()
    # palette = sns.color_palette('muted', n_colors=len(sources))
    
    for i, source in enumerate(sources):
        subset = df[df['Source'] == source]
        # Drop NaNs
        subset = subset.dropna(subset=['Model LMC', 'Generalization Gap'])
        
        if len(subset) > 1 and subset['Model LMC'].nunique() > 1:
            try:
                # Simple linear regression for trend
                z = np.polyfit(subset['Model LMC'], subset['Generalization Gap'], 1)
                p = np.poly1d(z)
                
                x_range = np.linspace(subset['Model LMC'].min(), subset['Model LMC'].max(), 100)
                plt.plot(x_range, p(x_range), color=SOURCE_PALETTE.get(source, 'black'), linestyle='--', alpha=0.5, linewidth=1.5)
            except Exception as e:
                print(f"Could not fit trend line for {source}: {e}")

    plt.title('Overfitting vs Complexity (Gen Gap vs LMC)', fontweight='normal', fontsize=16)
    plt.xlabel('Model LMC', fontsize=12)
    plt.ylabel('Generalization Gap (Val - Train)', fontsize=12)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(PLOTS_DIR, 'scatter_overfitting_vs_complexity.png')
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
    # palette = sns.color_palette('muted', n_colors=len(sources))
    # source_colors = {source: palette[i] for i, source in enumerate(sources)}
    source_colors = SOURCE_PALETTE
    
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
        palette=SOURCE_PALETTE,
        linewidth=1.5,
        alpha=0.9,
        legend=False
    )
    
    # Add preoptimization shaded region
    if EPOCH_OFFSET > 0:
        plt.axvspan(df_melted['Epoch'].min(), EPOCH_OFFSET, alpha=0.15, color='gray', zorder=0)
    
    plt.title('Training vs Validation vs Test Loss', fontweight='normal', fontsize=16)
    plt.ylabel('Loss', fontsize=12)
    
    # Custom Legend
    legend_elements = []
    
    # Preoptimization section
    if EPOCH_OFFSET > 0:
        from matplotlib.patches import Rectangle
        legend_elements.append(Line2D([0], [0], color='none', label=r'$\bf{Region}$'))
        legend_elements.append(Rectangle((0, 0), 1, 1, fc='gray', alpha=0.15, label='Preoptimization'))
        legend_elements.append(Line2D([0], [0], color='none', label=''))
    
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
        value_name='Loss'
    )
    
    # Facet by Loss Type
    # Changed to col_wrap=2 for 2x2 layout
    g = sns.FacetGrid(df_melted, col="Loss Type", hue="Source", col_wrap=2, 
                      height=6, aspect=1.4, sharey=False, palette=SOURCE_PALETTE)
    g.map(sns.lineplot, "Epoch", "Loss", errorbar=('se', 1.96), linewidth=4.0)
    g.add_legend(title=r'$\bf{Source}$', fontsize=18, title_fontsize=20)
    g.set_titles("{col_name}", size=20, fontweight='bold')
    g.set_axis_labels("Epoch", "Loss", fontsize=18, fontweight='bold')
    
    # Improve tick readability and add preoptimization shade
    for ax in g.axes.flat:
        ax.tick_params(labelsize=16)
        # Add preoptimization shaded region
        if EPOCH_OFFSET > 0:
            ax.axvspan(df_melted['Epoch'].min(), EPOCH_OFFSET, alpha=0.15, color='gray', zorder=0)
        # Add preoptimization shaded region
        if EPOCH_OFFSET > 0:
            ax.axvspan(df_melted['Epoch'].min(), EPOCH_OFFSET, alpha=0.15, color='gray', zorder=0)

    # g.fig.suptitle('Loss Comparison by Type', fontsize=24, fontweight='bold')
    g.fig.subplots_adjust(top=0.9)
    
    output_path = os.path.join(PLOTS_DIR, 'comparison_train_val_test_loss_faceted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_complexity_metrics_faceted(master_df):
    """Plots complexity metrics in faceted subplots."""
    if master_df is None: return
    
    cols = ['Model LMC', 'Entropy', 'Disequilibrium']
    available_cols = [c for c in cols if c in master_df.columns]
    
    if not available_cols: return

    df_melted = master_df.melt(
        id_vars=['Epoch', 'Source', 'Run_ID'], 
        value_vars=available_cols,
        var_name='Metric', 
        value_name='Value'
    )
    
    g = sns.FacetGrid(df_melted, col="Metric", hue="Source", height=6, aspect=1.4, sharey=False, palette=SOURCE_PALETTE)
    g.map(sns.lineplot, "Epoch", "Value", errorbar=('se', 1.96), linewidth=4.0)
    g.add_legend(title=r'$\bf{Source}$', fontsize=18, title_fontsize=20)
    g.set_titles("{col_name}", size=20, fontweight='bold')
    g.set_axis_labels("Epoch", "Value", fontsize=18, fontweight='bold')
    
    # Add preoptimization shade and improve tick readability
    for ax in g.axes.flat:
        ax.tick_params(labelsize=16)
        # Add preoptimization shaded region
        if EPOCH_OFFSET > 0:
            ax.axvspan(df_melted['Epoch'].min(), EPOCH_OFFSET, alpha=0.15, color='gray', zorder=0)

    # g.fig.suptitle('Complexity Metrics Comparison', fontsize=24, fontweight='bold')
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
    
    # Filter data based on config
    master_df = filter_data(master_df)

    # 1. Plot Metrics
    plot_all_metrics(master_df)
    
    # 3. New Plots
    plot_generalization_gap(master_df)
    plot_scatter_phase_space(master_df, 'Model LMC', 'Validation Loss')
    plot_train_vs_val_loss(master_df)
    plot_test_loss_barplot(master_df)
    plot_complexity_performance_path(master_df)
    plot_global_best_metrics_barplot(master_df)
    plot_final_metrics_barplot(master_df)
    plot_train_val_test_loss_comparison(master_df)
    plot_complexity_metrics_comparison(master_df)
    plot_adaptive_mechanism(master_df)
    plot_overfitting_vs_complexity(master_df)
    
    # Faceted Plots
    plot_train_val_test_loss_faceted(master_df)
    plot_complexity_metrics_faceted(master_df)
    plot_adaptive_mechanism_faceted(master_df)
    
    print("Done.")

if __name__ == "__main__":
    main()
