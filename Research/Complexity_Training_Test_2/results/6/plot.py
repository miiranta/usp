import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

class PlotConfig:
    # Input/Output
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(_BASE_DIR, "output")
    PLOTS_DIR = os.path.join(_BASE_DIR, "plots")
    
    # Plot Styling
    STYLE = "whitegrid"
    CONTEXT = "paper"
    PALETTE = "viridis"   # For preconditioning levels
    CONTROL_COLOR = "black"
    CONTROL_STYLE = "--"
    
    # Figure Sizes
    FIG_SIZE_WIDE = (12, 6)
    FIG_SIZE_SQUARE = (8, 8)
    DPI = 300
    
    # Config
    CI_LEVEL = 0.95
    SMOOTHING_WINDOW = 1  # No smoothing by default
    
    # Text
    FONT_SCALE = 1.2
    
def configure_plotting():
    """Apply global plotting styles"""
    sns.set_theme(style=PlotConfig.STYLE, context=PlotConfig.CONTEXT, font_scale=PlotConfig.FONT_SCALE)
    if not os.path.exists(PlotConfig.PLOTS_DIR):
        os.makedirs(PlotConfig.PLOTS_DIR)

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

def load_data(input_dir=PlotConfig.INPUT_DIR):
    """
    Reads all CSV files from the input directory structure.
    Expected structure: output/{config_name}/results_run_{id}.csv
    """
    all_files = glob.glob(os.path.join(input_dir, "*", "*.csv"))
    
    if not all_files:
        print(f"No CSV files found in {input_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(all_files)} files. Loading...")
    
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # Ensure critical columns exist
            if 'val_loss' not in df.columns:
                continue
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)
    
    # Fill standard NaNs if any (e.g., metric_val might be 0 or NaN for control)
    full_df['metric_name'] = full_df['metric_name'].fillna('unknown')
    
    return full_df

def get_aggregated_data(df, group_cols, metric_col):
    """
    Aggregates data by calculating mean and CI for a specific metric.
    Returns a DataFrame with mean, and bounds for CI.
    """
    # Filter for valid values
    data = df.dropna(subset=[metric_col])
    
    # GroupBy
    grouped = data.groupby(group_cols)[metric_col]
    
    agg_df = grouped.agg(['mean', 'count', 'std']).reset_index()
    
    # Calculate 95% Confidence Interval
    # CI = 1.96 * (std / sqrt(n))
    agg_df['ci'] = 1.96 * (agg_df['std'] / np.sqrt(agg_df['count']))
    agg_df['lower'] = agg_df['mean'] - agg_df['ci']
    agg_df['upper'] = agg_df['mean'] + agg_df['ci']
    
    return agg_df

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_learning_curves(df):
    """
    Creates learning curve plots organized by preconditioning level.
    For each P value, shows all metrics compared to control.
    Generates separate plots for training and validation curves.
    """
    print("Generating Learning Curves (by Preconditioning Level)...")
    
    val_data = df[df['log_type'] == 'validation'].copy()
    if val_data.empty:
        print("No validation data found.")
        return

    control_df = val_data[val_data['metric_name'] == 'control']
    experiment_df = val_data[val_data['metric_name'] != 'control']
    
    metrics = sorted(experiment_df['metric_name'].unique())
    precond_levels = sorted(experiment_df['precond_batches'].unique())
    
    if not metrics:
        print("No experiment metrics found.")
        return
    
    # Color palette for metrics
    metric_colors = sns.color_palette("husl", n_colors=len(metrics))
    metric_color_map = {metric: metric_colors[i] for i, metric in enumerate(metrics)}
    
    # === VALIDATION CURVES ===
    n_cols = min(3, len(precond_levels))
    n_rows = int(np.ceil(len(precond_levels) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if len(precond_levels) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    control_agg_val = get_aggregated_data(control_df, ['epoch'], 'val_loss')
    
    for idx, pb in enumerate(precond_levels):
        ax = axes[idx]
        
        # Plot control
        ax.plot(control_agg_val['epoch'], control_agg_val['mean'], 
                label='Control', color=PlotConfig.CONTROL_COLOR, 
                linestyle=PlotConfig.CONTROL_STYLE, linewidth=2.5, zorder=10)
        ax.fill_between(control_agg_val['epoch'], control_agg_val['lower'], control_agg_val['upper'], 
                        color=PlotConfig.CONTROL_COLOR, alpha=0.1)
        
        # Plot all metrics for this P
        for metric in metrics:
            metric_data = experiment_df[(experiment_df['metric_name'] == metric) & 
                                       (experiment_df['precond_batches'] == pb)]
            if metric_data.empty:
                continue
            
            subset_agg = get_aggregated_data(metric_data, ['epoch'], 'val_loss')
            
            ax.plot(subset_agg['epoch'], subset_agg['mean'], 
                   label=metric, color=metric_color_map[metric], linewidth=1.5, alpha=0.8)
            ax.fill_between(subset_agg['epoch'], subset_agg['lower'], subset_agg['upper'], 
                           color=metric_color_map[metric], alpha=0.1)
        
        ax.set_title(f"Precond = {pb} batches", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(precond_levels), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Validation Loss: All Metrics by Preconditioning Level", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(PlotConfig.PLOTS_DIR, "validation_curves_by_precond.png"), 
                dpi=PlotConfig.DPI, bbox_inches='tight')
    plt.close()
    
    # === TRAINING CURVES ===
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if len(precond_levels) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    control_agg_train = get_aggregated_data(control_df, ['epoch'], 'train_loss')
    
    for idx, pb in enumerate(precond_levels):
        ax = axes[idx]
        
        # Plot control
        ax.plot(control_agg_train['epoch'], control_agg_train['mean'], 
                label='Control', color=PlotConfig.CONTROL_COLOR, 
                linestyle=PlotConfig.CONTROL_STYLE, linewidth=2.5, zorder=10)
        ax.fill_between(control_agg_train['epoch'], control_agg_train['lower'], control_agg_train['upper'], 
                        color=PlotConfig.CONTROL_COLOR, alpha=0.1)
        
        # Plot all metrics for this P
        for metric in metrics:
            metric_data = experiment_df[(experiment_df['metric_name'] == metric) & 
                                       (experiment_df['precond_batches'] == pb)]
            if metric_data.empty:
                continue
            
            subset_agg = get_aggregated_data(metric_data, ['epoch'], 'train_loss')
            
            ax.plot(subset_agg['epoch'], subset_agg['mean'], 
                   label=metric, color=metric_color_map[metric], linewidth=1.5, alpha=0.8)
            ax.fill_between(subset_agg['epoch'], subset_agg['lower'], subset_agg['upper'], 
                           color=metric_color_map[metric], alpha=0.1)
        
        ax.set_title(f"Precond = {pb} batches", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(precond_levels), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Training Loss: All Metrics by Preconditioning Level", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(PlotConfig.PLOTS_DIR, "training_curves_by_precond.png"), 
                dpi=PlotConfig.DPI, bbox_inches='tight')
    plt.close()
    
    # Export data to CSV
    csv_data_val = []
    csv_data_train = []
    
    control_agg_val_csv = get_aggregated_data(control_df, ['epoch'], 'val_loss')
    control_agg_train_csv = get_aggregated_data(control_df, ['epoch'], 'train_loss')
    
    for _, row in control_agg_val_csv.iterrows():
        csv_data_val.append({
            'metric_name': 'control',
            'precond_batches': 0,
            'epoch': row['epoch'],
            'mean': row['mean'],
            'ci': row['ci']
        })
    
    for _, row in control_agg_train_csv.iterrows():
        csv_data_train.append({
            'metric_name': 'control',
            'precond_batches': 0,
            'epoch': row['epoch'],
            'mean': row['mean'],
            'ci': row['ci']
        })
    
    for metric in metrics:
        for pb in precond_levels:
            metric_data = experiment_df[(experiment_df['metric_name'] == metric) & 
                                       (experiment_df['precond_batches'] == pb)]
            if not metric_data.empty:
                val_agg = get_aggregated_data(metric_data, ['epoch'], 'val_loss')
                train_agg = get_aggregated_data(metric_data, ['epoch'], 'train_loss')
                
                for _, row in val_agg.iterrows():
                    csv_data_val.append({
                        'metric_name': metric,
                        'precond_batches': pb,
                        'epoch': row['epoch'],
                        'mean': row['mean'],
                        'ci': row['ci']
                    })
                
                for _, row in train_agg.iterrows():
                    csv_data_train.append({
                        'metric_name': metric,
                        'precond_batches': pb,
                        'epoch': row['epoch'],
                        'mean': row['mean'],
                        'ci': row['ci']
                    })
    
    pd.DataFrame(csv_data_val).to_csv(os.path.join(PlotConfig.PLOTS_DIR, "validation_curves_by_precond.csv"), index=False)
    pd.DataFrame(csv_data_train).to_csv(os.path.join(PlotConfig.PLOTS_DIR, "training_curves_by_precond.csv"), index=False)


def plot_efficiency_gap(df):
    """
    Plots the distance in batches between best control val loss 
    and when the metrics reached it.
    """
    print("Generating Efficiency Gap Analysis...")
    
    val_data = df[df['log_type'] == 'validation']
    
    # 1. Determine Target Loss (from mean of best control runs, or best of mean control curve)
    # Approach: Calculate mean curve for control, find min value.
    control_df = val_data[val_data['metric_name'] == 'control']
    control_agg = get_aggregated_data(control_df, ['epoch'], 'val_loss')
    target_loss = control_agg['mean'].min()
    target_loss_std = control_agg['std'].mean() # Just for reference
    
    print(f"  Target Control Loss (Best of Mean Curve): {target_loss:.4f}")
    
    # 2. Find time-to-target for every single RUN of every experiment
    # For robust statistics, we calculate simple "epochs to target" or "batches to target"
    # Batches is better. 'batches_processed' column.
    
    results = []
    
    # Iterate over every run
    # Group by Metric, Precond, RunID
    groups = df.groupby(['metric_name', 'precond_batches', 'run_id'])
    
    for (metric, precond, run_id), group in groups:
        if metric == 'control': continue
        
        # Sort by batches
        run_data = group.sort_values('batches_processed')
        
        # Check if validation data or batch data is better? 
        # validation `val_loss` is only once per epoch. 
        # `ce_loss` is available batch-wise but noisy.
        # User asked for "best control VAL loss", so let's stick to val_loss (epoch resolution).
        
        # Filter for validation entries
        run_val = run_data[run_data['log_type'] == 'validation']
        
        # Find first epoch where val_loss <= target_loss
        reached = run_val[run_val['val_loss'] <= target_loss]
        
        if not reached.empty:
            first_occurrence = reached.iloc[0]
            batches_to_target = first_occurrence['batches_processed']
            results.append({
                'metric_name': metric,
                'precond_batches': precond,
                'run_id': run_id,
                'batches_to_target': batches_to_target,
                'reached': True
            })
        else:
            # Did not reach target
            results.append({
                'metric_name': metric,
                'precond_batches': precond,
                'run_id': run_id,
                'batches_to_target': np.nan, # Handle later
                'reached': False
            })
            
    res_df = pd.DataFrame(results)
    
    if res_df.empty:
        print("  No runs reached the target loss of control.")
        return

    # Visualizing
    # We want to see: For each metric, how fast did it reach it vs precond batches?
    # Use only reached runs? Or handle 'not reached'?
    # For now, drop not reached or set to max. Let's drop for "Speed" analysis.
    
    plot_df = res_df[res_df['reached'] == True]
    
    if plot_df.empty:
         print("  No runs reached the target.")
         return
         
    # Aggregate for plot
    # Mean batches to target w/ CI
    summary = plot_df.groupby(['metric_name', 'precond_batches'])['batches_to_target'].agg(['mean', 'count', 'std']).reset_index()
    summary['ci'] = 1.96 * (summary['std'] / np.sqrt(summary['count']))
    
    # Plot
    # Bar chart: X=Metric, Hue=Precond, Y=Batches
    
    plt.figure(figsize=PlotConfig.FIG_SIZE_WIDE)
    ax = sns.barplot(data=plot_df, x='metric_name', y='batches_to_target', hue='precond_batches', 
                palette=PlotConfig.PALETTE, errorbar=('ci', 95), capsize=.1)
    
    # Add value labels with CI on bars
    for container in ax.containers:
        # Get the bars and compute labels with CI
        labels = []
        for bar in container:
            height = bar.get_height()
            if not np.isnan(height):
                # Find matching data to get CI
                labels.append(f'{height:.0f}')
            else:
                labels.append('')
        ax.bar_label(container, labels=labels, fontsize=8, padding=3)
    
    # Calculate Control Best Batches correctly
    best_idx = control_agg['mean'].idxmin()
    best_epoch = control_agg.loc[best_idx, 'epoch']
    # Get approximate batches for that epoch from original data (taking mean)
    control_best_batches = control_df[control_df['epoch'] == best_epoch]['batches_processed'].mean()
    
    plt.axhline(y=control_best_batches, 
                color='red', linestyle='--', label=f'Control Best (approx)', linewidth=2)
    
    plt.title(f"Efficiency: Batches to Reach Control Best Loss ({target_loss:.4f})")
    plt.xlabel("Metric")
    plt.ylabel("Batches Processed")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Precond Batches")
    plt.tight_layout()
    plt.savefig(os.path.join(PlotConfig.PLOTS_DIR, "efficiency_gap.png"), dpi=PlotConfig.DPI)
    plt.close()
    
    # Export data to CSV
    summary['control_best_batches'] = control_best_batches
    summary['target_loss'] = target_loss
    summary.to_csv(os.path.join(PlotConfig.PLOTS_DIR, "efficiency_gap.csv"), index=False)


def plot_best_val_losses(df):
    """
    Graph with all best val losses.
    """
    print("Generating Best Loss Summary...")
    
    # For each run, find min val_loss
    val_data = df[df['log_type'] == 'validation']
    best_losses = val_data.groupby(['metric_name', 'precond_batches', 'run_id'])['val_loss'].min().reset_index()
    
    plt.figure(figsize=PlotConfig.FIG_SIZE_WIDE)
    
    # Sort for better visuals: Control first, then others
    best_losses['sort_key'] = best_losses['metric_name'].apply(lambda x: ' AAA_Control' if x == 'control' else x)
    best_losses = best_losses.sort_values('sort_key')
    
    ax = sns.barplot(data=best_losses, x='metric_name', y='val_loss', hue='precond_batches',
                palette='magma', errorbar=('ci', 95), capsize=.1)
    
    # Calculate aggregated statistics for CI labels
    agg_stats = best_losses.groupby(['metric_name', 'precond_batches'])['val_loss'].agg(['mean', 'std', 'count']).reset_index()
    agg_stats['ci'] = 1.96 * (agg_stats['std'] / np.sqrt(agg_stats['count']))
    
    # Add value labels with CI on bars
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            if not np.isnan(height):
                # Find the CI for this bar (approximate match)
                matching = agg_stats[np.abs(agg_stats['mean'] - height) < 0.001]
                if len(matching) > 0:
                    ci = matching.iloc[0]['ci']
                    labels.append(f'{height:.3f}\n±{ci:.3f}')
                else:
                    labels.append(f'{height:.3f}')
            else:
                labels.append('')
        ax.bar_label(container, labels=labels, fontsize=7, padding=3)
    
    plt.title("Best Validation Loss Achieved")
    plt.xlabel("Metric")
    plt.ylabel("Minimum Validation Loss")
    plt.ylim(bottom=best_losses['val_loss'].min() * 0.9) # Zoom in a bit but don't mislead
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Precond Batches")
    plt.tight_layout()
    plt.savefig(os.path.join(PlotConfig.PLOTS_DIR, "summary_best_val_loss.png"), dpi=PlotConfig.DPI)
    plt.close()
    
    # Export data to CSV
    agg_stats.to_csv(os.path.join(PlotConfig.PLOTS_DIR, "summary_best_val_loss.csv"), index=False)

def plot_metric_evolution_during_precond(df):
    """
    Plots how the metrics change during the preconditioning phase.
    Creates one subplot per preconditioning level showing all metrics.
    """
    print("Generating Metric Evolution plots...")
    
    batch_data = df[df['log_type'] == 'batch'].copy()
    if batch_data.empty:
        print("  No batch-level data found.")
        return

    # Filter for preconditioning phase only
    precond_data = batch_data[batch_data['mode'] == 'precond']
    
    if precond_data.empty:
        print("  No preconditioning data found.")
        return
    
    # Remove control
    precond_data = precond_data[precond_data['metric_name'] != 'control']
    
    metrics = sorted(precond_data['metric_name'].unique())
    precond_levels = sorted(precond_data['precond_batches'].unique())
    
    if not metrics:
        print("  No metrics found in preconditioning phase.")
        return
    
    # Color palette for metrics
    metric_colors = sns.color_palette("husl", n_colors=len(metrics))
    metric_color_map = {metric: metric_colors[i] for i, metric in enumerate(metrics)}
    
    # Create grid
    n_cols = min(3, len(precond_levels))
    n_rows = int(np.ceil(len(precond_levels) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if len(precond_levels) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, pb in enumerate(precond_levels):
        ax = axes[idx]
        
        # Plot each metric for this precond level
        for metric in metrics:
            metric_data = precond_data[(precond_data['metric_name'] == metric) & 
                                       (precond_data['precond_batches'] == pb)]
            if metric_data.empty:
                continue
            
            # Aggregate by batch
            agg = metric_data.groupby('batch')['metric_val'].agg(['mean', 'std', 'count']).reset_index()
            agg['ci'] = 1.96 * (agg['std'] / np.sqrt(agg['count']))
            
            ax.plot(agg['batch'], agg['mean'], label=metric, 
                   color=metric_color_map[metric], linewidth=1.5, alpha=0.8)
            ax.fill_between(agg['batch'], agg['mean'] - agg['ci'], agg['mean'] + agg['ci'],
                           color=metric_color_map[metric], alpha=0.15)
        
        ax.set_title(f"Precond = {pb} batches", fontsize=11, fontweight='bold')
        ax.set_xlabel("Batch")
        ax.set_ylabel("Metric Value")
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(precond_levels), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Metric Evolution During Preconditioning Phase", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(PlotConfig.PLOTS_DIR, "metric_evolution_by_precond.png"), 
                dpi=PlotConfig.DPI, bbox_inches='tight')
    plt.close()
    
    # Export data to CSV
    csv_data = []
    for pb in precond_levels:
        for metric in metrics:
            metric_data = precond_data[(precond_data['metric_name'] == metric) & 
                                       (precond_data['precond_batches'] == pb)]
            if not metric_data.empty:
                agg = metric_data.groupby('batch')['metric_val'].agg(['mean', 'std', 'count']).reset_index()
                agg['ci'] = 1.96 * (agg['std'] / np.sqrt(agg['count']))
                
                for _, row in agg.iterrows():
                    csv_data.append({
                        'metric_name': metric,
                        'precond_batches': pb,
                        'batch': row['batch'],
                        'mean': row['mean'],
                        'ci': row['ci']
                    })
    
    pd.DataFrame(csv_data).to_csv(os.path.join(PlotConfig.PLOTS_DIR, "metric_evolution_by_precond.csv"), index=False)

# ============================================================================
# CRITICAL FIGURES FOR PUBLICATION
# ============================================================================

def plot_dose_response_curve(df):
    """
    Figure 1: Preconditioning dose-response curve
    Shows final validation loss vs preconditioning steps (log scale)
    """
    print("Generating Dose-Response Curve (CRITICAL FIGURE 1)...")
    
    val_data = df[df['log_type'] == 'validation']
    
    # Get final validation loss for each run
    final_losses = []
    for (metric, precond, run_id), group in val_data.groupby(['metric_name', 'precond_batches', 'run_id']):
        min_loss = group['val_loss'].min()
        final_losses.append({
            'metric_name': metric,
            'precond_batches': precond,
            'run_id': run_id,
            'final_val_loss': min_loss
        })
    
    res_df = pd.DataFrame(final_losses)
    
    # Aggregate
    agg = res_df.groupby(['metric_name', 'precond_batches'])['final_val_loss'].agg(['mean', 'std', 'count']).reset_index()
    agg['ci'] = 1.96 * (agg['std'] / np.sqrt(agg['count']))
    
    plt.figure(figsize=(10, 6))
    
    metrics = sorted(res_df['metric_name'].unique())
    metric_colors = sns.color_palette("husl", n_colors=len(metrics))
    
    for i, metric in enumerate(metrics):
        metric_data = agg[agg['metric_name'] == metric]
        
        if metric == 'control':
            plt.axhline(y=metric_data['mean'].iloc[0], 
                       color=PlotConfig.CONTROL_COLOR, 
                       linestyle=PlotConfig.CONTROL_STYLE, 
                       linewidth=2.5, label='Control', zorder=10)
            plt.fill_between([0.5, agg['precond_batches'].max()*2],
                           metric_data['mean'].iloc[0] - metric_data['ci'].iloc[0],
                           metric_data['mean'].iloc[0] + metric_data['ci'].iloc[0],
                           color=PlotConfig.CONTROL_COLOR, alpha=0.1)
        else:
            plt.errorbar(metric_data['precond_batches'], metric_data['mean'], 
                        yerr=metric_data['ci'], label=metric, 
                        color=metric_colors[i], marker='o', markersize=6,
                        linewidth=2, capsize=4, capthick=1.5)
    
    plt.xscale('log')
    plt.xlabel("Preconditioning Steps (batches)", fontsize=12, fontweight='bold')
    plt.ylabel("Final Validation Loss", fontsize=12, fontweight='bold')
    plt.title("Preconditioning Dose–Response Curve", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(PlotConfig.PLOTS_DIR, "fig1_dose_response.png"), dpi=PlotConfig.DPI)
    plt.close()
    
    # Export data to CSV
    agg.to_csv(os.path.join(PlotConfig.PLOTS_DIR, "fig1_dose_response.csv"), index=False)

def plot_compute_efficiency_frontier(df):
    """
    Figure 2: Compute-efficiency frontier
    Validation loss vs training compute (6ND), control vs best metrics
    """
    print("Generating Compute-Efficiency Frontier (CRITICAL FIGURE 2)...")
    
    val_data = df[df['log_type'] == 'validation'].copy()
    
    # Select control and best 2-3 metrics based on final performance
    final_perf = val_data.groupby(['metric_name', 'precond_batches'])['val_loss'].min().reset_index()
    best_configs = final_perf.nsmallest(4, 'val_loss')  # Control + top 3
    
    plt.figure(figsize=(10, 7))
    
    # Plot control
    control_data = val_data[val_data['metric_name'] == 'control']
    control_agg = get_aggregated_data(control_data, ['cumulative_compute'], 'val_loss')
    
    plt.plot(control_agg['cumulative_compute'], control_agg['mean'],
            label='Control', color=PlotConfig.CONTROL_COLOR,
            linestyle=PlotConfig.CONTROL_STYLE, linewidth=3, zorder=10, marker='.')
    plt.fill_between(control_agg['cumulative_compute'], control_agg['lower'], control_agg['upper'],
                    color=PlotConfig.CONTROL_COLOR, alpha=0.1)
    
    # Plot best metrics
    # Count non-control configs to set proper color palette size
    non_control_configs = best_configs[best_configs['metric_name'] != 'control']
    colors = sns.color_palette("bright", n_colors=max(1, len(non_control_configs)))
    color_idx = 0
    
    for _, row in best_configs.iterrows():
        if row['metric_name'] == 'control':
            continue
        
        metric_data = val_data[(val_data['metric_name'] == row['metric_name']) & 
                               (val_data['precond_batches'] == row['precond_batches'])]
        
        metric_agg = get_aggregated_data(metric_data, ['cumulative_compute'], 'val_loss')
        
        label = f"{row['metric_name']} (P={int(row['precond_batches'])})"
        plt.plot(metric_agg['cumulative_compute'], metric_agg['mean'],
                label=label, color=colors[color_idx], linewidth=2, marker='o', markersize=4)
        plt.fill_between(metric_agg['cumulative_compute'], metric_agg['lower'], metric_agg['upper'],
                        color=colors[color_idx], alpha=0.15)
        color_idx += 1
    
    plt.xscale('log')
    plt.xlabel("Training Compute (FLOPs, 6ND formula)\n*Metric computation excluded*", 
              fontsize=12, fontweight='bold')
    plt.ylabel("Validation Loss", fontsize=12, fontweight='bold')
    plt.title("Compute-Efficiency Frontier", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(PlotConfig.PLOTS_DIR, "fig2_compute_frontier.png"), dpi=PlotConfig.DPI)
    plt.close()
    
    # Export data to CSV
    csv_data = []
    
    # Control data
    for _, row in control_agg.iterrows():
        csv_data.append({
            'metric_name': 'control',
            'precond_batches': 0,
            'cumulative_compute': row['cumulative_compute'],
            'mean': row['mean'],
            'ci': row['ci']
        })
    
    # Best metrics data
    for _, config in best_configs.iterrows():
        if config['metric_name'] == 'control':
            continue
        
        metric_data = val_data[(val_data['metric_name'] == config['metric_name']) & 
                               (val_data['precond_batches'] == config['precond_batches'])]
        metric_agg = get_aggregated_data(metric_data, ['cumulative_compute'], 'val_loss')
        
        for _, row in metric_agg.iterrows():
            csv_data.append({
                'metric_name': config['metric_name'],
                'precond_batches': config['precond_batches'],
                'cumulative_compute': row['cumulative_compute'],
                'mean': row['mean'],
                'ci': row['ci']
            })
    
    pd.DataFrame(csv_data).to_csv(os.path.join(PlotConfig.PLOTS_DIR, "fig2_compute_frontier.csv"), index=False)

def plot_early_training_acceleration(df):
    """
    Figure 3: Early-training acceleration
    Shows validation loss vs CE training steps with zoomed inset
    """
    print("Generating Early-Training Acceleration (CRITICAL FIGURE 3)...")
    
    val_data = df[df['log_type'] == 'validation'].copy()
    
    # Select control and top performer
    final_perf = val_data.groupby(['metric_name', 'precond_batches'])['val_loss'].min().reset_index()
    best_config = final_perf[final_perf['metric_name'] != 'control'].nsmallest(1, 'val_loss').iloc[0]
    
    fig, (ax_main, ax_inset) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Main plot
    control_data = val_data[val_data['metric_name'] == 'control']
    control_agg = get_aggregated_data(control_data, ['batches_processed'], 'val_loss')
    
    ax_main.plot(control_agg['batches_processed'], control_agg['mean'],
                label='Control', color=PlotConfig.CONTROL_COLOR,
                linestyle=PlotConfig.CONTROL_STYLE, linewidth=2.5)
    ax_main.fill_between(control_agg['batches_processed'], control_agg['lower'], control_agg['upper'],
                        color=PlotConfig.CONTROL_COLOR, alpha=0.1)
    
    best_data = val_data[(val_data['metric_name'] == best_config['metric_name']) & 
                         (val_data['precond_batches'] == best_config['precond_batches'])]
    best_agg = get_aggregated_data(best_data, ['batches_processed'], 'val_loss')
    
    label = f"{best_config['metric_name']} (P={int(best_config['precond_batches'])})"
    ax_main.plot(best_agg['batches_processed'], best_agg['mean'],
                label=label, color='red', linewidth=2.5)
    ax_main.fill_between(best_agg['batches_processed'], best_agg['lower'], best_agg['upper'],
                        color='red', alpha=0.15)
    
    ax_main.set_xlabel("CE Training Steps (batches)", fontsize=12, fontweight='bold')
    ax_main.set_ylabel("Validation Loss", fontsize=12, fontweight='bold')
    ax_main.set_title("Full Training Trajectory", fontsize=12)
    ax_main.legend(fontsize=10)
    ax_main.grid(True, alpha=0.3)
    
    # Inset: first 500 steps
    max_steps = 500
    control_early = control_agg[control_agg['batches_processed'] <= max_steps]
    best_early = best_agg[best_agg['batches_processed'] <= max_steps]
    
    ax_inset.plot(control_early['batches_processed'], control_early['mean'],
                 color=PlotConfig.CONTROL_COLOR, linestyle=PlotConfig.CONTROL_STYLE, linewidth=2.5)
    ax_inset.fill_between(control_early['batches_processed'], control_early['lower'], control_early['upper'],
                         color=PlotConfig.CONTROL_COLOR, alpha=0.1)
    
    ax_inset.plot(best_early['batches_processed'], best_early['mean'],
                 color='red', linewidth=2.5)
    ax_inset.fill_between(best_early['batches_processed'], best_early['lower'], best_early['upper'],
                         color='red', alpha=0.15)
    
    ax_inset.set_xlabel(f"First {max_steps} CE Steps", fontsize=12, fontweight='bold')
    ax_inset.set_ylabel("Validation Loss", fontsize=12, fontweight='bold')
    ax_inset.set_title("Early Acceleration (Zoomed)", fontsize=12)
    ax_inset.grid(True, alpha=0.3)
    
    plt.suptitle("Early-Training Acceleration Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PlotConfig.PLOTS_DIR, "fig3_early_acceleration.png"), dpi=PlotConfig.DPI, bbox_inches='tight')
    plt.close()
    
    # Export data to CSV
    csv_data = []
    
    for _, row in control_agg.iterrows():
        csv_data.append({
            'metric_name': 'control',
            'precond_batches': 0,
            'batches_processed': row['batches_processed'],
            'mean': row['mean'],
            'ci': row['ci']
        })
    
    for _, row in best_agg.iterrows():
        csv_data.append({
            'metric_name': best_config['metric_name'],
            'precond_batches': best_config['precond_batches'],
            'batches_processed': row['batches_processed'],
            'mean': row['mean'],
            'ci': row['ci']
        })
    
    pd.DataFrame(csv_data).to_csv(os.path.join(PlotConfig.PLOTS_DIR, "fig3_early_acceleration.csv"), index=False)

def plot_fixed_compute_comparison(df):
    """
    Figure 4: Fixed-compute comparison
    Fix total CE steps, compare final loss across preconditioning levels
    """
    print("Generating Fixed-Compute Comparison (RECOMMENDED FIGURE 4)...")
    
    val_data = df[df['log_type'] == 'validation']
    
    # Determine a reasonable fixed compute budget (e.g., max batches processed by control)
    max_batches = val_data[val_data['metric_name'] == 'control']['batches_processed'].max()
    
    # For each config, get loss at this fixed batch count (or closest)
    results = []
    for (metric, precond, run_id), group in val_data.groupby(['metric_name', 'precond_batches', 'run_id']):
        # Find closest epoch to target
        closest = group.iloc[(group['batches_processed'] - max_batches).abs().argsort()[:1]]
        if len(closest) > 0:
            results.append({
                'metric_name': metric,
                'precond_batches': precond,
                'run_id': run_id,
                'val_loss': closest.iloc[0]['val_loss']
            })
    
    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=res_df, x='metric_name', y='val_loss', hue='precond_batches',
                    palette='coolwarm', errorbar=('ci', 95), capsize=.1)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=7, padding=3)
    
    plt.title(f"Fixed-Compute Comparison (at ~{max_batches:.0f} CE batches)", 
             fontsize=14, fontweight='bold')
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Precond Batches", fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PlotConfig.PLOTS_DIR, "fig4_fixed_compute.png"), dpi=PlotConfig.DPI)
    plt.close()
    
    # Export data to CSV
    agg_stats = res_df.groupby(['metric_name', 'precond_batches'])['val_loss'].agg(['mean', 'std', 'count']).reset_index()
    agg_stats['ci'] = 1.96 * (agg_stats['std'] / np.sqrt(agg_stats['count']))
    agg_stats['fixed_batches'] = max_batches
    agg_stats.to_csv(os.path.join(PlotConfig.PLOTS_DIR, "fig4_fixed_compute.csv"), index=False)

def plot_metric_ranking_integrated(df):
    """
    Integrated Multi-Criteria Ranking of Metrics
    Combines: 1) Final Performance, 2) Compute Efficiency, 3) Early Acceleration, 4) Stability
    """
    print("Generating Integrated Metric Ranking (COMPREHENSIVE ANALYSIS)...")
    
    val_data = df[df['log_type'] == 'validation'].copy()
    
    # Exclude control from ranking
    val_data_metrics = val_data[val_data['metric_name'] != 'control']
    
    if val_data_metrics.empty:
        print("No metrics found for ranking. Skipping.")
        return
    
    ranking_data = []
    
    for (metric, precond), group in val_data_metrics.groupby(['metric_name', 'precond_batches']):
        # Criterion 1: Final Performance (lower is better)
        final_loss = group.groupby('run_id')['val_loss'].min().mean()
        final_loss_std = group.groupby('run_id')['val_loss'].min().std()
        
        # Criterion 2: Compute Efficiency (inverse of area under curve - higher is better)
        # Calculate AUC of validation loss over training compute
        # Lower AUC = reaches good performance faster = MORE efficient
        auc_values = []
        for run_id, run_group in group.groupby('run_id'):
            sorted_group = run_group.sort_values('cumulative_compute')
            # Approximate integral using trapezoidal rule
            if len(sorted_group) > 1:
                auc = np.trapz(sorted_group['val_loss'], sorted_group['cumulative_compute'])
                auc_values.append(auc)
        
        # Store the average AUC (will invert during normalization)
        auc_efficiency = np.mean(auc_values) if auc_values else np.nan
        
        # Criterion 3: Early Training Acceleration (performance at 25% of training)
        early_checkpoint = group['batches_processed'].max() * 0.25
        early_performance = []
        for run_id, run_group in group.groupby('run_id'):
            closest = run_group.iloc[(run_group['batches_processed'] - early_checkpoint).abs().argsort()[:1]]
            if len(closest) > 0:
                early_performance.append(closest.iloc[0]['val_loss'])
        early_loss = np.mean(early_performance) if early_performance else np.nan
        
        # Criterion 4: Stability (lower variance across runs is better)
        stability = final_loss_std if final_loss_std is not None else 0
        
        ranking_data.append({
            'metric_name': metric,
            'precond_batches': int(precond),
            'final_loss': final_loss,
            'auc_efficiency': auc_efficiency,  # Lower AUC is better
            'early_loss': early_loss,
            'stability': stability
        })
    
    rank_df = pd.DataFrame(ranking_data)
    
    # Normalize each criterion to [0, 1] for fair comparison
    # For metrics where lower is better, invert the scale so higher normalized score = better
    rank_df['final_loss_norm'] = 1 - (rank_df['final_loss'] - rank_df['final_loss'].min()) / (rank_df['final_loss'].max() - rank_df['final_loss'].min() + 1e-10)
    
    # For AUC: lower is better (smaller area = faster learning), so invert
    rank_df['compute_eff_norm'] = 1 - (rank_df['auc_efficiency'] - rank_df['auc_efficiency'].min()) / (rank_df['auc_efficiency'].max() - rank_df['auc_efficiency'].min() + 1e-10)
    
    rank_df['early_loss_norm'] = 1 - (rank_df['early_loss'] - rank_df['early_loss'].min()) / (rank_df['early_loss'].max() - rank_df['early_loss'].min() + 1e-10)
    rank_df['stability_norm'] = 1 - (rank_df['stability'] - rank_df['stability'].min()) / (rank_df['stability'].max() - rank_df['stability'].min() + 1e-10)
    
    # Calculate composite score (weighted average)
    # Weights: Final Performance (40%), Compute Efficiency (30%), Early Acceleration (20%), Stability (10%)
    rank_df['composite_score'] = (
        0.40 * rank_df['final_loss_norm'] +
        0.30 * rank_df['compute_eff_norm'] +
        0.20 * rank_df['early_loss_norm'] +
        0.10 * rank_df['stability_norm']
    )
    
    # Sort by composite score (descending - higher is better)
    rank_df = rank_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    rank_df['rank'] = range(1, len(rank_df) + 1)
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Composite Score Ranking (Horizontal Bar Chart)
    ax1 = axes[0]
    rank_df_top = rank_df.head(10)  # Show top 10
    
    colors_rank = sns.color_palette("RdYlGn_r", n_colors=len(rank_df_top))
    
    y_pos = np.arange(len(rank_df_top))
    labels = [f"{row['metric_name']} (P={row['precond_batches']})" for _, row in rank_df_top.iterrows()]
    
    bars = ax1.barh(y_pos, rank_df_top['composite_score'], color=colors_rank)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()  # Highest score at top
    ax1.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
    ax1.set_title('Top 10 Metrics by Integrated Ranking', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, rank_df_top['composite_score'])):
        ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Criterion Breakdown for Top 5 (Grouped Bar Chart)
    ax2 = axes[1]
    rank_df_top5 = rank_df.head(5)
    
    criteria = ['final_loss_norm', 'compute_eff_norm', 'early_loss_norm', 'stability_norm']
    criteria_labels = ['Final\nPerf.', 'Compute\nEfficiency', 'Early\nAccel.', 'Stability']
    
    x = np.arange(len(criteria_labels))
    width = 0.15
    
    colors_breakdown = sns.color_palette("Set2", n_colors=len(rank_df_top5))
    
    for i, (_, row) in enumerate(rank_df_top5.iterrows()):
        values = [row[c] for c in criteria]
        label = f"{row['metric_name']} (P={row['precond_batches']})"
        ax2.bar(x + i * width, values, width, label=label, color=colors_breakdown[i])
    
    ax2.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Evaluation Criteria', fontsize=12, fontweight='bold')
    ax2.set_title('Criterion Breakdown - Top 5 Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(criteria_labels, fontsize=10)
    ax2.legend(loc='upper right', fontsize=8, ncol=1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(PlotConfig.PLOTS_DIR, "fig5_metric_ranking_integrated.png"), dpi=PlotConfig.DPI)
    plt.close()
    auc
    # Export ranking data to CSV
    export_df = rank_df[['rank', 'metric_name', 'precond_batches', 'composite_score', 
                         'final_loss', 'auc_efficiency', 'early_loss', 'stability']]
    export_df.to_csv(os.path.join(PlotConfig.PLOTS_DIR, "fig5_metric_ranking.csv"), index=False)
    
    # Print top 5 to console
    print("\n" + "="*60)
    print("TOP 5 METRICS (Integrated Ranking)")
    print("="*60)
    for _, row in rank_df.head(5).iterrows():
        print(f"  #{int(row['rank'])}: {row['metric_name']} (P={int(row['precond_batches'])}) "
              f"- Score: {row['composite_score']:.4f}")
        print(f"       Final Loss: {row['final_loss']:.4f} | "
              f"Early Loss: {row['early_loss']:.4f} | "
              f"Stability: {row['stability']:.4f}")
    print("="*60)

def main():
    configure_plotting()
    
    print("Loading data...")
    df = load_data()
    
    if df.empty:
        print("No data found. Exiting.")
        return
        
    print(f"Loaded {len(df)} records.")
    
    print("\n" + "="*60)
    print("CRITICAL FIGURES (Main Paper)")
    print("="*60)
    
    # Figure 1: Dose-response curve
    plot_dose_response_curve(df)
    
    # Figure 2: Compute-efficiency frontier
    plot_compute_efficiency_frontier(df)
    
    # Figure 3: Early-training acceleration
    plot_early_training_acceleration(df)
    
    print("\n" + "="*60)
    print("RECOMMENDED FIGURES (Appendix)")
    print("="*60)
    
    # Figure 4: Fixed-compute comparison
    plot_fixed_compute_comparison(df)
    
    # Figure 5: Integrated Metric Ranking
    plot_metric_ranking_integrated(df)
    
    print("\n" + "="*60)
    print("SUPPLEMENTARY FIGURES")
    print("="*60)
    
    # Existing plots
    plot_learning_curves(df)
    plot_efficiency_gap(df)
    plot_best_val_losses(df)
    plot_metric_evolution_during_precond(df)
    
    print(f"\nDone. All plots saved to '{PlotConfig.PLOTS_DIR}'")

if __name__ == "__main__":
    main()
