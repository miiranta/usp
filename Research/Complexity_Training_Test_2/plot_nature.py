import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

# Set style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory for plots
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots_nature")
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data(root_dir=None):
    """
    Traverses the output directory structure:
    output/
      {metric_name}_{direction}/
        H{hidden}_L{layers}/
          results_run_{run}.csv
    """
    if root_dir is None:
        root_dir = os.path.join(SCRIPT_DIR, "output_nature")
    data_list = []
    
    # Only look for first-level directories in output/
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} not found.")
        return pd.DataFrame()

    metric_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for m_dir in metric_dirs:
        # Check if it's a valid metric folder (format: name_min or name_max)
        if not ('_min' in m_dir or '_max' in m_dir):
            continue
            
        is_control = 'shannon' in m_dir and 'control' in m_dir # Adjusted based on folder naming
        # Or check if folder name exactly matches control logic used in train_nature
        # In train_nature: output_dir = f"output/{clean_name}_{direction}"
        
        # Parse metric name
        if '_min' in m_dir:
            metric_name = m_dir.replace('_min', '')
            direction = 'min'
        else:
            metric_name = m_dir.replace('_max', '')
            direction = 'max'
            
        full_m_dir = os.path.join(root_dir, m_dir)
        
        # Look for size directories (H{}_L{})
        size_dirs = [d for d in os.listdir(full_m_dir) if os.path.isdir(os.path.join(full_m_dir, d)) and d.startswith('H')]
        
        for s_dir in size_dirs:
            # Parse size
            match = re.match(r'H(\d+)_L(\d+)', s_dir)
            if not match:
                continue
            hidden_dim = int(match.group(1))
            num_layers = int(match.group(2))
            
            # Approximate parameter count (roughly proportional to H^2 * L)
            # Embedding: V * H
            # Layers: L * (12 * H^2) roughly
            # Head: H * V
            # Simplified proxy for sorting:
            param_proxy = num_layers * (hidden_dim ** 2)
            
            full_s_dir = os.path.join(full_m_dir, s_dir)
            
            # Look for CSV files
            csv_files = glob.glob(os.path.join(full_s_dir, "results_run_*.csv"))
            
            for csv_file in csv_files:
                try:
                    # Read the summary section (lines 1-8 approx)
                    # We need to parse the file carefully as it has multiple sections
                    # Structure:
                    # Metric, name
                    # ... Test Results ...
                    # ...
                    # ... Training Summary ...
                    # Final Training Loss, ...
                    # Final Validation Loss, ...
                    # Final Metric Value, ...
                    # Total FLOPs, ...  <-- Ensure this exists
                    # Run Number, ...
                    # []
                    # Header for timeseries
                    
                    # Let's read the whole file to pandas first, dealing with the header mess
                    # Actually, easier to read lines for summary, then pandas for history
                    
                    with open(csv_file, 'r') as f:
                        lines = f.readlines()
                    
                    total_flops = None
                    final_val_loss = None
                    final_test_loss_wiki = None
                    final_test_loss_shake = None
                    
                    # Extract summary values
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) < 2: continue
                        
                        if 'Total FLOPs' in parts[0]:
                            total_flops = float(parts[1])
                        elif 'Final Validation Loss' in parts[0]:
                            final_val_loss = float(parts[1])
                        elif 'Test Loss (WikiText-2)' in parts[0]:
                            final_test_loss_wiki = float(parts[1])
                        elif 'Test Loss (Tiny-Shakespeare)' in parts[0]:
                            final_test_loss_shake = float(parts[1])
                            
                    # Read history
                    # Find where the data starts
                    start_idx = 0
                    for i, line in enumerate(lines):
                        if 'Step FLOPs' in line and 'Epoch' in line:
                            start_idx = i
                            break
                    
                    if start_idx == 0:
                        # Fallback for old CSV format if necessary, but train_nature was updated
                        continue
                        
                    # Create temporary buffer for pandas
                    from io import StringIO
                    history_str = "".join(lines[start_idx:])
                    df_hist = pd.read_csv(StringIO(history_str))
                    
                    # Cumulative FLOPs
                    if 'Step FLOPs' in df_hist.columns:
                        df_hist['Cumulative FLOPs'] = df_hist['Step FLOPs'].cumsum()
                    
                    # Calculate Total Model FLOPs from history summing 'Model FLOPs'
                    total_model_flops = df_hist['Model FLOPs'].sum() if 'Model FLOPs' in df_hist.columns else total_flops

                    # Calculate BEST losses from history
                    best_val_loss = df_hist['Validation Loss'].min() if 'Validation Loss' in df_hist.columns else final_val_loss
                    best_test_loss_wiki = df_hist['Test Loss Wiki'].min() if 'Test Loss Wiki' in df_hist.columns else final_test_loss_wiki
                    best_test_loss_shake = df_hist['Test Loss Shakespeare'].min() if 'Test Loss Shakespeare' in df_hist.columns else final_test_loss_shake

                    # Add to data list
                    run_data = {
                        'metric': metric_name,
                        'hidden': hidden_dim,
                        'layers': num_layers,
                        'params': param_proxy,
                        'run': int(os.path.basename(csv_file).split('_')[-1].split('.')[0]),
                        'total_flops': total_flops,
                        'total_model_flops': total_model_flops,
                        'final_val_loss': final_val_loss,
                        'final_test_loss_wiki': final_test_loss_wiki,
                        'final_test_loss_shake': final_test_loss_shake,
                        'best_val_loss': best_val_loss,
                        'best_test_loss_wiki': best_test_loss_wiki,
                        'best_test_loss_shake': best_test_loss_shake,
                        'is_control': 'control' in metric_name or 'shannon' in metric_name,
                        'history': df_hist
                    }
                    data_list.append(run_data)
                    
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")

    return pd.DataFrame(data_list)

def plot_scaling_laws(df):
    """
    1. The scaling-law plot (System Efficiency)
    X-axis: log10(Total FLOPs)
    Y-axis: log10(Best Validation loss)
    """
    print("Plotting Scaling Laws (System Efficiency)...")
    plt.figure(figsize=(10, 8))
    
    # Filter valid data
    df = df.dropna(subset=['total_flops', 'best_val_loss']) 
    
    # Use Best Validation Loss as requested
    df['log_flops'] = np.log10(df['total_flops'])
    df['log_val_loss'] = np.log10(df['best_val_loss'])
    
    # Separate Control and Others
    control_df = df[df['is_control']]
    metrics_df = df[~df['is_control']]
    
    # Get unique metrics
    unique_metrics = metrics_df['metric'].unique()
    
    colors = sns.color_palette("muted", len(unique_metrics) + 1)
    
    # Plot Control
    if not control_df.empty:
        sns.scatterplot(
            data=control_df, x='log_flops', y='log_val_loss', 
            label='Baseline', color='black', s=100, marker='o'
        )
        # Fit Line
        if control_df['log_flops'].nunique() > 1:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(control_df['log_flops'], control_df['log_val_loss'])
                x_vals = np.array([control_df['log_flops'].min(), control_df['log_flops'].max()])
                plt.plot(x_vals, intercept + slope * x_vals, 'k--', linewidth=2, label=f'Baseline Fit (α={-slope:.3f})')
            except ValueError:
                pass
    
    # Plot Metrics
    for i, metric in enumerate(unique_metrics):
        m_df = metrics_df[metrics_df['metric'] == metric]
        color = colors[i]
        
        sns.scatterplot(
            data=m_df, x='log_flops', y='log_val_loss', 
            label=f'{metric}', color=color, s=80, marker='s', alpha=0.7
        )
        
        # Fit Line
        if m_df['log_flops'].nunique() > 1:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(m_df['log_flops'], m_df['log_val_loss'])
                x_vals = np.array([m_df['log_flops'].min(), m_df['log_flops'].max()])
                plt.plot(x_vals, intercept + slope * x_vals, '-', color=color, linewidth=2, label=f'{metric} Fit (α={-slope:.3f})')
            except ValueError:
                pass

    plt.xlabel(r'$\log_{10}(\text{Total FLOPs})$')
    plt.ylabel(r'$\log_{10}(\text{Best Validation Loss})$')
    plt.title('System Efficiency: Validation Loss vs Total Compute')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "1_scaling_law_system_efficiency.png"), dpi=300)
    plt.close()

def plot_loss_vs_flops_trajectory(df):
    """
    2. Loss vs FLOPs (trajectory)
    X-axis: cumulative FLOPs during training
    Y-axis: validation loss
    """
    print("Plotting Loss vs FLOPs Trajectory...")
    plt.figure(figsize=(12, 8))
    
    # Pick the largest model size for this visualization to show the effect clearly
    max_params = df['params'].max()
    large_models = df[df['params'] == max_params]
    
    if large_models.empty:
        return

    # Plot trajectories
    unique_metrics = large_models['metric'].unique()
    
    for metric in unique_metrics:
        # Average over runs
        runs = large_models[large_models['metric'] == metric]
        
        # We need to align runs. A simple way is to concat all histories and plot trace
        all_hist = []
        for _, row in runs.iterrows():
            hist = row['history'].copy()
            hist['metric'] = metric
            all_hist.append(hist)
            
        if not all_hist: continue
        
        combined_hist = pd.concat(all_hist)
        
        # Convert FLOPs to log scale for x-axis often helps, but request says "cumulative FLOPs"
        # Often log-log is used here too, but let's stick to linear-log or log-log
        
        sns.lineplot(
            data=combined_hist, x='Cumulative FLOPs', y='Validation Loss',
            label=metric, linewidth=2.5, alpha=0.8
        )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cumulative FLOPs (log scale)')
    plt.ylabel('Validation Loss (log scale)')
    plt.title(f'Training Trajectory: Loss vs Compute (Model Size: Largest)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "3_loss_vs_flops_trajectory.png"), dpi=300)
    plt.close()

def plot_optimizer_cost_fraction(df):
    """
    3. Optimizer cost fraction
    Stacked area plot of FLOP components over training for one representative run.
    """
    print("Plotting Optimizer Cost Fraction...")
    
    # Pick a non-control run, preferably specific metric
    # Let's pick the largest model, metric run 1
    target_models = df[(~df['is_control']) & (df['params'] == df['params'].max())]
    if target_models.empty:
        return
        
    run_row = target_models.iloc[0]
    hist = run_row['history']
    metric_name = run_row['metric']
    
    if 'Model FLOPs' not in hist.columns:
        print("FLOP breakdown columns not found in history.")
        return
        
    plt.figure(figsize=(10, 6))
    
    epochs = hist['Epoch']
    model_flops = hist['Model FLOPs']
    metric_flops = hist['Metric FLOPs']
    opt_flops = hist['Opt FLOPs']
    
    # Stackplot
    plt.stackplot(epochs, model_flops, opt_flops, metric_flops, 
                  labels=['Model Forward/Backward', 'Optimizer (AdamW)', 'Metric Calculation'],
                  colors=['#3498db', '#95a5a6', '#e74c3c'], alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('FLOPs per Epoch')
    plt.title(f'Compute Breakdown: {metric_name} (Largest Model)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "4_compute_breakdown_stacked.png"), dpi=300)
    plt.close()
    
    # Normalized version (Percentage)
    plt.figure(figsize=(10, 6))
    total = model_flops + opt_flops + metric_flops
    plt.stackplot(epochs, 
                  model_flops/total * 100, 
                  opt_flops/total * 100, 
                  metric_flops/total * 100,
                  labels=['Model', 'Optimizer', 'Metric'],
                  colors=['#3498db', '#95a5a6', '#e74c3c'], alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Percentage of Compute (%)')
    plt.title(f'Compute Cost Distribution: {metric_name}')
    plt.legend(loc='lower right')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "4_compute_breakdown_percent.png"), dpi=300)
    plt.close()

def plot_universality(df):
    """
    4. Universality plot (Wiki vs Shakespeare)
    Scaling laws on different datasets.
    """
    print("Plotting Universality...")
    
    datasets = [
        ('best_val_loss', 'WikiText-2 (Best Val)'),
        ('best_test_loss_shake', 'Tiny Shakespeare (Best Test)')
    ]
    
    for col, name in datasets:
        plt.figure(figsize=(10, 8))
        
        sub_df = df.dropna(subset=['total_flops', col])
        sub_df['log_flops'] = np.log10(sub_df['total_flops'])
        sub_df['log_loss'] = np.log10(sub_df[col])
        
        control_df = sub_df[sub_df['is_control']]
        metrics_df = sub_df[~sub_df['is_control']]
        unique_metrics = metrics_df['metric'].unique()
        colors = sns.color_palette("muted", len(unique_metrics) + 1)
        
        # Plot Control
        if not control_df.empty:
            sns.scatterplot(
                data=control_df, x='log_flops', y='log_loss', 
                label='Baseline', color='black', s=100, marker='o'
            )
            # Fit
            if control_df['log_flops'].nunique() > 1:
                try:
                    slope, intercept, _, _, _ = stats.linregress(control_df['log_flops'], control_df['log_loss'])
                    x_vals = np.array([control_df['log_flops'].min(), control_df['log_flops'].max()])
                    plt.plot(x_vals, intercept + slope * x_vals, 'k--', label=f'Baseline α={-slope:.3f}')
                except ValueError:
                    pass

        # Plot Metrics
        for i, metric in enumerate(unique_metrics):
            m_df = metrics_df[metrics_df['metric'] == metric]
            color = colors[i]
            sns.scatterplot(
                data=m_df, x='log_flops', y='log_loss', 
                label=metric, color=color, s=80, marker='s', alpha=0.7
            )
            # Fit
            if m_df['log_flops'].nunique() > 1:
                try:
                    slope, intercept, _, _, _ = stats.linregress(m_df['log_flops'], m_df['log_loss'])
                    x_vals = np.array([m_df['log_flops'].min(), m_df['log_flops'].max()])
                    plt.plot(x_vals, intercept + slope * x_vals, '-', color=color, label=f'{metric} α={-slope:.3f}')
                except ValueError:
                    pass
        
        plt.xlabel(r'$\log_{10}(\text{Total FLOPs})$')
        plt.ylabel(r'$\log_{10}(\text{Best Loss})$')
        plt.title(f'Universality: Scaling on {name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        
        # Clean name: remove parentheses and replacing spaces
        clean_name = name.split(' (')[0].replace(' ', '_')
        plt.savefig(os.path.join(PLOT_DIR, f"5_universality_{clean_name}.png"), dpi=300)
        plt.close()

def plot_optimizer_efficiency(df):
    """
    2. Optimizer Efficiency
    X-axis: log10(Total Model FLOPs) -> Effective training work
    Y-axis: log10(Best Validation loss)
    """
    print("Plotting Optimizer Efficiency...")
    plt.figure(figsize=(10, 8))
    
    # Filter valid data
    df = df.dropna(subset=['total_model_flops', 'best_val_loss']) 
    
    df['log_model_flops'] = np.log10(df['total_model_flops'])
    df['log_val_loss'] = np.log10(df['best_val_loss'])
    
    # Separate Control and Others
    control_df = df[df['is_control']]
    metrics_df = df[~df['is_control']]
    
    # Get unique metrics
    unique_metrics = metrics_df['metric'].unique()
    colors = sns.color_palette("muted", len(unique_metrics) + 1)
    
    # Plot Control
    if not control_df.empty:
        sns.scatterplot(
            data=control_df, x='log_model_flops', y='log_val_loss', 
            label='Baseline', color='black', s=100, marker='o'
        )
        # Fit Line
        if control_df['log_model_flops'].nunique() > 1:
            try:
                slope, intercept, _, _, _ = stats.linregress(control_df['log_model_flops'], control_df['log_val_loss'])
                x_vals = np.array([control_df['log_model_flops'].min(), control_df['log_model_flops'].max()])
                plt.plot(x_vals, intercept + slope * x_vals, 'k--', linewidth=2, label=f'Baseline Fit (α={-slope:.3f})')
            except ValueError:
                pass
    
    # Plot Metrics
    for i, metric in enumerate(unique_metrics):
        m_df = metrics_df[metrics_df['metric'] == metric]
        color = colors[i]
        
        sns.scatterplot(
            data=m_df, x='log_model_flops', y='log_val_loss', 
            label=f'{metric}', color=color, s=80, marker='s', alpha=0.7
        )
        
        # Fit Line
        if m_df['log_model_flops'].nunique() > 1:
            try:
                slope, intercept, _, _, _ = stats.linregress(m_df['log_model_flops'], m_df['log_val_loss'])
                x_vals = np.array([m_df['log_model_flops'].min(), m_df['log_model_flops'].max()])
                plt.plot(x_vals, intercept + slope * x_vals, '-', color=color, linewidth=2, label=f'{metric} Fit (α={-slope:.3f})')
            except ValueError:
                pass

    plt.xlabel(r'$\log_{10}(\text{Model FLOPs})$')
    plt.ylabel(r'$\log_{10}(\text{Best Validation Loss})$')
    plt.title('Optimizer Efficiency: Loss vs Effective Training Compute')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "2_scaling_law_optimizer_efficiency.png"), dpi=300)
    plt.close()

def plot_learning_curves(df):
    """
    6. Learning Curves
    X-axis: Epoch
    Y-axis: Training and Validation Loss
    """
    print("Plotting Learning Curves...")
    
    # Pick the largest model size
    max_params = df['params'].max()
    large_models = df[df['params'] == max_params]
    
    if large_models.empty:
        return

    # Prepare data for plotting
    control_models = large_models[large_models['is_control']]
    metric_models = large_models[~large_models['is_control']]
    unique_metrics = metric_models['metric'].unique()
    colors = sns.color_palette("muted", len(unique_metrics))

    # --- Training Loss ---
    plt.figure(figsize=(10, 8))
    
    # Plot Control
    if not control_models.empty:
        all_hist = []
        for _, row in control_models.iterrows():
            hist = row['history'].copy()
            all_hist.append(hist)
        
        if all_hist:
            combined_hist = pd.concat(all_hist)
            sns.lineplot(
                data=combined_hist, x='Epoch', y='Training Loss',
                label='Baseline', color='black', linewidth=2.5, alpha=0.8
            )

    # Plot Metrics
    for i, metric in enumerate(unique_metrics):
        runs = metric_models[metric_models['metric'] == metric]
        color = colors[i]
        
        all_hist = []
        for _, row in runs.iterrows():
            hist = row['history'].copy()
            all_hist.append(hist)
            
        if all_hist:
            combined_hist = pd.concat(all_hist)
            sns.lineplot(
                data=combined_hist, x='Epoch', y='Training Loss',
                label=metric, color=color, linewidth=2.5, alpha=0.8
            )

    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Learning Curves: Training Loss (Model Size: Largest)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "6_learning_curves_train.png"), dpi=300)
    plt.close()

    # --- Validation Loss ---
    plt.figure(figsize=(10, 8))
    
    # Plot Control
    if not control_models.empty:
        all_hist = []
        for _, row in control_models.iterrows():
            hist = row['history'].copy()
            all_hist.append(hist)
        
        if all_hist:
            combined_hist = pd.concat(all_hist)
            sns.lineplot(
                data=combined_hist, x='Epoch', y='Validation Loss',
                label='Baseline', color='black', linewidth=2.5, alpha=0.8
            )

    # Plot Metrics
    for i, metric in enumerate(unique_metrics):
        runs = metric_models[metric_models['metric'] == metric]
        color = colors[i]
        
        all_hist = []
        for _, row in runs.iterrows():
            hist = row['history'].copy()
            all_hist.append(hist)
            
        if all_hist:
            combined_hist = pd.concat(all_hist)
            sns.lineplot(
                data=combined_hist, x='Epoch', y='Validation Loss',
                label=metric, color=color, linewidth=2.5, alpha=0.8
            )

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(f'Learning Curves: Validation Loss (Model Size: Largest)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "6_learning_curves_val.png"), dpi=300)
    plt.close()

def main():
    print("Loading data...")
    df = load_data()
    
    if df.empty:
        print("No data found in output/ directory.")
        return
        
    print(f"Loaded {len(df)} runs.")
    
    # 1. Scaling Law Plot
    plot_scaling_laws(df)
    
    # 2. Optimizer Efficiency
    plot_optimizer_efficiency(df)
    
    # 3. Trajectory Plot
    plot_loss_vs_flops_trajectory(df)
    
    # 4. Optimizer Cost
    plot_optimizer_cost_fraction(df)
    
    # 5. Universality
    plot_universality(df)
    
    # 6. Learning Curves
    plot_learning_curves(df)
    
    print(f"All plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
