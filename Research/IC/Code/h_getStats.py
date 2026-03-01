import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "eval_results/plots")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "stats_results")

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'global_plots', 'ARIMA'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'global_plots', 'LSTM'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'global_plots', 'Combined'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'aggregated_plots'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'heatmaps'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'box_plots'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'improvement_analysis'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'summary_csvs'), exist_ok=True)

def load_all_csvs():
    all_data = []
    
    # Iterate through all subdirectories
    for subdir in Path(INPUT_FOLDER).iterdir():
        if subdir.is_dir():
            csv_path = subdir / "model_comparison_results.csv"
            if csv_path.exists():
                # Read CSV
                df = pd.read_csv(csv_path, sep='|')
                # Add run identifier
                df['run_id'] = subdir.name
                all_data.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def categorize_dataset(dataset_name):
    if dataset_name == 'baseline':
        return 'Baseline'
    elif dataset_name == 'interpolated':
        return 'Without Correction'
    elif dataset_name == 'optimized':
        return 'With Correction'
    else:
        return 'Unknown'

def create_global_plots(df):
    metrics = ['R2', 'RMSE', 'MAE', 'MSE']
    
    # Add category column
    df['category'] = df['dataset'].apply(categorize_dataset)
    
    # Define color palette
    color_palette = {
        'Baseline': '#e74c3c',
        'Without Correction': '#f39c12',
        'With Correction': '#2ecc71'
    }
    
    # Process each model separately
    for model in df['method'].unique():
        model_data = df[df['method'] == model].copy()
        
        for metric in metrics:
            # Prepare data for plotting
            plot_data = model_data[['run_id', 'category', metric]].copy()
            
            # Sort by metric value (best to worst)
            # For R2, higher is better; for others, lower is better
            if metric == 'R2':
                plot_data = plot_data.sort_values(metric, ascending=False)
            else:
                plot_data = plot_data.sort_values(metric, ascending=True)
            
            # Calculate figure width based on number of data points
            num_points = len(plot_data)
            # Minimum 16 inches, add 0.3 inches per data point beyond 50
            fig_width = max(16, 16 + (num_points - 50) * 0.3) if num_points > 50 else 16
            
            plt.figure(figsize=(fig_width, 8))
            
            # Create bar plot
            ax = plt.subplot(111)
            
            x_labels = []
            x_positions = []
            colors = []
            values = []
            
            for idx, (_, row) in enumerate(plot_data.iterrows()):
                # Remove the "x--" prefix from run_id (e.g., "8--phi-4..." becomes "phi-4...")
                run_id_clean = '--'.join(row['run_id'].split('--')[1:]) if '--' in row['run_id'] else row['run_id']
                # Just use the run_id without category label
                x_labels.append(f"{run_id_clean}")
                x_positions.append(idx)
                colors.append(color_palette[row['category']])
                values.append(row[metric])
            
            bars = ax.bar(x_positions, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels inside each bar, rotated 90 degrees
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                       f'{value:.4f}', ha='center', va='center', fontsize=5, rotation=90, color='white', fontweight='bold')
            
            # Customize plot
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            
            if metric == 'R2':
                ax.set_title(f'{model} - Global {metric} Comparison (Best to Worst - Higher is Better)', fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'{model} - Global {metric} Comparison (Best to Worst - Lower is Better)', fontsize=14, fontweight='bold')
            
            # Show ALL labels, rotated 90 degrees with smaller font
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=90, ha='center', fontsize=6)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color_palette[cat], label=cat, alpha=0.8, edgecolor='black') 
                              for cat in ['Baseline', 'Without Correction', 'With Correction']]
            ax.legend(handles=legend_elements, loc='best', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, 'global_plots', model, f'global_{metric.lower()}_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Created global plots for {model}")

def create_combined_global_plots(df):
    metrics = ['R2', 'RMSE', 'MAE', 'MSE']
    
    # Add category column
    df['category'] = df['dataset'].apply(categorize_dataset)
    
    # Define color palette for categories
    color_palette = {
        'Baseline': '#e74c3c',
        'Without Correction': '#f39c12',
        'With Correction': '#2ecc71'
    }
    
    for metric in metrics:
        # Prepare data for both models
        lstm_data = df[df['method'] == 'LSTM'].copy()
        arima_data = df[df['method'] == 'ARIMA'].copy()
        
        # Add model column for identification
        lstm_data['model'] = 'LSTM'
        arima_data['model'] = 'ARIMA'
        
        # Combine both datasets
        combined_data = pd.concat([lstm_data, arima_data], ignore_index=True)
        
        # Sort by metric value
        if metric == 'R2':
            combined_data = combined_data.sort_values(metric, ascending=False)
        else:
            combined_data = combined_data.sort_values(metric, ascending=True)
        
        # Calculate figure dimensions
        num_points = len(combined_data)
        fig_width = max(20, 20 + (num_points - 100) * 0.15) if num_points > 100 else 20
        
        # Create single plot
        plt.figure(figsize=(fig_width, 8))
        ax = plt.subplot(111)
        
        x_labels = []
        x_positions = []
        colors = []
        values = []
        
        for idx, (_, row) in enumerate(combined_data.iterrows()):
            # Clean run_id
            run_id_clean = '--'.join(row['run_id'].split('--')[1:]) if '--' in row['run_id'] else row['run_id']
            # Add model name in bold to the label
            label = f"$\\bf{{{row['model']}}}$ - {run_id_clean}"
            x_labels.append(label)
            x_positions.append(idx)
            colors.append(color_palette[row['category']])
            values.append(row[metric])
        
        bars = ax.bar(x_positions, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels inside each bar, rotated 90 degrees
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                   f'{value:.4f}', ha='center', va='center', fontsize=5, rotation=90, color='white', fontweight='bold')
        
        # Customize plot
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        
        if metric == 'R2':
            ax.set_title(f'LSTM vs ARIMA - {metric} Comparison (Best to Worst - Higher is Better)', 
                        fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'LSTM vs ARIMA - {metric} Comparison (Best to Worst - Lower is Better)', 
                        fontsize=14, fontweight='bold')
        
        # Show ALL labels, rotated 90 degrees with smaller font
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=90, ha='center', fontsize=5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_palette[cat], label=cat, alpha=0.8, edgecolor='black') 
                          for cat in ['Baseline', 'Without Correction', 'With Correction']]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'global_plots', 'Combined', f'combined_{metric.lower()}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Created combined global plots (LSTM vs ARIMA)")

def create_aggregated_plots(df):
    metrics = ['R2', 'RMSE', 'MAE', 'MSE']
    
    df['category'] = df['dataset'].apply(categorize_dataset)
    
    color_palette = {
        'Baseline': '#e74c3c',
        'Without Correction': '#f39c12',
        'With Correction': '#2ecc71'
    }
    
    # Fixed order for categories
    category_order = ['Baseline', 'Without Correction', 'With Correction']
    
    for metric in metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Average by category (FIXED ORDER)
        avg_by_category = df.groupby('category')[metric].agg(['mean', 'std']).reset_index()
        # Ensure correct order
        avg_by_category['category'] = pd.Categorical(avg_by_category['category'], 
                                                     categories=category_order, 
                                                     ordered=True)
        avg_by_category = avg_by_category.sort_values('category')
        
        bars1 = ax1.bar(avg_by_category['category'], avg_by_category['mean'], 
                       color=[color_palette[cat] for cat in avg_by_category['category']],
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.errorbar(avg_by_category['category'], avg_by_category['mean'], 
                    yerr=avg_by_category['std'], fmt='none', color='black', 
                    capsize=5, linewidth=2)
        
        ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'Average {metric}', fontsize=12, fontweight='bold')
        ax1.set_title(f'Average {metric} by Category (with std)', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Average by model and category (FIXED ORDER)
        avg_by_model_cat = df.groupby(['method', 'category'])[metric].mean().reset_index()
        
        x = np.arange(len(avg_by_model_cat['method'].unique()))
        width = 0.25
        
        for i, category in enumerate(category_order):
            data = avg_by_model_cat[avg_by_model_cat['category'] == category]
            models = data['method'].values
            values = data[metric].values
            
            # Create x positions for this category
            x_pos = []
            for model in models:
                model_idx = list(avg_by_model_cat['method'].unique()).index(model)
                x_pos.append(model_idx)
            
            ax2.bar([p + i * width for p in x_pos], values, width, 
                   label=category, color=color_palette[category], 
                   alpha=0.8, edgecolor='black', linewidth=1)
        
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'Average {metric}', fontsize=12, fontweight='bold')
        ax2.set_title(f'Average {metric} by Model and Category', fontsize=13, fontweight='bold')
        ax2.set_xticks([p + width for p in x])
        ax2.set_xticklabels(avg_by_model_cat['method'].unique())
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'aggregated_plots', f'aggregated_{metric.lower()}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created aggregated plot for {metric}")

def create_summary_csv(df):
    df['category'] = df['dataset'].apply(categorize_dataset)
    
    metrics = ['R2', 'RMSE', 'MAE', 'MSE']
    
    # Overall statistics
    overall_stats = []
    for metric in metrics:
        overall_stats.append({
            'Metric': metric,
            'Overall_Mean': df[metric].mean(),
            'Overall_Std': df[metric].std(),
            'Overall_Min': df[metric].min(),
            'Overall_Max': df[metric].max(),
            'Overall_Median': df[metric].median()
        })
    
    overall_df = pd.DataFrame(overall_stats)
    
    # Statistics by category (FIXED ORDER)
    category_order = ['Baseline', 'Without Correction', 'With Correction']
    category_stats = []
    for metric in metrics:
        for category in category_order:
            cat_data = df[df['category'] == category][metric]
            category_stats.append({
                'Metric': metric,
                'Category': category,
                'Mean': cat_data.mean(),
                'Std': cat_data.std(),
                'Min': cat_data.min(),
                'Max': cat_data.max(),
                'Median': cat_data.median(),
                'Count': len(cat_data)
            })
    
    category_df = pd.DataFrame(category_stats)
    
    # Statistics by model
    model_stats = []
    for metric in metrics:
        for model in df['method'].unique():
            model_data = df[df['method'] == model][metric]
            model_stats.append({
                'Metric': metric,
                'Model': model,
                'Mean': model_data.mean(),
                'Std': model_data.std(),
                'Min': model_data.min(),
                'Max': model_data.max(),
                'Median': model_data.median(),
                'Count': len(model_data)
            })
    
    model_df = pd.DataFrame(model_stats)
    
    # Statistics by model and category (FIXED ORDER)
    model_category_stats = []
    for metric in metrics:
        for model in df['method'].unique():
            for category in category_order:
                mc_data = df[(df['method'] == model) & (df['category'] == category)][metric]
                if len(mc_data) > 0:
                    model_category_stats.append({
                        'Metric': metric,
                        'Model': model,
                        'Category': category,
                        'Mean': mc_data.mean(),
                        'Std': mc_data.std(),
                        'Min': mc_data.min(),
                        'Max': mc_data.max(),
                        'Median': mc_data.median(),
                        'Count': len(mc_data)
                    })
    
    model_category_df = pd.DataFrame(model_category_stats)
    
    # Save all summary statistics
    overall_df.to_csv(os.path.join(OUTPUT_FOLDER, 'summary_csvs', 'summary_overall.csv'), index=False)
    category_df.to_csv(os.path.join(OUTPUT_FOLDER, 'summary_csvs', 'summary_by_category.csv'), index=False)
    model_df.to_csv(os.path.join(OUTPUT_FOLDER, 'summary_csvs', 'summary_by_model.csv'), index=False)
    model_category_df.to_csv(os.path.join(OUTPUT_FOLDER, 'summary_csvs', 'summary_by_model_and_category.csv'), index=False)
    
    print("Created summary CSV files")
    
    return overall_df, category_df, model_df, model_category_df

def create_improvement_analysis(df):
    df['category'] = df['dataset'].apply(categorize_dataset)
    
    metrics = ['R2', 'RMSE', 'MAE', 'MSE']
    
    improvement_data = []
    
    for run_id in df['run_id'].unique():
        run_data = df[df['run_id'] == run_id]
        
        for model in run_data['method'].unique():
            model_data = run_data[run_data['method'] == model]
            
            baseline_row = model_data[model_data['category'] == 'Baseline']
            without_corr_row = model_data[model_data['category'] == 'Without Correction']
            with_corr_row = model_data[model_data['category'] == 'With Correction']
            
            if len(baseline_row) > 0 and len(without_corr_row) > 0 and len(with_corr_row) > 0:
                for metric in metrics:
                    baseline_val = baseline_row[metric].values[0]
                    without_val = without_corr_row[metric].values[0]
                    with_val = with_corr_row[metric].values[0]
                    
                    # Calculate improvement (for R2, higher is better; for others, lower is better)
                    if metric == 'R2':
                        improvement_without = ((without_val - baseline_val) / abs(baseline_val)) * 100 if baseline_val != 0 else 0
                        improvement_with = ((with_val - baseline_val) / abs(baseline_val)) * 100 if baseline_val != 0 else 0
                    else:
                        improvement_without = ((baseline_val - without_val) / baseline_val) * 100 if baseline_val != 0 else 0
                        improvement_with = ((baseline_val - with_val) / baseline_val) * 100 if baseline_val != 0 else 0
                    
                    improvement_data.append({
                        'Run_ID': run_id,
                        'Model': model,
                        'Metric': metric,
                        'Baseline_Value': baseline_val,
                        'Without_Correction_Value': without_val,
                        'With_Correction_Value': with_val,
                        'Improvement_Without_Correction_%': improvement_without,
                        'Improvement_With_Correction_%': improvement_with
                    })
    
    improvement_df = pd.DataFrame(improvement_data)
    improvement_df.to_csv(os.path.join(OUTPUT_FOLDER, 'improvement_analysis', 'improvement_analysis.csv'), index=False)
    
    # Create improvement visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        metric_data = improvement_df[improvement_df['Metric'] == metric]
        
        avg_improvement = metric_data.groupby('Model')[['Improvement_Without_Correction_%', 'Improvement_With_Correction_%']].mean()
        
        x = np.arange(len(avg_improvement.index))
        width = 0.35
        
        ax = axes[idx]
        bars1 = ax.bar(x - width/2, avg_improvement['Improvement_Without_Correction_%'], width, 
                      label='Without Correction', color='#f39c12', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, avg_improvement['Improvement_With_Correction_%'], width, 
                      label='With Correction', color='#2ecc71', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Improvement (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Average Improvement in {metric} from Baseline', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(avg_improvement.index)
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'improvement_analysis', 'improvement_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created improvement analysis")

def create_heatmaps(df):
    df['category'] = df['dataset'].apply(categorize_dataset)
    
    metrics = ['R2', 'RMSE', 'MAE', 'MSE']
    category_order = ['Baseline', 'Without Correction', 'With Correction']
    
    # Heatmap 1: Average performance by model and category
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        pivot_data = df.groupby(['method', 'category'])[metric].mean().reset_index()
        pivot_table = pivot_data.pivot(index='method', columns='category', values=metric)
        
        # Reorder columns (FIXED ORDER)
        pivot_table = pivot_table[category_order]
        
        ax = axes[idx]
        
        # Determine if higher is better
        if metric == 'R2':
            cmap = 'YlGn'  # Higher is better - green for high values
            fmt = '.4f'
        else:
            cmap = 'YlOrRd_r'  # Lower is better - green for low values
            fmt = '.4f'
        
        sns.heatmap(pivot_table, annot=True, fmt=fmt, cmap=cmap, 
                   ax=ax, cbar_kws={'label': metric}, linewidths=1, linecolor='black')
        ax.set_title(f'Average {metric} by Model and Category', fontsize=12, fontweight='bold')
        ax.set_xlabel('Category', fontsize=11, fontweight='bold')
        ax.set_ylabel('Model', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'heatmaps', 'heatmap_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created performance heatmap")
    
    # Heatmap 2: Standard deviation by model and category
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        pivot_data = df.groupby(['method', 'category'])[metric].std().reset_index()
        pivot_table = pivot_data.pivot(index='method', columns='category', values=metric)
        
        # Reorder columns (FIXED ORDER)
        pivot_table = pivot_table[category_order]
        
        ax = axes[idx]
        
        sns.heatmap(pivot_table, annot=True, fmt='.5f', cmap='Reds', 
                   ax=ax, cbar_kws={'label': f'{metric} Std Dev'}, linewidths=1, linecolor='black')
        ax.set_title(f'Standard Deviation of {metric} by Model and Category', fontsize=12, fontweight='bold')
        ax.set_xlabel('Category', fontsize=11, fontweight='bold')
        ax.set_ylabel('Model', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'heatmaps', 'heatmap_variability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created variability heatmap")

def create_box_plots(df):
    df['category'] = df['dataset'].apply(categorize_dataset)
    
    metrics = ['R2', 'RMSE', 'MAE', 'MSE']
    category_order = ['Baseline', 'Without Correction', 'With Correction']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    color_palette = {
        'Baseline': '#e74c3c',
        'Without Correction': '#f39c12',
        'With Correction': '#2ecc71'
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data
        plot_data = df[['method', 'category', metric]].copy()
        
        # Create box plot (FIXED ORDER)
        positions = []
        data_to_plot = []
        labels = []
        colors = []
        
        pos = 0
        for model in sorted(df['method'].unique()):  # Sort models alphabetically
            for category in category_order:  # Use fixed order
                data = plot_data[(plot_data['method'] == model) & (plot_data['category'] == category)][metric]
                if len(data) > 0:
                    positions.append(pos)
                    data_to_plot.append(data.values)
                    labels.append(f'{model}\n{category}')
                    colors.append(color_palette[category])
                    pos += 1
            pos += 0.5  # Add space between models
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='darkblue'),
                       meanprops=dict(linewidth=2, color='red', linestyle='--'))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'Distribution of {metric} Across Models and Categories', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_palette['Baseline'], label='Baseline', alpha=0.7),
            Patch(facecolor=color_palette['Without Correction'], label='Without Correction', alpha=0.7),
            Patch(facecolor=color_palette['With Correction'], label='With Correction', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'box_plots', 'box_plots_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created box plot distributions")

def main():
    print("Loading CSV files...")
    df = load_all_csvs()
    print(f"Loaded {len(df)} rows from {df['run_id'].nunique()} runs")
    
    print("\nCreating global plots...")
    create_global_plots(df)
    
    print("\nCreating combined global plots (LSTM vs ARIMA)...")
    create_combined_global_plots(df)
    
    print("\nCreating aggregated plots...")
    create_aggregated_plots(df)
    
    print("\nCreating heatmaps...")
    create_heatmaps(df)
    
    print("\nCreating box plots...")
    create_box_plots(df)
    
    print("\nCreating summary CSVs...")
    create_summary_csv(df)
    
    print("\nCreating improvement analysis...")
    create_improvement_analysis(df)
    
    print(f"\nAll statistics and plots have been saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()

