import os
import numpy as np
import pandas as pd
import warnings
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
from scipy.optimize import curve_fit

# ================================================================================

# --------------------------------------------------------------
# 1 COMPLEXITY vs FILTER

def plot_filter_vs_complexity(appended_benchmarks_df, OUTPUT_FOLDER):

    # Remove NaN values and filter = 0 for regression
    mask = ~(appended_benchmarks_df['filter'].isna() | appended_benchmarks_df['complexity'].isna())
    x_data = appended_benchmarks_df.loc[mask, 'filter'].values
    y_data = appended_benchmarks_df.loc[mask, 'complexity'].values

    # Try different functions and find the best fit
    best_r2 = -np.inf
    best_name = None
    best_params = None
    best_func = None
    best_equation = None

    # ALL DATA
    FUNCTIONS_TO_TEST = {
            'linear': {
                'func': lambda x, a, b: a * x + b,
                'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
                'initial_guess': [1, 1]
            },
            'quadratic': {
                'func': lambda x, a, b, c: a * x**2 + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'exponential': {
                'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
                'initial_guess': [1, 0.1, 1]
            },
            'logarithmic': {
                'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'power': {
                'func': lambda x, a, b, c: a * (x + 1)**b + c,
                'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
                'initial_guess': [1, 0.5, 1]
            }
        }

    for name, func_info in FUNCTIONS_TO_TEST.items():
        try:
            params, _ = curve_fit(func_info['func'], x_data, y_data, p0=func_info['initial_guess'], maxfev=10000)
            y_pred = func_info['func'](x_data, *params)
            r2 = 1 - (np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2))
            
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                best_params = params
                best_func = func_info['func']
                best_equation = func_info['equation'](params)
        except:
            print(f"Curve fitting failed for function: {name}")
            # Skip if curve fitting fails for this function
            pass

    # Create regression line with best fit
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = best_func(x_line, *best_params)

    # Get maximum complexity for each filter
    max_complexity_by_filter = appended_benchmarks_df.groupby('filter')['complexity'].max().reset_index()
    x_max = max_complexity_by_filter['filter'].values
    y_max = max_complexity_by_filter['complexity'].values

    # Try different functions and find the best fit for maximum values
    best_r2_max = -np.inf
    best_name_max = None
    best_params_max = None
    best_func_max = None
    best_equation_max = None

    # MAXIMUM VALUES
    FUNCTIONS_TO_TEST = {
        'linear': {
            'func': lambda x, a, b: a * x + b,
            'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
            'initial_guess': [1, 1]
        },
        'quadratic': {
            'func': lambda x, a, b, c: a * x**2 + b * x + c,
            'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
            'initial_guess': [1, 1, 1]
        },
        'exponential': {
            'func': lambda x, a, b, c: a * np.exp(b * x) + c,
            'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
            'initial_guess': [1, 0.1, 1]
        },
        'logarithmic': {
            'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
            'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
            'initial_guess': [1, 1, 1]
        },
        'power': {
            'func': lambda x, a, b, c: a * (x + 1)**b + c,
            'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
            'initial_guess': [1, 0.5, 1]
        }
    }

    for name, func_info in FUNCTIONS_TO_TEST.items():
        try:
            params, _ = curve_fit(func_info['func'], x_max, y_max, p0=func_info['initial_guess'], maxfev=10000)
            y_pred = func_info['func'](x_max, *params)
            r2 = 1 - (np.sum((y_max - y_pred)**2) / np.sum((y_max - np.mean(y_max))**2))
            
            if r2 > best_r2_max:
                best_r2_max = r2
                best_name_max = name
                best_params_max = params
                best_func_max = func_info['func']
                best_equation_max = func_info['equation'](params)
        except:
            # Skip if curve fitting fails for this function
            pass

    # Create regression line for maximum values
    x_line_max = np.linspace(x_max.min(), x_max.max(), 100)
    y_line_max = best_func_max(x_line_max, *best_params_max)

    # Create output folder for filter vs complexity
    filter_complexity_folder = os.path.join(OUTPUT_FOLDER, '1_filter_vs_complexity')
    if not os.path.exists(filter_complexity_folder):
        os.makedirs(filter_complexity_folder)

    # Plot scatter with regression lines using seaborn
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(x=appended_benchmarks_df['filter'], y=appended_benchmarks_df['complexity'],
                    alpha=0.6, label='Data', ax=ax)
    
    # Add model names below each data point
    for idx, row in appended_benchmarks_df.iterrows():
        if pd.notna(row['filter']) and pd.notna(row['complexity']):
            ax.text(row['filter'], row['complexity'], row['model'], 
                   fontsize=6, alpha=0.5, ha='center', va='top', color='gray')
    
    # Regression lines
    sns.lineplot(x=x_line, y=y_line, color='red', linewidth=2, linestyle='-',
                 label=f'Best fit all data ({best_name})\n{best_equation}\nR² = {best_r2:.6f}', ax=ax)
    sns.scatterplot(x=x_max, y=y_max, color='green', s=100, alpha=0.8, marker='D',
                    label='Maximum values', zorder=5, ax=ax)
    sns.lineplot(x=x_line_max, y=y_line_max, color='green', linewidth=2, linestyle='--',
                 label=f'Best fit max ({best_name_max})\n{best_equation_max}\nR² = {best_r2_max:.6f}', ax=ax)
    ax.set_xlabel('Filter')
    ax.set_ylabel('Complexity')
    ax.set_title('Filter vs Complexity')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5, fontsize=8)
    
    fig.savefig(os.path.join(filter_complexity_folder, 'regression.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Get average complexity for each filter (excluding filter = 0)
    avg_complexity_by_filter = appended_benchmarks_df.groupby('filter')['complexity'].mean().reset_index()
    x_avg = avg_complexity_by_filter['filter'].values
    y_avg = avg_complexity_by_filter['complexity'].values

    # Plot bar graph for average complexity using seaborn
    fig, ax = plt.subplots(figsize=(12, 9))
    avg_df = pd.DataFrame({'filter': x_avg, 'avg_complexity': y_avg})
    sns.barplot(data=avg_df, x='filter', y='avg_complexity', color='steelblue', alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
    ax.set_xlabel('Filter')
    ax.set_ylabel('Average Complexity')
    ax.set_title('Average Complexity by Filter Value')
    ax.grid(axis='y', alpha=0.3)
    # Add value labels on top of bars (slightly above each bar to avoid overlap)
    if len(y_avg) > 0:
        y_range = y_avg.max() - y_avg.min()
        if y_range == 0:
            offset = 0.01 * (y_avg.max() if y_avg.max() != 0 else 1)
        else:
            offset = y_range * 0.02
    else:
        offset = 0.01
    # Annotate bars using patch positions so labels align with seaborn's categorical positioning
    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        ax.text(x, height + offset, f"{height:.4f}", ha='center', va='bottom', fontsize=8, rotation=90)
    # Adjust y-axis limits to accommodate rotated text labels
    y_max_val = ax.get_ylim()[1]
    ax.set_ylim(0, y_max_val * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(filter_complexity_folder, 'average_complexity_bar.png'), bbox_inches='tight')
    plt.close(fig)

    # Plot bar graph for maximum complexity using seaborn
    fig, ax = plt.subplots(figsize=(12, 9))
    max_df = pd.DataFrame({'filter': x_max, 'max_complexity': y_max})
    sns.barplot(data=max_df, x='filter', y='max_complexity', color='darkgreen', alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
    ax.set_xlabel('Filter')
    ax.set_ylabel('Maximum Complexity')
    ax.set_title('Maximum Complexity by Filter Value')
    ax.grid(axis='y', alpha=0.3)
    # Add value labels on top of bars (slightly above each bar to avoid overlap)
    if len(y_max) > 0:
        y_range_max = y_max.max() - y_max.min()
        if y_range_max == 0:
            offset_max = 0.01 * (y_max.max() if y_max.max() != 0 else 1)
        else:
            offset_max = y_range_max * 0.02
    else:
        offset_max = 0.01
    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        ax.text(x, height + offset_max, f"{height:.4f}", ha='center', va='bottom', fontsize=8, rotation=90)
    # Adjust y-axis limits to accommodate rotated text labels
    y_max_val = ax.get_ylim()[1]
    ax.set_ylim(0, y_max_val * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(filter_complexity_folder, 'maximum_complexity_bar.png'), bbox_inches='tight')
    plt.close(fig)

    # Save regression information to text file
    with open(os.path.join(filter_complexity_folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
        f.write("FILTER VS COMPLEXITY - REGRESSION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("ALL DATA REGRESSION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best fit function: {best_name}\n")
        f.write(f"Expression: {best_equation}\n")
        f.write(f"R² score: {best_r2:.16f}\n")
        f.write(f"Parameters: {best_params}\n\n")
        
        f.write("MAXIMUM VALUES REGRESSION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best fit function: {best_name_max}\n")
        f.write(f"Expression: {best_equation_max}\n")
        f.write(f"R² score: {best_r2_max:.16f}\n")
        f.write(f"Parameters: {best_params_max}\n")

    # Save average complexity data to text file
    with open(os.path.join(filter_complexity_folder, 'average_complexity.txt'), 'w', encoding='utf-8') as f:
        f.write("FILTER VS AVERAGE COMPLEXITY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Filter':<15} {'Average Complexity':<20}\n")
        f.write("-" * 70 + "\n")
        for x, y in zip(x_avg, y_avg):
            f.write(f"{x:<15} {y:<20.16f}\n")
        f.write("\n")
        f.write(f"Overall mean: {y_avg.mean():.16f}\n")
        f.write(f"Overall std: {y_avg.std():.16f}\n")

    # Save maximum complexity data to text file
    with open(os.path.join(filter_complexity_folder, 'maximum_complexity.txt'), 'w', encoding='utf-8') as f:
        f.write("FILTER VS MAXIMUM COMPLEXITY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Filter':<15} {'Maximum Complexity':<20}\n")
        f.write("-" * 70 + "\n")
        for x, y in zip(x_max, y_max):
            f.write(f"{x:<15} {y:<20.16f}\n")


# --------------------------------------------------------------
# 1.1 FILTER vs NUMBER OF BINS

def plot_filter_vs_bin_count(appended_benchmarks_df, OUTPUT_FOLDER):

    # Remove NaN values for regression
    mask = ~(appended_benchmarks_df['filter'].isna() | appended_benchmarks_df['bin_count'].isna())
    x_data = appended_benchmarks_df.loc[mask, 'filter'].values
    y_data = appended_benchmarks_df.loc[mask, 'bin_count'].values

    # Try different functions and find the best fit
    best_r2 = -np.inf
    best_name = None
    best_params = None
    best_func = None
    best_equation = None

    FUNCTIONS_TO_TEST = {
            'linear': {
                'func': lambda x, a, b: a * x + b,
                'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
                'initial_guess': [1, 1]
            },
            'quadratic': {
                'func': lambda x, a, b, c: a * x**2 + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'exponential': {
                'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
                'initial_guess': [1, 0.1, 1]
            },
            'logarithmic': {
                'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'power': {
                'func': lambda x, a, b, c: a * (x + 1)**b + c,
                'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
                'initial_guess': [1, 0.5, 1]
            }
        }

    for name, func_info in FUNCTIONS_TO_TEST.items():
        try:
            params, _ = curve_fit(func_info['func'], x_data, y_data, p0=func_info['initial_guess'], maxfev=10000)
            y_pred = func_info['func'](x_data, *params)
            r2 = 1 - (np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2))
            
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                best_params = params
                best_func = func_info['func']
                best_equation = func_info['equation'](params)
        except:
            # Skip if curve fitting fails for this function
            pass

    # Create regression line with best fit
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = best_func(x_line, *best_params)

    # Get maximum bin_count for each filter
    max_bins_by_filter = appended_benchmarks_df.groupby('filter')['bin_count'].max().reset_index()
    x_max = max_bins_by_filter['filter'].values
    y_max = max_bins_by_filter['bin_count'].values

    # Try different functions and find the best fit for maximum values
    best_r2_max = -np.inf
    best_name_max = None
    best_params_max = None
    best_func_max = None
    best_equation_max = None

    for name, func_info in FUNCTIONS_TO_TEST.items():
        try:
            params, _ = curve_fit(func_info['func'], x_max, y_max, p0=func_info['initial_guess'], maxfev=10000)
            y_pred = func_info['func'](x_max, *params)
            r2 = 1 - (np.sum((y_max - y_pred)**2) / np.sum((y_max - np.mean(y_max))**2))
            
            if r2 > best_r2_max:
                best_r2_max = r2
                best_name_max = name
                best_params_max = params
                best_func_max = func_info['func']
                best_equation_max = func_info['equation'](params)
        except:
            # Skip if curve fitting fails for this function
            pass

    # Create regression line for maximum values
    x_line_max = np.linspace(x_max.min(), x_max.max(), 100)
    y_line_max = best_func_max(x_line_max, *best_params_max)

    # Create output folder
    folder = os.path.join(OUTPUT_FOLDER, '1_1_filter_vs_bin_count')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Plot scatter with regression lines using seaborn
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(x=appended_benchmarks_df['filter'], y=appended_benchmarks_df['bin_count'], alpha=0.6, label='Data', ax=ax)
    
    # Add model names below each data point
    for idx, row in appended_benchmarks_df.iterrows():
        if pd.notna(row['filter']) and pd.notna(row['bin_count']):
            ax.text(row['filter'], row['bin_count'], row['model'], 
                   fontsize=6, alpha=0.5, ha='center', va='top', color='gray')
    
    sns.lineplot(x=x_line, y=y_line, color='red', linewidth=2, linestyle='-', label=f'Best fit all data ({best_name})\n{best_equation}\nR² = {best_r2:.6f}', ax=ax)
    sns.scatterplot(x=x_max, y=y_max, color='green', s=100, alpha=0.8, marker='D', label='Maximum values', zorder=5, ax=ax)
    sns.lineplot(x=x_line_max, y=y_line_max, color='green', linewidth=2, linestyle='--', label=f'Best fit max ({best_name_max})\n{best_equation_max}\nR² = {best_r2_max:.6f}', ax=ax)
    ax.set_xlabel('Filter')
    ax.set_ylabel('Number of Bins')
    ax.set_title('Filter vs Number of Bins')
    
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5, fontsize=8)
    
    fig.savefig(os.path.join(folder, 'regression.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Average bin_count by filter
    avg_bins_by_filter = appended_benchmarks_df.groupby('filter')['bin_count'].mean().reset_index()
    x_avg = avg_bins_by_filter['filter'].values
    y_avg = avg_bins_by_filter['bin_count'].values

    fig, ax = plt.subplots(figsize=(12, 9))
    avg_df = pd.DataFrame({'filter': x_avg, 'avg_bins': y_avg})
    sns.barplot(data=avg_df, x='filter', y='avg_bins', color='steelblue', alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
    ax.set_xlabel('Filter')
    ax.set_ylabel('Average Number of Bins')
    ax.set_title('Average Number of Bins by Filter Value')
    ax.grid(axis='y', alpha=0.3)
    # Add value labels on top of bars (slightly above each bar to avoid overlap)
    if len(y_avg) > 0:
        y_range = y_avg.max() - y_avg.min()
        if y_range == 0:
            offset = 0.01 * (y_avg.max() if y_avg.max() != 0 else 1)
        else:
            offset = y_range * 0.02
    else:
        offset = 0.01
    # Use bar patch positions for annotations so labels align with plotted bars
    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        ax.text(x, height + offset, f"{height:.4f}", ha='center', va='bottom', fontsize=8, rotation=90)
    # Adjust y-axis limits to accommodate rotated text labels
    y_max_val = ax.get_ylim()[1]
    ax.set_ylim(0, y_max_val * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'average_bins_bar.png'), bbox_inches='tight')
    plt.close(fig)

    # Maximum bin_count bar
    fig, ax = plt.subplots(figsize=(12, 9))
    max_df = pd.DataFrame({'filter': x_max, 'max_bins': y_max})
    sns.barplot(data=max_df, x='filter', y='max_bins', color='darkgreen', alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
    ax.set_xlabel('Filter')
    ax.set_ylabel('Maximum Number of Bins')
    ax.set_title('Maximum Number of Bins by Filter Value')
    ax.grid(axis='y', alpha=0.3)
    # Add value labels on top of bars (slightly above each bar to avoid overlap)
    if len(y_max) > 0:
        y_range_max = y_max.max() - y_max.min()
        if y_range_max == 0:
            offset_max = 0.01 * (y_max.max() if y_max.max() != 0 else 1)
        else:
            offset_max = y_range_max * 0.02
    else:
        offset_max = 0.01
    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        ax.text(x, height + offset_max, f"{height:.4f}", ha='center', va='bottom', fontsize=8, rotation=90)
    # Adjust y-axis limits to accommodate rotated text labels
    y_max_val = ax.get_ylim()[1]
    ax.set_ylim(0, y_max_val * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'maximum_bins_bar.png'), bbox_inches='tight')
    plt.close(fig)

    # Save regression info
    with open(os.path.join(folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
        f.write("FILTER VS NUMBER OF BINS - REGRESSION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write("ALL DATA REGRESSION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best fit function: {best_name}\n")
        f.write(f"Expression: {best_equation}\n")
        f.write(f"R² score: {best_r2:.16f}\n")
        f.write(f"Parameters: {best_params}\n\n")
        f.write("MAXIMUM VALUES REGRESSION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best fit function: {best_name_max}\n")
        f.write(f"Expression: {best_equation_max}\n")
        f.write(f"R² score: {best_r2_max:.16f}\n")
        f.write(f"Parameters: {best_params_max}\n")


# --------------------------------------------------------------
# 2 COMPLEXITY vs TYPES

def plot_complexity_vs_types(appended_benchmarks_df, OUTPUT_FOLDER):

    # Convert types list to string for grouping (join with '-')
    appended_benchmarks_df['types_str'] = appended_benchmarks_df['types'].apply(lambda x: '-'.join(sorted(x)) if isinstance(x, list) and len(x) > 0 else 'unknown')

    # Remove rows with NaN complexity
    types_df = appended_benchmarks_df[~appended_benchmarks_df['complexity'].isna()].copy()

    # Get average complexity for each type combination
    avg_complexity_by_types = types_df.groupby('types_str')['complexity'].mean().reset_index()
    avg_complexity_by_types = avg_complexity_by_types.sort_values('complexity', ascending=False)
    x_types_avg = avg_complexity_by_types['types_str'].values
    y_types_avg = avg_complexity_by_types['complexity'].values

    # Get maximum complexity for each type combination
    max_complexity_by_types = types_df.groupby('types_str')['complexity'].max().reset_index()
    max_complexity_by_types = max_complexity_by_types.sort_values('complexity', ascending=False)
    x_types_max = max_complexity_by_types['types_str'].values
    y_types_max = max_complexity_by_types['complexity'].values

    # Create output folder for types vs complexity
    types_complexity_folder = os.path.join(OUTPUT_FOLDER, '2_types_vs_complexity')
    if not os.path.exists(types_complexity_folder):
        os.makedirs(types_complexity_folder)

    # Plot bar graph for average complexity by types using seaborn
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(x='complexity', y='types_str', data=avg_complexity_by_types, color='steelblue', alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
    ax.set_xlabel('Average Complexity')
    ax.set_ylabel('Types')
    ax.set_title('Average Complexity by Type Combination')
    ax.grid(axis='x', alpha=0.3)
    # Add value labels on bars using patch positions (horizontal bars)
    x_min, x_max_lim = ax.get_xlim()
    x_range_lim = x_max_lim - x_min if (x_max_lim - x_min) != 0 else 1
    x_offset = x_range_lim * 0.01
    for p in ax.patches:
        width = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(width + x_offset, y, f' {width:.4f}', va='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(types_complexity_folder, 'average_complexity_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot bar graph for maximum complexity by types using seaborn
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(x='complexity', y='types_str', data=max_complexity_by_types, color='darkgreen', alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
    ax.set_xlabel('Maximum Complexity')
    ax.set_ylabel('Types')
    ax.set_title('Maximum Complexity by Type Combination')
    ax.grid(axis='x', alpha=0.3)
    # Add value labels on bars using patch positions (horizontal bars)
    x_min, x_max_lim = ax.get_xlim()
    x_range_lim = x_max_lim - x_min if (x_max_lim - x_min) != 0 else 1
    x_offset = x_range_lim * 0.01
    for p in ax.patches:
        width = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(width + x_offset, y, f' {width:.4f}', va='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(types_complexity_folder, 'maximum_complexity_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save average complexity data to text file
    with open(os.path.join(types_complexity_folder, 'average_complexity.txt'), 'w', encoding='utf-8') as f:
        f.write("TYPES VS AVERAGE COMPLEXITY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Types':<30} {'Average Complexity':<20} {'Count':<10}\n")
        f.write("-" * 70 + "\n")
        for types_str in x_types_avg:
            avg_val = avg_complexity_by_types[avg_complexity_by_types['types_str'] == types_str]['complexity'].values[0]
            count = len(types_df[types_df['types_str'] == types_str])
            f.write(f"{types_str:<30} {avg_val:<20.16f} {count:<10}\n")
        f.write("\n")
        f.write(f"Overall mean: {y_types_avg.mean():.16f}\n")
        f.write(f"Overall std: {y_types_avg.std():.16f}\n")
        f.write(f"Number of unique type combinations: {len(x_types_avg)}\n")

    # Save maximum complexity data to text file
    with open(os.path.join(types_complexity_folder, 'maximum_complexity.txt'), 'w', encoding='utf-8') as f:
        f.write("TYPES VS MAXIMUM COMPLEXITY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Types':<30} {'Maximum Complexity':<20} {'Count':<10}\n")
        f.write("-" * 70 + "\n")
        for types_str in x_types_max:
            max_val = max_complexity_by_types[max_complexity_by_types['types_str'] == types_str]['complexity'].values[0]
            count = len(types_df[types_df['types_str'] == types_str])
            f.write(f"{types_str:<30} {max_val:<20.16f} {count:<10}\n")
        f.write("\n")
        f.write(f"Overall max of maxes: {y_types_max.max():.16f}\n")
        f.write(f"Number of unique type combinations: {len(x_types_max)}\n")

# --------------------------------------------------------------
# 3 COMPLEXITY vs NUMBER OF PARAMS

def plot_complexity_vs_num_params(appended_benchmarks_df, OUTPUT_FOLDER):

    # Remove NaN values for regression
    mask_params = ~(appended_benchmarks_df['count'].isna() | appended_benchmarks_df['complexity'].isna())
    x_data_params = appended_benchmarks_df.loc[mask_params, 'count'].values
    y_data_params = appended_benchmarks_df.loc[mask_params, 'complexity'].values

    # Try different functions and find the best fit
    best_r2_params = -np.inf
    best_name_params = None
    best_params_params = None
    best_func_params = None
    best_equation_params = None

    # ALL DATA
    FUNCTIONS_TO_TEST_PARAMS = {
        'linear': {
            'func': lambda x, a, b: a * x + b,
            'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
            'initial_guess': [1, 1]
        },
        'quadratic': {
            'func': lambda x, a, b, c: a * x**2 + b * x + c,
            'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
            'initial_guess': [1, 1, 1]
        },
        'exponential': {
            'func': lambda x, a, b, c: a * np.exp(b * x) + c,
            'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
            'initial_guess': [1, 0.1, 1]
        },
        'logarithmic': {
            'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
            'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
            'initial_guess': [1, 1, 1]
        },
        'power': {
            'func': lambda x, a, b, c: a * (x + 1)**b + c,
            'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
            'initial_guess': [1, 0.5, 1]
        }
    }

    for name, func_info in FUNCTIONS_TO_TEST_PARAMS.items():
        try:
            params, _ = curve_fit(func_info['func'], x_data_params, y_data_params, p0=func_info['initial_guess'], maxfev=10000)
            y_pred = func_info['func'](x_data_params, *params)
            r2 = 1 - (np.sum((y_data_params - y_pred)**2) / np.sum((y_data_params - np.mean(y_data_params))**2))
            
            if r2 > best_r2_params:
                best_r2_params = r2
                best_name_params = name
                best_params_params = params
                best_func_params = func_info['func']
                best_equation_params = func_info['equation'](params)
        except:
            print(f"Curve fitting failed for function: {name}")
            # Skip if curve fitting fails for this function
            pass

    # Create regression line with best fit
    x_line_params = np.linspace(x_data_params.min(), x_data_params.max(), 100)
    y_line_params = best_func_params(x_line_params, *best_params_params)

    # Create output folder for params vs complexity
    params_complexity_folder = os.path.join(OUTPUT_FOLDER, '3_params_vs_complexity')
    if not os.path.exists(params_complexity_folder):
        os.makedirs(params_complexity_folder)

    # Plot scatter with regression line using seaborn
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(x=x_data_params, y=y_data_params, alpha=0.6, label='Data', ax=ax)
    
    # Add model names below each data point
    mask_params_valid = ~(appended_benchmarks_df['count'].isna() | appended_benchmarks_df['complexity'].isna())
    for idx, row in appended_benchmarks_df[mask_params_valid].iterrows():
        ax.text(row['count'], row['complexity'], row['model'], 
               fontsize=6, alpha=0.5, ha='center', va='top', color='gray')
    
    sns.lineplot(x=x_line_params, y=y_line_params, color='red', linewidth=2, linestyle='-', label=f'Best fit ({best_name_params})\n{best_equation_params}\nR² = {best_r2_params:.6f}', ax=ax)
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Complexity')
    ax.set_title('Number of Parameters vs Complexity')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5, fontsize=8)
    
    fig.savefig(os.path.join(params_complexity_folder, 'regression.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Calculate number of bins using Freedman-Diaconis rule for parameter count
    # bin_width = 2 * IQR / n^(1/3)
    q75_params, q25_params = np.percentile(x_data_params, [75, 25])
    iqr_params = q75_params - q25_params
    n_params = len(x_data_params)
    bin_width_params = 2 * iqr_params / (n_params ** (1/3))
    if bin_width_params > 0:
        n_bins_fd = int(np.ceil((x_data_params.max() - x_data_params.min()) / bin_width_params))
        # Limit the number of bins to a reasonable range
    else:
        n_bins_fd = 30  # Default if calculation fails

    # Create equal-width bins in log scale to avoid empty gaps
    min_val = x_data_params.min()
    max_val = x_data_params.max()
    
    # Use log scale for binning
    log_min = np.log10(max(min_val, 1))
    log_max = np.log10(max_val + 1)
    
    # Create bins in log space using Freedman-Diaconis result directly
    n_log_bins = n_bins_fd  # Use FD result directly without modification
    log_bins = np.linspace(log_min, log_max, n_log_bins + 1)
    param_bins = 10 ** log_bins
    param_bins[0] = min_val  # Ensure exact min
    param_bins[-1] = max_val  # Ensure exact max
    
    bin_centers = []
    bin_labels = []
    avg_complexity_per_bin = []
    bin_counts = []
    bin_models = []  # Store models for each bin

    for i in range(len(param_bins) - 1):
        mask_bin = (x_data_params >= param_bins[i]) & (x_data_params < param_bins[i + 1])
        if i == len(param_bins) - 2:  # Include the last edge
            mask_bin = (x_data_params >= param_bins[i]) & (x_data_params <= param_bins[i + 1])
        
        if mask_bin.any():
            bin_centers.append((param_bins[i] + param_bins[i + 1]) / 2)
            # Format labels as order of magnitude - only show upper bound
            upper_mag = np.log10(max(param_bins[i+1], 1))
            bin_labels.append(f'10^{upper_mag:.1f}')
            avg_complexity_per_bin.append(y_data_params[mask_bin].mean())
            bin_counts.append(mask_bin.sum())
            # Get models in this bin
            models_in_bin = appended_benchmarks_df.loc[mask_params & mask_bin, 'model'].tolist()
            bin_models.append(models_in_bin)

    # Plot histogram: average complexity per parameter count interval using seaborn
    fig, ax = plt.subplots(figsize=(22, 16))
    hist_df = pd.DataFrame({'bin_range': bin_labels, 'avg_complexity': avg_complexity_per_bin})
    bars = ax.bar(range(len(bin_labels)), avg_complexity_per_bin, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_ylabel('Average Complexity', fontsize=12)
    ax.set_title(f'Average Complexity by Parameter Count', fontsize=13, pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Set x-axis ticks at transitions (between bars) instead of centers
    n_bins = len(bin_labels)
    # Create tick positions at transitions: -0.5, 0.5, 1.5, ..., n_bins-0.5
    transition_positions = [i - 0.5 for i in range(n_bins + 1)]
    
    # Create transition labels from param_bins (actual boundaries)
    transition_labels = []
    for i in range(n_bins + 1):
        if i < len(param_bins):
            mag = np.log10(max(param_bins[i], 1))
            transition_labels.append(f'10^{mag:.1f}')
        else:
            transition_labels.append('')
    
    ax.set_xticks(transition_positions)
    ax.set_xticklabels(transition_labels, rotation=90, ha='center', fontsize=8, fontweight='bold')
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.tick_params(axis='x', which='major', length=5)
    ax.set_xlabel('Number of Parameters', fontsize=12, labelpad=10)
    
    # Get max height for ylim adjustment
    max_height = max(avg_complexity_per_bin) if avg_complexity_per_bin else 1
    
    # Add model names above each bar (not inside)
    for i, (bar, models, count, avg_comp) in enumerate(zip(bars, bin_models, bin_counts, avg_complexity_per_bin)):
        # Get unique model names and sort them
        unique_models = sorted(set(models))
        
        bar_height = bar.get_height()
        bar_width = bar.get_width()
        
        if len(unique_models) > 0:
            # Position text ABOVE the bar (starting above the top with more spacing)
            text_y_position = bar_height + 0.002
            
            # Show all model names, each on its own line (matplotlib will wrap automatically within bbox)
            models_text = '\n'.join(unique_models)
            
            # Calculate approximate width in inches for wrapping
            # Each bar width in data coordinates needs to be converted
            fig_width_inches = 22
            data_width = n_bins
            bar_width_inches = (bar_width / data_width) * fig_width_inches * 0.9  # 90% of bar width for padding
            
            ax.text(bar.get_x() + bar_width / 2, text_y_position, 
                   models_text,
                   ha='center', va='bottom', fontsize=4.5, color='black', 
                   fontweight='normal', rotation=0,
                   wrap=True,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

    # Adjust y-axis limits to add space at top for model names
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)  # Add 15% space for model names
    
    # Add overall statistics text
    overall_avg = np.average(avg_complexity_per_bin, weights=bin_counts)
    stats_text = f'Overall Avg: {overall_avg:.4f}\nTotal samples: {sum(bin_counts)}\nNumber of bins: {len(bin_labels)}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, top=0.98)  # Margin for labels and top space
    fig.savefig(os.path.join(params_complexity_folder, 'average_complexity_histogram.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    # Save regression information to text file
    with open(os.path.join(params_complexity_folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
        f.write("NUMBER OF PARAMETERS VS COMPLEXITY - REGRESSION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("REGRESSION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best fit function: {best_name_params}\n")
        f.write(f"Expression: {best_equation_params}\n")
        f.write(f"R² score: {best_r2_params:.16f}\n")
        f.write(f"Parameters: {best_params_params}\n\n")
        
        f.write("HISTOGRAM INFORMATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Method: Freedman-Diaconis rule\n")
        f.write(f"Number of bins: {n_bins_fd}\n")
        f.write(f"Bin width: {bin_width_params:.2f} parameters\n")
        f.write(f"IQR: {iqr_params:.2f}\n")

    # Save average complexity per bin to text file with model lists
    with open(os.path.join(params_complexity_folder, 'average_complexity_per_bin.txt'), 'w', encoding='utf-8') as f:
        f.write("AVERAGE COMPLEXITY PER PARAMETER COUNT RANGE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Param Range Start':<20} {'Param Range End':<20} {'Avg Complexity':<20} {'Count':<10}\n")
        f.write("-" * 70 + "\n")
        for i, (center, avg_comp, count) in enumerate(zip(bin_centers, avg_complexity_per_bin, bin_counts)):
            range_start = param_bins[i]
            range_end = param_bins[i + 1]
            f.write(f"{range_start:<20.2f} {range_end:<20.2f} {avg_comp:<20.16f} {count:<10}\n")
        f.write("\n")
        overall_avg = np.average(avg_complexity_per_bin, weights=bin_counts)
        f.write(f"Overall weighted average: {overall_avg:.16f}\n")
        f.write(f"Total data points: {sum(bin_counts)}\n")

    # Save models per bin to a separate text file
    with open(os.path.join(params_complexity_folder, 'models_per_bin.txt'), 'w', encoding='utf-8') as f:
        f.write("MODELS IN EACH PARAMETER COUNT RANGE\n")
        f.write("=" * 100 + "\n\n")
        for i, (label, models) in enumerate(zip(bin_labels, bin_models)):
            range_start = param_bins[i]
            range_end = param_bins[i + 1]
            f.write(f"Bin {i+1}: {label} (Range: {range_start:.2f} - {range_end:.2f})\n")
            f.write(f"Number of models: {len(models)}\n")
            f.write("Models:\n")
            for model in sorted(set(models)):  # Use set to remove duplicates, then sort
                f.write(f"  - {model}\n")
            f.write("\n" + "-" * 100 + "\n\n")

    # Save overall statistics to text file
    with open(os.path.join(params_complexity_folder, 'statistics.txt'), 'w', encoding='utf-8') as f:
        f.write("PARAMETER COUNT VS COMPLEXITY - STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("COMPLEXITY STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of data points: {len(y_data_params)}\n")
        f.write(f"Mean complexity: {y_data_params.mean():.16f}\n")
        f.write(f"Median complexity: {np.median(y_data_params):.16f}\n")
        f.write(f"Std complexity: {y_data_params.std():.16f}\n")
        f.write(f"Min complexity: {y_data_params.min():.16f}\n")
        f.write(f"Max complexity: {y_data_params.max():.16f}\n\n")
        
        f.write("PARAMETER COUNT STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Min parameter count: {x_data_params.min():.0f}\n")
        f.write(f"Max parameter count: {x_data_params.max():.0f}\n")
        f.write(f"Mean parameter count: {x_data_params.mean():.2f}\n")
        f.write(f"Median parameter count: {np.median(x_data_params):.2f}\n")
        f.write(f"Q1 (25%): {q25_params:.2f}\n")
        f.write(f"Q3 (75%): {q75_params:.2f}\n")
        f.write(f"IQR: {iqr_params:.2f}\n")

# --------------------------------------------------------------
# 4 COMPLEXITY vs NUMBER OF BINS

def plot_complexity_vs_num_bins(appended_benchmarks_df, OUTPUT_FOLDER):

    # Remove NaN values for regression
    mask_bins = ~(appended_benchmarks_df['bin_count'].isna() | appended_benchmarks_df['complexity'].isna())
    x_data_bins = appended_benchmarks_df.loc[mask_bins, 'bin_count'].values
    y_data_bins = appended_benchmarks_df.loc[mask_bins, 'complexity'].values

    # Try different functions and find the best fit
    best_r2_bins = -np.inf
    best_name_bins = None
    best_params_bins = None
    best_func_bins = None
    best_equation_bins = None

    # ALL DATA
    FUNCTIONS_TO_TEST_BINS = {
        'linear': {
            'func': lambda x, a, b: a * x + b,
            'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
            'initial_guess': [1, 1]
        },
        'quadratic': {
            'func': lambda x, a, b, c: a * x**2 + b * x + c,
            'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
            'initial_guess': [1, 1, 1]
        },
        'exponential': {
            'func': lambda x, a, b, c: a * np.exp(b * x) + c,
            'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
            'initial_guess': [1, 0.1, 1]
        },
        'logarithmic': {
            'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
            'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
            'initial_guess': [1, 1, 1]
        },
        'power': {
            'func': lambda x, a, b, c: a * (x + 1)**b + c,
            'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
            'initial_guess': [1, 0.5, 1]
        }
    }

    for name, func_info in FUNCTIONS_TO_TEST_BINS.items():
        try:
            params, _ = curve_fit(func_info['func'], x_data_bins, y_data_bins, p0=func_info['initial_guess'], maxfev=10000)
            y_pred = func_info['func'](x_data_bins, *params)
            r2 = 1 - (np.sum((y_data_bins - y_pred)**2) / np.sum((y_data_bins - np.mean(y_data_bins))**2))
            
            if r2 > best_r2_bins:
                best_r2_bins = r2
                best_name_bins = name
                best_params_bins = params
                best_func_bins = func_info['func']
                best_equation_bins = func_info['equation'](params)
        except:
            print(f"Curve fitting failed for function: {name}")
            # Skip if curve fitting fails for this function
            pass

    # Create regression line with best fit
    x_line_bins = np.linspace(x_data_bins.min(), x_data_bins.max(), 100)
    y_line_bins = best_func_bins(x_line_bins, *best_params_bins)

    # Create output folder for bins vs complexity
    bins_complexity_folder = os.path.join(OUTPUT_FOLDER, '4_bins_vs_complexity')
    if not os.path.exists(bins_complexity_folder):
        os.makedirs(bins_complexity_folder)

    # Plot scatter with regression line using seaborn
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(x=x_data_bins, y=y_data_bins, alpha=0.6, label='Data', ax=ax)
    
    # Add model names below each data point
    mask_bins_valid = ~(appended_benchmarks_df['bin_count'].isna() | appended_benchmarks_df['complexity'].isna())
    for idx, row in appended_benchmarks_df[mask_bins_valid].iterrows():
        ax.text(row['bin_count'], row['complexity'], row['model'], 
               fontsize=6, alpha=0.5, ha='center', va='top', color='gray')
    
    sns.lineplot(x=x_line_bins, y=y_line_bins, color='red', linewidth=2, linestyle='-', label=f'Best fit ({best_name_bins})\n{best_equation_bins}\nR² = {best_r2_bins:.6f}', ax=ax)
    ax.set_xlabel('Number of Bins')
    ax.set_ylabel('Complexity')
    ax.set_title('Number of Bins vs Complexity')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5, fontsize=8)
    
    fig.savefig(os.path.join(bins_complexity_folder, 'regression.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Calculate number of bins using Freedman-Diaconis rule for bin count
    # bin_width = 2 * IQR / n^(1/3)
    q75_bins, q25_bins = np.percentile(x_data_bins, [75, 25])
    iqr_bins = q75_bins - q25_bins
    n_bins_data = len(x_data_bins)
    bin_width_bins = 2 * iqr_bins / (n_bins_data ** (1/3))
    if bin_width_bins > 0:
        n_bins_fd_bins = int(np.ceil((x_data_bins.max() - x_data_bins.min()) / bin_width_bins))
        # Limit the number of bins to a reasonable range
    else:
        n_bins_fd_bins = 30  # Default if calculation fails

    # Create equal-width bins in log scale to avoid empty gaps
    min_val_bins = x_data_bins.min()
    max_val_bins = x_data_bins.max()
    
    # Use log scale for binning
    log_min_bins = np.log10(max(min_val_bins, 1))
    log_max_bins = np.log10(max_val_bins + 1)
    
    # Create bins in log space using Freedman-Diaconis result directly
    n_log_bins = n_bins_fd_bins  # Use FD result directly without modification
    log_bins_array = np.linspace(log_min_bins, log_max_bins, n_log_bins + 1)
    bin_bins = 10 ** log_bins_array
    bin_bins[0] = min_val_bins  # Ensure exact min
    bin_bins[-1] = max_val_bins  # Ensure exact max
    
    bin_centers_bins = []
    bin_labels_bins = []
    avg_complexity_per_bin_bins = []
    bin_counts_bins = []
    bin_models_bins = []  # Store models for each bin

    for i in range(len(bin_bins) - 1):
        mask_bin = (x_data_bins >= bin_bins[i]) & (x_data_bins < bin_bins[i + 1])
        if i == len(bin_bins) - 2:  # Include the last edge
            mask_bin = (x_data_bins >= bin_bins[i]) & (x_data_bins <= bin_bins[i + 1])
        
        if mask_bin.any():
            bin_centers_bins.append((bin_bins[i] + bin_bins[i + 1]) / 2)
            # Format labels as order of magnitude - show range
            lower_mag = np.log10(max(bin_bins[i], 1))
            upper_mag = np.log10(max(bin_bins[i+1], 1))
            bin_labels_bins.append(f'10^{lower_mag:.1f}-10^{upper_mag:.1f}')
            avg_complexity_per_bin_bins.append(y_data_bins[mask_bin].mean())
            bin_counts_bins.append(mask_bin.sum())
            # Get models in this bin
            models_in_bin = appended_benchmarks_df.loc[mask_bins & mask_bin, 'model'].tolist()
            bin_models_bins.append(models_in_bin)

    # Plot histogram: average complexity per bin count interval using seaborn
    fig, ax = plt.subplots(figsize=(22, 16))
    hist_df = pd.DataFrame({'bin_range': bin_labels_bins, 'avg_complexity': avg_complexity_per_bin_bins})
    bars = ax.bar(range(len(bin_labels_bins)), avg_complexity_per_bin_bins, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_ylabel('Average Complexity', fontsize=12)
    ax.set_title(f'Average Complexity by Bin Count', fontsize=13, pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Set x-axis ticks at transitions (between bars) instead of centers
    n_bins = len(bin_labels_bins)
    # Create tick positions at transitions: -0.5, 0.5, 1.5, ..., n_bins-0.5
    transition_positions = [i - 0.5 for i in range(n_bins + 1)]
    
    # Create transition labels from bin_bins (actual boundaries)
    transition_labels = []
    for i in range(n_bins + 1):
        if i < len(bin_bins):
            mag = np.log10(max(bin_bins[i], 1))
            transition_labels.append(f'10^{mag:.1f}')
        else:
            transition_labels.append('')
    
    ax.set_xticks(transition_positions)
    ax.set_xticklabels(transition_labels, rotation=90, ha='center', fontsize=8, fontweight='bold')
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.tick_params(axis='x', which='major', length=5)
    ax.set_xlabel('Number of Bins', fontsize=12, labelpad=10)
    
    # Get max height for ylim adjustment
    max_height = max(avg_complexity_per_bin_bins) if avg_complexity_per_bin_bins else 1
    
    # Add model names above each bar (not inside)
    for i, (bar, models, count, avg_comp) in enumerate(zip(bars, bin_models_bins, bin_counts_bins, avg_complexity_per_bin_bins)):
        # Get unique model names and sort them
        unique_models = sorted(set(models))
        
        bar_height = bar.get_height()
        bar_width = bar.get_width()
        
        if len(unique_models) > 0:
            # Position text ABOVE the bar (starting above the top with more spacing)
            text_y_position = bar_height + 0.002
            
            # Show all model names, each on its own line (matplotlib will wrap automatically within bbox)
            models_text = '\n'.join(unique_models)
            
            # Calculate approximate width in inches for wrapping
            # Each bar width in data coordinates needs to be converted
            fig_width_inches = 22
            data_width = n_bins
            bar_width_inches = (bar_width / data_width) * fig_width_inches * 0.9  # 90% of bar width for padding
            
            ax.text(bar.get_x() + bar_width / 2, text_y_position, 
                   models_text,
                   ha='center', va='bottom', fontsize=4.5, color='black', 
                   fontweight='normal', rotation=0,
                   wrap=True,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

    # Adjust y-axis limits to add space at top for model names
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)  # Add 15% space for model names
    
    # Add overall statistics text
    overall_avg_bins = np.average(avg_complexity_per_bin_bins, weights=bin_counts_bins)
    stats_text = f'Overall Avg: {overall_avg_bins:.4f}\nTotal samples: {sum(bin_counts_bins)}\nNumber of bins: {len(bin_labels_bins)}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, top=0.98)  # Margin for labels and top space
    fig.savefig(os.path.join(bins_complexity_folder, 'average_complexity_histogram.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    # Save regression information to text file
    with open(os.path.join(bins_complexity_folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
        f.write("NUMBER OF BINS VS COMPLEXITY - REGRESSION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("REGRESSION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best fit function: {best_name_bins}\n")
        f.write(f"Expression: {best_equation_bins}\n")
        f.write(f"R² score: {best_r2_bins:.16f}\n")
        f.write(f"Parameters: {best_params_bins}\n\n")
        
        f.write("HISTOGRAM INFORMATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Method: Freedman-Diaconis rule\n")
        f.write(f"Number of bins: {n_bins_fd_bins}\n")
        f.write(f"Bin width: {bin_width_bins:.2f} bins\n")
        f.write(f"IQR: {iqr_bins:.2f}\n")

    # Save average complexity per bin to text file with model lists
    with open(os.path.join(bins_complexity_folder, 'average_complexity_per_bin.txt'), 'w', encoding='utf-8') as f:
        f.write("AVERAGE COMPLEXITY PER BIN COUNT RANGE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Bin Range Start':<20} {'Bin Range End':<20} {'Avg Complexity':<20} {'Count':<10}\n")
        f.write("-" * 70 + "\n")
        for i, (center, avg_comp, count) in enumerate(zip(bin_centers_bins, avg_complexity_per_bin_bins, bin_counts_bins)):
            range_start = bin_bins[i]
            range_end = bin_bins[i + 1]
            f.write(f"{range_start:<20.2f} {range_end:<20.2f} {avg_comp:<20.16f} {count:<10}\n")
        f.write("\n")
        overall_avg_bins = np.average(avg_complexity_per_bin_bins, weights=bin_counts_bins)
        f.write(f"Overall weighted average: {overall_avg_bins:.16f}\n")
        f.write(f"Total data points: {sum(bin_counts_bins)}\n")

    # Save models per bin to a separate text file
    with open(os.path.join(bins_complexity_folder, 'models_per_bin.txt'), 'w', encoding='utf-8') as f:
        f.write("MODELS IN EACH BIN COUNT RANGE\n")
        f.write("=" * 100 + "\n\n")
        for i, (label, models) in enumerate(zip(bin_labels_bins, bin_models_bins)):
            range_start = bin_bins[i]
            range_end = bin_bins[i + 1]
            f.write(f"Bin {i+1}: {label} (Range: {range_start:.2f} - {range_end:.2f})\n")
            f.write(f"Number of models: {len(models)}\n")
            f.write("Models:\n")
            for model in sorted(set(models)):  # Use set to remove duplicates, then sort
                f.write(f"  - {model}\n")
            f.write("\n" + "-" * 100 + "\n\n")

    # Save overall statistics to text file
    with open(os.path.join(bins_complexity_folder, 'statistics.txt'), 'w', encoding='utf-8') as f:
        f.write("BIN COUNT VS COMPLEXITY - STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("COMPLEXITY STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of data points: {len(y_data_bins)}\n")
        f.write(f"Mean complexity: {y_data_bins.mean():.16f}\n")
        f.write(f"Median complexity: {np.median(y_data_bins):.16f}\n")
        f.write(f"Std complexity: {y_data_bins.std():.16f}\n")
        f.write(f"Min complexity: {y_data_bins.min():.16f}\n")
        f.write(f"Max complexity: {y_data_bins.max():.16f}\n\n")
        
        f.write("BIN COUNT STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Min bin count: {x_data_bins.min():.0f}\n")
        f.write(f"Max bin count: {x_data_bins.max():.0f}\n")
        f.write(f"Mean bin count: {x_data_bins.mean():.2f}\n")
        f.write(f"Median bin count: {np.median(x_data_bins):.2f}\n")
        f.write(f"Q1 (25%): {q25_bins:.2f}\n")
        f.write(f"Q3 (75%): {q75_bins:.2f}\n")
        f.write(f"IQR: {iqr_bins:.2f}\n")

# ================================================================================
   
# --------------------------------------------------------------
# 5 FOR EACH BENCHMARK - COMPLEXITY vs BENCHMARK

def analyze_benchmarks_vs_complexity(appended_benchmarks_df, OUTPUT_FOLDER):
    # Get rows in header that start with "BENCH"
    BENCH_ROWS_NAMES = appended_benchmarks_df.columns[appended_benchmarks_df.columns.str.startswith('BENCH')].tolist()

    # Create output folder for benchmarks vs complexity
    benchmarks_complexity_folder = os.path.join(OUTPUT_FOLDER, '5_benchmarks_vs_complexity')
    if not os.path.exists(benchmarks_complexity_folder):
        os.makedirs(benchmarks_complexity_folder)

    # Dictionary to store all regression results
    all_benchmark_results = {}

    # Process each benchmark column
    for bench_name in BENCH_ROWS_NAMES:
        print(f"Processing benchmark: {bench_name}")
        
        # Remove NaN values for regression
        mask_bench = ~(appended_benchmarks_df[bench_name].isna() | appended_benchmarks_df['complexity'].isna())
        x_data_bench = appended_benchmarks_df.loc[mask_bench, bench_name].values
        y_data_bench = appended_benchmarks_df.loc[mask_bench, 'complexity'].values
        
        # Skip if not enough data points
        if len(x_data_bench) < 3:
            print(f"  Skipping {bench_name}: insufficient data points ({len(x_data_bench)})")
            continue
        
        # Try different functions and find the best fit
        best_r2_bench = -np.inf
        best_name_bench = None
        best_params_bench = None
        best_func_bench = None
        best_equation_bench = None
        
        FUNCTIONS_TO_TEST_BENCH = {
            'linear': {
                'func': lambda x, a, b: a * x + b,
                'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
                'initial_guess': [1, 1]
            },
            'quadratic': {
                'func': lambda x, a, b, c: a * x**2 + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'exponential': {
                'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
                'initial_guess': [1, 0.1, 1]
            },
            'logarithmic': {
                'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'power': {
                'func': lambda x, a, b, c: a * (x + 1)**b + c,
                'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
                'initial_guess': [1, 0.5, 1]
            }
        }
        
        for name, func_info in FUNCTIONS_TO_TEST_BENCH.items():
            try:
                params, _ = curve_fit(func_info['func'], x_data_bench, y_data_bench, 
                                        p0=func_info['initial_guess'], maxfev=10000)
                y_pred = func_info['func'](x_data_bench, *params)
                r2 = 1 - (np.sum((y_data_bench - y_pred)**2) / np.sum((y_data_bench - np.mean(y_data_bench))**2))
                
                if r2 > best_r2_bench:
                    best_r2_bench = r2
                    best_name_bench = name
                    best_params_bench = params
                    best_func_bench = func_info['func']
                    best_equation_bench = func_info['equation'](params)
            except:
                # Skip if curve fitting fails for this function
                pass
        
        # If no fit was successful, skip this benchmark
        if best_func_bench is None:
            print(f"  Skipping {bench_name}: curve fitting failed for all functions")
            continue
        
        # Create regression line with best fit
        x_line_bench = np.linspace(x_data_bench.min(), x_data_bench.max(), 100)
        y_line_bench = best_func_bench(x_line_bench, *best_params_bench)
        
        # Calculate linear regression for comparison
        linear_params, _ = curve_fit(lambda x, a, b: a * x + b, x_data_bench, y_data_bench, 
                                     p0=[1, 1], maxfev=10000)
        y_pred_linear = linear_params[0] * x_data_bench + linear_params[1]
        r2_linear = 1 - (np.sum((y_data_bench - y_pred_linear)**2) / np.sum((y_data_bench - np.mean(y_data_bench))**2))
        y_line_linear = linear_params[0] * x_line_bench + linear_params[1]
        linear_equation = f'y = {linear_params[0]:.16f}x + {linear_params[1]:.16f}'
        
        # Create subfolder for this benchmark
        bench_clean_name = bench_name.replace('BENCH-', '').replace('_', '-').lower()
        bench_folder = os.path.join(benchmarks_complexity_folder, bench_clean_name)
        if not os.path.exists(bench_folder):
            os.makedirs(bench_folder)
        
        # Plot scatter with both regression lines
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(x_data_bench, y_data_bench, alpha=0.6, label='Data')
        
        # Add model names below each data point
        for idx, (x, y) in enumerate(zip(x_data_bench, y_data_bench)):
            model_name = appended_benchmarks_df.loc[mask_bench, 'model'].iloc[idx]
            ax.text(x, y, f'  {model_name}', fontsize=6, alpha=0.5, color='gray', ha='center', va='top')
        
        ax.plot(x_line_bench, y_line_bench, 'r-', linewidth=2, label=f'Free regression ({best_name_bench})\n{best_equation_bench}\nR² = {best_r2_bench:.6f}')
        ax.plot(x_line_bench, y_line_linear, 'b--', linewidth=2, label=f'Linear regression\n{linear_equation}\nR² = {r2_linear:.6f}')
        ax.set_xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
        ax.set_ylabel('Complexity')
        ax.set_title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Complexity')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5, fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(bench_folder, 'regression.png'))
        plt.close()
        
        # Calculate Pearson correlation coefficient
        correlation = np.corrcoef(x_data_bench, y_data_bench)[0, 1]
        
        # Save regression information to text file
        with open(os.path.join(bench_folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{bench_name} VS COMPLEXITY - REGRESSION ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("REGRESSION:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Best fit function: {best_name_bench}\n")
            f.write(f"Expression: {best_equation_bench}\n")
            f.write(f"R² score: {best_r2_bench:.16f}\n")
            f.write(f"Pearson correlation: {correlation:.16f}\n")
            f.write(f"Parameters: {best_params_bench}\n")
        
        # Save statistics to text file
        with open(os.path.join(bench_folder, 'statistics.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{bench_name} VS COMPLEXITY - STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("COMPLEXITY STATISTICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Number of data points: {len(y_data_bench)}\n")
            f.write(f"Mean complexity: {y_data_bench.mean():.16f}\n")
            f.write(f"Median complexity: {np.median(y_data_bench):.16f}\n")
            f.write(f"Std complexity: {y_data_bench.std():.16f}\n")
            f.write(f"Min complexity: {y_data_bench.min():.16f}\n")
            f.write(f"Max complexity: {y_data_bench.max():.16f}\n\n")
            
            f.write(f"{bench_name} STATISTICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Min score: {x_data_bench.min():.16f}\n")
            f.write(f"Max score: {x_data_bench.max():.16f}\n")
            f.write(f"Mean score: {x_data_bench.mean():.16f}\n")
            f.write(f"Median score: {np.median(x_data_bench):.16f}\n")
            f.write(f"Std score: {x_data_bench.std():.16f}\n")
        
        # Store results for summary
        all_benchmark_results[bench_name] = {
            'best_function': best_name_bench,
            'r2_score': best_r2_bench,
            'r2_linear': r2_linear,
            'correlation': correlation,
            'equation': best_equation_bench,
            'linear_equation': linear_equation,
            'data_points': len(x_data_bench)
        }
        
        print(f"  Completed {bench_name}: R² = {best_r2_bench:.6f}, Correlation = {correlation:.6f}")

    # ==============================================================================
    # PROCESS "ALL" - Concatenate all benchmark data with Min-Max normalization
    # ==============================================================================
    print("\nProcessing 'all' (concatenated benchmark data with Min-Max normalization)...")
    
    # Collect all data points from all benchmarks with Min-Max normalization per benchmark
    all_x_data = []
    all_y_data = []
    all_model_names = []
    
    for bench_name in BENCH_ROWS_NAMES:
        mask_bench = ~(appended_benchmarks_df[bench_name].isna() | appended_benchmarks_df['complexity'].isna())
        x_data_bench = appended_benchmarks_df.loc[mask_bench, bench_name].values
        y_data_bench = appended_benchmarks_df.loc[mask_bench, 'complexity'].values
        model_names_bench = appended_benchmarks_df.loc[mask_bench, 'model'].values
        
        if len(x_data_bench) >= 3:
            # Apply Min-Max normalization to scale to [0, 1]
            # Best model (max score) = 1.0, Worst model (min score) = 0.0
            x_min = x_data_bench.min()
            x_max = x_data_bench.max()
            if x_max > x_min:  # Avoid division by zero
                x_normalized = (x_data_bench - x_min) / (x_max - x_min)
            else:
                x_normalized = np.ones_like(x_data_bench) * 0.5  # All same value
            
            all_x_data.extend(x_normalized)
            all_y_data.extend(y_data_bench)
            all_model_names.extend(model_names_bench)
    
    # Convert to numpy arrays
    all_x_data = np.array(all_x_data)
    all_y_data = np.array(all_y_data)
    all_model_names = np.array(all_model_names)
    
    if len(all_x_data) >= 3:
        # Try different functions and find the best fit
        best_r2_all = -np.inf
        best_name_all = None
        best_params_all = None
        best_func_all = None
        best_equation_all = None
        
        FUNCTIONS_TO_TEST_ALL = {
            'linear': {
                'func': lambda x, a, b: a * x + b,
                'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
                'initial_guess': [1, 1]
            },
            'quadratic': {
                'func': lambda x, a, b, c: a * x**2 + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'exponential': {
                'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
                'initial_guess': [1, 0.1, 1]
            },
            'logarithmic': {
                'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'power': {
                'func': lambda x, a, b, c: a * (x + 1)**b + c,
                'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
                'initial_guess': [1, 0.5, 1]
            }
        }
        
        for name, func_info in FUNCTIONS_TO_TEST_ALL.items():
            try:
                params, _ = curve_fit(func_info['func'], all_x_data, all_y_data, 
                                    p0=func_info['initial_guess'], maxfev=10000)
                y_pred = func_info['func'](all_x_data, *params)
                r2 = 1 - (np.sum((all_y_data - y_pred)**2) / np.sum((all_y_data - np.mean(all_y_data))**2))
                
                if r2 > best_r2_all:
                    best_r2_all = r2
                    best_name_all = name
                    best_params_all = params
                    best_func_all = func_info['func']
                    best_equation_all = func_info['equation'](params)
            except:
                pass
        
        if best_func_all is not None:
            # Create regression line with best fit
            x_line_all = np.linspace(all_x_data.min(), all_x_data.max(), 100)
            y_line_all = best_func_all(x_line_all, *best_params_all)
            
            # Calculate linear regression for comparison
            linear_params, _ = curve_fit(lambda x, a, b: a * x + b, all_x_data, all_y_data, 
                                        p0=[1, 1], maxfev=10000)
            y_pred_linear = linear_params[0] * all_x_data + linear_params[1]
            r2_linear_all = 1 - (np.sum((all_y_data - y_pred_linear)**2) / np.sum((all_y_data - np.mean(all_y_data))**2))
            y_line_linear = linear_params[0] * x_line_all + linear_params[1]
            linear_equation_all = f'y = {linear_params[0]:.16f}x + {linear_params[1]:.16f}'
            
            # Create subfolder for "all"
            all_folder = os.path.join(benchmarks_complexity_folder, 'all')
            if not os.path.exists(all_folder):
                os.makedirs(all_folder)
            
            # Plot scatter with both regression lines
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.scatter(all_x_data, all_y_data, alpha=0.6, label='Data')
            
            # Add model names below each data point
            for idx, (x, y) in enumerate(zip(all_x_data, all_y_data)):
                model_name = all_model_names[idx]
                ax.text(x, y, f'  {model_name}', fontsize=6, alpha=0.5, color='gray', ha='center', va='top')
            
            ax.plot(x_line_all, y_line_all, 'r-', linewidth=2, label=f'Free regression ({best_name_all})\n{best_equation_all}\nR² = {best_r2_all:.6f}')
            ax.plot(x_line_all, y_line_linear, 'b--', linewidth=2, label=f'Linear regression\n{linear_equation_all}\nR² = {r2_linear_all:.6f}')
            ax.set_xlabel('Normalized Benchmark Score (All Benchmarks, Min-Max [0,1])')
            ax.set_ylabel('Complexity')
            ax.set_title('All Benchmarks vs Complexity (Normalized)')
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5, fontsize=8)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(all_folder, 'regression.png'))
            plt.close()
            
            # Calculate Pearson correlation coefficient
            correlation_all = np.corrcoef(all_x_data, all_y_data)[0, 1]
            
            # Save regression information to text file
            with open(os.path.join(all_folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
                f.write("ALL BENCHMARKS VS COMPLEXITY - REGRESSION ANALYSIS\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("REGRESSION:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Best fit function: {best_name_all}\n")
                f.write(f"Expression: {best_equation_all}\n")
                f.write(f"R² score: {best_r2_all:.16f}\n")
                f.write(f"Pearson correlation: {correlation_all:.16f}\n")
                f.write(f"Parameters: {best_params_all}\n")
            
            # Save statistics to text file
            with open(os.path.join(all_folder, 'statistics.txt'), 'w', encoding='utf-8') as f:
                f.write("ALL BENCHMARKS VS COMPLEXITY - STATISTICS\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("COMPLEXITY STATISTICS:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Number of data points: {len(all_y_data)}\n")
                f.write(f"Mean complexity: {all_y_data.mean():.16f}\n")
                f.write(f"Median complexity: {np.median(all_y_data):.16f}\n")
                f.write(f"Std complexity: {all_y_data.std():.16f}\n")
                f.write(f"Min complexity: {all_y_data.min():.16f}\n")
                f.write(f"Max complexity: {all_y_data.max():.16f}\n\n")
                
                f.write("BENCHMARK SCORE STATISTICS:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Min score: {all_x_data.min():.16f}\n")
                f.write(f"Max score: {all_x_data.max():.16f}\n")
                f.write(f"Mean score: {all_x_data.mean():.16f}\n")
                f.write(f"Median score: {np.median(all_x_data):.16f}\n")
                f.write(f"Std score: {all_x_data.std():.16f}\n")
            
            # Store results for summary
            all_benchmark_results['ALL'] = {
                'best_function': best_name_all,
                'r2_score': best_r2_all,
                'r2_linear': r2_linear_all,
                'correlation': correlation_all,
                'equation': best_equation_all,
                'linear_equation': linear_equation_all,
                'data_points': len(all_x_data)
            }
            
            print(f"  Completed 'all': R² = {best_r2_all:.6f}, Correlation = {correlation_all:.6f}")
    
    # ==============================================================================
    # END OF "ALL" PROCESSING
    # ==============================================================================

    # Create summary file with all benchmark results
    with open(os.path.join(benchmarks_complexity_folder, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("BENCHMARKS VS COMPLEXITY - SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Benchmark':<35} {'Best Fit':<15} {'R² Score':<15} {'Correlation':<15} {'Data Points':<15}\n")
        f.write("-" * 100 + "\n")
        
        # Sort by R² score (descending), but always put "ALL" last
        sorted_results = sorted(all_benchmark_results.items(), key=lambda x: x[1]['r2_score'], reverse=True)
        # Move "ALL" to the end if it exists
        sorted_results = [item for item in sorted_results if item[0] != 'ALL'] + \
                        [item for item in sorted_results if item[0] == 'ALL']
        
        for bench_name, results in sorted_results:
            f.write(f"{bench_name:<35} {results['best_function']:<15} {results['r2_score']:<15.16f} "
                    f"{results['correlation']:<15.16f} {results['data_points']:<15}\n")
        
        # Calculate and add averages
        if len(all_benchmark_results) > 0:
            avg_r2 = sum(r['r2_score'] for r in all_benchmark_results.values()) / len(all_benchmark_results)
            avg_corr = sum(r['correlation'] for r in all_benchmark_results.values()) / len(all_benchmark_results)
            f.write("-" * 100 + "\n")
            f.write(f"{'AVERAGE':<35} {'':<15} {avg_r2:<15.16f} {avg_corr:<15.16f}\n")
        
        f.write("\n\n")
        f.write("DETAILED EQUATIONS:\n")
        f.write("-" * 100 + "\n")
        for bench_name, results in sorted_results:
            f.write(f"\n{bench_name}:\n")
            f.write(f"  {results['equation']}\n")

    # Create comparison plot: R² scores for all benchmarks
    if len(all_benchmark_results) > 0:
        fig = plt.figure(figsize=(14, 8))
        
        # Separate "ALL" from other benchmarks
        bench_items = [(name, results) for name, results in all_benchmark_results.items() if name != 'ALL']
        all_item = [(name, results) for name, results in all_benchmark_results.items() if name == 'ALL']
        
        # Sort other benchmarks by absolute R² score (highest to lowest)
        bench_items_sorted = sorted(bench_items, key=lambda x: abs(x[1]['r2_score']), reverse=True)
        
        # Combine: sorted benchmarks + ALL at the end
        combined_items = bench_items_sorted + all_item
        
        bench_names_short = [name.replace('BENCH-', '').replace('_', ' ') for name, _ in combined_items]
        r2_scores = [results['r2_score'] for _, results in combined_items]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(x=r2_scores, y=bench_names_short, color='steelblue', alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
        ax.set_yticklabels(bench_names_short, fontsize=9)
        ax.set_xlabel('R² Score')
        ax.set_ylabel('Benchmark')
        ax.set_title('Regression Quality: R² Scores (Free Regression) for Each Benchmark vs Complexity')
        ax.grid(axis='x', alpha=0.3)
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(ax.patches, r2_scores)):
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(width + 0.002, y, f'{width:.4f}', va='center', fontsize=8, ha='left')
        fig.tight_layout()
        fig.savefig(os.path.join(benchmarks_complexity_folder, 'r2_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Create comparison plot: R² scores for LINEAR regression
        # Re-sort by absolute r2_linear (highest to lowest), with ALL at bottom
        r2_linear_combined = [(name, results) for name, results in all_benchmark_results.items() if name != 'ALL']
        r2_linear_all = [(name, results) for name, results in all_benchmark_results.items() if name == 'ALL']
        r2_linear_sorted = sorted(r2_linear_combined, key=lambda x: abs(x[1]['r2_linear']), reverse=True)
        r2_linear_combined_items = r2_linear_sorted + r2_linear_all
        
        bench_names_short_r2lin = [name.replace('BENCH-', '').replace('_', ' ') for name, _ in r2_linear_combined_items]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        r2_linear_scores = [results['r2_linear'] for _, results in r2_linear_combined_items]
        sns.barplot(x=r2_linear_scores, y=bench_names_short_r2lin, color='cornflowerblue', alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
        ax.set_yticklabels(bench_names_short_r2lin, fontsize=9)
        ax.set_xlabel('R² Score')
        ax.set_ylabel('Benchmark')
        ax.set_title('Regression Quality: R² Scores (Linear Regression) for Each Benchmark vs Complexity')
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, score) in enumerate(zip(ax.patches, r2_linear_scores)):
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(width + 0.002, y, f'{width:.4f}', va='center', fontsize=8, ha='left')
        fig.tight_layout()
        fig.savefig(os.path.join(benchmarks_complexity_folder, 'r2_linear_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Create comparison plot: Correlation coefficients for all benchmarks
        # Re-sort by absolute correlation (highest to lowest), with ALL at bottom
        corr_combined = [(name, results) for name, results in all_benchmark_results.items() if name != 'ALL']
        corr_all = [(name, results) for name, results in all_benchmark_results.items() if name == 'ALL']
        corr_sorted = sorted(corr_combined, key=lambda x: abs(x[1]['correlation']), reverse=True)
        corr_combined_items = corr_sorted + corr_all
        
        bench_names_short_corr = [name.replace('BENCH-', '').replace('_', ' ') for name, _ in corr_combined_items]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        correlations = [results['correlation'] for _, results in corr_combined_items]
        colors = ['green' if c > 0 else 'red' for c in correlations]
        sns.barplot(x=correlations, y=bench_names_short_corr, palette=colors, alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
        ax.set_yticklabels(bench_names_short_corr, fontsize=9)
        ax.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
        ax.set_ylabel('Benchmark', fontsize=11)
        ax.set_title('Correlation: Benchmark Score vs Complexity', fontsize=12, pad=15)
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add correlation values as text outside bars
        for i, (bar, corr) in enumerate(zip(ax.patches, correlations)):
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            # Place text outside the bar end
            x_offset = 0.005 if width > 0 else -0.005
            x_pos = width + x_offset
            ha_align = 'left' if width > 0 else 'right'
            ax.text(x_pos, y, f'{width:.4f}', va='center', ha=ha_align, fontsize=9, fontweight='bold')
        
        fig.tight_layout()
        fig.subplots_adjust(right=0.95, left=0.1)
        fig.savefig(os.path.join(benchmarks_complexity_folder, 'correlation_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\nCompleted benchmark analysis. Processed {len(all_benchmark_results)} benchmarks.")

    # --------------------------------------------------------------
    # Plot graphs: FOR EACH BENCHMARK: FOR EACH TYPE: FOR EACH FILTER: - COMPLEXITY vs BENCHMARK

    print("\n" + "="*70)
    print("Starting detailed benchmark analysis by type and filter...")
    print("="*70)

    # Create main output folder
    detailed_benchmarks_folder = os.path.join(OUTPUT_FOLDER, '5_2_detailed_benchmarks_analysis')
    if not os.path.exists(detailed_benchmarks_folder):
        os.makedirs(detailed_benchmarks_folder)

    # Dictionary to store all detailed results
    all_detailed_results = []

    # Create types_str column if it doesn't exist
    if 'types_str' not in appended_benchmarks_df.columns:
        appended_benchmarks_df['types_str'] = appended_benchmarks_df['types'].apply(
            lambda x: '-'.join(sorted(x)) if isinstance(x, list) and len(x) > 0 else 'unknown'
        )

    # Get unique types and filters
    unique_types = appended_benchmarks_df['types_str'].unique()
    unique_filters = sorted([f for f in appended_benchmarks_df['filter'].unique() if pd.notna(f)])

    # Process each benchmark
    for bench_name in BENCH_ROWS_NAMES:
        print(f"\nProcessing benchmark: {bench_name}")
        
        # Create folder for this benchmark
        bench_clean_name = bench_name.replace('BENCH-', '').replace('_', '-').lower()
        bench_main_folder = os.path.join(detailed_benchmarks_folder, bench_clean_name)
        if not os.path.exists(bench_main_folder):
            os.makedirs(bench_main_folder)
        
        # Counter for combinations processed
        combinations_processed = 0
        
        # Process each type
        for type_str in unique_types:
            # Create folder for this type
            type_clean_name = type_str.replace('-', '_').replace(' ', '_').lower()
            if type_clean_name == 'unknown':
                continue  # Skip unknown types
            
            type_folder = os.path.join(bench_main_folder, type_clean_name)
            if not os.path.exists(type_folder):
                os.makedirs(type_folder)
            
            # Process each filter
            for filter_val in unique_filters:
                # Filter data for this specific combination
                mask_combo = (
                    (appended_benchmarks_df['types_str'] == type_str) &
                    (appended_benchmarks_df['filter'] == filter_val) &
                    ~(appended_benchmarks_df[bench_name].isna()) &
                    ~(appended_benchmarks_df['complexity'].isna())
                )
                
                x_data_combo = appended_benchmarks_df.loc[mask_combo, bench_name].values
                y_data_combo = appended_benchmarks_df.loc[mask_combo, 'complexity'].values
                
                # Skip if not enough data points
                if len(x_data_combo) < 3:
                    continue
                
                # Try different functions and find the best fit
                best_r2_combo = -np.inf
                best_name_combo = None
                best_params_combo = None
                best_func_combo = None
                best_equation_combo = None
                
                FUNCTIONS_TO_TEST_COMBO = {
                    'linear': {
                        'func': lambda x, a, b: a * x + b,
                        'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
                        'initial_guess': [1, 1]
                    },
                    'quadratic': {
                        'func': lambda x, a, b, c: a * x**2 + b * x + c,
                        'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
                        'initial_guess': [1, 1, 1]
                    },
                    'exponential': {
                        'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                        'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
                        'initial_guess': [1, 0.1, 1]
                    },
                    'logarithmic': {
                        'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
                        'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
                        'initial_guess': [1, 1, 1]
                    },
                    'power': {
                        'func': lambda x, a, b, c: a * (x + 1)**b + c,
                        'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
                        'initial_guess': [1, 0.5, 1]
                    }
                }
                
                for name, func_info in FUNCTIONS_TO_TEST_COMBO.items():
                    try:
                        params, _ = curve_fit(func_info['func'], x_data_combo, y_data_combo, 
                                                p0=func_info['initial_guess'], maxfev=10000)
                        y_pred = func_info['func'](x_data_combo, *params)
                        r2 = 1 - (np.sum((y_data_combo - y_pred)**2) / np.sum((y_data_combo - np.mean(y_data_combo))**2))
                        
                        if r2 > best_r2_combo:
                            best_r2_combo = r2
                            best_name_combo = name
                            best_params_combo = params
                            best_func_combo = func_info['func']
                            best_equation_combo = func_info['equation'](params)
                    except:
                        pass
                
                # Skip if no fit was successful
                if best_func_combo is None:
                    continue
                
                # Calculate Pearson correlation coefficient
                correlation_combo = np.corrcoef(x_data_combo, y_data_combo)[0, 1]
                
                # Create regression line with best fit
                x_line_combo = np.linspace(x_data_combo.min(), x_data_combo.max(), 100)
                y_line_combo = best_func_combo(x_line_combo, *best_params_combo)
                
                # Calculate linear regression for comparison
                linear_params_combo, _ = curve_fit(lambda x, a, b: a * x + b, x_data_combo, y_data_combo, 
                                                   p0=[1, 1], maxfev=10000)
                y_pred_linear_combo = linear_params_combo[0] * x_data_combo + linear_params_combo[1]
                r2_linear_combo = 1 - (np.sum((y_data_combo - y_pred_linear_combo)**2) / np.sum((y_data_combo - np.mean(y_data_combo))**2))
                y_line_linear_combo = linear_params_combo[0] * x_line_combo + linear_params_combo[1]
                linear_equation_combo = f'y = {linear_params_combo[0]:.16f}x + {linear_params_combo[1]:.16f}'
                
                # Create filename for this filter
                filter_clean_name = f"filter_{float(filter_val)}"
                
                # Plot scatter with both regression lines
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.scatter(x_data_combo, y_data_combo, alpha=0.6, label='Data')
                
                # Add model names below each data point
                for idx, (x, y) in enumerate(zip(x_data_combo, y_data_combo)):
                    model_name = appended_benchmarks_df.loc[mask_combo, 'model'].iloc[idx]
                    ax.text(x, y, f'  {model_name}', fontsize=6, alpha=0.5, color='gray', ha='center', va='top')
                
                ax.plot(x_line_combo, y_line_combo, 'r-', linewidth=2, label=f'Free regression ({best_name_combo})\n{best_equation_combo}\nR² = {best_r2_combo:.6f}')
                ax.plot(x_line_combo, y_line_linear_combo, 'b--', linewidth=2, label=f'Linear regression\nR² = {r2_linear_combo:.6f}\nN = {len(x_data_combo)}')
                ax.set_xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
                ax.set_ylabel('Complexity')
                ax.set_title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Complexity\nType: {type_str}, Filter: {float(filter_val)} sigma')
                ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5, fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(type_folder, f'{filter_clean_name}_regression.png'))
                plt.close()
                
                # Save statistics to text file
                with open(os.path.join(type_folder, f'{filter_clean_name}_stats.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"{bench_name} VS COMPLEXITY\n")
                    f.write(f"Type: {type_str}\n")
                    f.write(f"Filter: {float(filter_val)} sigma\n")
                    f.write("=" * 70 + "\n\n")
                    
                    f.write("REGRESSION:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Best fit function: {best_name_combo}\n")
                    f.write(f"Expression: {best_equation_combo}\n")
                    f.write(f"R² score: {best_r2_combo:.16f}\n")
                    f.write(f"Pearson correlation: {correlation_combo:.16f}\n")
                    f.write(f"Parameters: {best_params_combo}\n\n")
                    
                    f.write("DATA STATISTICS:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Number of data points: {len(x_data_combo)}\n")
                    f.write(f"Complexity mean: {y_data_combo.mean():.16f}\n")
                    f.write(f"Complexity std: {y_data_combo.std():.16f}\n")
                    f.write(f"Benchmark score mean: {x_data_combo.mean():.16f}\n")
                    f.write(f"Benchmark score std: {x_data_combo.std():.16f}\n")
                
                # Store results for summary
                all_detailed_results.append({
                    'benchmark': bench_name,
                    'type': type_str,
                    'filter': float(filter_val),
                    'best_function': best_name_combo,
                    'r2_score': best_r2_combo,
                    'r2_linear': r2_linear_combo,
                    'pearson_correlation': correlation_combo,
                    'abs_correlation': abs(correlation_combo),
                    'data_points': len(x_data_combo),
                    'equation': best_equation_combo,
                    'linear_equation': linear_equation_combo,
                    'complexity_mean': y_data_combo.mean(),
                    'complexity_std': y_data_combo.std(),
                    'benchmark_mean': x_data_combo.mean(),
                    'benchmark_std': x_data_combo.std()
                })
                
                combinations_processed += 1
        
        print(f"  Processed {combinations_processed} combinations for {bench_name}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_detailed_results)

    # Save complete results sorted by absolute correlation (descending)
    results_df_sorted = results_df.sort_values('abs_correlation', ascending=False)
    results_df_sorted.to_csv(os.path.join(detailed_benchmarks_folder, 'all_combinations_by_correlation.csv'), 
                            index=False, encoding='utf-8')

    # For each benchmark, create individual CSV files
    print("\n" + "="*70)
    print("Creating CSV files for each benchmark...")
    print("="*70)

    for bench_name in BENCH_ROWS_NAMES:
        bench_results = results_df[results_df['benchmark'] == bench_name]
        
        if len(bench_results) == 0:
            continue
        
        # Sort by absolute correlation (descending)
        bench_results_sorted = bench_results.sort_values('abs_correlation', ascending=False)
        
        # Get benchmark folder
        bench_clean_name = bench_name.replace('BENCH-', '').replace('_', '-').lower()
        bench_main_folder = os.path.join(detailed_benchmarks_folder, bench_clean_name)
        
        # Save all combinations for this benchmark (sorted by correlation)
        bench_results_sorted.to_csv(os.path.join(bench_main_folder, 'all_combinations.csv'), 
                                    index=False, encoding='utf-8')
        
        # Save top 20 combinations (or all if less than 20)
        top_n = min(20, len(bench_results_sorted))
        top_results = bench_results_sorted.head(top_n)
        top_results.to_csv(os.path.join(bench_main_folder, 'top_20_combinations.csv'), 
                        index=False, encoding='utf-8')
        
        # Create top 20 visualization for this benchmark
        if len(top_results) > 0:
            # Sort by absolute pearson correlation (ascending so highest abs appears on top in barh)
            top_results_sorted_viz = top_results.sort_values('abs_correlation', ascending=True)
            
            fig = plt.figure(figsize=(14, max(8, top_n * 0.4)))
            
            # Create labels
            labels = [f"{row['type'][:25]}, Filter={int(row['filter'])}" 
                    for _, row in top_results_sorted_viz.iterrows()]
            correlations = top_results_sorted_viz['pearson_correlation'].values
            
            colors = ['green' if c > 0 else 'red' for c in correlations]
            
            plt.barh(range(len(labels)), correlations, color=colors, alpha=0.7, edgecolor='black')
            plt.yticks(range(len(labels)), labels, fontsize=9)
            plt.xlabel('Pearson Correlation Coefficient')
            plt.ylabel('Type - Filter Combination')
            plt.title(f'Top {top_n} Combinations by Absolute Pearson Correlation\n{bench_name.replace("BENCH-", "").replace("_", " ")} vs Complexity')
            plt.grid(axis='x', alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            
            # Add value labels on bars
            for i, corr in enumerate(correlations):
                x_offset = 0.003 if corr > 0 else -0.003
                ha_align = 'left' if corr > 0 else 'right'
                plt.text(corr + x_offset, i, f'{corr:.4f}', va='center', fontsize=8, ha=ha_align)
            
            plt.tight_layout()
            plt.savefig(os.path.join(bench_main_folder, 'top_20_correlations.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Created CSV files and plot for {bench_name} ({len(bench_results_sorted)} combinations, top {top_n} saved)")

    # Create a pivot summary showing best correlation for each benchmark-type-filter combination
    pivot_summary = results_df.pivot_table(
        values='abs_correlation',
        index=['benchmark', 'type'],
        columns='filter',
        aggfunc='max'
    )
    pivot_summary.to_csv(os.path.join(detailed_benchmarks_folder, 'correlation_matrix.csv'), encoding='utf-8')

    # Create summary text file with overall statistics
    with open(os.path.join(detailed_benchmarks_folder, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("DETAILED BENCHMARK ANALYSIS - SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Total combinations analyzed: {len(results_df)}\n")
        f.write(f"Benchmarks analyzed: {results_df['benchmark'].nunique()}\n")
        f.write(f"Types analyzed: {results_df['type'].nunique()}\n")
        f.write(f"Filters analyzed: {sorted(results_df['filter'].unique())}\n\n")
        
        f.write("TOP 20 COMBINATIONS BY ABSOLUTE PEARSON CORRELATION:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Rank':<6} {'Benchmark':<30} {'Type':<20} {'Filter':<8} {'Correlation':<15} {'R²':<15} {'N':<8}\n")
        f.write("-" * 100 + "\n")
        
        top_20 = results_df_sorted.head(20)
        for idx, (_, row) in enumerate(top_20.iterrows(), 1):
            bench_short = row['benchmark'].replace('BENCH-', '')
            f.write(f"{idx:<6} {bench_short:<30} {row['type']:<20} {row['filter']:<8} "
                    f"{row['pearson_correlation']:<15.8f} {row['r2_score']:<15.8f} {row['data_points']:<8}\n")
        
        f.write("\n\n")
        f.write("BEST COMBINATION FOR EACH BENCHMARK:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Benchmark':<30} {'Type':<20} {'Filter':<8} {'Correlation':<15} {'R²':<15} {'Function':<15}\n")
        f.write("-" * 100 + "\n")
        
        for bench_name in BENCH_ROWS_NAMES:
            bench_best = results_df[results_df['benchmark'] == bench_name].sort_values('abs_correlation', ascending=False)
            if len(bench_best) > 0:
                best = bench_best.iloc[0]
                bench_short = best['benchmark'].replace('BENCH-', '')
                f.write(f"{bench_short:<30} {best['type']:<20} {best['filter']:<8} "
                        f"{best['pearson_correlation']:<15.8f} {best['r2_score']:<15.8f} {best['best_function']:<15}\n")
        
        f.write("\n\n")
        f.write("STATISTICS BY BENCHMARK:\n")
        f.write("-" * 100 + "\n")
        
        for bench_name in BENCH_ROWS_NAMES:
            bench_data = results_df[results_df['benchmark'] == bench_name]
            if len(bench_data) > 0:
                f.write(f"\n{bench_name.replace('BENCH-', '')}:\n")
                f.write(f"  Combinations analyzed: {len(bench_data)}\n")
                f.write(f"  Mean correlation: {bench_data['abs_correlation'].mean():.8f}\n")
                f.write(f"  Max correlation: {bench_data['abs_correlation'].max():.8f}\n")
                f.write(f"  Min correlation: {bench_data['abs_correlation'].min():.8f}\n")
                f.write(f"  Mean R²: {bench_data['r2_score'].mean():.8f}\n")

    # Create visualization: Top 20 correlations
    if len(results_df) > 0:
        fig = plt.figure(figsize=(16, 10))
        top_20_plot = results_df_sorted.head(20).copy()
        
        # Sort by absolute value (descending so highest appears on top)
        # For barh, we need to sort in reverse order so highest index has highest value on top
        top_20_plot_sorted = top_20_plot.sort_values('abs_correlation', ascending=True)
        
        # Create labels
        labels = [f"{row['benchmark'].replace('BENCH-', '')[:20]}\n{row['type'][:15]}, F={float(row['filter'])}" 
                for _, row in top_20_plot_sorted.iterrows()]
        correlations = top_20_plot_sorted['pearson_correlation'].values
        
        colors = ['green' if c > 0 else 'red' for c in correlations]
        
        plt.barh(range(len(labels)), correlations, color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(range(len(labels)), labels, fontsize=8)
        plt.xlabel('Pearson Correlation Coefficient')
        plt.ylabel('Benchmark - Type - Filter Combination')
        plt.title('Top 20 Combinations by Absolute Pearson Correlation\n(Benchmark vs Complexity)')
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars
        for i, corr in enumerate(correlations):
            x_offset = 0.003 if corr > 0 else -0.003
            ha_align = 'left' if corr > 0 else 'right'
            plt.text(corr + x_offset, i, f'{corr:.4f}', va='center', fontsize=7, ha=ha_align)
        
        plt.tight_layout()
        plt.savefig(os.path.join(detailed_benchmarks_folder, 'top_20_correlations.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "="*70)
    print(f"Detailed analysis complete!")
    print(f"Total combinations analyzed: {len(results_df)}")
    print(f"Results saved to: {detailed_benchmarks_folder}")
    print("="*70) 

# ================================================================================
   
# --------------------------------------------------------------
# 6 FOR EACH BENCHMARK - PARAM COUNT vs BENCHMARK

def analyze_param_count_vs_benchmarks(appended_benchmarks_df, OUTPUT_FOLDER):
    # Get rows in header that start with "BENCH"
    BENCH_ROWS_NAMES = appended_benchmarks_df.columns[appended_benchmarks_df.columns.str.startswith('BENCH')].tolist()


    print("\n" + "="*70)
    print("Starting PARAM COUNT vs BENCHMARK analysis...")
    print("="*70)

    # Create main output folder
    param_benchmarks_folder = os.path.join(OUTPUT_FOLDER, '6_param_count_vs_benchmarks')
    if not os.path.exists(param_benchmarks_folder):
        os.makedirs(param_benchmarks_folder)

    # Dictionary to store all results
    all_param_benchmark_results = {}

    # Process each benchmark column
    for bench_name in BENCH_ROWS_NAMES:
        print(f"Processing benchmark: {bench_name}")
        
        # Remove NaN values for regression
        mask_bench = ~(appended_benchmarks_df[bench_name].isna() | appended_benchmarks_df['count'].isna())
        x_data_bench = appended_benchmarks_df.loc[mask_bench, bench_name].values
        y_data_bench = appended_benchmarks_df.loc[mask_bench, 'count'].values
        
        # Skip if not enough data points
        if len(x_data_bench) < 3:
            print(f"  Skipping {bench_name}: insufficient data points ({len(x_data_bench)})")
            continue
        
        # Try different functions and find the best fit
        best_r2_bench = -np.inf
        best_name_bench = None
        best_params_bench = None
        best_func_bench = None
        best_equation_bench = None
        
        FUNCTIONS_TO_TEST_BENCH = {
            'linear': {
                'func': lambda x, a, b: a * x + b,
                'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
                'initial_guess': [1, 1]
            },
            'quadratic': {
                'func': lambda x, a, b, c: a * x**2 + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'exponential': {
                'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
                'initial_guess': [1, 0.1, 1]
            },
            'logarithmic': {
                'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'power': {
                'func': lambda x, a, b, c: a * (x + 1)**b + c,
                'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
                'initial_guess': [1, 0.5, 1]
            }
        }
        
        for name, func_info in FUNCTIONS_TO_TEST_BENCH.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params, _ = curve_fit(func_info['func'], x_data_bench, y_data_bench, 
                                        p0=func_info['initial_guess'], maxfev=10000)
                y_pred = func_info['func'](x_data_bench, *params)
                r2 = 1 - (np.sum((y_data_bench - y_pred)**2) / np.sum((y_data_bench - np.mean(y_data_bench))**2))
                
                if r2 > best_r2_bench:
                    best_r2_bench = r2
                    best_name_bench = name
                    best_params_bench = params
                    best_func_bench = func_info['func']
                    best_equation_bench = func_info['equation'](params)
            except:
                # Skip if curve fitting fails for this function
                pass
        
        # If no fit was successful, skip this benchmark
        if best_func_bench is None:
            print(f"  Skipping {bench_name}: curve fitting failed for all functions")
            continue
        
        # Create regression line with best fit
        x_line_bench = np.linspace(x_data_bench.min(), x_data_bench.max(), 100)
        y_line_bench = best_func_bench(x_line_bench, *best_params_bench)
        
        # Calculate linear regression for comparison
        linear_params, _ = curve_fit(lambda x, a, b: a * x + b, x_data_bench, y_data_bench, 
                                     p0=[1, 1], maxfev=10000)
        y_pred_linear = linear_params[0] * x_data_bench + linear_params[1]
        r2_linear = 1 - (np.sum((y_data_bench - y_pred_linear)**2) / np.sum((y_data_bench - np.mean(y_data_bench))**2))
        y_line_linear = linear_params[0] * x_line_bench + linear_params[1]
        linear_equation = f'y = {linear_params[0]:.16f}x + {linear_params[1]:.16f}'
        
        # Create subfolder for this benchmark
        bench_clean_name = bench_name.replace('BENCH-', '').replace('_', '-').lower()
        bench_folder = os.path.join(param_benchmarks_folder, bench_clean_name)
        if not os.path.exists(bench_folder):
            os.makedirs(bench_folder)
        
        # Plot scatter with both regression lines
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(x_data_bench, y_data_bench, alpha=0.6, label='Data')
        
        # Add model names below each data point
        for idx, (x, y) in enumerate(zip(x_data_bench, y_data_bench)):
            model_name = appended_benchmarks_df.loc[mask_bench, 'model'].iloc[idx]
            ax.text(x, y, f'  {model_name}', fontsize=6, alpha=0.5, color='gray', ha='center', va='top')
        
        ax.plot(x_line_bench, y_line_bench, 'r-', linewidth=2, label=f'Free regression ({best_name_bench})\n{best_equation_bench}\nR² = {best_r2_bench:.6f}')
        ax.plot(x_line_bench, y_line_linear, 'b--', linewidth=2, label=f'Linear regression\n{linear_equation}\nR² = {r2_linear:.6f}')
        ax.set_xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
        ax.set_ylabel('Parameter Count')
        ax.set_title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Parameter Count')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5, fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(bench_folder, 'regression.png'))
        plt.close()
        
        # Calculate Pearson correlation coefficient
        correlation = np.corrcoef(x_data_bench, y_data_bench)[0, 1]
        
        # Save regression information to text file
        with open(os.path.join(bench_folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{bench_name} VS PARAMETER COUNT - REGRESSION ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("REGRESSION:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Best fit function: {best_name_bench}\n")
            f.write(f"Expression: {best_equation_bench}\n")
            f.write(f"R² score: {best_r2_bench:.16f}\n")
            f.write(f"Pearson correlation: {correlation:.16f}\n")
            f.write(f"Parameters: {best_params_bench}\n")
        
        # Save statistics to text file
        with open(os.path.join(bench_folder, 'statistics.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{bench_name} VS PARAMETER COUNT - STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("PARAMETER COUNT STATISTICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Number of data points: {len(y_data_bench)}\n")
            f.write(f"Mean parameter count: {y_data_bench.mean():.16f}\n")
            f.write(f"Median parameter count: {np.median(y_data_bench):.16f}\n")
            f.write(f"Std parameter count: {y_data_bench.std():.16f}\n")
            f.write(f"Min parameter count: {y_data_bench.min():.16f}\n")
            f.write(f"Max parameter count: {y_data_bench.max():.16f}\n\n")
            
            f.write(f"{bench_name} STATISTICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Min score: {x_data_bench.min():.16f}\n")
            f.write(f"Max score: {x_data_bench.max():.16f}\n")
            f.write(f"Mean score: {x_data_bench.mean():.16f}\n")
            f.write(f"Median score: {np.median(x_data_bench):.16f}\n")
            f.write(f"Std score: {x_data_bench.std():.16f}\n")
        
        # Store results for summary
        all_param_benchmark_results[bench_name] = {
            'best_function': best_name_bench,
            'r2_score': best_r2_bench,
            'r2_linear': r2_linear,
            'correlation': correlation,
            'equation': best_equation_bench,
            'linear_equation': linear_equation,
            'data_points': len(x_data_bench)
        }
        
        print(f"  Completed {bench_name}: R² = {best_r2_bench:.6f}, Correlation = {correlation:.6f}")

    # ==============================================================================
    # PROCESS "ALL" - Concatenate all benchmark data for param count with Min-Max normalization
    # ==============================================================================
    print("\nProcessing 'all' (concatenated benchmark data with Min-Max normalization)...")
    
    # Collect all data points from all benchmarks with Min-Max normalization per benchmark
    all_x_data_param = []
    all_y_data_param = []
    all_model_names_param = []
    
    for bench_name in BENCH_ROWS_NAMES:
        mask_bench = ~(appended_benchmarks_df[bench_name].isna() | appended_benchmarks_df['count'].isna())
        x_data_bench = appended_benchmarks_df.loc[mask_bench, bench_name].values
        y_data_bench = appended_benchmarks_df.loc[mask_bench, 'count'].values
        model_names_bench = appended_benchmarks_df.loc[mask_bench, 'model'].values
        
        if len(x_data_bench) >= 3:
            # Apply Min-Max normalization to scale to [0, 1]
            # Best model (max score) = 1.0, Worst model (min score) = 0.0
            x_min = x_data_bench.min()
            x_max = x_data_bench.max()
            if x_max > x_min:  # Avoid division by zero
                x_normalized = (x_data_bench - x_min) / (x_max - x_min)
            else:
                x_normalized = np.ones_like(x_data_bench) * 0.5  # All same value
            
            all_x_data_param.extend(x_normalized)
            all_y_data_param.extend(y_data_bench)
            all_model_names_param.extend(model_names_bench)
    
    # Convert to numpy arrays
    all_x_data_param = np.array(all_x_data_param)
    all_y_data_param = np.array(all_y_data_param)
    all_model_names_param = np.array(all_model_names_param)
    
    if len(all_x_data_param) >= 3:
        # Try different functions and find the best fit
        best_r2_all_param = -np.inf
        best_name_all_param = None
        best_params_all_param = None
        best_func_all_param = None
        best_equation_all_param = None
        
        FUNCTIONS_TO_TEST_ALL_PARAM = {
            'linear': {
                'func': lambda x, a, b: a * x + b,
                'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
                'initial_guess': [1, 1]
            },
            'quadratic': {
                'func': lambda x, a, b, c: a * x**2 + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'exponential': {
                'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
                'initial_guess': [1, 0.1, 1]
            },
            'logarithmic': {
                'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'power': {
                'func': lambda x, a, b, c: a * (x + 1)**b + c,
                'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
                'initial_guess': [1, 0.5, 1]
            }
        }
        
        for name, func_info in FUNCTIONS_TO_TEST_ALL_PARAM.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params, _ = curve_fit(func_info['func'], all_x_data_param, all_y_data_param, 
                                        p0=func_info['initial_guess'], maxfev=10000)
                y_pred = func_info['func'](all_x_data_param, *params)
                r2 = 1 - (np.sum((all_y_data_param - y_pred)**2) / np.sum((all_y_data_param - np.mean(all_y_data_param))**2))
                
                if r2 > best_r2_all_param:
                    best_r2_all_param = r2
                    best_name_all_param = name
                    best_params_all_param = params
                    best_func_all_param = func_info['func']
                    best_equation_all_param = func_info['equation'](params)
            except:
                pass
        
        if best_func_all_param is not None:
            # Create regression line with best fit
            x_line_all_param = np.linspace(all_x_data_param.min(), all_x_data_param.max(), 100)
            y_line_all_param = best_func_all_param(x_line_all_param, *best_params_all_param)
            
            # Calculate linear regression for comparison
            # Use better initial guess based on data scale
            y_range = all_y_data_param.max() - all_y_data_param.min()
            x_range = all_x_data_param.max() - all_x_data_param.min() if all_x_data_param.max() != all_x_data_param.min() else 1
            initial_slope = y_range / x_range
            initial_intercept = all_y_data_param.mean()
            
            linear_params_all, _ = curve_fit(lambda x, a, b: a * x + b, all_x_data_param, all_y_data_param, 
                                        p0=[initial_slope, initial_intercept], maxfev=10000)
            y_pred_linear_param = linear_params_all[0] * all_x_data_param + linear_params_all[1]
            r2_linear_all_param = 1 - (np.sum((all_y_data_param - y_pred_linear_param)**2) / np.sum((all_y_data_param - np.mean(all_y_data_param))**2))
            y_line_linear_param = linear_params_all[0] * x_line_all_param + linear_params_all[1]
            linear_equation_all_param = f'y = {linear_params_all[0]:.16f}x + {linear_params_all[1]:.16f}'
            
            # Create subfolder for "all"
            all_folder_param = os.path.join(param_benchmarks_folder, 'all')
            if not os.path.exists(all_folder_param):
                os.makedirs(all_folder_param)
            
            # Plot scatter with both regression lines
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.scatter(all_x_data_param, all_y_data_param, alpha=0.6, label='Data')
            
            # Add model names below each data point
            for idx, (x, y) in enumerate(zip(all_x_data_param, all_y_data_param)):
                model_name = all_model_names_param[idx]
                ax.text(x, y, f'  {model_name}', fontsize=6, alpha=0.5, color='gray', ha='center', va='top')
            
            ax.plot(x_line_all_param, y_line_all_param, 'r-', linewidth=2, label=f'Free regression ({best_name_all_param})\n{best_equation_all_param}\nR² = {best_r2_all_param:.6f}')
            ax.plot(x_line_all_param, y_line_linear_param, 'b--', linewidth=2, label=f'Linear regression\n{linear_equation_all_param}\nR² = {r2_linear_all_param:.6f}')
            ax.set_xlabel('Normalized Benchmark Score (All Benchmarks, Min-Max [0,1])')
            ax.set_ylabel('Parameter Count')
            ax.set_title('All Benchmarks vs Parameter Count (Normalized)')
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5, fontsize=8)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(all_folder_param, 'regression.png'))
            plt.close()
            
            # Calculate Pearson correlation coefficient
            correlation_all_param = np.corrcoef(all_x_data_param, all_y_data_param)[0, 1]
            
            # Save regression information to text file
            with open(os.path.join(all_folder_param, 'regression_info.txt'), 'w', encoding='utf-8') as f:
                f.write("ALL BENCHMARKS VS PARAMETER COUNT - REGRESSION ANALYSIS\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("REGRESSION:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Best fit function: {best_name_all_param}\n")
                f.write(f"Expression: {best_equation_all_param}\n")
                f.write(f"R² score: {best_r2_all_param:.16f}\n")
                f.write(f"Pearson correlation: {correlation_all_param:.16f}\n")
                f.write(f"Parameters: {best_params_all_param}\n")
            
            # Save statistics to text file
            with open(os.path.join(all_folder_param, 'statistics.txt'), 'w', encoding='utf-8') as f:
                f.write("ALL BENCHMARKS VS PARAMETER COUNT - STATISTICS\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("PARAMETER COUNT STATISTICS:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Number of data points: {len(all_y_data_param)}\n")
                f.write(f"Mean parameter count: {all_y_data_param.mean():.16f}\n")
                f.write(f"Median parameter count: {np.median(all_y_data_param):.16f}\n")
                f.write(f"Std parameter count: {all_y_data_param.std():.16f}\n")
                f.write(f"Min parameter count: {all_y_data_param.min():.16f}\n")
                f.write(f"Max parameter count: {all_y_data_param.max():.16f}\n\n")
                
                f.write("BENCHMARK SCORE STATISTICS:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Min score: {all_x_data_param.min():.16f}\n")
                f.write(f"Max score: {all_x_data_param.max():.16f}\n")
                f.write(f"Mean score: {all_x_data_param.mean():.16f}\n")
                f.write(f"Median score: {np.median(all_x_data_param):.16f}\n")
                f.write(f"Std score: {all_x_data_param.std():.16f}\n")
            
            # Store results for summary
            all_param_benchmark_results['ALL'] = {
                'best_function': best_name_all_param,
                'r2_score': best_r2_all_param,
                'r2_linear': r2_linear_all_param,
                'correlation': correlation_all_param,
                'equation': best_equation_all_param,
                'linear_equation': linear_equation_all_param,
                'data_points': len(all_x_data_param)
            }
            
            print(f"  Completed 'all': R² = {best_r2_all_param:.6f}, Correlation = {correlation_all_param:.6f}")
    
    # ==============================================================================
    # END OF "ALL" PROCESSING
    # ==============================================================================

    # Create summary file with all benchmark results
    with open(os.path.join(param_benchmarks_folder, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("BENCHMARKS VS PARAMETER COUNT - SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Benchmark':<35} {'Best Fit':<15} {'R² Score':<15} {'Correlation':<15} {'Data Points':<15}\n")
        f.write("-" * 100 + "\n")
        
        # Sort by R² score (descending), but always put "ALL" last
        sorted_results = sorted(all_param_benchmark_results.items(), key=lambda x: x[1]['r2_score'], reverse=True)
        # Move "ALL" to the end if it exists
        sorted_results = [item for item in sorted_results if item[0] != 'ALL'] + \
                        [item for item in sorted_results if item[0] == 'ALL']
        
        for bench_name, results in sorted_results:
            f.write(f"{bench_name:<35} {results['best_function']:<15} {results['r2_score']:<15.16f} "
                    f"{results['correlation']:<15.16f} {results['data_points']:<15}\n")
        
        # Calculate and add averages
        if len(all_param_benchmark_results) > 0:
            avg_r2 = sum(r['r2_score'] for r in all_param_benchmark_results.values()) / len(all_param_benchmark_results)
            avg_corr = sum(r['correlation'] for r in all_param_benchmark_results.values()) / len(all_param_benchmark_results)
            f.write("-" * 100 + "\n")
            f.write(f"{'AVERAGE':<35} {'':<15} {avg_r2:<15.16f} {avg_corr:<15.16f}\n")
        
        f.write("\n\n")
        f.write("DETAILED EQUATIONS:\n")
        f.write("-" * 100 + "\n")
        for bench_name, results in sorted_results:
            f.write(f"\n{bench_name}:\n")
            f.write(f"  {results['equation']}\n")

    # Create comparison plot: R² scores for all benchmarks
    if len(all_param_benchmark_results) > 0:
        fig = plt.figure(figsize=(14, 8))
        
        # Separate "ALL" from other benchmarks
        bench_items_param = [(name, results) for name, results in all_param_benchmark_results.items() if name != 'ALL']
        all_item_param = [(name, results) for name, results in all_param_benchmark_results.items() if name == 'ALL']
        
        # Sort other benchmarks by absolute R² score (highest to lowest)
        bench_items_param_sorted = sorted(bench_items_param, key=lambda x: abs(x[1]['r2_score']), reverse=True)
        
        # Combine: sorted benchmarks + ALL at the end
        combined_items_param = bench_items_param_sorted + all_item_param
        
        bench_names_short = [name.replace('BENCH-', '').replace('_', ' ') for name, _ in combined_items_param]
        r2_scores = [results['r2_score'] for _, results in combined_items_param]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(x=r2_scores, y=bench_names_short, color='steelblue', alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
        ax.set_yticklabels(bench_names_short, fontsize=9)
        ax.set_xlabel('R² Score')
        ax.set_ylabel('Benchmark')
        ax.set_title('Regression Quality: R² Scores (Free Regression) for Each Benchmark vs Parameter Count')
        ax.grid(axis='x', alpha=0.3)
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(ax.patches, r2_scores)):
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(width + 0.002, y, f'{width:.4f}', va='center', fontsize=8, ha='left')
        fig.tight_layout()
        fig.savefig(os.path.join(param_benchmarks_folder, 'r2_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Create comparison plot: R² scores for LINEAR regression
        # Re-sort by absolute r2_linear (highest to lowest), with ALL at bottom
        r2_linear_param_combined = [(name, results) for name, results in all_param_benchmark_results.items() if name != 'ALL']
        r2_linear_param_all = [(name, results) for name, results in all_param_benchmark_results.items() if name == 'ALL']
        r2_linear_param_sorted = sorted(r2_linear_param_combined, key=lambda x: abs(x[1]['r2_linear']), reverse=True)
        r2_linear_param_combined_items = r2_linear_param_sorted + r2_linear_param_all
        
        bench_names_short_r2lin_param = [name.replace('BENCH-', '').replace('_', ' ') for name, _ in r2_linear_param_combined_items]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        r2_linear_scores = [results['r2_linear'] for _, results in r2_linear_param_combined_items]
        sns.barplot(x=r2_linear_scores, y=bench_names_short_r2lin_param, color='cornflowerblue', alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
        ax.set_yticklabels(bench_names_short_r2lin_param, fontsize=9)
        ax.set_xlabel('R² Score')
        ax.set_ylabel('Benchmark')
        ax.set_title('Regression Quality: R² Scores (Linear Regression) for Each Benchmark vs Parameter Count')
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, score) in enumerate(zip(ax.patches, r2_linear_scores)):
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(width + 0.002, y, f'{width:.4f}', va='center', fontsize=8, ha='left')
        fig.tight_layout()
        fig.savefig(os.path.join(param_benchmarks_folder, 'r2_linear_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Create comparison plot: Correlation coefficients for all benchmarks
        # Re-sort by absolute correlation (highest to lowest), with ALL at bottom
        corr_param_combined = [(name, results) for name, results in all_param_benchmark_results.items() if name != 'ALL']
        corr_param_all = [(name, results) for name, results in all_param_benchmark_results.items() if name == 'ALL']
        corr_param_sorted = sorted(corr_param_combined, key=lambda x: abs(x[1]['correlation']), reverse=True)
        corr_param_combined_items = corr_param_sorted + corr_param_all
        
        bench_names_short_corr_param = [name.replace('BENCH-', '').replace('_', ' ') for name, _ in corr_param_combined_items]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        correlations = [results['correlation'] for _, results in corr_param_combined_items]
        colors = ['green' if c > 0 else 'red' for c in correlations]
        sns.barplot(x=correlations, y=bench_names_short_corr_param, palette=colors, alpha=0.7, edgecolor='black', ax=ax, errorbar=None)
        ax.set_yticklabels(bench_names_short_corr_param, fontsize=9)
        ax.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
        ax.set_ylabel('Benchmark', fontsize=11)
        ax.set_title('Correlation: Benchmark Score vs Parameter Count', fontsize=12, pad=15)
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add correlation values as text outside bars
        for i, (bar, corr) in enumerate(zip(ax.patches, correlations)):
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            # Place text outside the bar end
            x_offset = 0.005 if width > 0 else -0.005
            x_pos = width + x_offset
            ha_align = 'left' if width > 0 else 'right'
            ax.text(x_pos, y, f'{width:.4f}', va='center', ha=ha_align, fontsize=9, fontweight='bold')
        
        fig.tight_layout()
        fig.subplots_adjust(right=0.95, left=0.1)
        fig.savefig(os.path.join(param_benchmarks_folder, 'correlation_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\nCompleted parameter count analysis. Processed {len(all_param_benchmark_results)} benchmarks.")

    # --------------------------------------------------------------
    # Plot graphs: FOR EACH BENCHMARK: FOR EACH TYPE: FOR EACH FILTER: - PARAM COUNT vs BENCHMARK

    print("\n" + "="*70)
    print("Starting detailed PARAM COUNT analysis by type and filter...")
    print("="*70)

    # Create main output folder
    detailed_param_benchmarks_folder = os.path.join(OUTPUT_FOLDER, '6_2_detailed_param_count_analysis')
    if not os.path.exists(detailed_param_benchmarks_folder):
        os.makedirs(detailed_param_benchmarks_folder)

    # Dictionary to store all detailed results
    all_detailed_param_results = []

    # Create types_str column if it doesn't exist
    if 'types_str' not in appended_benchmarks_df.columns:
        appended_benchmarks_df['types_str'] = appended_benchmarks_df['types'].apply(
            lambda x: '-'.join(sorted(x)) if isinstance(x, list) and len(x) > 0 else 'unknown'
        )

    # Get unique types and filters (local to this function)
    unique_types = appended_benchmarks_df['types_str'].unique()
    unique_filters = sorted([f for f in appended_benchmarks_df['filter'].unique() if pd.notna(f)])

    # Process each benchmark
    for bench_name in BENCH_ROWS_NAMES:
        print(f"\nProcessing benchmark: {bench_name}")
        
        # Create folder for this benchmark
        bench_clean_name = bench_name.replace('BENCH-', '').replace('_', '-').lower()
        bench_main_folder = os.path.join(detailed_param_benchmarks_folder, bench_clean_name)
        if not os.path.exists(bench_main_folder):
            os.makedirs(bench_main_folder)
        
        # Counter for combinations processed
        combinations_processed = 0
        
        # Process each type
        for type_str in unique_types:
            # Create folder for this type
            type_clean_name = type_str.replace('-', '_').replace(' ', '_').lower()
            if type_clean_name == 'unknown':
                continue  # Skip unknown types
            
            type_folder = os.path.join(bench_main_folder, type_clean_name)
            if not os.path.exists(type_folder):
                os.makedirs(type_folder)
            
            # Process each filter
            for filter_val in unique_filters:
                # Filter data for this specific combination
                mask_combo = (
                    (appended_benchmarks_df['types_str'] == type_str) &
                    (appended_benchmarks_df['filter'] == filter_val) &
                    ~(appended_benchmarks_df[bench_name].isna()) &
                    ~(appended_benchmarks_df['count'].isna())
                )
                
                x_data_combo = appended_benchmarks_df.loc[mask_combo, bench_name].values
                y_data_combo = appended_benchmarks_df.loc[mask_combo, 'count'].values
                
                # Skip if not enough data points
                if len(x_data_combo) < 3:
                    continue
                
                # Try different functions and find the best fit
                best_r2_combo = -np.inf
                best_name_combo = None
                best_params_combo = None
                best_func_combo = None
                best_equation_combo = None
                
                FUNCTIONS_TO_TEST_COMBO = {
                    'linear': {
                        'func': lambda x, a, b: a * x + b,
                        'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
                        'initial_guess': [1, 1]
                    },
                    'quadratic': {
                        'func': lambda x, a, b, c: a * x**2 + b * x + c,
                        'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
                        'initial_guess': [1, 1, 1]
                    },
                    'exponential': {
                        'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                        'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
                        'initial_guess': [1, 0.1, 1]
                    },
                    'logarithmic': {
                        'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
                        'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
                        'initial_guess': [1, 1, 1]
                    },
                    'power': {
                        'func': lambda x, a, b, c: a * (x + 1)**b + c,
                        'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
                        'initial_guess': [1, 0.5, 1]
                    }
                }
                
                for name, func_info in FUNCTIONS_TO_TEST_COMBO.items():
                    try:
                        params, _ = curve_fit(func_info['func'], x_data_combo, y_data_combo, 
                                                p0=func_info['initial_guess'], maxfev=10000)
                        y_pred = func_info['func'](x_data_combo, *params)
                        r2 = 1 - (np.sum((y_data_combo - y_pred)**2) / np.sum((y_data_combo - np.mean(y_data_combo))**2))
                        
                        if r2 > best_r2_combo:
                            best_r2_combo = r2
                            best_name_combo = name
                            best_params_combo = params
                            best_func_combo = func_info['func']
                            best_equation_combo = func_info['equation'](params)
                    except:
                        pass
                
                # Skip if no fit was successful
                if best_func_combo is None:
                    continue
                
                # Calculate Pearson correlation coefficient
                correlation_combo = np.corrcoef(x_data_combo, y_data_combo)[0, 1]
                
                # Create regression line with best fit
                x_line_combo = np.linspace(x_data_combo.min(), x_data_combo.max(), 100)
                y_line_combo = best_func_combo(x_line_combo, *best_params_combo)
                
                # Calculate linear regression for comparison
                linear_params_combo, _ = curve_fit(lambda x, a, b: a * x + b, x_data_combo, y_data_combo, 
                                                   p0=[1, 1], maxfev=10000)
                y_pred_linear_combo = linear_params_combo[0] * x_data_combo + linear_params_combo[1]
                r2_linear_combo = 1 - (np.sum((y_data_combo - y_pred_linear_combo)**2) / np.sum((y_data_combo - np.mean(y_data_combo))**2))
                y_line_linear_combo = linear_params_combo[0] * x_line_combo + linear_params_combo[1]
                linear_equation_combo = f'y = {linear_params_combo[0]:.16f}x + {linear_params_combo[1]:.16f}'
                
                # Create filename for this filter
                filter_clean_name = f"filter_{float(filter_val)}"
                
                # Plot scatter with both regression lines
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.scatter(x_data_combo, y_data_combo, alpha=0.6, label='Data')
                
                # Add model names below each data point
                for idx, (x, y) in enumerate(zip(x_data_combo, y_data_combo)):
                    model_name = appended_benchmarks_df.loc[mask_combo, 'model'].iloc[idx]
                    ax.text(x, y, f'  {model_name}', fontsize=6, alpha=0.5, color='gray', ha='center', va='top')
                
                ax.plot(x_line_combo, y_line_combo, 'r-', linewidth=2, label=f'Free regression ({best_name_combo})\n{best_equation_combo}\nR² = {best_r2_combo:.6f}')
                ax.plot(x_line_combo, y_line_linear_combo, 'b--', linewidth=2, label=f'Linear regression\nR² = {r2_linear_combo:.6f}\nN = {len(x_data_combo)}')
                ax.set_xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
                ax.set_ylabel('Parameter Count')
                ax.set_title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Parameter Count\nType: {type_str}, Filter: {float(filter_val)} sigma')
                ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5, fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(type_folder, f'{filter_clean_name}_regression.png'))
                plt.close()
                
                # Save statistics to text file
                with open(os.path.join(type_folder, f'{filter_clean_name}_stats.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"{bench_name} VS PARAMETER COUNT\n")
                    f.write(f"Type: {type_str}\n")
                    f.write(f"Filter: {float(filter_val)} sigma\n")
                    f.write("=" * 70 + "\n\n")
                    
                    f.write("REGRESSION:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Best fit function: {best_name_combo}\n")
                    f.write(f"Expression: {best_equation_combo}\n")
                    f.write(f"R² score: {best_r2_combo:.16f}\n")
                    f.write(f"Pearson correlation: {correlation_combo:.16f}\n")
                    f.write(f"Parameters: {best_params_combo}\n\n")
                    
                    f.write("DATA STATISTICS:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Number of data points: {len(x_data_combo)}\n")
                    f.write(f"Parameter count mean: {y_data_combo.mean():.16f}\n")
                    f.write(f"Parameter count std: {y_data_combo.std():.16f}\n")
                    f.write(f"Benchmark score mean: {x_data_combo.mean():.16f}\n")
                    f.write(f"Benchmark score std: {x_data_combo.std():.16f}\n")
                
                # Store results for summary
                all_detailed_param_results.append({
                    'benchmark': bench_name,
                    'type': type_str,
                    'filter': float(filter_val),
                    'best_function': best_name_combo,
                    'r2_score': best_r2_combo,
                    'r2_linear': r2_linear_combo,
                    'pearson_correlation': correlation_combo,
                    'abs_correlation': abs(correlation_combo),
                    'data_points': len(x_data_combo),
                    'equation': best_equation_combo,
                    'linear_equation': linear_equation_combo,
                    'param_count_mean': y_data_combo.mean(),
                    'param_count_std': y_data_combo.std(),
                    'benchmark_mean': x_data_combo.mean(),
                    'benchmark_std': x_data_combo.std()
                })
                
                combinations_processed += 1
        
        print(f"  Processed {combinations_processed} combinations for {bench_name}")

    # Convert results to DataFrame
    param_results_df = pd.DataFrame(all_detailed_param_results)

    # Save complete results sorted by absolute correlation (descending)
    param_results_df_sorted = param_results_df.sort_values('abs_correlation', ascending=False)
    param_results_df_sorted.to_csv(os.path.join(detailed_param_benchmarks_folder, 'all_combinations_by_correlation.csv'), 
                            index=False, encoding='utf-8')

    # For each benchmark, create individual CSV files
    print("\n" + "="*70)
    print("Creating CSV files for each benchmark...")
    print("="*70)

    for bench_name in BENCH_ROWS_NAMES:
        bench_results = param_results_df[param_results_df['benchmark'] == bench_name]
        
        if len(bench_results) == 0:
            continue
        
        # Sort by absolute correlation (descending)
        bench_results_sorted = bench_results.sort_values('abs_correlation', ascending=False)
        
        # Get benchmark folder
        bench_clean_name = bench_name.replace('BENCH-', '').replace('_', '-').lower()
        bench_main_folder = os.path.join(detailed_param_benchmarks_folder, bench_clean_name)
        
        # Save all combinations for this benchmark (sorted by correlation)
        bench_results_sorted.to_csv(os.path.join(bench_main_folder, 'all_combinations.csv'), 
                                    index=False, encoding='utf-8')
        
        # Save top 20 combinations (or all if less than 20)
        top_n = min(20, len(bench_results_sorted))
        top_results = bench_results_sorted.head(top_n)
        top_results.to_csv(os.path.join(bench_main_folder, 'top_20_combinations.csv'), 
                        index=False, encoding='utf-8')
        
        # Create top 20 visualization for this benchmark
        if len(top_results) > 0:
            # Sort by absolute pearson correlation (ascending so highest abs appears on top in barh)
            top_results_sorted_viz = top_results.sort_values('abs_correlation', ascending=True)
            
            fig = plt.figure(figsize=(14, max(8, top_n * 0.4)))
            
            # Create labels
            labels = [f"{row['type'][:25]}, Filter={int(row['filter'])}" 
                    for _, row in top_results_sorted_viz.iterrows()]
            correlations = top_results_sorted_viz['pearson_correlation'].values
            
            colors = ['green' if c > 0 else 'red' for c in correlations]
            
            plt.barh(range(len(labels)), correlations, color=colors, alpha=0.7, edgecolor='black')
            plt.yticks(range(len(labels)), labels, fontsize=9)
            plt.xlabel('Pearson Correlation Coefficient')
            plt.ylabel('Type - Filter Combination')
            plt.title(f'Top {top_n} Combinations by Absolute Pearson Correlation\n{bench_name.replace("BENCH-", "").replace("_", " ")} vs Parameter Count')
            plt.grid(axis='x', alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            
            # Add value labels on bars
            for i, corr in enumerate(correlations):
                x_offset = 0.003 if corr > 0 else -0.003
                ha_align = 'left' if corr > 0 else 'right'
                plt.text(corr + x_offset, i, f'{corr:.4f}', va='center', fontsize=8, ha=ha_align)
            
            plt.tight_layout()
            plt.savefig(os.path.join(bench_main_folder, 'top_20_correlations.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Created CSV files and plot for {bench_name} ({len(bench_results_sorted)} combinations, top {top_n} saved)")

    # Create a pivot summary showing best correlation for each benchmark-type-filter combination
    param_pivot_summary = param_results_df.pivot_table(
        values='abs_correlation',
        index=['benchmark', 'type'],
        columns='filter',
        aggfunc='max'
    )
    param_pivot_summary.to_csv(os.path.join(detailed_param_benchmarks_folder, 'correlation_matrix.csv'), encoding='utf-8')

    # Create summary text file with overall statistics
    with open(os.path.join(detailed_param_benchmarks_folder, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("DETAILED PARAMETER COUNT ANALYSIS - SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Total combinations analyzed: {len(param_results_df)}\n")
        f.write(f"Benchmarks analyzed: {param_results_df['benchmark'].nunique()}\n")
        f.write(f"Types analyzed: {param_results_df['type'].nunique()}\n")
        f.write(f"Filters analyzed: {sorted(param_results_df['filter'].unique())}\n\n")
        
        f.write("TOP 20 COMBINATIONS BY ABSOLUTE PEARSON CORRELATION:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Rank':<6} {'Benchmark':<30} {'Type':<20} {'Filter':<8} {'Correlation':<15} {'R²':<15} {'N':<8}\n")
        f.write("-" * 100 + "\n")
        
        top_20 = param_results_df_sorted.head(20)
        for idx, (_, row) in enumerate(top_20.iterrows(), 1):
            bench_short = row['benchmark'].replace('BENCH-', '')
            f.write(f"{idx:<6} {bench_short:<30} {row['type']:<20} {row['filter']:<8} "
                    f"{row['pearson_correlation']:<15.8f} {row['r2_score']:<15.8f} {row['data_points']:<8}\n")
        
        f.write("\n\n")
        f.write("BEST COMBINATION FOR EACH BENCHMARK:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Benchmark':<30} {'Type':<20} {'Filter':<8} {'Correlation':<15} {'R²':<15} {'Function':<15}\n")
        f.write("-" * 100 + "\n")
        
        for bench_name in BENCH_ROWS_NAMES:
            bench_best = param_results_df[param_results_df['benchmark'] == bench_name].sort_values('abs_correlation', ascending=False)
            if len(bench_best) > 0:
                best = bench_best.iloc[0]
                bench_short = best['benchmark'].replace('BENCH-', '')
                f.write(f"{bench_short:<30} {best['type']:<20} {best['filter']:<8} "
                        f"{best['pearson_correlation']:<15.8f} {best['r2_score']:<15.8f} {best['best_function']:<15}\n")
        
        f.write("\n\n")
        f.write("STATISTICS BY BENCHMARK:\n")
        f.write("-" * 100 + "\n")
        
        for bench_name in BENCH_ROWS_NAMES:
            bench_data = param_results_df[param_results_df['benchmark'] == bench_name]
            if len(bench_data) > 0:
                f.write(f"\n{bench_name.replace('BENCH-', '')}:\n")
                f.write(f"  Combinations analyzed: {len(bench_data)}\n")
                f.write(f"  Mean correlation: {bench_data['abs_correlation'].mean():.8f}\n")
                f.write(f"  Max correlation: {bench_data['abs_correlation'].max():.8f}\n")
                f.write(f"  Min correlation: {bench_data['abs_correlation'].min():.8f}\n")
                f.write(f"  Mean R²: {bench_data['r2_score'].mean():.8f}\n")

    # Create visualization: Top 20 correlations
    if len(param_results_df) > 0:
        fig = plt.figure(figsize=(16, 10))
        top_20_plot = param_results_df_sorted.head(20).copy()
        
        # Sort by absolute value (ascending so highest abs appears on top in barh)
        top_20_plot_sorted = top_20_plot.sort_values('abs_correlation', ascending=True)
        
        # Create labels
        labels = [f"{row['benchmark'].replace('BENCH-', '')[:20]}\n{row['type'][:15]}, F={float(row['filter'])}" 
                for _, row in top_20_plot_sorted.iterrows()]
        correlations = top_20_plot_sorted['pearson_correlation'].values
        
        colors = ['green' if c > 0 else 'red' for c in correlations]
        
        plt.barh(range(len(labels)), correlations, color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(range(len(labels)), labels, fontsize=8)
        plt.xlabel('Pearson Correlation Coefficient')
        plt.ylabel('Benchmark - Type - Filter Combination')
        plt.title('Top 20 Combinations by Absolute Pearson Correlation\n(Benchmark vs Parameter Count)')
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars
        for i, corr in enumerate(correlations):
            x_offset = 0.003 if corr > 0 else -0.003
            ha_align = 'left' if corr > 0 else 'right'
            plt.text(corr + x_offset, i, f'{corr:.4f}', va='center', fontsize=7, ha=ha_align)
        
        plt.tight_layout()
        plt.savefig(os.path.join(detailed_param_benchmarks_folder, 'top_20_correlations.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "="*70)
    print(f"Detailed parameter count analysis complete!")
    print(f"Total combinations analyzed: {len(param_results_df)}")
    print(f"Results saved to: {detailed_param_benchmarks_folder}")
    print("="*70)

# ================================================================================
   
# --------------------------------------------------------------
# 7 MAGICAL VARIABLE EXPLORE

def equation_exploration(appended_benchmarks_df, OUTPUT_FOLDER):

    # Create output subfolder for equation exploration
    equation_output_folder = os.path.join(OUTPUT_FOLDER, '7_equation_exploration')
    if not os.path.exists(equation_output_folder):
        os.makedirs(equation_output_folder)
    
    # Get rows in header that start with "BENCH"
    BENCH_ROWS_NAMES = appended_benchmarks_df.columns[appended_benchmarks_df.columns.str.startswith('BENCH')].tolist()
    unique_filters = sorted([f for f in appended_benchmarks_df['filter'].unique() if pd.notna(f)])
    
    # Store results
    results = []
    
    # Define the three test groups with their configurations
    test_groups = {
        'combined': {
            'columns': ['count', 'complexity'],
            'folder': 'combined',
            'description': 'count + complexity'
        },
        'only_count': {
            'columns': ['count'],
            'folder': 'only_count',
            'description': 'count only'
        },
        'only_complexity': {
            'columns': ['complexity'],
            'folder': 'only_complexity',
            'description': 'complexity only'
        }
    }
    
    for filter_val in unique_filters:
        # Filter only rows with this filter
        filtered_df = appended_benchmarks_df[appended_benchmarks_df['filter'] == filter_val]
        
        for benchmark in BENCH_ROWS_NAMES:
            
            # Process each test group
            for group_name, group_config in test_groups.items():
                print(f"\nFilter {filter_val}, Benchmark {benchmark}, Group {group_name}:")
                
                # Prepare columns for this group
                columns_to_use = group_config['columns'] + [benchmark]
                exploration_df = filtered_df[columns_to_use].copy()
                
                # Remove rows with invalid values
                exploration_df = exploration_df.dropna(subset=columns_to_use)
                for col in group_config['columns']:
                    exploration_df = exploration_df[exploration_df[col] > 0]
                
                if len(exploration_df) < 5:  # Need at least 5 data points
                    print(f"  Not enough data points ({len(exploration_df)})")
                    continue
                
                print(f"  Data points: {len(exploration_df)}")
                
                # Prepare data for PySR
                X = exploration_df[group_config['columns']].values
                if len(group_config['columns']) == 1:
                    X = X.reshape(-1, 1)  # Reshape for single variable
                y = exploration_df[benchmark].values
                
                # Configure PySR
                model = PySRRegressor(
                    # Binary operators: arithmetic and comparison
                    binary_operators=[
                        "+", "-", "*", "/",         # Basic arithmetic
                        "^",                        # Power
                        "max", "min",               # Comparison
                    ],
                    # Unary operators organized by category
                    unary_operators=[
                        # Exponential
                        "exp",
                        # Logarithm
                        "log",
                        # Basic transformations
                        "abs",
                        # Rounding
                        "ceil", "floor",
                    ],
                    niterations=500,
                    populations=10,
                    population_size=500,
                    maxsize=20, 
                    model_selection="best",  # or "accuracy"
                    verbosity=0,
                    progress=False
                )
                
                try:
                    # Fit the model
                    print(f"  Running symbolic regression...")
                    model.fit(X, y)
                    
                    # Get the best equations
                    equations = model.equations_
                    
                    # Save equations to file
                    benchmark_clean = benchmark.replace('BENCH-', '')
                    
                    evals_folder = os.path.join(equation_output_folder, 'evals')
                    group_folder = os.path.join(evals_folder, group_config['folder'])
                    if not os.path.exists(group_folder):
                        os.makedirs(group_folder)
                    
                    output_file = os.path.join(
                        group_folder,
                        f'filter_{filter_val}_{benchmark_clean}.csv'
                    )
                    equations.to_csv(output_file, index=False)
                    
                    # Get best equation
                    # find lowest-loss equation
                    min_loss = equations["loss"].min()
                    # choose the simplest equation within 5% of best loss
                    subset = equations[equations["loss"] <= 1.05 * min_loss]
                    best_row = subset.loc[subset["complexity"].idxmin()]
                    best_equation = best_row["equation"]
                    best_score = best_row["score"]
                    
                    # Calculate prediction error metrics
                    y_pred = model.predict(X)
                    mse = np.mean((y - y_pred) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y - y_pred))
                    
                    print(f"  Best equation: {best_equation}")
                    print(f"  R² score: {best_score:.4f}")
                    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                    
                    # Store results
                    results.append({
                        'filter': filter_val,
                        'benchmark': benchmark,
                        'group': group_name,
                        'equation': str(best_equation),
                        'r2_score': best_score,
                        'rmse': rmse,
                        'mae': mae,
                        'mse': mse,
                        'n_points': len(exploration_df)
                    })
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
    
    # Save summary.csv
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        summary_csv_path = os.path.join(equation_output_folder, 'summary.csv')
        results_df.to_csv(summary_csv_path, index=False)
        print(f"\n{'='*70}")
        print(f"Equation exploration complete!")
        print(f"Total equations found: {len(results)}")
        print(f"  Combined: {len(results_df[results_df['group'] == 'combined'])}")
        print(f"  Only count: {len(results_df[results_df['group'] == 'only_count'])}")
        print(f"  Only complexity: {len(results_df[results_df['group'] == 'only_complexity'])}")
        print(f"Summary saved to: {summary_csv_path}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("Warning: No equations were successfully generated!")
        print(f"{'='*70}\n")
        
def analyze_equation_exploration(OUTPUT_FOLDER):
    
    # Create output subfolder for equation exploration
    equation_output_folder = os.path.join(OUTPUT_FOLDER, '7_equation_exploration')
    if not os.path.exists(equation_output_folder):
        os.makedirs(equation_output_folder)
    
    # Create plot folders for each group
    plot_folders = {
        'combined': os.path.join(equation_output_folder, 'plots', 'combined'),
        'only_count': os.path.join(equation_output_folder, 'plots', 'only_count'),
        'only_complexity': os.path.join(equation_output_folder, 'plots', 'only_complexity')
    }
    
    for folder in plot_folders.values():
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Read the summary.csv file
    summary_csv_path = os.path.join(equation_output_folder, 'summary.csv')
    
    if not os.path.exists(summary_csv_path):
        print(f"Summary CSV not found at: {summary_csv_path}")
        return
    
    print("\n" + "="*70)
    print("Starting equation exploration plot generation...")
    print("="*70)
    
    # Read the CSV
    summary_df = pd.read_csv(summary_csv_path)
    
    # Define ranges for x0 (count) and x1 (complexity)
    # Count: 1 to 100 billion (1e11)
    # Complexity: 0.01 to 1.0 (avoid 0 for some functions)
    count_range = np.logspace(0, 11, 500)  # 1 to 100 billion, log-spaced
    
    # Create multiple complexity values to plot as separate curves
    # From 0.01 to 1.0 with 20 different values
    complexity_values = np.linspace(0.01, 1.0, 20)
    
    # Process each equation
    for idx, row in summary_df.iterrows():
        filter_val = int(row['filter'])
        benchmark = row['benchmark'].replace('BENCH-', '')
        equation_str = row['equation']
        r2_score = row['r2_score']
        n_points = row['n_points']
        group = row['group']  # Get the group (combined, only_count, only_complexity)
        
        print(f"\nProcessing: Filter {filter_val}, {benchmark}, Group {group}")
        print(f"  Equation: {equation_str}")
        print(f"  R² score: {r2_score:.4f}")
        
        try:
            # Convert the equation string to a SymPy expression
            import sympy as sp
            
            # Determine number of variables based on group
            if group == 'combined':
                # Both x0 (count) and x1 (complexity)
                x0, x1 = sp.symbols('x0 x1')
                symbols = (x0, x1)
            elif group == 'only_count':
                # Only x0 (count)
                x0 = sp.symbols('x0')
                symbols = (x0,)
            elif group == 'only_complexity':
                # Only x0 (but represents complexity in this case)
                x0 = sp.symbols('x0')
                symbols = (x0,)
            else:
                print(f"  Warning: Unknown group '{group}', skipping")
                continue
            
            # Parse the equation with custom functions
            local_dict = {
                'x0': symbols[0] if len(symbols) >= 1 else None,
                'x1': symbols[1] if len(symbols) >= 2 else None,
                'log': sp.log,
                'log2': lambda x: sp.log(x, 2),
                'log10': lambda x: sp.log(x, 10),
                'exp': sp.exp,
                'exp2': lambda x: 2**x,
                'sqrt': sp.sqrt,
                'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
                'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
                'ceiling': sp.ceiling, 'floor': sp.floor,
                'sign': sp.sign, 'Abs': sp.Abs,
                'Mod': sp.Mod, 'Piecewise': sp.Piecewise
            }
            expr = sp.sympify(equation_str, locals=local_dict)
            
            # Create a numerical function from the symbolic expression
            numpy_funcs = {
                'Mod': np.mod,
                'ceiling': np.ceil,
                'floor': np.floor,
                'sign': np.sign,
                'Abs': np.abs
            }
            func = sp.lambdify(symbols, expr, modules=[numpy_funcs, 'numpy'])
            
            # Get the plot folder for this group
            current_plot_folder = plot_folders[group]
            
            # ============================================================
            # PLOTTING LOGIC DEPENDS ON GROUP
            # ============================================================
            
            if group == 'combined':
                # Plot with multiple complexity curves (original behavior)
                # Create the plot
                fig, ax = plt.subplots(figsize=(14, 10))
                
                # Use a colormap for gradual color changes
                cmap = plt.cm.viridis
                colors = [cmap(i / len(complexity_values)) for i in range(len(complexity_values))]
                
                # Plot a curve for each complexity value
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    for i, complexity_val in enumerate(complexity_values):
                        try:
                            # Evaluate function with this complexity value across all count values
                            y_values = func(count_range, complexity_val)
                            
                            # If function returns a scalar (doesn't depend on x0), broadcast it
                            if np.isscalar(y_values) or (isinstance(y_values, np.ndarray) and y_values.shape == ()):
                                y_values = np.full_like(count_range, y_values)
                            elif isinstance(y_values, np.ndarray) and len(y_values) == 1:
                                y_values = np.full_like(count_range, y_values[0])
                            
                            # Handle any invalid values
                            y_values = np.nan_to_num(y_values, nan=0, posinf=1e6, neginf=-1e6)
                            
                            # Plot this curve
                            label = f'Complexity = {complexity_val:.2f}'
                            ax.plot(count_range, y_values, color=colors[i], linewidth=1.5, 
                                   label=label, alpha=0.8)
                            
                        except Exception as e:
                            print(f"  Warning: Could not evaluate for complexity={complexity_val:.2f}: {e}")
                            continue
                
                # Configure the plot
                ax.set_xlabel('Count (x0) - Parameter Count', fontsize=12, fontweight='bold')
                ax.set_ylabel('Benchmark Score', fontsize=12, fontweight='bold')
                ax.set_xscale('log')
                ax.set_title(f'Filter {filter_val}, {benchmark} [Combined]\n{equation_str}\nR² = {r2_score:.6f}, N = {n_points}',
                            fontsize=11, pad=15)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add legend with smaller font and multiple columns
                ax.legend(loc='best', bbox_to_anchor=(1, 0.5), fontsize=8, 
                         ncol=1, frameon=True, fancybox=True, shadow=True, framealpha=0.5)
                
                # Create the plot filename
                plot_filename = f"filter_{filter_val}_{benchmark.replace(' ', '_').lower()}.png"
                plot_path = os.path.join(current_plot_folder, plot_filename)
                
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved plot to: {group}/plots/{plot_filename}")
                
                # ============================================================
                # CREATE SECOND PLOT: Linear scale, count from 1 to 1000
                # ============================================================
                
                count_range_linear = np.linspace(1, 1000, 500)  # 1 to 1000, linear-spaced
                
                fig, ax = plt.subplots(figsize=(14, 10))
                
                # Use the same colormap for consistency
                cmap = plt.cm.viridis
                colors = [cmap(i / len(complexity_values)) for i in range(len(complexity_values))]
                
                # Plot a curve for each complexity value
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    for i, complexity_val in enumerate(complexity_values):
                        try:
                            # Evaluate function with this complexity value across all count values
                            y_values = func(count_range_linear, complexity_val)
                            
                            # If function returns a scalar (doesn't depend on x0), broadcast it
                            if np.isscalar(y_values) or (isinstance(y_values, np.ndarray) and y_values.shape == ()):
                                y_values = np.full_like(count_range_linear, y_values)
                            elif isinstance(y_values, np.ndarray) and len(y_values) == 1:
                                y_values = np.full_like(count_range_linear, y_values[0])
                            
                            # Handle any invalid values
                            y_values = np.nan_to_num(y_values, nan=0, posinf=1e6, neginf=-1e6)
                            
                            # Plot this curve
                            label = f'Complexity = {complexity_val:.2f}'
                            ax.plot(count_range_linear, y_values, color=colors[i], linewidth=1.5, 
                                   label=label, alpha=0.8)
                            
                        except Exception as e:
                            print(f"  Warning: Could not evaluate for complexity={complexity_val:.2f} (linear): {e}")
                            continue
                
                # Configure the plot
                ax.set_xlabel('Count (x0) - Parameter Count', fontsize=12, fontweight='bold')
                ax.set_ylabel('Benchmark Score', fontsize=12, fontweight='bold')
                # Linear scale - no set_xscale('log')
                ax.set_title(f'Filter {filter_val}, {benchmark} [Combined] (Linear Scale: 1-1000)\n{equation_str}\nR² = {r2_score:.6f}, N = {n_points}',
                            fontsize=11, pad=15)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add legend with smaller font and multiple columns
                ax.legend(loc='best', bbox_to_anchor=(1, 0.5), fontsize=8, 
                         ncol=1, frameon=True, fancybox=True, shadow=True, framealpha=0.5)
                
                # Create the plot filename for linear version
                plot_filename_linear = f"filter_{filter_val}_{benchmark.replace(' ', '_').lower()}_linear.png"
                plot_path_linear = os.path.join(current_plot_folder, plot_filename_linear)
                
                plt.tight_layout()
                plt.savefig(plot_path_linear, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved linear plot to: {group}/plots/{plot_filename_linear}")
                
            elif group == 'only_count':
                # Plot with count only (log scale)
                fig, ax = plt.subplots(figsize=(14, 10))
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    try:
                        # Evaluate function with count values
                        y_values = func(count_range)
                        
                        # If function returns a scalar, broadcast it
                        if np.isscalar(y_values) or (isinstance(y_values, np.ndarray) and y_values.shape == ()):
                            y_values = np.full_like(count_range, y_values)
                        elif isinstance(y_values, np.ndarray) and len(y_values) == 1:
                            y_values = np.full_like(count_range, y_values[0])
                        
                        # Handle any invalid values
                        y_values = np.nan_to_num(y_values, nan=0, posinf=1e6, neginf=-1e6)
                        
                        # Plot the curve
                        ax.plot(count_range, y_values, color='steelblue', linewidth=2, 
                               label='Prediction', alpha=0.8)
                        
                    except Exception as e:
                        print(f"  Warning: Could not evaluate equation: {e}")
                
                # Configure the plot
                ax.set_xlabel('Count (x0) - Parameter Count', fontsize=12, fontweight='bold')
                ax.set_ylabel('Benchmark Score', fontsize=12, fontweight='bold')
                ax.set_xscale('log')
                ax.set_title(f'Filter {filter_val}, {benchmark} [Count Only]\n{equation_str}\nR² = {r2_score:.6f}, N = {n_points}',
                            fontsize=11, pad=15)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.8)
                
                # Create the plot filename
                plot_filename = f"filter_{filter_val}_{benchmark.replace(' ', '_').lower()}.png"
                plot_path = os.path.join(current_plot_folder, plot_filename)
                
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved plot to: {group}/plots/{plot_filename}")
                
                # Linear scale version
                count_range_linear = np.linspace(1, 1000, 500)
                fig, ax = plt.subplots(figsize=(14, 10))
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    try:
                        y_values = func(count_range_linear)
                        if np.isscalar(y_values) or (isinstance(y_values, np.ndarray) and y_values.shape == ()):
                            y_values = np.full_like(count_range_linear, y_values)
                        elif isinstance(y_values, np.ndarray) and len(y_values) == 1:
                            y_values = np.full_like(count_range_linear, y_values[0])
                        y_values = np.nan_to_num(y_values, nan=0, posinf=1e6, neginf=-1e6)
                        ax.plot(count_range_linear, y_values, color='steelblue', linewidth=2, 
                               label='Prediction', alpha=0.8)
                    except Exception as e:
                        print(f"  Warning: Could not evaluate equation (linear): {e}")
                
                ax.set_xlabel('Count (x0) - Parameter Count', fontsize=12, fontweight='bold')
                ax.set_ylabel('Benchmark Score', fontsize=12, fontweight='bold')
                ax.set_title(f'Filter {filter_val}, {benchmark} [Count Only] (Linear Scale: 1-1000)\n{equation_str}\nR² = {r2_score:.6f}, N = {n_points}',
                            fontsize=11, pad=15)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.5)
                
                plot_filename_linear = f"filter_{filter_val}_{benchmark.replace(' ', '_').lower()}_linear.png"
                plot_path_linear = os.path.join(current_plot_folder, plot_filename_linear)
                
                plt.tight_layout()
                plt.savefig(plot_path_linear, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved linear plot to: {group}/plots/{plot_filename_linear}")
                
            elif group == 'only_complexity':
                # Plot with complexity only
                complexity_range = np.linspace(0.01, 1.0, 500)  # Complexity from 0.01 to 1.0
                
                fig, ax = plt.subplots(figsize=(14, 10))
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    try:
                        # Evaluate function with complexity values
                        y_values = func(complexity_range)
                        
                        # If function returns a scalar, broadcast it
                        if np.isscalar(y_values) or (isinstance(y_values, np.ndarray) and y_values.shape == ()):
                            y_values = np.full_like(complexity_range, y_values)
                        elif isinstance(y_values, np.ndarray) and len(y_values) == 1:
                            y_values = np.full_like(complexity_range, y_values[0])
                        
                        # Handle any invalid values
                        y_values = np.nan_to_num(y_values, nan=0, posinf=1e6, neginf=-1e6)
                        
                        # Plot the curve
                        ax.plot(complexity_range, y_values, color='darkgreen', linewidth=2, 
                               label='Prediction', alpha=0.8)
                        
                    except Exception as e:
                        print(f"  Warning: Could not evaluate equation: {e}")
                
                # Configure the plot
                ax.set_xlabel('Complexity (x0)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Benchmark Score', fontsize=12, fontweight='bold')
                ax.set_title(f'Filter {filter_val}, {benchmark} [Complexity Only]\n{equation_str}\nR² = {r2_score:.6f}, N = {n_points}',
                            fontsize=11, pad=15)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.8)
                
                # Create the plot filename
                plot_filename = f"filter_{filter_val}_{benchmark.replace(' ', '_').lower()}.png"
                plot_path = os.path.join(current_plot_folder, plot_filename)
                
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved plot to: {group}/plots/{plot_filename}")
            
        except Exception as e:
            print(f"  Error processing equation: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("Equation exploration plot generation complete!")
    for group, folder in plot_folders.items():
        print(f"  {group} plots saved to: {folder}")
    print("="*70)

def create_equation_rankings(OUTPUT_FOLDER):
    
    # Create output subfolder for equation exploration
    equation_output_folder = os.path.join(OUTPUT_FOLDER, '7_equation_exploration')
    if not os.path.exists(equation_output_folder):
        os.makedirs(equation_output_folder)
    
    # Create rankings folder
    rankings_folder = os.path.join(equation_output_folder, 'rankings')
    if not os.path.exists(rankings_folder):
        os.makedirs(rankings_folder)
    
    # Read the summary.csv file
    summary_csv_path = os.path.join(equation_output_folder, 'summary.csv')
    
    if not os.path.exists(summary_csv_path):
        print(f"Summary CSV not found at: {summary_csv_path}")
        return
    
    print("\n" + "="*70)
    print("Starting equation rankings generation...")
    print("="*70)
    
    # Read the CSV
    summary_df = pd.read_csv(summary_csv_path)
    
    # Error columns to visualize
    error_columns = ['rmse', 'mae', 'mse']
    
    # Check if error columns exist
    missing_cols = [col for col in error_columns if col not in summary_df.columns]
    if missing_cols:
        print(f"Warning: Missing error columns: {missing_cols}")
        error_columns = [col for col in error_columns if col in summary_df.columns]
        if not error_columns:
            print("No error columns found. Cannot create rankings.")
            return
    
    # Process each group
    groups = summary_df['group'].unique()
    
    for group in groups:
        print(f"\nProcessing group: {group}")
        
        # Filter data for this group
        group_df = summary_df[summary_df['group'] == group].copy()
        
        if len(group_df) == 0:
            print(f"  No data for group {group}")
            continue
        
        # Calculate average error (mean of all error columns)
        group_df['avg_error'] = group_df[error_columns].mean(axis=1)
        
        # Sort by average error (ascending - lower is better)
        group_df = group_df.sort_values('avg_error', ascending=True)
        
        # Get top 20 (or all if less than 20)
        top_n = min(20, len(group_df))
        top_df = group_df.head(top_n)
        
        print(f"  Creating heatmap for top {top_n} functions")
        
        # Create labels combining filter, benchmark, and average error
        labels = []
        for _, row in top_df.iterrows():
            filter_val = int(row['filter'])
            benchmark = row['benchmark'].replace('BENCH-', '')
            avg_err = row['avg_error']
            equation = row['equation']
            # Truncate benchmark name if too long
            if len(benchmark) > 20:
                benchmark = benchmark[:17] + '...'
            label = f"F{filter_val}-{benchmark} (Avg: {avg_err:.4f})\n{equation}"
            labels.append(label)
        
        # Extract error values for heatmap
        error_data = top_df[error_columns].values
        
        # Create the heatmap (increased height to accommodate two-line labels)
        fig, ax = plt.subplots(figsize=(14, max(10, top_n * 0.6)))

        # Use seaborn heatmap for better styling and annotations
        sns.heatmap(
            error_data,
            ax=ax,
            cmap='YlOrRd',
            annot=True,
            fmt='.4f',
            cbar_kws={'label': 'Error Value'},
            linewidths=0.5,
            linecolor='white'
        )

        # Set ticks and labels
        ax.set_xticks(np.arange(len(error_columns)) + 0.5)
        ax.set_xticklabels([col.upper() for col in error_columns], fontsize=11, fontweight='bold')
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(labels, fontsize=8, linespacing=1.2)
        
        # Set title
        ax.set_title(f'Top {top_n} Functions by Average Error - Group: {group.replace("_", " ").title()}\n'
                    f'(Lower values are better)', 
                    fontsize=12, fontweight='bold', pad=15)
        
        # Set labels
        ax.set_xlabel('Error Measures', fontsize=11, fontweight='bold')
        ax.set_ylabel('Filter-Benchmark (Average Error)', fontsize=11, fontweight='bold')
        
        # Add grid
        ax.set_xticks(np.arange(len(error_columns)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(labels)) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'ranking_{group}.png'
        plot_path = os.path.join(rankings_folder, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved ranking heatmap to: rankings/{plot_filename}")
        
        # Also save the ranking data to CSV
        csv_filename = f'ranking_{group}.csv'
        csv_path = os.path.join(rankings_folder, csv_filename)
        
        # Select relevant columns for CSV
        ranking_cols = ['filter', 'benchmark', 'equation', 'r2_score', 'avg_error'] + error_columns + ['n_points']
        top_df[ranking_cols].to_csv(csv_path, index=False)
        
        print(f"  Saved ranking data to: rankings/{csv_filename}")
    
    # Create a summary comparison across groups
    print("\n  Creating cross-group comparison...")
    
    # For each group, get the best function (lowest avg error)
    best_per_group = []
    for group in groups:
        group_df = summary_df[summary_df['group'] == group].copy()
        if len(group_df) > 0:
            group_df['avg_error'] = group_df[error_columns].mean(axis=1)
            best_row = group_df.loc[group_df['avg_error'].idxmin()]
            best_per_group.append({
                'group': group,
                'filter': best_row['filter'],
                'benchmark': best_row['benchmark'],
                'avg_error': best_row['avg_error'],
                'r2_score': best_row['r2_score'],
                **{col: best_row[col] for col in error_columns}
            })
    
    if len(best_per_group) > 0:
        best_df = pd.DataFrame(best_per_group)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(best_df))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for each group
        
        bars = ax.bar(x_pos, best_df['avg_error'], color=colors[:len(best_df)], alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Error (Lower is Better)', fontsize=12, fontweight='bold')
        ax.set_title('Best Function per Group - Average Error Comparison', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([g.replace('_', ' ').title() for g in best_df['group']], fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, best_df['avg_error'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(rankings_folder, 'best_per_group.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save summary CSV
        best_df.to_csv(os.path.join(rankings_folder, 'best_per_group.csv'), index=False)
        
        print(f"  Saved cross-group comparison to: rankings/best_per_group.png")
    
    print("\n" + "="*70)
    print("Equation rankings generation complete!")
    print(f"Rankings saved to: {rankings_folder}")
    print("="*70)
            
            
# ================================================================================

# --------------------------------------------------------------
# 8 FOR EACH BENCHMARK - MAGICAL VARIABLE vs BENCHMARK

def analyze_magical_var_vs_benchmarks(appended_benchmarks_df, OUTPUT_FOLDER):
    # Get rows in header that start with "BENCH"
    BENCH_ROWS_NAMES = appended_benchmarks_df.columns[appended_benchmarks_df.columns.str.startswith('BENCH')].tolist()

    # Create magical variable: complexity * param count
    appended_benchmarks_df['magical_var'] = appended_benchmarks_df['complexity'] * appended_benchmarks_df['count']

    print("\n" + "="*70)
    print("Starting MAGICAL VARIABLE (Complexity × Param Count) vs BENCHMARK analysis...")
    print("="*70)

    # Create main output folder
    magical_benchmarks_folder = os.path.join(OUTPUT_FOLDER, '8_magical_var_vs_benchmarks')
    if not os.path.exists(magical_benchmarks_folder):
        os.makedirs(magical_benchmarks_folder)

    # Dictionary to store all results
    all_magical_benchmark_results = {}

    # Process each benchmark column
    for bench_name in BENCH_ROWS_NAMES:
        print(f"Processing benchmark: {bench_name}")
        
        # Remove NaN values for regression
        mask_bench = ~(appended_benchmarks_df[bench_name].isna() | appended_benchmarks_df['magical_var'].isna())
        x_data_bench = appended_benchmarks_df.loc[mask_bench, bench_name].values
        y_data_bench = appended_benchmarks_df.loc[mask_bench, 'magical_var'].values
        
        # Skip if not enough data points
        if len(x_data_bench) < 3:
            print(f"  Skipping {bench_name}: insufficient data points ({len(x_data_bench)})")
            continue
        
        # Try different functions and find the best fit
        best_r2_bench = -np.inf
        best_name_bench = None
        best_params_bench = None
        best_func_bench = None
        best_equation_bench = None
        
        FUNCTIONS_TO_TEST_BENCH = {
            'linear': {
                'func': lambda x, a, b: a * x + b,
                'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
                'initial_guess': [1, 1]
            },
            'quadratic': {
                'func': lambda x, a, b, c: a * x**2 + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'exponential': {
                'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
                'initial_guess': [1, 0.1, 1]
            },
            'logarithmic': {
                'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
                'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
                'initial_guess': [1, 1, 1]
            },
            'power': {
                'func': lambda x, a, b, c: a * (x + 1)**b + c,
                'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
                'initial_guess': [1, 0.5, 1]
            }
        }
        
        for name, func_info in FUNCTIONS_TO_TEST_BENCH.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params, _ = curve_fit(func_info['func'], x_data_bench, y_data_bench, 
                                        p0=func_info['initial_guess'], maxfev=10000)
                y_pred = func_info['func'](x_data_bench, *params)
                r2 = 1 - (np.sum((y_data_bench - y_pred)**2) / np.sum((y_data_bench - np.mean(y_data_bench))**2))
                
                if r2 > best_r2_bench:
                    best_r2_bench = r2
                    best_name_bench = name
                    best_params_bench = params
                    best_func_bench = func_info['func']
                    best_equation_bench = func_info['equation'](params)
            except:
                # Skip if curve fitting fails for this function
                pass
        
        # If no fit was successful, skip this benchmark
        if best_func_bench is None:
            print(f"  Skipping {bench_name}: curve fitting failed for all functions")
            continue
        
        # Create regression line with best fit
        x_line_bench = np.linspace(x_data_bench.min(), x_data_bench.max(), 100)
        y_line_bench = best_func_bench(x_line_bench, *best_params_bench)
        
        # Create subfolder for this benchmark
        bench_clean_name = bench_name.replace('BENCH-', '').replace('_', '-').lower()
        bench_folder = os.path.join(magical_benchmarks_folder, bench_clean_name)
        if not os.path.exists(bench_folder):
            os.makedirs(bench_folder)
        
        # Plot scatter with regression line
        fig = plt.figure(figsize=(12, 9))
        plt.scatter(x_data_bench, y_data_bench, alpha=0.6, label='Data')
        plt.plot(x_line_bench, y_line_bench, 'r-', linewidth=2, label=f'Best fit ({best_name_bench})\n{best_equation_bench}\nR² = {best_r2_bench:.6f}')
        plt.xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
        plt.ylabel('Magical Variable (Complexity × Param Count)')
        plt.title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Magical Variable')
        plt.legend(loc='best', bbox_to_anchor=(0.48, -0.18), ncol=1, frameon=True, fancybox=True, shadow=True, fontsize=8)
        
        plt.savefig(os.path.join(bench_folder, 'regression.png'))
        plt.close()
        
        # Calculate Pearson correlation coefficient
        correlation = np.corrcoef(x_data_bench, y_data_bench)[0, 1]
        
        # Save regression information to text file
        with open(os.path.join(bench_folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{bench_name} VS MAGICAL VARIABLE - REGRESSION ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("REGRESSION:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Best fit function: {best_name_bench}\n")
            f.write(f"Expression: {best_equation_bench}\n")
            f.write(f"R² score: {best_r2_bench:.16f}\n")
            f.write(f"Pearson correlation: {correlation:.16f}\n")
            f.write(f"Parameters: {best_params_bench}\n")
        
        # Save statistics to text file
        with open(os.path.join(bench_folder, 'statistics.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{bench_name} VS MAGICAL VARIABLE - STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("MAGICAL VARIABLE STATISTICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Number of data points: {len(y_data_bench)}\n")
            f.write(f"Mean magical variable: {y_data_bench.mean():.16f}\n")
            f.write(f"Median magical variable: {np.median(y_data_bench):.16f}\n")
            f.write(f"Std magical variable: {y_data_bench.std():.16f}\n")
            f.write(f"Min magical variable: {y_data_bench.min():.16f}\n")
            f.write(f"Max magical variable: {y_data_bench.max():.16f}\n\n")
            
            f.write(f"{bench_name} STATISTICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Min score: {x_data_bench.min():.16f}\n")
            f.write(f"Max score: {x_data_bench.max():.16f}\n")
            f.write(f"Mean score: {x_data_bench.mean():.16f}\n")
            f.write(f"Median score: {np.median(x_data_bench):.16f}\n")
            f.write(f"Std score: {x_data_bench.std():.16f}\n")
        
        # Store results for summary
        all_magical_benchmark_results[bench_name] = {
            'best_function': best_name_bench,
            'r2_score': best_r2_bench,
            'correlation': correlation,
            'equation': best_equation_bench,
            'data_points': len(x_data_bench)
        }
        
        print(f"  Completed {bench_name}: R² = {best_r2_bench:.6f}, Correlation = {correlation:.6f}")

    # Create summary file with all benchmark results
    with open(os.path.join(magical_benchmarks_folder, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("BENCHMARKS VS MAGICAL VARIABLE - SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Benchmark':<35} {'Best Fit':<15} {'R² Score':<15} {'Correlation':<15} {'Data Points':<15}\n")
        f.write("-" * 100 + "\n")
        
        # Sort by R² score (descending)
        sorted_results = sorted(all_magical_benchmark_results.items(), key=lambda x: x[1]['r2_score'], reverse=True)
        
        for bench_name, results in sorted_results:
            f.write(f"{bench_name:<35} {results['best_function']:<15} {results['r2_score']:<15.16f} "
                    f"{results['correlation']:<15.16f} {results['data_points']:<15}\n")
        
        # Calculate and add averages
        if len(all_magical_benchmark_results) > 0:
            avg_r2 = sum(r['r2_score'] for r in all_magical_benchmark_results.values()) / len(all_magical_benchmark_results)
            avg_corr = sum(r['correlation'] for r in all_magical_benchmark_results.values()) / len(all_magical_benchmark_results)
            f.write("-" * 100 + "\n")
            f.write(f"{'AVERAGE':<35} {'':<15} {avg_r2:<15.16f} {avg_corr:<15.16f}\n")
        
        f.write("\n\n")
        f.write("DETAILED EQUATIONS:\n")
        f.write("-" * 100 + "\n")
        for bench_name, results in sorted_results:
            f.write(f"\n{bench_name}:\n")
            f.write(f"  {results['equation']}\n")

    # Create comparison plot: R² scores for all benchmarks
    if len(all_magical_benchmark_results) > 0:
        fig = plt.figure(figsize=(14, 8))
        bench_names_short = [name.replace('BENCH-', '').replace('_', ' ') for name in all_magical_benchmark_results.keys()]
        r2_scores = [results['r2_score'] for results in all_magical_benchmark_results.values()]
        
        # Sort by R² score
        sorted_indices = np.argsort(r2_scores)[::-1]
        bench_names_short = [bench_names_short[i] for i in sorted_indices]
        r2_scores = [r2_scores[i] for i in sorted_indices]
        
        plt.barh(range(len(bench_names_short)), r2_scores, color='steelblue', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(bench_names_short)), bench_names_short, fontsize=9)
        plt.xlabel('R² Score')
        plt.ylabel('Benchmark')
        plt.title('Regression Quality: R² Scores for Each Benchmark vs Magical Variable')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (name, score) in enumerate(zip(bench_names_short, r2_scores)):
            plt.text(score, i, f' {score:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(magical_benchmarks_folder, 'r2_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create comparison plot: Correlation coefficients for all benchmarks
        fig = plt.figure(figsize=(14, 8))
        correlations = [all_magical_benchmark_results[list(all_magical_benchmark_results.keys())[i]]['correlation'] 
                    for i in sorted_indices]
        
        colors = ['green' if c > 0 else 'red' for c in correlations]
        plt.barh(range(len(bench_names_short)), correlations, color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(range(len(bench_names_short)), bench_names_short, fontsize=9)
        plt.xlabel('Pearson Correlation Coefficient')
        plt.ylabel('Benchmark')
        plt.title('Correlation: Benchmark Score vs Magical Variable')
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars
        for i, (name, corr) in enumerate(zip(bench_names_short, correlations)):
            plt.text(corr, i, f' {corr:.4f}', va='center', fontsize=8, 
                    ha='left' if corr > 0 else 'right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(magical_benchmarks_folder, 'correlation_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nCompleted magical variable analysis. Processed {len(all_magical_benchmark_results)} benchmarks.")

    # --------------------------------------------------------------
    # Plot graphs: FOR EACH BENCHMARK: FOR EACH TYPE: FOR EACH FILTER: - MAGICAL VARIABLE vs BENCHMARK

    print("\n" + "="*70)
    print("Starting detailed MAGICAL VARIABLE analysis by type and filter...")
    print("="*70)

    # Create main output folder
    detailed_magical_benchmarks_folder = os.path.join(OUTPUT_FOLDER, '8_2_detailed_magical_var_analysis')
    if not os.path.exists(detailed_magical_benchmarks_folder):
        os.makedirs(detailed_magical_benchmarks_folder)

    # Dictionary to store all detailed results
    all_detailed_magical_results = []

    # Create types_str column if it doesn't exist
    if 'types_str' not in appended_benchmarks_df.columns:
        appended_benchmarks_df['types_str'] = appended_benchmarks_df['types'].apply(
            lambda x: '-'.join(sorted(x)) if isinstance(x, list) and len(x) > 0 else 'unknown'
        )

    # Get unique types and filters (local to this function)
    unique_types = appended_benchmarks_df['types_str'].unique()
    unique_filters = sorted([f for f in appended_benchmarks_df['filter'].unique() if pd.notna(f)])

    # Process each benchmark
    for bench_name in BENCH_ROWS_NAMES:
        print(f"\nProcessing benchmark: {bench_name}")
        
        # Create folder for this benchmark
        bench_clean_name = bench_name.replace('BENCH-', '').replace('_', '-').lower()
        bench_main_folder = os.path.join(detailed_magical_benchmarks_folder, bench_clean_name)
        if not os.path.exists(bench_main_folder):
            os.makedirs(bench_main_folder)
        
        # Counter for combinations processed
        combinations_processed = 0
        
        # Process each type
        for type_str in unique_types:
            # Create folder for this type
            type_clean_name = type_str.replace('-', '_').replace(' ', '_').lower()
            if type_clean_name == 'unknown':
                continue  # Skip unknown types
            
            type_folder = os.path.join(bench_main_folder, type_clean_name)
            if not os.path.exists(type_folder):
                os.makedirs(type_folder)
            
            # Process each filter
            for filter_val in unique_filters:
                # Filter data for this specific combination
                mask_combo = (
                    (appended_benchmarks_df['types_str'] == type_str) &
                    (appended_benchmarks_df['filter'] == filter_val) &
                    ~(appended_benchmarks_df[bench_name].isna()) &
                    ~(appended_benchmarks_df['magical_var'].isna())
                )
                
                x_data_combo = appended_benchmarks_df.loc[mask_combo, bench_name].values
                y_data_combo = appended_benchmarks_df.loc[mask_combo, 'magical_var'].values
                
                # Skip if not enough data points
                if len(x_data_combo) < 3:
                    continue
                
                # Try different functions and find the best fit
                best_r2_combo = -np.inf
                best_name_combo = None
                best_params_combo = None
                best_func_combo = None
                best_equation_combo = None
                
                FUNCTIONS_TO_TEST_COMBO = {
                    'linear': {
                        'func': lambda x, a, b: a * x + b,
                        'equation': lambda params: f'y = {params[0]:.16f}x + {params[1]:.16f}',
                        'initial_guess': [1, 1]
                    },
                    'quadratic': {
                        'func': lambda x, a, b, c: a * x**2 + b * x + c,
                        'equation': lambda params: f'y = {params[0]:.16f}x² + {params[1]:.16f}x + {params[2]:.16f}',
                        'initial_guess': [1, 1, 1]
                    },
                    'exponential': {
                        'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                        'equation': lambda params: f'y = {params[0]:.16f}·e^({params[1]:.16f}x) + {params[2]:.16f}',
                        'initial_guess': [1, 0.1, 1]
                    },
                    'logarithmic': {
                        'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
                        'equation': lambda params: f'y = {params[0]:.16f}·ln(x+1) + {params[1]:.16f}x + {params[2]:.16f}',
                        'initial_guess': [1, 1, 1]
                    },
                    'power': {
                        'func': lambda x, a, b, c: a * (x + 1)**b + c,
                        'equation': lambda params: f'y = {params[0]:.16f}·(x+1)^{params[1]:.16f} + {params[2]:.16f}',
                        'initial_guess': [1, 0.5, 1]
                    }
                }
                
                for name, func_info in FUNCTIONS_TO_TEST_COMBO.items():
                    try:
                        params, _ = curve_fit(func_info['func'], x_data_combo, y_data_combo, 
                                                p0=func_info['initial_guess'], maxfev=10000)
                        y_pred = func_info['func'](x_data_combo, *params)
                        r2 = 1 - (np.sum((y_data_combo - y_pred)**2) / np.sum((y_data_combo - np.mean(y_data_combo))**2))
                        
                        if r2 > best_r2_combo:
                            best_r2_combo = r2
                            best_name_combo = name
                            best_params_combo = params
                            best_func_combo = func_info['func']
                            best_equation_combo = func_info['equation'](params)
                    except:
                        pass
                
                # Skip if no fit was successful
                if best_func_combo is None:
                    continue
                
                # Calculate Pearson correlation coefficient
                correlation_combo = np.corrcoef(x_data_combo, y_data_combo)[0, 1]
                
                # Create regression line with best fit
                x_line_combo = np.linspace(x_data_combo.min(), x_data_combo.max(), 100)
                y_line_combo = best_func_combo(x_line_combo, *best_params_combo)
                
                # Create filename for this filter
                filter_clean_name = f"filter_{float(filter_val)}"
                
                # Plot scatter with regression line
                fig = plt.figure(figsize=(12, 9))
                plt.scatter(x_data_combo, y_data_combo, alpha=0.6, label='Data')
                plt.plot(x_line_combo, y_line_combo, 'r-', linewidth=2, label=f'Best fit ({best_name_combo})\n{best_equation_combo}\nR² = {best_r2_combo:.6f}')
                plt.xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
                plt.ylabel('Magical Variable (Complexity × Param Count)')
                plt.title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Magical Variable\nType: {type_str}, Filter: {float(filter_val)} sigma')
                plt.legend(loc='best', bbox_to_anchor=(0.48, -0.18), ncol=1, frameon=True, fancybox=True, shadow=True, fontsize=8)
                
                plt.savefig(os.path.join(type_folder, f'{filter_clean_name}_regression.png'))
                plt.close()
                
                # Save statistics to text file
                with open(os.path.join(type_folder, f'{filter_clean_name}_stats.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"{bench_name} VS MAGICAL VARIABLE\n")
                    f.write(f"Type: {type_str}\n")
                    f.write(f"Filter: {float(filter_val)} sigma\n")
                    f.write("=" * 70 + "\n\n")
                    
                    f.write("REGRESSION:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Best fit function: {best_name_combo}\n")
                    f.write(f"Expression: {best_equation_combo}\n")
                    f.write(f"R² score: {best_r2_combo:.16f}\n")
                    f.write(f"Pearson correlation: {correlation_combo:.16f}\n")
                    f.write(f"Parameters: {best_params_combo}\n\n")
                    
                    f.write("DATA STATISTICS:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Number of data points: {len(x_data_combo)}\n")
                    f.write(f"Magical variable mean: {y_data_combo.mean():.16f}\n")
                    f.write(f"Magical variable std: {y_data_combo.std():.16f}\n")
                    f.write(f"Benchmark score mean: {x_data_combo.mean():.16f}\n")
                    f.write(f"Benchmark score std: {x_data_combo.std():.16f}\n")
                
                # Store results for summary
                all_detailed_magical_results.append({
                    'benchmark': bench_name,
                    'type': type_str,
                    'filter': float(filter_val),
                    'best_function': best_name_combo,
                    'r2_score': best_r2_combo,
                    'pearson_correlation': correlation_combo,
                    'abs_correlation': abs(correlation_combo),
                    'data_points': len(x_data_combo),
                    'equation': best_equation_combo,
                    'magical_var_mean': y_data_combo.mean(),
                    'magical_var_std': y_data_combo.std(),
                    'benchmark_mean': x_data_combo.mean(),
                    'benchmark_std': x_data_combo.std()
                })
                
                combinations_processed += 1
        
        print(f"  Processed {combinations_processed} combinations for {bench_name}")

    # Convert results to DataFrame
    magical_results_df = pd.DataFrame(all_detailed_magical_results)

    # Save complete results sorted by absolute correlation (descending)
    magical_results_df_sorted = magical_results_df.sort_values('abs_correlation', ascending=False)
    magical_results_df_sorted.to_csv(os.path.join(detailed_magical_benchmarks_folder, 'all_combinations_by_correlation.csv'), 
                            index=False, encoding='utf-8')

    # For each benchmark, create individual CSV files
    print("\n" + "="*70)
    print("Creating CSV files for each benchmark...")
    print("="*70)

    for bench_name in BENCH_ROWS_NAMES:
        bench_results = magical_results_df[magical_results_df['benchmark'] == bench_name]
        
        if len(bench_results) == 0:
            continue
        
        # Sort by absolute correlation (descending)
        bench_results_sorted = bench_results.sort_values('abs_correlation', ascending=False)
        
        # Get benchmark folder
        bench_clean_name = bench_name.replace('BENCH-', '').replace('_', '-').lower()
        bench_main_folder = os.path.join(detailed_magical_benchmarks_folder, bench_clean_name)
        
        # Save all combinations for this benchmark (sorted by correlation)
        bench_results_sorted.to_csv(os.path.join(bench_main_folder, 'all_combinations.csv'), 
                                    index=False, encoding='utf-8')
        
        # Save top 20 combinations (or all if less than 20)
        top_n = min(20, len(bench_results_sorted))
        top_results = bench_results_sorted.head(top_n)
        top_results.to_csv(os.path.join(bench_main_folder, 'top_20_combinations.csv'), 
                        index=False, encoding='utf-8')
        
        # Create top 20 visualization for this benchmark
        if len(top_results) > 0:
            fig = plt.figure(figsize=(14, max(8, top_n * 0.4)))
            
            # Create labels
            labels = [f"{row['type'][:25]}, Filter={int(row['filter'])}" 
                    for _, row in top_results.iterrows()]
            correlations = top_results['pearson_correlation'].values
            
            colors = ['green' if c > 0 else 'red' for c in correlations]
            
            plt.barh(range(len(labels)), correlations, color=colors, alpha=0.7, edgecolor='black')
            plt.yticks(range(len(labels)), labels, fontsize=9)
            plt.xlabel('Pearson Correlation Coefficient')
            plt.ylabel('Type - Filter Combination')
            plt.title(f'Top {top_n} Combinations by Absolute Pearson Correlation\n{bench_name.replace("BENCH-", "").replace("_", " ")} vs Magical Variable')
            plt.grid(axis='x', alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            
            # Add value labels on bars
            for i, corr in enumerate(correlations):
                plt.text(corr, i, f' {corr:.4f}', va='center', fontsize=8, 
                        ha='left' if corr > 0 else 'right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(bench_main_folder, 'top_20_correlations.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Created CSV files and plot for {bench_name} ({len(bench_results_sorted)} combinations, top {top_n} saved)")

    # Create a pivot summary showing best correlation for each benchmark-type-filter combination
    magical_pivot_summary = magical_results_df.pivot_table(
        values='abs_correlation',
        index=['benchmark', 'type'],
        columns='filter',
        aggfunc='max'
    )
    magical_pivot_summary.to_csv(os.path.join(detailed_magical_benchmarks_folder, 'correlation_matrix.csv'), encoding='utf-8')

    # Create summary text file with overall statistics
    with open(os.path.join(detailed_magical_benchmarks_folder, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("DETAILED MAGICAL VARIABLE ANALYSIS - SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Total combinations analyzed: {len(magical_results_df)}\n")
        f.write(f"Benchmarks analyzed: {magical_results_df['benchmark'].nunique()}\n")
        f.write(f"Types analyzed: {magical_results_df['type'].nunique()}\n")
        f.write(f"Filters analyzed: {sorted(magical_results_df['filter'].unique())}\n\n")
        
        f.write("TOP 20 COMBINATIONS BY ABSOLUTE PEARSON CORRELATION:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Rank':<6} {'Benchmark':<30} {'Type':<20} {'Filter':<8} {'Correlation':<15} {'R²':<15} {'N':<8}\n")
        f.write("-" * 100 + "\n")
        
        top_20 = magical_results_df_sorted.head(20)
        for idx, (_, row) in enumerate(top_20.iterrows(), 1):
            bench_short = row['benchmark'].replace('BENCH-', '')
            f.write(f"{idx:<6} {bench_short:<30} {row['type']:<20} {row['filter']:<8} "
                    f"{row['pearson_correlation']:<15.8f} {row['r2_score']:<15.8f} {row['data_points']:<8}\n")
        
        f.write("\n\n")
        f.write("BEST COMBINATION FOR EACH BENCHMARK:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Benchmark':<30} {'Type':<20} {'Filter':<8} {'Correlation':<15} {'R²':<15} {'Function':<15}\n")
        f.write("-" * 100 + "\n")
        
        for bench_name in BENCH_ROWS_NAMES:
            bench_best = magical_results_df[magical_results_df['benchmark'] == bench_name].sort_values('abs_correlation', ascending=False)
            if len(bench_best) > 0:
                best = bench_best.iloc[0]
                bench_short = best['benchmark'].replace('BENCH-', '')
                f.write(f"{bench_short:<30} {best['type']:<20} {best['filter']:<8} "
                        f"{best['pearson_correlation']:<15.8f} {best['r2_score']:<15.8f} {best['best_function']:<15}\n")
        
        f.write("\n\n")
        f.write("STATISTICS BY BENCHMARK:\n")
        f.write("-" * 100 + "\n")
        
        for bench_name in BENCH_ROWS_NAMES:
            bench_data = magical_results_df[magical_results_df['benchmark'] == bench_name]
            if len(bench_data) > 0:
                f.write(f"\n{bench_name.replace('BENCH-', '')}:\n")
                f.write(f"  Combinations analyzed: {len(bench_data)}\n")
                f.write(f"  Mean correlation: {bench_data['abs_correlation'].mean():.8f}\n")
                f.write(f"  Max correlation: {bench_data['abs_correlation'].max():.8f}\n")
                f.write(f"  Min correlation: {bench_data['abs_correlation'].min():.8f}\n")
                f.write(f"  Mean R²: {bench_data['r2_score'].mean():.8f}\n")

    # Create visualization: Top 20 correlations
    if len(magical_results_df) > 0:
        fig = plt.figure(figsize=(16, 10))
        top_20_plot = magical_results_df_sorted.head(20).copy()
        
        # Create labels
        labels = [f"{row['benchmark'].replace('BENCH-', '')[:20]}\n{row['type'][:15]}, F={float(row['filter'])}" 
                for _, row in top_20_plot.iterrows()]
        correlations = top_20_plot['pearson_correlation'].values
        
        colors = ['green' if c > 0 else 'red' for c in correlations]
        
        plt.barh(range(len(labels)), correlations, color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(range(len(labels)), labels, fontsize=8)
        plt.xlabel('Pearson Correlation Coefficient')
        plt.ylabel('Benchmark - Type - Filter Combination')
        plt.title('Top 20 Combinations by Absolute Pearson Correlation\n(Benchmark vs Magical Variable: Complexity × Param Count)')
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars
        for i, corr in enumerate(correlations):
            plt.text(corr, i, f' {corr:.4f}', va='center', fontsize=7, 
                    ha='left' if corr > 0 else 'right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(detailed_magical_benchmarks_folder, 'top_20_correlations.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "="*70)
    print(f"Detailed magical variable analysis complete!")
    print(f"Total combinations analyzed: {len(magical_results_df)}")
    print(f"Results saved to: {detailed_magical_benchmarks_folder}")
    print("="*70)
