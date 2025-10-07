import os
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'appended_csvs'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'info_results'))

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

FILTER_DEFAULT_REPLACE = 0  # Replace filter "0 sigma" with this value

# Load appended_benchmarks.csv
appended_benchmarks_path = os.path.join(INPUT_FOLDER, 'appended_benchmarks.csv')
appended_benchmarks_df = pd.read_csv(appended_benchmarks_path)

# Split rows in types at "-" to a list
appended_benchmarks_df['types'] = appended_benchmarks_df['types'].apply(lambda x: x.split('-') if pd.notnull(x) else [])

# Parse filter column to an int "0 sigma" to 0
def parse_filter(value):
    if pd.isnull(value):
        return None
    try:
        return int(value.split()[0])
    except (ValueError, IndexError):
        return None
appended_benchmarks_df['filter'] = appended_benchmarks_df['filter'].apply(parse_filter)

# Change filter 0
appended_benchmarks_df['filter'] = appended_benchmarks_df['filter'].replace(0, FILTER_DEFAULT_REPLACE)

# Get rows in header that start with "BENCH"
BENCH_ROWS_NAMES = appended_benchmarks_df.columns[appended_benchmarks_df.columns.str.startswith('BENCH')].tolist()

# File structure
# model,types,filter,count,min,max,mean,std,bin_count,shannon_entropy,desequilibrium,complexity,BENCH-OPEN_LLM_AVERAGE,BENCH-LMARENA_SCORE,BENCH-MMLU_5,BENCH-MMLU_PRO_5

# ================================================================================

# --------------------------------------------------------------
# Plot graphs: COMPLEXITY vs FILTER

# Remove NaN values and filter = 0 for regression
mask = ~(appended_benchmarks_df['filter'].isna() | appended_benchmarks_df['complexity'].isna()) & (appended_benchmarks_df['filter'] != FILTER_DEFAULT_REPLACE)
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
        'cubic': {
            'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
            'equation': lambda params: f'y = {params[0]:.16f}x³ + {params[1]:.16f}x² + {params[2]:.16f}x + {params[3]:.16f}',
            'initial_guess': [1, 1, 1, 1]
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

# Get maximum complexity for each filter (excluding filter = 0)
max_complexity_by_filter = appended_benchmarks_df[appended_benchmarks_df['filter'] != FILTER_DEFAULT_REPLACE].groupby('filter')['complexity'].max().reset_index()
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
    'cubic': {
        'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
        'equation': lambda params: f'y = {params[0]:.16f}x³ + {params[1]:.16f}x² + {params[2]:.16f}x + {params[3]:.16f}',
        'initial_guess': [1, 1, 1, 1]
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
filter_complexity_folder = os.path.join(OUTPUT_FOLDER, 'filter_vs_complexity')
if not os.path.exists(filter_complexity_folder):
    os.makedirs(filter_complexity_folder)

# Plot scatter with regression lines
fig = plt.figure(figsize=(12, 9))
# Filter out filter = 0 from the scatter plot
filtered_df = appended_benchmarks_df[appended_benchmarks_df['filter'] != FILTER_DEFAULT_REPLACE]
plt.scatter(filtered_df['filter'], filtered_df['complexity'], alpha=0.6, label='Data')
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Best fit all data ({best_name})')
plt.scatter(x_max, y_max, color='green', s=100, alpha=0.8, marker='D', label='Maximum values', zorder=5)
plt.plot(x_line_max, y_line_max, 'g--', linewidth=2, label=f'Best fit max ({best_name_max})')
plt.xlabel('Filter')
plt.ylabel('Complexity')
plt.title('Filter vs Complexity')
plt.legend(loc='lower center', bbox_to_anchor=(0.48, -0.19), ncol=2, frameon=True, fancybox=True, shadow=True)
plt.subplots_adjust(bottom=0.28)
plt.figtext(0.5, 0.14, f'All data: {best_equation}     R² = {best_r2:.16f}', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.figtext(0.5, 0.11, f'Max values: {best_equation_max}     R² = {best_r2_max:.16f}', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
plt.savefig(os.path.join(filter_complexity_folder, 'regression.png'))
plt.close()

# Get average complexity for each filter (excluding filter = 0)
avg_complexity_by_filter = appended_benchmarks_df[appended_benchmarks_df['filter'] != FILTER_DEFAULT_REPLACE].groupby('filter')['complexity'].mean().reset_index()
x_avg = avg_complexity_by_filter['filter'].values
y_avg = avg_complexity_by_filter['complexity'].values

# Plot bar graph for average complexity
fig = plt.figure(figsize=(12, 9))
plt.bar(x_avg, y_avg, color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Filter')
plt.ylabel('Average Complexity')
plt.title('Average Complexity by Filter Value')
plt.grid(axis='y', alpha=0.3)
# Add value labels on top of bars
for i, (x, y) in enumerate(zip(x_avg, y_avg)):
    plt.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=9)
plt.savefig(os.path.join(filter_complexity_folder, 'average_complexity_bar.png'))
plt.close()

# Plot bar graph for maximum complexity
fig = plt.figure(figsize=(12, 9))
plt.bar(x_max, y_max, color='darkgreen', alpha=0.7, edgecolor='black')
plt.xlabel('Filter')
plt.ylabel('Maximum Complexity')
plt.title('Maximum Complexity by Filter Value')
plt.grid(axis='y', alpha=0.3)
# Add value labels on top of bars
for i, (x, y) in enumerate(zip(x_max, y_max)):
    plt.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=9)
plt.savefig(os.path.join(filter_complexity_folder, 'maximum_complexity_bar.png'))
plt.close()

# Save regression information to text file
with open(os.path.join(filter_complexity_folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
    f.write("FILTER VS COMPLEXITY - REGRESSION ANALYSIS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("ALL DATA REGRESSION:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Best fit function: {best_name}\n")
    f.write(f"Equation: {best_equation}\n")
    f.write(f"R² score: {best_r2:.16f}\n")
    f.write(f"Parameters: {best_params}\n\n")
    
    f.write("MAXIMUM VALUES REGRESSION:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Best fit function: {best_name_max}\n")
    f.write(f"Equation: {best_equation_max}\n")
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
# Plot graphs: COMPLEXITY vs TYPES

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
types_complexity_folder = os.path.join(OUTPUT_FOLDER, 'types_vs_complexity')
if not os.path.exists(types_complexity_folder):
    os.makedirs(types_complexity_folder)

# Plot bar graph for average complexity by types
fig = plt.figure(figsize=(14, 10))
plt.barh(range(len(x_types_avg)), y_types_avg, color='steelblue', alpha=0.7, edgecolor='black')
plt.yticks(range(len(x_types_avg)), x_types_avg, fontsize=8)
plt.xlabel('Average Complexity')
plt.ylabel('Types')
plt.title('Average Complexity by Type Combination')
plt.grid(axis='x', alpha=0.3)
# Add value labels on bars
for i, (x, y) in enumerate(zip(x_types_avg, y_types_avg)):
    plt.text(y, i, f' {y:.4f}', va='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(types_complexity_folder, 'average_complexity_bar.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot bar graph for maximum complexity by types
fig = plt.figure(figsize=(14, 10))
plt.barh(range(len(x_types_max)), y_types_max, color='darkgreen', alpha=0.7, edgecolor='black')
plt.yticks(range(len(x_types_max)), x_types_max, fontsize=8)
plt.xlabel('Maximum Complexity')
plt.ylabel('Types')
plt.title('Maximum Complexity by Type Combination')
plt.grid(axis='x', alpha=0.3)
# Add value labels on bars
for i, (x, y) in enumerate(zip(x_types_max, y_types_max)):
    plt.text(y, i, f' {y:.4f}', va='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(types_complexity_folder, 'maximum_complexity_bar.png'), dpi=150, bbox_inches='tight')
plt.close()

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
# Plot graphs: COMPLEXITY vs NUMBER OF PARAMS

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
    'cubic': {
        'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
        'equation': lambda params: f'y = {params[0]:.16f}x³ + {params[1]:.16f}x² + {params[2]:.16f}x + {params[3]:.16f}',
        'initial_guess': [1, 1, 1, 1]
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
params_complexity_folder = os.path.join(OUTPUT_FOLDER, 'params_vs_complexity')
if not os.path.exists(params_complexity_folder):
    os.makedirs(params_complexity_folder)

# Plot scatter with regression line
fig = plt.figure(figsize=(12, 9))
plt.scatter(x_data_params, y_data_params, alpha=0.6, label='Data')
plt.plot(x_line_params, y_line_params, 'r-', linewidth=2, label=f'Best fit ({best_name_params})')
plt.xlabel('Number of Parameters')
plt.ylabel('Complexity')
plt.title('Number of Parameters vs Complexity')
plt.legend(loc='lower center', bbox_to_anchor=(0.48, -0.18), ncol=1, frameon=True, fancybox=True, shadow=True)
plt.subplots_adjust(bottom=0.22)
plt.figtext(0.5, 0.07, f'{best_equation_params}     R² = {best_r2_params:.16f}', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.savefig(os.path.join(params_complexity_folder, 'regression.png'))
plt.close()

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

# Create bins for parameter count and calculate average complexity for each bin
param_bins = np.linspace(x_data_params.min(), x_data_params.max(), n_bins_fd + 1)
bin_centers = []
avg_complexity_per_bin = []
bin_counts = []

for i in range(n_bins_fd):
    mask_bin = (x_data_params >= param_bins[i]) & (x_data_params < param_bins[i + 1])
    if i == n_bins_fd - 1:  # Include the last edge
        mask_bin = (x_data_params >= param_bins[i]) & (x_data_params <= param_bins[i + 1])
    
    if mask_bin.any():
        bin_centers.append((param_bins[i] + param_bins[i + 1]) / 2)
        avg_complexity_per_bin.append(y_data_params[mask_bin].mean())
        bin_counts.append(mask_bin.sum())

# Plot histogram: average complexity per parameter count interval
fig = plt.figure(figsize=(14, 9))
bars = plt.bar(bin_centers, avg_complexity_per_bin, width=(param_bins[1] - param_bins[0]) * 0.9, 
               color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Number of Parameters')
plt.ylabel('Average Complexity')
plt.title(f'Average Complexity by Parameter Count Range (Freedman-Diaconis: {n_bins_fd} bins)')
plt.grid(axis='y', alpha=0.3)

# Add overall statistics text
overall_avg = np.average(avg_complexity_per_bin, weights=bin_counts)
stats_text = f'Overall Avg: {overall_avg:.4f}\nTotal samples: {sum(bin_counts)}'
plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(params_complexity_folder, 'average_complexity_histogram.png'))
plt.close()

# Save regression information to text file
with open(os.path.join(params_complexity_folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
    f.write("NUMBER OF PARAMETERS VS COMPLEXITY - REGRESSION ANALYSIS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("REGRESSION:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Best fit function: {best_name_params}\n")
    f.write(f"Equation: {best_equation_params}\n")
    f.write(f"R² score: {best_r2_params:.16f}\n")
    f.write(f"Parameters: {best_params_params}\n\n")
    
    f.write("HISTOGRAM INFORMATION:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Method: Freedman-Diaconis rule\n")
    f.write(f"Number of bins: {n_bins_fd}\n")
    f.write(f"Bin width: {bin_width_params:.2f} parameters\n")
    f.write(f"IQR: {iqr_params:.2f}\n")

# Save average complexity per bin to text file
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
# Plot graphs: COMPLEXITY vs NUMBER OF BINS

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
    'cubic': {
        'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
        'equation': lambda params: f'y = {params[0]:.16f}x³ + {params[1]:.16f}x² + {params[2]:.16f}x + {params[3]:.16f}',
        'initial_guess': [1, 1, 1, 1]
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
bins_complexity_folder = os.path.join(OUTPUT_FOLDER, 'bins_vs_complexity')
if not os.path.exists(bins_complexity_folder):
    os.makedirs(bins_complexity_folder)

# Plot scatter with regression line
fig = plt.figure(figsize=(12, 9))
plt.scatter(x_data_bins, y_data_bins, alpha=0.6, label='Data')
plt.plot(x_line_bins, y_line_bins, 'r-', linewidth=2, label=f'Best fit ({best_name_bins})')
plt.xlabel('Number of Bins')
plt.ylabel('Complexity')
plt.title('Number of Bins vs Complexity')
plt.legend(loc='lower center', bbox_to_anchor=(0.48, -0.18), ncol=1, frameon=True, fancybox=True, shadow=True)
plt.subplots_adjust(bottom=0.22)
plt.figtext(0.5, 0.07, f'{best_equation_bins}     R² = {best_r2_bins:.16f}', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.savefig(os.path.join(bins_complexity_folder, 'regression.png'))
plt.close()

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

# Create bins for bin count and calculate average complexity for each bin
bin_bins = np.linspace(x_data_bins.min(), x_data_bins.max(), n_bins_fd_bins + 1)
bin_centers_bins = []
avg_complexity_per_bin_bins = []
bin_counts_bins = []

for i in range(n_bins_fd_bins):
    mask_bin = (x_data_bins >= bin_bins[i]) & (x_data_bins < bin_bins[i + 1])
    if i == n_bins_fd_bins - 1:  # Include the last edge
        mask_bin = (x_data_bins >= bin_bins[i]) & (x_data_bins <= bin_bins[i + 1])
    
    if mask_bin.any():
        bin_centers_bins.append((bin_bins[i] + bin_bins[i + 1]) / 2)
        avg_complexity_per_bin_bins.append(y_data_bins[mask_bin].mean())
        bin_counts_bins.append(mask_bin.sum())

# Plot histogram: average complexity per bin count interval
fig = plt.figure(figsize=(14, 9))
bars = plt.bar(bin_centers_bins, avg_complexity_per_bin_bins, width=(bin_bins[1] - bin_bins[0]) * 0.9, 
               color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Number of Bins')
plt.ylabel('Average Complexity')
plt.title(f'Average Complexity by Bin Count Range (Freedman-Diaconis: {n_bins_fd_bins} bins)')
plt.grid(axis='y', alpha=0.3)

# Add overall statistics text
overall_avg_bins = np.average(avg_complexity_per_bin_bins, weights=bin_counts_bins)
stats_text = f'Overall Avg: {overall_avg_bins:.4f}\nTotal samples: {sum(bin_counts_bins)}'
plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(bins_complexity_folder, 'average_complexity_histogram.png'))
plt.close()

# Save regression information to text file
with open(os.path.join(bins_complexity_folder, 'regression_info.txt'), 'w', encoding='utf-8') as f:
    f.write("NUMBER OF BINS VS COMPLEXITY - REGRESSION ANALYSIS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("REGRESSION:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Best fit function: {best_name_bins}\n")
    f.write(f"Equation: {best_equation_bins}\n")
    f.write(f"R² score: {best_r2_bins:.16f}\n")
    f.write(f"Parameters: {best_params_bins}\n\n")
    
    f.write("HISTOGRAM INFORMATION:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Method: Freedman-Diaconis rule\n")
    f.write(f"Number of bins: {n_bins_fd_bins}\n")
    f.write(f"Bin width: {bin_width_bins:.2f} bins\n")
    f.write(f"IQR: {iqr_bins:.2f}\n")

# Save average complexity per bin to text file
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
# Plot graphs: FOR EACH BENCHMARK - COMPLEXITY vs BENCHMARK

if False:

    # Create output folder for benchmarks vs complexity
    benchmarks_complexity_folder = os.path.join(OUTPUT_FOLDER, 'benchmarks_vs_complexity')
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
            'cubic': {
                'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
                'equation': lambda params: f'y = {params[0]:.16f}x³ + {params[1]:.16f}x² + {params[2]:.16f}x + {params[3]:.16f}',
                'initial_guess': [1, 1, 1, 1]
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
        
        # Create subfolder for this benchmark
        bench_clean_name = bench_name.replace('BENCH-', '').replace('_', '-').lower()
        bench_folder = os.path.join(benchmarks_complexity_folder, bench_clean_name)
        if not os.path.exists(bench_folder):
            os.makedirs(bench_folder)
        
        # Plot scatter with regression line
        fig = plt.figure(figsize=(12, 9))
        plt.scatter(x_data_bench, y_data_bench, alpha=0.6, label='Data')
        plt.plot(x_line_bench, y_line_bench, 'r-', linewidth=2, label=f'Best fit ({best_name_bench})')
        plt.xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
        plt.ylabel('Complexity')
        plt.title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Complexity')
        plt.legend(loc='lower center', bbox_to_anchor=(0.48, -0.18), ncol=1, frameon=True, fancybox=True, shadow=True)
        plt.subplots_adjust(bottom=0.22)
        plt.figtext(0.5, 0.07, f'{best_equation_bench}     R² = {best_r2_bench:.16f}', 
                    ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
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
            f.write(f"Equation: {best_equation_bench}\n")
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
            'correlation': correlation,
            'equation': best_equation_bench,
            'data_points': len(x_data_bench)
        }
        
        print(f"  Completed {bench_name}: R² = {best_r2_bench:.6f}, Correlation = {correlation:.6f}")

    # Create summary file with all benchmark results
    with open(os.path.join(benchmarks_complexity_folder, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("BENCHMARKS VS COMPLEXITY - SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Benchmark':<35} {'Best Fit':<15} {'R² Score':<15} {'Correlation':<15} {'Data Points':<15}\n")
        f.write("-" * 100 + "\n")
        
        # Sort by R² score (descending)
        sorted_results = sorted(all_benchmark_results.items(), key=lambda x: x[1]['r2_score'], reverse=True)
        
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
        bench_names_short = [name.replace('BENCH-', '').replace('_', ' ') for name in all_benchmark_results.keys()]
        r2_scores = [results['r2_score'] for results in all_benchmark_results.values()]
        
        # Sort by R² score
        sorted_indices = np.argsort(r2_scores)[::-1]
        bench_names_short = [bench_names_short[i] for i in sorted_indices]
        r2_scores = [r2_scores[i] for i in sorted_indices]
        
        plt.barh(range(len(bench_names_short)), r2_scores, color='steelblue', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(bench_names_short)), bench_names_short, fontsize=9)
        plt.xlabel('R² Score')
        plt.ylabel('Benchmark')
        plt.title('Regression Quality: R² Scores for Each Benchmark vs Complexity')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (name, score) in enumerate(zip(bench_names_short, r2_scores)):
            plt.text(score, i, f' {score:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(benchmarks_complexity_folder, 'r2_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create comparison plot: Correlation coefficients for all benchmarks
        fig = plt.figure(figsize=(14, 8))
        correlations = [all_benchmark_results[list(all_benchmark_results.keys())[i]]['correlation'] 
                    for i in sorted_indices]
        
        colors = ['green' if c > 0 else 'red' for c in correlations]
        plt.barh(range(len(bench_names_short)), correlations, color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(range(len(bench_names_short)), bench_names_short, fontsize=9)
        plt.xlabel('Pearson Correlation Coefficient')
        plt.ylabel('Benchmark')
        plt.title('Correlation: Benchmark Score vs Complexity')
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars
        for i, (name, corr) in enumerate(zip(bench_names_short, correlations)):
            plt.text(corr, i, f' {corr:.4f}', va='center', fontsize=8, 
                    ha='left' if corr > 0 else 'right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(benchmarks_complexity_folder, 'correlation_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nCompleted benchmark analysis. Processed {len(all_benchmark_results)} benchmarks.")

    # --------------------------------------------------------------
    # Plot graphs: FOR EACH BENCHMARK: FOR EACH TYPE: FOR EACH FILTER: - COMPLEXITY vs BENCHMARK

    print("\n" + "="*70)
    print("Starting detailed benchmark analysis by type and filter...")
    print("="*70)

    # Create main output folder
    detailed_benchmarks_folder = os.path.join(OUTPUT_FOLDER, 'detailed_benchmarks_analysis')
    if not os.path.exists(detailed_benchmarks_folder):
        os.makedirs(detailed_benchmarks_folder)

    # Dictionary to store all detailed results
    all_detailed_results = []

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
                    'cubic': {
                        'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
                        'equation': lambda params: f'y = {params[0]:.16f}x³ + {params[1]:.16f}x² + {params[2]:.16f}x + {params[3]:.16f}',
                        'initial_guess': [1, 1, 1, 1]
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
                filter_clean_name = f"filter_{int(filter_val)}"
                
                # Plot scatter with regression line
                fig = plt.figure(figsize=(12, 9))
                plt.scatter(x_data_combo, y_data_combo, alpha=0.6, label='Data')
                plt.plot(x_line_combo, y_line_combo, 'r-', linewidth=2, label=f'Best fit ({best_name_combo})')
                plt.xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
                plt.ylabel('Complexity')
                plt.title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Complexity\nType: {type_str}, Filter: {int(filter_val)} sigma')
                plt.legend(loc='lower center', bbox_to_anchor=(0.48, -0.18), ncol=1, frameon=True, fancybox=True, shadow=True)
                plt.subplots_adjust(bottom=0.28)
                
                # Add equation and statistics
                stats_text = f'{best_equation_combo}\nR² = {best_r2_combo:.16f}\nCorrelation = {correlation_combo:.16f}\nN = {len(x_data_combo)}'
                plt.figtext(0.5, 0.07, stats_text, ha='center', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.savefig(os.path.join(type_folder, f'{filter_clean_name}_regression.png'))
                plt.close()
                
                # Save statistics to text file
                with open(os.path.join(type_folder, f'{filter_clean_name}_stats.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"{bench_name} VS COMPLEXITY\n")
                    f.write(f"Type: {type_str}\n")
                    f.write(f"Filter: {int(filter_val)} sigma\n")
                    f.write("=" * 70 + "\n\n")
                    
                    f.write("REGRESSION:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Best fit function: {best_name_combo}\n")
                    f.write(f"Equation: {best_equation_combo}\n")
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
                    'filter': int(filter_val),
                    'best_function': best_name_combo,
                    'r2_score': best_r2_combo,
                    'pearson_correlation': correlation_combo,
                    'abs_correlation': abs(correlation_combo),
                    'data_points': len(x_data_combo),
                    'equation': best_equation_combo,
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
            plt.title(f'Top {top_n} Combinations by Absolute Pearson Correlation\n{bench_name.replace("BENCH-", "").replace("_", " ")} vs Complexity')
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
        
        # Create labels
        labels = [f"{row['benchmark'].replace('BENCH-', '')[:20]}\n{row['type'][:15]}, F={int(row['filter'])}" 
                for _, row in top_20_plot.iterrows()]
        correlations = top_20_plot['pearson_correlation'].values
        
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
            plt.text(corr, i, f' {corr:.4f}', va='center', fontsize=7, 
                    ha='left' if corr > 0 else 'right')
        
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
# Plot graphs: FOR EACH BENCHMARK - PARAM COUNT vs BENCHMARK

if False:

    print("\n" + "="*70)
    print("Starting PARAM COUNT vs BENCHMARK analysis...")
    print("="*70)

    # Create main output folder
    param_benchmarks_folder = os.path.join(OUTPUT_FOLDER, 'param_count_vs_benchmarks')
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
            'cubic': {
                'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
                'equation': lambda params: f'y = {params[0]:.16f}x³ + {params[1]:.16f}x² + {params[2]:.16f}x + {params[3]:.16f}',
                'initial_guess': [1, 1, 1, 1]
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
        bench_folder = os.path.join(param_benchmarks_folder, bench_clean_name)
        if not os.path.exists(bench_folder):
            os.makedirs(bench_folder)
        
        # Plot scatter with regression line
        fig = plt.figure(figsize=(12, 9))
        plt.scatter(x_data_bench, y_data_bench, alpha=0.6, label='Data')
        plt.plot(x_line_bench, y_line_bench, 'r-', linewidth=2, label=f'Best fit ({best_name_bench})')
        plt.xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
        plt.ylabel('Parameter Count')
        plt.title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Parameter Count')
        plt.legend(loc='lower center', bbox_to_anchor=(0.48, -0.18), ncol=1, frameon=True, fancybox=True, shadow=True)
        plt.subplots_adjust(bottom=0.22)
        plt.figtext(0.5, 0.07, f'{best_equation_bench}     R² = {best_r2_bench:.16f}', 
                    ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
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
            f.write(f"Equation: {best_equation_bench}\n")
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
            'correlation': correlation,
            'equation': best_equation_bench,
            'data_points': len(x_data_bench)
        }
        
        print(f"  Completed {bench_name}: R² = {best_r2_bench:.6f}, Correlation = {correlation:.6f}")

    # Create summary file with all benchmark results
    with open(os.path.join(param_benchmarks_folder, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("BENCHMARKS VS PARAMETER COUNT - SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Benchmark':<35} {'Best Fit':<15} {'R² Score':<15} {'Correlation':<15} {'Data Points':<15}\n")
        f.write("-" * 100 + "\n")
        
        # Sort by R² score (descending)
        sorted_results = sorted(all_param_benchmark_results.items(), key=lambda x: x[1]['r2_score'], reverse=True)
        
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
        bench_names_short = [name.replace('BENCH-', '').replace('_', ' ') for name in all_param_benchmark_results.keys()]
        r2_scores = [results['r2_score'] for results in all_param_benchmark_results.values()]
        
        # Sort by R² score
        sorted_indices = np.argsort(r2_scores)[::-1]
        bench_names_short = [bench_names_short[i] for i in sorted_indices]
        r2_scores = [r2_scores[i] for i in sorted_indices]
        
        plt.barh(range(len(bench_names_short)), r2_scores, color='steelblue', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(bench_names_short)), bench_names_short, fontsize=9)
        plt.xlabel('R² Score')
        plt.ylabel('Benchmark')
        plt.title('Regression Quality: R² Scores for Each Benchmark vs Parameter Count')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (name, score) in enumerate(zip(bench_names_short, r2_scores)):
            plt.text(score, i, f' {score:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(param_benchmarks_folder, 'r2_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create comparison plot: Correlation coefficients for all benchmarks
        fig = plt.figure(figsize=(14, 8))
        correlations = [all_param_benchmark_results[list(all_param_benchmark_results.keys())[i]]['correlation'] 
                    for i in sorted_indices]
        
        colors = ['green' if c > 0 else 'red' for c in correlations]
        plt.barh(range(len(bench_names_short)), correlations, color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(range(len(bench_names_short)), bench_names_short, fontsize=9)
        plt.xlabel('Pearson Correlation Coefficient')
        plt.ylabel('Benchmark')
        plt.title('Correlation: Benchmark Score vs Parameter Count')
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars
        for i, (name, corr) in enumerate(zip(bench_names_short, correlations)):
            plt.text(corr, i, f' {corr:.4f}', va='center', fontsize=8, 
                    ha='left' if corr > 0 else 'right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(param_benchmarks_folder, 'correlation_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nCompleted parameter count analysis. Processed {len(all_param_benchmark_results)} benchmarks.")

    # --------------------------------------------------------------
    # Plot graphs: FOR EACH BENCHMARK: FOR EACH TYPE: FOR EACH FILTER: - PARAM COUNT vs BENCHMARK

    print("\n" + "="*70)
    print("Starting detailed PARAM COUNT analysis by type and filter...")
    print("="*70)

    # Create main output folder
    detailed_param_benchmarks_folder = os.path.join(OUTPUT_FOLDER, 'detailed_param_count_analysis')
    if not os.path.exists(detailed_param_benchmarks_folder):
        os.makedirs(detailed_param_benchmarks_folder)

    # Dictionary to store all detailed results
    all_detailed_param_results = []

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
                    'cubic': {
                        'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
                        'equation': lambda params: f'y = {params[0]:.16f}x³ + {params[1]:.16f}x² + {params[2]:.16f}x + {params[3]:.16f}',
                        'initial_guess': [1, 1, 1, 1]
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
                filter_clean_name = f"filter_{int(filter_val)}"
                
                # Plot scatter with regression line
                fig = plt.figure(figsize=(12, 9))
                plt.scatter(x_data_combo, y_data_combo, alpha=0.6, label='Data')
                plt.plot(x_line_combo, y_line_combo, 'r-', linewidth=2, label=f'Best fit ({best_name_combo})')
                plt.xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
                plt.ylabel('Parameter Count')
                plt.title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Parameter Count\nType: {type_str}, Filter: {int(filter_val)} sigma')
                plt.legend(loc='lower center', bbox_to_anchor=(0.48, -0.18), ncol=1, frameon=True, fancybox=True, shadow=True)
                plt.subplots_adjust(bottom=0.28)
                
                # Add equation and statistics
                stats_text = f'{best_equation_combo}\nR² = {best_r2_combo:.16f}\nCorrelation = {correlation_combo:.16f}\nN = {len(x_data_combo)}'
                plt.figtext(0.5, 0.07, stats_text, ha='center', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.savefig(os.path.join(type_folder, f'{filter_clean_name}_regression.png'))
                plt.close()
                
                # Save statistics to text file
                with open(os.path.join(type_folder, f'{filter_clean_name}_stats.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"{bench_name} VS PARAMETER COUNT\n")
                    f.write(f"Type: {type_str}\n")
                    f.write(f"Filter: {int(filter_val)} sigma\n")
                    f.write("=" * 70 + "\n\n")
                    
                    f.write("REGRESSION:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Best fit function: {best_name_combo}\n")
                    f.write(f"Equation: {best_equation_combo}\n")
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
                    'filter': int(filter_val),
                    'best_function': best_name_combo,
                    'r2_score': best_r2_combo,
                    'pearson_correlation': correlation_combo,
                    'abs_correlation': abs(correlation_combo),
                    'data_points': len(x_data_combo),
                    'equation': best_equation_combo,
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
            plt.title(f'Top {top_n} Combinations by Absolute Pearson Correlation\n{bench_name.replace("BENCH-", "").replace("_", " ")} vs Parameter Count')
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
        
        # Create labels
        labels = [f"{row['benchmark'].replace('BENCH-', '')[:20]}\n{row['type'][:15]}, F={int(row['filter'])}" 
                for _, row in top_20_plot.iterrows()]
        correlations = top_20_plot['pearson_correlation'].values
        
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
            plt.text(corr, i, f' {corr:.4f}', va='center', fontsize=7, 
                    ha='left' if corr > 0 else 'right')
        
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
# Plot graphs: MAGICAL VARIABLE EXPLORE

if True:

    print("\n" + "="*70)
    print("Starting EQUATION EXPLORATION (Complexity & Param Count) vs BENCHMARKS...")
    print("="*70)

    # Create main output folder
    equation_exploration_folder = os.path.join(OUTPUT_FOLDER, 'equation_exploration')
    if not os.path.exists(equation_exploration_folder):
        os.makedirs(equation_exploration_folder)

    # Define various equations to test
    # Theory: performance_roof = f(count), complexity = how close to roof
    # We'll test various formulations

    MIN_STOPPING_CORRELATION = 0.99
    VARIABLES_TO_USE = ["complexity", "count"] # ["complexity", "count"]
    VARIABLES_MANDATORY_TO_USE = []

    print("\n" + "="*80)
    print("EVOLUTIONARY EQUATION DISCOVERY MODE")
    print("="*80)
    
    import ga_evolution
    
    # Filter to only default filter for equation discovery
    appended_benchmarks_df = appended_benchmarks_df[appended_benchmarks_df['filter'] == FILTER_DEFAULT_REPLACE]
    appended_benchmarks_df = appended_benchmarks_df[appended_benchmarks_df['types'].str.len() == 3]
    appended_benchmarks_df = appended_benchmarks_df[~appended_benchmarks_df['types'].apply(lambda x: 'bias' in x if isinstance(x, list) else False)]
    print(appended_benchmarks_df)
    
    # Run the evolutionary algorithm
    top_equations, evolution_history = ga_evolution.run_evolution(
        appended_benchmarks_df=appended_benchmarks_df,
        bench_rows_names=BENCH_ROWS_NAMES,
        output_folder=OUTPUT_FOLDER,
        min_stopping_correlation=MIN_STOPPING_CORRELATION,
        population_size=500,          # Number of equations per generation
        max_generations=10000,         # Maximum generations
        top_n_to_keep=4,           # How many top equations to save
        elite_size=10,               # How many best equations to keep unchanged
        mandatory_vars=VARIABLES_MANDATORY_TO_USE,  # Variables that MUST be in every equation
        allowed_vars=VARIABLES_TO_USE,  # Variables that CAN be used (restricted set)
        simplicity_weight=0.01,      # Weight for simplicity (higher = prefer simpler equations)
        resume_from_checkpoint='auto',  # 'auto' to find latest, or specific file like 'top_equations_gen10.py'
        max_stagnation=100000000000,          # Stop after N generations without improvement (default: 15)
        adaptive_mutation=True,     # Increase mutation when stuck (default: True)
        diversity_injection_rate=0.4  # Add 20% random equations when stagnating (default: 0.2)
    )
    
    # Convert top equations to the format expected by the rest of the code
    EQUATIONS_TO_TEST = ga_evolution.equations_to_dict(top_equations)
    
    print("\n" + "="*80)
    print(f"Evolution complete! Using top {len(EQUATIONS_TO_TEST)} equations for analysis.")
    print("="*80)
        
    # Store results for all equations and benchmarks
    all_equation_results = []
    
    # Track failed fits for reporting
    failed_fits = {}  # {equation_id: reason}

    # STEP 1: Collect ALL benchmark data at once (for fitting shared parameters)
    print("\nCollecting data from all benchmarks for shared parameter fitting...")
    
    all_benchmarks_data = {}
    for bench_name in BENCH_ROWS_NAMES:
        # Get valid data for this benchmark
        mask_bench = ~(
            appended_benchmarks_df[bench_name].isna() | 
            appended_benchmarks_df['complexity'].isna() | 
            appended_benchmarks_df['count'].isna()
        )
        
        benchmark_values = appended_benchmarks_df.loc[mask_bench, bench_name].values
        complexity_values = appended_benchmarks_df.loc[mask_bench, 'complexity'].values
        count_values = appended_benchmarks_df.loc[mask_bench, 'count'].values
        
        # Skip if not enough data points
        if len(benchmark_values) < 10:
            print(f"  Skipping {bench_name}: insufficient data points ({len(benchmark_values)})")
            continue
        
        all_benchmarks_data[bench_name] = {
            'benchmark_values': benchmark_values,
            'complexity_values': complexity_values,
            'count_values': count_values
        }
        print(f"  {bench_name}: {len(benchmark_values)} data points")
    
    # Check if we have data from all benchmarks
    if len(all_benchmarks_data) < len(BENCH_ROWS_NAMES):
        print(f"\n⚠️  WARNING: Only {len(all_benchmarks_data)}/{len(BENCH_ROWS_NAMES)} benchmarks have sufficient data!")
    
    # Concatenate all benchmark data for shared parameter fitting
    all_benchmark_values = np.concatenate([data['benchmark_values'] for data in all_benchmarks_data.values()])
    all_complexity_values = np.concatenate([data['complexity_values'] for data in all_benchmarks_data.values()])
    all_count_values = np.concatenate([data['count_values'] for data in all_benchmarks_data.values()])
    
    print(f"\nTotal combined data points: {len(all_benchmark_values)}")
    
    # STEP 2: Fit each equation ONCE using ALL benchmark data combined
    print(f"\nFitting {len(EQUATIONS_TO_TEST)} equations with shared parameters across all benchmarks...")
    
    for eq_name, eq_info in EQUATIONS_TO_TEST.items():
        try:
            # Create wrapper function for curve_fit (takes x as single array)
            def fit_func(x, *params):
                complexity = x[0]
                count = x[1]
                return eq_info['func'](complexity, count, *params)
            
            # Prepare data for curve_fit (ALL benchmarks combined)
            x_data = np.vstack([all_complexity_values, all_count_values])
            
            # Fit the equation ONCE with all data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shared_params, _ = curve_fit(
                    fit_func, 
                    x_data, 
                    all_benchmark_values,
                    p0=eq_info['initial_guess'],
                    maxfev=5000000
                )
            
            # STEP 3: Calculate correlation for EACH benchmark using the SAME shared parameters
            benchmark_correlations = []
            benchmark_r2_scores = []
            all_benchmarks_valid = True  # Track if all benchmarks have valid predictions
            
            for bench_name, data in all_benchmarks_data.items():
                # Use shared parameters to predict this benchmark
                predictions = eq_info['func'](data['complexity_values'], data['count_values'], *shared_params)
                
                # Validate predictions
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    all_benchmarks_valid = False
                    break
                
                # Check if predictions have variance
                if np.std(predictions) < 1e-10:
                    all_benchmarks_valid = False
                    break
                
                # Calculate R² score
                ss_res = np.sum((data['benchmark_values'] - predictions)**2)
                ss_tot = np.sum((data['benchmark_values'] - np.mean(data['benchmark_values']))**2)
                r2 = 1 - (ss_res / ss_tot)
                
                # Calculate Pearson correlation
                correlation = np.corrcoef(data['benchmark_values'], predictions)[0, 1]
                
                # Validate correlation
                if np.isnan(correlation) or np.isinf(correlation):
                    all_benchmarks_valid = False
                    break
                
                benchmark_correlations.append(abs(correlation))
                benchmark_r2_scores.append(r2)
                
                # Store individual benchmark results
                all_equation_results.append({
                    'benchmark': bench_name,
                    'equation_id': eq_name,
                    'equation_name': eq_info['name'],
                    'r2_score': r2,
                    'pearson_correlation': correlation,
                    'abs_correlation': abs(correlation),
                    'data_points': len(data['benchmark_values']),
                    'params': shared_params.tolist(),  # SAME params for all benchmarks
                    'rmse': np.sqrt(np.mean((data['benchmark_values'] - predictions)**2))
                })
            
            # Only calculate average if ALL benchmarks are valid
            if not all_benchmarks_valid:
                raise ValueError("Invalid predictions for one or more benchmarks")
            
            # STEP 4: Calculate average correlation across all benchmarks
            avg_correlation = np.mean(benchmark_correlations)
            avg_r2 = np.mean(benchmark_r2_scores)
            
            print(f"  ✓ {eq_info['name'][:60]}: Avg|Corr|={avg_correlation:.6f}, Avg R²={avg_r2:.6f}")
            
        except Exception as e:
            # Skip if fitting fails
            failed_fits[eq_name] = f"Exception: {type(e).__name__}: {str(e)}"
            print(f"  ✗ {eq_info['name'][:60]}: FAILED ({type(e).__name__})")
            continue
    
    print(f"\nFitting complete! Tested {len(EQUATIONS_TO_TEST)} equations with shared parameters.")

    # Report equations that had fitting failures
    if len(failed_fits) > 0:
        print(f"\n⚠️  FITTING FAILURES DETECTED:")
        print(f"Total equations with failures: {len(failed_fits)}")
        
        # Show details for failed equations
        if len(failed_fits) > 0:
            print(f"\nEquations that failed ({len(failed_fits)}):")
            for eq_id, reason in list(failed_fits.items())[:10]:
                eq_name = EQUATIONS_TO_TEST[eq_id]['name'] if eq_id in EQUATIONS_TO_TEST else eq_id
                print(f"  • {eq_name[:70]}")
                print(f"     → {reason}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_equation_results)

    if len(results_df) == 0:
        print("No successful equation fits. Exiting equation exploration.")
    else:
        print(f"\nTotal successful fits: {len(results_df)}")
        
        # Count total benchmarks that were tested
        num_total_benchmarks = len(all_benchmarks_data)
        
        # Calculate average correlation for each equation across all benchmarks
        # NOTE: With shared parameters, each equation should have EXACTLY one entry per benchmark
        equation_avg_stats = results_df.groupby(['equation_id', 'equation_name']).agg({
            'abs_correlation': ['mean', 'std', 'min', 'max'],
            'r2_score': ['mean', 'std', 'min', 'max'],
            'data_points': 'sum',  # Total data points across all benchmarks
            'benchmark': 'nunique'  # Should be equal to num_total_benchmarks
        }).reset_index()
        
        equation_avg_stats.columns = ['equation_id', 'equation_name', 
                                       'avg_abs_corr', 'std_abs_corr', 'min_abs_corr', 'max_abs_corr',
                                       'avg_r2', 'std_r2', 'min_r2', 'max_r2',
                                       'total_data_points', 'num_benchmarks']
        
        # Verify all equations fitted all benchmarks (they should with shared parameters)
        equations_with_all_benchmarks = equation_avg_stats[equation_avg_stats['num_benchmarks'] == num_total_benchmarks].copy()
        equations_partial = equation_avg_stats[equation_avg_stats['num_benchmarks'] < num_total_benchmarks].copy()
        
        if len(equations_partial) > 0:
            print(f"\n⚠️  WARNING: {len(equations_partial)} equations don't have all benchmarks (this shouldn't happen with shared params!):")
            for _, row in equations_partial.iterrows():
                missing_count = num_total_benchmarks - int(row['num_benchmarks'])
                print(f"   • {row['equation_name'][:70]} - Has {int(row['num_benchmarks'])}/{num_total_benchmarks} benchmarks ({missing_count} missing)")
        
        # Check for duplicate equation names and keep only the best performing one
        duplicate_names = equations_with_all_benchmarks[equations_with_all_benchmarks.duplicated('equation_name', keep=False)]
        if len(duplicate_names) > 0:
            print(f"\n⚠️  WARNING: Found {len(duplicate_names)} equations with DUPLICATE NAMES!")
            print("Keeping only the best-performing version of each duplicate:")
            for name in duplicate_names['equation_name'].unique():
                dupes = duplicate_names[duplicate_names['equation_name'] == name]
                best_idx = dupes['avg_abs_corr'].idxmax()
                print(f"   • '{name}' appears {len(dupes)} times")
                print(f"     → IDs: {dupes['equation_id'].tolist()}")
                print(f"     → Keeping ID '{dupes.loc[best_idx, 'equation_id']}' (best avg correlation: {dupes.loc[best_idx, 'avg_abs_corr']:.6f})")
            
            # Remove duplicates, keeping the one with highest average correlation
            equations_with_all_benchmarks = equations_with_all_benchmarks.sort_values('avg_abs_corr', ascending=False).drop_duplicates('equation_name', keep='first')
            print(f"   → After deduplication: {len(equations_with_all_benchmarks)} unique equations\n")
        
        # Use only equations that fitted all benchmarks
        if len(equations_with_all_benchmarks) == 0:
            print("\n❌ ERROR: No equations successfully fitted with shared parameters!")
            print("Cannot generate ranking. Exiting equation exploration.")
            # Still save the results we have
            equation_avg_stats.sort_values('avg_abs_corr', ascending=False).to_csv(
                os.path.join(equation_exploration_folder, 'equation_ranking_by_avg_correlation_INCOMPLETE.csv'),
                index=False, encoding='utf-8'
            )
        else:
            print(f"\n✓ {len(equations_with_all_benchmarks)} UNIQUE equations successfully fitted ALL {num_total_benchmarks} benchmarks")
            print(f"  (After removing duplicates and invalid data)")
            equation_avg_stats_sorted = equations_with_all_benchmarks.sort_values('avg_abs_corr', ascending=False)
            
            # Save equation ranking (only complete equations)
            equation_avg_stats_sorted.to_csv(
                os.path.join(equation_exploration_folder, 'equation_ranking_by_avg_correlation.csv'),
                index=False, encoding='utf-8'
            )
            
            # Save all detailed results
            results_df_sorted = results_df.sort_values(['abs_correlation'], ascending=False)
            results_df_sorted.to_csv(
                os.path.join(equation_exploration_folder, 'all_equation_benchmark_combinations.csv'),
                index=False, encoding='utf-8'
            )
            
            # Get equation IDs that are complete (for filtering later)
            complete_equation_ids = set(equations_with_all_benchmarks['equation_id'].values)
            
            # Create summary text file
            with open(os.path.join(equation_exploration_folder, 'summary.txt'), 'w', encoding='utf-8') as f:
                f.write("EQUATION EXPLORATION - SUMMARY (SHARED PARAMETERS)\n")
                f.write("=" * 120 + "\n\n")
                f.write("Testing various equations combining Complexity and Parameter Count to predict Benchmark scores\n")
                f.write("METHOD: Each equation uses SHARED parameters fitted across ALL benchmarks simultaneously\n")
                f.write("RANKING: Based on average correlation across all benchmarks using the same parameter set\n\n")
                
                f.write(f"Total equations tested: {len(EQUATIONS_TO_TEST)}\n")
                f.write(f"Benchmarks analyzed: {results_df['benchmark'].nunique()}\n")
                f.write(f"Successful fits: {len(results_df)}\n")
                f.write(f"Equations with ALL benchmarks fitted: {len(equations_with_all_benchmarks)}\n")
                f.write(f"Equations with PARTIAL fits (excluded): {len(equations_partial)}\n\n")
                
                if len(equations_partial) > 0:
                    f.write("EXCLUDED EQUATIONS (Failed to fit ALL benchmarks):\n")
                    f.write("-" * 120 + "\n")
                    for _, row in equations_partial.iterrows():
                        f.write(f"  • {row['equation_name'][:80]} - Fitted {int(row['num_benchmarks'])}/{num_total_benchmarks} benchmarks\n")
                    f.write("\n\n")
                
                f.write("EQUATION RANKING BY AVERAGE ABSOLUTE CORRELATION:\n")
                f.write("(Only equations that successfully fitted ALL benchmarks)\n")
                f.write("-" * 120 + "\n")
                f.write(f"{'Rank':<6} {'Equation':<80} {'Avg|Corr|':<12} {'Std':<12} {'Avg R²':<12} {'N':<6}\n")
                f.write("-" * 120 + "\n")
                
                for rank_idx, (_, row) in enumerate(equation_avg_stats_sorted.iterrows(), 1):
                    f.write(f"{rank_idx:<6} {row['equation_name']:<80} {row['avg_abs_corr']:<12.6f} "
                           f"{row['std_abs_corr']:<12.6f} {row['avg_r2']:<12.6f} {int(row['num_benchmarks']):<6}\n")
                
                f.write("\n\n")
                f.write("TOP 20 INDIVIDUAL BENCHMARK-EQUATION COMBINATIONS:\n")
                f.write("(Only from equations that fitted ALL benchmarks)\n")
                f.write("-" * 120 + "\n")
                f.write(f"{'Rank':<6} {'Benchmark':<35} {'Equation':<60} {'|Corr|':<10} {'R²':<10}\n")
                f.write("-" * 120 + "\n")
                
                # Filter to only include complete equations
                results_df_complete = results_df[results_df['equation_id'].isin(complete_equation_ids)]
                results_df_complete_sorted = results_df_complete.sort_values('abs_correlation', ascending=False)
                top_20 = results_df_complete_sorted.head(20)
                for idx, (_, row) in enumerate(top_20.iterrows(), 1):
                    bench_short = row['benchmark'].replace('BENCH-', '')
                    f.write(f"{idx:<6} {bench_short:<35} {row['equation_name'][:58]:<60} "
                           f"{row['abs_correlation']:<10.6f} {row['r2_score']:<10.6f}\n")
                
                f.write("\n\n")
                f.write("BEST EQUATION FOR EACH BENCHMARK:\n")
                f.write("(Only from equations that fitted ALL benchmarks)\n")
                f.write("-" * 120 + "\n")
                f.write(f"{'Benchmark':<35} {'Equation':<60} {'Correlation':<12} {'R²':<10}\n")
                f.write("-" * 120 + "\n")
                
                for bench_name in BENCH_ROWS_NAMES:
                    bench_results = results_df_complete[results_df_complete['benchmark'] == bench_name]
                    if len(bench_results) > 0:
                        best = bench_results.sort_values('abs_correlation', ascending=False).iloc[0]
                        bench_short = best['benchmark'].replace('BENCH-', '')
                        f.write(f"{bench_short:<35} {best['equation_name'][:58]:<60} "
                               f"{best['pearson_correlation']:<12.6f} {best['r2_score']:<10.6f}\n")
                
                f.write("\n\n")
                f.write("SHARED PARAMETERS FOR TOP 5 EQUATIONS:\n")
                f.write("(These parameters are SHARED across all benchmarks - same values used for all)\n")
                f.write("-" * 120 + "\n")
                
                # Get top 5 equations (not individual benchmark combinations)
                top_5_equations = equation_avg_stats_sorted.head(5)
                for idx, (_, eq_row) in enumerate(top_5_equations.iterrows(), 1):
                    f.write(f"{idx}. {eq_row['equation_name']}\n")
                    f.write(f"   Average Correlation: {eq_row['avg_abs_corr']:.6f} ± {eq_row['std_abs_corr']:.6f}\n")
                    f.write(f"   Average R²: {eq_row['avg_r2']:.6f}\n")
                    
                    # Get the params from any benchmark entry (they're all the same)
                    eq_entry = results_df[results_df['equation_id'] == eq_row['equation_id']].iloc[0]
                    params = eq_entry['params']
                    
                    # Format the parameters nicely
                    param_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
                    param_strs = []
                    for i, param_val in enumerate(params):
                        if i < len(param_letters):
                            # Format with scientific notation if very small/large
                            if abs(param_val) < 0.001 or abs(param_val) > 1000:
                                param_strs.append(f"{param_letters[i]}={param_val:.2e}")
                            else:
                                param_strs.append(f"{param_letters[i]}={param_val:.4f}")
                    
                    f.write(f"   Shared parameters: {', '.join(param_strs)}\n")
                    
                    # Show per-benchmark correlations with these shared parameters
                    eq_benchmarks = results_df[results_df['equation_id'] == eq_row['equation_id']]
                    f.write(f"   Per-benchmark correlations:\n")
                    for _, bench_row in eq_benchmarks.iterrows():
                        bench_short = bench_row['benchmark'].replace('BENCH-', '')
                        f.write(f"     • {bench_short}: {bench_row['pearson_correlation']:.6f}\n")
                    f.write("\n")
                
                f.write("Note: ALL parameters shown above are SHARED - the same values are used across all benchmarks.\n")
                f.write("Detailed results for all equations can be found in:\n")
                f.write("all_equation_benchmark_combinations.csv\n")
            
            # Create visualization: Top equations by average correlation (only complete equations)
            print(f"\n✓ Creating ranking visualization with {len(equation_avg_stats_sorted)} complete equations")
            
            # VERIFICATION: Print first few equations to console
            print("\n  Top 5 equations in ranking visualization:")
            for rank_idx, (_, row) in enumerate(equation_avg_stats_sorted.head(5).iterrows(), 1):
                print(f"    {rank_idx}. {row['equation_name'][:60]} | Fitted {int(row['num_benchmarks'])}/{num_total_benchmarks} benchmarks | Avg|Corr|={row['avg_abs_corr']:.6f}")
            
            fig = plt.figure(figsize=(16, max(10, len(equation_avg_stats_sorted) * 0.4)))
            
            # Reverse the order so highest correlations appear at the top
            labels = [row['equation_name'][:70] for _, row in equation_avg_stats_sorted.iterrows()][::-1]
            avg_corrs = equation_avg_stats_sorted['avg_abs_corr'].values[::-1]
            std_corrs = equation_avg_stats_sorted['std_abs_corr'].values[::-1]
            
            y_pos = range(len(labels))
            
            plt.barh(y_pos, avg_corrs, xerr=std_corrs, alpha=0.7, color='steelblue', 
                    edgecolor='black', capsize=3)
            plt.yticks(y_pos, labels, fontsize=8)
            plt.xlabel('Average Absolute Pearson Correlation', fontsize=10)
            plt.ylabel('Equation', fontsize=10)
            plt.title('Equation Performance Ranking\n(Average Correlation Across All Benchmarks - Shared Parameters Method)', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
            
            # Add timestamp to verify this is the new version
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            plt.text(0.99, 0.01, f'Generated: {timestamp}', 
                    transform=fig.transFigure, fontsize=6, ha='right', va='bottom', alpha=0.5)
            
            # Add value labels
            for i, (corr, std) in enumerate(zip(avg_corrs, std_corrs)):
                plt.text(corr + std + 0.01, i, f'{corr:.4f}±{std:.4f}', 
                        va='center', fontsize=7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(equation_exploration_folder, 'equation_ranking.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create heatmap: equations vs benchmarks (only complete equations)
            # Get the list of equation IDs (not names) that fitted all benchmarks
            complete_equation_ids_heatmap = set(equations_with_all_benchmarks['equation_id'].values)
            complete_equation_names = set(equations_with_all_benchmarks['equation_name'].values)
            
            print(f"\n  Heatmap filtering: {len(complete_equation_ids_heatmap)} unique equation IDs")
            print(f"  Heatmap filtering: {len(complete_equation_names)} unique equation names")
            
            # Filter results to only include complete equations (by ID for accuracy)
            results_df_for_heatmap = results_df[results_df['equation_id'].isin(complete_equation_ids_heatmap)]
            
            print(f"  After filtering by name: {len(results_df_for_heatmap)} rows")
            print(f"  Unique equations in filtered data: {results_df_for_heatmap['equation_name'].nunique()}")
            print(f"  Unique benchmarks in filtered data: {results_df_for_heatmap['benchmark'].nunique()}")
            
            # Create pivot table
            pivot_corr = results_df_for_heatmap.pivot_table(
                values='abs_correlation',
                index='equation_name',
                columns='benchmark',
                aggfunc='max'
            )
            
            print(f"  Pivot table shape: {pivot_corr.shape} (equations x benchmarks)")
            print(f"  Equations before dropna: {len(pivot_corr)}")
            
            # Check which equations have NaN values
            equations_with_nan = pivot_corr[pivot_corr.isna().any(axis=1)]
            if len(equations_with_nan) > 0:
                print(f"\n  ⚠️  WARNING: {len(equations_with_nan)} equations have NaN values in pivot table:")
                for eq_name in equations_with_nan.index[:5]:  # Show first 5
                    missing_benchmarks = equations_with_nan.loc[eq_name][equations_with_nan.loc[eq_name].isna()].index.tolist()
                    print(f"     • {eq_name[:60]} - Missing: {[b.replace('BENCH-', '') for b in missing_benchmarks]}")
            
            # Filter out any equations that don't have all benchmarks (safety check)
            pivot_corr = pivot_corr.dropna()
            
            print(f"  Equations after dropna: {len(pivot_corr)}")
            
            # Sort by average correlation
            pivot_corr['_avg'] = pivot_corr.mean(axis=1)
            pivot_corr = pivot_corr.sort_values('_avg', ascending=False)
            pivot_corr = pivot_corr.drop('_avg', axis=1)
            
            print(f"\n✓ Heatmap will show {len(pivot_corr)} equations (all with complete benchmark fits)")
            
            fig = plt.figure(figsize=(16, max(10, len(pivot_corr) * 0.3)))
            
            # Create heatmap
            im = plt.imshow(pivot_corr.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            
            # Set ticks and labels
            plt.xticks(range(len(pivot_corr.columns)), 
                      [col.replace('BENCH-', '') for col in pivot_corr.columns], 
                      rotation=45, ha='right', fontsize=8)
            plt.yticks(range(len(pivot_corr.index)), 
                      [idx[:70] for idx in pivot_corr.index], 
                      fontsize=7)
            
            plt.xlabel('Benchmark', fontsize=10)
            plt.ylabel('Equation', fontsize=10)
            plt.title('Correlation Heatmap: Equations vs Benchmarks\n(Using Shared Parameters Across All Benchmarks)', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('|Correlation|', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(equation_exploration_folder, 'correlation_heatmap.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            print("\n" + "="*70)
            print("Equation exploration complete! (SHARED PARAMETERS METHOD)")
            print(f"Results saved to: {equation_exploration_folder}")
            print(f"\nEquations with complete fits: {len(equations_with_all_benchmarks)}")
            print(f"Equations excluded (partial fits): {len(equations_partial)}")
            print("\nTop 5 equations by average correlation across all benchmarks:")
            print("(Using SHARED parameters - same values for all benchmarks)")
            for rank_idx, (_, row) in enumerate(equation_avg_stats_sorted.head(5).iterrows(), 1):
                print(f"  {rank_idx}. {row['equation_name'][:70]}")
                print(f"     Avg |Corr|: {row['avg_abs_corr']:.6f} ± {row['std_abs_corr']:.6f}")
                print(f"     Avg R²: {row['avg_r2']:.6f}")
            print("="*70)

# ================================================================================

# --------------------------------------------------------------
# Plot graphs: FOR EACH BENCHMARK - MAGICAL VARIABLE vs BENCHMARK
if False:

    # Create magical variable: complexity * param count
    appended_benchmarks_df['magical_var'] = appended_benchmarks_df['complexity'] * appended_benchmarks_df['count']

    print("\n" + "="*70)
    print("Starting MAGICAL VARIABLE (Complexity × Param Count) vs BENCHMARK analysis...")
    print("="*70)

    # Create main output folder
    magical_benchmarks_folder = os.path.join(OUTPUT_FOLDER, 'magical_var_vs_benchmarks')
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
            'cubic': {
                'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
                'equation': lambda params: f'y = {params[0]:.16f}x³ + {params[1]:.16f}x² + {params[2]:.16f}x + {params[3]:.16f}',
                'initial_guess': [1, 1, 1, 1]
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
        plt.plot(x_line_bench, y_line_bench, 'r-', linewidth=2, label=f'Best fit ({best_name_bench})')
        plt.xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
        plt.ylabel('Magical Variable (Complexity × Param Count)')
        plt.title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Magical Variable')
        plt.legend(loc='lower center', bbox_to_anchor=(0.48, -0.18), ncol=1, frameon=True, fancybox=True, shadow=True)
        plt.subplots_adjust(bottom=0.22)
        plt.figtext(0.5, 0.07, f'{best_equation_bench}     R² = {best_r2_bench:.16f}', 
                    ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
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
            f.write(f"Equation: {best_equation_bench}\n")
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
    detailed_magical_benchmarks_folder = os.path.join(OUTPUT_FOLDER, 'detailed_magical_var_analysis')
    if not os.path.exists(detailed_magical_benchmarks_folder):
        os.makedirs(detailed_magical_benchmarks_folder)

    # Dictionary to store all detailed results
    all_detailed_magical_results = []

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
                    'cubic': {
                        'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
                        'equation': lambda params: f'y = {params[0]:.16f}x³ + {params[1]:.16f}x² + {params[2]:.16f}x + {params[3]:.16f}',
                        'initial_guess': [1, 1, 1, 1]
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
                filter_clean_name = f"filter_{int(filter_val)}"
                
                # Plot scatter with regression line
                fig = plt.figure(figsize=(12, 9))
                plt.scatter(x_data_combo, y_data_combo, alpha=0.6, label='Data')
                plt.plot(x_line_combo, y_line_combo, 'r-', linewidth=2, label=f'Best fit ({best_name_combo})')
                plt.xlabel(bench_name.replace('BENCH-', '').replace('_', ' '))
                plt.ylabel('Magical Variable (Complexity × Param Count)')
                plt.title(f'{bench_name.replace("BENCH-", "").replace("_", " ")} vs Magical Variable\nType: {type_str}, Filter: {int(filter_val)} sigma')
                plt.legend(loc='lower center', bbox_to_anchor=(0.48, -0.18), ncol=1, frameon=True, fancybox=True, shadow=True)
                plt.subplots_adjust(bottom=0.28)
                
                # Add equation and statistics
                stats_text = f'{best_equation_combo}\nR² = {best_r2_combo:.16f}\nCorrelation = {correlation_combo:.16f}\nN = {len(x_data_combo)}'
                plt.figtext(0.5, 0.07, stats_text, ha='center', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.savefig(os.path.join(type_folder, f'{filter_clean_name}_regression.png'))
                plt.close()
                
                # Save statistics to text file
                with open(os.path.join(type_folder, f'{filter_clean_name}_stats.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"{bench_name} VS MAGICAL VARIABLE\n")
                    f.write(f"Type: {type_str}\n")
                    f.write(f"Filter: {int(filter_val)} sigma\n")
                    f.write("=" * 70 + "\n\n")
                    
                    f.write("REGRESSION:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Best fit function: {best_name_combo}\n")
                    f.write(f"Equation: {best_equation_combo}\n")
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
                    'filter': int(filter_val),
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
        labels = [f"{row['benchmark'].replace('BENCH-', '')[:20]}\n{row['type'][:15]}, F={int(row['filter'])}" 
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