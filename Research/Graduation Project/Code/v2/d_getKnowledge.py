import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'appended_csvs'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'info_results'))

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

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
appended_benchmarks_df['filter'] = appended_benchmarks_df['filter'].replace(0, 0)

# Get rows in header that start with "BENCH"
BENCH_ROWS_NAMES = appended_benchmarks_df.columns[appended_benchmarks_df.columns.str.startswith('BENCH')].tolist()

# File structure
# model,types,filter,count,min,max,mean,std,bin_count,shannon_entropy,desequilibrium,complexity,BENCH-OPEN_LLM_AVERAGE,BENCH-LMARENA_SCORE,BENCH-MMLU_5,BENCH-MMLU_PRO_5

# ================================================================================

# --------------------------------------------------------------
# Plot graphs: COMPLEXITY vs FILTER

# Remove NaN values and filter = 0 for regression
mask = ~(appended_benchmarks_df['filter'].isna() | appended_benchmarks_df['complexity'].isna()) & (appended_benchmarks_df['filter'] != 0)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
max_complexity_by_filter = appended_benchmarks_df[appended_benchmarks_df['filter'] != 0].groupby('filter')['complexity'].max().reset_index()
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
filtered_df = appended_benchmarks_df[appended_benchmarks_df['filter'] != 0]
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
avg_complexity_by_filter = appended_benchmarks_df[appended_benchmarks_df['filter'] != 0].groupby('filter')['complexity'].mean().reset_index()
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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