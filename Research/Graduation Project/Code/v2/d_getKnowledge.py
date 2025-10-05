import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

# Plot graphs: FILTER vs COMPLEXITY
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
        'equation': lambda params: f'y = {params[0]:.8f}x + {params[1]:.8f}',
        'initial_guess': [1, 1]
    },
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
        'equation': lambda params: f'y = {params[0]:.8f}x + {params[1]:.8f}',
        'initial_guess': [1, 1]
    },
    'quadratic': {
        'func': lambda x, a, b, c: a * x**2 + b * x + c,
        'equation': lambda params: f'y = {params[0]:.8f}x² + {params[1]:.8f}x + {params[2]:.8f}',
        'initial_guess': [1, 1, 1]
    },
    'cubic': {
        'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
        'equation': lambda params: f'y = {params[0]:.8f}x³ + {params[1]:.8f}x² + {params[2]:.8f}x + {params[3]:.8f}',
        'initial_guess': [1, 1, 1, 1]
    },
    'exponential': {
        'func': lambda x, a, b, c: a * np.exp(b * x) + c,
        'equation': lambda params: f'y = {params[0]:.8f}·e^({params[1]:.8f}x) + {params[2]:.8f}',
        'initial_guess': [1, 0.1, 1]
    },
    'logarithmic': {
        'func': lambda x, a, b, c: a * np.log(x + 1) + b * x + c,
        'equation': lambda params: f'y = {params[0]:.8f}·ln(x+1) + {params[1]:.8f}x + {params[2]:.8f}',
        'initial_guess': [1, 1, 1]
    },
    'power': {
        'func': lambda x, a, b, c: a * (x + 1)**b + c,
        'equation': lambda params: f'y = {params[0]:.8f}·(x+1)^{params[1]:.8f} + {params[2]:.8f}',
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
filtered_df = appended_benchmarks_df[appended_benchmarks_df['filter'] != 0]
plt.scatter(filtered_df['filter'], filtered_df['complexity'], alpha=0.6, label='Data')
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Best fit all data ({best_name})')
plt.scatter(x_max, y_max, color='green', s=100, alpha=0.8, marker='D', label='Maximum values', zorder=5)
plt.plot(x_line_max, y_line_max, 'g--', linewidth=2, label=f'Best fit max ({best_name_max})')
plt.xlabel('Filter')
plt.ylabel('Complexity')
plt.title('Filter vs Complexity')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=True, fancybox=True, shadow=True)
plt.subplots_adjust(bottom=0.28)
plt.figtext(0.5, 0.14, f'All data: {best_equation}     R² = {best_r2:.8f}', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.figtext(0.5, 0.11, f'Max values: {best_equation_max}     R² = {best_r2_max:.8f}', 
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
    f.write(f"R² score: {best_r2:.8f}\n")
    f.write(f"Parameters: {best_params}\n\n")
    
    f.write("MAXIMUM VALUES REGRESSION:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Best fit function: {best_name_max}\n")
    f.write(f"Equation: {best_equation_max}\n")
    f.write(f"R² score: {best_r2_max:.8f}\n")
    f.write(f"Parameters: {best_params_max}\n")

# Save average complexity data to text file
with open(os.path.join(filter_complexity_folder, 'average_complexity.txt'), 'w', encoding='utf-8') as f:
    f.write("FILTER VS AVERAGE COMPLEXITY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"{'Filter':<15} {'Average Complexity':<20}\n")
    f.write("-" * 70 + "\n")
    for x, y in zip(x_avg, y_avg):
        f.write(f"{x:<15} {y:<20.8f}\n")
    f.write("\n")
    f.write(f"Overall mean: {y_avg.mean():.8f}\n")
    f.write(f"Overall std: {y_avg.std():.8f}\n")

# Save maximum complexity data to text file
with open(os.path.join(filter_complexity_folder, 'maximum_complexity.txt'), 'w', encoding='utf-8') as f:
    f.write("FILTER VS MAXIMUM COMPLEXITY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"{'Filter':<15} {'Maximum Complexity':<20}\n")
    f.write("-" * 70 + "\n")
    for x, y in zip(x_max, y_max):
        f.write(f"{x:<15} {y:<20.8f}\n")

# --------------------------------------------------------------

# Plot graphs: EACH BENCHMARK vs COMPLEXITY


# --------------------------------------------------------------



# ================================================================================