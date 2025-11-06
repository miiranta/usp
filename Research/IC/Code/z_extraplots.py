import csv
import os
import datetime
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "info_and_graphs")
EVAL_FOLDER = os.path.join(SCRIPT_FOLDER, "eval_results")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "info_and_graphs")
DATASETS_FOLDER = os.path.join(os.path.dirname(SCRIPT_FOLDER), "Datasets", "Grades")

def _date_key(d):
    """Parse date in DD/MM/YYYY format and return sortable tuple"""
    day, month, year = map(int, d.split('/'))
    return (year, month, day)

def parse_date_to_datetime(d):
    """Parse date in DD/MM/YYYY format and return datetime object"""
    day, month, year = map(int, d.split('/'))
    return datetime.datetime(year, month, day)

def get_available_models(result_csv_data):
    """Extract unique model names from CSV data"""
    models = set()
    
    for row in result_csv_data[1:]:
        if len(row) > 1:
            model = row[1].strip()
            if model:
                models.add(model)
    
    response = sorted(models)
    
    # Move human to the end if present
    if 'human' in response:
        response.remove('human')
        response.append('human')
    
    return response

def get_grade_avarage_model(result_csv_data, model):
    """Calculate average grade for a specific model"""
    total_grades = 0
    count = 0
    
    for row in result_csv_data[1:]:
        if row[1].strip() == model:
            try:
                grade = int(row[2])
                total_grades += grade
                count += 1
            except ValueError:
                print(f"Invalid grade value in row: {row}")
    
    if count == 0:
        return None
    
    return total_grades / count

def plot_average_by_date_and_model_with_ipca():
    """
    Plot average grade by date and model with IPCA overlay using cubic spline interpolation.
    IPCA data is in YYYY-MM-DD format (monthly), CSV data is in DD/MM/YYYY format.
    """
    
    # Load appended results CSV
    csv_file = os.path.join(INPUT_FOLDER, "appended_results.csv")
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return
    
    result_csv_data = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if row:
                result_csv_data.append(row)
    
    if not result_csv_data or len(result_csv_data) < 2:
        print("No data found in appended_results.csv")
        return
    
    # Extract unique dates and sort them
    dates_set = {row[0].strip() for row in result_csv_data[1:]}
    if not dates_set:
        print("No date rows to plot.")
        return

    dates_sorted = sorted(dates_set, key=_date_key)

    models = get_available_models(result_csv_data)
    if not models:
        print("No models found to plot.")
        return
    
    # Load IPCA data
    ipca_file = os.path.join(EVAL_FOLDER, "ipca_data.csv")
    if not os.path.exists(ipca_file):
        print(f"Error: {ipca_file} not found.")
        return
    
    print("Loading IPCA data from file...")
    ipca_df = pd.read_csv(ipca_file)
    ipca_df['date'] = pd.to_datetime(ipca_df['date'])
    
    # Convert dates_sorted (DD/MM/YYYY) to datetime objects for interpolation
    dates_datetime = [parse_date_to_datetime(d) for d in dates_sorted]
    
    # Prepare IPCA data for cubic spline interpolation
    ipca_dates = ipca_df['date'].values
    ipca_values = ipca_df['ipca_monthly'].values
    
    # Convert dates to numeric values (days since epoch) for interpolation
    ipca_dates_numeric = pd.to_datetime(ipca_dates).astype('int64') / 10**9 / 86400  # days since epoch
    target_dates_numeric = [d.timestamp() / 86400 for d in dates_datetime]  # days since epoch
    
    # Create cubic spline interpolator
    cs = CubicSpline(ipca_dates_numeric, ipca_values)
    
    # Interpolate IPCA values for our target dates
    ipca_series = []
    for target_numeric in target_dates_numeric:
        # Check if target date is within the range of IPCA data
        if target_numeric < ipca_dates_numeric[0] or target_numeric > ipca_dates_numeric[-1]:
            ipca_series.append(float('nan'))
        else:
            ipca_series.append(float(cs(target_numeric)))
    
    print(f"Interpolated IPCA values for {len(ipca_series)} dates using cubic spline.")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(29, 9))
    ax2 = ax.twinx()
    x_positions = list(range(len(dates_sorted)))

    # Plot model grades
    for model in models:
        series = []
        for d in dates_sorted:
            grades = [float(row[2]) for row in result_csv_data[1:] if row[0].strip() == d and row[1].strip() == model]
            if grades:
                series.append(sum(grades) / len(grades))
            else:
                series.append(float('nan'))
        line, = ax.plot(x_positions, series, marker='o', label=model)

        try:
            avg = get_grade_avarage_model(result_csv_data, model)
            if avg is not None:
                ax.axhline(avg, color=line.get_color(), linestyle=':', linewidth=1, alpha=0.8, zorder=0)
        except Exception:
            pass

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)

    # Plot IPCA
    ax2.plot(x_positions, ipca_series, color='blue', linestyle='-', linewidth=2, label='IPCA (%)', alpha=0.7)

    # Configure x-axis
    num_date_labels = 80
    display_indices = list(range(0, len(dates_sorted), max(1, len(dates_sorted) // num_date_labels)))

    tick_positions = [x_positions[i] for i in display_indices]
    tick_labels = [dates_sorted[i] for i in display_indices]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=60, ha='right')
    ax.tick_params(axis='x', pad=12)

    # Configure y-axes with aligned zeros
    ax.set_ylabel('Average Grade')
    ax2.set_ylabel('IPCA (%)', color='black')
    ax.set_title('Average Grade and IPCA by Date')
    
    # Align the zero point on both y-axes
    y1_min, y1_max = ax.get_ylim()
    y2_min, y2_max = ax2.get_ylim()
    
    # Calculate ratios to align zero
    y1_range_below = abs(y1_min)
    y1_range_above = abs(y1_max)
    y2_range_below = abs(y2_min)
    y2_range_above = abs(y2_max)
    
    # Use the maximum ratio to set symmetric or proportional limits
    ratio = max(y1_range_below / y2_range_below if y2_range_below != 0 else 1,
                y1_range_above / y2_range_above if y2_range_above != 0 else 1)
    
    # Adjust y2 limits to align zeros with 15% padding
    padding = 1.6
    if y2_range_below != 0 or y2_range_above != 0:
        new_y2_min = -y1_range_below / ratio if y1_range_below != 0 else y2_min
        new_y2_max = y1_range_above / ratio if y1_range_above != 0 else y2_max
        y2_range = new_y2_max - new_y2_min
        new_y2_min -= y2_range * padding
        new_y2_max += y2_range * padding
        ax2.set_ylim(new_y2_min, new_y2_max)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(top=0.9)

    output_file_path = os.path.join(OUTPUT_FOLDER, "average_grade_by_date_with_ipca.png")
    fig.savefig(output_file_path)
    plt.close(fig)
    
    print(f"✓ Plot saved to: {output_file_path}")

def load_human_dataset(file_path):
    """Load a human dataset CSV file and return the data"""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
    
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if row:
                data.append(row)
    
    return data

def get_grade_average_dataset(dataset_data):
    """Calculate average grade for a dataset"""
    total_grades = 0
    count = 0
    
    for row in dataset_data[1:]:  # Skip header
        try:
            grade = int(row[2])
            total_grades += grade
            count += 1
        except (ValueError, IndexError):
            continue
    
    if count == 0:
        return None
    
    return total_grades / count

def get_grade_confidence_interval_dataset(dataset_data, confidence=0.95):
    """Calculate confidence interval for a dataset"""
    grades = []
    
    for row in dataset_data[1:]:  # Skip header
        try:
            grade = int(row[2])
            grades.append(grade)
        except (ValueError, IndexError):
            continue
    
    if not grades:
        return None
    
    mean = sum(grades) / len(grades)
    
    if len(grades) == 1:
        return (mean, mean)
    
    variance = sum((x - mean) ** 2 for x in grades) / (len(grades) - 1)
    std_dev = math.sqrt(variance)
    
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * (std_dev / math.sqrt(len(grades)))
    
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    return confidence_interval

def get_grade_average_global(result_csv_data):
    """Calculate global average grade"""
    total_grades = 0
    count = 0
    
    for row in result_csv_data[1:]:
        try:
            grade = int(row[2])
            total_grades += grade
            count += 1
        except (ValueError, IndexError):
            continue
    
    if count == 0:
        return None
    
    return total_grades / count

def get_grade_confidence_interval_global(result_csv_data, confidence=0.95):
    """Calculate global confidence interval"""
    grades = []
    
    for row in result_csv_data[1:]:
        try:
            grade = int(row[2])
            grades.append(grade)
        except (ValueError, IndexError):
            continue
    
    if not grades:
        return None
    
    mean = sum(grades) / len(grades)
    
    if len(grades) == 1:
        return (mean, mean)
    
    variance = sum((x - mean) ** 2 for x in grades) / (len(grades) - 1)
    std_dev = math.sqrt(variance)
    
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * (std_dev / math.sqrt(len(grades)))
    
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    return confidence_interval

def get_grade_frequency_global(result_csv_data):
    """Calculate grade frequency globally"""
    grade_frequency = {}
    
    for row in result_csv_data[1:]:
        try:
            grade = int(row[2])
            grade_frequency[grade] = grade_frequency.get(grade, 0) + 1
        except (ValueError, IndexError):
            continue
    
    return grade_frequency

def get_grade_frequency_dataset(dataset_data):
    """Calculate grade frequency for a dataset"""
    grade_frequency = {}
    
    for row in dataset_data[1:]:
        try:
            grade = int(row[2])
            grade_frequency[grade] = grade_frequency.get(grade, 0) + 1
        except (ValueError, IndexError):
            continue
    
    return grade_frequency

def get_grade_frequency_model(result_csv_data, model):
    """Calculate grade frequency for a specific model"""
    grade_frequency = {}
    
    for row in result_csv_data[1:]:
        if row[1].strip() == model:
            try:
                grade = int(row[2])
                grade_frequency[grade] = grade_frequency.get(grade, 0) + 1
            except (ValueError, IndexError):
                continue
    
    return grade_frequency

def get_grade_standard_deviation_global(result_csv_data):
    """Calculate standard deviation globally"""
    grades = []
    
    for row in result_csv_data[1:]:
        try:
            grade = int(row[2])
            grades.append(grade)
        except (ValueError, IndexError):
            continue
    
    if not grades:
        return None
    
    if len(grades) == 1:
        return 0.0
    
    mean = sum(grades) / len(grades)
    variance = sum((x - mean) ** 2 for x in grades) / (len(grades) - 1)
    std_dev = math.sqrt(variance)
    
    return std_dev

def get_grade_standard_deviation_dataset(dataset_data):
    """Calculate standard deviation for a dataset"""
    grades = []
    
    for row in dataset_data[1:]:
        try:
            grade = int(row[2])
            grades.append(grade)
        except (ValueError, IndexError):
            continue
    
    if not grades:
        return None
    
    if len(grades) == 1:
        return 0.0
    
    mean = sum(grades) / len(grades)
    variance = sum((x - mean) ** 2 for x in grades) / (len(grades) - 1)
    std_dev = math.sqrt(variance)
    
    return std_dev

def get_grade_standard_deviation_model(result_csv_data, model):
    """Calculate standard deviation for a specific model"""
    grades = []
    
    for row in result_csv_data[1:]:
        if row[1].strip() == model:
            try:
                grade = int(row[2])
                grades.append(grade)
            except (ValueError, IndexError):
                continue
    
    if not grades:
        return None
    
    if len(grades) == 1:
        return 0.0
    
    mean = sum(grades) / len(grades)
    variance = sum((x - mean) ** 2 for x in grades) / (len(grades) - 1)
    std_dev = math.sqrt(variance)
    
    return std_dev

def get_grade_average_model(result_csv_data, model):
    """Calculate average grade for a specific model"""
    total_grades = 0
    count = 0
    
    for row in result_csv_data[1:]:
        if row[1].strip() == model:
            try:
                grade = int(row[2])
                total_grades += grade
                count += 1
            except (ValueError, IndexError):
                continue
    
    if count == 0:
        return None
    
    return total_grades / count

def get_grade_confidence_interval_model(result_csv_data, model, confidence=0.95):
    """Calculate confidence interval for a specific model"""
    grades = []
    
    for row in result_csv_data[1:]:
        if row[1].strip() == model:
            try:
                grade = int(row[2])
                grades.append(grade)
            except (ValueError, IndexError):
                continue
    
    if not grades:
        return None
    
    mean = sum(grades) / len(grades)
    
    if len(grades) == 1:
        return (mean, mean)
    
    variance = sum((x - mean) ** 2 for x in grades) / (len(grades) - 1)
    std_dev = math.sqrt(variance)
    
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * (std_dev / math.sqrt(len(grades)))
    
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    return confidence_interval

def plot_average_and_confidence_interval_by_dataset(confidence=0.95):
    """
    Plot average grade with confidence intervals for all models and human datasets.
    Includes: All individual models, all human datasets (Specialist Cezio, Conciliated, Open),
    Global models only, Global humans only
    """
    
    # Load appended results CSV for models
    csv_file = os.path.join(INPUT_FOLDER, "appended_results.csv")
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return
    
    result_csv_data = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if row:
                result_csv_data.append(row)
    
    if not result_csv_data or len(result_csv_data) < 2:
        print("No data found in appended_results.csv")
        return
    
    # Get all available models
    models = get_available_models(result_csv_data)
    if not models:
        print("No models found to plot.")
        return
    
    # Load human datasets
    dataset_files = {
        'Human (Specialist)': os.path.join(DATASETS_FOLDER, 'human_cezio_350.csv'),
        'Human (Conciliated)': os.path.join(DATASETS_FOLDER, 'human_conciliado_220.csv'),
        'Human (Open)': os.path.join(DATASETS_FOLDER, 'human_open_278.csv')
    }
    
    datasets = {}
    for name, file_path in dataset_files.items():
        data = load_human_dataset(file_path)
        if data:
            datasets[name] = data
        else:
            print(f"Warning: Could not load {name} dataset")
    
    if not datasets:
        print("No human datasets loaded.")
        return
    
    # Combine all human data for Global humans only
    all_human_data = [result_csv_data[0]]  # Keep header
    for dataset_data in datasets.values():
        all_human_data.extend(dataset_data[1:])  # Skip headers
    
    # Create plot
    dataset_names = list(datasets.keys())
    labels = models + dataset_names + ['Global (models only)', 'Global (humans only)']
    num_items = len(labels)
    
    fig, ax = plt.subplots(figsize=(max(8, num_items * 0.8 + 4), 9))
    x_positions = list(range(num_items))
    
    cmap = plt.get_cmap('tab20')
    
    # Plot each model
    for i, model in enumerate(models):
        avg = get_grade_average_model(result_csv_data, model)
        ci = get_grade_confidence_interval_model(result_csv_data, model, confidence)
        
        color = cmap(i % cmap.N)
        
        if avg is not None and ci is not None:
            lower = avg - ci[0]
            upper = ci[1] - avg
            
            ax.errorbar([i], [avg], yerr=[[lower], [upper]], fmt='o',
                       color=color, ecolor=color, capsize=5, markersize=7, linestyle='none')
            ax.axhline(avg, color=color, linestyle=':', linewidth=1, alpha=0.8, zorder=0)
        else:
            ax.plot([i], [float('nan')], marker='o', color=color)
    
    # Plot each human dataset
    for i, (name, dataset_data) in enumerate(datasets.items()):
        pos = len(models) + i
        avg = get_grade_average_dataset(dataset_data)
        ci = get_grade_confidence_interval_dataset(dataset_data, confidence)
        
        color = cmap(pos % cmap.N)
        
        if avg is not None and ci is not None:
            lower = avg - ci[0]
            upper = ci[1] - avg
            
            ax.errorbar([pos], [avg], yerr=[[lower], [upper]], fmt='o',
                       color=color, ecolor=color, capsize=5, markersize=7, linestyle='none')
            ax.axhline(avg, color=color, linestyle=':', linewidth=1, alpha=0.8, zorder=0)
        else:
            ax.plot([pos], [float('nan')], marker='o', color=color)
    
    # Plot Global models only
    global_models_pos = len(models) + len(datasets)
    global_models_color = cmap(global_models_pos % cmap.N)
    
    avg_global_models = get_grade_average_global(result_csv_data)
    ci_global_models = get_grade_confidence_interval_global(result_csv_data, confidence)
    
    if avg_global_models is not None and ci_global_models is not None:
        lower_gm = avg_global_models - ci_global_models[0]
        upper_gm = ci_global_models[1] - avg_global_models
        ax.errorbar([global_models_pos], [avg_global_models], yerr=[[lower_gm], [upper_gm]], fmt='o',
                   color=global_models_color, ecolor=global_models_color, capsize=5, markersize=7, linestyle='none')
        ax.axhline(avg_global_models, color=global_models_color, linestyle=':', linewidth=1, alpha=0.8, zorder=0)
    else:
        ax.plot([global_models_pos], [float('nan')], marker='o', color=global_models_color)
    
    # Plot Global humans only
    global_humans_pos = len(models) + len(datasets) + 1
    global_humans_color = cmap(global_humans_pos % cmap.N)
    
    avg_global_humans = get_grade_average_global(all_human_data)
    ci_global_humans = get_grade_confidence_interval_global(all_human_data, confidence)
    
    if avg_global_humans is not None and ci_global_humans is not None:
        lower_gh = avg_global_humans - ci_global_humans[0]
        upper_gh = ci_global_humans[1] - avg_global_humans
        ax.errorbar([global_humans_pos], [avg_global_humans], yerr=[[lower_gh], [upper_gh]], fmt='o',
                   color=global_humans_color, ecolor=global_humans_color, capsize=5, markersize=7, linestyle='none')
        ax.axhline(avg_global_humans, color=global_humans_color, linestyle=':', linewidth=1, alpha=0.8, zorder=0)
    else:
        ax.plot([global_humans_pos], [float('nan')], marker='o', color=global_humans_color)
    
    # Configure plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Average Grade')
    ax.set_title(f"Average Grade by Model and Dataset (Interval with {int(confidence*100)}% Confidence)")
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
    
    plt.tight_layout()
    output_file_path = os.path.join(OUTPUT_FOLDER, f"average_ci_by_dataset_{int(confidence*100)}perc.png")
    fig.savefig(output_file_path)
    plt.close(fig)
    
    print(f"✓ Plot saved to: {output_file_path}")

def generate_info_extra():
    """
    Generate info_extra.txt with statistics for models and human datasets.
    Similar to info.txt but with separate human datasets instead of combined.
    """
    
    # Load appended results CSV for models
    csv_file = os.path.join(INPUT_FOLDER, "appended_results.csv")
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return
    
    result_csv_data = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if row:
                result_csv_data.append(row)
    
    if not result_csv_data or len(result_csv_data) < 2:
        print("No data found in appended_results.csv")
        return
    
    # Get all available models
    models = get_available_models(result_csv_data)
    
    # Load human datasets
    dataset_files = {
        'Specialist': os.path.join(DATASETS_FOLDER, 'human_cezio_350.csv'),
        'Conciliated': os.path.join(DATASETS_FOLDER, 'human_conciliado_220.csv'),
        'Open': os.path.join(DATASETS_FOLDER, 'human_open_278.csv')
    }
    
    datasets = {}
    for name, file_path in dataset_files.items():
        data = load_human_dataset(file_path)
        if data:
            datasets[name] = data
    
    # Combine all human data for Global humans only
    all_human_data = [result_csv_data[0]]  # Keep header
    for dataset_data in datasets.values():
        all_human_data.extend(dataset_data[1:])  # Skip headers
    
    # Confidence levels to check
    confidences_to_check = [0.95, 0.99]
    
    # Write info_extra.txt
    info_file_path = os.path.join(OUTPUT_FOLDER, "info_extra.txt")
    with open(info_file_path, 'w', encoding='utf-8') as f:
        
        # General Information
        f.write("--------\n")
        f.write("General Information:\n")
        f.write(f"Models data: {len(result_csv_data) - 1} rows\n")
        f.write(f"Human data (combined): {len(all_human_data) - 1} rows\n")
        
        if models:
            f.write(f"Available models: {', '.join(models)}\n")
        
        f.write(f"\nHuman datasets:\n")
        for name, dataset_data in datasets.items():
            f.write(f"  - {name}: {len(dataset_data) - 1} rows\n")
        
        unique_phrases_models = {row[3].strip() for row in result_csv_data[1:] if len(row) > 3}
        f.write(f"\nUnique phrases evaluated (models): {len(unique_phrases_models)}\n")
        
        unique_dates_models = {row[0].strip() for row in result_csv_data[1:]}
        f.write(f"Unique dates evaluated (models): {len(unique_dates_models)}\n")
        
        # Average
        f.write("--------\n")
        f.write("Average Grades:\n")
        
        avg_global_models = get_grade_average_global(result_csv_data)
        if avg_global_models is not None:
            f.write(f"Global (models only): {avg_global_models:.64f}\n")
        
        avg_global_humans = get_grade_average_global(all_human_data)
        if avg_global_humans is not None:
            f.write(f"Global (humans only): {avg_global_humans:.64f}\n")
        
        f.write("\nBy model:\n")
        for model in models:
            avg = get_grade_average_model(result_csv_data, model)
            if avg is not None:
                f.write(f"  - {model}: {avg:.64f}\n")
        
        f.write("\nBy human dataset:\n")
        for name, dataset_data in datasets.items():
            avg = get_grade_average_dataset(dataset_data)
            if avg is not None:
                f.write(f"  - {name}: {avg:.64f}\n")
        
        # Frequency
        f.write("--------\n")
        f.write("Grade Frequency:\n")
        
        freq_global_models = get_grade_frequency_global(result_csv_data)
        if freq_global_models:
            f.write("Global (models only): ")
            for grade, frequency in sorted(freq_global_models.items()):
                f.write(f"[{grade}]: {frequency} ")
            f.write("\n")
        
        freq_global_humans = get_grade_frequency_global(all_human_data)
        if freq_global_humans:
            f.write("Global (humans only): ")
            for grade, frequency in sorted(freq_global_humans.items()):
                f.write(f"[{grade}]: {frequency} ")
            f.write("\n")
        
        f.write("\nBy model:\n")
        for model in models:
            freq = get_grade_frequency_model(result_csv_data, model)
            if freq:
                f.write(f"  - {model}: ")
                for grade, frequency in sorted(freq.items()):
                    f.write(f"[{grade}]: {frequency} ")
                f.write("\n")
        
        f.write("\nBy human dataset:\n")
        for name, dataset_data in datasets.items():
            freq = get_grade_frequency_dataset(dataset_data)
            if freq:
                f.write(f"  - {name}: ")
                for grade, frequency in sorted(freq.items()):
                    f.write(f"[{grade}]: {frequency} ")
                f.write("\n")
        
        # Standard Deviation
        f.write("--------\n")
        f.write("Standard Deviation:\n")
        
        std_global_models = get_grade_standard_deviation_global(result_csv_data)
        if std_global_models is not None:
            f.write(f"Global (models only): {std_global_models:.64f}\n")
        
        std_global_humans = get_grade_standard_deviation_global(all_human_data)
        if std_global_humans is not None:
            f.write(f"Global (humans only): {std_global_humans:.64f}\n")
        
        f.write("\nBy model:\n")
        for model in models:
            std = get_grade_standard_deviation_model(result_csv_data, model)
            if std is not None:
                f.write(f"  - {model}: {std:.64f}\n")
        
        f.write("\nBy human dataset:\n")
        for name, dataset_data in datasets.items():
            std = get_grade_standard_deviation_dataset(dataset_data)
            if std is not None:
                f.write(f"  - {name}: {std:.64f}\n")
        
        # Confidence Intervals
        for confidence in confidences_to_check:
            f.write("--------\n")
            f.write(f"{int(confidence*100)}% Confidence Interval:\n")
            
            ci_global_models = get_grade_confidence_interval_global(result_csv_data, confidence)
            if ci_global_models is not None:
                f.write(f"Global (models only): ({ci_global_models[0]:.64f}, {ci_global_models[1]:.64f})\n")
            
            ci_global_humans = get_grade_confidence_interval_global(all_human_data, confidence)
            if ci_global_humans is not None:
                f.write(f"Global (humans only): ({ci_global_humans[0]:.64f}, {ci_global_humans[1]:.64f})\n")
            
            f.write("\nBy model:\n")
            for model in models:
                ci = get_grade_confidence_interval_model(result_csv_data, model, confidence)
                if ci is not None:
                    f.write(f"  - {model}: ({ci[0]:.64f}, {ci[1]:.64f})\n")
            
            f.write("\nBy human dataset:\n")
            for name, dataset_data in datasets.items():
                ci = get_grade_confidence_interval_dataset(dataset_data, confidence)
                if ci is not None:
                    f.write(f"  - {name}: ({ci[0]:.64f}, {ci[1]:.64f})\n")
        
        f.write("--------\n")
    
    print(f"✓ Info file saved to: {info_file_path}")

if __name__ == "__main__":
    plot_average_by_date_and_model_with_ipca()
    plot_average_and_confidence_interval_by_dataset(confidence=0.95)
    plot_average_and_confidence_interval_by_dataset(confidence=0.99)
    generate_info_extra()
