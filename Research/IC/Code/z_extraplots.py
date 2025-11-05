import csv
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "info_and_graphs")
EVAL_FOLDER = os.path.join(SCRIPT_FOLDER, "eval_results")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "info_and_graphs")

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

if __name__ == "__main__":
    plot_average_by_date_and_model_with_ipca()
