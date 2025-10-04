import os
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import numpy as np
import re
import torch
import torch.optim as optim
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
from itertools import combinations

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "info_and_graphs")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "optimized_results")
INTERPOLATED_FOLDER = os.path.join(OUTPUT_FOLDER, "interpolated")
OPTIMIZED_FOLDER = os.path.join(OUTPUT_FOLDER, "optimized")

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

if not os.path.exists(INTERPOLATED_FOLDER):
    os.makedirs(INTERPOLATED_FOLDER)

if not os.path.exists(OPTIMIZED_FOLDER):
    os.makedirs(OPTIMIZED_FOLDER)

##

def _date_key(d):
    day, month, year = map(int, d.split('/'))
    return (year, month, day)

def get_unique_models(csv_data):
    """Extract unique model names from CSV data."""
    models = set()
    for row in csv_data:
        if len(row) >= 2:
            models.add(row[1])
    return sorted(list(models))

def filter_csv_by_models(csv_data, model_names):
    """Filter CSV data to only include rows from specified models."""
    if not model_names:
        return csv_data
    model_set = set(model_names)
    return [row for row in csv_data if len(row) >= 2 and row[1] in model_set]

def get_csv_lines(filename):
    filepath = os.path.join(INPUT_FOLDER, filename)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None
    with open(filepath, "r", encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.replace("\n", "").split("|") for line in lines]
        if lines:
            lines.remove(lines[0])
        return sorted(lines, key=lambda x: _date_key(x[0]))

##

def optimize_equation_parameters(equation, model_avgs, human_avgs, title="optimization", full_model_names=None, epochs=5000, lr=0.01):
    """
    equation: A string equation with parameters (a, b, c, d, ...) and variable x
                Example: "a * x + b" or "a * x**2 + b * x + c"
                x represents the average model prediction per date
    model_avgs: Dictionary of {date: average_grade} for models (interpolated)
    human_avgs: Dictionary of {date: average_grade} for humans (interpolated)
    title: Title for saving output files
    full_model_names: List of full model names for display in plots
    """
    
    # Check if optimization already exists
    output_csv_filename = f"{title}_optimized.csv"
    output_csv_path = os.path.join(OPTIMIZED_FOLDER, output_csv_filename)
    
    if os.path.exists(output_csv_path):
        print(f"✓ Optimization already exists, loading from: {output_csv_path}")
        
        # Load existing optimization results
        optimized_predictions = []
        matching_dates = []
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    matching_dates.append(parts[0])
                    optimized_predictions.append(float(parts[1]))
        
        # Calculate MSE from loaded data
        model_grades = []
        human_grades = []
        common_dates = sorted(set(model_avgs.keys()) & set(human_avgs.keys()), key=_date_key)
        
        for date in common_dates:
            model_grades.append(model_avgs[date])
            human_grades.append(human_avgs[date])
        
        # Calculate MSE
        if len(optimized_predictions) == len(human_grades):
            mse = sum((opt - hum) ** 2 for opt, hum in zip(optimized_predictions, human_grades)) / len(human_grades)
            print(f"✓ Loaded cached results with MSE: {mse:.10f}")
            
            # Extract parameters from equation - we'll return dummy params since we're loading from cache
            param_names = sorted(set(re.findall(r'\b[a-z]\b', equation)) - {'x'})
            best_params = {name: 0.0 for name in param_names}  # Dummy values
            
            return best_params, mse
        else:
            print(f"⚠️ Cached data size mismatch, recomputing...")
    
    print(f"Computing new optimization for: {title}")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get matching dates and their averages
    model_grades = []
    human_grades = []
    matching_dates = []
    
    common_dates = sorted(set(model_avgs.keys()) & set(human_avgs.keys()), key=_date_key)
    
    for date in common_dates:
        model_grades.append(model_avgs[date])
        human_grades.append(human_avgs[date])
        matching_dates.append(date)
    
    if not model_grades:
        print("No valid pairs to optimize.")
        print(f"Model dates: {len(model_avgs)}")
        print(f"Human dates: {len(human_avgs)}")
        return None, None
    
    print(f"Found {len(model_grades)} dates with matching data (using interpolated averages)")
    
    # Extract parameter names from equation (a, b, c, d, ...)
    param_names = sorted(set(re.findall(r'\b[a-z]\b', equation)) - {'x'})
    
    if not param_names:
        print("No parameters found in equation.")
        return None, None
    
    print(f"Optimizing parameters: {param_names}")
    print(f"Equation: {equation}")
    
    # Convert to PyTorch tensors and move to device
    X = torch.tensor(model_grades, dtype=torch.float32, device=device)
    y = torch.tensor(human_grades, dtype=torch.float32, device=device)
    
    # Initialize parameters as trainable tensors on device
    params = {}
    for name in param_names:
        params[name] = torch.tensor(1.0, requires_grad=True, dtype=torch.float32, device=device)
    
    # Create optimizer
    optimizer = optim.Adam(params.values(), lr=lr)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    losses = []
    best_loss = float('inf')
    best_params = None
    
    # Compile the equation once for vectorized operations
    # Replace single letter variables with params
    vectorized_equation = equation
    for name in param_names:
        vectorized_equation = vectorized_equation.replace(name, f"params['{name}']")
    vectorized_equation = vectorized_equation.replace('x', 'X')
    
    print(f"Vectorized equation: {vectorized_equation}")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute predictions using vectorized equation
        try:
            # Create evaluation context for vectorized operations
            eval_context = {
                'params': params,
                'X': X,
                'torch': torch,
                'abs': torch.abs
            }
            
            # Evaluate the vectorized equation
            predictions = eval(vectorized_equation, {"__builtins__": {}}, eval_context)
        except Exception as e:
            print(f"Error in equation evaluation: {e}")
            return None, None
        
        # Compute MSE loss
        loss = torch.mean((predictions - y) ** 2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        loss_val = loss.item()
        losses.append(loss_val)
        
        # Track best parameters
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = {name: params[name].item() for name in param_names}
        
        # Print initial loss
        if epoch == 0:
            print(f"Initial Loss: {loss_val:.10f}")
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_val:.10f}")
    
    print(f"\nOptimization complete!")
    print(f"Best parameters: {best_params}")
    print(f"Final MSE: {best_loss:.10f}")
    
    # Make predictions with best parameters (on CPU for final results)
    optimized_predictions = []
    for x_val in model_grades:
        eval_context = best_params.copy()
        eval_context['x'] = x_val
        eval_context['abs'] = abs
        
        try:
            pred = eval(equation, {"__builtins__": {}}, eval_context)
            optimized_predictions.append(float(pred))
        except:
            optimized_predictions.append(float('nan'))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Line plot comparing predictions by date
    x_positions = list(range(len(model_grades)))
    
    ax1.plot(x_positions, model_grades, marker='o', label='Original Model', color='blue', alpha=0.6, markersize=4, linewidth=1)
    ax1.plot(x_positions, human_grades, marker='o', label='Human (Ground Truth)', color='red', markersize=4, linewidth=1)
    ax1.plot(x_positions, optimized_predictions, marker='s', label='Optimized Model', color='green', markersize=3, linewidth=1)
    ax1.fill_between(x_positions, optimized_predictions, human_grades, alpha=0.3, color='lightgreen')
    
    ax1.set_xlabel('Date Index')
    ax1.set_ylabel('Average Grade')
    
    # Create detailed title with model names and equation
    title_text = f'Parameter Optimization Results (MSE: {best_loss:.4f})'
    ax1.set_title(title_text, fontsize=12, fontweight='bold')
    
    # Add model names and equation as text below the plot
    if full_model_names:
        models_text = 'Models: ' + ', '.join(full_model_names)
        ax1.text(0.5, -0.15, models_text, transform=ax1.transAxes, 
                ha='center', va='top', fontsize=9, wrap=True)
        ax1.text(0.5, -0.20, f'Equation: {equation}', transform=ax1.transAxes,
                ha='center', va='top', fontsize=9, style='italic')
    else:
        ax1.text(0.5, -0.12, f'Equation: {equation}', transform=ax1.transAxes,
                ha='center', va='top', fontsize=9, style='italic')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training loss over time
    ax2.plot(losses, color='purple', linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Loss Over Time', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale to better see convergence
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12 if full_model_names else 0.08, hspace=0.45)
    
    output_filename = f"{title}_parameter_optimization_results.png"
    output_file_path = os.path.join(OPTIMIZED_FOLDER, output_filename)
    fig.savefig(output_file_path, dpi=150)
    plt.close(fig)
    print(f"Optimization plot saved to {output_file_path}")
    
    # Save optimization results to CSV (only Date and Model_Optimized)
    output_csv_filename = f"{title}_optimized.csv"
    output_csv_path = os.path.join(OPTIMIZED_FOLDER, output_csv_filename)
    with open(output_csv_path, 'w', encoding='utf-8') as f:
        f.write("Date|Model_Optimized\n")
        for i, date in enumerate(matching_dates):
            f.write(f"{date}|{optimized_predictions[i]:.10f}\n")
    
    print(f"Optimization results CSV saved to {output_csv_path}")
    
    return best_params, best_loss

##

def calculate_and_save_daily_averages(title, csv_data):
    # Check if interpolated CSV already exists
    output_csv_filename = f"{title}_daily_averages_interpolated.csv"
    output_csv_path = os.path.join(INTERPOLATED_FOLDER, output_csv_filename)
    
    if os.path.exists(output_csv_path):
        print(f"✓ Loading existing interpolated data from: {output_csv_path}")
        # Load existing data
        interpolated_date_averages = {}
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    date_str = parts[0]
                    avg = float(parts[1])
                    interpolated_date_averages[date_str] = avg
        print(f"✓ Loaded {len(interpolated_date_averages)} interpolated daily values from cache")
        return interpolated_date_averages
    
    print(f"Computing new interpolation for: {title}")
    
    # Collect grades by date
    grades_by_date = defaultdict(list)
    for row in csv_data:
        date = row[0]
        try:
            grade = float(row[2])
            grades_by_date[date].append(grade)
        except (ValueError, IndexError):
            pass
    
    # Calculate averages for existing dates
    date_averages = {}
    sorted_dates = sorted(grades_by_date.keys(), key=_date_key)
    
    for date in sorted_dates:
        if grades_by_date[date]:
            avg = sum(grades_by_date[date]) / len(grades_by_date[date])
            date_averages[date] = avg
    
    if not date_averages:
        print(f"No valid data found for {title}")
        return None
    
    print(f"Calculated averages for {len(date_averages)} dates with data")
    
    # Convert dates to datetime objects for interpolation
    date_objects = [datetime.strptime(date, '%d/%m/%Y') for date in sorted_dates]
    
    # Create numeric representation (days since first date)
    first_date = date_objects[0]
    x_days = [(d - first_date).days for d in date_objects]
    y_averages = [date_averages[date] for date in sorted_dates]
    
    # Generate all dates in the range
    last_date = date_objects[-1]
    all_dates = []
    current_date = first_date
    while current_date <= last_date:
        all_dates.append(current_date)
        current_date += timedelta(days=1)
    
    all_x_days = [(d - first_date).days for d in all_dates]
    
    # Apply Cubic Spline Interpolation
    cs = CubicSpline(x_days, y_averages)
    interpolated_averages = cs(all_x_days)
    
    # Create dictionary with all dates (interpolated)
    interpolated_date_averages = {}
    for date, avg in zip(all_dates, interpolated_averages):
        date_str = date.strftime('%d/%m/%Y')
        interpolated_date_averages[date_str] = float(avg)
    
    print(f"Interpolated to {len(interpolated_date_averages)} daily values (from {len(date_averages)} original dates)")
    
    # Save to CSV (with interpolated values)
    output_csv_filename = f"{title}_daily_averages_interpolated.csv"
    output_csv_path = os.path.join(INTERPOLATED_FOLDER, output_csv_filename)
    with open(output_csv_path, 'w', encoding='utf-8') as f:
        f.write("Date|Average Grade\n")
        for date in all_dates:
            date_str = date.strftime('%d/%m/%Y')
            f.write(f"{date_str}|{interpolated_date_averages[date_str]:.10f}\n")
    
    print(f"CSV saved to {output_csv_path}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot interpolated data
    all_date_strings = [d.strftime('%d/%m/%Y') for d in all_dates]
    all_averages = [interpolated_date_averages[d] for d in all_date_strings]
    x_positions = list(range(len(all_dates)))
    
    ax.plot(x_positions, all_averages, color='lightblue', linewidth=1, label='Interpolated', alpha=0.7)
    
    # Overlay original data points
    original_x_positions = [all_date_strings.index(date) for date in sorted_dates]
    original_averages = [date_averages[date] for date in sorted_dates]
    ax.scatter(original_x_positions, original_averages, color='blue', s=20, zorder=5, label='Original Data', edgecolors='black', linewidths=0.5)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Set x-axis labels (show subset of dates for readability)
    num_labels = min(40, len(all_dates))
    step = max(1, len(all_dates) // num_labels)
    tick_positions = x_positions[::step]
    tick_labels = all_date_strings[::step]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Grade')
    ax.set_title(f'Average Grade by Date - {title} \n{len(all_dates)} total days | {len(date_averages)} original dates (Cubic Spline Interpolation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    output_plot_filename = f"{title}_daily_averages_interpolated.png"
    output_plot_path = os.path.join(INTERPOLATED_FOLDER, output_plot_filename)
    fig.savefig(output_plot_path, dpi=150)
    plt.close(fig)
    
    print(f"Plot saved to {output_plot_path}")
    
    return interpolated_date_averages

##

EQS_TO_TRY = [
    "x + a",
    "b * x + a",
    "c * x**2 + b * x + a",
    "d * x**3 + c * x**2 + b * x + a",
]

def main():
    # Check if required CSV files exist
    models_csv_path = os.path.join(INPUT_FOLDER, "appended_results.csv")
    human_csv_path = os.path.join(INPUT_FOLDER, "appended_results_human.csv")
    
    if not os.path.exists(models_csv_path):
        print(f"ERROR: Models CSV file not found: {models_csv_path}")
        print("Please ensure 'appended_results.csv' exists in the 'info_and_graphs' folder.")
        return
    
    if not os.path.exists(human_csv_path):
        print(f"ERROR: Human CSV file not found: {human_csv_path}")
        print("Please ensure 'appended_results_human.csv' exists in the 'info_and_graphs' folder.")
        return
    
    # Load CSV data
    models_csv = get_csv_lines("appended_results.csv")
    human_csv = get_csv_lines("appended_results_human.csv")
    
    if models_csv is None or human_csv is None:
        print("ERROR: Failed to load CSV files.")
        return
    
    # Calculate human averages once (same for all runs)
    print("\n" + "="*80)
    print("CALCULATING HUMAN AVERAGES (GROUND TRUTH)")
    print("="*80)
    human_avgs = calculate_and_save_daily_averages(
        title="human",
        csv_data=human_csv
    )
    
    # Get all unique models
    all_models = get_unique_models(models_csv)
    print(f"\n{'='*80}")
    print(f"Found {len(all_models)} unique models:")
    for i, model in enumerate(all_models, 1):
        print(f"  {i}. {model}")
    print("="*80)
    
    # Generate all combinations of models (1 to all models)
    all_combinations = []
    for r in range(1, len(all_models) + 1):
        all_combinations.extend(combinations(all_models, r))
    
    print(f"\nTotal combinations to test: {len(all_combinations)}")
    print(f"Equations to test per combination: {len(EQS_TO_TRY)}")
    print(f"Total runs: {len(all_combinations) * len(EQS_TO_TRY)}")
    print("="*80 + "\n")
    
    # Store all results
    all_results = []
    
    # Test each combination with each equation
    run_number = 0
    total_runs = len(all_combinations) * len(EQS_TO_TRY)
    
    for combo_idx, model_combo in enumerate(all_combinations, 1):
        # Create title for this combination using model names separated by dashes
        model_short_names = [m.split('/')[-1] for m in model_combo]
        combo_name = "-".join(model_short_names)
        
        # If the name is too long, truncate and add indicator
        if len(combo_name) > 150:
            combo_name = f"combo_{combo_idx}_{len(model_combo)}_models"
        
        # Filter CSV to only include selected models
        filtered_csv = filter_csv_by_models(models_csv, model_combo)
        
        print(f"\n{'='*80}")
        print(f"COMBINATION {combo_idx}/{len(all_combinations)}: {len(model_combo)} model(s)")
        print(f"Models: {', '.join(model_combo)}")
        print(f"Filtered data: {len(filtered_csv)} rows")
        print("="*80)
        
        # Calculate model averages for this combination
        model_avgs = calculate_and_save_daily_averages(
            title=f"model_{combo_name}",
            csv_data=filtered_csv
        )
        
        if model_avgs is None:
            print(f"⚠️ No valid data for combination {combo_idx}, skipping...")
            continue
        
        # Test each equation
        for eq_idx, equation in enumerate(EQS_TO_TRY, 1):
            run_number += 1
            print(f"\n{'-'*80}")
            print(f"RUN {run_number}/{total_runs}")
            print(f"Combination {combo_idx}/{len(all_combinations)} | Equation {eq_idx}/{len(EQS_TO_TRY)}")
            print(f"Equation: {equation}")
            print(f"-"*80)
            
            # Create unique title for this run: model-model-model--eq1
            run_title = f"{combo_name}--eq{eq_idx}"
            
            # Optimize
            best_params, mse = optimize_equation_parameters(
                title=run_title,
                equation=equation,
                model_avgs=model_avgs,
                human_avgs=human_avgs,
                full_model_names=list(model_combo),
                epochs=1000,
                lr=0.01
            )
            
            if best_params is not None and mse is not None:
                # Store results
                result = {
                    'models': ', '.join(model_combo),
                    'num_models': len(model_combo),
                    'equation': equation,
                    'mse': mse,
                    'params': best_params,
                    'run_title': run_title
                }
                all_results.append(result)
                print(f"✓ Run {run_number} completed successfully")
            else:
                print(f"✗ Run {run_number} failed")
    
    # Sort results by MSE (lowest first)
    all_results.sort(key=lambda x: x['mse'])
    
    # Save comprehensive results CSV
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("="*80)
    
    results_csv_path = os.path.join(OUTPUT_FOLDER, "all_optimization_results.csv")
    with open(results_csv_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("Rank|Models|Num_Models|Equation|MSE|Parameters|Run_Title\n")
        
        # Write results
        for rank, result in enumerate(all_results, 1):
            params_str = "; ".join([f"{k}={v:.10f}" for k, v in result['params'].items()])
            f.write(f"{rank}|{result['models']}|{result['num_models']}|")
            f.write(f"{result['equation']}|{result['mse']:.10f}|")
            f.write(f"{params_str}|{result['run_title']}\n")
    
    print(f"✓ Results saved to: {results_csv_path}")
    print(f"\nTotal successful runs: {len(all_results)}/{total_runs}")
    
    # Print top 10 results
    print(f"\n{'='*80}")
    print("TOP 10 RESULTS (Lowest MSE)")
    print("="*80)
    for i, result in enumerate(all_results[:10], 1):
        print(f"\n{i}. MSE: {result['mse']:.10f}")
        print(f"   Models ({result['num_models']}): {result['models']}")
        print(f"   Equation: {result['equation']}")
        print(f"   Parameters: {result['params']}")
    
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE!")
    print("="*80)

        
if __name__ == "__main__":
    main()