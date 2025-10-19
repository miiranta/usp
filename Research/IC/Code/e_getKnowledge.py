import csv
import os
import math
import time
import matplotlib.pyplot as plt
import requests
import datetime
from scipy import stats

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "csvs")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "info_and_graphs")

confidences_to_check = [0.95, 0.99]

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    
##

def _date_key(d):
    day, month, year = map(int, d.split('/'))
    return (year, month, day)

def read_csv(file_path):
    data = []
    data_human = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        for row in reader:
            if row:
                if 'human' in row[1].strip().lower():
                    data_human.append(row)
                else:
                    data.append(row)
    return data, data_human

def get_appended_csvs():
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the input directory.")
        return [], []
    
    result_csv_data = []
    result_human_data = []
    header = None
    for csv_file in csv_files:
        file_path = os.path.join(INPUT_FOLDER, csv_file)
        csv_data, csv_human_data = read_csv(file_path)
        if csv_data:
            header = csv_data[0]
            csv_data = csv_data[1:]
            result_csv_data.extend(csv_data)
        if csv_human_data:
            csv_human_data = csv_human_data[0:]
            result_human_data.extend(csv_human_data)
       
    result_csv_data.sort(key=lambda row: (row[1], _date_key(row[0])))
    result_human_data.sort(key=lambda row: (row[1], _date_key(row[0])))
    
    # Remove grades
    grades_to_remove = {'-2', '-3', '0'}
    for row in result_csv_data[:]:
        if row[2].strip() in grades_to_remove:
            result_csv_data.remove(row)
    for row in result_human_data[:]:
        if row[2].strip() in grades_to_remove:
            result_human_data.remove(row)
    
    # Check if any grade values are invalid (anything other than -1, 0, 1)
    valid_grades = {-1, 1}
    for row in result_csv_data:
        try:
            grade = int(row[2])
            if grade not in valid_grades:
                print(f"Warning: Invalid grade value '{grade}' found in row: {row}\n")
        except ValueError:
            print(f"Warning: Non-integer grade value '{row[2]}' found in row: {row}\n")
    for row in result_human_data:
        try:
            grade = int(row[2])
            if grade not in valid_grades:
                print(f"Warning: Invalid grade value '{grade}' found in row: {row}\n")
        except ValueError:
            print(f"Warning: Non-integer grade value '{row[2]}' found in row: {row}\n")
       
    if header:
        result_csv_data.insert(0, header)
        result_human_data.insert(0, header)

    return result_csv_data, result_human_data

##

def get_available_models(result_csv_data):
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

##

def get_grade_avarage_global(result_csv_data):
    total_grades = 0
    count = 0
    
    for row in result_csv_data[1:]: 
        try:
            grade = int(row[2]) 
            total_grades += grade
            count += 1
        except ValueError:
            print(f"Invalid grade value in row: {row}")
    
    if count == 0:
        print("No valid grades found.")
        return None
    
    average_grade = total_grades / count
    return average_grade

def get_grade_avarage_model(result_csv_data, model):
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
        print(f"No valid grades found for model: {model}")
        return None
    
    average_grade = total_grades / count
    return average_grade

def get_grade_frequency_global(result_csv_data):
    grade_frequency = {}
    
    for row in result_csv_data[1:]:
        try:
            grade = int(row[2])
            if grade in grade_frequency:
                grade_frequency[grade] += 1
            else:
                grade_frequency[grade] = 1
        except ValueError:
            print(f"Invalid grade value in row: {row}")
    
    return grade_frequency

def get_grade_frequency_model(result_csv_data, model):
    grade_frequency = {}
    
    for row in result_csv_data[1:]:
        if row[1].strip() == model:
            try:
                grade = int(row[2])
                if grade in grade_frequency:
                    grade_frequency[grade] += 1
                else:
                    grade_frequency[grade] = 1
            except ValueError:
                print(f"Invalid grade value in row: {row}")
    
    return grade_frequency

def get_grade_standard_deviation_global(result_csv_data):
    grades = []
    
    for row in result_csv_data[1:]:
        try:
            grade = int(row[2])
            grades.append(grade)
        except ValueError:
            print(f"Invalid grade value in row: {row}")
    
    if not grades:
        print("No valid grades found.")
        return None

    if len(grades) == 1:
        return 0.0

    mean = sum(grades) / len(grades)
    variance = sum((x - mean) ** 2 for x in grades) / (len(grades) - 1)
    std_dev = math.sqrt(variance)

    return std_dev

def get_grade_standard_deviation_model(result_csv_data, model):
    grades = []
    
    for row in result_csv_data[1:]:
        if row[1].strip() == model:
            try:
                grade = int(row[2])
                grades.append(grade)
            except ValueError:
                print(f"Invalid grade value in row: {row}")
    
    if not grades:
        print(f"No valid grades found for model: {model}")
        return None

    if len(grades) == 1:
        return 0.0

    mean = sum(grades) / len(grades)
    variance = sum((x - mean) ** 2 for x in grades) / (len(grades) - 1)
    std_dev = math.sqrt(variance)

    return std_dev

def get_grade_confidence_interval_global(result_csv_data, confidence=0.95):
    grades = []
    
    for row in result_csv_data[1:]:
        try:
            grade = int(row[2])
            grades.append(grade)
        except ValueError:
            print(f"Invalid grade value in row: {row}")
    
    if not grades:
        print("No valid grades found.")
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

def get_grade_confidence_interval_model(result_csv_data, model, confidence=0.95):
    grades = []
    
    for row in result_csv_data[1:]:
        if row[1].strip() == model:
            try:
                grade = int(row[2])
                grades.append(grade)
            except ValueError:
                print(f"Invalid grade value in row: {row}")
    
    if not grades:
        print(f"No valid grades found for model: {model}")
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

def get_grade_discordance_per_phrase(result_csv_data): # PER PHRASE 1/n Sum (grade - average_grade)^2
    phrase_dict = {}

    for row in result_csv_data[1:]:
        date = row[0].strip()
        phrase = row[3].strip()
        phrase_key = f"{phrase} | {date}"
        try:
            grade = int(row[2])
            if phrase_key in phrase_dict:
                phrase_dict[phrase_key].append(grade)
            else:
                phrase_dict[phrase_key] = [grade]
        except ValueError:
            print(f"Invalid grade value in row: {row}")

    discordance_per_phrase = {}
    phrase_counts = {}
    phrase_grades = {}

    for phrase, grades in phrase_dict.items():
        if not grades:
            continue
        mean = sum(grades) / len(grades)
        variance = sum((x - mean) ** 2 for x in grades) / len(grades)
        discordance_per_phrase[phrase] = variance
        phrase_counts[phrase] = len(grades)
        phrase_grades[phrase] = list(grades)

    discordance_per_phrase = dict(sorted(discordance_per_phrase.items(), key=lambda item: (-item[1], -phrase_counts[item[0]])))

    return discordance_per_phrase, phrase_counts, phrase_grades

## 

def plot_average_by_date_and_model(result_csv_data):
    dates_set = {row[0].strip() for row in result_csv_data[1:]}
    if not dates_set:
        print("No date rows to plot.")
        return

    dates_sorted = sorted(dates_set, key=_date_key)

    models = get_available_models(result_csv_data)
    if not models:
        print("No models found to plot.")
        return
    
    # Get SELIC data
    min_date = dates_sorted[0]
    max_date = dates_sorted[-1]
    selic_file = os.path.join(OUTPUT_FOLDER, "selic_data.csv")
    if os.path.exists(selic_file):
        print("Loading SELIC data from file...")
        
        # Load from file
        with open(selic_file, 'r', encoding='utf-8') as f:
            selic_reader = csv.reader(f)
            next(selic_reader)  # skip header
            selic_dict = {}
            for row in selic_reader:
                if row:
                    date = row[0]
                    value = float(row[1])
                    selic_dict[date] = value
    else:
        print("Fetching SELIC data from API...")
        
        # Fetch SELIC data in chunks of 10 years
        def parse_date(d):
            day, month, year = map(int, d.split('/'))
            return datetime.datetime(year, month, day)
        
        min_dt = parse_date(min_date)
        max_dt = parse_date(max_date)
        selic_dict = {}
        current_start = min_dt
        while current_start <= max_dt:
            current_end = min(current_start.replace(year=current_start.year + 10), max_dt)
            start_str = current_start.strftime('%d/%m/%Y')
            end_str = current_end.strftime('%d/%m/%Y')
            selic_url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=csv&dataInicial={start_str}&dataFinal={end_str}"
            try:
                response = requests.get(selic_url)
                response.raise_for_status()
                selic_data = response.text
                selic_reader = csv.reader(selic_data.splitlines(), delimiter=';')
                next(selic_reader)  # skip header
                for row in selic_reader:
                    if row:
                        date = row[0]
                        value = float(row[1].replace(',', '.'))
                        selic_dict[date] = value
            except Exception as e:
                print(f"Failed to fetch SELIC data for {start_str} to {end_str}: {e}")
            current_start = current_end + datetime.timedelta(days=1)
        
        # Save to file
        with open(selic_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['data', 'valor'])
            sorted_selic = sorted(selic_dict.items(), key=lambda x: _date_key(x[0]))
            for date, value in sorted_selic:
                writer.writerow([date, value])
    
    # Create SELIC series
    selic_series = []
    for d in dates_sorted:
        if d in selic_dict:
            selic_series.append(selic_dict[d])
        else:
            selic_series.append(float('nan'))
    
    # Plot without SELIC
    fig, ax = plt.subplots(figsize=(29, 9))
    x_positions = list(range(len(dates_sorted)))

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

    num_date_labels = 80
    display_indices = list(range(0, len(dates_sorted), max(1, len(dates_sorted) // num_date_labels)))

    tick_positions = [x_positions[i] for i in display_indices]
    tick_labels = [dates_sorted[i] for i in display_indices]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=60, ha='right')
    ax.tick_params(axis='x', pad=12)

    #ax.set_xlabel('Date')
    ax.set_ylabel('Average Grade')
    ax.set_title('Average Grade by Date')

    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(top=0.9)

    output_file_path = os.path.join(OUTPUT_FOLDER, "average_grade_by_date.png")
    fig.savefig(output_file_path)
    plt.close(fig)
    
    # Plot with SELIC
    fig, ax = plt.subplots(figsize=(29, 9))
    ax2 = ax.twinx()

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

    # Plot SELIC
    ax2.plot(x_positions, selic_series, color='red', linestyle='-', linewidth=2, label='SELIC (%)', alpha=0.7)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=60, ha='right')
    ax.tick_params(axis='x', pad=12)

    ax.set_ylabel('Average Grade')
    ax2.set_ylabel('SELIC (%)', color='black')
    ax.set_title('Average Grade and SELIC by Date')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(top=0.9)

    output_file_path_selic = os.path.join(OUTPUT_FOLDER, "average_grade_by_date_with_selic.png")
    fig.savefig(output_file_path_selic)
    plt.close(fig)

def plot_average_and_confidence_interval_by_model(result_csv_data, result_csv_data_human, confidence=0.95):
    models = get_available_models(result_csv_data + result_csv_data_human)
    if not models:
        print("No models found to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.8 + 4), 9))
    x_positions = list(range(len(models) + 2))  # +2 for global
    labels = models + ['Global (models only)', 'Global (models + human)']

    cmap = plt.get_cmap('tab20')

    for i, model in enumerate(models):
        avg = get_grade_avarage_model(result_csv_data + result_csv_data_human, model)
        ci = get_grade_confidence_interval_model(result_csv_data + result_csv_data_human, model, confidence)
        pos = i

        color = cmap(i % cmap.N)

        if avg is not None and ci is not None:
            lower = avg - ci[0]
            upper = ci[1] - avg

            ax.errorbar([pos], [avg], yerr=[[lower], [upper]], fmt='o', color=color, ecolor=color, capsize=5, markersize=7, linestyle='none')
            ax.axhline(avg, color=color, linestyle=':', linewidth=1, alpha=0.8, zorder=0)
        else:
            ax.plot([pos], [float('nan')], marker='o', color=color)

    avg_global = get_grade_avarage_global(result_csv_data)
    ci_global = get_grade_confidence_interval_global(result_csv_data, confidence)
    global_pos = len(models)
    global_color = cmap(len(models) % cmap.N)
    
    avg_global_human = get_grade_avarage_global(result_csv_data + result_csv_data_human)
    ci_global_human = get_grade_confidence_interval_global(result_csv_data + result_csv_data_human, confidence)
    global_pos_human = len(models) + 1
    global_color_human = cmap((len(models) + 1) % cmap.N)

    if avg_global is not None and ci_global is not None:
        lower_g = avg_global - ci_global[0]
        upper_g = ci_global[1] - avg_global
        ax.errorbar([global_pos], [avg_global], yerr=[[lower_g], [upper_g]], fmt='o',
                    color=global_color, ecolor=global_color, capsize=5, markersize=7, linestyle='none')
        ax.axhline(avg_global, color=global_color, linestyle=':', linewidth=1, alpha=0.8, zorder=0)
    else:
        ax.plot([global_pos], [float('nan')], marker='o', color=global_color)
    
    if avg_global_human is not None and ci_global_human is not None:
        lower_gh = avg_global_human - ci_global_human[0]
        upper_gh = ci_global_human[1] - avg_global_human
        ax.errorbar([global_pos_human], [avg_global_human], yerr=[[lower_gh], [upper_gh]], fmt='o',
                    color=global_color_human, ecolor=global_color_human, capsize=5, markersize=7, linestyle='none')
        ax.axhline(avg_global_human, color=global_color_human, linestyle=':', linewidth=1, alpha=0.8, zorder=0)
    else:
        ax.plot([global_pos_human], [float('nan')], marker='o', color=global_color_human)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Average Grade')
    ax.set_title(f"Average Grade by Model (Interval with {int(confidence*100)}% Confidence)")

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)

    plt.tight_layout()
    output_file_path = os.path.join(OUTPUT_FOLDER, "average_ci_by_model_" + str(int(confidence*100)) + "perc.png")
    fig.savefig(output_file_path)
    plt.close(fig)

def plot_discordance_histogram(discordance_dict, type_str):
    if not discordance_dict:
        print(f"No discordance data to plot for {type_str}.")
        return
    
    discordances = list(discordance_dict.values())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(discordances, bins=15, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Discordance')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Discordance Histogram ({type_str.capitalize()})")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_FOLDER, f"discordance_histogram_{type_str}.png")
    fig.savefig(output_path)
    plt.close(fig)

##

def main():
    global confidences_to_check
    
    # Append CSVs

    result_csv_data, result_csv_data_human = get_appended_csvs()

    output_file_path = os.path.join(OUTPUT_FOLDER, "appended_results.csv")
    with open(output_file_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter="|")
        for row in result_csv_data:
            writer.writerow(row)
            
    output_human_file_path = os.path.join(OUTPUT_FOLDER, "appended_results_human.csv")
    with open(output_human_file_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter="|")
        for row in result_csv_data_human:
            writer.writerow(row)

    # Plots
    
    plot_average_by_date_and_model(result_csv_data)
    
    for confidence in confidences_to_check:
        plot_average_and_confidence_interval_by_model(result_csv_data, result_csv_data_human, confidence)
    
    # Discordance histograms
    
    discordance_models, _, _ = get_grade_discordance_per_phrase(result_csv_data)
    plot_discordance_histogram(discordance_models, "models")
    
    discordance_human, _, _ = get_grade_discordance_per_phrase(result_csv_data_human)
    plot_discordance_histogram(discordance_human, "humans")
    
    # Info.txt
    
    freq_avg_file_path = os.path.join(OUTPUT_FOLDER, "info.txt")
    with open(freq_avg_file_path, 'w', encoding='utf-8') as f:
        
        available_model = get_available_models(result_csv_data + result_csv_data_human)

        f.write("--------\n")
        f.write("General Information:\n")
        f.write(f"Found {len(result_csv_data)} rows in total across all CSV files.\n")
        if available_model:
            f.write(f"Available models: {', '.join(available_model)}\n")
        unique_phrases = {row[3].strip() for row in result_csv_data[1:]}
        f.write(f"Unique phrases evaluated: {len(unique_phrases)}\n")    
        unique_dates = {row[0].strip() for row in result_csv_data[1:]}
        f.write(f"Unique dates evaluated: {len(unique_dates)}\n")
        max_rows_per_model = 0
        for model in available_model:
            model_rows = sum(1 for row in result_csv_data[1:] if row[1].strip() == model)
            if model_rows > max_rows_per_model:
                max_rows_per_model = model_rows
        f.write(f"Maximum number rows for a single model: {max_rows_per_model}\n")
        
        
        # Average
        
        f.write("--------\n")
        
        average_grade_models = get_grade_avarage_global(result_csv_data)
        if average_grade_models is not None:
            f.write(f"Average grade (global models only): {average_grade_models:.64f}\n")
            
        average_grade_all = get_grade_avarage_global(result_csv_data + result_csv_data_human)
        if average_grade_all is not None:
            f.write(f"Average grade (global models + human): {average_grade_all:.64f}\n")
            
        for model in available_model:
            average_grade_model = get_grade_avarage_model(result_csv_data + result_csv_data_human, model)
            if average_grade_model is not None:
                f.write(f" > Average grade for {model}: {average_grade_model:.64f}\n")
        
        # Frequency
        
        f.write("--------\n")
        
        frequency_grade_models = get_grade_frequency_global(result_csv_data)
        if frequency_grade_models:
            f.write("Grade frequency (global models only): ")
            for grade, frequency in sorted(frequency_grade_models.items()):
                f.write(f"[{grade}]: {frequency} ")
            f.write("\n")
        
        frequency_grade_all = get_grade_frequency_global(result_csv_data + result_csv_data_human)
        if frequency_grade_all:
            f.write("Grade frequency (global models + human): ")
            for grade, frequency in sorted(frequency_grade_all.items()):
                f.write(f"[{grade}]: {frequency} ")
            f.write("\n")
        
        for model in available_model:
            frequency_grade_model = get_grade_frequency_model(result_csv_data + result_csv_data_human, model)
            if frequency_grade_model:
                f.write(f" > Grade frequency for {model}: ")
                for grade, frequency in sorted(frequency_grade_model.items()):
                    f.write(f"[{grade}]: {frequency} ")
                f.write("\n")
                
        # Standard Deviation
        
        f.write("--------\n")
        
        std_dev_models = get_grade_standard_deviation_global(result_csv_data)
        if std_dev_models is not None:
            f.write(f"Standard Deviation (global models only): {std_dev_models:.64f}\n")
            
        std_dev_all = get_grade_standard_deviation_global(result_csv_data + result_csv_data_human)
        if std_dev_all is not None:
            f.write(f"Standard Deviation (global models + human): {std_dev_all:.64f}\n")
            
        for model in available_model:
            std_dev_model = get_grade_standard_deviation_model(result_csv_data + result_csv_data_human, model)
            if std_dev_model is not None:
                f.write(f" > Standard Deviation for {model}: {std_dev_model:.64f}")
            f.write("\n")
        
        # Confidence Interval (between models)
        
        for confidence in confidences_to_check:
            
            f.write("--------\n")
            
            conf_int_models = get_grade_confidence_interval_global(result_csv_data, confidence)
            if conf_int_models is not None:
                f.write(f"{int(confidence*100)}% Confidence Interval (global models only): ({conf_int_models[0]:.64f}, {conf_int_models[1]:.64f})\n")
            
            conf_int_all = get_grade_confidence_interval_global(result_csv_data + result_csv_data_human, confidence)
            if conf_int_all is not None:
                f.write(f"{int(confidence*100)}% Confidence Interval (global models + human): ({conf_int_all[0]:.64f}, {conf_int_all[1]:.64f})\n")
            
            for model in available_model:
                conf_int_model = get_grade_confidence_interval_model(result_csv_data + result_csv_data_human, model, confidence)
                if conf_int_model is not None:
                    f.write(f" > {int(confidence*100)}% Confidence Interval for {model}: ({conf_int_model[0]:.64f}, {conf_int_model[1]:.64f})\n")
       
        # Discordance per phrase (between models)

        f.write("--------\n")
        
        discordance_per_phrase, phrase_counts, phrase_grades = get_grade_discordance_per_phrase(result_csv_data)

        f.write("Top 100 phrases with highest discordance (variance):\n")
        for i, (phrase, discordance) in enumerate(discordance_per_phrase.items()):
            if i >= 100:
                break
            count = phrase_counts.get(phrase, 0)
            grades_list = phrase_grades.get(phrase, [])
            f.write(f"{i + 1}. Discordance: {discordance:.64f} | Count: {count} | Grades: {grades_list} | Phrase: {phrase}\n")

        f.write("--------\n")
        
        f.write("Top 100 phrases with lowest discordance (variance):\n")
        for i, (phrase, discordance) in enumerate(sorted(discordance_per_phrase.items(), key=lambda item: (item[1], -phrase_counts[item[0]]))):
            if i >= 100:
                break
            count = phrase_counts.get(phrase, 0)
            grades_list = phrase_grades.get(phrase, [])
            f.write(f"{i + 1}. Discordance: {discordance:.64f} | Count: {count} | Grades: {grades_list} | Phrase: {phrase}\n")

        f.write("--------\n\n")

        if result_csv_data_human:
            discordance_per_phrase_human, phrase_counts_human, phrase_grades_human = get_grade_discordance_per_phrase(result_csv_data_human)

            f.write("Top 100 phrases with highest discordance (variance) for humans:\n")
            for i, (phrase, discordance) in enumerate(discordance_per_phrase_human.items()):
                if i >= 100:
                    break
                count = phrase_counts_human.get(phrase, 0)
                grades_list = phrase_grades_human.get(phrase, [])
                f.write(f"{i + 1}. Discordance: {discordance:.64f} | Count: {count} | Grades: {grades_list} | Phrase: {phrase}\n")

            f.write("--------\n")
            
            f.write("Top 100 phrases with lowest discordance (variance) for humans:\n")
            for i, (phrase, discordance) in enumerate(sorted(discordance_per_phrase_human.items(), key=lambda item: (item[1], -phrase_counts_human[item[0]]))):
                if i >= 100:
                    break
                count = phrase_counts_human.get(phrase, 0)
                grades_list = phrase_grades_human.get(phrase, [])
                f.write(f"{i + 1}. Discordance: {discordance:.64f} | Count: {count} | Grades: {grades_list} | Phrase: {phrase}\n")

            f.write("--------\n\n")

    print("Processing completed.")
    
if __name__ == "__main__":
    main()
