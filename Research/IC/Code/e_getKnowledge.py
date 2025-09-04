import csv
import os
import math
import matplotlib.pyplot as plt

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "csvs")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "info_and_graphs")

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
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        for row in reader:
            if row:
                data.append(row)
    return data

def get_appended_csvs():
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the input directory.")
        return []
    
    result_csv_data = []
    header = None
    for csv_file in csv_files:
        file_path = os.path.join(INPUT_FOLDER, csv_file)
        csv_data = read_csv(file_path)
        if csv_data:
            header = csv_data[0]
            csv_data = csv_data[1:]
            result_csv_data.extend(csv_data)
       
    result_csv_data.sort(key=lambda row: (row[1], _date_key(row[0])))
    
    # Check if any grade values are invalid (anything other than -1, 0, 1)
    valid_grades = {-1, 0, 1}
    for row in result_csv_data:
        try:
            grade = int(row[2])
            if grade not in valid_grades:
                print(f"Warning: Invalid grade value '{grade}' found in row: {row}\n")
        except ValueError:
            print(f"Warning: Non-integer grade value '{row[2]}' found in row: {row}\n")
    
       
    if header:
        result_csv_data.insert(0, header)

    return result_csv_data

##

def get_available_models():
    models = set()
    result_csv_data = get_appended_csvs()
    
    for row in result_csv_data[1:]:
        if len(row) > 1:
            model = row[1].strip()
            if model:
                models.add(model)
    
    return sorted(models)

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
    
    mean = sum(grades) / len(grades)
    variance = sum((x - mean) ** 2 for x in grades) / (len(grades) - 1)
    std_dev = math.sqrt(variance)
    
    return std_dev

## 

def plot_average_by_date_and_model(result_csv_data):
    dates_set = {row[0].strip() for row in result_csv_data[1:]}
    if not dates_set:
        print("No date rows to plot.")
        return

    dates_sorted = sorted(dates_set, key=_date_key)

    models = get_available_models()
    if not models:
        print("No models found to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(24, 6))
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

    ax.set_xlabel('Date')
    ax.set_ylabel('Average Grade')
    ax.set_title('Average Grade by Date')

    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)

    output_file_path = os.path.join(OUTPUT_FOLDER, "average_grade_by_date.png")
    fig.savefig(output_file_path)
    plt.close(fig)

##

def main():
    # Append CSVs

    result_csv_data = get_appended_csvs()

    output_file_path = os.path.join(OUTPUT_FOLDER, "appended_results.csv")
    with open(output_file_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter="|")
        for row in result_csv_data:
            writer.writerow(row)

    # Plot dates
    
    plot_average_by_date_and_model(result_csv_data)
    
    # Frequency and average
    
    freq_avg_file_path = os.path.join(OUTPUT_FOLDER, "info.txt")
    with open(freq_avg_file_path, 'w', encoding='utf-8') as f:
        
        available_model = get_available_models()
        
        f.write("--------\n")
        f.write(f"Found {len(result_csv_data)} rows in total across all CSV files.\n")
        if available_model:
            f.write(f"Available models: {', '.join(available_model)}\n")
        f.write("--------\n")
        
        # Average
        
        average_grade = get_grade_avarage_global(result_csv_data)
        
        if average_grade is not None:
            f.write(f"Average grade (global): {average_grade:.64f}\n")
            
        for model in available_model:
            average_grade_model = get_grade_avarage_model(result_csv_data, model)
            if average_grade_model is not None:
                f.write(f" > Average grade for {model}: {average_grade_model:.64f}\n")
        
        # Frequency
        
        frequency_grade = get_grade_frequency_global(result_csv_data)
        
        if frequency_grade:
            f.write("Grade frequency (global): ")
            for grade, frequency in sorted(frequency_grade.items()):
                f.write(f"[{grade}]: {frequency} ")
            f.write("\n")
        
        for model in available_model:
            frequency_grade_model = get_grade_frequency_model(result_csv_data, model)
            if frequency_grade_model:
                f.write(f" > Grade frequency for {model}: ")
                for grade, frequency in sorted(frequency_grade_model.items()):
                    f.write(f"[{grade}]: {frequency} ")
                f.write("\n")
                
        # Standard Deviation
        
        std_dev = get_grade_standard_deviation_global(result_csv_data)
        
        if std_dev is not None:
            f.write(f"Standard Deviation (global): {std_dev:.64f}\n")
            
        for model in available_model:
            std_dev_model = get_grade_standard_deviation_model(result_csv_data, model)
            if std_dev_model is not None:
                f.write(f" > Standard Deviation for {model}: {std_dev_model:.64f}")
            f.write("\n")
            
        f.write("--------\n")
    
    print("Processing completed.")
    
if __name__ == "__main__":
    main()
