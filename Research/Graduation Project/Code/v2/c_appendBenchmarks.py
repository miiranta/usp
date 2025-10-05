import os
import csv

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'csvs'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'appended_csvs'))

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def main():
    benchmarks_path = os.path.join(INPUT_FOLDER, 'benchmarks.csv')
    summary_path = os.path.join(INPUT_FOLDER, 'summary.csv')
    
    if not os.path.exists(benchmarks_path):
        print(f"benchmarks.csv not found in {INPUT_FOLDER}")
        return
    
    if not os.path.exists(summary_path):
        print(f"summary.csv not found in {INPUT_FOLDER}")
        return
    
    # Read benchmarks.csv (using | as delimiter)
    benchmarks_data = {}
    benchmark_columns = []
    with open(benchmarks_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        # Find the model column (could be 'MODELS', 'model', 'Model', etc.)
        model_col = None
        for col in reader.fieldnames:
            if col.upper() == 'MODELS' or col.lower() == 'model':
                model_col = col
                break
        
        if model_col is None:
            print(f"Could not find model column in benchmarks.csv. Available columns: {reader.fieldnames}")
            return
        
        benchmark_columns = [col for col in reader.fieldnames if col != model_col]
        
        for row in reader:
            model = row[model_col]
            benchmarks_data[model] = {col: row[col] for col in benchmark_columns}
    
    # Read summary.csv and append benchmark data
    output_rows = []
    with open(summary_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        summary_columns = reader.fieldnames
        
        for row in reader:
            model = row['model']
            # Add benchmark data if model exists in benchmarks
            if model in benchmarks_data:
                row.update(benchmarks_data[model])
            else:
                # Add empty values for benchmarks if model not found
                for col in benchmark_columns:
                    row[col] = ''
            output_rows.append(row)
    
    # Write output
    output_path = os.path.join(OUTPUT_FOLDER, 'appended_benchmarks.csv')
    output_columns = summary_columns + benchmark_columns
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"Appended CSV saved to {output_path}")
    print(f"Merged {len(output_rows)} rows with {len(benchmark_columns)} benchmark columns")

if __name__ == '__main__':
    main()