import os

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'csvs'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'appended_csvs'))

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    
# ====================================

def get_csv_files():    
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    return csv_files

def main():
    csv_files = get_csv_files()
    
    all_lines = []
    header = None
    
    for csv_file in csv_files:
        with open(os.path.join(INPUT_FOLDER, csv_file), 'r') as file:
            lines = file.readlines()
            if not header:
                header = lines[0]
            all_lines.extend(lines[1:])  # Skip header for subsequent files
    
    output_file_path = os.path.join(OUTPUT_FOLDER, 'appended_benchmarks.csv')
    with open(output_file_path, 'w') as outfile:
        outfile.write(header)
        outfile.writelines(all_lines)
    
    print(f"Appended CSV saved to {output_file_path}")

