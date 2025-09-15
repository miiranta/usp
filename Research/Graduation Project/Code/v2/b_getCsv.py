import os

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'results'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'csvs'))

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ====================================

class Info:
    def __init__(self):
        self.folder = ""
        self.subdir = ""
        self.unparsed_lines = []
        # simple fields
        self.model = ""
        self.types = ""
        self.filter = ""
        # data stats
        self.count = None
        self.min = None
        self.max = None
        self.mean = None
        self.std = None
        # histogram stats
        self.bin_count = None
        self.shannon_entropy = None
        self.desequilibrium = None
        self.complexity = None

# ====================================

def get_subdirs():
    folders = [f for f in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, f))]
    
    non_empty_subdirs = []
    for folder in folders:
        subdirs = [d for d in os.listdir(os.path.join(INPUT_FOLDER, folder)) if os.path.isdir(os.path.join(INPUT_FOLDER, folder, d))]
        for subdir in subdirs:
            full_path = os.path.join(INPUT_FOLDER, folder, subdir)
            if os.listdir(full_path):
                info = Info()
                info.folder = folder
                info.subdir = subdir
                non_empty_subdirs.append(info)
    
    return non_empty_subdirs

def get_info_from_txt(info):
    in_dir = os.path.join(INPUT_FOLDER, info.folder, info.subdir)
    txt_files = [f for f in os.listdir(in_dir) if f.endswith('.txt')]
    with open(os.path.join(in_dir, txt_files[0]), 'r') as file:
        lines = file.readlines()
    info.unparsed_lines = lines

    parse_lines(info)

def write_csv(infos):
    out_path = os.path.join(OUTPUT_FOLDER, "summary.csv")
    with open(out_path, 'w') as file:
        headers = [
            'model', 'types', 'filter',
            'count', 'min', 'max', 'mean', 'std',
            'bin_count', 'shannon_entropy', 'desequilibrium', 'complexity'
        ]
        file.write(','.join(headers) + '\n')
        
        for info in infos:
            values = [
                info.model, info.types, info.filter,
                str(info.count), str(info.min), str(info.max), str(info.mean), str(info.std),
                str(info.bin_count), str(info.shannon_entropy), str(info.desequilibrium), str(info.complexity)
            ]
            file.write(','.join(values) + '\n')

# ====================================

def parse_lines(info):
    lines = info.unparsed_lines

    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        low = s.lower()
        if low.startswith('model:'):
            info.model = s.split(':', 1)[1].strip()
            continue
        if low.startswith('types:'):
            info.types = '-'.join([t.strip() for t in s.split(':', 1)[1].split(',') if t.strip()])
            continue
        if low.startswith('filter:'):
            info.filter = s.split(':', 1)[1].strip()
            continue

        if s.startswith('>'):
            s2 = s.lstrip('> ').strip()
        else:
            s2 = s

        if ':' in s2:
            key, val = s2.split(':', 1)
            key = key.strip().lower()
            val = val.strip()
            num = None
            
            try:
                if '.' in val or 'e' in val.lower():
                    num = float(val, precision=64)
                else:
                    num = int(val)
            except Exception:
                num = None

            if key == 'count':
                info.count = num if num is not None else val
            elif key == 'min':
                info.min = num if num is not None else val
            elif key == 'max':
                info.max = num if num is not None else val
            elif key == 'mean':
                info.mean = num if num is not None else val
            elif key == 'standard deviation':
                info.std = num if num is not None else val
            elif key == 'bin count':
                info.bin_count = num if num is not None else val
            elif key == 'shannon entropy':
                info.shannon_entropy = num if num is not None else val
            elif key == 'desequilibrium':
                info.desequilibrium = num if num is not None else val
            elif key == 'complexity':
                info.complexity = num if num is not None else val

# ====================================

def main():
    dirs = get_subdirs()
    
    for info in dirs:
        get_info_from_txt(info)
        
    write_csv(dirs)
    
if __name__ == "__main__":
    main()

