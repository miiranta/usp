import os
import time
import torch
import numpy as np
from transformers import AutoModel
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""

MODEL_NAME = 'default'
MODEL_FOLDER = 'model'
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, 'results', MODEL_FOLDER, MODEL_NAME)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

MODEL_DATA_ARRAYS = dict() # param_type -> Data

class Data:
    DATA = None
    
    COUNT = 0
    MIN = None
    MAX = None
    
    MEAN = None
    STANDARD_DEVIATION = None
    
class Histogram:
    BINS = np.array([])
    PROBS = np.array([])
    
    SHANNON_ENTROPY = None
    DESEQUILIBRIUM = None
    COMPLEXITY = None

# ==================================== DATA

def calc_bin_amount(data): # Freedman-Diaconis rule
    q75, q25 = np.percentile(data.DATA, [75 ,25])
    iqr = q75 - q25
    bin_width = 2 * iqr * (len(data.DATA) ** (-1/3))
    if bin_width == 0:
        print("Error: bin_width is 0")
        exit(1)
    bin_amount = int(np.ceil((data.MAX - data.MIN) / bin_width))
    return max(bin_amount, 1)
    
def calc_data_stats(data):
    data.COUNT = len(data.DATA)
    data.MIN = np.min(data.DATA)
    data.MAX = np.max(data.DATA)
    
    data.MEAN = np.mean(data.DATA)
    data.STANDARD_DEVIATION = np.std(data.DATA)

# ==================================== HISTOGRAM

def calc_histogram(data, histogram):
    bin_amount = calc_bin_amount(data)
    bins = np.linspace(data.MIN, data.MAX, bin_amount + 1)
    counts, _ = np.histogram(data.DATA, bins=bins)
    probs = counts / data.COUNT
    histogram.BINS = bins
    histogram.PROBS = probs
    
def calc_histogram_stats(histogram):
    histogram.SHANNON_ENTROPY = calc_shannon_entropy(histogram.PROBS)
    histogram.DESEQUILIBRIUM = calc_desequilibrium(histogram.PROBS)
    histogram.COMPLEXITY = calc_complexity(histogram.SHANNON_ENTROPY, histogram.DESEQUILIBRIUM)

def plot_histogram(histogram):
    plt.figure()
    bin_centers = (histogram.BINS[:-1] + histogram.BINS[1:]) / 2
    plt.bar(bin_centers, histogram.PROBS, width=np.diff(histogram.BINS), edgecolor='black', align='center')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.title('Histogram')
    plt.grid(True)
    safe_model_name = MODEL_NAME.replace('/', '_')
    path = os.path.join(OUTPUT_FOLDER, f'{safe_model_name}_histogram.png')
    plt.savefig(path)
    plt.close()

# ====================================

def remove_data_outliers(data, sigma=0):
    if sigma <= 0:
        return data
    
    filtered_data = Data()
    filtered_data.DATA = data.DATA[
        (data.DATA >= data.MEAN - sigma * data.STANDARD_DEVIATION) &
        (data.DATA <= data.MEAN + sigma * data.STANDARD_DEVIATION)
    ]
    calc_data_stats(filtered_data)
    return filtered_data

# ====================================

def calc_shannon_entropy(probs): # H = -Σ p(x) log(p(x))
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def calc_desequilibrium(probs): # D = Σ (p(x) - 1/n)^2
    n = len(probs)
    uniform_prob = 1.0 / n
    return np.sum((probs - uniform_prob) ** 2)

def calc_complexity(H, D): # C = H * D
    return H * D

# ====================================

def param_to_numpy(param):
	if param is None:
		return torch.tensor([])
	t = param.detach()
	if t.numel() == 0:
		return torch.tensor([])
	if t.dtype in (torch.bfloat16, torch.float16):
		t = t.to(torch.float32)
	return t.view(-1).numpy()

def classify_param_name(name: str) -> str:
    lname = name.lower()
    # embeddings
    if 'embed' in lname or 'embedding' in lname:
        return 'embedding'
    # layernorm / norm
    if 'layernorm' in lname or 'ln' in lname or 'norm' in lname:
        return 'norm'
    # typical head / output / classifier modules
    if 'head' in lname or 'lm_head' in lname or 'classifier' in lname or 'pooler' in lname or 'out_proj' in lname:
        return 'head'
    # suffix-based
    if 'bias' in lname:
        return 'bias'
    if 'weight' in lname:
        return 'weight'
    # fallback
    return 'other'

# ====================================

def combinations(types):
    if len(types) == 0:
        return [[]]
    first, *rest = types
    without_first = combinations(rest)
    with_first = [[first] + combo for combo in without_first]
    return with_first + without_first

def set_model_name(model_folder, model_name):
    global MODEL_NAME, OUTPUT_FOLDER, MODEL_FOLDER
    MODEL_NAME = model_name
    MODEL_FOLDER = model_folder
    OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, 'results', MODEL_FOLDER, MODEL_NAME)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

def write_down(text):
    safe_model_name = MODEL_NAME.replace('/', '_')
    path = os.path.join(OUTPUT_FOLDER, f'{safe_model_name}.txt')
    with open(path, 'a') as f:
        f.write(text + '\n')

def write_down_data_stats(data):
    write_down("Data Stats:")
    write_down(f" > Count: {data.COUNT}")
    write_down(f" > Min: {data.MIN}")
    write_down(f" > Max: {data.MAX}")
    write_down(f" > Mean: {data.MEAN}")
    write_down(f" > Standard Deviation: {data.STANDARD_DEVIATION}")

def write_down_histogram(histogram):
    write_down("Bins:")
    write_down(np.array2string(histogram.BINS, separator=', '))
    write_down("Probs:")
    write_down(np.array2string(histogram.PROBS, separator=', '))

def write_down_histogram_stats(histogram):
    write_down("Histogram Stats:")
    write_down(f" > Shannon Entropy: {histogram.SHANNON_ENTROPY}")
    write_down(f" > Desequilibrium: {histogram.DESEQUILIBRIUM}")
    write_down(f" > Complexity: {histogram.COMPLEXITY}")

def write_down_all(data, histogram):
    write_down("=== DATA STATS ===")
    write_down_data_stats(data)
    write_down("\n=== HISTOGRAM STATS ===")
    write_down_histogram_stats(histogram)
    write_down("\n=== HISTOGRAM ===")
    write_down_histogram(histogram)

# ==================================== MAIN

MODELS_TO_TEST = [
    'openai/gpt-oss-20b',
]
TYPES_TO_TEST = ['weight', 'bias', 'norm', 'embedding', 'head', 'other'] # Parameter types to analyze
FILTERS_TO_TEST = [0, 1, 2, 3, 4] # Number of standard deviations for outlier removal

def main():
    global MODEL_DATA_ARRAYS
    
    for model in MODELS_TO_TEST:
        
        start_timer = time.time()
        
        print(f"Loading model {model}...")
        model = AutoModel.from_pretrained(
            model,
            device_map = "cpu",
        )
        
        # Tally sizes per parameter type
        print("Step 1: Tallying parameter sizes per type...")
        sizes = {t: 0 for t in TYPES_TO_TEST}
        for name, param in model.named_parameters():
            t = classify_param_name(name)
            sizes[t] += param.numel()

        # Allocate contiguous numpy arrays per type
        print("Step 2: Allocating contiguous arrays...")
        MODEL_DATA_ARRAYS = {t: Data() for t in TYPES_TO_TEST}
        for t in TYPES_TO_TEST:
            MODEL_DATA_ARRAYS[t].DATA = np.empty(sizes[t], dtype=np.float32)

        # Copy flattened params directly into preallocated buffers
        print("Step 3: Copying data into preallocated arrays...")
        for name, param in model.named_parameters():
            t = classify_param_name(name)
            array = param_to_numpy(param)
            offset = MODEL_DATA_ARRAYS[t].COUNT
            length = len(array)
            MODEL_DATA_ARRAYS[t].DATA[offset:offset+length] = array
            MODEL_DATA_ARRAYS[t].COUNT += length
        
        # How many parameters in total? # Just a check
        total_params = sum(len(MODEL_DATA_ARRAYS[t].DATA) for t in TYPES_TO_TEST)
        print(f"Total parameters: {total_params}")
        for t in TYPES_TO_TEST:
            print(f"-{t}: {len(MODEL_DATA_ARRAYS[t].DATA)}")

        for types in combinations(TYPES_TO_TEST):
            if len(types) == 0:
                continue

            for filter in FILTERS_TO_TEST:
                
                testing_name = f"{model}_" + "-".join(types) + f"_filter{filter}"
                set_model_name(model.replace('/', '_'), testing_name)
                print(f" > Testing {testing_name}...")
                
                # Already done?
                safe_model_name = MODEL_NAME.replace('/', '_')
                path = os.path.join(OUTPUT_FOLDER, f'{safe_model_name}.txt')
                if os.path.exists(path):
                    print(" > > Already done, skipping.")
                    continue
                
                # Merge data
                print(" > > Merging data...")
                merged_data = Data()
                merged_data.DATA = np.concatenate([MODEL_DATA_ARRAYS[t].DATA for t in types], axis=0)
                
                print(" > > Filtering...")
                calc_data_stats(merged_data)
                filtered_data = remove_data_outliers(merged_data, sigma=filter)
                
                print(" > > Calculating histogram...")
                histogram = Histogram()
                calc_histogram(filtered_data, histogram)
                calc_histogram_stats(histogram)
                plot_histogram(histogram)
                
                print(" > > Writing down results...")
                write_down_all(filtered_data, histogram)
                
                print(" > > Done.")
                
        del model
        del MODEL_DATA_ARRAYS
        MODEL_DATA_ARRAYS = dict()
        
        total_time = time.time() - start_timer
        print(f"Total execution time: {total_time:.2f} seconds.")
        
if __name__ == "__main__":
    main()

