import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import time
import torch
import torch.cuda
import numpy as np
from transformers import AutoModel
import matplotlib.pyplot as plt
from numba import jit, prange

if not torch.cuda.is_available():
    print("CUDA is not available. Please ensure you have a compatible GPU and the necessary drivers installed.")
    exit(1)
device = torch.device("cuda")

MODEL_NAME = 'default'
MODEL_FOLDER = 'model'
OUTPUT_FOLDER = None
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

MODEL_DATA_ARRAYS = dict() # param_type -> Data

class Data:
    def __init__(self):
        self._params = []  # raw parameter tensors

        self.COUNT = 0
        self.MIN = None
        self.MAX = None
        self.MEAN = 0.0
        self.STANDARD_DEVIATION = 0.0

    @property
    def DATA(self):
        return self

    def append(self, param_tensor: torch.Tensor):
        self._params.append(param_tensor)

    def __len__(self):
        return len(self._params)

    def __iter__(self):
        for p in self._params:
            yield self._param_to_tensor(p)

    def __getitem__(self, idx):
        return self._param_to_tensor(self._params[idx])

    def _param_to_tensor(self, param):
        if param is None:
            return torch.tensor([])
        t = param.detach()
        if t.numel() == 0:
            return torch.tensor([])
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.to(torch.float32)
        return t.view(-1).cpu()
    
class MergedData:
    def __init__(self):
        self._data_objects = [] # List of Data objects to merge
        
        self.COUNT = 0
        self.MIN = None
        self.MAX = None
        self.MEAN = 0.0
        self.STANDARD_DEVIATION = 0.0
        
    @property
    def DATA(self):
        return self
    
    def append(self, data_object: Data):
        self._data_objects.append(data_object)
                
    def __getitem__(self, index):
        current_index = 0
        for obj in self._data_objects:
            for arr in obj.DATA:
                if current_index == index:
                    return arr
                current_index += 1
        raise IndexError("Index out of range")
    
    def __iter__(self):
        for obj in self._data_objects:
            for arr in obj.DATA:
                yield arr
    
    def __len__(self):
        return sum(len(obj.DATA) for obj in self._data_objects)
    
class FilteredData:
    def __init__(self, data_object, lower_bound, upper_bound):
        self._data_object = data_object
        self.lower = lower_bound
        self.upper = upper_bound

        self.COUNT = 0
        self.MIN = None
        self.MAX = None
        self.MEAN = 0.0
        self.STANDARD_DEVIATION = 0.0

    @property
    def DATA(self):
        return self

    def __len__(self):
        return len(self._data_object.DATA)

    def __iter__(self):
        for arr in self._data_object.DATA:
            if len(arr) == 0:
                continue
            filtered = self._filter_array(arr)
            if len(filtered) > 0:
                yield filtered

    def __getitem__(self, idx):
        arr = self._data_object.DATA[idx]
        if len(arr) == 0:
            return arr
        return self._filter_array(arr)
        
    def _filter_array(self, arr):
        arr_gpu = arr.to(device)
        mask = (arr_gpu >= self.lower) & (arr_gpu <= self.upper)
        return arr_gpu[mask].cpu()
    
class Histogram:
    def __init__(self):
        self.BINS = 0
        self.HIST = None
        self.PROBS = None
        self.DATA_MIN = None
        self.DATA_MAX = None
        self.BIN_WIDTH = None
        
        self.SHANNON_ENTROPY = 0.0
        self.DESEQUILIBRIUM = 0.0
        self.COMPLEXITY = 0.0
  
# ==================================== DATA

def calc_bin_amount(data): # Freedman-Diaconis rule
    n = data.COUNT
    if n == 0:
        return 1
        
    sample_size = min(100000, n)

    data_arrays = [arr.numpy() for arr in data.DATA]
    samples_np = sample_values(data_arrays, sample_size)
    samples = samples_np[~np.isnan(samples_np)].tolist()
    
    if len(samples) == 0:
        return 1
        
    sample_tensor = torch.tensor(samples, device=device)
    q75 = torch.quantile(sample_tensor, 0.75).item()
    q25 = torch.quantile(sample_tensor, 0.25).item()
    iqr = q75 - q25
    
    bin_width = 2 * iqr * (n ** (-1/3))
    if bin_width == 0:
        print("Error: bin_width is 0")
        exit(1)
        
    bin_amount = int(torch.ceil(torch.tensor((data.MAX - data.MIN) / bin_width)).item())
    return max(bin_amount, 1)
    
def calc_data_stats(data):
    data.COUNT = 0
    data.MIN = None
    data.MAX = None
    
    data.MEAN = 0.0
    data.STANDARD_DEVIATION = 0.0
    
    total_sum = 0.0
    total_sq_sum = 0.0
    for arr_tensor in data.DATA:
        data.COUNT += len(arr_tensor)
        
        arr_min = torch.min(arr_tensor).item()
        arr_max = torch.max(arr_tensor).item()
        if data.MIN is None or arr_min < data.MIN:
            data.MIN = arr_min
        if data.MAX is None or arr_max > data.MAX:
            data.MAX = arr_max
            
        total_sum += torch.sum(arr_tensor).item()
        total_sq_sum += torch.sum(arr_tensor ** 2).item()
    
    if data.COUNT == 0:
        data.MEAN = 0.0
        data.STANDARD_DEVIATION = 0.0
        return
        
    data.MEAN = total_sum / data.COUNT
    variance = (total_sq_sum / data.COUNT) - (data.MEAN ** 2)
    data.STANDARD_DEVIATION = variance ** 0.5

@jit(parallel=True)
def sample_values(data_arrays, sample_size):
    samples = np.empty(sample_size, dtype=np.float32)
    for i in prange(sample_size):
        random_array_idx = np.random.randint(0, len(data_arrays))
        random_array = data_arrays[random_array_idx]
        if len(random_array) == 0:
            samples[i] = np.nan
        else:
            random_value_idx = np.random.randint(0, len(random_array))
            samples[i] = random_array[random_value_idx]
    return samples

# ==================================== HISTOGRAM

def calc_histogram(data, histogram):
    bin_amount = calc_bin_amount(data)
    if bin_amount <= 0:
        print("Error: bin_amount is 0")
        exit(1)
    histogram.BINS = bin_amount
    histogram.DATA_MIN = data.MIN
    histogram.DATA_MAX = data.MAX
    print(f" > > > > Bin amount: {bin_amount}")
    
    bin_width = (data.MAX - data.MIN) / bin_amount
    histogram.BIN_WIDTH = bin_width
    counts = torch.zeros(bin_amount, dtype=torch.long, device=device)
    
    for arr_tensor in data.DATA:
        arr_gpu = arr_tensor.to(device)
        bin_indices = ((arr_gpu - data.MIN) / bin_width).long()
        bin_indices = torch.clamp(bin_indices, 0, bin_amount - 1)
        counts += torch.bincount(bin_indices, minlength=bin_amount)
        
    # Convert counts to probabilities
    total_count = torch.sum(counts).item()
    counts_np = counts.cpu().numpy()
    histogram.HIST = counts_np
    histogram.PROBS = counts_np / total_count if total_count > 0 else counts_np
    
def calc_histogram_stats(histogram):
    histogram.SHANNON_ENTROPY = calc_shannon_entropy(histogram.PROBS)
    histogram.DESEQUILIBRIUM = calc_desequilibrium(histogram.PROBS)
    histogram.COMPLEXITY = calc_complexity(histogram.SHANNON_ENTROPY, histogram.DESEQUILIBRIUM)

def plot_histogram(histogram):
    if histogram.HIST is None or len(histogram.HIST) == 0:
        print("Warning: Empty histogram, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Parameter Histogram for {MODEL_NAME}", fontsize=16)
    
    bin_edges = np.linspace(histogram.DATA_MIN, histogram.DATA_MAX, histogram.BINS + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_lefts = bin_edges[:-1]
    
    max_plot_bins = 10000
    if len(histogram.HIST) > max_plot_bins:
        downsample_factor = len(histogram.HIST) // max_plot_bins
        
        downsampled_hist = []
        downsampled_lefts = []

        for i in range(0, len(histogram.HIST), downsample_factor):
            end_idx = min(i + downsample_factor, len(histogram.HIST))
            bin_group = histogram.HIST[i:end_idx]
            downsampled_hist.append(np.sum(bin_group))
            downsampled_lefts.append(bin_lefts[i])
        
        downsampled_bin_width = bin_width * downsample_factor
        
        ax.bar(
            downsampled_lefts,
            downsampled_hist,
            width=downsampled_bin_width,
            align='edge',
            color='blue',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
        )
        
        title = f"Histogram (Counts) - {len(downsampled_hist):,} bins (downsampled from {len(histogram.HIST):,})"
        print(f" > > > Plotting histogram with {len(downsampled_hist)} bins (downsampled from {len(histogram.HIST)})")
    else:
        ax.bar(
            bin_lefts,
            histogram.HIST,
            width=bin_width,
            align='edge',
            color='blue',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
        )
        
        title = f"Histogram (Counts) - {len(histogram.HIST):,} bins"
        print(f" > > > Plotting histogram with {len(histogram.HIST)} bins")
    
    ax.set_title(title)
    ax.set_xlabel("Value Range")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if OUTPUT_FOLDER is not None:
        safe_model_name = MODEL_NAME.replace('/', '-')
        path = os.path.join(OUTPUT_FOLDER, f'{safe_model_name}_histogram.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
    
    plt.close()

# ====================================

def remove_data_outliers(data, sigma=0):
    if sigma <= 0:
        return data
    lower_bound = data.MEAN - sigma * data.STANDARD_DEVIATION
    upper_bound = data.MEAN + sigma * data.STANDARD_DEVIATION
    return FilteredData(data, lower_bound, upper_bound)

# ====================================

def calc_shannon_entropy(probs): # H = -Σ p(x) log(p(x))
    probs_tensor = torch.from_numpy(probs).to(device)
    probs_tensor = probs_tensor[probs_tensor > 0]
    return -torch.sum(probs_tensor * torch.log(probs_tensor)).item()

def calc_desequilibrium(probs): # D = Σ (p(x) - 1/n)^2
    probs_tensor = torch.from_numpy(probs).to(device)
    n = len(probs_tensor)
    if n == 0:
        return 0.0
    uniform_prob = 1.0 / n
    return torch.sum((probs_tensor - uniform_prob) ** 2).item()

def calc_complexity(H, D): # C = H * D
    return H * D

# ====================================

def classify_param_name(name: str) -> str:
    lname = name.lower()
    # embeddings
    if 'embed' in lname or 'embedding' in lname:
        return 'embedding'
    # layernorm / norm
    if 'layernorm' in lname or 'ln' in lname or 'norm' in lname:
        return 'norm'
    # suffix-based
    if 'bias' in lname:
        return 'bias'
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
    MODEL_NAME = model_name.replace('/', '-')
    MODEL_FOLDER = model_folder
    OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, 'results', MODEL_FOLDER, MODEL_NAME)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

def write_down(text):
    global OUTPUT_FOLDER
    if OUTPUT_FOLDER is None:
        return
    safe_model_name = MODEL_NAME.replace('/', '-')
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
    hist_list = histogram.HIST.tolist()
    write_down('[' + ', '.join(map(str, hist_list)) + ']')
    write_down("Probs:")
    probs_list = histogram.PROBS.tolist()
    write_down('[' + ', '.join(map(str, probs_list)) + ']')

def write_down_histogram_stats(histogram):
    write_down("Histogram Stats:")
    write_down(f" > Bin Count: {histogram.BINS}")
    write_down(f" > Shannon Entropy: {histogram.SHANNON_ENTROPY}")
    write_down(f" > Desequilibrium: {histogram.DESEQUILIBRIUM}")
    write_down(f" > Complexity: {histogram.COMPLEXITY}")

def write_down_all(data, histogram):
    write_down("=== DATA STATS ===")
    write_down_data_stats(data)
    write_down("\n=== HISTOGRAM STATS ===")
    write_down_histogram_stats(histogram)
    #write_down("\n=== HISTOGRAM ===")
    #write_down_histogram(histogram)

# ==================================== MAIN

MODELS_TO_TEST = [
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b',
]
TYPES_TO_TEST = ['bias', 'norm', 'embedding', 'other'] # Parameter types to analyze
FILTERS_TO_TEST = [0, 1, 2, 3, 4] # Number of standard deviations for outlier removal

def main():
    global MODEL_DATA_ARRAYS
    
    
    
    # MODELS
    for model_name in MODELS_TO_TEST:
        start_timer = time.time()

        # Load model
        print(f"Loading model {model_name}...")
        
        original_cuda_available = torch.cuda.is_available
        original_get_device_capability = torch.cuda.get_device_capability
        original_get_device_properties = torch.cuda.get_device_properties
        torch.cuda.is_available = lambda: False
        torch.cuda.get_device_capability = lambda device=None: (0, 0)
        torch.cuda.get_device_properties = lambda device: None
        
        try:
            loaded_model = AutoModel.from_pretrained(
                model_name,
                device_map="cpu",
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_safetensors=True,
                attn_implementation="eager",
            )
            
            print(f"Successfully loaded {model_name}")
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Skipping this model...")
            continue
            
        finally:
            torch.cuda.is_available = original_cuda_available
            torch.cuda.get_device_capability = original_get_device_capability
            torch.cuda.get_device_properties = original_get_device_properties
        
            torch.cuda.empty_cache()
        
        # Allocate data arrays
        print("Allocating data arrays...")
        for t in TYPES_TO_TEST:
            MODEL_DATA_ARRAYS[t] = Data()
        
        # Extract parameters
        print("Storing references...")
        for name, param in loaded_model.named_parameters():
            ptype = classify_param_name(name)
            if ptype not in TYPES_TO_TEST:
                continue
            if param is None or param.numel() == 0:
                continue
            MODEL_DATA_ARRAYS[ptype].append(param)
            
        # Calculate data stats
        print("Calculating data stats...")
        for ptype in MODEL_DATA_ARRAYS:
            print(f" -{ptype}...")
            calc_data_stats(MODEL_DATA_ARRAYS[ptype])
        
        print("Value counts:")
        total_count = 0
        for ptype in MODEL_DATA_ARRAYS:
            total_count += MODEL_DATA_ARRAYS[ptype].COUNT
            print(f"-{ptype}: {MODEL_DATA_ARRAYS[ptype].COUNT}")
        print(f"--total: {total_count}")
        
        
        
        # TYPES
        for types in combinations(TYPES_TO_TEST):
            if len(types) == 0:
                continue
            
            # Merge data
            print(" > Merging data for types: " + ", ".join(types))
            merged_data = MergedData()
            data_to_merge = [MODEL_DATA_ARRAYS[t] for t in types]
            for d in data_to_merge:
                merged_data.append(d)
            
            # Calculate merged data stats
            print(" > Calculating merged data stats...")
            calc_data_stats(merged_data)
            print(f" > Merged data count: {merged_data.COUNT}")
            
            
            
            # FILTERS
            for filter in FILTERS_TO_TEST:
                
                testing_name = f"{model_name}_" + "-".join(types) + f"_filter-{filter}"
                set_model_name(model_name.replace('/', '-'), testing_name)
                print(f" > > Testing {testing_name}...")
                
                # Already done?
                safe_model_name = MODEL_NAME.replace('/', '-')
                path = os.path.join(OUTPUT_FOLDER, f'{safe_model_name}.txt')
                if os.path.exists(path):
                    print(" > > > Already done, skipping.")
                    continue

                # Filter outliers
                if filter > 0:
                    print(" > > > Filtering...")
                    filtered_data = remove_data_outliers(merged_data, sigma=filter)
                    calc_data_stats(filtered_data)
                else:
                    filtered_data = merged_data
                
                print(" > > > Calculating histogram...")
                histogram = Histogram()
                calc_histogram(filtered_data, histogram)
                calc_histogram_stats(histogram)
                plot_histogram(histogram)
                
                print(" > > > Writing down results...")
                write_down(f"Model: {model_name}")
                write_down(f"Types: {', '.join(types)}")
                write_down(f"Filter: {filter} sigma")
                write_down_all(filtered_data, histogram)
                
                print(" > > Done.")

        del loaded_model
        MODEL_DATA_ARRAYS = dict()
        
        # Force garbage collection and clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        total_time = time.time() - start_timer
        print(f"Total execution time for {model_name}: {total_time:.2f} seconds.")
        print("-" * 50)
        
if __name__ == "__main__":
    main()

