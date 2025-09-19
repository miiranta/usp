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
from huggingface_hub import login
from dotenv import load_dotenv

if not torch.cuda.is_available():
    print("CUDA is not available. Please ensure you have a compatible GPU and the necessary drivers installed.")
    exit(1)
device = torch.device("cuda")

MODEL_NAME = 'default'
MODEL_FOLDER = 'model'
OUTPUT_FOLDER = None
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

MODEL_DATA_ARRAYS = dict() # param_type -> Data

load_dotenv(os.path.join(SCRIPT_FOLDER, '.env'))
login(token=os.getenv('hugging_face_token'))

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
        RESONABLE_MIN = -1e30
        RESONABLE_MAX = 1e30
        
        if param is None:
            return torch.tensor([])
        t = param.detach()
        if t.numel() == 0:
            return torch.tensor([])
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.to(torch.float32)
        t = torch.clamp(t, RESONABLE_MIN, RESONABLE_MAX)
        return t.view(-1).cpu()
    
    def _get_values_at_indices(self, indices):
        sizes = [p.numel() for p in self._params]
        cumsum = [0]
        for s in sizes:
            cumsum.append(cumsum[-1] + s)
        samples_values = []
        
        sorted_pairs = sorted(enumerate(indices), key=lambda x: x[1])
        current_param_idx = -1
        current_param_flat = None
        
        for orig_pos, idx in sorted_pairs:
            for i in range(len(cumsum)-1):
                if cumsum[i] <= idx < cumsum[i+1]:
                    local_idx = idx - cumsum[i]
                    
                    if i != current_param_idx:
                        param = self._params[i]
                        if param is None:
                            continue
                        current_param_flat = param.detach().cpu().view(-1)
                        current_param_idx = i
                    
                    value = current_param_flat[local_idx].item()
                    samples_values.append((orig_pos, value))
                    break
        
        samples_values.sort()
        if len(samples_values) == 0:
            return torch.tensor([], dtype=torch.float32)
        return torch.tensor([val for _, val in samples_values], dtype=torch.float32)
    
    def get_random_sample(self, sample_size):
        if self.COUNT == 0:
            return torch.tensor([], dtype=torch.float32)
        random_idxs = torch.randint(0, max(1, self.COUNT), (sample_size,), dtype=torch.long)
        return self._get_values_at_indices(random_idxs)
    
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
    
    def _get_values_at_indices(self, indices):
        sizes = [obj.COUNT for obj in self._data_objects]
        cumsum = [0]
        for s in sizes:
            cumsum.append(cumsum[-1] + s)
        samples_values = []
        
        sorted_pairs = sorted(enumerate(indices), key=lambda x: x[1])
        
        obj_indices = {}
        for orig_pos, idx in sorted_pairs:
            for i in range(len(cumsum)-1):
                if cumsum[i] <= idx < cumsum[i+1]:
                    local_idx = idx - cumsum[i]
                    if i not in obj_indices:
                        obj_indices[i] = []
                    obj_indices[i].append((orig_pos, local_idx))
                    break
        
        for obj_idx in sorted(obj_indices.keys()):
            local_requests = obj_indices[obj_idx]
            local_indices = [local_idx for _, local_idx in local_requests]
            
            sub_values = self._data_objects[obj_idx]._get_values_at_indices(torch.tensor(local_indices, dtype=torch.long))
            
            for (orig_pos, _), value in zip(local_requests, sub_values.tolist()):
                samples_values.append((orig_pos, float(value)))
    
        samples_values.sort()
        if len(samples_values) == 0:
            return torch.tensor([], dtype=torch.float32)
        return torch.tensor([val for _, val in samples_values], dtype=torch.float32)
    
    def get_random_sample(self, sample_size):
        if self.COUNT == 0:
            return torch.tensor([], dtype=torch.float32)
        random_idxs = torch.randint(0, max(1, self.COUNT), (sample_size,), dtype=torch.long)
        return self._get_values_at_indices(random_idxs)
    
class FilteredData:
    def __init__(self, data_object, lower_bound, upper_bound):
        self._data_object = data_object # MergedData
        if not isinstance(self._data_object, MergedData):
            raise ValueError("FilteredData currently only supports wrapping MergedData objects")
        
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
        mask = (arr >= float(self.lower)) & (arr <= float(self.upper))
        return arr[mask]
    
    def get_random_sample(self, sample_size):
        if self._data_object.COUNT == 0:
            return torch.tensor([], dtype=torch.float32)
        random_idxs = torch.randint(0, self._data_object.COUNT, (sample_size,), dtype=torch.long)
        
        complete = False
        result_values = []
        while not complete:
            values = self._data_object._get_values_at_indices(random_idxs)
            result_values.extend(self._filter_array(values).tolist())
            
            if len(result_values) >= sample_size:
                complete = True
            else:
                needed = sample_size - len(result_values)
                random_idxs = torch.randint(0, self._data_object.COUNT, (needed,), dtype=torch.long)
    
        return torch.tensor(result_values, dtype=torch.float32)
    
class Histogram:
    def __init__(self):
        self.BINS = 0
        self.HIST = None
        self.PROBS = None
        self.DATA_MIN = None
        self.DATA_MAX = None
        
        self.SHANNON_ENTROPY = 0.0
        self.DESEQUILIBRIUM = 0.0
        self.COMPLEXITY = 0.0
  
# ==================================== DATA

def calc_bin_amount(data): # Freedman-Diaconis rule
    n = data.COUNT
    if n == 0:
        return 1
        
    sample_size = min(100000, n)

    samples_tensor = data.get_random_sample(sample_size)
    if samples_tensor.numel() == 0:
        return 1
        
    samples = samples_tensor[~torch.isnan(samples_tensor)].tolist()
    
    if len(samples) == 0:
        return 1
        
    sample_tensor = torch.tensor(samples, device=device)
    q75 = torch.quantile(sample_tensor, 0.75).item()
    q25 = torch.quantile(sample_tensor, 0.25).item()
    iqr = q75 - q25
    
    bin_width = 2 * iqr * (n ** (-1/3))

    try:    
        torch_bin_amount = torch.ceil(torch.tensor((data.MAX - data.MIN) / bin_width)).item()
        bin_amount = int(torch_bin_amount)
    except:
        bin_amount = data.COUNT
        
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

# ==================================== HISTOGRAM

def calc_histogram(data, histogram):
    bin_amount = calc_bin_amount(data)
    if bin_amount <= 0:
        print("Error: bin_amount is 0")
        exit(1)

    BIN_CAP = 2000000000 # 2 billion ~8GB for float32
    if bin_amount > BIN_CAP:
        print(f" > > > > Capping bin amount {bin_amount} to total data count {BIN_CAP}")
        bin_amount = BIN_CAP

    histogram.BINS = bin_amount
    histogram.DATA_MIN = data.MIN
    histogram.DATA_MAX = data.MAX
    print(f" > > > > Bin amount: {bin_amount}")
    print(f" > > > > Data range: [{data.MIN}, {data.MAX}]")
    
    counts = torch.zeros(bin_amount, dtype=torch.long, device='cpu')

    CHUNK_ELEMS = 500000000 # 500 million ~2GB for float32
    chunk_list = []
    chunk_count = 0

    def _process_and_accumulate(concat_tensor):
        start = time.time()

        GPU_BATCH_ELEMS = 1000000000 # 1 billion ~4GB for float32
        processed_elems = 0
        total_elems = concat_tensor.numel()
        t_cpu = concat_tensor.view(-1)

        for i in range(0, total_elems, GPU_BATCH_ELEMS):
            sub = t_cpu[i:i+GPU_BATCH_ELEMS]
            sub_cuda = sub.to(device)
            hist = torch.histc(sub_cuda, bins=bin_amount, min=data.MIN, max=data.MAX)
            counts.add_(hist.to(torch.long).to('cpu'))
            del hist
            del sub_cuda
            processed_elems += sub.numel()

        elapsed = time.time() - start
        print(f" > > > > chunk processed: elems={processed_elems:,}, time={elapsed:.3f}s")
        return

    for arr_tensor in data.DATA:
        if arr_tensor.numel() == 0:
            continue
        t = arr_tensor.to('cpu')
        chunk_list.append(t)
        chunk_count += t.numel()

        if chunk_count >= CHUNK_ELEMS:
            concat = torch.cat(chunk_list)
            _process_and_accumulate(concat)
            del concat
            chunk_list = []
            chunk_count = 0

    if chunk_count > 0 and len(chunk_list) > 0:
        concat = torch.cat(chunk_list)
        _process_and_accumulate(concat)
        del concat

    # Convert counts to probabilities
    print(" > > > > Converting counts to probabilities...")
    total_count = int(counts.sum().item())
    histogram.HIST = counts.clone()
    histogram.PROBS = counts.float() / total_count if total_count > 0 else counts.float()
    
def calc_histogram_stats(histogram):
    histogram.SHANNON_ENTROPY = calc_shannon_entropy(histogram.PROBS)
    histogram.DESEQUILIBRIUM = calc_desequilibrium(histogram.PROBS)
    histogram.COMPLEXITY = calc_complexity(histogram.SHANNON_ENTROPY, histogram.DESEQUILIBRIUM)

def plot_histogram(histogram):
    if histogram.HIST is None or len(histogram.HIST) == 0:
        print("Warning: Empty histogram, skipping plot")
        return
    if histogram.DATA_MIN is None or histogram.DATA_MAX is None:
        print("Warning: Histogram has no data range (DATA_MIN/DATA_MAX None), skipping plot")
        return
    if not isinstance(histogram.BINS, int) or histogram.BINS <= 0:
        print("Warning: Histogram has invalid BINS value, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Parameter Histogram for {MODEL_NAME}", fontsize=16)
    
    bin_edges = np.linspace(histogram.DATA_MIN, histogram.DATA_MAX, histogram.BINS + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_lefts = bin_edges[:-1]
    
    max_plot_bins = 10000
    hist_tensor = histogram.HIST

    if hist_tensor.numel() > max_plot_bins:
        downsample_factor = hist_tensor.numel() // max_plot_bins + 1

        downsampled_hist = []
        downsampled_lefts = []

        for i in range(0, hist_tensor.numel(), downsample_factor):
            end_idx = min(i + downsample_factor, hist_tensor.numel())
            bin_group = hist_tensor[i:end_idx]
            downsampled_hist.append(int(bin_group.sum().item()))
            downsampled_lefts.append(float(bin_lefts[i]))

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

        title = f"Histogram (Counts) - {len(downsampled_hist):,} bins (downsampled from {hist_tensor.numel():,})"
        print(f" > > > > Plotting histogram with {len(downsampled_hist)} bins (downsampled from {hist_tensor.numel()})")
    else:
        ax.bar(
            bin_lefts,
            hist_tensor.cpu().numpy(),
            width=bin_width,
            align='edge',
            color='blue',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
        )

        title = f"Histogram (Counts) - {hist_tensor.numel():,} bins"
        print(f" > > > > Plotting histogram with {hist_tensor.numel()} bins")
    
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
    probs_tensor = probs.to(device)
    probs_tensor = probs_tensor[probs_tensor > 0]
    return -torch.sum(probs_tensor * torch.log(probs_tensor)).item()

def calc_desequilibrium(probs): # D = Σ (p(x) - 1/n)^2
    probs_tensor = probs.to(device)
    n = probs_tensor.numel()
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
    count = data.COUNT if getattr(data, 'COUNT', None) is not None else 0
    data_min = data.MIN if getattr(data, 'MIN', None) is not None else 0.0
    data_max = data.MAX if getattr(data, 'MAX', None) is not None else 0.0
    mean = data.MEAN if getattr(data, 'MEAN', None) is not None else 0.0
    std = data.STANDARD_DEVIATION if getattr(data, 'STANDARD_DEVIATION', None) is not None else 0.0

    write_down(f" > Count: {count}")
    write_down(f" > Min: {data_min}")
    write_down(f" > Max: {data_max}")
    write_down(f" > Mean: {mean}")
    write_down(f" > Standard Deviation: {std}")

def write_down_histogram(histogram):
    write_down("Bins:")
    hist_list = histogram.HIST.tolist()
    write_down('[' + ', '.join(map(str, hist_list)) + ']')
    write_down("Probs:")
    probs_list = histogram.PROBS.tolist()
    write_down('[' + ', '.join(map(str, probs_list)) + ']')

def write_down_histogram_stats(histogram):
    write_down("Histogram Stats:")
    bins = histogram.BINS if getattr(histogram, 'BINS', None) is not None else 0
    H = histogram.SHANNON_ENTROPY if getattr(histogram, 'SHANNON_ENTROPY', None) is not None else 0.0
    D = histogram.DESEQUILIBRIUM if getattr(histogram, 'DESEQUILIBRIUM', None) is not None else 0.0
    C = histogram.COMPLEXITY if getattr(histogram, 'COMPLEXITY', None) is not None else 0.0

    write_down(f" > Bin Count: {bins}")
    write_down(f" > Shannon Entropy: {H}")
    write_down(f" > Desequilibrium: {D}")
    write_down(f" > Complexity: {C}")

def write_down_all(data, histogram):
    write_down("=== DATA STATS ===")
    write_down_data_stats(data)
    write_down("\n=== HISTOGRAM STATS ===")
    write_down_histogram_stats(histogram)
    #write_down("\n=== HISTOGRAM ===")
    #write_down_histogram(histogram)

# ==================================== MAIN

# MODELS:

# 1. Selection of the biggest companies by marketcap that post open models for text generation on HuggingFace
# - OpenAI, Google, Meta, Microsoft

# 2. For each company, select every model that:
# - Is a transformer-based language model
# - Is open weights (gated or not)
# - Is text only
# - Is the original model (not a fine-tuned version for sppecific tasks)
# - Have less then 150B parameters (due to hardware limitations)

MODELS_TO_TEST = [
    # META
    #'meta-llama/Llama-4-Scout-17B-16E',
    
    #'meta-llama/Llama-3.2-3B',
    #'meta-llama/Llama-3.2-1B',
    
    #'meta-llama/Llama-3.1-70B',
    #'meta-llama/Llama-3.1-8B',
    
    #'meta-llama/Meta-Llama-3-70B',
    #'meta-llama/Meta-Llama-3-8B',
    
    'meta-llama/Llama-2-70b',
    'meta-llama/Llama-2-13b',
    'meta-llama/Llama-2-7b',
    
    # GOOGLE
    'google/gemma-3n-E4B',
    'google/gemma-3n-E2B',
    
    'google/gemma-3-27b-pt',
    'google/gemma-3-12b-pt',
    'google/gemma-3-4b-pt',
    'google/gemma-3-1b-pt',
    'google/gemma-3-270m',
    
    'google/gemma-2-27b',
    'google/gemma-2-9b',
    'google/gemma-2-2b',
    'google/gemma-2-2b-GGUF',
    
    'google/gemma-7b',
    'google/gemma-7b-GGUF',
    'google/gemma-2b',
    'google/gemma-2b-GGUF',
    
    'google/recurrentgemma-9b',
    'google/recurrentgemma-2b',
    
    # MICROSOFT
    'microsoft/Phi-4-mini-flash-reasoning',
    'microsoft/Phi-4-mini-reasoning',
    'microsoft/Phi-4-mini-reasoning-onnx',
    'microsoft/Phi-4-reasoning',
    'microsoft/Phi-4-reasoning-onnx',
    'microsoft/Phi-4-reasoning-plus',
    'microsoft/Phi-4-reasoning-plus-onnx',
    'microsoft/phi-4',
    'microsoft/phi-4-onnx',
    'microsoft/phi-4-gguf',
    
    'microsoft/phi-2',
    
    'microsoft/phi-1_5',
    
    'microsoft/phi-1',
    
    'microsoft/bitnet-b1.58-2B-4T',
    'microsoft/bitnet-b1.58-2B-4T-bf16',
    'microsoft/bitnet-b1.58-2B-4T-gguf',
    
    # OPENAI
    #'openai/gpt-oss-120b',
    #'openai/gpt-oss-20b',
    
    'openai-community/gpt2-xl',
    'openai-community/gpt2-large',
    'openai-community/gpt2-medium',
    'openai-community/gpt2',
    
    'openai-community/openai-gpt',
]

# TYPES:

# The types of parameters most commonly found in transformer-based language models
# - Biases
# - Normalization parameters (LayerNorm, BatchNorm, etc)
# - Embedding parameters (token embeddings, positional embeddings, etc)
# - Other parameters (weights of linear layers, attention layers, etc)

TYPES_TO_TEST = ['bias', 'norm', 'embedding', 'other']

# FILTERS:

# Different levels of outlier removal to test
# - 0: No filtering (infinite sigma)
# - 1: removes weights that differ more than 1 standard deviation from the mean
# - 2: '' 2 standard deviations
# - 3: '' 3 standard deviations
# - 4: '' 4 standard deviations

FILTERS_TO_TEST = [0, 1, 2, 3, 4]

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
                    print(" > > > Calculating filtered data stats...")
                    calc_data_stats(filtered_data)
                    print(f" > > > > Filtered data count: {filtered_data.COUNT} (removed {merged_data.COUNT - filtered_data.COUNT} | {(merged_data.COUNT - filtered_data.COUNT) / max(1, merged_data.COUNT) * 100:.2f}%)")
                else:
                    filtered_data = merged_data
                
                print(" > > > Calculating histogram...")
                histogram = Histogram()
                calc_histogram(filtered_data, histogram)
                print(" > > > Calculating histogram stats...")
                calc_histogram_stats(histogram)
                print(" > > > Plotting histogram...")
                plot_histogram(histogram)
                
                print(" > > > Writing down results...")
                write_down(f"Model: {model_name}")
                write_down(f"Types: {', '.join(types)}")
                write_down(f"Filter: {filter} sigma")
                write_down("")
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

