from transformers import AutoModel
import torch
import matplotlib.pyplot as plt

# Plot histogram
def plot_param_histogram(parameters, bins=50):
	import numpy as np
	plt.ion()
	fig, ax = plt.subplots(figsize=(10, 6))
	all_values = []
	min_val, max_val = None, None
	for i, p in enumerate(parameters):
		vals = p.detach().cpu().flatten().numpy()
		all_values.extend(vals)
		# Update min/max
		cur_min, cur_max = vals.min(), vals.max()
		min_val = cur_min if min_val is None else min(min_val, cur_min)
		max_val = cur_max if max_val is None else max(max_val, cur_max)
		if (i+1) % 5 == 0 or i == 0:
			ax.clear()
			ax.hist(all_values, bins=bins, range=(min_val, max_val), color='blue', alpha=0.7)
			ax.set_title(f"Histogram of Model Parameters ({bins} bins, {i+1} tensors)")
			ax.set_xlabel("Parameter Value")
			ax.set_ylabel("Frequency")
			ax.grid(True)
			plt.pause(0.01)
	# Final plot
	ax.clear()
	ax.hist(all_values, bins=bins, range=(min_val, max_val), color='blue', alpha=0.7)
	ax.set_title(f"Histogram of Model Parameters ({bins} bins, all tensors)")
	ax.set_xlabel("Parameter Value")
	ax.set_ylabel("Frequency")
	ax.grid(True)
	plt.ioff()
	plt.show(block=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model_name = "openai/gpt-oss-20b"
model = AutoModel.from_pretrained(
	model_name,
	device_map="auto",
	offload_folder="offload",
	low_cpu_mem_usage=True
)

# Count parameters
param_count = sum(p.numel() for p in model.parameters())
print(f"Total parameter count: {param_count}")

# Plot histogram of parameters
plot_param_histogram(model.parameters(), bins=50)
