import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModel
import torch
import matplotlib.pyplot as plt
import numpy as np

class ModelWeights:
	def __init__(self, name):
		self.name = name
		self.count = 0

		self.max_weight = None
		self.min_weight = None

		self.bins = 1000
		self.histograms = [
			np.zeros(self.bins, dtype=int),
			np.zeros(self.bins, dtype=float),
			np.zeros(self.bins, dtype=int),
			np.zeros(self.bins, dtype=float),
		]
  
		self.count_outside_ranges = dict()

	def add_weights(self, weights):
		if weights.size == 0:
			return
     
		# Count weights
		self.count += weights.size

		# Update min and max weights
		if self.min_weight is None:
			self.min_weight = np.min(weights)
		else:
			self.min_weight = min(self.min_weight, np.min(weights))

		if self.max_weight is None:
			self.max_weight = np.max(weights)
		else:
			self.max_weight = max(self.max_weight, np.max(weights))
   
		ranges_to_check = [3, 5, 10, 50, 100, 300, 500, 1000, 2000, 3000]
		for r in ranges_to_check:
			if np.max(weights) > r or np.min(weights) < -r:
				outside_range = np.sum((weights > r) | (weights < -r))
				if r not in self.count_outside_ranges:
					self.count_outside_ranges[r] = 0
				self.count_outside_ranges[r] += outside_range
   
	def add_weights_histogram(self, weights):
		if weights.size == 0:
			return

		counts, _ = np.histogram(weights, bins=self.bins, range=(self.min_weight, self.max_weight), density=False)
		self.histograms[0] += counts
  
		counts, _ = np.histogram(weights, bins=self.bins, range=(self.min_weight, self.max_weight), density=True)
		self.histograms[1] += counts
  
		tanh_weights = np.tanh(weights)
		counts, _ = np.histogram(tanh_weights, bins=self.bins, range=(-1, 1), density=False)
		self.histograms[2] += counts
  
		counts, _ = np.histogram(tanh_weights, bins=self.bins, range=(-1, 1), density=True)
		self.histograms[3] += counts

	def plot_histogram(self):
		if self.count == 0:
			print("No weights to plot.")
			return
		
		fig, axs = plt.subplots(2, 2, figsize=(15, 10))
		fig.suptitle(f"Weight Histograms for {self.name}", fontsize=16)
  
		titles = [
			"Histogram of Weights (Counts)",
			"Histogram of Weights (Density)",
			"Histogram of Tanh(Weights) (Counts)",
			"Histogram of Tanh(Weights) (Density)"
		]
		xlabels = [
			"Weights",
			"Weights",
			"Tanh(Weights)",
			"Tanh(Weights)"
		]
		ranges = [
			(self.min_weight, self.max_weight),
			(self.min_weight, self.max_weight),
			(-1, 1),
			(-1, 1)
		]
  
		for i, ax in enumerate(axs.flat):
			ax.bar(
				np.linspace(ranges[i][0], ranges[i][1], self.bins),
				self.histograms[i],
				width=(ranges[i][1] - ranges[i][0]) / self.bins,
				color='blue',
				alpha=0.7
			)
			ax.set_title(titles[i])
			ax.set_xlabel(xlabels[i])
			ax.set_ylabel("Counts" if i % 2 == 0 else "Density")
			ax.grid(True)
   
		plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  
		SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
		plt.savefig(os.path.join(SCRIPT_DIR, f"{self.name}_histogram.png"))
		plt.close()
   
def param_to_numpy(param):
    if param is None:
        return np.array([])
    t = param.detach()
    if t.numel() == 0:
        return np.array([])
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    arr = t.view(-1).numpy()
    return arr
 
def main():
	# Load model
	model_name = "openai/gpt-oss-20b"
	model = AutoModel.from_pretrained(
		model_name,
		device_map="cpu",
		low_cpu_mem_usage=True
	)
 
	# Initialize full array of weights
	modelWeights = ModelWeights(model_name.replace("/", "_"))
 
	# Extract weights
	for param in model.parameters():
		array_of_weights = param_to_numpy(param)
		modelWeights.add_weights(array_of_weights)
  
	# Histogram
	for param in model.parameters():
		array_of_weights = param_to_numpy(param)
		modelWeights.add_weights_histogram(array_of_weights)
	modelWeights.plot_histogram()

	# Count parameters
	param_count_torch = sum(p.numel() for p in model.parameters())
	print(f"Weight count (by Torch): {param_count_torch}")
	print(f"Weight count (by NumPy): {modelWeights.count}")
 
	# Print min and max weights
	print(f"Min weight: {modelWeights.min_weight}")
	print(f"Max weight: {modelWeights.max_weight}")
 
	# Print debug info
	print(f"Count outside range: {modelWeights.count_outside_ranges}")
 
if __name__ == "__main__":
	main()

