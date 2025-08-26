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
		self.histogram = np.zeros(self.bins, dtype=int)
		self.bins_probabilities = np.zeros(self.bins, dtype=float)
		self.bins_probabilities_calc = False
  
		self.shannon_information_k = 1
		self.shannon_information = 0
		self.shannon_information_calc = False
  
		self.desequilibrium = 0
		self.desequilibrium_calc = False
  
		self.lmc_complexity = 0
  
		self.count_outside_ranges = dict()

	def calculate_bins_probabilities(self):
		if self.count == 0:
			return
		for n in range(self.bins):
			self.bins_probabilities[n] = self.histogram[n] / self.count
   
		self.bins_probabilities_calc = True

	def calculate_shannon_information(self):
		if not self.bins_probabilities_calc:
			self.calculate_bins_probabilities()
     
		self.shannon_information = 0
		for p in self.bins_probabilities:
			if p > 0:
				self.shannon_information += p * np.log2(p)
			#else:
				#print_and_write(model_name,"Warning: Probability is zero, skipping in Shannon calculation.")

		self.shannon_information *= (-1) * self.shannon_information_k
  
		self.shannon_information_calc = True

	def calculate_desequilibrium(self):
		if not self.bins_probabilities_calc:
			self.calculate_bins_probabilities()
	 
		self.desequilibrium = 0
		for p in self.bins_probabilities:
			self.desequilibrium += (p - 1 / self.bins) ** 2
   
			#if p == 0:
				#print_and_write(model_name,"Warning: Probability is zero.")
    
		self.desequilibrium_calc = True
    
	def calculate_lmc_complexity(self):
		if not self.shannon_information_calc:
			self.calculate_shannon_information()
		if not self.desequilibrium_calc:
			self.calculate_desequilibrium()
		self.lmc_complexity = self.shannon_information * self.desequilibrium
		
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
   
		# Count weights outside certain ranges for debugging
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
		self.histogram += counts

	def plot_histogram(self):
		if self.count == 0:
			print_and_write(model_name,"No weights to plot.")
			return
		
		fig, ax = plt.subplots(figsize=(12, 6))
		fig.suptitle(f"Weight Histogram for {self.name}", fontsize=16)
  
		title = "Histogram of Weights (Counts)"
		xlabel = "Weights"
		range_ = (self.min_weight, self.max_weight)
		bin_edges = np.linspace(range_[0], range_[1], self.bins + 1)
		bin_width = bin_edges[1] - bin_edges[0]
		bin_lefts = bin_edges[:-1]

		ax.bar(
			bin_lefts,
			self.histogram,
			width=bin_width,
			align='edge',
			color='blue',
			alpha=0.7,
			edgecolor='black',
			linewidth=0.5,
		)
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		ax.set_ylabel("Counts")
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
 
def print_and_write(filename, text):
	print(text)

	safe_filename = filename.replace(os.sep, "_")
	safe_filename = safe_filename.replace("/", "_")
	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	filepath = os.path.join(SCRIPT_DIR, safe_filename + "_results.txt")

	with open(filepath, "a") as f:
		f.write(str(text) + "\n")
    
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
 
 	# Calculate information theory metrics
	modelWeights.calculate_bins_probabilities()
	modelWeights.calculate_shannon_information()
	modelWeights.calculate_desequilibrium()
	modelWeights.calculate_lmc_complexity()

	# Count parameters
	param_count_torch = sum(p.numel() for p in model.parameters())
	print_and_write(model_name,f"Weight count (by Torch): {param_count_torch}")
	print_and_write(model_name,f"Weight count (by NumPy): {modelWeights.count}")
 
	# Print min and max weights
	print_and_write(model_name,f"Min weight: {modelWeights.min_weight}")
	print_and_write(model_name,f"Max weight: {modelWeights.max_weight}")
 
	# Print information theory metrics
	print_and_write(model_name,f"Bins probabilities: {modelWeights.bins_probabilities}")
	print_and_write(model_name,f"Shannon information: {modelWeights.shannon_information}")
	print_and_write(model_name,f"Desequilibrium: {modelWeights.desequilibrium}")
	print_and_write(model_name,f"LMC Complexity: {modelWeights.lmc_complexity}")
 
	# Histogram data
	print_and_write(model_name,f"Histogram: {modelWeights.histogram}")
 
	# Print debug info
	print_and_write(model_name,f"Count outside range: {modelWeights.count_outside_ranges}")
 
def test():
	model_name = "test"
	modelWeights = ModelWeights(model_name)
	weights = np.array([-1, 0, 1, 0, 0])
	weights2 = np.array([-2, 0, 2])
 
	modelWeights.add_weights(weights)
	modelWeights.add_weights(weights2)
	modelWeights.add_weights_histogram(weights)
	modelWeights.add_weights_histogram(weights2)
 
	modelWeights.plot_histogram()
 
	print_and_write(model_name,modelWeights.histogram)
 
	modelWeights.calculate_bins_probabilities()
	print_and_write(model_name,f"Bins probabilities: {modelWeights.bins_probabilities}")
 
	modelWeights.calculate_shannon_information()
	print_and_write(model_name,modelWeights.shannon_information)
 
	modelWeights.calculate_desequilibrium()
	print_and_write(model_name,modelWeights.desequilibrium)
 
	modelWeights.calculate_lmc_complexity()
	print_and_write(model_name,modelWeights.lmc_complexity)

if __name__ == "__main__":
	main()
	#test()
 
 
 
 

