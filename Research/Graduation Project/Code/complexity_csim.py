import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModel
import torch
import matplotlib.pyplot as plt
import numpy as np

class ModelWeights:
	def __init__(self, name, param_type="weights"):
		self.name = name
		self.param_type = param_type  # "weights" or "bias"
		self.count = 0
  
		self.sum = 0
		self.sum_sq = 0
		self.remove_outliers_std = 3 #0 for no filter

		self.max_weight = None
		self.min_weight = None

		self.bins = None
		self.histogram = None
		self.bins_probabilities = None
		self.bins_probabilities_calc = False
  
		self.shannon_information_k = 1
		self.shannon_information = 0
		self.shannon_information_calc = False
  
		self.desequilibrium = 0
		self.desequilibrium_calc = False
  
		self.lmc_complexity = 0

		self.weight_arrays = []

	def set_bins(self):
		if self.count > 0:
			self.mean = self.sum / self.count
			self.std = np.sqrt(self.sum_sq / self.count - self.mean**2)
			self.bins = int(2 * (self.count ** (1/3))) # Rice's Rule
		else:
			self.mean = 0
			self.std = 0
			self.bins = 1
   
		self.histogram = np.zeros(self.bins, dtype=int)
		self.bins_probabilities = np.zeros(self.bins, dtype=float)

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

		self.shannon_information *= (-1) * self.shannon_information_k
  
		self.shannon_information_calc = True

	def calculate_desequilibrium(self):
		if not self.bins_probabilities_calc:
			self.calculate_bins_probabilities()
	 
		self.desequilibrium = 0
		for p in self.bins_probabilities:
			self.desequilibrium += (p - 1 / self.bins) ** 2
   
		self.desequilibrium_calc = True
    
	def calculate_lmc_complexity(self):
		if not self.shannon_information_calc:
			self.calculate_shannon_information()
		if not self.desequilibrium_calc:
			self.calculate_desequilibrium()
		self.lmc_complexity = self.shannon_information * self.desequilibrium
		
	def add_weights(self, weights):
		if weights.numel() == 0:
			return
     
		# Convert to numpy and store for later use
		if isinstance(weights, torch.Tensor):
			numpy_weights = weights.numpy()
		else:
			numpy_weights = weights
		self.weight_arrays.append(numpy_weights)
		
		# Count weights
		self.count += weights.numel()
		self.sum += torch.sum(weights).item()
		self.sum_sq += torch.sum(weights**2).item()

		# Update min and max weights
		if self.min_weight is None:
			self.min_weight = torch.min(weights).item()
		else:
			self.min_weight = min(self.min_weight, torch.min(weights).item())

		if self.max_weight is None:
			self.max_weight = torch.max(weights).item()
		else:
			self.max_weight = max(self.max_weight, torch.max(weights).item())
   
	def build_histogram_from_stored_arrays(self):
		if not self.weight_arrays:
			return
			
		all_weights = np.concatenate(self.weight_arrays)
		
		if self.remove_outliers_std > 0 and self.std > 0:
			print(f"Removing outliers beyond {self.remove_outliers_std} standard deviations for {self.param_type}.")
			sigma = self.remove_outliers_std
			lower = self.mean - sigma * self.std
			upper = self.mean + sigma * self.std
			mask = (all_weights >= lower) & (all_weights <= upper)
			filtered_weights = all_weights[mask]
		else:
			filtered_weights = all_weights
			
		counts, _ = np.histogram(filtered_weights, bins=self.bins, range=(self.min_weight, self.max_weight), density=False)
		self.histogram = counts

	def plot_histogram(self):
		if self.count == 0:
			print(f"No {self.param_type} to plot.")
			return
		
		fig, ax = plt.subplots(figsize=(12, 6))
		fig.suptitle(f"{self.param_type.title()} Histogram for {self.name}", fontsize=16)
  
		title = f"Histogram of {self.param_type.title()} (Counts)"
		xlabel = self.param_type.title()
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
		plt.savefig(os.path.join(SCRIPT_DIR, f"{self.name}_{self.param_type}_{self.remove_outliers_std}_histogram.png"))
		plt.close()

def param_to_tensor(param):
	if param is None:
		return torch.tensor([])
	t = param.detach()
	if t.numel() == 0:
		return torch.tensor([])
	if t.dtype in (torch.bfloat16, torch.float16):
		t = t.to(torch.float32)
	return t.view(-1)
 
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
	print("Loading model...")
	model_name = "openai/gpt-oss-20b"
	model = AutoModel.from_pretrained(
		model_name,
		device_map = "cpu",
	)
 
	# Initialize separate trackers for weights and biases
	model_name_clean = model_name.replace("/", "_")
	modelWeights = ModelWeights(model_name_clean, "weights")
	modelBiases = ModelWeights(model_name_clean, "bias")
 
	# Extract weights and biases
	print("Extracting weights and biases...")
	for name, param in model.named_parameters():
		array_of_params = param_to_tensor(param)
		
		if 'bias' in name.lower():
			modelBiases.add_weights(array_of_params)
		else:
			modelWeights.add_weights(array_of_params)
  
	# Set bins
	print("Setting bins...")
	modelWeights.set_bins()
	modelBiases.set_bins()
	
	# Build histograms from stored arrays
	print("Building histograms from stored arrays...")
	modelWeights.build_histogram_from_stored_arrays()
	modelBiases.build_histogram_from_stored_arrays()
	
	# Plot histograms
	print("Plotting histograms...")
	modelWeights.plot_histogram()
	modelBiases.plot_histogram()
 
	# Calculate information theory metrics for weights
	print("Calculating information theory metrics...")
	modelWeights.calculate_bins_probabilities()
	modelWeights.calculate_shannon_information()
	modelWeights.calculate_desequilibrium()
	modelWeights.calculate_lmc_complexity()
	
	# Calculate information theory metrics for biases
	modelBiases.calculate_bins_probabilities()
	modelBiases.calculate_shannon_information()
	modelBiases.calculate_desequilibrium()
	modelBiases.calculate_lmc_complexity()

	# Count parameters
	param_count_torch = sum(p.numel() for p in model.parameters())
	print_and_write(model_name, f"Total parameter count (by Torch): {param_count_torch}")
	print_and_write(model_name, f"Weight count (by NumPy): {modelWeights.count}")
	print_and_write(model_name, f"Bias count (by NumPy): {modelBiases.count}")
 
	# Print weights information
	print_and_write(model_name, "\n=== WEIGHTS INFORMATION ===")
	print_and_write(model_name, f"Number of bins for weights: {modelWeights.bins}")
	print_and_write(model_name, f"Mean weight: {modelWeights.mean}")
	print_and_write(model_name, f"Std weight: {modelWeights.std}")
	print_and_write(model_name, f"Min weight: {modelWeights.min_weight}")
	print_and_write(model_name, f"Max weight: {modelWeights.max_weight}")
	print_and_write(model_name, f"Weights bins probabilities: {modelWeights.bins_probabilities}")
	print_and_write(model_name, f"Weights Shannon information: {modelWeights.shannon_information}")
	print_and_write(model_name, f"Weights Desequilibrium: {modelWeights.desequilibrium}")
	print_and_write(model_name, f"Weights LMC Complexity: {modelWeights.lmc_complexity}")
	print_and_write(model_name, f"Weights histogram: {modelWeights.histogram}")
	
	# Print biases information
	print_and_write(model_name, "\n=== BIASES INFORMATION ===")
	print_and_write(model_name, f"Number of bins for biases: {modelBiases.bins}")
	print_and_write(model_name, f"Mean bias: {modelBiases.mean}")
	print_and_write(model_name, f"Std bias: {modelBiases.std}")
	print_and_write(model_name, f"Min bias: {modelBiases.min_weight}")
	print_and_write(model_name, f"Max bias: {modelBiases.max_weight}")
	print_and_write(model_name, f"Biases bins probabilities: {modelBiases.bins_probabilities}")
	print_and_write(model_name, f"Biases Shannon information: {modelBiases.shannon_information}")
	print_and_write(model_name, f"Biases Desequilibrium: {modelBiases.desequilibrium}")
	print_and_write(model_name, f"Biases LMC Complexity: {modelBiases.lmc_complexity}")
	print_and_write(model_name, f"Biases histogram: {modelBiases.histogram}")
 
def test():
	model_name = "test"
	modelWeights = ModelWeights(model_name, "weights")
	modelBiases = ModelWeights(model_name, "bias")
	
	weights = torch.tensor([-1, 0, 1, 0, 0], dtype=torch.float32)
	weights2 = torch.tensor([-2, 0, 2], dtype=torch.float32)
	biases = torch.tensor([0.1, -0.1, 0.05], dtype=torch.float32)
 
	# Add weights and biases
	modelWeights.add_weights(weights)
	modelWeights.add_weights(weights2)
	modelBiases.add_weights(biases)
	
	# Set bins
	modelWeights.set_bins()
	modelBiases.set_bins()
	
	# Build histograms from stored arrays
	modelWeights.build_histogram_from_stored_arrays()
	modelBiases.build_histogram_from_stored_arrays()
 
	modelWeights.plot_histogram()
	modelBiases.plot_histogram()
 
	print_and_write(model_name, f"Weights histogram: {modelWeights.histogram}")
	print_and_write(model_name, f"Mean weight: {modelWeights.mean}")
	print_and_write(model_name, f"Std weight: {modelWeights.std}")
	print_and_write(model_name, f"Biases histogram: {modelBiases.histogram}")
	print_and_write(model_name, f"Mean bias: {modelBiases.mean}")
	print_and_write(model_name, f"Std bias: {modelBiases.std}")
 
	modelWeights.calculate_bins_probabilities()
	print_and_write(model_name, f"Weights bins probabilities: {modelWeights.bins_probabilities}")
 
	modelWeights.calculate_shannon_information()
	print_and_write(model_name, f"Weights Shannon information: {modelWeights.shannon_information}")
 
	modelWeights.calculate_desequilibrium()
	print_and_write(model_name, f"Weights desequilibrium: {modelWeights.desequilibrium}")
 
	modelWeights.calculate_lmc_complexity()
	print_and_write(model_name, f"Weights LMC complexity: {modelWeights.lmc_complexity}")
	
	modelBiases.calculate_bins_probabilities()
	print_and_write(model_name, f"Biases bins probabilities: {modelBiases.bins_probabilities}")
 
	modelBiases.calculate_shannon_information()
	print_and_write(model_name, f"Biases Shannon information: {modelBiases.shannon_information}")
 
	modelBiases.calculate_desequilibrium()
	print_and_write(model_name, f"Biases desequilibrium: {modelBiases.desequilibrium}")
 
	modelBiases.calculate_lmc_complexity()
	print_and_write(model_name, f"Biases LMC complexity: {modelBiases.lmc_complexity}")

if __name__ == "__main__":
	main()
	#test()





