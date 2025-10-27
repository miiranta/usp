import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('csvs/benchmarks.csv', sep='|')

# Set the MODELS column as index
df.set_index('MODELS', inplace=True)

# Filter out models that do not have any benchmarks
df = df[df.notna().any(axis=1)]

# Filter out gemma-3n models
df = df[~df.index.str.contains('gemma-3n', case=False, na=False)]

# Create a binary matrix: 1 for available (non-empty), 0 for not available (empty)
availability_matrix = df.notna().astype(int)

# Create a matrix with values: keep original values where available, NaN where not
value_matrix = df.copy()

# Create annotations: show values where available, empty string where not
annotations = df.astype(str).replace('nan', '')

# Set up the figure
plt.figure(figsize=(14, 12))

# Create a mask for missing values
mask = df.isna()

# Create a binary matrix for coloring: 1 for available, 0 for not available
binary_matrix = (~mask).astype(int)

# Create the heatmap with binary colors only
ax = sns.heatmap(
    binary_matrix,
    annot=annotations,
    fmt='',
    cmap=['#ff6b6b', '#90ee90'],  # Red for 0 (unavailable), Green for 1 (available)
    cbar=False,  # Remove color bar since we only have binary values
    linewidths=0.5,
    linecolor='gray',
    vmin=0,
    vmax=1
)

# Customize the plot
plt.title('Benchmark Availability Matrix', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Benchmarks', fontsize=12, fontweight='bold')
plt.ylabel('Models', fontsize=12, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
output_path = 'benchmarks_availability.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Show the plot
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Total models: {len(df)}")
print(f"Total benchmarks: {len(df.columns)}")
print(f"\nBenchmark coverage per model:")
coverage = (df.notna().sum(axis=1) / len(df.columns) * 100).round(2)
print(coverage.sort_values(ascending=False).head(10))
print(f"\nModel coverage per benchmark:")
model_coverage = (df.notna().sum(axis=0) / len(df) * 100).round(2)
print(model_coverage.sort_values(ascending=False))
