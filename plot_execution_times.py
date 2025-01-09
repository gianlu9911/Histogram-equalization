import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
seq_df = pd.read_csv('execution_times_sequential.csv')
cuda_df = pd.read_csv('execution_times_cuda.csv')

# Merge the dataframes on matching columns (Width, Height, Channels, and Method)
merged_df = pd.merge(seq_df, cuda_df, on=['Width', 'Height', 'Channels', 'Method'], suffixes=('_seq', '_cuda'))

# Calculate speedup
merged_df['Speedup'] = merged_df['ExecutionTime(ms)_seq'] / merged_df['ExecutionTime(ms)_cuda']

# Separate grayscale and RGB versions
grayscale_df = merged_df[merged_df['Channels'] == 1]
rgb_df = merged_df[merged_df['Channels'] == 3]

# Plot speedup for grayscale and RGB separately
plt.figure(figsize=(10, 6))

# Grayscale plot
plt.scatter(grayscale_df['Width'], grayscale_df['Height'], c=grayscale_df['Speedup'], cmap='Blues', label='Grayscale', marker='o')
# RGB plot
plt.scatter(rgb_df['Width'], rgb_df['Height'], c=rgb_df['Speedup'], cmap='Reds', label='RGB', marker='^')

# Set plot labels and title
plt.title('Speedup of CUDA over Sequential Execution', fontsize=14)
plt.xlabel('Image Width', fontsize=12)
plt.ylabel('Image Height', fontsize=12)
plt.colorbar(label='Speedup')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
