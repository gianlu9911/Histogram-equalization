import pandas as pd
import matplotlib.pyplot as plt

# Load the sequential execution times data
sequential_df = pd.read_csv('execution_times_sequential.csv')

# Load the parallel execution times (CUDA) data
cuda_df = pd.read_csv('execution_times_cuda.csv')

# Filter only 'Total execution' stage in the CUDA data
cuda_df_filtered = cuda_df[cuda_df['Stage'] == 'Total execution']

# Create an empty list to store the results
results = []

# Loop through the sequential data
for _, seq_row in sequential_df.iterrows():
    width = seq_row['Width']
    channels = seq_row['Channels']
    method = seq_row['Method']
    seq_time = seq_row['ExecutionTime(ms)']
    
    # Loop through the CUDA data to find the corresponding entry
    for _, cuda_row in cuda_df_filtered.iterrows():
        if cuda_row['Width'] == width and cuda_row['Channels'] == channels:
            cuda_time = cuda_row['Time (ms)']
            
            # Calculate speedup
            speedup = seq_time / cuda_time
            
            # Append the result to the list
            results.append([speedup, width, channels, method])
            break  # Found the matching entry, no need to loop further for this combination

# Create a DataFrame from the results
results_df = pd.DataFrame(results, columns=['Speedup', 'Width', 'Channels', 'Method'])

# Create a plot
plt.figure(figsize=(12, 7))

# List of all methods to plot
methods = ['OpenCV Grayscale', 'OpenCV Color', 'Manual Grayscale', 'Manual Color']

# Define colors for each method for better distinction
colors = ['b', 'g', 'r', 'c']

# Plot speedup curves for each method
for method, color in zip(methods, colors):
    method_df = results_df[results_df['Method'] == method]
    method_df = method_df.sort_values('Width')  # Sort by Width to ensure the curve is plotted correctly
    plt.plot(method_df['Width'], method_df['Speedup'], label=method, marker='o', markersize=8, color=color, linewidth=2)

# Customize the plot
plt.xlabel('Resolution (Width)', fontsize=14)
plt.ylabel('Speedup', fontsize=14)
plt.title('Speedup vs Resolution for Different Methods', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()  # Enable minor gridlines
plt.tight_layout()

# Show the plot
plt.show()
