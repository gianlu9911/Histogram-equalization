import pandas as pd
import matplotlib.pyplot as plt

# Load the sequential and CUDA execution times
sequential_file = "execution_times_sequential.csv"
cuda_file = "execution_times_cuda.csv"

# Read the CSV files
sequential_df = pd.read_csv(sequential_file)
cuda_df = pd.read_csv(cuda_file)

# List of image sizes to consider for plotting
sizes = [128, 256, 512, 1024, 2048]

# Create a 2x3 grid for subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Flatten axes array for easier iteration
axes = axes.flatten()

# Loop through each image size and plot in the corresponding subplot
for i, size in enumerate(sizes):
    ax = axes[i]
    
    # Filter the data for the current image size
    seq_data = sequential_df[sequential_df['Image Size'] == size]
    cuda_data = cuda_df[cuda_df['Image Size'] == size]

    # If there is no data for a specific image size, skip it
    if seq_data.empty or cuda_data.empty:
        continue

    # Sequential total time for the current image size (we take the first row as there's only one sequential time per size)
    sequential_total_time = seq_data['Total Time (ms)'].values[0]

    # Loop through each block configuration
    for block in cuda_data['Blocks'].unique():
        block_data = cuda_data[cuda_data['Blocks'] == block]
        
        # We need to extract threads and calculate speedup for each configuration of threads
        x_threads = block_data['Threads'].values
        y_speedup = sequential_total_time / block_data['Total Time (ms)'].values

        # Plot the speedup for the current block configuration
        ax.plot(x_threads, y_speedup, label=f'Blocks = {block}', marker='o')

    # Set labels and title for each subplot
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(f'Speedup vs Number of Threads for Image Size {size}', fontsize=14)
    ax.grid(True)
    ax.legend(title="Block Configuration")

# Hide the last unused subplot
axes[-1].axis('off')

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
