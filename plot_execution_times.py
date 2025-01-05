import pandas as pd
import matplotlib.pyplot as plt
import math

# Load the CSV data from the execution_times.csv file
df = pd.read_csv('execution_times.csv')

# Function to calculate speedup for equalization time
def calculate_speedup(row, sequential_time):
    return sequential_time / row['Equalization Time (ms)']

# Sequential execution time (threads=1, blocks=1) for each image size
sequential_data = df[(df['Threads'] == 1) & (df['Blocks'] == 1)]

# Create a dictionary to store sequential times for different image sizes
sequential_times = {
    size: sequential_data[sequential_data['Image Size'] == size]['Equalization Time (ms)'].values[0]
    for size in df['Image Size'].unique()
}

# Number of unique image sizes
num_sizes = len(sequential_times)

# Calculate the number of rows and columns for subplots (square-like arrangement)
ncols = 2  # Number of columns in the subplot grid
nrows = math.ceil(num_sizes / ncols)  # Calculate rows needed

# Create subplots: dynamically adjust the number of rows based on the number of image sizes
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 5))  # Adjust the figsize accordingly
axes = axes.flatten()  # Flatten the axes array to easily iterate over it

# Define colors and markers for different blocks and image sizes
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 's', '^', 'D', 'v', 'p', '*']

# Plot speedup for each image size and blocks configuration
for idx, size in enumerate(sequential_times.keys()):
    df_size = df[df['Image Size'] == size].copy()  # Use .copy() to avoid SettingWithCopyWarning
    sequential_time = sequential_times[size]
    
    # Calculate speedup for each row
    df_size.loc[:, 'Speedup'] = df_size.apply(lambda row: calculate_speedup(row, sequential_time), axis=1)
    
    # Loop over different blocks to use different markers
    for block_idx, block in enumerate(df_size['Blocks'].unique()):
        # Only consider blocks greater than 1
        if block > 1:
            df_block = df_size[df_size['Blocks'] == block]
            
            # Plot the speedup with different markers for blocks
            axes[idx].plot(df_block['Threads'],  # Number of threads on the x-axis
                           df_block['Speedup'],
                           color=colors[block_idx % len(colors)],  # Use a different color for each size
                           marker=markers[block_idx % len(markers)],  # Use a different marker for each block value
                           label=f'Blocks: {block}')
    
    axes[idx].set_title(f'Speedup for {size}x{size} Image')
    axes[idx].set_xlabel('Number of Threads')
    axes[idx].set_ylabel('Speedup')
    axes[idx].legend()
    axes[idx].grid(True)

# Remove any unused subplots if there are fewer than the maximum number of subplots
for i in range(num_sizes, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
