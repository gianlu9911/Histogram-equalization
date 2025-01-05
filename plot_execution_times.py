import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv('execution_times.csv')

def calculate_speedup(row, sequential_time):
    return sequential_time / row['Equalization Time (ms)']

# Sequential execution time (threads=1, blocks=1) for each image size
sequential_data = df[(df['Threads'] == 1) & (df['Blocks'] == 1)]

# Create a dictionary to store sequential times for different image sizes
sequential_times = {
    size: sequential_data[sequential_data['Image Size'] == size]['Equalization Time (ms)'].values[0]
    for size in df['Image Size'].unique()
}

num_sizes = len(sequential_times)

ncols = 2  
nrows = math.ceil(num_sizes / ncols) 

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 5))  # Adjust the figsize accordingly
axes = axes.flatten()  # Flatten the axes array to easily iterate over it

# Different colors and markers for different blocks and image sizes
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 's', '^', 'D', 'v', 'p', '*']

for idx, size in enumerate(sequential_times.keys()):
    df_size = df[df['Image Size'] == size].copy() 
    sequential_time = sequential_times[size]
    
    df_size.loc[:, 'Speedup'] = df_size.apply(lambda row: calculate_speedup(row, sequential_time), axis=1)
    
    for block_idx, block in enumerate(df_size['Blocks'].unique()):
        # Only consider blocks greater than 1 - don't want random points to appear
        if block > 1:
            df_block = df_size[df_size['Blocks'] == block]
            
            axes[idx].plot(df_block['Threads'],  # Number of threads on the x-axis
                           df_block['Speedup'],
                           color=colors[block_idx % len(colors)], 
                           marker=markers[block_idx % len(markers)],  
                           label=f'Blocks: {block}')
    
    axes[idx].set_title(f'Speedup for {size}x{size} Image')
    axes[idx].set_xlabel('Number of Threads')
    axes[idx].set_ylabel('Speedup')
    axes[idx].legend()
    axes[idx].grid(True)

# Remove any unused subplots if there are fewer than the maximum number of subplots
for i in range(num_sizes, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
