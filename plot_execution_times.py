import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
sequential_df = pd.read_csv('execution_times_sequential.csv')
cuda_df = pd.read_csv('execution_times_cuda.csv')

# Filter the rows corresponding to Manual Grayscale and Manual YCbCr Color methods
sequential_grayscale = sequential_df[sequential_df['Method'] == 'Manual Grayscale']
sequential_ycbcr = sequential_df[sequential_df['Method'] == 'Manual YCbCr Color']

# Prepare lists to store speedup values
speedup_grayscale = []
speedup_ycbcr = []

# Iterate over sequential grayscale entries
for index, row in sequential_grayscale.iterrows():
    width = row['Width']
    height = row['Height']
    channels = row['Channels']
    
    # Find the matching parallel CUDA entry for Grayscale
    matching_cuda = cuda_df[(cuda_df['Width'] == width) & (cuda_df['Height'] == height) &
                            (cuda_df['Channels'] == channels) & (cuda_df['Method'] == 'Grayscale')]
    
    # Compute the speedup if a match is found
    if not matching_cuda.empty:
        sequential_time = row['ExecutionTime(ms)']
        cuda_time = matching_cuda['ExecutionTime(ms)'].values[0]
        speedup = sequential_time / cuda_time
        tile_width = matching_cuda['TileWidth'].values[0]
        tile_height = matching_cuda['TileHeight'].values[0]
        block_width = matching_cuda['BlockWidth'].values[0]
        block_height = matching_cuda['BlockHeight'].values[0]
        speedup_grayscale.append((width, speedup, tile_width, tile_height, block_width, block_height))

# Iterate over sequential YCbCr entries
for index, row in sequential_ycbcr.iterrows():
    width = row['Width']
    height = row['Height']
    channels = row['Channels']
    
    # Find the matching parallel CUDA entry for YCbCr
    matching_cuda = cuda_df[(cuda_df['Width'] == width) & (cuda_df['Height'] == height) &
                            (cuda_df['Channels'] == channels) & (cuda_df['Method'] == 'YCbCr')]
    
    # Compute the speedup if a match is found
    if not matching_cuda.empty:
        sequential_time = row['ExecutionTime(ms)']
        cuda_time = matching_cuda['ExecutionTime(ms)'].values[0]
        speedup = sequential_time / cuda_time
        tile_width = matching_cuda['TileWidth'].values[0]
        tile_height = matching_cuda['TileHeight'].values[0]
        block_width = matching_cuda['BlockWidth'].values[0]
        block_height = matching_cuda['BlockHeight'].values[0]
        speedup_ycbcr.append((width, speedup, tile_width, tile_height, block_width, block_height))

# Convert the results into DataFrames
df_grayscale_speedup = pd.DataFrame(speedup_grayscale, columns=['Width', 'Speedup', 'TileWidth', 'TileHeight', 'BlockWidth', 'BlockHeight'])
df_ycbcr_speedup = pd.DataFrame(speedup_ycbcr, columns=['Width', 'Speedup', 'TileWidth', 'TileHeight', 'BlockWidth', 'BlockHeight'])

# Plot the speedup results
plt.figure(figsize=(10, 6))

# Plot Grayscale speedup
plt.plot(df_grayscale_speedup['Width'], df_grayscale_speedup['Speedup'], label='Grayscale', marker='o')

# Plot YCbCr speedup
plt.plot(df_ycbcr_speedup['Width'], df_ycbcr_speedup['Speedup'], label='YCbCr', marker='x')

# Adding labels and title with block and tile size
tile_width = df_grayscale_speedup['TileWidth'].values[0]  # Assume all entries have the same tile width
tile_height = df_grayscale_speedup['TileHeight'].values[0]  # Assume all entries have the same tile height
block_width = df_grayscale_speedup['BlockWidth'].values[0]  # Assume all entries have the same block width
block_height = df_grayscale_speedup['BlockHeight'].values[0]  # Assume all entries have the same block height

plt.xlabel('Width')
plt.ylabel('Speedup (Sequential / CUDA)')
plt.title(f'Speedup Comparison: Manual vs CUDA for Grayscale and YCbCr\nTile: {tile_width}x{tile_height}, Block: {block_width}x{block_height}')
plt.legend()

plt.grid(True)
plt.show()
