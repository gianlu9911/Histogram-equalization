import pandas as pd
import matplotlib.pyplot as plt

sequential_file = "execution_times_sequential.csv"
cuda_file = "execution_times_cuda.csv"

sequential_df = pd.read_csv(sequential_file)
cuda_df = pd.read_csv(cuda_file)

sizes = [128, 256, 512, 1024, 2048]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes = axes.flatten()

for i, size in enumerate(sizes):
    ax = axes[i]
    
    seq_data = sequential_df[sequential_df['Image Size'] == size]
    cuda_data = cuda_df[cuda_df['Image Size'] == size]

    if seq_data.empty or cuda_data.empty:
        continue

    sequential_total_time = seq_data['Total Time (ms)'].values[0]

    for block in cuda_data['Blocks'].unique():
        block_data = cuda_data[cuda_data['Blocks'] == block]
        
        x_threads = block_data['Threads'].values
        y_speedup = sequential_total_time / block_data['Total Time (ms)'].values

        ax.plot(x_threads, y_speedup, label=f'Blocks = {block}', marker='o')

    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(f'Speedup vs Number of Threads for Image Size {size}', fontsize=14)
    ax.grid(True)
    ax.legend(title="Block Configuration")

axes[-1].axis('off') #remove last empty plot

plt.tight_layout()
plt.show()
