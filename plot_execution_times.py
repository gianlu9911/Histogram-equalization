import pandas as pd
import matplotlib.pyplot as plt

# Load data
sequential_file = 'execution_times_sequential.csv'
cuda_file = 'execution_times_cuda.csv'

df_sequential = pd.read_csv(sequential_file)
df_cuda = pd.read_csv(cuda_file)

# Define a function to map methods dynamically
def map_method(row):
    if row['Channels'] == 1 and row['Method'] != 'OpenCV Grayscale':
        return 'grayscale'
    elif row['Channels'] == 3 and row['Method'] != 'OpenCV Color':
        return 'rgb'
    return None

# Apply mapping to sequential data
df_sequential['MappedMethod'] = df_sequential.apply(map_method, axis=1)
df_cuda['MappedMethod'] = df_cuda['Method'].str.strip().str.lower()

# Debug: Check mapped methods
print("\nMapped Sequential Methods:", df_sequential['MappedMethod'].unique())
print("CUDA Methods:", df_cuda['MappedMethod'].unique())

# Standardize key columns and ensure data types match
for col in ['Width', 'Height', 'Channels']:
    df_sequential[col] = df_sequential[col].astype(int)
    df_cuda[col] = df_cuda[col].astype(int)

# Use MappedMethod for alignment
columns_to_match = ['Width', 'Height', 'Channels', 'MappedMethod']
df_sequential_matched = df_sequential.set_index(columns_to_match)
df_cuda_matched = df_cuda.set_index(columns_to_match)

# Align rows and calculate speedup
common_index = df_sequential_matched.index.intersection(df_cuda_matched.index)

if common_index.empty:
    print("No common rows found. Check your data for inconsistencies!")
else:
    print("\nCommon Rows Found:")
    print(common_index)

# Calculate speedup
speedup = (df_sequential_matched.loc[common_index, 'ExecutionTime(ms)'] /
           df_cuda_matched.loc[common_index, 'ExecutionTime(ms)'])

# Convert to DataFrame for plotting
speedup_df = speedup.reset_index(name='Speedup')

# Debug: Print calculated speedup
print("\nSpeedup Data:")
print(speedup_df)

# Plot the speedup if there's data
if speedup_df.empty:
    print("No speedup data to plot. Ensure your sequential and CUDA data files align correctly.")
else:
    # Plotting
    plt.figure(figsize=(10, 6))
    for method in speedup_df['MappedMethod'].unique():
        method_data = speedup_df[speedup_df['MappedMethod'] == method]
        plt.plot(method_data['Width'], method_data['Speedup'], marker='o', label=f'Method: {method}')

    plt.title('Speedup Comparison')
    plt.xlabel('Width (Pixels)')
    plt.ylabel('Speedup')
    plt.legend()
    plt.grid(True)
    plt.show()
