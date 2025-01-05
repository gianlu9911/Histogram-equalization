# Histogram-Equalization
This project implements histogram equalization for image processing using both sequential and CUDA parallel computing approaches. The goal is to enhance the contrast of images.

## Project Structure
- **cuda_main.cu**: CUDA implementation of histogram equalization.
- **sequential_main.cpp**: Sequential implementation of histogram equalization.
- **include/Utility.h**: Header file containing the functions used for the sequential implementation.
- **include/UtilityCuda.cu**: Source file containing CUDA kernels and utility functions for the CUDA implementation.
- **execution_times.csv**: CSV file to store execution times for different image sizes and configurations.
- **images/**: Directory containing sample images for testing.
- **outputs/**: Directory to store output images after processing.
- **plot_execution_times.py**: Script to plot the execution times and speedup of the CUDA implementation compared to the sequential implementation.
- **profiling/**: Directory containing profiling reports.


## About execution_times.csv

The `execution_times.csv` file contains the execution times for different image sizes and configurations. It is used to analyze and plot the performance of the sequential and CUDA implementations.

### Columns

- **Image Size**: The size of the image (e.g., 512x512, 1024x1024).
- **Blocks**: The number of CUDA blocks used in the execution.
- **Threads**: The number of CUDA threads per block.
- **Execution Time**: The time taken to process the image in milliseconds.

