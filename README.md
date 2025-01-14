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


