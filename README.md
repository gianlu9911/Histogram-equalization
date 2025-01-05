# Histogram-Equalization
This project implements histogram equalization for image processing using both sequential and CUDA parallel computing approaches. The goal is to enhance the contrast of images by transforming the values in an intensity image so that the histogram of the output image is approximately flat.

## Project Structure
- **cuda_main.cu**: CUDA implementation of histogram equalization.
- **sequential_main.cpp**: sequential implementation of histogram equalization.
- **include/Utility.h**: Header file containing the functions used for the sequential implementation.
- **include/UtilityCuda.cu**: Source file containing CUDA kernels and utility functions for the CUDA implementation.
- **execution_times.csv**: CSV file to store execution times for different image sizes.
- **images/**: Directory containing sample images for testing.
- **outputs/**: Directory to store output images after processing.
