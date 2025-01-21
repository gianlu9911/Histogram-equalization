#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/stat.h>

#define HISTOGRAM_SIZE 256
#define TILE_SIZE 32

using namespace cv;

// Warm-up kernel to stabilize CUDA performance
__global__ void warmup_kernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Perform a dummy computation to engage GPU cores
    float temp = 0.0f;
    for (int i = 0; i < 100; i++) {
        temp += sinf(idx * 0.01f); // Arbitrary computation
    }
}


__global__ void calculate_histogram_tiled(unsigned char* d_image, int* d_histogram, int rows, int cols) {
    __shared__ int local_histogram[HISTOGRAM_SIZE];  // Shared memory histogram

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;  // Unique thread ID in block
    int global_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int global_y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Step 1: Initialize shared memory histogram using all threads
    if (thread_id < HISTOGRAM_SIZE) {
        local_histogram[thread_id] = 0;
    }
    __syncthreads();

    // Step 2: Coalesced memory access for image pixels
    if (global_x < cols && global_y < rows) {
        unsigned char pixel = d_image[global_y * cols + global_x];
        atomicAdd(&local_histogram[pixel], 1);  // Efficient atomic update within shared memory
    }
    __syncthreads();

    // Step 3: Efficient shared-to-global histogram update
    if (thread_id < HISTOGRAM_SIZE) {
        atomicAdd(&d_histogram[thread_id], local_histogram[thread_id]);  // Minimized global atomics
    }
}



// Prefix sum kernel to calculate the CDF of the histogram
__global__ void prefix_sum_kernel(int* d_histogram, int* d_cdf, int size) {
    __shared__ int temp[HISTOGRAM_SIZE];  // Shared memory to hold the histogram

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;

    if (idx < size) {
        temp[thread_id] = d_histogram[idx];
    } else {
        temp[thread_id] = 0;
    }

    __syncthreads();

    // Perform prefix sum (scan) in shared memory
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (thread_id >= offset) {
            temp[thread_id] += temp[thread_id - offset];
        }
        __syncthreads();
    }

    // Write the result back to the global memory
    if (idx < size) {
        d_cdf[idx] = temp[thread_id];
    }
}

// Optimized kernel for histogram equalization using tiling
__global__ void histogram_equalization_tiled(unsigned char* d_image, int* d_cdf, int rows, int cols, int cdf_min, int cdf_max) {
    __shared__ unsigned char tile[TILE_SIZE][TILE_SIZE];  // Shared memory tile

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = d_image[y * cols + x];
    }
    __syncthreads();

    if (x < cols && y < rows) {
        unsigned char val = tile[threadIdx.y][threadIdx.x];
        tile[threadIdx.y][threadIdx.x] = (unsigned char)(((d_cdf[val] - cdf_min) * 255) / (cdf_max - cdf_min));
    }
    __syncthreads();

    if (x < cols && y < rows) {
        d_image[y * cols + x] = tile[threadIdx.y][threadIdx.x];
    }
}

void saveExecutionTimesToCSV(const std::vector<std::tuple<int, int, int, std::string, double, int, int, int, int>>& executionTimes) {
    std::ofstream file;
    bool isNewFile = false;

    // Check if the file exists and is empty
    struct stat buffer;
    if (stat("execution_times_cuda.csv", &buffer) != 0) {
        isNewFile = true; // The file doesn't exist, so we'll create it
    } else if (buffer.st_size == 0) {
        isNewFile = true; // The file exists but is empty
    }

    file.open("execution_times_cuda.csv", std::ios::app); // Open in append mode

    // If the file is new or empty, write the header
    if (isNewFile) {
        file << "Width,Height,Channels,Method,ExecutionTime(ms),BlockWidth,BlockHeight,TileWidth,TileHeight\n";
    }

    // Write the execution times to the file
    for (const auto& execTime : executionTimes) {
        int width, height, channels, blockWidth, blockHeight, tileWidth, tileHeight;
        std::string method;
        double time;
        std::tie(width, height, channels, method, time, blockWidth, blockHeight, tileWidth, tileHeight) = execTime;
        file << width << "," << height << "," << channels << "," << method << "," << time << ","
             << blockWidth << "," << blockHeight << "," << tileWidth << "," << tileHeight << "\n";
    }

    file.close();
}