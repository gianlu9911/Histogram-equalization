#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32  

#define CUDA_CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, #call); \
        fprintf(stderr, "Error code: %d, Reason: %s\n", error, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void equalizeRGBImage(const unsigned char* d_image, unsigned char* d_output, int width, int height,
    const unsigned char* d_cdf_r, const unsigned char* d_cdf_g, const unsigned char* d_cdf_b,
    int tile_width, int tile_height)
{
// Calculate global thread coordinates
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

// Allocate shared memory for a tile of the image
__shared__ unsigned char tile_r[BLOCK_SIZE][BLOCK_SIZE];
__shared__ unsigned char tile_g[BLOCK_SIZE][BLOCK_SIZE];
__shared__ unsigned char tile_b[BLOCK_SIZE][BLOCK_SIZE];

// Thread coordinates within the block
int tx = threadIdx.x;
int ty = threadIdx.y;

// Compute global pixel coordinates
int global_x = blockIdx.x * blockDim.x + threadIdx.x;
int global_y = blockIdx.y * blockDim.y + threadIdx.y;

// Load the tile into shared memory
if (global_x < width && global_y < height) {
int idx = (global_y * width + global_x) * 3;
tile_r[ty][tx] = d_image[idx];        // Red channel
tile_g[ty][tx] = d_image[idx + 1];    // Green channel
tile_b[ty][tx] = d_image[idx + 2];    // Blue channel
}

// Synchronize to make sure all threads have loaded their tile into shared memory
__syncthreads();

// Perform the histogram equalization on the tile
if (global_x < width && global_y < height) {
int idx = (global_y * width + global_x) * 3;

// Apply the CDF to the corresponding channels
unsigned char r_eq = d_cdf_r[tile_r[ty][tx]];
unsigned char g_eq = d_cdf_g[tile_g[ty][tx]];
unsigned char b_eq = d_cdf_b[tile_b[ty][tx]];

d_output[idx] = r_eq;
d_output[idx + 1] = g_eq;
d_output[idx + 2] = b_eq;
}
}

__global__ void computeHistogram(const unsigned char* d_image, int* d_hist_r, int* d_hist_g, int* d_hist_b, int width, int height)
{
    __shared__ int local_hist_r[256];
    __shared__ int local_hist_g[256];
    __shared__ int local_hist_b[256];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < 256) {
        local_hist_r[tid] = 0;
        local_hist_g[tid] = 0;
        local_hist_b[tid] = 0;
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = d_image[idx];
        unsigned char g = d_image[idx + 1];
        unsigned char b = d_image[idx + 2];

        atomicAdd(&local_hist_r[r], 1);
        atomicAdd(&local_hist_g[g], 1);
        atomicAdd(&local_hist_b[b], 1);
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&d_hist_r[tid], local_hist_r[tid]);
        atomicAdd(&d_hist_g[tid], local_hist_g[tid]);
        atomicAdd(&d_hist_b[tid], local_hist_b[tid]);
    }
}


__global__ void computeCDF(int* d_hist, unsigned char* d_cdf, int width, int height)
{
    int idx = threadIdx.x;
    if (idx >= 256) return;

    int cdf_accum = 0;
    for (int i = 0; i <= idx; ++i) {
        cdf_accum += d_hist[i];
    }

    d_cdf[idx] = (unsigned char)(cdf_accum * 255 / (width * height));
}

///////////// GRAYSCALE
__global__ void computeHistogramGrayscale(const unsigned char* d_image, int* d_hist, int width, int height) {
    // Define the tile size from the BLOCK_SIZE macro
    __shared__ int tile_hist[256];  // Shared memory for storing the histogram of the tile

    // Initialize shared memory to zero
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < 256) {
        tile_hist[tid] = 0;
    }
    __syncthreads();

    // Calculate the global pixel index
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < height) {
        unsigned char pixel = d_image[y * width + x];  // Access the pixel value
        atomicAdd(&tile_hist[pixel], 1);  // Increment the histogram value in shared memory
    }
    __syncthreads();

    // Write the results from shared memory to global memory
    if (tid < 256) {
        atomicAdd(&d_hist[tid], tile_hist[tid]);
    }
}
__global__ void computeCDFGrayscale(int* d_hist, unsigned char* d_cdf, int width, int height) {
    int idx = threadIdx.x;

    if (idx < 256) {
        int cdf_accum = 0;
        // Accumulate the histogram values to compute CDF
        for (int i = 0; i <= idx; ++i) {
            cdf_accum += d_hist[i];
        }

        // Normalize and store the CDF value
        d_cdf[idx] = (unsigned char)(cdf_accum * 255 / (width * height));
    }
}
__global__ void equalizeGrayscaleImage(const unsigned char* d_image, unsigned char* d_output, int width, int height, const unsigned char* d_cdf) {
    // Shared memory tile for grayscale pixels
    __shared__ unsigned char tile[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load the image data into shared memory (tile-based loading)
    if (x < width && y < height) {
        tile[ty][tx] = d_image[y * width + x];
    }
    __syncthreads();

    // Perform histogram equalization in the tile
    if (x < width && y < height) {
        unsigned char pixel_eq = d_cdf[tile[ty][tx]];  // Map pixel to equalized value
        d_output[y * width + x] = pixel_eq;
    }
}

void writeTotalExecutionTimeToCSV(int width, int height, int channels, 
    float totalExecutionTime, int blocks, int threads, 
    const std::string& imageType) {
// Open the CSV file in append mode
std::ofstream file("../execution_times_cuda.csv", std::ios::app);

if (file.is_open()) {
// Check if the file is empty and if so, write the header
file.seekp(0, std::ios::end);  // Move to the end of the file
if (file.tellp() == 0) {
// File is empty, write the header
file << "Width,Height,Channels,Method,ExecutionTime(ms),Blocks,Threads\n";
}

// Determine the method based on image type (RGB or Grayscale)
std::string method = (imageType == "RGB") ? "RGB" : "Grayscale";

// Write the total execution data to the file
file << width << ","
<< height << ","
<< channels << ","
<< method << ","
<< totalExecutionTime << ","
<< blocks << ","
<< threads << "\n";

file.close();
} else {
std::cerr << "Error: Could not open CSV file for writing!" << std::endl;
}
}