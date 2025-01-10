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


#define BLOCK_SIZE 32

__global__ void computeHistogramGrayscale(const unsigned char* d_image, int* d_hist, int width, int height) {
    __shared__ int tile_hist[256];

    // Initialize shared memory
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < 256) {
        tile_hist[tid] = 0;
    }
    __syncthreads();

    // Compute global pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned char pixel = d_image[y * width + x];
        atomicAdd(&tile_hist[pixel], 1);
    }
    __syncthreads();

    // Update global histogram
    if (tid < 256) {
        atomicAdd(&d_hist[tid], tile_hist[tid]);
    }
}

__global__ void computeCDFGrayscale(int* d_hist, unsigned char* d_cdf, int width, int height) {
    int idx = threadIdx.x;

    if (idx < 256) {
        int total_pixels = width * height;
        int cdf_accum = 0;

        for (int i = 0; i <= idx; ++i) {
            cdf_accum += d_hist[i];
        }

        d_cdf[idx] = (unsigned char)((cdf_accum * 255.0f) / total_pixels);
    }
}

__global__ void equalizeGrayscaleImage(unsigned char* d_image, unsigned char* d_output, int width, int height, unsigned char* d_cdf) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Check for valid pixel location
    if (x < width && y < height) {
        unsigned char pixel = d_image[y * width + x];
        
        // Use shared memory for fast CDF access
        __shared__ unsigned char shared_cdf[256];
        if (tx == 0 && ty == 0) {
            // Copy CDF values into shared memory (can be done once per block)
            for (int i = 0; i < 256; i++) {
                shared_cdf[i] = d_cdf[i];
            }
        }
        __syncthreads();

        // Perform the equalization using shared CDF
        d_output[y * width + x] = shared_cdf[pixel];
    }
}
