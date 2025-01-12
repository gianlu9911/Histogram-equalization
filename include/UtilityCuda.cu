#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32  



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

__global__ void computeHistogramGrayscale(const unsigned char* d_image, int* d_hist, int width, int height)
{
    __shared__ int local_hist[256];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < 256) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char gray_value = d_image[idx];

        atomicAdd(&local_hist[gray_value], 1);
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&d_hist[tid], local_hist[tid]);
    }
}

__global__ void computeCDFGrayscale(int* d_hist, unsigned char* d_cdf, int width, int height)
{
    int idx = threadIdx.x;
    if (idx >= 256) return;

    int cdf_accum = 0;
    for (int i = 0; i <= idx; ++i) {
        cdf_accum += d_hist[i];
    }

    // Normalize the CDF and store it in the d_cdf array
    d_cdf[idx] = (unsigned char)(cdf_accum * 255 / (width * height));
}

__global__ void equalizeGrayscaleImage(const unsigned char* d_image, unsigned char* d_output, int width, int height,
    const unsigned char* d_cdf, int tile_width, int tile_height)
{
    // Calculate global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Allocate shared memory for a tile of the image
    __shared__ unsigned char tile[BLOCK_SIZE][BLOCK_SIZE];

    // Thread coordinates within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute global pixel coordinates
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load the tile into shared memory
    if (global_x < width && global_y < height) {
        int idx = global_y * width + global_x;
        tile[ty][tx] = d_image[idx];  // Grayscale channel
    }

    // Synchronize to make sure all threads have loaded their tile into shared memory
    __syncthreads();

    // Perform the histogram equalization on the tile
    if (global_x < width && global_y < height) {
        int idx = global_y * width + global_x;

        // Apply the CDF to the pixel
        unsigned char eq_value = d_cdf[tile[ty][tx]];

        // Store the equalized pixel in the output image
        d_output[idx] = eq_value;
    }
}
