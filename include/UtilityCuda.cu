#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

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

// CUDA kernel for histogram equalization of RGB channels
__global__ void equalizeRGBImage(const unsigned char* d_image, unsigned char* d_output, int width, int height, const unsigned char* d_cdf_r, const unsigned char* d_cdf_g, const unsigned char* d_cdf_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    unsigned char r = d_image[idx];
    unsigned char g = d_image[idx + 1];
    unsigned char b = d_image[idx + 2];

    unsigned char r_eq = d_cdf_r[r];
    unsigned char g_eq = d_cdf_g[g];
    unsigned char b_eq = d_cdf_b[b];

    d_output[idx] = r_eq;
    d_output[idx + 1] = g_eq;
    d_output[idx + 2] = b_eq;
}

__global__ void computeHistogram(const unsigned char* d_image, int* d_hist_r, int* d_hist_g, int* d_hist_b, int width, int height)
{
    __shared__ int local_hist_r[256];
    __shared__ int local_hist_g[256];
    __shared__ int local_hist_b[256];

    // Initialize shared memory
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < 256) {
        local_hist_r[tid] = 0;
        local_hist_g[tid] = 0;
        local_hist_b[tid] = 0;
    }
    __syncthreads();

    // Calculate global indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = d_image[idx];
        unsigned char g = d_image[idx + 1];
        unsigned char b = d_image[idx + 2];

        // Accumulate into shared memory
        atomicAdd(&local_hist_r[r], 1);
        atomicAdd(&local_hist_g[g], 1);
        atomicAdd(&local_hist_b[b], 1);
    }
    __syncthreads();

    // Write shared memory results back to global memory
    if (tid < 256) {
        atomicAdd(&d_hist_r[tid], local_hist_r[tid]);
        atomicAdd(&d_hist_g[tid], local_hist_g[tid]);
        atomicAdd(&d_hist_b[tid], local_hist_b[tid]);
    }
}


// CUDA kernel to compute the CDF for each color channel
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

void normalizeHistogram(int* hist, int size, int height)
{
    int max_value = *std::max_element(hist, hist + size);
    for (int i = 0; i < size; ++i) {
        hist[i] = cv::saturate_cast<int>(height * hist[i] / max_value);
    }
}

void drawHistogram(int* hist, int size, cv::Mat& image, const cv::Scalar& color)
{
    for (int i = 1; i < size; ++i) {
        cv::line(image, cv::Point(i - 1, image.rows - hist[i - 1]),
                 cv::Point(i, image.rows - hist[i]), color, 1, 8, 0);
    }
}

__global__ void equalizeRGBImageTiled(const unsigned char* d_image, unsigned char* d_output, int width, int height, const unsigned char* d_cdf_r, const unsigned char* d_cdf_g, const unsigned char* d_cdf_b)
{
    __shared__ unsigned char tile[16][16][3]; // Shared memory tile for RGB channels

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        // Load data into shared memory
        tile[ty][tx][0] = d_image[idx];
        tile[ty][tx][1] = d_image[idx + 1];
        tile[ty][tx][2] = d_image[idx + 2];
    }
    __syncthreads();

    if (x < width && y < height) {
        // Perform histogram equalization
        unsigned char r_eq = d_cdf_r[tile[ty][tx][0]];
        unsigned char g_eq = d_cdf_g[tile[ty][tx][1]];
        unsigned char b_eq = d_cdf_b[tile[ty][tx][2]];

        int idx = (y * width + x) * 3;
        d_output[idx] = r_eq;
        d_output[idx + 1] = g_eq;
        d_output[idx + 2] = b_eq;
    }
}
