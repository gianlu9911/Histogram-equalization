#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <iomanip>

// CUDA error checking 
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

// Histogram Kernel with Shared Memory
__global__ void compute_histogram(const unsigned char* d_input, int* d_hist, int width, int height) {
    __shared__ int hist_shared[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x < 256) hist_shared[threadIdx.x] = 0;
    __syncthreads();

    if (idx < width * height) {
        atomicAdd(&hist_shared[d_input[idx]], 1);
    }
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&d_hist[threadIdx.x], hist_shared[threadIdx.x]);
    }
}


__global__ void compute_cdf(int* d_hist, int* d_cdf, int total_pixels) {
    __shared__ int hist_shared[256];
    __shared__ int cdf_shared[256];
    int idx = threadIdx.x;

    if (idx < 256) hist_shared[idx] = d_hist[idx];
    __syncthreads();

    if (idx < 256) {
        cdf_shared[idx] = 0;
        for (int i = 0; i <= idx; ++i) {
            cdf_shared[idx] += hist_shared[i];
        }
        cdf_shared[idx] = (cdf_shared[idx] * 255) / total_pixels;
    }
    __syncthreads();

    if (idx < 256) d_cdf[idx] = cdf_shared[idx];
}

// Equalization Kernel with Shared Memory
__global__ void equalize_image(unsigned char* d_output, const unsigned char* d_input, const int* d_cdf, int width, int height) {
    __shared__ int cdf_shared[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x < 256) {
        cdf_shared[threadIdx.x] = d_cdf[threadIdx.x];
    }
    __syncthreads();

    if (idx < width * height) {
        d_output[idx] = cdf_shared[d_input[idx]];
    }
}
=======
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <utility>

__global__ void compute_histogram(const unsigned char* d_input, int* d_hist, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < width * height) {
        atomicAdd(&d_hist[d_input[idx]], 1);
    }
}

__global__ void compute_cdf(int* d_hist, int* d_cdf, int total_pixels) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 256) {
        d_cdf[idx] = 0;
        for (int i = 0; i <= idx; ++i) {
            d_cdf[idx] += d_hist[i];
        }
        d_cdf[idx] = (d_cdf[idx] * 255) / total_pixels;
    }
}

__global__ void equalize_image(unsigned char* d_output, const unsigned char* d_input, const int* d_cdf, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < width * height) {
        d_output[idx] = d_cdf[d_input[idx]];
    }
}
=======
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <utility>

__global__ void compute_histogram(const unsigned char* d_input, int* d_hist, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < width * height) {
        atomicAdd(&d_hist[d_input[idx]], 1);
    }
}

__global__ void compute_cdf(int* d_hist, int* d_cdf, int total_pixels) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 256) {
        d_cdf[idx] = 0;
        for (int i = 0; i <= idx; ++i) {
            d_cdf[idx] += d_hist[i];
        }
        d_cdf[idx] = (d_cdf[idx] * 255) / total_pixels;
    }
}

__global__ void equalize_image(unsigned char* d_output, const unsigned char* d_input, const int* d_cdf, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < width * height) {
        d_output[idx] = d_cdf[d_input[idx]];
    }
}
>>>>>>> fa413aa0b83c7d85b8316d3827d75df66d921c36
>>>>>>> 82380bd8a704aa1303f781b68e5e46aed0cfa4c8
