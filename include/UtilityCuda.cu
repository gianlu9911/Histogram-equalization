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
