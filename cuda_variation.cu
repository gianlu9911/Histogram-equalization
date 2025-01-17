#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 256

using namespace cv;

// CUDA kernel to compute the histogram
__global__ void compute_histogram(unsigned char* d_image, int* d_histogram, int img_size, int padded_width, int padded_height) {
    __shared__ int local_hist[HISTOGRAM_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize shared memory histogram with coalesced memory access
    for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Unrolling loop for better memory access pattern
    for (int i = tid; i < img_size; i += stride * 4) {
        atomicAdd(&local_hist[d_image[i]], 1);
        if (i + stride < img_size) atomicAdd(&local_hist[d_image[i + stride]], 1);
        if (i + 2 * stride < img_size) atomicAdd(&local_hist[d_image[i + 2 * stride]], 1);
        if (i + 3 * stride < img_size) atomicAdd(&local_hist[d_image[i + 3 * stride]], 1);
    }
    __syncthreads();

    // Reduce atomic contention by distributing updates
    for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
        atomicAdd(&d_histogram[i], local_hist[i]);
    }
}

// CUDA kernel for inclusive scan (prefix sum) to compute the CDF
__global__ void inclusive_scan(int* d_histogram, int* d_cdf, int size) {
    __shared__ int temp_cdf[HISTOGRAM_SIZE];

    int tid = threadIdx.x;
    if (tid < HISTOGRAM_SIZE) {
        temp_cdf[tid] = d_histogram[tid];
    }
    __syncthreads();

    // Parallel inclusive scan (Hillis-Steele)
    for (int stride = 1; stride < HISTOGRAM_SIZE; stride *= 2) {
        int val = (tid >= stride) ? temp_cdf[tid - stride] : 0;
        __syncthreads();
        temp_cdf[tid] += val;
        __syncthreads();
    }

    if (tid < HISTOGRAM_SIZE) {
        d_cdf[tid] = temp_cdf[tid];
    }
}

// CUDA kernel to equalize the image using the CDF
__global__ void equalize_image(unsigned char* d_image, int* d_cdf, int img_size, int padded_width, int padded_height, int cdf_min, int cdf_max) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Process multiple pixels per thread, accounting for padded size
    for (int i = tid; i < img_size; i += stride * 4) {
        int row = i / padded_width;
        int col = i % padded_width;
        if (row < padded_height && col < padded_width) {
            int pixel_val = d_image[i];
            d_image[i] = (unsigned char)(((d_cdf[pixel_val] - cdf_min) * 255) / (cdf_max - cdf_min));
        }

        if (i + stride < img_size) {
            row = (i + stride) / padded_width;
            col = (i + stride) % padded_width;
            if (row < padded_height && col < padded_width) {
                int pixel_val = d_image[i + stride];
                d_image[i + stride] = (unsigned char)(((d_cdf[pixel_val] - cdf_min) * 255) / (cdf_max - cdf_min));
            }
        }

        if (i + 2 * stride < img_size) {
            row = (i + 2 * stride) / padded_width;
            col = (i + 2 * stride) % padded_width;
            if (row < padded_height && col < padded_width) {
                int pixel_val = d_image[i + 2 * stride];
                d_image[i + 2 * stride] = (unsigned char)(((d_cdf[pixel_val] - cdf_min) * 255) / (cdf_max - cdf_min));
            }
        }

        if (i + 3 * stride < img_size) {
            row = (i + 3 * stride) / padded_width;
            col = (i + 3 * stride) % padded_width;
            if (row < padded_height && col < padded_width) {
                int pixel_val = d_image[i + 3 * stride];
                d_image[i + 3 * stride] = (unsigned char)(((d_cdf[pixel_val] - cdf_min) * 255) / (cdf_max - cdf_min));
            }
        }
    }
}

void histogram_equalization_cuda(Mat& img) {
    int img_size = img.rows * img.cols;
    int padded_width = ((img.cols + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    int padded_height = ((img.rows + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    unsigned char* d_image;
    int *d_histogram, *d_cdf;

    // Allocate pinned memory on host
    unsigned char* h_image;
    int* h_histogram;
    int* h_cdf;

    cudaMallocHost((void**)&h_image, padded_width * padded_height * sizeof(unsigned char));
    cudaMallocHost((void**)&h_histogram, HISTOGRAM_SIZE * sizeof(int));
    cudaMallocHost((void**)&h_cdf, HISTOGRAM_SIZE * sizeof(int));

    // Copy image to padded memory
    memcpy(h_image, img.data, img_size * sizeof(unsigned char));

    // Pad image with zeros
    for (int i = img_size; i < padded_width * padded_height; i++) {
        h_image[i] = 0;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_image, padded_width * padded_height * sizeof(unsigned char));
    cudaMalloc((void**)&d_histogram, HISTOGRAM_SIZE * sizeof(int));
    cudaMalloc((void**)&d_cdf, HISTOGRAM_SIZE * sizeof(int));

    // Copy data to device memory
    cudaMemcpy(d_image, h_image, padded_width * padded_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, HISTOGRAM_SIZE * sizeof(int));

    // Launch optimized histogram kernel
    int num_blocks = (padded_width * padded_height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_histogram<<<num_blocks, BLOCK_SIZE>>>(d_image, d_histogram, padded_width * padded_height, padded_width, padded_height);
    cudaDeviceSynchronize();

    // Compute CDF using inclusive scan (prefix sum)
    inclusive_scan<<<1, HISTOGRAM_SIZE>>>(d_histogram, d_cdf, HISTOGRAM_SIZE);
    cudaDeviceSynchronize();

    // Copy CDF back to pinned memory (host)
    cudaMemcpy(h_cdf, d_cdf, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Find min and max CDF values for equalization
    int cdf_min = 0;
    int cdf_max = h_cdf[HISTOGRAM_SIZE - 1];

    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        if (h_cdf[i] > 0) {
            cdf_min = h_cdf[i];
            break;
        }
    }

    // Apply equalization
    equalize_image<<<num_blocks, BLOCK_SIZE>>>(d_image, d_cdf, padded_width * padded_height, padded_width, padded_height, cdf_min, cdf_max);
    cudaDeviceSynchronize();

    // Copy the result back to pinned memory (host)
    cudaMemcpy(h_image, d_image, padded_width * padded_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Copy back to OpenCV Mat and display/save
    memcpy(img.data, h_image, img_size * sizeof(unsigned char));

    // Free memory
    cudaFree(d_image);
    cudaFree(d_histogram);
    cudaFree(d_cdf);
    cudaFreeHost(h_image);
    cudaFreeHost(h_histogram);
    cudaFreeHost(h_cdf);
}

int main() {
    Mat img = imread("images/img4.bmp", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    histogram_equalization_cuda(img);

    imwrite("outputs/equalized.jpg", img);
    imshow("Equalized Image", img);
    waitKey(0);

    return 0;
}
