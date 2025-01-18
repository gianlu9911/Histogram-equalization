#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <iostream>

#define HISTOGRAM_SIZE 256
#define TILE_SIZE 32    // Increased tile size for better parallelism

using namespace cv;

// Kernel to calculate the histogram of the image with shared memory reduction
__global__ void calculate_histogram(unsigned char* d_image, int* d_histogram, int rows, int cols) {
    __shared__ int local_histogram[HISTOGRAM_SIZE];  // Shared memory for local histogram

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize the shared histogram for each block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            local_histogram[i] = 0;
        }
    }
    __syncthreads();

    // Thread performs histogram computation within the block
    if (x < cols && y < rows) {
        atomicAdd(&local_histogram[d_image[y * cols + x]], 1);
    }

    __syncthreads();

    // After all threads have processed, reduce local_histogram to global histogram
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            atomicAdd(&d_histogram[i], local_histogram[i]);
        }
    }
}

// Optimized kernel for histogram equalization using tiling
__global__ void histogram_equalization_tiled(unsigned char* d_image, int* d_cdf, int rows, int cols, int cdf_min, int cdf_max) {
    __shared__ unsigned char tile[TILE_SIZE][TILE_SIZE];  // Shared memory tile

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load tile into shared memory (boundary check)
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = d_image[y * cols + x];
    }
    __syncthreads();

    // Perform histogram equalization for the tile (use shared memory)
    if (x < cols && y < rows) {
        unsigned char val = tile[threadIdx.y][threadIdx.x];
        tile[threadIdx.y][threadIdx.x] = (unsigned char)(((d_cdf[val] - cdf_min) * 255) / (cdf_max - cdf_min));
    }
    __syncthreads();

    // Store back the result to global memory
    if (x < cols && y < rows) {
        d_image[y * cols + x] = tile[threadIdx.y][threadIdx.x];
    }
}

void histogram_equalization_cuda(Mat& img) {
    int rows = img.rows, cols = img.cols;
    int img_size = rows * cols;
    int* h_histogram;
    int* h_cdf;
    unsigned char* h_pinned_image;

    // Allocate pinned memory
    cudaHostAlloc((void**)&h_pinned_image, img_size * sizeof(unsigned char), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_histogram, HISTOGRAM_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_cdf, HISTOGRAM_SIZE * sizeof(int), cudaHostAllocDefault);

    memcpy(h_pinned_image, img.data, img_size * sizeof(unsigned char));

    // Allocate device memory
    thrust::device_vector<unsigned char> d_image(h_pinned_image, h_pinned_image + img_size);
    thrust::device_vector<int> d_histogram(HISTOGRAM_SIZE, 0);
    thrust::device_vector<int> d_cdf(HISTOGRAM_SIZE, 0);

    // Compute histogram using the shared memory kernel
    dim3 block(32, 32);  // Larger block size for better parallelism
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    calculate_histogram<<<grid, block>>>(thrust::raw_pointer_cast(d_image.data()), thrust::raw_pointer_cast(d_histogram.data()), rows, cols);
    cudaDeviceSynchronize();

    // Compute CDF using thrust
    thrust::inclusive_scan(d_histogram.begin(), d_histogram.end(), d_cdf.begin());

    // Copy CDF to host
    thrust::copy(d_cdf.begin(), d_cdf.end(), h_cdf);
    int cdf_min = *std::min_element(h_cdf, h_cdf + HISTOGRAM_SIZE);
    int cdf_max = h_cdf[HISTOGRAM_SIZE - 1];

    // If cdf_min == cdf_max, all pixels are the same, skip equalization
    if (cdf_min == cdf_max) {
        std::cout << "Skipping equalization, all pixels are the same!" << std::endl;
        return;
    }

    // Grid size for tiling (increased block size)
    dim3 grid_tiling((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block_tiling(TILE_SIZE, TILE_SIZE);

    // Launch tiled histogram equalization
    histogram_equalization_tiled<<<grid_tiling, block_tiling>>>(thrust::raw_pointer_cast(d_image.data()), thrust::raw_pointer_cast(d_cdf.data()), rows, cols, cdf_min, cdf_max);
    cudaDeviceSynchronize();

    // Copy back the result
    thrust::copy(d_image.begin(), d_image.end(), h_pinned_image);
    memcpy(img.data, h_pinned_image, img_size * sizeof(unsigned char));

    // Free pinned memory
    cudaFreeHost(h_pinned_image);
    cudaFreeHost(h_histogram);
    cudaFreeHost(h_cdf);
}

int main() {
    Mat img = imread("images/img4.bmp", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    // Start CUDA event to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    histogram_equalization_cuda(img);

    // End CUDA event to measure execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Histogram equalization (CUDA) completed in " << milliseconds << " ms." << std::endl;

    imwrite("outputs/equalized.jpg", img);

    imshow("Equalized Image", img);
    waitKey(0);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
