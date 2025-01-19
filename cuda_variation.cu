#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <iostream>

#define HISTOGRAM_SIZE 256
#define TILE_SIZE 32    // Increased tile size for better parallelism

using namespace cv;

// Warm-up kernel to stabilize CUDA performance
__global__ void warmup_kernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        // Dummy operation
    }
}

// Kernel to calculate the histogram of the image with shared memory reduction
__global__ void calculate_histogram(unsigned char* d_image, int* d_histogram, int rows, int cols) {
    __shared__ int local_histogram[HISTOGRAM_SIZE];  // Shared memory for local histogram

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            local_histogram[i] = 0;
        }
    }
    __syncthreads();

    if (x < cols && y < rows) {
        atomicAdd(&local_histogram[d_image[y * cols + x]], 1);
    }

    __syncthreads();

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

void histogram_equalization_cuda(Mat& img) {
    int rows = img.rows, cols = img.cols;
    int img_size = rows * cols;
    int* h_histogram;
    int* h_cdf;
    unsigned char* h_pinned_image;

    cudaHostAlloc((void**)&h_pinned_image, img_size * sizeof(unsigned char), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_histogram, HISTOGRAM_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_cdf, HISTOGRAM_SIZE * sizeof(int), cudaHostAllocDefault);
    memcpy(h_pinned_image, img.data, img_size * sizeof(unsigned char));

    thrust::device_vector<unsigned char> d_image(h_pinned_image, h_pinned_image + img_size);
    thrust::device_vector<int> d_histogram(HISTOGRAM_SIZE, 0);
    thrust::device_vector<int> d_cdf(HISTOGRAM_SIZE, 0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record total start time
    cudaEventRecord(start);

    // Warm-up kernel
    cudaEventRecord(start);
    warmup_kernel<<<1, 1>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float warmup_time;
    cudaEventElapsedTime(&warmup_time, start, stop);
    std::cout << "Warm-up Time: " << warmup_time << " ms." << std::endl;

    // Histogram Calculation
    cudaEventRecord(start);
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    calculate_histogram<<<grid, block>>>(thrust::raw_pointer_cast(d_image.data()), thrust::raw_pointer_cast(d_histogram.data()), rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float histogram_time;
    cudaEventElapsedTime(&histogram_time, start, stop);
    std::cout << "Histogram Calculation Time: " << histogram_time << " ms." << std::endl;

    // CDF Computation
    cudaEventRecord(start);
    thrust::inclusive_scan(d_histogram.begin(), d_histogram.end(), d_cdf.begin());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cdf_time;
    cudaEventElapsedTime(&cdf_time, start, stop);
    std::cout << "CDF Computation Time: " << cdf_time << " ms." << std::endl;

    // Copy CDF data from device to host
    thrust::copy(d_cdf.begin(), d_cdf.end(), h_cdf);
    int cdf_min = *std::min_element(h_cdf, h_cdf + HISTOGRAM_SIZE);
    int cdf_max = h_cdf[HISTOGRAM_SIZE - 1];

    if (cdf_min == cdf_max) {
        std::cout << "Skipping equalization, all pixels are the same!" << std::endl;
        return;
    }

    // Histogram Equalization using tiling
    dim3 grid_tiling((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block_tiling(TILE_SIZE, TILE_SIZE);

    cudaEventRecord(start);
    histogram_equalization_tiled<<<grid_tiling, block_tiling>>>(thrust::raw_pointer_cast(d_image.data()), thrust::raw_pointer_cast(d_cdf.data()), rows, cols, cdf_min, cdf_max);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float equalization_time;
    cudaEventElapsedTime(&equalization_time, start, stop);
    std::cout << "Histogram Equalization Time: " << equalization_time << " ms." << std::endl;

    // Copy the result back to the image
    thrust::copy(d_image.begin(), d_image.end(), h_pinned_image);
    memcpy(img.data, h_pinned_image, img_size * sizeof(unsigned char));

    // Free host memory
    cudaFreeHost(h_pinned_image);
    cudaFreeHost(h_histogram);
    cudaFreeHost(h_cdf);

    // Record total end time and calculate total execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    std::cout << "Total Execution Time: " << total_time << " ms." << std::endl;
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
