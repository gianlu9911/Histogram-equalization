#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

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
void histogram_equalization_cuda(cv::Mat& img) {
    int rows = img.rows, cols = img.cols;
    int img_size = rows * cols;
    int* h_histogram;
    int* h_cdf;
    unsigned char* h_pinned_image;

    cudaHostAlloc((void**)&h_pinned_image, img_size * sizeof(unsigned char), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_histogram, HISTOGRAM_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_cdf, HISTOGRAM_SIZE * sizeof(int), cudaHostAllocDefault);
    memcpy(h_pinned_image, img.data, img_size * sizeof(unsigned char));

    int* d_histogram;
    int* d_cdf;
    unsigned char* d_image;

    cudaMalloc((void**)&d_image, img_size * sizeof(unsigned char));
    cudaMalloc((void**)&d_histogram, HISTOGRAM_SIZE * sizeof(int));
    cudaMalloc((void**)&d_cdf, HISTOGRAM_SIZE * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record total start time
    cudaEventRecord(start);

    // Memory transfer from host to device (image)
    cudaEventRecord(start);
    cudaMemcpy(d_image, h_pinned_image, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float mem_transfer_time;
    cudaEventElapsedTime(&mem_transfer_time, start, stop);
    std::cout << "Memory Transfer Time (Host to Device): " << mem_transfer_time << " ms." << std::endl;

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
    calculate_histogram_tiled<<<grid, block>>>(d_image, d_histogram, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float histogram_time;
    cudaEventElapsedTime(&histogram_time, start, stop);
    std::cout << "Histogram Calculation Time: " << histogram_time << " ms." << std::endl;

    // CDF Computation using Prefix Sum
    cudaEventRecord(start);
    int block_size = 256;
    int grid_size = (HISTOGRAM_SIZE + block_size - 1) / block_size;
    prefix_sum_kernel<<<grid_size, block_size>>>(d_histogram, d_cdf, HISTOGRAM_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cdf_time;
    cudaEventElapsedTime(&cdf_time, start, stop);
    std::cout << "CDF Computation Time (Prefix Sum): " << cdf_time << " ms." << std::endl;

    // Copy CDF data from device to host
    cudaMemcpy(h_cdf, d_cdf, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
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
    histogram_equalization_tiled<<<grid_tiling, block_tiling>>>(d_image, d_cdf, rows, cols, cdf_min, cdf_max);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float equalization_time;
    cudaEventElapsedTime(&equalization_time, start, stop);
    std::cout << "Histogram Equalization Time: " << equalization_time << " ms." << std::endl;
    
    // Memory transfer back from device to host (image)
    cudaEventRecord(start);
    cudaMemcpy(h_pinned_image, d_image, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float mem_transfer_back_time;
    cudaEventElapsedTime(&mem_transfer_back_time, start, stop);
    std::cout << "Memory Transfer Time (Device to Host): " << mem_transfer_back_time << " ms." << std::endl;

    // Copy the result back to the image
    memcpy(img.data, h_pinned_image, img_size * sizeof(unsigned char));

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_histogram);
    cudaFree(d_cdf);

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
    // Mat img = imread("images/img4.bmp", IMREAD_COLOR);
    Mat img = imread("images/img4.bmp", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    // Check number of channels and decide on processing
    if (img.channels() == 1) {
        std::cout << "Processing single-channel image (grayscale)..." << std::endl;
        histogram_equalization_cuda(img);
    } else if (img.channels() == 3) {
        std::cout << "Processing multi-channel image (RGB), converting to YCbCr..." << std::endl;

        // Convert RGB to YCbCr
        cv::Mat ycbcr_img;
        cv::cvtColor(img, ycbcr_img, cv::COLOR_BGR2YCrCb);

        // Split the YCbCr image into Y, Cb, and Cr channels
        std::vector<cv::Mat> channels(3);
        cv::split(ycbcr_img, channels);

        // Extract the Y channel
        cv::Mat& y_channel = channels[0];

        // Use the old histogram equalization function for the Y channel
        histogram_equalization_cuda(y_channel);

        // Merge the Y, Cb, and Cr channels back
        cv::merge(channels, ycbcr_img);

        // Convert YCbCr back to RGB
        cv::cvtColor(ycbcr_img, img, cv::COLOR_YCrCb2BGR);
    } else {
        std::cout << "Unsupported number of channels: " << img.channels() << std::endl;
        return -1;
    }

    imwrite("outputs/equalized.jpg", img);
    imshow("Equalized Image", img);
    waitKey(0);

    return 0;
}

