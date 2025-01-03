<<<<<<< HEAD
=======
#include "UtilityCuda.cu"
>>>>>>> fa413aa0b83c7d85b8316d3827d75df66d921c36
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <utility>
#include <cuda_profiler_api.h>


// Optimized Histogram Kernel with Shared Memory
__global__ void compute_histogram(const unsigned char* d_input, int* d_hist, int width, int height) {
    __shared__ int hist_shared[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize shared memory histogram
    if (threadIdx.x < 256) hist_shared[threadIdx.x] = 0;
    __syncthreads();

    // Compute histogram in shared memory
    if (idx < width * height) {
        atomicAdd(&hist_shared[d_input[idx]], 1);
    }
    __syncthreads();

    // Transfer shared memory histogram to global memory
    if (threadIdx.x < 256) {
        atomicAdd(&d_hist[threadIdx.x], hist_shared[threadIdx.x]);
    }
}

// Optimized CDF Kernel
__global__ void compute_cdf(int* d_hist, int* d_cdf, int total_pixels) {
    __shared__ int hist_shared[256];
    __shared__ int cdf_shared[256];

    int idx = threadIdx.x;

    // Load histogram into shared memory
    if (idx < 256) hist_shared[idx] = d_hist[idx];
    __syncthreads();

    // Compute CDF in shared memory
    if (idx < 256) {
        cdf_shared[idx] = 0;
        for (int i = 0; i <= idx; ++i) {
            cdf_shared[idx] += hist_shared[i];
        }
        cdf_shared[idx] = (cdf_shared[idx] * 255) / total_pixels;
    }
    __syncthreads();

    // Transfer CDF back to global memory
    if (idx < 256) d_cdf[idx] = cdf_shared[idx];
}

// Optimized Equalization Kernel with Shared Memory
__global__ void equalize_image(unsigned char* d_output, const unsigned char* d_input, const int* d_cdf, int width, int height) {
    __shared__ int cdf_shared[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load CDF into shared memory
    if (threadIdx.x < 256) {
        cdf_shared[threadIdx.x] = d_cdf[threadIdx.x];
    }
    __syncthreads();

    // Equalize image
    if (idx < width * height) {
        d_output[idx] = cdf_shared[d_input[idx]];
    }
}

int main() {
    std::string inputPath = "../images/img1.bmp";
    std::string outputPath = "../outputs/";

    cv::Mat img = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << inputPath << std::endl;
        return -1;
    }

    std::vector<std::pair<int, int>> sizes = {
        {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}
    };

    for (auto size : sizes) {
        std::cout << "Processing image with size: " << size.first << "x" << size.second << std::endl;

        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(size.first, size.second));

        int width = resized_img.cols;
        int height = resized_img.rows;

        unsigned char *d_input, *d_output;
        int *d_hist, *d_cdf;
        cudaMalloc(&d_input, width * height * sizeof(unsigned char));
        cudaMalloc(&d_output, width * height * sizeof(unsigned char));
        cudaMalloc(&d_hist, 256 * sizeof(int));
        cudaMalloc(&d_cdf, 256 * sizeof(int));

        cudaMemcpy(d_input, resized_img.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemset(d_hist, 0, 256 * sizeof(int));

        cudaEvent_t start, stop;
        float milliseconds = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int threads = 256;
        int blocks_histogram = (width * height + threads - 1) / threads;

        // Start profiling
        cudaProfilerStart();

        // Histogram computation
        cudaEventRecord(start);
        compute_histogram<<<blocks_histogram, threads>>>(d_input, d_hist, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Time for compute_histogram (" << size.first << "x" << size.second << "): " << milliseconds << " ms" << std::endl;

        // CDF computation
        cudaEventRecord(start);
        compute_cdf<<<1, 256>>>(d_hist, d_cdf, width * height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Time for compute_cdf (" << size.first << "x" << size.second << "): " << milliseconds << " ms" << std::endl;

        // Equalize image
        cudaEventRecord(start);
        equalize_image<<<blocks_histogram, threads>>>(d_output, d_input, d_cdf, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Time for equalize_image (" << size.first << "x" << size.second << "): " << milliseconds << " ms" << std::endl;

        // Stop profiling
        cudaProfilerStop();

        unsigned char* h_output = new unsigned char[width * height];
        cudaMemcpy(h_output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        std::string outputFilePath = outputPath + "img_equalized_" + std::to_string(size.first) + "x" + std::to_string(size.second) + "_cuda" + ".bmp";
        cv::Mat outputImg(height, width, CV_8UC1, h_output);
        cv::imwrite(outputFilePath, outputImg);
        std::cout << "Equalized image saved to " << outputFilePath << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_hist);
        cudaFree(d_cdf);

        delete[] h_output;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
