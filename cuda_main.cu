#include <iostream>
#include <opencv2/opencv.hpp>
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

int main() {
    std::string inputPath = "../images/img1.bmp";
    std::string outputPath = "../outputs/";

    // Read input image
    cv::Mat img = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << inputPath << std::endl;
        return -1;
    }

    // Define sizes to test
    std::vector<std::pair<int, int>> sizes = {
        {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}
    };

    for (auto size : sizes) {
        std::cout << "Processing image with size: " << size.first << "x" << size.second << std::endl;

        // Resize the image
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

        // Copy resized image to device 
        cudaMemcpy(d_input, resized_img.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Init histogram
        cudaMemset(d_hist, 0, 256 * sizeof(int));

        // CUDA Event for timing
        cudaEvent_t start, stop;
        float milliseconds = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record start time before the histogram computation
        cudaEventRecord(start);

        // Launch compute_histogram kernel
        int threads = 256;
        int blocks = (width * height + threads - 1) / threads;
        compute_histogram<<<blocks, threads>>>(d_input, d_hist, width, height);

        // Record stop time after compute_histogram execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  // Wait for the stop event to be recorded
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Time for compute_histogram (" << size.first << "x" << size.second << "): " << milliseconds << " ms" << std::endl;

        // Record start time before compute_cdf kernel
        cudaEventRecord(start);

        // Launch compute_cdf kernel
        compute_cdf<<<1, 256>>>(d_hist, d_cdf, width * height);

        // Record stop time after compute_cdf execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  // Wait for the stop event to be recorded
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Time for compute_cdf (" << size.first << "x" << size.second << "): " << milliseconds << " ms" << std::endl;

        // Record start time before equalize_image kernel
        cudaEventRecord(start);

        // Launch equalize_image kernel
        equalize_image<<<blocks, threads>>>(d_output, d_input, d_cdf, width, height);

        // Record stop time after equalize_image execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  // Wait for the stop event to be recorded
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Time for equalize_image (" << size.first << "x" << size.second << "): " << milliseconds << " ms" << std::endl;

        // Copy result back to host
        unsigned char* h_output = new unsigned char[width * height];
        cudaMemcpy(h_output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Save the result image using OpenCV
        std::string outputFilePath = outputPath + "img_equalized_" + std::to_string(size.first) + "x" + std::to_string(size.second) + ".bmp";
        cv::Mat outputImg(height, width, CV_8UC1, h_output);
        cv::imwrite(outputFilePath, outputImg);
        std::cout << "Equalized image saved to " << outputFilePath << std::endl;

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_hist);
        cudaFree(d_cdf);

        // Free host memory
        delete[] h_output;

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
