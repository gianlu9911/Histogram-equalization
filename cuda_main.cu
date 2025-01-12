#include "UtilityCuda.cu"


#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void warmUpKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // No operation, just to warm up the device
    if (idx == 0) {
        printf("CUDA device warmed up!\n");
    }
}


void equalizeImageWithCUDAGrayscale(const cv::Mat& inputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    const int TILE_WIDTH = 16;  // Tile width for shared memory
    const int TILE_HEIGHT = 16; // Tile height for shared memory

    // Device memory allocation
    unsigned char *d_image, *d_output;
    int *d_hist;
    unsigned char *d_cdf;

    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start overall time measurement
    CUDA_CHECK(cudaEventRecord(start));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_image, width * height * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_image, inputImage.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, 256 * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_cdf, 256 * sizeof(unsigned char)));

    // Warm-up kernel
    warmUpKernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Histogram computation
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    computeHistogramGrayscale<<<grid, block>>>(d_image, d_hist, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // CDF computation
    computeCDFGrayscale<<<1, 256>>>(d_hist, d_cdf, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Equalization
    equalizeGrayscaleImage<<<grid, block>>>(d_image, d_output, width, height, d_cdf);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    cv::Mat outputImage(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(outputImage.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_cdf));

    // Save the output
    cv::imwrite("../outputs/cuda_equalized_grayscale_image_" + std::to_string(width) + "_" + std::to_string(height) + ".jpg", outputImage);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate the total elapsed time
    float totalMilliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalMilliseconds, start, stop));
    std::cout << "Total Histogram Equalization (Grayscale) executed in " << totalMilliseconds << " ms.\n";

    // Write execution times to CSV file
    std::ofstream csvFile("../execution_times_cuda.csv", std::ios::app);
    if (csvFile.is_open()) {
        csvFile << "Total execution," 
                << totalMilliseconds << ","
                << width << ","
                << height << ","
                << 1 << ","  // Grayscale image has 1 channel
                << TILE_WIDTH << ","
                << TILE_HEIGHT << ","
                << block.x << ","
                << block.y << "\n";
        csvFile.close();
    } else {
        std::cerr << "Error: Could not open CSV file for logging." << std::endl;
    }
}


void equalizeImageWithCUDA(const cv::Mat& inputImage, int tile_width, int tile_height) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();  // Get the number of channels

    // Allocate memory for input image and output image
    unsigned char *d_image, *d_output;
    int *d_hist_r, *d_hist_g, *d_hist_b;
    unsigned char *d_cdf_r, *d_cdf_g, *d_cdf_b;

    // Allocate device memory (ensure it's the correct size)
    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    size_t histSize = 256 * sizeof(int);

    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_hist_r, histSize);
    cudaMalloc(&d_hist_g, histSize);
    cudaMalloc(&d_hist_b, histSize);
    cudaMalloc(&d_cdf_r, 256 * sizeof(unsigned char));
    cudaMalloc(&d_cdf_g, 256 * sizeof(unsigned char));
    cudaMalloc(&d_cdf_b, 256 * sizeof(unsigned char));

    // Copy image data from host to device
    cudaMemcpy(d_image, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    // Initialize histograms to zero (memset on the device)
    cudaMemset(d_hist_r, 0, histSize);
    cudaMemset(d_hist_g, 0, histSize);
    cudaMemset(d_hist_b, 0, histSize);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up kernel
    warmUpKernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Open the CSV file in append mode
    std::ofstream csvFile("../execution_times_cuda.csv", std::ios::app);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file!" << std::endl;
        return;
    }

    // Write the header if the file is empty
    if (csvFile.tellp() == 0) {
        csvFile << "Stage,Time (ms),Width,Height,Channels,TileWidth,TileHeight" << std::endl;
    }

    // Record total execution start time
    cudaEventRecord(start);

    // Timing the histogram computation kernel
    cudaEvent_t stage_start, stage_stop;
    cudaEventCreate(&stage_start);
    cudaEventCreate(&stage_stop);
    cudaEventRecord(stage_start);
    
    // Define block and grid sizes
    dim3 block(tile_width, tile_height);
    dim3 grid((width + tile_width - 1) / tile_width, (height + tile_height - 1) / tile_height);
    // Launch the compute histogram kernel
    computeHistogram<<<grid, block>>>(d_image, d_hist_r, d_hist_g, d_hist_b, width, height);
    cudaEventRecord(stage_stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, stage_start, stage_stop);
    std::cout << "Histogram computation time " << milliseconds << " ms" << std::endl;
    csvFile << "Histogram computation," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    // Timing the CDF computation kernel
    cudaEventRecord(stage_start);
    // Compute the CDF for each channel (each one requires its own kernel)
    computeCDF<<<dim3(1, 1), 256>>>(d_hist_r, d_cdf_r, width, height);
    computeCDF<<<dim3(1, 1), 256>>>(d_hist_g, d_cdf_g, width, height);
    computeCDF<<<dim3(1, 1), 256>>>(d_hist_b, d_cdf_b, width, height);
    cudaEventRecord(stage_stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, stage_start, stage_stop);
    std::cout << "CDF computation time " << milliseconds << " ms" << std::endl;
    csvFile << "CDF computation," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    // Timing the equalization kernel
    cudaEventRecord(stage_start);
    // Launch the equalization kernel
    equalizeRGBImage<<<grid, block>>>(d_image, d_output, width, height, d_cdf_r, d_cdf_g, d_cdf_b, tile_width, tile_height);
    cudaEventRecord(stage_stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, stage_start, stage_stop);
    std::cout << "Equalization time " << milliseconds << " ms" << std::endl;
    csvFile << "Equalization," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    // Record total execution end time and calculate total time
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total execution time " << milliseconds << " ms" << std::endl;

    // Save the total execution time in the CSV
    csvFile << "Total execution," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    // Copy the result back to the host
    cudaMemcpy(inputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Construct the output filename
    std::stringstream outputFileName;
    outputFileName << "../outputs/RGB_output_image_" << width << "x" << height << "_CUDA.jpg";

    // Save the processed image
    if (!cv::imwrite(outputFileName.str(), inputImage)) {
        std::cerr << "Error: Could not save the image!" << std::endl;
    } else {
        std::cout << "Image saved as " << outputFileName.str() << std::endl;
    }

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_hist_r);
    cudaFree(d_hist_g);
    cudaFree(d_hist_b);
    cudaFree(d_cdf_r);
    cudaFree(d_cdf_g);
    cudaFree(d_cdf_b);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(stage_start);
    cudaEventDestroy(stage_stop);

    // Close the CSV file
    csvFile.close();
}





int main() {
    // Load the original image as grayscale and color
    cv::Mat inputImageGray = cv::imread("../images/img2.bmp", cv::IMREAD_GRAYSCALE);
    if (inputImageGray.empty()) {
        std::cerr << "Error: Could not load grayscale image!" << std::endl;
        return -1;
    }

    cv::Mat inputImageColor = cv::imread("../images/img2.bmp", cv::IMREAD_COLOR);
    if (inputImageColor.empty()) {
        std::cerr << "Error: Could not load color image!" << std::endl;
        return -1;
    }

    // List of sizes to process (square resolutions)
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};

    // Loop through each resolution
    for (int size : sizes) {
        // Resize images to the current resolution (size x size)
        cv::Mat resizedGray, resizedColor;
        cv::resize(inputImageGray, resizedGray, cv::Size(size, size));
        cv::resize(inputImageColor, resizedColor, cv::Size(size, size));

        // Process grayscale image
        //std::cout << "Processing grayscale image at resolution: " << size << "x" << size << std::endl;
        //equalizeImageWithCUDAGrayscale(resizedGray);

        // Process color image
        std::cout << "Processing color image at resolution: " 
                  << size << "x" << size << std::endl;
        equalizeImageWithCUDA(resizedColor, 16, 16);
    }

    return 0;
}
