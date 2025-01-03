#include "Utility Cuda.cu"
#include <opencv2/opencv.hpp>

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

        unsigned char* h_output = new unsigned char[width * height];
        cudaMemcpy(h_output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Save the result image using OpenCV
        std::string outputFilePath = outputPath + "img_equalized_" + std::to_string(size.first) + "x" + std::to_string(size.second) + ".bmp";
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
