#include "UtilityCuda.cu"

void equalizeImageWithCUDA(const cv::Mat& inputImage)
{
    int width = inputImage.cols;
    int height = inputImage.rows;

    // Allocate device memory
    unsigned char* d_image;
    unsigned char* d_output;
    int *d_hist_r, *d_hist_g, *d_hist_b;
    unsigned char *d_cdf_r, *d_cdf_g, *d_cdf_b;

    // Timing variables
    cudaEvent_t start, stop, histStart, histStop, cdfStart, cdfStop, eqStart, eqStop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&histStart));
    CUDA_CHECK(cudaEventCreate(&histStop));
    CUDA_CHECK(cudaEventCreate(&cdfStart));
    CUDA_CHECK(cudaEventCreate(&cdfStop));
    CUDA_CHECK(cudaEventCreate(&eqStart));
    CUDA_CHECK(cudaEventCreate(&eqStop));

    // Start timing for the entire process
    CUDA_CHECK(cudaEventRecord(start));

    // Flatten the input image and transfer to device memory
    CUDA_CHECK(cudaMalloc(&d_image, width * height * 3 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_image, inputImage.data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char)));

    // Allocate device memory for histograms
    CUDA_CHECK(cudaMalloc(&d_hist_r, 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist_g, 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist_b, 256 * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_hist_r, 0, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist_g, 0, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist_b, 0, 256 * sizeof(int)));

    // Start timing for histogram computation
    CUDA_CHECK(cudaEventRecord(histStart));

    // Compute histogram on the device
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    computeHistogram<<<grid, block>>>(d_image, d_hist_r, d_hist_g, d_hist_b, width, height);

    // Stop timing for histogram computation
    CUDA_CHECK(cudaEventRecord(histStop));
    CUDA_CHECK(cudaEventSynchronize(histStop));

    float histTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&histTime, histStart, histStop));
    std::cout << "Histogram computation time: " << histTime << " ms" << std::endl;

    // Copy the histograms from device to host
    int h_hist_r[256] = {0};
    int h_hist_g[256] = {0};
    int h_hist_b[256] = {0};

    CUDA_CHECK(cudaMemcpy(h_hist_r, d_hist_r, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hist_g, d_hist_g, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hist_b, d_hist_b, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    // Normalize the histograms
    int max_height = 256; // Height for displaying the histogram images
    normalizeHistogram(h_hist_r, 256, max_height);
    normalizeHistogram(h_hist_g, 256, max_height);
    normalizeHistogram(h_hist_b, 256, max_height);

    // Allocate memory for CDFs
    CUDA_CHECK(cudaMalloc(&d_cdf_r, 256 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_cdf_g, 256 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_cdf_b, 256 * sizeof(unsigned char)));

    // Start timing for CDF computation
    CUDA_CHECK(cudaEventRecord(cdfStart));

    // Compute CDFs on the device
    computeCDF<<<1, 256>>>(d_hist_r, d_cdf_r, width, height);
    computeCDF<<<1, 256>>>(d_hist_g, d_cdf_g, width, height);
    computeCDF<<<1, 256>>>(d_hist_b, d_cdf_b, width, height);

    // Stop timing for CDF computation
    CUDA_CHECK(cudaEventRecord(cdfStop));
    CUDA_CHECK(cudaEventSynchronize(cdfStop));

    float cdfTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cdfTime, cdfStart, cdfStop));
    std::cout << "CDF computation time: " << cdfTime << " ms" << std::endl;

    // Start timing for histogram equalization
    CUDA_CHECK(cudaEventRecord(eqStart));

    // Launch the kernel to apply histogram equalization
    equalizeRGBImageTiled<<<grid, block>>>(d_image, d_output, width, height, d_cdf_r, d_cdf_g, d_cdf_b);

    // Stop timing for histogram equalization
    CUDA_CHECK(cudaEventRecord(eqStop));
    CUDA_CHECK(cudaEventSynchronize(eqStop));

    float eqTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&eqTime, eqStart, eqStop));
    std::cout << "Histogram equalization time: " << eqTime << " ms" << std::endl;

    // Stop timing for the entire process
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    std::cout << "Total execution time: " << totalTime << " ms" << std::endl;

    // Copy result back to host
    cv::Mat outputImage(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(outputImage.data, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_hist_r));
    CUDA_CHECK(cudaFree(d_hist_g));
    CUDA_CHECK(cudaFree(d_hist_b));
    CUDA_CHECK(cudaFree(d_cdf_r));
    CUDA_CHECK(cudaFree(d_cdf_g));
    CUDA_CHECK(cudaFree(d_cdf_b));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(histStart));
    CUDA_CHECK(cudaEventDestroy(histStop));
    CUDA_CHECK(cudaEventDestroy(cdfStart));
    CUDA_CHECK(cudaEventDestroy(cdfStop));
    CUDA_CHECK(cudaEventDestroy(eqStart));
    CUDA_CHECK(cudaEventDestroy(eqStop));

    // Save the processed image
    cv::imwrite("../outputs/cuda_equalized_RGB_image.jpg", outputImage);
}


void equalizeImageWithCUDAGrayscale(const cv::Mat& inputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;

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

    // ----------------------------------------
    // Compute Histogram
    // ----------------------------------------

    cudaEvent_t histStart, histStop;
    CUDA_CHECK(cudaEventCreate(&histStart));
    CUDA_CHECK(cudaEventCreate(&histStop));
    CUDA_CHECK(cudaEventRecord(histStart));

    // Compute histogram on the device
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    computeHistogramGrayscale<<<grid, block>>>(d_image, d_hist, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(histStop));
    CUDA_CHECK(cudaEventSynchronize(histStop));

    float histMilliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&histMilliseconds, histStart, histStop));
    std::cout << "Histogram computation time: " << histMilliseconds << " ms.\n";

    // ----------------------------------------
    // Compute CDF
    // ----------------------------------------

    cudaEvent_t cdfStart, cdfStop;
    CUDA_CHECK(cudaEventCreate(&cdfStart));
    CUDA_CHECK(cudaEventCreate(&cdfStop));
    CUDA_CHECK(cudaEventRecord(cdfStart));

    // Compute CDF on the device
    computeCDFGrayscale<<<1, 256>>>(d_hist, d_cdf, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(cdfStop));
    CUDA_CHECK(cudaEventSynchronize(cdfStop));

    float cdfMilliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&cdfMilliseconds, cdfStart, cdfStop));
    std::cout << "CDF computation time: " << cdfMilliseconds << " ms.\n";

    // ----------------------------------------
    // Histogram Equalization
    // ----------------------------------------

    cudaEvent_t equalizationStart, equalizationStop;
    CUDA_CHECK(cudaEventCreate(&equalizationStart));
    CUDA_CHECK(cudaEventCreate(&equalizationStop));
    CUDA_CHECK(cudaEventRecord(equalizationStart));

    // Perform histogram equalization
    equalizeGrayscaleImage<<<grid, block>>>(d_image, d_output, width, height, d_cdf);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(equalizationStop));
    CUDA_CHECK(cudaEventSynchronize(equalizationStop));

    float equalizationMilliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&equalizationMilliseconds, equalizationStart, equalizationStop));
    std::cout << "Histogram equalization time: " << equalizationMilliseconds << " ms.\n";

    // Stop overall time measurement
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate the total elapsed time
    float totalMilliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalMilliseconds, start, stop));

    std::cout << "Total Histogram Equalization (Grayscale) executed in " << totalMilliseconds << " ms.\n";

    // Copy result back to host
    cv::Mat outputImage(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(outputImage.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_cdf));

    // Save the output
    cv::imwrite("../outputs/cuda_equalized_grayscale_image.jpg", outputImage);
}


int main() {
    // Load the image as grayscale
    cv::Mat inputImageGray = cv::imread("../images/img2.bmp", cv::IMREAD_GRAYSCALE); // Load in grayscale
    if (inputImageGray.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Process grayscale image
    std::cout << "Processing grayscale image..." << std::endl;
    equalizeImageWithCUDAGrayscale(inputImageGray);

    // Load the image as color (RGB)
    cv::Mat inputImageColor = cv::imread("../images/img2.bmp", cv::IMREAD_COLOR); // Load in color (3 channels)
    if (inputImageColor.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Process color image
    std::cout << "Processing color image..." << std::endl;
    equalizeImageWithCUDA(inputImageColor);

    return 0;
}
