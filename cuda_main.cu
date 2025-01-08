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

    // Create black images for plotting the histograms
    cv::Mat hist_img_r = cv::Mat::zeros(cv::Size(256, max_height), CV_8UC3);
    cv::Mat hist_img_g = cv::Mat::zeros(cv::Size(256, max_height), CV_8UC3);
    cv::Mat hist_img_b = cv::Mat::zeros(cv::Size(256, max_height), CV_8UC3);

    // Draw the histograms on the black images
    //drawHistogram(h_hist_r, 256, hist_img_r, cv::Scalar(0, 0, 255)); // Red histogram in red color
    //drawHistogram(h_hist_g, 256, hist_img_g, cv::Scalar(0, 255, 0)); // Green histogram in green color
    //drawHistogram(h_hist_b, 256, hist_img_b, cv::Scalar(255, 0, 0)); // Blue histogram in blue color

    // Combine the individual histograms into one image
    cv::Mat combined_hist;
    cv::hconcat(hist_img_r, hist_img_g, combined_hist);
    cv::hconcat(combined_hist, hist_img_b, combined_hist);

    // Display the combined histogram image
    //cv::imshow("RGB Histograms", combined_hist);
    //cv::waitKey(0);

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

    // Launch the kernel to apply histogram equalization
    CUDA_CHECK(cudaEventRecord(eqStart));
    equalizeRGBImageTiled<<<grid, block>>>(d_image, d_output, width, height, d_cdf_r, d_cdf_g, d_cdf_b);
    CUDA_CHECK(cudaEventRecord(eqStop));
    CUDA_CHECK(cudaEventSynchronize(eqStop));

    // Stop timing for histogram equalization
    float eqTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&eqTime, eqStart, eqStop));
    std::cout << "Histogram equalization time: " << eqTime << " ms" << std::endl;

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
    cv::imwrite("outputs/my_image_cuda.jpg", outputImage);
}

int main()
{
    // Load an image with OpenCV
    cv::Mat inputImage = cv::imread("images/f.jpg", cv::IMREAD_COLOR); // Load in color (3 channels)
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Equalize the image using CUDA
    equalizeImageWithCUDA(inputImage);

    return 0;
}
