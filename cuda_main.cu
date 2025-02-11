#include "UtilityCuda.cu"

void histogram_equalization_cuda(cv::Mat& img, std::string method) {
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
    dim3 block(TILE_SIZE, TILE_SIZE);  // Block dimensions
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
    dim3 block_tiling(TILE_SIZE, TILE_SIZE);  // Tile dimensions

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

    // Save execution times in CSV format
    std::vector<std::tuple<int, int, int, std::string, double, int, int, int, int>> executionTimes;
    int ch = 3;
    if (method == "Grayscale") {
        ch = 1;
    }
    executionTimes.emplace_back(cols, rows, ch, method, total_time, block.x, block.y, TILE_SIZE, TILE_SIZE);

    // Save all execution times to CSV file
    saveExecutionTimesToCSV(executionTimes);
}

int main() {

    std::vector size = {128, 256, 1024, 2048};
    for (int s : size) {
        Mat img_color = imread("images/img4.bmp", IMREAD_COLOR);
        cv::resize(img_color, img_color, cv::Size(s, s));
    Mat img_gray = imread("images/img4.bmp", IMREAD_GRAYSCALE);
    cv::resize(img_gray, img_gray, cv::Size(s, s));
    if (img_gray.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    
    std::cout << "Processing single-channel image (grayscale)..." << std::endl;
    histogram_equalization_cuda(img_gray, "Grayscale");

    std::cout << "Processing multi-channel image (RGB), converting to YCbCr..." << std::endl;
    cv::Mat ycbcr_img;
    cv::cvtColor(img_color, ycbcr_img, cv::COLOR_BGR2YCrCb);

    // Split the YCbCr image into Y, Cb, and Cr channels
    std::vector<cv::Mat> channels(3);
    cv::split(ycbcr_img, channels);

    // Extract the Y channel
    cv::Mat& y_channel = channels[0];

     // Use the old histogram equalization function for the Y channel
    histogram_equalization_cuda(y_channel, "YCbCr");

    // Merge the Y, Cb, and Cr channels back
    cv::merge(channels, ycbcr_img);

    // Convert YCbCr back to RGB
    cv::cvtColor(ycbcr_img, img_color, cv::COLOR_YCrCb2BGR);
   

    imwrite("outputs/equalized.jpg", img_color);
    imshow("Equalized color Image", img_color);
    imshow("Equalized grayscale Image", img_gray);
    waitKey(0);
    }
    

    return 0;
}

