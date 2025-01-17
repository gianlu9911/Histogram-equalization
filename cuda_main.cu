#include "UtilityCuda.cu"


void equalizeImageWithCUDAGrayscale(const cv::Mat& inputImage, int tile_width, int tile_height) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();  

    // Allocate memory for input image and output image
    unsigned char *d_image, *d_output;
    int *d_hist;
    unsigned char *d_cdf;

    size_t imageSize = width * height * sizeof(unsigned char);  
    size_t histSize = 256 * sizeof(int);

    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_hist, histSize);
    cudaMalloc(&d_cdf, 256 * sizeof(unsigned char));

    // Check for memory allocation errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA memory allocation error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Copy image data from host to device
    cudaMemcpy(d_image, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    // Initialize histogram to zero (memset on the device)
    cudaMemset(d_hist, 0, histSize);

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

    cudaEvent_t stage_start, stage_stop;
    cudaEventCreate(&stage_start);
    cudaEventCreate(&stage_stop);
    cudaEventRecord(stage_start);
    
    // Define block and grid sizes
    dim3 block(tile_width, tile_height);
    dim3 grid((width + tile_width - 1) / tile_width, (height + tile_height - 1) / tile_height);

    // Launch the compute histogram kernel
    computeHistogramGrayscale<<<grid, block>>>(d_image, d_hist, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after computeHistogramGrayscale kernel: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaEventRecord(stage_stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, stage_start, stage_stop);
    std::cout << "Histogram computation time " << milliseconds << " ms" << std::endl;
    csvFile << "Histogram computation," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    cudaEventRecord(stage_start);
    computeCDF<<<dim3(1, 1), 256>>>(d_hist, d_cdf, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after computeCDF kernel: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    cudaEventRecord(stage_stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, stage_start, stage_stop);
    std::cout << "CDF computation time " << milliseconds << " ms" << std::endl;
    csvFile << "CDF computation," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    cudaEventRecord(stage_start);
    equalizeGrayscaleImage<<<grid, block>>>(d_image, d_output, width, height, d_cdf, tile_width, tile_height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after equalizeGrayscaleImage kernel: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaEventRecord(stage_stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, stage_start, stage_stop);
    std::cout << "Equalization time " << milliseconds << " ms" << std::endl;
    csvFile << "Equalization," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total execution time " << milliseconds << " ms" << std::endl;

    csvFile << "Total execution," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    // Copy the result back to the host
    cudaMemcpy(inputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    std::stringstream outputFileName;
    outputFileName << "../outputs/grayscale_output_image_" << width << "x" << height << "_CUDA.jpg";

    if (!cv::imwrite(outputFileName.str(), inputImage)) {
        std::cerr << "Error: Could not save the image!" << std::endl;
    } else {
        std::cout << "Image saved as " << outputFileName.str() << std::endl;
    }

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_hist);
    cudaFree(d_cdf);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(stage_start);
    cudaEventDestroy(stage_stop);

    csvFile.close();
}

void equalizeImageWithCUDA(const cv::Mat& inputImage, int tile_width, int tile_height) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels(); 

    // Allocate memory for input image and output image
    unsigned char *d_image, *d_output;
    int *d_hist_r, *d_hist_g, *d_hist_b;
    unsigned char *d_cdf_r, *d_cdf_g, *d_cdf_b;

    // Allocate device memory 
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up kernel
    warmUpKernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::ofstream csvFile("../execution_times_cuda.csv", std::ios::app);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file!" << std::endl;
        return;
    }

    if (csvFile.tellp() == 0) {
        csvFile << "Stage,Time (ms),Width,Height,Channels,TileWidth,TileHeight" << std::endl;
    }

    cudaEventRecord(start);

    cudaEvent_t stage_start, stage_stop;
    cudaEventCreate(&stage_start);
    cudaEventCreate(&stage_stop);
    cudaEventRecord(stage_start);
    
    // Define block and grid sizes
    dim3 block(tile_width, tile_height);
    dim3 grid((width + tile_width - 1) / tile_width, (height + tile_height - 1) / tile_height);

    computeHistogram<<<grid, block>>>(d_image, d_hist_r, d_hist_g, d_hist_b, width, height);
    cudaEventRecord(stage_stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, stage_start, stage_stop);
    std::cout << "Histogram computation time " << milliseconds << " ms" << std::endl;
    csvFile << "Histogram computation," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    cudaEventRecord(stage_start);
    computeCDF<<<dim3(1, 1), 256>>>(d_hist_r, d_cdf_r, width, height);
    computeCDF<<<dim3(1, 1), 256>>>(d_hist_g, d_cdf_g, width, height);
    computeCDF<<<dim3(1, 1), 256>>>(d_hist_b, d_cdf_b, width, height);
    cudaEventRecord(stage_stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, stage_start, stage_stop);
    std::cout << "CDF computation time " << milliseconds << " ms" << std::endl;
    csvFile << "CDF computation," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    cudaEventRecord(stage_start);
    equalizeRGBImage<<<grid, block>>>(d_image, d_output, width, height, d_cdf_r, d_cdf_g, d_cdf_b, tile_width, tile_height);
    cudaEventRecord(stage_stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, stage_start, stage_stop);
    std::cout << "Equalization time " << milliseconds << " ms" << std::endl;
    csvFile << "Equalization," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total execution time " << milliseconds << " ms" << std::endl;

    csvFile << "Total execution," << milliseconds << "," << width << "," << height << "," << channels << "," << tile_width << "," << tile_height << std::endl;

    cudaMemcpy(inputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    std::stringstream outputFileName;
    outputFileName << "../outputs/RGB_output_image_" << width << "x" << height << "_CUDA.jpg";
    if (!cv::imwrite(outputFileName.str(), inputImage)) {
        std::cerr << "Error: Could not save the image!" << std::endl;
    } else {
        std::cout << "Image saved as " << outputFileName.str() << std::endl;
    }

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_hist_r);
    cudaFree(d_hist_g);
    cudaFree(d_hist_b);
    cudaFree(d_cdf_r);
    cudaFree(d_cdf_g);
    cudaFree(d_cdf_b);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(stage_start);
    cudaEventDestroy(stage_stop);

    csvFile.close();
}



int main() {
    cv::Mat inputImageGray = cv::imread("images/img2.bmp", cv::IMREAD_GRAYSCALE);
    if (inputImageGray.empty()) {
        std::cerr << "Error: Could not load grayscale image!" << std::endl;
        return -1;
    }

    cv::Mat inputImageColor = cv::imread("images/img2.bmp", cv::IMREAD_COLOR);
    if (inputImageColor.empty()) {
        std::cerr << "Error: Could not load color image!" << std::endl;
        return -1;
    }



    std::vector<int> sizes = {128, 256, 512, 1024, 2048};

    for (int size : sizes) {
        // Resize images 
        cv::Mat resizedGray, resizedColor;
        cv::resize(inputImageGray, resizedGray, cv::Size(size, size));
        cv::resize(inputImageColor, resizedColor, cv::Size(size, size));

        std::cout << "Processing grayscale image at resolution: " << size << "x" << size << std::endl;
        equalizeImageWithCUDAGrayscale(resizedGray, 32,32);

        std::cout << "Processing color image at resolution: " << size << "x" << size << std::endl;
        equalizeImageWithCUDA(resizedColor, 32,32);
    }

    return 0;
}
