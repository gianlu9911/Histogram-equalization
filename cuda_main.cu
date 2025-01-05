#include "UtilityCuda.cu"

int main() {
    std::string inputPath = "../images/img1.bmp";

    cv::Mat img = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << inputPath << std::endl;
        return -1;
    }

    std::string csvPath = "../execution_times_cuda.csv";

    // Open the CSV file in append mode
    std::ofstream csvFile(csvPath, std::ios::out | std::ios::app);

    // Write the header if the file is empty
    if (csvFile.tellp() == 0) {
        csvFile << "Image Size,Histogram Time (ms),CDF Time (ms),Equalization Time (ms),Total Time (ms),Threads,Blocks\n";
    }

    std::vector<int> sizes = {128, 256, 512, 1024, 2048}; // size test
    std::vector<int> thread_configs = {32, 64, 128, 256, 512, 1024};   // threads configurations
    std::vector<int> block_configs = {32, 256, 1024, 1024*8};      // block configurations

    for (int size : sizes) {
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(size, size));

        int width = resized_img.cols;
        int height = resized_img.rows;

        unsigned char *d_input, *d_output;
        int *d_hist, *d_cdf;
        CUDA_CHECK(cudaMalloc(&d_input, width * height * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cdf, 256 * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_input, resized_img.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_hist, 0, 256 * sizeof(int)));

        cudaEvent_t start, stop;
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        float total_execution_time = 0.0f;  // To track the total execution time for all operations

        for (int threads : thread_configs) {
            for (int blocks : block_configs) {
                int blocks_histogram = (width * height + threads - 1) / threads;

                float histogram_time = 0.0f;
                float cdf_time = 0.0f;
                float equalization_time = 0.0f;

                // Histogram computation
                CUDA_CHECK(cudaEventRecord(start));
                compute_histogram<<<blocks_histogram, threads>>>(d_input, d_hist, width, height);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
                histogram_time = milliseconds;

                // CDF computation
                CUDA_CHECK(cudaEventRecord(start));
                compute_cdf<<<1, 256>>>(d_hist, d_cdf, width * height);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
                cdf_time = milliseconds;

                // Equalize image
                CUDA_CHECK(cudaEventRecord(start));
                equalize_image<<<blocks_histogram, threads>>>(d_output, d_input, d_cdf, width, height);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
                equalization_time = milliseconds;

                // Calculate total time for this configuration
                float total_time = histogram_time + cdf_time + equalization_time;
                total_execution_time += total_time;

                std::cout << "Image Size: " << size 
                          << ", Histogram Time (ms): " << histogram_time 
                          << ", CDF Time (ms): " << cdf_time 
                          << ", Equalization Time (ms): " << equalization_time 
                          << ", Total Time (ms): " << total_time 
                          << ", Threads: " << threads
                          << ", Blocks: " << blocks << std::endl;

                // Save results for each thread/block configuration
                csvFile << size << ","
                        << histogram_time << ","
                        << cdf_time << ","
                        << equalization_time << ","
                        << total_time << ","
                        << threads << ","
                        << blocks << "\n";
            }
        }



        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_hist));
        CUDA_CHECK(cudaFree(d_cdf));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    csvFile.close();

    std::cout << "Execution times saved to " << csvPath << std::endl;
    return 0;
}
