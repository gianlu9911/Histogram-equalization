#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>

void computeHistogram(const cv::Mat& img, std::vector<int>& histogram) {
    histogram.assign(256, 0);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            histogram[img.at<uchar>(i, j)]++;
        }
    }
}

void computeCDF(const std::vector<int>& histogram, std::vector<int>& cdf) {
    cdf.assign(256, 0);
    cdf[0] = histogram[0];
    for (size_t i = 1; i < histogram.size(); ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }
}

void equalizeImage(cv::Mat& img, const std::vector<int>& cdf) {
    int totalPixels = img.rows * img.cols;
    int minCDF = *std::min_element(cdf.begin(), cdf.end());
    std::vector<uchar> lut(256);

    #pragma omp parallel for
    for (size_t i = 0; i < cdf.size(); ++i) {
        lut[i] = static_cast<uchar>(((cdf[i] - minCDF) * 255) / (totalPixels - minCDF));
    }

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            img.at<uchar>(i, j) = lut[img.at<uchar>(i, j)];
        }
    }
}

void histogramEqualizationOPENCV(const std::string& imagePath) {
    // Read the input image in grayscale
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not read the image. Check the file path." << std::endl;
        return;
    }

    // Vector of different sizes to resize the image
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};

    // Open the CSV file for appending
    std::ofstream outputFile("../execution_times_sequential.csv", std::ios::app);

    // Write the header if the file is empty
    if (outputFile.tellp() == 0) {
        outputFile << "Image Size,Histogram Time (ms),CDF Time (ms),Equalization Time (ms),Total Time (ms),Threads,Blocks\n";
    }

    // Loop through the sizes and perform histogram equalization for each
    for (int size : sizes) {
        // Resize the image to the desired size (size x size)
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(size, size));

        // Start measuring time for histogram computation
        auto start = std::chrono::high_resolution_clock::now();
        
        // Perform histogram equalization
        cv::Mat equalizedImage;
        cv::equalizeHist(resizedImage, equalizedImage);
        
        // End measuring time for histogram computation
        auto end = std::chrono::high_resolution_clock::now();
        
        // Measure the duration in microseconds, then convert to milliseconds with precision
        auto duration = std::chrono::duration<double>(end - start); // Duration in seconds
        double histogram_time = duration.count() * 1000.0;  // Convert to milliseconds

        // CDF computation (simulated)
        start = std::chrono::high_resolution_clock::now();
        // Simulate CDF calculation (This would typically be another operation)
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end - start);
        double cdf_time = duration.count() * 1000.0; // Convert to milliseconds

        // Equalization time (same as histogram time in this case as both happen in one step)
        double equalization_time = histogram_time;

        // Calculate total time in milliseconds
        double total_time = histogram_time + cdf_time + equalization_time;

        // Output to console
        std::cout << "Image Size: " << size
                  << ", Histogram Time (ms): " << histogram_time
                  << ", CDF Time (ms): " << cdf_time
                  << ", Equalization Time (ms): " << equalization_time
                  << ", Total Time (ms): " << total_time
                  << ", Threads: N/A, Blocks: N/A" << std::endl;

        // Save results to CSV
        outputFile << size << ","
                   << histogram_time << ","
                   << cdf_time << ","
                   << equalization_time << ","
                   << total_time << ","
                   << "N/A" << ","
                   << "N/A" << "\n"; // Placeholder for threads and blocks since they're not used in this code
    }

    // Close the output file
    outputFile.close();

}


