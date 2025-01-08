#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

// Function to save execution times and image information to a CSV file
void saveExecutionTimesToCSV(const std::vector<std::tuple<int, int, int, std::string, double>>& times, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file for writing." << std::endl;
        return;
    }

    file << "Width,Height,Channels,Method,ExecutionTime(ms)\n";

    for (const auto& entry : times) {
        file << std::get<0>(entry) << ","
             << std::get<1>(entry) << ","
             << std::get<2>(entry) << ","
             << std::get<3>(entry) << ","
             << std::get<4>(entry) << "\n";
    }

    file.close();
}

// Function to equalize histogram for grayscale images using OpenCV
double equalizeHistogramGrayOpenCV(const std::string& inputPath, const std::string& outputPath, bool printTime = true) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat img = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << inputPath << std::endl;
        return 0.0;
    }
    cv::Mat equalizedImg;
    cv::equalizeHist(img, equalizedImg);
    cv::imwrite(outputPath, equalizedImg);

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration<double, std::milli>(end - start).count();
    if (printTime) {
        std::cout << "OpenCV Grayscale histogram equalization time: " << executionTime << " ms" << std::endl;
    }

    return executionTime;
}

// Function to equalize histogram for colored images using OpenCV
double equalizeHistogramColorOpenCV(const std::string& inputPath, const std::string& outputPath, bool printTime = true) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << inputPath << std::endl;
        return 0.0;
    }
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (auto& channel : channels) {
        cv::equalizeHist(channel, channel);
    }
    cv::Mat equalizedImg;
    cv::merge(channels, equalizedImg);
    cv::imwrite(outputPath, equalizedImg);

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration<double, std::milli>(end - start).count();
    if (printTime) {
        std::cout << "OpenCV Color histogram equalization time: " << executionTime << " ms" << std::endl;
    }

    return executionTime;
}

// Function to equalize the histogram for grayscale images without OpenCV
double equalizeHistogramGrayManual(const cv::Mat& inputImage, cv::Mat& outputImage, bool printTime = true) {
    auto start = std::chrono::high_resolution_clock::now();

    if (inputImage.channels() != 1) {
        std::cerr << "Error: Input image must be grayscale!" << std::endl;
        return 0.0;
    }

    int hist[256] = {0};
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            hist[inputImage.at<uchar>(y, x)]++;
        }
    }

    int cdf[256] = {0};
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    int totalPixels = inputImage.rows * inputImage.cols;
    uchar cdfMin = cdf[0];
    uchar lut[256];
    for (int i = 0; i < 256; ++i) {
        lut[i] = static_cast<uchar>(255.0 * (cdf[i] - cdfMin) / (totalPixels - cdfMin));
    }

    outputImage = inputImage.clone();
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            outputImage.at<uchar>(y, x) = lut[inputImage.at<uchar>(y, x)];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration<double, std::milli>(end - start).count();
    if (printTime) {
        std::cout << "Manual Grayscale histogram equalization time: " << executionTime << " ms" << std::endl;
    }

    return executionTime;
}

// Function to equalize the histogram for color images without OpenCV
double equalizeHistogramColorManual(const cv::Mat& inputImage, cv::Mat& outputImage, bool printTime = true) {
    auto start = std::chrono::high_resolution_clock::now();

    if (inputImage.channels() != 3) {
        std::cerr << "Error: Input image must be color (3 channels)!" << std::endl;
        return 0.0;
    }

    std::vector<cv::Mat> channels;
    cv::split(inputImage, channels);
    for (auto& channel : channels) {
        cv::Mat equalizedChannel;
        equalizeHistogramGrayManual(channel, equalizedChannel, false); // No printing inside
        channel = equalizedChannel;
    }
    cv::merge(channels, outputImage);

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration<double, std::milli>(end - start).count();
    if (printTime) {
        std::cout << "Manual Color histogram equalization time: " << executionTime << " ms" << std::endl;
    }

    return executionTime;
}
