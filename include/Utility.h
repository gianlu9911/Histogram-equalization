#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

void saveExecutionTimesToCSV(const std::vector<std::tuple<int, int, int, std::string, double>>& executionTimes, const std::string& filename) {
    std::ofstream file(filename, std::ios::app); // Open in append mode

    // Write header if the file is empty
    if (file.tellp() == 0) {
        file << "Width,Height,Channels,Method,ExecutionTime(ms)" << std::endl;
    }

    // Write execution times data to CSV
    for (const auto& execTime : executionTimes) {
        file << std::get<0>(execTime) << ","
             << std::get<1>(execTime) << ","
             << std::get<2>(execTime) << ","
             << std::get<3>(execTime) << ","
             << std::get<4>(execTime) << std::endl;
    }

    file.close();
}

// Function to equalize histogram for grayscale images using OpenCV
double equalizeHistogramGrayOpenCV(const cv::Mat& inputImage, cv::Mat& outputImage, bool printTime = true) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat equalizedImg;
    cv::equalizeHist(inputImage, equalizedImg);
    outputImage = equalizedImg;

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration<double, std::milli>(end - start).count();
    if (printTime) {
        std::cout << "OpenCV Grayscale histogram equalization time: " << executionTime << " ms" << std::endl;
    }

    return executionTime;
}

// Function to equalize histogram for colored images using OpenCV
double equalizeHistogramColorOpenCV(const cv::Mat& inputImage, cv::Mat& outputImage, bool printTime = true) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<cv::Mat> channels;
    cv::split(inputImage, channels);
    for (auto& channel : channels) {
        cv::equalizeHist(channel, channel);
    }
    cv::Mat equalizedImg;
    cv::merge(channels, equalizedImg);
    outputImage = equalizedImg;

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration<double, std::milli>(end - start).count();
    if (printTime) {
        std::cout << "OpenCV Color histogram equalization time: " << executionTime << " ms" << std::endl;
    }

    return executionTime;
}

// Function to equalize the histogram for grayscale images manually
double equalizeHistogramGrayManual(const cv::Mat& inputImage, cv::Mat& outputImage, bool printTime = true) {
    auto start = std::chrono::high_resolution_clock::now();

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

// Function to equalize the histogram for color images manually
double equalizeHistogramColorManual(const cv::Mat& inputImage, cv::Mat& outputImage, bool printTime = true) {
    auto start = std::chrono::high_resolution_clock::now();

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