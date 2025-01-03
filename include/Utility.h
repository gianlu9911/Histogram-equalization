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