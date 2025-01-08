#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

// Function to equalize histogram for grayscale images using OpenCV
void equalizeHistogramGrayOpenCV(const std::string& inputPath, const std::string& outputPath) {
    cv::Mat img = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << inputPath << std::endl;
        return;
    }
    cv::Mat equalizedImg;
    cv::equalizeHist(img, equalizedImg);
    cv::imwrite(outputPath, equalizedImg);
}

// Function to equalize histogram for colored images using OpenCV
void equalizeHistogramColorOpenCV(const std::string& inputPath, const std::string& outputPath) {
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << inputPath << std::endl;
        return;
    }
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (auto& channel : channels) {
        cv::equalizeHist(channel, channel);
    }
    cv::Mat equalizedImg;
    cv::merge(channels, equalizedImg);
    cv::imwrite(outputPath, equalizedImg);
}

// Function to equalize the histogram for grayscale images without OpenCV
void equalizeHistogramGrayManual(const cv::Mat& inputImage, cv::Mat& outputImage) {
    if (inputImage.channels() != 1) {
        std::cerr << "Error: Input image must be grayscale!" << std::endl;
        return;
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
}

// Function to equalize the histogram for color images without OpenCV
void equalizeHistogramColorManual(const cv::Mat& inputImage, cv::Mat& outputImage) {
    if (inputImage.channels() != 3) {
        std::cerr << "Error: Input image must be color (3 channels)!" << std::endl;
        return;
    }

    std::vector<cv::Mat> channels;
    cv::split(inputImage, channels);
    for (auto& channel : channels) {
        cv::Mat equalizedChannel;
        equalizeHistogramGrayManual(channel, equalizedChannel);
        channel = equalizedChannel;
    }
    cv::merge(channels, outputImage);
}

// Example main function
int main() {
    std::string inputGrayPath = "../images/img1.bmp";
    std::string outputGrayOpenCVPath = "../outputs/gray_equalized_opencv.bmp";
    std::string outputGrayManualPath = "../outputs/gray_equalized_manual.bmp";

    std::string inputColorPath = "../images/img1.bmp";
    std::string outputColorOpenCVPath = "../outputs/color_equalized_opencv.bmp";
    std::string outputColorManualPath = "../outputs/color_equalized_manual.bmp";

    // Grayscale Equalization
    equalizeHistogramGrayOpenCV(inputGrayPath, outputGrayOpenCVPath);
    cv::Mat grayImg = cv::imread(inputGrayPath, cv::IMREAD_GRAYSCALE);
    cv::Mat equalizedGrayImg;
    equalizeHistogramGrayManual(grayImg, equalizedGrayImg);
    cv::imwrite(outputGrayManualPath, equalizedGrayImg);

    // Color Equalization
    equalizeHistogramColorOpenCV(inputColorPath, outputColorOpenCVPath);
    cv::Mat colorImg = cv::imread(inputColorPath);
    cv::Mat equalizedColorImg;
    equalizeHistogramColorManual(colorImg, equalizedColorImg);
    cv::imwrite(outputColorManualPath, equalizedColorImg);

    return 0;
}
