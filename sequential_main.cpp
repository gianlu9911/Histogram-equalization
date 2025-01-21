#include "Utility.h"

int equalize_histogram_sequential(int size) {

    bool prints = true ;

    std::string inputGrayPath = "../images/img1.bmp";
    std::string outputGrayOpenCVPath = "../outputs/gray_equalized_opencv_" + std::to_string(size) + ".bmp";
    std::string outputGrayManualPath = "../outputs/gray_equalized_manual_" + std::to_string(size) + ".bmp";

    std::string inputColorPath = "../images/img1.bmp";
    std::string outputColorOpenCVPath = "../outputs/color_equalized_opencv_" + std::to_string(size) + ".bmp";
    std::string outputColorManualPath = "../outputs/color_equalized_manual_" + std::to_string(size) + ".bmp";
    std::string outputColorYCbCrManualPath = "../outputs/color_equalized_ycbcr_manual_" + std::to_string(size) + ".bmp";

    std::vector<std::tuple<int, int, int, std::string, double>> executionTimes;

    cv::Mat originalImg = cv::imread(inputGrayPath);

    // Resize the image to the desired size (size x size dimensions)
    int newSize = size; 
    if (newSize <= 0) return -1;
    cv::Mat resizedImg;
    cv::resize(originalImg, resizedImg, cv::Size(newSize, newSize)); // Resize image to (size, size)

    // Grayscale Equalization (OpenCV)
    cv::Mat grayImg = resizedImg.clone();
    cv::cvtColor(resizedImg, grayImg, cv::COLOR_BGR2GRAY); // Convert to grayscale
    double timeGrayOpenCV = equalizeHistogramGrayOpenCV(grayImg, grayImg, prints); // Measure only the equalization
    executionTimes.emplace_back(grayImg.cols, grayImg.rows, grayImg.channels(), "OpenCV Grayscale", timeGrayOpenCV);

    // Grayscale Equalization (Manual)
    cv::Mat equalizedGrayImg;
    double timeGrayManual = equalizeHistogramGrayManual(grayImg, equalizedGrayImg, prints);
    cv::imwrite(outputGrayManualPath, equalizedGrayImg);
    executionTimes.emplace_back(equalizedGrayImg.cols, equalizedGrayImg.rows, equalizedGrayImg.channels(), "Manual Grayscale", timeGrayManual);

    // Color Equalization (OpenCV)
    cv::Mat colorImg = resizedImg.clone();
    double timeColorOpenCV = equalizeHistogramColorOpenCV(colorImg, colorImg, prints);
    executionTimes.emplace_back(colorImg.cols, colorImg.rows, colorImg.channels(), "OpenCV Color", timeColorOpenCV);

    // Color Equalization (Manual)
    cv::Mat equalizedColorImg;
    double timeColorManual = equalizeHistogramColorManual(colorImg, equalizedColorImg, prints);
    cv::imwrite(outputColorManualPath, equalizedColorImg);
    executionTimes.emplace_back(equalizedColorImg.cols, equalizedColorImg.rows, equalizedColorImg.channels(), "Manual Color", timeColorManual);

    // Color Equalization in YCbCr (Manual)
    cv::Mat equalizedColorYCbCrImg;
    double timeColorYCbCrManual = equalizeHistogramRGB_YCbCrManual(colorImg, equalizedColorYCbCrImg, prints);
    cv::imwrite(outputColorYCbCrManualPath, equalizedColorYCbCrImg);
    executionTimes.emplace_back(equalizedColorYCbCrImg.cols, equalizedColorYCbCrImg.rows, equalizedColorYCbCrImg.channels(), "Manual YCbCr Color", timeColorYCbCrManual);

    // Color Equalization in YCbCr (OpenCV)
    cv::Mat equalizedColorYCbCrImgOpencv;
    double timeColorYCbCrOpenCV = equalizeHistogramRGB_YCbCrOpenCV(colorImg, equalizedColorYCbCrImgOpencv, prints);
    cv::imwrite(outputColorYCbCrManualPath, equalizedColorYCbCrImgOpencv);
    executionTimes.emplace_back(equalizedColorYCbCrImgOpencv.cols, equalizedColorYCbCrImgOpencv.rows, equalizedColorYCbCrImgOpencv.channels(), "OpenCV YCbCr Color", timeColorYCbCrOpenCV);

    saveExecutionTimesToCSV(executionTimes, "../execution_times_sequential.csv");

    return 0;
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};

    for (int s : sizes) {
        equalize_histogram_sequential(s);
    }

    return 0;
}