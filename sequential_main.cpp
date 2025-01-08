#include "Utility.h"


int main() {
    std::string inputGrayPath = "../images/img1.bmp";
    std::string outputGrayOpenCVPath = "../outputs/gray_equalized_opencv.bmp";
    std::string outputGrayManualPath = "../outputs/gray_equalized_manual.bmp";

    std::string inputColorPath = "../images/img1.bmp";
    std::string outputColorOpenCVPath = "../outputs/color_equalized_opencv.bmp";
    std::string outputColorManualPath = "../outputs/color_equalized_manual.bmp";

    std::vector<std::tuple<int, int, int, std::string, double>> executionTimes;

    // Grayscale Equalization (OpenCV)
    cv::Mat grayImg = cv::imread(inputGrayPath, cv::IMREAD_GRAYSCALE);
    double timeGrayOpenCV = equalizeHistogramGrayOpenCV(inputGrayPath, outputGrayOpenCVPath, false);
    executionTimes.emplace_back(grayImg.cols, grayImg.rows, grayImg.channels(), "OpenCV Grayscale", timeGrayOpenCV);

    // Grayscale Equalization (Manual)
    cv::Mat equalizedGrayImg;
    double timeGrayManual = equalizeHistogramGrayManual(grayImg, equalizedGrayImg, false);
    cv::imwrite(outputGrayManualPath, equalizedGrayImg);
    executionTimes.emplace_back(grayImg.cols, grayImg.rows, grayImg.channels(), "Manual Grayscale", timeGrayManual);

    // Color Equalization (OpenCV)
    cv::Mat colorImg = cv::imread(inputColorPath);
    double timeColorOpenCV = equalizeHistogramColorOpenCV(inputColorPath, outputColorOpenCVPath, false);
    executionTimes.emplace_back(colorImg.cols, colorImg.rows, colorImg.channels(), "OpenCV Color", timeColorOpenCV);

    // Color Equalization (Manual)
    cv::Mat equalizedColorImg;
    double timeColorManual = equalizeHistogramColorManual(colorImg, equalizedColorImg, false);
    cv::imwrite(outputColorManualPath, equalizedColorImg);
    executionTimes.emplace_back(colorImg.cols, colorImg.rows, colorImg.channels(), "Manual Color", timeColorManual);

    saveExecutionTimesToCSV(executionTimes, "../execution_times_sequential.csv");

    return 0;
}
