#include "Utility.h"

int main() {
    std::string inputPath = "../images/img1.bmp";
    std::string csvPath = "../execution_times_sequential.csv";

    cv::Mat img = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << inputPath << std::endl;
        return -1;
    }

    std::vector<int> sizes = {128, 256, 512, 1024, 2048};

    std::ofstream csvFile(csvPath, std::ios::app);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file at " << csvPath << std::endl;
        return -1;
    }

    csvFile.seekp(0, std::ios::end);
    if (csvFile.tellp() == 0) {
        csvFile << "Image Size,Histogram Time (ms),CDF Time (ms),Equalization Time (ms),Total Time (ms),Threads,Blocks\n";
    }

    std::vector<int> histogram(256), cdf(256);
    cv::Mat resizedImg;

    for (const auto& size : sizes) {
        auto start = std::chrono::high_resolution_clock::now();

        cv::resize(img, resizedImg, cv::Size(size, size));

        auto histStart = std::chrono::high_resolution_clock::now();
        computeHistogram(resizedImg, histogram);
        auto histEnd = std::chrono::high_resolution_clock::now();
        double histTime = std::chrono::duration<double, std::milli>(histEnd - histStart).count();

        auto cdfStart = std::chrono::high_resolution_clock::now();
        computeCDF(histogram, cdf);
        auto cdfEnd = std::chrono::high_resolution_clock::now();
        double cdfTime = std::chrono::duration<double, std::milli>(cdfEnd - cdfStart).count();

        auto eqStart = std::chrono::high_resolution_clock::now();
        equalizeImage(resizedImg, cdf);
        auto eqEnd = std::chrono::high_resolution_clock::now();
        double eqTime = std::chrono::duration<double, std::milli>(eqEnd - eqStart).count();

        auto end = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Processed size " << size
                  << " in " << totalTime << " ms (Histogram: " << histTime
                  << " ms, CDF: " << cdfTime << " ms, Equalization: " << eqTime << " ms)\n";

        int threads = 1; // Fixed value
        int blocks = 1;  // Fixed value
        csvFile << size << ","
                << histTime << "," << cdfTime << "," << eqTime << ","
                << totalTime << "," << threads << "," << blocks << "\n";

        std::string outputPath = "../outputs/resized_" + std::to_string(size) + ".bmp";
        cv::imwrite(outputPath, resizedImg);
    }

    csvFile.close();
    return 0;
}
