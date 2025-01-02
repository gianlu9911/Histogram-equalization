#include "Utility.h"

int main() {
    std::string inputPath = "../images/img1.bmp";

    cv::Mat img = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << inputPath << std::endl;
        return -1;
    }

    std::vector<std::pair<int, int>> sizes = {
        {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}
    };

    std::cout << "Original Image Size: " << img.cols << "x" << img.rows << std::endl;

    for (const auto& size : sizes) {
        // Resize the image
        cv::Mat resizedImg;
        cv::resize(img, resizedImg, cv::Size(size.first, size.second));

        auto start = std::chrono::high_resolution_clock::now();
        equalizeHistogram(resizedImg.data, resizedImg.cols, resizedImg.rows);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Processed size " << size.first << "x" << size.second << " in " << elapsed << " ms" << std::endl;

        std::string outputPath = "../outputs/resized_" + std::to_string(size.first) + "x" + std::to_string(size.second) + ".bmp";
        cv::imwrite(outputPath, resizedImg);
    }

    return 0;
}
