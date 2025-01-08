#include "Utility.h"

int main2() {
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


int main3() { //for opencv version, make me cuter
    std::string imagePath = "../images/img1.bmp"; 
        histogramEqualizationOPENCV(imagePath);
return 0;
}


int main4() {
    std::string inputImagePath = "../images/img1.bmp";  
    std::string outputImagePath = "../outputs/equalized_YUV_img1.bmp"; 

    equalizeColorImageYUV(inputImagePath, outputImagePath);

    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

void equalizeRGBChannels(const std::string& inputPath, const std::string& outputPath) {
    // Read the input image
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) {
        std::cerr << "Error: Could not read the image at " << inputPath << std::endl;
        return;
    }

    // Resize the image (e.g., to 4096x4096 pixels)
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(4096, 4096));  // Resize to a larger size (e.g., 4096x4096)

    // Split the resized image into its R, G, and B channels
    std::vector<cv::Mat> rgbChannels;
    cv::split(resizedImg, rgbChannels);

    // Measure the time for histogram equalization on all channels
    auto start = std::chrono::high_resolution_clock::now();
    for (auto& channel : rgbChannels) {
        cv::equalizeHist(channel, channel);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Merge the channels back together
    cv::Mat equalizedImg;
    cv::merge(rgbChannels, equalizedImg);

    // Save the equalized image
    cv::imwrite(outputPath, equalizedImg);

    // Calculate and print precise equalization time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double milliseconds = duration / 1000.0; // Convert microseconds to milliseconds
    std::cout << "RGB Channels Histogram Equalization Time: " << std::fixed << std::setprecision(3) << milliseconds << " ms" << std::endl;
}

int main5() {
    std::string inputImagePath = "../images/img1.bmp";  // Replace with the path to your input image
    std::string outputImagePath = "../outputs/rgb_equalized_image.jpg";  // Path to save the equalized image

    equalizeRGBChannels(inputImagePath, outputImagePath);

    return 0;
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

// Function to compute the histogram for each color channel
void computeHistogram(const cv::Mat& image, int* hist_r, int* hist_g, int* hist_b)
{
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            hist_r[pixel[2]]++; // Red channel
            hist_g[pixel[1]]++; // Green channel
            hist_b[pixel[0]]++; // Blue channel
        }
    }
}

// Function to compute the CDF for each color channel
void computeCDF(int* hist, unsigned char* cdf, int totalPixels)
{
    int cdf_accum = 0;
    for (int i = 0; i < 256; ++i)
    {
        cdf_accum += hist[i];
        cdf[i] = (unsigned char)((float)cdf_accum * 255 / totalPixels);
    }
}

// Function to apply histogram equalization
void equalizeImage(const cv::Mat& inputImage, cv::Mat& outputImage)
{
    int width = inputImage.cols;
    int height = inputImage.rows;
    int totalPixels = width * height;

    // Histograms for each color channel
    int hist_r[256] = {0}, hist_g[256] = {0}, hist_b[256] = {0};

    // Start timing histogram computation
    auto start = std::chrono::high_resolution_clock::now();

    // Compute histograms for each color channel
    computeHistogram(inputImage, hist_r, hist_g, hist_b);

    // Stop timing histogram computation
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> histogramDuration = end - start;
    std::cout << "Histogram computation time: " << histogramDuration.count() * 1000 << " ms" << std::endl;

    // CDF arrays for each channel
    unsigned char cdf_r[256], cdf_g[256], cdf_b[256];

    // Start timing CDF computation
    start = std::chrono::high_resolution_clock::now();

    // Compute CDFs for each channel
    computeCDF(hist_r, cdf_r, totalPixels);
    computeCDF(hist_g, cdf_g, totalPixels);
    computeCDF(hist_b, cdf_b, totalPixels);

    // Stop timing CDF computation
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cdfDuration = end - start;
    std::cout << "CDF computation time: " << cdfDuration.count() * 1000 << " ms" << std::endl;

    // Start timing histogram equalization
    start = std::chrono::high_resolution_clock::now();

    // Apply the CDF to equalize the image
    outputImage = inputImage.clone();
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            cv::Vec3b& pixel = outputImage.at<cv::Vec3b>(y, x);
            pixel[0] = cdf_b[pixel[0]]; // Apply equalization to blue channel
            pixel[1] = cdf_g[pixel[1]]; // Apply equalization to green channel
            pixel[2] = cdf_r[pixel[2]]; // Apply equalization to red channel
        }
    }

    // Stop timing histogram equalization
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> equalizationDuration = end - start;
    std::cout << "Histogram equalization time: " << equalizationDuration.count() * 1000 << " ms" << std::endl;
}

int main()
{
    // Load an image with OpenCV
    cv::Mat inputImage = cv::imread("../images/f.jpg", cv::IMREAD_COLOR); // Load in color (3 channels)
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Initialize output image
    cv::Mat outputImage;

    // Start overall execution timer
    auto start = std::chrono::high_resolution_clock::now();

    // Equalize the image in a sequential way
    equalizeImage(inputImage, outputImage);

    // Stop overall execution timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> overallDuration = end - start;
    std::cout << "Total execution time: " << overallDuration.count() * 1000 << " ms" << std::endl;

    // Save the processed image
    cv::imwrite("../outputs/my_image_sequential.jpg", outputImage);

    return 0;
}


