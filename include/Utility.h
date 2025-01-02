#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <numeric>
#include <cmath>

void equalizeHistogram(uchar* pdata, int width, int height, int max_val = 255) {
    int total = width * height;
    const int n_bins = max_val + 1;

    // Compute histogram and initialize LUT
    int hist[n_bins] = {0};
    int lut[n_bins] = {0};

    // Histogram computation
    for (int i = 0; i < total; ++i) {
        hist[pdata[i]]++;
    }

    // Build LUT from cumulative histogram
    int sum = 0;
    float scale = (n_bins - 1.f) / total;

    for (int i = 0; i < n_bins; ++i) {
        sum += hist[i];
        lut[i] = std::max(0, std::min(int(round(sum * scale)), max_val));
    }

    // Apply equalization
    for (int i = 0; i < total; ++i) {
        pdata[i] = lut[pdata[i]];
    }
}