#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    // Set the image path directly
    String imagePath = "../images/img1.bmp";  // Change the path as needed

    // Read the image in grayscale
    Mat src = imread(imagePath, IMREAD_GRAYSCALE);

    if (src.empty()) {
        cerr << "Error: Unable to open the image." << endl;
        return EXIT_FAILURE;
    }

    // Calculate histogram for grayscale image
    int histSize = 256;
    const float range[] = { 0, 256 }; 
    const float* histRange[] = { range };

    Mat hist;
    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, histRange);

    // Normalize the histogram
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0));

    normalize(hist, hist, 0, hist_h, NORM_MINMAX);

    // Plot the histogram
    for (int i = 1; i < histSize; i++) {
        line(histImage, 
             Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
             Scalar(255), 2, 8, 0);
    }

    // Show the image and its histogram
    imshow("Source Image", src);
    imshow("Histogram", histImage);
    waitKey();
    return EXIT_SUCCESS;
}
