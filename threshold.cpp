/*
 * threshold.cpp
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Written from scratch. Uses the ISODATA algorithm (iterative k-means, K=2)
 * to find a dynamic threshold rather than a fixed value. Works on grayscale
 * after a light blur to suppress high-frequency noise.
 */

#include "threshold.h"
#include <cmath>
#include <vector>

// Samples 1/16 of the pixels and runs iterative k-means with K=2 until the
// two cluster means converge. Returns the midpoint as the threshold value.
static int isodataThreshold(const cv::Mat &gray) {
    std::vector<uchar> samples;
    samples.reserve((gray.rows / 4) * (gray.cols / 4));
    for (int r = 0; r < gray.rows; r += 4)
        for (int c = 0; c < gray.cols; c += 4)
            samples.push_back(gray.at<uchar>(r, c));

    // Start means at the quarter-points of the intensity range
    double m0 = 64.0, m1 = 192.0;

    for (int iter = 0; iter < 100; ++iter) {
        double sum0 = 0, sum1 = 0;
        int    cnt0 = 0, cnt1 = 0;

        for (uchar v : samples) {
            if (std::abs((double)v - m0) <= std::abs((double)v - m1)) {
                sum0 += v; cnt0++;
            } else {
                sum1 += v; cnt1++;
            }
        }

        double new0 = cnt0 ? sum0 / cnt0 : m0;
        double new1 = cnt1 ? sum1 / cnt1 : m1;

        if (std::abs(new0 - m0) < 0.5 && std::abs(new1 - m1) < 0.5) break;
        m0 = new0;
        m1 = new1;
    }

    return static_cast<int>((m0 + m1) / 2.0);
}

void applyThreshold(const cv::Mat &src, cv::Mat &dst) {
    // Convert to HSV so we can darken highly saturated pixels before thresholding.
    // This lets colored objects (e.g. bright blue clip) read as dark against the
    // white (low-saturation) background, which ISODATA would otherwise miss.
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Mat gray(src.rows, src.cols, CV_8UC1);
    for (int r = 0; r < hsv.rows; r++) {
        for (int c = 0; c < hsv.cols; c++) {
            cv::Vec3b p = hsv.at<cv::Vec3b>(r, c);
            uchar sat = p[1];   // 0-255
            uchar val = p[2];   // 0-255 (brightness)
            // Pull saturated pixels toward dark: the more saturated, the darker.
            uchar adjusted = (uchar)(val * (1.0f - 0.85f * (sat / 255.0f)));
            gray.at<uchar>(r, c) = adjusted;
        }
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    int t = isodataThreshold(blurred);
    cv::threshold(blurred, dst, t, 255, cv::THRESH_BINARY_INV);
}
