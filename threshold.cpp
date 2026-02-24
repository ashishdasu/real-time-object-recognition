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
    cv::Mat gray, blurred;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    int t = isodataThreshold(blurred);
    // Objects are darker than the white background, so invert the binary result
    cv::threshold(blurred, dst, t, 255, cv::THRESH_BINARY_INV);
}
