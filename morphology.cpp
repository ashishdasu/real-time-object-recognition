/*
 * morphology.cpp
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Written from scratch using direct pixel access. Operates on single-channel
 * binary (CV_8UC1) images where 255 = foreground, 0 = background.
 */

#include "morphology.h"

void erode(const cv::Mat &src, cv::Mat &dst, int kernelSize) {
    dst = cv::Mat::zeros(src.size(), src.type());
    int half = kernelSize / 2;

    for (int r = half; r < src.rows - half; r++) {
        for (int c = half; c < src.cols - half; c++) {
            bool allFg = true;
            // A pixel survives erosion only if its entire neighborhood is foreground
            for (int kr = -half; kr <= half && allFg; kr++)
                for (int kc = -half; kc <= half && allFg; kc++)
                    if (src.at<uchar>(r + kr, c + kc) == 0)
                        allFg = false;

            dst.at<uchar>(r, c) = allFg ? 255 : 0;
        }
    }
}

void dilate(const cv::Mat &src, cv::Mat &dst, int kernelSize) {
    dst = cv::Mat::zeros(src.size(), src.type());
    int half = kernelSize / 2;

    for (int r = half; r < src.rows - half; r++) {
        for (int c = half; c < src.cols - half; c++) {
            bool anyFg = false;
            // A pixel becomes foreground if any neighbor is foreground
            for (int kr = -half; kr <= half && !anyFg; kr++)
                for (int kc = -half; kc <= half && !anyFg; kc++)
                    if (src.at<uchar>(r + kr, c + kc) == 255)
                        anyFg = true;

            dst.at<uchar>(r, c) = anyFg ? 255 : 0;
        }
    }
}

void applyMorphology(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat tmp;
    // Closing: fills holes from logos, buttons, and camera cutouts
    dilate(src, tmp, 7);
    erode(tmp, dst, 7);
}
