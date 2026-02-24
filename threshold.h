/*
 * threshold.h
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Separates dark objects from a light background using a dynamically
 * computed threshold (ISODATA / two-class iterative k-means).
 */

#pragma once
#include <opencv2/opencv.hpp>

// Converts src to a binary mask: 255 = foreground (dark object), 0 = background.
void applyThreshold(const cv::Mat &src, cv::Mat &dst);
