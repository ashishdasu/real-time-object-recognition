/*
 * morphology.h
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Erosion and dilation on binary images using a rectangular structuring
 * element with direct pixel access. Chained into a closing operation
 * (dilate -> erode) to fill holes left by object textures and markings.
 */

#pragma once
#include <opencv2/opencv.hpp>

// Erode a binary image: a pixel stays 255 only if every pixel in its
// (kernelSize x kernelSize) neighborhood is also 255.
void erode(const cv::Mat& src, cv::Mat& dst, int kernelSize);

// Dilate a binary image: a pixel becomes 255 if any pixel in its
// (kernelSize x kernelSize) neighborhood is 255.
void dilate(const cv::Mat& src, cv::Mat& dst, int kernelSize);

// Closing (dilate then erode): fills small holes inside objects
// without significantly changing their outer boundary.
void applyMorphology(const cv::Mat& src, cv::Mat& dst);
