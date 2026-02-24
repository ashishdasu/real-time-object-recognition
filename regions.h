/*
 * regions.h
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Connected components segmentation using OpenCV's connectedComponentsWithStats.
 * Filters out small regions and those touching the image border, then
 * renders the remaining regions as a color-coded map for display.
 */

#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct RegionInfo {
    int id;
    int area;
    cv::Point2d centroid;
    cv::Rect boundingBox;
};

// Label type returned by connectedComponents
using RegionMap = cv::Mat;

// Run connected components on a cleaned binary image. Filters regions that
// are too small or touch the image border. Results sorted by area descending.
void segmentRegions(const cv::Mat &binary,
                    RegionMap &regionMap,
                    std::vector<RegionInfo> &regions);

// Render a BGR color map with each region drawn in a distinct color.
cv::Mat drawRegions(const RegionMap &regionMap,
                    const std::vector<RegionInfo> &regions);
