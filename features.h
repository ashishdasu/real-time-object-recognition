/*
 * features.h
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Computes a rotation, scale, and translation invariant feature vector
 * for each segmented region using OpenCV moments. Also computes the axis
 * of least central moment and oriented bounding box for visualization.
 */

#pragma once
#include <opencv2/opencv.hpp>
#include "regions.h"
#include <vector>

struct FeatureVec {
    int         regionId;
    cv::Point2d centroid;
    double      angle;      // orientation of major axis, radians
    double      hu[4];      // first 4 log-scaled Hu moments
    double      percentFilled;  // area / oriented bounding box area
    double      aspectRatio;    // major axis length / minor axis length

    // Bounding box corners for overlay drawing (not used as features)
    cv::Point2f obbCorners[4];
    double      axisLength;     // half-length of major axis for drawing
};

// Compute feature vectors for all regions. Draws OBB and axis onto overlay.
void computeFeatures(const cv::Mat &src,
                     const RegionMap &regionMap,
                     const std::vector<RegionInfo> &regions,
                     std::vector<FeatureVec> &fvecs);

// Return a copy of src with OBB and axis overlays drawn on it.
cv::Mat drawFeatureOverlay(const cv::Mat &src,
                           const std::vector<FeatureVec> &fvecs);
