/*
 * classifier.h
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Nearest-neighbor classification using scaled Euclidean distance.
 * Flags "Unknown" when the nearest neighbor distance exceeds a threshold,
 * which handles objects not present in the training database.
 */

#pragma once
#include "features.h"
#include "database.h"
#include <string>

// 1-nearest-neighbor. Returns "Unknown" if distance exceeds unknownThresh.
std::string classifyFeature(const FeatureVec &fv,
                            const FeatureDB &db,
                            double unknownThresh = 3.0);

// K-nearest-neighbor with majority vote. Returns "Unknown" if the average
// distance of the K winners exceeds unknownThresh.
std::string classifyFeatureKNN(const FeatureVec &fv,
                               const FeatureDB &db,
                               int k = 3,
                               double unknownThresh = 3.0);

// Draw the classified label on the image at each region's centroid.
// Uses KNN by default.
void classifyAndLabel(cv::Mat &display,
                      const std::vector<FeatureVec> &fvecs,
                      const FeatureDB &db);
