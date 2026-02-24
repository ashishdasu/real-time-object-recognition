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

// Classify a feature vector against the DB. Returns the nearest label,
// or "Unknown" if the scaled distance exceeds unknownThresh.
std::string classifyFeature(const FeatureVec &fv,
                            const FeatureDB &db,
                            double unknownThresh = 3.0);

// Draw the classified label on the image at each region's centroid.
void classifyAndLabel(cv::Mat &display,
                      const std::vector<FeatureVec> &fvecs,
                      const FeatureDB &db);
