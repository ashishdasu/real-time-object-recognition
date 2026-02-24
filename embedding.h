/*
 * embedding.h
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * One-shot classification using ResNet18 embeddings. Pre-processes each
 * region (rotate to align axis, crop OBB, resize to 224x224), runs it
 * through a pre-trained network, and classifies using sum-squared distance
 * on the 512-dim embedding vector.
 */

#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "features.h"
#include "regions.h"
#include <string>
#include <vector>

struct EmbeddingDB {
    std::vector<std::string> labels;
    std::vector<cv::Mat>     embeddings;  // each is 1x512 CV_32F
};

// Load the ResNet18 ONNX model. Returns false if the file is not found.
bool loadNetwork(const std::string &modelPath, cv::dnn::Net &net);

// Extract the 512-dim embedding for the region described by fv.
// Returns false if the ROI is degenerate (too small to crop).
bool getRegionEmbedding(const cv::Mat &frame,
                        const FeatureVec &fv,
                        const RegionMap &regionMap,
                        cv::dnn::Net &net,
                        cv::Mat &embedding);

// Nearest-neighbor classification on the embedding DB using SSD distance.
// Returns "Unknown" if the best distance exceeds unknownThresh.
std::string classifyEmbedding(const cv::Mat &embedding,
                              const EmbeddingDB &db,
                              double unknownThresh = 5e6);

void addEmbeddingSample(EmbeddingDB &db,
                        const std::string &label,
                        const cv::Mat &embedding);

bool saveEmbeddingDB(const std::string &path, const EmbeddingDB &db);
bool loadEmbeddingDB(const std::string &path, EmbeddingDB &db);
