/*
 * embedding.cpp
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Uses prepEmbeddingImage and getEmbedding from utilities.cpp to pre-process
 * each region and extract a ResNet18 embedding. Classification uses
 * sum-squared distance (SSD) on the 512-dim feature vector.
 */

#include "embedding.h"
#include <cmath>
#include <limits>
#include <iostream>

// Declared in utilities.cpp
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage,
                        int cx, int cy, float theta,
                        float minE1, float maxE1,
                        float minE2, float maxE2,
                        int debug);

int getEmbedding(cv::Mat &src, cv::Mat &embedding,
                 cv::dnn::Net &net, int debug);

bool loadNetwork(const std::string &modelPath, cv::dnn::Net &net) {
    try {
        net = cv::dnn::readNet(modelPath);
    } catch (const cv::Exception &e) {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return false;
    }
    return !net.empty();
}

// Project all foreground pixels in the region onto the primary and secondary
// axes to find the actual spatial extents (not just the eigenvalue estimate).
static void computeAxisExtents(const RegionMap &regionMap, int regionId,
                                const cv::Point2d &centroid, double angle,
                                float &minE1, float &maxE1,
                                float &minE2, float &maxE2) {
    double ca = std::cos(angle), sa = std::sin(angle);
    minE1 = minE2 = std::numeric_limits<float>::max();
    maxE1 = maxE2 = std::numeric_limits<float>::lowest();

    for (int r = 0; r < regionMap.rows; r++) {
        const int *row = regionMap.ptr<int>(r);
        for (int c = 0; c < regionMap.cols; c++) {
            if (row[c] != regionId) continue;
            double dx = c - centroid.x;
            double dy = r - centroid.y;
            float p1 = (float)(dx * ca + dy * sa);
            float p2 = (float)(-dx * sa + dy * ca);
            if (p1 < minE1) minE1 = p1;
            if (p1 > maxE1) maxE1 = p1;
            if (p2 < minE2) minE2 = p2;
            if (p2 > maxE2) maxE2 = p2;
        }
    }
}

bool getRegionEmbedding(const cv::Mat &frame,
                        const FeatureVec &fv,
                        const RegionMap &regionMap,
                        cv::dnn::Net &net,
                        cv::Mat &embedding) {
    float minE1, maxE1, minE2, maxE2;
    computeAxisExtents(regionMap, fv.regionId, fv.centroid, fv.angle,
                       minE1, maxE1, minE2, maxE2);

    if ((maxE1 - minE1) < 5 || (maxE2 - minE2) < 5) return false;

    cv::Mat frameCopy = frame.clone();
    cv::Mat roi;
    prepEmbeddingImage(frameCopy, roi,
                       (int)fv.centroid.x, (int)fv.centroid.y,
                       (float)fv.angle,
                       minE1, maxE1, minE2, maxE2, 0);

    if (roi.empty()) return false;

    getEmbedding(roi, embedding, net, 0);
    return true;
}

std::string classifyEmbedding(const cv::Mat &embedding,
                              const EmbeddingDB &db,
                              double unknownThresh) {
    if (db.embeddings.empty()) return "Unknown";

    double bestDist = std::numeric_limits<double>::max();
    std::string bestLabel = "Unknown";

    for (size_t i = 0; i < db.embeddings.size(); i++) {
        cv::Mat diff = embedding - db.embeddings[i];
        double ssd = diff.dot(diff);
        if (ssd < bestDist) {
            bestDist  = ssd;
            bestLabel = db.labels[i];
        }
    }

    return (bestDist < unknownThresh) ? bestLabel : "Unknown";
}

void addEmbeddingSample(EmbeddingDB &db,
                        const std::string &label,
                        const cv::Mat &embedding) {
    db.labels.push_back(label);
    db.embeddings.push_back(embedding.clone());
}

bool saveEmbeddingDB(const std::string &path, const EmbeddingDB &db) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;

    fs << "count" << (int)db.labels.size();
    for (int i = 0; i < (int)db.labels.size(); i++) {
        fs << ("label_" + std::to_string(i)) << db.labels[i];
        fs << ("emb_"   + std::to_string(i)) << db.embeddings[i];
    }
    return true;
}

bool loadEmbeddingDB(const std::string &path, EmbeddingDB &db) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    db.labels.clear();
    db.embeddings.clear();

    int count = 0;
    fs["count"] >> count;
    for (int i = 0; i < count; i++) {
        std::string label;
        cv::Mat emb;
        fs["label_" + std::to_string(i)] >> label;
        fs["emb_"   + std::to_string(i)] >> emb;
        db.labels.push_back(label);
        db.embeddings.push_back(emb);
    }
    return true;
}
