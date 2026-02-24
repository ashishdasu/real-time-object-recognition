/*
 * classifier.cpp
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Scaled Euclidean distance: each feature dimension is divided by its
 * standard deviation across the training set so no single feature
 * dominates just because of its numerical scale.
 */

#include "classifier.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

static double scaledDistance(const FeatureVec &a, const FeatureVec &b,
                              const std::vector<double> &stdevs) {
    double fa[6] = { a.hu[0], a.hu[1], a.hu[2], a.hu[3], a.percentFilled, a.aspectRatio };
    double fb[6] = { b.hu[0], b.hu[1], b.hu[2], b.hu[3], b.percentFilled, b.aspectRatio };

    double dist = 0.0;
    for (int i = 0; i < 6; i++) {
        double d = (fa[i] - fb[i]) / stdevs[i];
        dist += d * d;
    }
    return std::sqrt(dist);
}

std::string classifyFeature(const FeatureVec &fv,
                            const FeatureDB &db,
                            double unknownThresh) {
    if (db.samples.empty()) return "Unknown";

    double bestDist = std::numeric_limits<double>::max();
    std::string bestLabel = "Unknown";

    for (size_t i = 0; i < db.samples.size(); i++) {
        double d = scaledDistance(fv, db.samples[i], db.stdevs);
        if (d < bestDist) {
            bestDist  = d;
            bestLabel = db.labels[i];
        }
    }

    return (bestDist < unknownThresh) ? bestLabel : "Unknown";
}

std::string classifyFeatureKNN(const FeatureVec &fv,
                               const FeatureDB &db,
                               int k,
                               double unknownThresh) {
    if (db.samples.empty()) return "Unknown";

    // Compute distance to every training sample and sort
    std::vector<std::pair<double, std::string>> dists;
    dists.reserve(db.samples.size());
    for (size_t i = 0; i < db.samples.size(); i++)
        dists.push_back({ scaledDistance(fv, db.samples[i], db.stdevs), db.labels[i] });

    std::sort(dists.begin(), dists.end());

    int K = std::min(k, (int)dists.size());

    // Check if even the nearest neighbor is too far
    double avgDist = 0.0;
    for (int i = 0; i < K; i++) avgDist += dists[i].first;
    avgDist /= K;
    if (avgDist > unknownThresh) return "Unknown";

    // Majority vote among K nearest
    std::map<std::string, int> votes;
    for (int i = 0; i < K; i++) votes[dists[i].second]++;

    return std::max_element(votes.begin(), votes.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; })->first;
}

void classifyAndLabel(cv::Mat &display,
                      const std::vector<FeatureVec> &fvecs,
                      const FeatureDB &db) {
    for (const auto &fv : fvecs) {
        std::string label = classifyFeatureKNN(fv, db);
        cv::Point pt((int)fv.centroid.x, (int)fv.centroid.y - 20);
        cv::putText(display, label, pt,
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 0}, 4);
        cv::putText(display, label, pt,
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 128}, 2);
    }
}
