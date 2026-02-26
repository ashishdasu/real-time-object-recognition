/*
 * regions.cpp
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Runs connected-component labeling on the binary mask and filters out
 * noise (small area) and border-touching components before returning
 * the N largest qualifying regions sorted by area.
 */

#include "regions.h"

#include <algorithm>

// Ignore any region smaller than this (noise, shadows, paper texture)
static const int MIN_AREA = 500;

// Only keep the N largest qualifying regions
static const int MAX_REGIONS = 5;

// Run connectedComponentsWithStats, drop anything too small or touching the
// border, return the N largest blobs sorted by area.
void segmentRegions(const cv::Mat& binary, RegionMap& regionMap,
                    std::vector<RegionInfo>& regions) {
    regions.clear();

    cv::Mat stats, centroids;
    int nLabels =
        cv::connectedComponentsWithStats(binary, regionMap, stats, centroids);

    for (int i = 1; i < nLabels; i++) {  // label 0 is background
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < MIN_AREA) continue;

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // Skip regions whose bounding box touches the image border — these are
        // usually the background leaking through or partial objects at the edge
        if (x <= 1 || y <= 1 || x + w >= binary.cols - 1 ||
            y + h >= binary.rows - 1)
            continue;

        RegionInfo ri;
        ri.id = i;
        ri.area = area;
        ri.centroid = {centroids.at<double>(i, 0), centroids.at<double>(i, 1)};
        ri.boundingBox = cv::Rect(x, y, w, h);
        regions.push_back(ri);
    }

    // Keep the largest regions — smaller ones are likely debris or partial
    // objects
    std::sort(regions.begin(), regions.end(),
              [](const RegionInfo& a, const RegionInfo& b) {
                  return a.area > b.area;
              });

    if ((int)regions.size() > MAX_REGIONS) regions.resize(MAX_REGIONS);
}

// Renders a color-coded BGR image where each accepted region is painted in a
// distinct color from a fixed palette. Draws a white dot at each centroid with
// an index number for quick visual reference during debugging.
cv::Mat drawRegions(const RegionMap& regionMap,
                    const std::vector<RegionInfo>& regions) {
    // Visually distinct BGR colors for up to MAX_REGIONS regions
    static const cv::Vec3b palette[] = {
        {100, 220, 255},  // yellow
        {100, 255, 100},  // green
        {255, 100, 100},  // blue
        {100, 100, 255},  // red
        {255, 100, 255},  // magenta
    };

    // Build a label -> palette index lookup to avoid per-pixel linear search
    int maxId = 0;
    for (const auto& r : regions) maxId = std::max(maxId, r.id);

    std::vector<int> lut(maxId + 1, -1);
    for (int i = 0; i < (int)regions.size(); i++) lut[regions[i].id] = i;

    cv::Mat output = cv::Mat::zeros(regionMap.size(), CV_8UC3);
    for (int r = 0; r < regionMap.rows; r++) {
        for (int c = 0; c < regionMap.cols; c++) {
            int label = regionMap.at<int>(r, c);
            if (label > 0 && label <= maxId && lut[label] >= 0)
                output.at<cv::Vec3b>(r, c) = palette[lut[label]];
        }
    }

    // Draw centroid dot and label index on each region for reference
    for (int i = 0; i < (int)regions.size(); i++) {
        cv::Point ct(regions[i].centroid.x, regions[i].centroid.y);
        cv::circle(output, ct, 5, {255, 255, 255}, -1);
        cv::putText(output, std::to_string(i + 1), ct + cv::Point(8, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2);
    }

    return output;
}
