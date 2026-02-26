/*
 * features.cpp
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Feature extraction using OpenCV's moments() function. The Hu moments
 * are invariant to translation, scale, and rotation. Percent filled and
 * aspect ratio add simple geometric discrimination on top of that.
 */

#include "features.h"
#include <cmath>

// Extract the binary mask for a single region from the full label map
static cv::Mat regionMask(const RegionMap &regionMap, int id) {
    cv::Mat mask;
    cv::compare(regionMap, id, mask, cv::CMP_EQ);
    return mask;
}

// Compute the oriented bounding box corners from centroid, angle, and axes
static void computeOBB(const cv::Point2d &centroid, double angle,
                        double halfMajor, double halfMinor,
                        cv::Point2f corners[4]) {
    double ca = std::cos(angle), sa = std::sin(angle);

    // Unit vectors along major and minor axes
    cv::Point2d major( ca,  sa);
    cv::Point2d minor(-sa,  ca);

    cv::Point2d c(centroid.x, centroid.y);
    cv::Point2d pts[4] = {
        c + halfMajor * major + halfMinor * minor,
        c + halfMajor * major - halfMinor * minor,
        c - halfMajor * major - halfMinor * minor,
        c - halfMajor * major + halfMinor * minor
    };
    for (int i = 0; i < 4; i++)
        corners[i] = { (float)pts[i].x, (float)pts[i].y };
}

// Compute 6 features per region: log-scaled Hu moments [0-3], percentFilled,
// aspectRatio. Also stores angle and OBB corners for the overlay display.
void computeFeatures(const cv::Mat &src,
                     const RegionMap &regionMap,
                     const std::vector<RegionInfo> &regions,
                     std::vector<FeatureVec> &fvecs) {
    fvecs.clear();

    for (const auto &ri : regions) {
        cv::Mat mask = regionMask(regionMap, ri.id);

        // Compute raw moments from the binary region mask
        cv::Moments m = cv::moments(mask, true);
        if (m.m00 < 1.0) continue;   // degenerate region

        FeatureVec fv;
        fv.regionId = ri.id;
        fv.centroid = { m.m10 / m.m00, m.m01 / m.m00 };

        // Orientation: angle of axis of least central moment.
        // Derived from the second-order central moments (mu20, mu11, mu02).
        // This gives the angle of the principal axis in radians.
        double num = 2.0 * m.mu11;
        double den = m.mu20 - m.mu02;
        fv.angle = 0.5 * std::atan2(num, den);

        // Eigenvalues of the covariance matrix give axis lengths
        double common = std::sqrt(4.0 * m.mu11 * m.mu11 +
                                  (m.mu20 - m.mu02) * (m.mu20 - m.mu02));
        double lambda1 = (m.mu20 + m.mu02 + common) / (2.0 * m.m00);
        double lambda2 = (m.mu20 + m.mu02 - common) / (2.0 * m.m00);
        lambda1 = std::max(lambda1, 1.0);
        lambda2 = std::max(lambda2, 1.0);

        double halfMajor = 2.0 * std::sqrt(lambda1);
        double halfMinor = 2.0 * std::sqrt(lambda2);
        fv.axisLength = halfMajor;

        // Aspect ratio: how elongated the region is
        fv.aspectRatio = halfMajor / halfMinor;

        // Oriented bounding box area for percent filled
        double obbArea = (2.0 * halfMajor) * (2.0 * halfMinor);
        fv.percentFilled = (obbArea > 1.0) ? m.m00 / obbArea : 0.0;

        computeOBB(fv.centroid, fv.angle, halfMajor, halfMinor, fv.obbCorners);

        // Hu moments — log-scaled to compress the dynamic range.
        // We use the first 4; higher-order ones are rarely stable.
        double huRaw[7];
        cv::HuMoments(m, huRaw);
        for (int i = 0; i < 4; i++) {
            double v = huRaw[i];
            fv.hu[i] = (v != 0.0) ? std::copysign(std::log10(std::abs(v)), v) : 0.0;
        }

        fvecs.push_back(fv);
    }
}

// Draw OBB, axis line, and fill/ar text on the image — used in the features
// debug view and batch output screenshots.
cv::Mat drawFeatureOverlay(const cv::Mat &src,
                           const std::vector<FeatureVec> &fvecs) {
    cv::Mat out = src.clone();

    for (const auto &fv : fvecs) {
        cv::Point ct((int)fv.centroid.x, (int)fv.centroid.y);

        // Primary axis line
        cv::Point axisEnd(ct.x + (int)(fv.axisLength * std::cos(fv.angle)),
                          ct.y + (int)(fv.axisLength * std::sin(fv.angle)));
        cv::line(out, ct, axisEnd, {0, 255, 0}, 2);
        cv::circle(out, ct, 5, {0, 255, 0}, -1);

        // Oriented bounding box
        for (int i = 0; i < 4; i++)
            cv::line(out, fv.obbCorners[i], fv.obbCorners[(i + 1) % 4],
                     {0, 200, 255}, 2);

        // Print a couple of feature values for live inspection
        std::string info = cv::format("fill=%.2f ar=%.2f",
                                      fv.percentFilled, fv.aspectRatio);
        cv::putText(out, info, ct + cv::Point(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
    }

    return out;
}
