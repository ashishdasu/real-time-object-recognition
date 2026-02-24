/*
 * main.cpp
 * CS5330 - Project 3: Real-time 2D Object Recognition
 * Ashish Dasu
 *
 * Main loop. Opens a webcam or file (argv[1]), runs the full pipeline,
 * and handles keyboard input for training and classification.
 *
 * Keys:
 *   d  - cycle debug view (Original → Threshold → Morphology → Regions → Features)
 *   n  - label current object and add to training database
 *   l  - load database from disk
 *   c  - toggle live classification overlay
 *   q  - quit
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "morphology.h"
#include "regions.h"
#include "features.h"
#include "database.h"
#include "classifier.h"

int main(int argc, char* argv[]) {
    cv::VideoCapture cap;

    if (argc > 1) {
        try {
            cap.open(std::stoi(argv[1]));
        } catch (...) {
            cap.open(argv[1]);
        }
    } else {
        cap.open(0);
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: could not open video source\n";
        return -1;
    }

    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Debug",  cv::WINDOW_AUTOSIZE);

    enum View { ORIGINAL, THRESHOLD, MORPHOLOGY, REGIONS, FEATURES, VIEW_COUNT };
    const char* viewNames[] = { "Original", "Threshold", "Morphology", "Regions", "Features" };
    View view = ORIGINAL;

    FeatureDB db;
    loadDatabase("data/feature_db.csv", db);
    std::cout << "DB loaded: " << db.labels.size() << " entries\n";

    bool classifyMode = false;

    std::cout << "d:cycle-view  n:label  l:reload-db  c:toggle-classify  a:auto-learn  q:quit\n";

    cv::Mat frame, thresholded, cleaned;
    RegionMap regionMap;
    std::vector<RegionInfo> regions;
    std::vector<FeatureVec> fvecs;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        applyThreshold(frame, thresholded);
        applyMorphology(thresholded, cleaned);
        segmentRegions(cleaned, regionMap, regions);
        computeFeatures(frame, regionMap, regions, fvecs);

        // Output window: feature overlay, plus classification if enabled
        cv::Mat output = drawFeatureOverlay(frame, fvecs);
        if (classifyMode && !db.samples.empty())
            classifyAndLabel(output, fvecs, db);
        cv::imshow("Output", output);

        // Debug window: selected pipeline stage
        switch (view) {
            case THRESHOLD: {
                cv::Mat vis;
                cv::cvtColor(thresholded, vis, cv::COLOR_GRAY2BGR);
                cv::putText(vis, viewNames[view], {10, 25},
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
                cv::imshow("Debug", vis);
                break;
            }
            case MORPHOLOGY: {
                cv::Mat vis;
                cv::cvtColor(cleaned, vis, cv::COLOR_GRAY2BGR);
                cv::putText(vis, viewNames[view], {10, 25},
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
                cv::imshow("Debug", vis);
                break;
            }
            case REGIONS:
                cv::imshow("Debug", drawRegions(regionMap, regions));
                break;
            case FEATURES:
                cv::imshow("Debug", drawFeatureOverlay(frame, fvecs));
                break;
            default:
                cv::imshow("Debug", frame);
        }

        char key = (char)cv::waitKey(10);
        if (key == 'q') break;

        if (key == 'd') {
            view = (View)((view + 1) % VIEW_COUNT);
            std::cout << "View: " << viewNames[view] << "\n";
        }

        if (key == 'c') {
            classifyMode = !classifyMode;
            std::cout << "Classify: " << (classifyMode ? "ON" : "OFF") << "\n";
        }

        if (key == 'l') {
            loadDatabase("data/feature_db.csv", db);
            std::cout << "DB reloaded: " << db.labels.size() << " entries\n";
        }

        if (key == 'n') {
            if (fvecs.empty()) {
                std::cout << "No object detected\n";
            } else {
                std::cout << "Label: ";
                std::string label;
                std::cin >> label;
                addSample(db, label, fvecs[0]);
                saveDatabase("data/feature_db.csv", db);
                std::cout << "Saved '" << label << "' (" << db.labels.size() << " total)\n";
            }
        }

        // Auto-learn: only fires when classify mode is on and the current
        // object is unrecognized, letting you build the DB on the fly
        if (key == 'a' && classifyMode) {
            if (fvecs.empty()) {
                std::cout << "No object detected\n";
            } else {
                std::string result = classifyFeatureKNN(fvecs[0], db);
                if (result == "Unknown") {
                    std::cout << "Unknown object — enter label to learn: ";
                    std::string label;
                    std::cin >> label;
                    addSample(db, label, fvecs[0]);
                    saveDatabase("data/feature_db.csv", db);
                    std::cout << "Learned '" << label << "' (" << db.labels.size() << " total)\n";
                } else {
                    std::cout << "Object already recognized as '" << result << "'\n";
                }
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
