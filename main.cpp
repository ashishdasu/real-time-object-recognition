/*
 * main.cpp
 * CS5330 - Project 3: Real-time 2D Object Recognition
 * Ashish Dasu
 *
 * Main loop. Opens a webcam or file (argv[1]), runs each pipeline stage,
 * and displays results. Press 'd' to cycle through pipeline views, 'q' to quit.
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "morphology.h"
#include "regions.h"

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

    cv::namedWindow("Output",  cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Debug",   cv::WINDOW_AUTOSIZE);

    // Cycles through what the debug window shows
    enum View { ORIGINAL, THRESHOLD, MORPHOLOGY, REGIONS, VIEW_COUNT };
    const char* viewNames[] = { "Original", "Threshold", "Morphology", "Regions" };
    View view = ORIGINAL;

    std::cout << "d: cycle view  q: quit\n";

    cv::Mat frame, thresholded, cleaned;
    RegionMap regionMap;
    std::vector<RegionInfo> regions;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        applyThreshold(frame, thresholded);
        applyMorphology(thresholded, cleaned);
        segmentRegions(cleaned, regionMap, regions);

        cv::imshow("Output", frame);

        // Debug window shows the selected pipeline stage
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
            default:
                cv::imshow("Debug", frame);
        }

        char key = (char)cv::waitKey(10);
        if (key == 'q') break;
        if (key == 'd') {
            view = (View)((view + 1) % VIEW_COUNT);
            std::cout << "View: " << viewNames[view] << "\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
