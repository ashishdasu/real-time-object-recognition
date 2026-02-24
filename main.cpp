/*
 * main.cpp
 * CS5330 - Project 3: Real-time 2D Object Recognition
 * Ashish Dasu
 *
 * Video capture loop. Opens a webcam or image/video file passed as
 * argv[1], displays the feed in a window, and exits on 'q'.
 */

#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    cv::VideoCapture cap;

    // Accept a device index (e.g. 0) or a file/video path as argv[1]
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

    cv::namedWindow("Object Recognition", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("Object Recognition", frame);

        if ((char)cv::waitKey(10) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
