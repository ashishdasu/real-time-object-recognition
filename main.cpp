/*
 * main.cpp
 * CS5330 - Project 3: Real-time 2D Object Recognition
 * Ashish Dasu
 *
 * Main loop. Opens a webcam or file (argv[1]), runs the full pipeline,
 * and handles keyboard input for training, classification, and evaluation.
 *
 * Keys:
 *   d  - cycle debug view (Original → Threshold → Morphology → Regions → Features)
 *   n  - label current object and add to training database
 *   l  - reload database from disk
 *   c  - toggle live classification overlay
 *   a  - auto-learn: label an unknown object on the spot (classify mode only)
 *   e  - record evaluation sample (prompts for true label, logs prediction)
 *   p  - print confusion matrix to terminal
 *   s  - save screenshot of both windows to disk
 *   q  - quit
 */

#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
#include "threshold.h"
#include "morphology.h"
#include "regions.h"
#include "features.h"
#include "database.h"
#include "classifier.h"
#include "embedding.h"

// Live capture mode: n=set label, SPACE=save frame, q=quit
static void runCapture(int camIndex) {
    cv::VideoCapture cap(camIndex);
    if (!cap.isOpened()) { std::cerr << "Cannot open camera " << camIndex << "\n"; return; }

    cv::namedWindow("Capture", cv::WINDOW_AUTOSIZE);
    std::string label = "unlabeled";
    int count = 1;

    std::cout << "Capture mode: click window, n=new label, SPACE=save, q=quit\n";

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;
        cv::resize(frame, frame, cv::Size(640, 480));

        cv::Mat display = frame.clone();
        cv::putText(display, "Label: " + label + " [" + std::to_string(count) + "]",
                    {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,0}, 2);
        cv::putText(display, "n=label  SPACE=save  q=quit",
                    {10, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {200,200,200}, 1);
        cv::imshow("Capture", display);

        char key = (char)cv::waitKey(30);
        if (key == 'q') break;

        if (key == ' ') {
            std::string path = "data/" + label + "_" + std::to_string(count) + ".jpg";
            cv::imwrite(path, frame);
            std::cout << "Saved: " << path << "\n";
            count++;
        }

        if (key == 'n') {
            std::cout << "Label: ";
            std::cin >> label;
            count = 1;
            for (const auto &e : fs::directory_iterator("data")) {
                std::string s = e.path().stem().string();
                if (s.rfind(label + "_", 0) == 0) count++;
            }
            std::cout << "Ready — next shot will be " << label << "_" << count << ".jpg\n";
        }
    }
    cv::destroyAllWindows();
}

// Extract features from every image in data/ and save to feature_db.csv
static void trainFeaturesFromData() {
    FeatureDB db;

    for (const auto &entry : fs::directory_iterator("data")) {
        if (entry.path().extension() != ".jpg" &&
            entry.path().extension() != ".png") continue;

        std::string stem = entry.path().stem().string();
        // Derive label by stripping trailing _N suffix
        std::string label = stem;
        auto underscore = stem.rfind('_');
        if (underscore != std::string::npos) {
            std::string suffix = stem.substr(underscore + 1);
            bool allDigits = !suffix.empty() &&
                std::all_of(suffix.begin(), suffix.end(), ::isdigit);
            if (allDigits) label = stem.substr(0, underscore);
        }

        cv::Mat frame = cv::imread(entry.path().string());
        if (frame.empty()) continue;
        cv::resize(frame, frame, cv::Size(640, 480));

        cv::Mat thresholded, cleaned;
        RegionMap regionMap;
        std::vector<RegionInfo> regions;
        std::vector<FeatureVec> fvecs;

        applyThreshold(frame, thresholded);
        applyMorphology(thresholded, cleaned);
        segmentRegions(cleaned, regionMap, regions);
        computeFeatures(frame, regionMap, regions, fvecs);

        if (fvecs.empty()) {
            std::cout << "No region: " << stem << "\n";
            continue;
        }

        addSample(db, label, fvecs[0]);
        std::cout << "Added: " << label << " (" << stem << ")\n";
    }

    saveDatabase("data/feature_db.csv", db);
    std::cout << "Feature DB saved: " << db.labels.size() << " entries\n";
}

// Process every .jpg in data/ and save all 5 pipeline stage images to screenshots/
static void runBatch() {
    fs::create_directories("screenshots");

    for (const auto &entry : fs::directory_iterator("data")) {
        if (entry.path().extension() != ".jpg" &&
            entry.path().extension() != ".png") continue;

        cv::Mat frame = cv::imread(entry.path().string());
        if (frame.empty()) continue;

        cv::resize(frame, frame, cv::Size(640, 480));

        cv::Mat thresholded, cleaned;
        RegionMap regionMap;
        std::vector<RegionInfo> regions;
        std::vector<FeatureVec> fvecs;

        applyThreshold(frame, thresholded);
        applyMorphology(thresholded, cleaned);
        segmentRegions(cleaned, regionMap, regions);
        computeFeatures(frame, regionMap, regions, fvecs);

        std::string stem = entry.path().stem().string();

        cv::Mat threshBGR, morphBGR;
        cv::cvtColor(thresholded, threshBGR, cv::COLOR_GRAY2BGR);
        cv::cvtColor(cleaned,     morphBGR,  cv::COLOR_GRAY2BGR);

        cv::imwrite("screenshots/" + stem + "_1_original.png",   frame);
        cv::imwrite("screenshots/" + stem + "_2_threshold.png",  threshBGR);
        cv::imwrite("screenshots/" + stem + "_3_morphology.png", morphBGR);
        cv::imwrite("screenshots/" + stem + "_4_regions.png",    drawRegions(regionMap, regions));
        cv::imwrite("screenshots/" + stem + "_5_features.png",   drawFeatureOverlay(frame, fvecs));

        std::cout << "Processed: " << stem << "\n";
    }
    std::cout << "Batch done. Images saved to screenshots/\n";
}

static void printConfusionMatrix(const std::vector<std::pair<std::string,std::string>> &results);

// Build embedding DB from test/*_1.jpg — one shot per object, same domain as test images.
static void trainEmbeddingsFromTest(EmbeddingDB &embDB, cv::dnn::Net &net) {
    if (!fs::exists("test")) { std::cerr << "No test/ directory\n"; return; }

    embDB.labels.clear();
    embDB.embeddings.clear();

    for (const auto &entry : fs::directory_iterator("test")) {
        std::string stem = entry.path().stem().string();
        // Only process _1 images
        if (stem.size() < 2 || stem.substr(stem.size() - 2) != "_1") continue;
        if (entry.path().extension() != ".jpg" &&
            entry.path().extension() != ".png") continue;

        std::string label = stem.substr(0, stem.size() - 2);
        cv::Mat frame = cv::imread(entry.path().string());
        if (frame.empty()) continue;
        cv::resize(frame, frame, cv::Size(640, 480));

        cv::Mat thresholded, cleaned;
        RegionMap regionMap;
        std::vector<RegionInfo> regions;
        std::vector<FeatureVec> fvecs;

        applyThreshold(frame, thresholded);
        applyMorphology(thresholded, cleaned);
        segmentRegions(cleaned, regionMap, regions);
        computeFeatures(frame, regionMap, regions, fvecs);

        if (fvecs.empty()) { std::cout << "No region: " << stem << "\n"; continue; }

        cv::Mat emb;
        if (getRegionEmbedding(frame, fvecs[0], regionMap, net, emb)) {
            addEmbeddingSample(embDB, label, emb);
            std::cout << "Embedded: " << label << "\n";
        } else {
            std::cout << "Embedding failed: " << stem << "\n";
        }
    }
    saveEmbeddingDB("data/embedding_db.yml", embDB);
    std::cout << "Embedding DB rebuilt: " << embDB.labels.size() << " entries\n";
}

// Draw text rotated 45 degrees into img, with the bottom of the text
// landing at anchor (center-x of a column cell, just above the grid).
static void drawRotatedLabel(cv::Mat &img, const std::string &text, cv::Point anchor) {
    const double scale = 0.42;
    const int thickness = 1;
    int baseline = 0;
    cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);

    // Render text onto a generously sized canvas so rotation doesn't clip it
    int pad = 6;
    int cw = sz.width + 2 * pad;
    int ch = sz.height + baseline + 2 * pad;
    // After 45° rotation the diagonal is sqrt(cw²+ch²) — use that as canvas side
    int side = (int)std::ceil(std::sqrt((double)cw*cw + (double)ch*ch)) + 4;
    cv::Mat canvas(side, side, CV_8UC3, cv::Scalar(255,255,255));
    // Place text in the centre of the canvas
    cv::Point textOrg((side - sz.width) / 2, (side + sz.height) / 2 - baseline);
    cv::putText(canvas, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, scale, {0,0,0}, thickness);

    // Rotate 45° clockwise around canvas centre
    cv::Point2f center(side / 2.0f, side / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, -45, 1.0);
    cv::Mat rotated;
    cv::warpAffine(canvas, rotated, rot, cv::Size(side, side),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255,255,255));

    // Stamp: anchor is the bottom-centre of where the text should land
    int x0 = anchor.x - side / 2;
    int y0 = anchor.y - side;
    for (int r = 0; r < rotated.rows; r++) {
        for (int c = 0; c < rotated.cols; c++) {
            int yr = y0 + r, xc = x0 + c;
            if (yr < 0 || yr >= img.rows || xc < 0 || xc >= img.cols) continue;
            cv::Vec3b px = rotated.at<cv::Vec3b>(r, c);
            if (px[0] < 230)
                img.at<cv::Vec3b>(yr, xc) = px;
        }
    }
}

// Save a confusion matrix as a PNG image
static void saveConfusionMatrixPNG(
        const std::vector<std::pair<std::string,std::string>> &results,
        const std::string &path) {

    if (results.empty()) return;

    std::set<std::string> classSet;
    for (const auto &r : results) { classSet.insert(r.first); classSet.insert(r.second); }
    std::vector<std::string> classes(classSet.begin(), classSet.end());
    int N = classes.size();

    std::map<std::string,int> idx;
    for (int i = 0; i < N; i++) idx[classes[i]] = i;

    std::vector<std::vector<int>> mat(N, std::vector<int>(N, 0));
    int correct = 0, total = (int)results.size();
    for (const auto &r : results) mat[idx[r.first]][idx[r.second]]++;
    for (int i = 0; i < N; i++) correct += mat[i][i];

    // Find max off-diagonal count for color scaling
    int maxErr = 1;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j && mat[i][j] > maxErr) maxErr = mat[i][j];
    int maxDiag = 1;
    for (int i = 0; i < N; i++)
        if (mat[i][i] > maxDiag) maxDiag = mat[i][i];

    const int cell = 72;
    const int leftMargin = 115;  // row labels
    const int topMargin  = 160;  // rotated column labels + padding
    const int bottomPad  = 45;
    const int rightPad   = 20;

    int W = leftMargin + N * cell + rightPad;
    int H = topMargin  + N * cell + bottomPad;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(252,252,252));

    // "Predicted" axis label centred above the columns
    {
        int baseline2 = 0;
        cv::Size lsz = cv::getTextSize("Predicted Label", cv::FONT_HERSHEY_SIMPLEX, 0.52, 1, &baseline2);
        cv::putText(img, "Predicted Label",
                    cv::Point(leftMargin + (N*cell - lsz.width)/2, 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.52, {80,80,80}, 1);
    }
    // "True Label" left of the rows
    {
        int baseline2 = 0;
        cv::Size lsz = cv::getTextSize("True Label", cv::FONT_HERSHEY_SIMPLEX, 0.52, 1, &baseline2);
        cv::putText(img, "True Label",
                    cv::Point(4, topMargin + (N*cell + lsz.width)/2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.52, {80,80,80}, 1);
    }

    // Rotated column headers
    for (int j = 0; j < N; j++) {
        cv::Point anchor(leftMargin + j*cell + cell/2 - 4, topMargin - 8);
        drawRotatedLabel(img, classes[j], anchor);
    }

    // Row headers
    for (int i = 0; i < N; i++) {
        int baseline = 0;
        cv::Size sz = cv::getTextSize(classes[i], cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &baseline);
        cv::putText(img, classes[i],
                    cv::Point(leftMargin - sz.width - 6, topMargin + i*cell + cell/2 + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, {0,0,0}, 1);
    }

    // Cells
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cv::Rect cr(leftMargin + j*cell, topMargin + i*cell, cell, cell);
            cv::Scalar bg;
            if (i == j && mat[i][i] > 0) {
                // Green intensity proportional to count
                float t = (maxDiag > 0) ? (float)mat[i][i] / maxDiag : 0;
                bg = cv::Scalar(100 + (int)(40*(1-t)), 180 + (int)(55*t), 100 + (int)(40*(1-t)));
            } else if (i == j) {
                bg = cv::Scalar(245, 245, 245);  // zero on diagonal — neutral
            } else if (mat[i][j] > 0) {
                // Red intensity proportional to count
                float t = (float)mat[i][j] / maxErr;
                bg = cv::Scalar(180 - (int)(60*t), 180 - (int)(80*t), 230);
            } else {
                bg = cv::Scalar(245, 245, 245);
            }
            cv::rectangle(img, cr, bg, cv::FILLED);
            cv::rectangle(img, cr, cv::Scalar(200,200,200), 1);

            if (mat[i][j] > 0) {
                std::string val = std::to_string(mat[i][j]);
                int baseline2 = 0;
                cv::Size vsz = cv::getTextSize(val, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline2);
                cv::putText(img, val,
                            cv::Point(cr.x + cell/2 - vsz.width/2,
                                      cr.y + cell/2 + vsz.height/2),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, {20,20,20}, 1);
            }
        }
    }

    // Accuracy line
    double pct = 100.0 * correct / total;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Accuracy: %d/%d  (%.1f%%)", correct, total, pct);
    cv::putText(img, buf, cv::Point(leftMargin, H - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, {40,40,40}, 1);

    cv::imwrite(path, img);
}

// For each _1 image in data/, save a KNN-classified result image
static void runClassifyKNN(FeatureDB &db) {
    fs::create_directories("screenshots");

    for (const auto &entry : fs::directory_iterator("data")) {
        if (entry.path().extension() != ".jpg" &&
            entry.path().extension() != ".png") continue;

        std::string stem = entry.path().stem().string();
        // Only use _1 images (one representative shot per object)
        if (stem.size() < 2 || stem.substr(stem.size() - 2) != "_1") continue;

        std::string label_stem = stem.substr(0, stem.size() - 2);

        cv::Mat frame = cv::imread(entry.path().string());
        if (frame.empty()) continue;
        cv::resize(frame, frame, cv::Size(640, 480));

        cv::Mat thresholded, cleaned;
        RegionMap regionMap;
        std::vector<RegionInfo> regions;
        std::vector<FeatureVec> fvecs;

        applyThreshold(frame, thresholded);
        applyMorphology(thresholded, cleaned);
        segmentRegions(cleaned, regionMap, regions);
        computeFeatures(frame, regionMap, regions, fvecs);

        cv::Mat result = drawFeatureOverlay(frame, fvecs);

        if (!fvecs.empty()) {
            std::string label = classifyFeatureKNN(fvecs[0], db, 3, 1e9);
            cv::Point pt((int)fvecs[0].centroid.x, (int)fvecs[0].centroid.y - 30);
            cv::putText(result, label, pt, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,0}, 4);
            cv::putText(result, label, pt, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
        }

        cv::imwrite("screenshots/" + label_stem + "_knn.png", result);
        std::cout << "KNN classified: " << label_stem << "\n";
    }
}

// For each _2 image in test/, save an embedding-classified result image
static void runClassifyEmbeddings(EmbeddingDB &embDB, cv::dnn::Net &net) {
    fs::create_directories("screenshots");

    std::string imgDir = fs::exists("test") ? "test" : "data";
    for (const auto &entry : fs::directory_iterator(imgDir)) {
        if (entry.path().extension() != ".jpg" &&
            entry.path().extension() != ".png") continue;

        std::string stem = entry.path().stem().string();
        if (stem.size() < 2 || stem.substr(stem.size() - 2) != "_2") continue;

        std::string label_stem = stem.substr(0, stem.size() - 2);

        cv::Mat frame = cv::imread(entry.path().string());
        if (frame.empty()) continue;
        cv::resize(frame, frame, cv::Size(640, 480));

        cv::Mat thresholded, cleaned;
        RegionMap regionMap;
        std::vector<RegionInfo> regions;
        std::vector<FeatureVec> fvecs;

        applyThreshold(frame, thresholded);
        applyMorphology(thresholded, cleaned);
        segmentRegions(cleaned, regionMap, regions);
        computeFeatures(frame, regionMap, regions, fvecs);

        cv::Mat result = drawFeatureOverlay(frame, fvecs);

        if (!fvecs.empty()) {
            cv::Mat emb;
            std::string label = "Unknown";
            if (getRegionEmbedding(frame, fvecs[0], regionMap, net, emb))
                label = classifyEmbedding(emb, embDB);
            cv::Point pt((int)fvecs[0].centroid.x, (int)fvecs[0].centroid.y - 30);
            cv::putText(result, label, pt, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,0}, 4);
            cv::putText(result, label, pt, cv::FONT_HERSHEY_SIMPLEX, 1.0, {255,128,0}, 2);
        }

        cv::imwrite("screenshots/" + label_stem + "_emb.png", result);
        std::cout << "Embedding classified: " << label_stem << "\n";
    }
}

// Run both classifiers on every image in data/ using the filename stem as the
// true label. Prints both confusion matrices and saves them as PNGs.
static void runEvaluate(FeatureDB &db, EmbeddingDB &embDB,
                        cv::dnn::Net &net, bool netReady) {
    std::vector<std::pair<std::string,std::string>> knnResults, embResults;

    // Prefer test/ for evaluation (multiple images per object).
    // Fall back to data/ if test/ doesn't exist.
    std::string evalDir = fs::exists("test") ? "test" : "data";
    std::cout << "Evaluating from: " << evalDir << "/\n";

    for (const auto &entry : fs::directory_iterator(evalDir)) {
        if (entry.path().extension() != ".jpg" &&
            entry.path().extension() != ".png") continue;

        // Strip trailing _1, _2 etc. so remote_1.jpg → true label "remote"
        std::string stem = entry.path().stem().string();
        std::string trueLabel = stem;
        auto underscore = stem.rfind('_');
        if (underscore != std::string::npos) {
            std::string suffix = stem.substr(underscore + 1);
            bool allDigits = !suffix.empty() &&
                std::all_of(suffix.begin(), suffix.end(), ::isdigit);
            if (allDigits) {
                trueLabel = stem.substr(0, underscore);
            }
        }
        cv::Mat frame = cv::imread(entry.path().string());
        if (frame.empty()) continue;
        cv::resize(frame, frame, cv::Size(640, 480));

        cv::Mat thresholded, cleaned;
        RegionMap regionMap;
        std::vector<RegionInfo> regions;
        std::vector<FeatureVec> fvecs;

        applyThreshold(frame, thresholded);
        applyMorphology(thresholded, cleaned);
        segmentRegions(cleaned, regionMap, regions);
        computeFeatures(frame, regionMap, regions, fvecs);

        if (fvecs.empty()) {
            std::cout << "No region detected: " << trueLabel << "\n";
            continue;
        }

        // Use a high threshold so evaluation always returns the nearest class
        // rather than Unknown — we want to see what it misclassifies as.
        std::string knnPred = classifyFeatureKNN(fvecs[0], db, 3, 1e9);
        knnResults.push_back({trueLabel, knnPred});
        std::cout << "[KNN] " << trueLabel << " → " << knnPred << "\n";

        // Skip _1 images for embedding evaluation — they were used for training
        bool isTrainImage = (evalDir == "test" && stem.size() >= 2 &&
                             stem.substr(stem.size() - 2) == "_1");
        if (netReady && !embDB.embeddings.empty() && !isTrainImage) {
            cv::Mat emb;
            std::string embPred = "Unknown";
            if (getRegionEmbedding(frame, fvecs[0], regionMap, net, emb))
                embPred = classifyEmbedding(emb, embDB);
            embResults.push_back({trueLabel, embPred});
            std::cout << "[EMB] " << trueLabel << " → " << embPred << "\n";
        }
    }

    fs::create_directories("screenshots");

    // Write raw results to CSV so plot_confusion.py can render them
    auto writeCSV = [](const std::string &path,
                       const std::vector<std::pair<std::string,std::string>> &res) {
        std::ofstream f(path);
        f << "true,predicted\n";
        for (const auto &r : res) f << r.first << "," << r.second << "\n";
    };
    writeCSV("data/results_knn.csv", knnResults);

    printConfusionMatrix(knnResults);
    saveConfusionMatrixPNG(knnResults, "screenshots/confusion_knn.png");

    if (!embResults.empty()) {
        std::cout << "\n--- Embedding Classifier ---\n";
        writeCSV("data/results_emb.csv", embResults);
        printConfusionMatrix(embResults);
        saveConfusionMatrixPNG(embResults, "screenshots/confusion_embedding.png");
    }
    std::cout << "Confusion matrix PNGs saved to screenshots/\n";
}

// Print a confusion matrix from (true, predicted) pairs
static void printConfusionMatrix(
        const std::vector<std::pair<std::string,std::string>> &results) {

    if (results.empty()) { std::cout << "No evaluation data.\n"; return; }

    // Collect the ordered set of class names
    std::set<std::string> classSet;
    for (const auto &r : results) {
        classSet.insert(r.first);
        classSet.insert(r.second);
    }
    std::vector<std::string> classes(classSet.begin(), classSet.end());
    int N = classes.size();

    // Map class name → index
    std::map<std::string,int> idx;
    for (int i = 0; i < N; i++) idx[classes[i]] = i;

    // Build NxN matrix  [true][predicted]
    std::vector<std::vector<int>> mat(N, std::vector<int>(N, 0));
    for (const auto &r : results)
        mat[idx[r.first]][idx[r.second]]++;

    // Print header
    const int W = 12;
    std::cout << "\n--- Confusion Matrix (rows=true, cols=predicted) ---\n";
    std::cout << std::setw(W) << "";
    for (const auto &c : classes) std::cout << std::setw(W) << c;
    std::cout << "\n";

    int correct = 0, total = results.size();
    for (int i = 0; i < N; i++) {
        std::cout << std::setw(W) << classes[i];
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(W) << mat[i][j];
            if (i == j) correct += mat[i][j];
        }
        std::cout << "\n";
    }
    std::cout << "\nAccuracy: " << correct << "/" << total
              << " (" << std::fixed << std::setprecision(1)
              << 100.0 * correct / total << "%)\n\n";
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--capture") {
        int cam = (argc > 2) ? std::stoi(argv[2]) : 1;
        runCapture(cam);
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "--train-features") {
        trainFeaturesFromData();
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "--batch") {
        runBatch();
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "--classify-knn") {
        FeatureDB db;
        loadDatabase("data/feature_db.csv", db);
        runClassifyKNN(db);
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "--classify-embeddings") {
        cv::dnn::Net net;
        bool netReady = loadNetwork("resnet18-v2-7.onnx", net);
        if (!netReady) { std::cerr << "Model not loaded\n"; return -1; }
        EmbeddingDB embDB;
        loadEmbeddingDB("data/embedding_db.yml", embDB);
        runClassifyEmbeddings(embDB, net);
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "--train-embeddings") {
        cv::dnn::Net net;
        bool netReady = loadNetwork("resnet18-v2-7.onnx", net);
        if (!netReady) { std::cerr << "Model not loaded\n"; return -1; }
        EmbeddingDB embDB;
        trainEmbeddingsFromTest(embDB, net);
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "--evaluate") {
        FeatureDB db;
        loadDatabase("data/feature_db.csv", db);
        cv::dnn::Net net;
        bool netReady = loadNetwork("resnet18-v2-7.onnx", net);
        EmbeddingDB embDB;
        loadEmbeddingDB("data/embedding_db.yml", embDB);
        runEvaluate(db, embDB, net, netReady);
        return 0;
    }

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

    // Load ResNet18 for embedding-based classification
    cv::dnn::Net net;
    bool netReady = loadNetwork("resnet18-v2-7.onnx", net);
    if (!netReady) std::cout << "Warning: could not load ResNet18 model\n";

    EmbeddingDB embDB;
    loadEmbeddingDB("data/embedding_db.yml", embDB);
    std::cout << "Embedding DB loaded: " << embDB.labels.size() << " entries\n";

    bool classifyMode  = false;
    bool embedMode     = false;   // use embedding classifier instead of hand features

    // Evaluation data: (true label, predicted label)
    std::vector<std::pair<std::string,std::string>> evalResults;

    std::cout << "d:view  n:label  l:reload  c:classify  a:auto-learn\n"
              << "b:save-embedding  m:embed-classify  e:eval  p:matrix  s:screenshot  q:quit\n";

    cv::Mat frame, lastFrame, thresholded, cleaned;
    RegionMap regionMap;
    std::vector<RegionInfo> regions;
    std::vector<FeatureVec> fvecs;

    while (true) {
        cap >> frame;
        // For static images, freeze on the last valid frame instead of exiting
        if (frame.empty()) {
            if (lastFrame.empty()) break;
            frame = lastFrame.clone();
        } else {
            cv::resize(frame, frame, cv::Size(640, 480));
            lastFrame = frame.clone();
        }

        applyThreshold(frame, thresholded);
        applyMorphology(thresholded, cleaned);
        segmentRegions(cleaned, regionMap, regions);
        computeFeatures(frame, regionMap, regions, fvecs);

        cv::Mat output = drawFeatureOverlay(frame, fvecs);
        if (embedMode && netReady && !embDB.embeddings.empty() && !fvecs.empty()) {
            cv::Mat emb;
            if (getRegionEmbedding(frame, fvecs[0], regionMap, net, emb)) {
                std::string label = classifyEmbedding(emb, embDB);
                cv::Point pt((int)fvecs[0].centroid.x, (int)fvecs[0].centroid.y - 20);
                cv::putText(output, "[E] " + label, pt,
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 0}, 4);
                cv::putText(output, "[E] " + label, pt,
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 128, 0}, 2);
            }
        } else if (classifyMode && !db.samples.empty()) {
            classifyAndLabel(output, fvecs, db);
        }
        cv::imshow("Output", output);

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

        if (key == 'a' && classifyMode) {
            if (fvecs.empty()) {
                std::cout << "No object detected\n";
            } else {
                std::string result = classifyFeatureKNN(fvecs[0], db);
                if (result == "Unknown") {
                    std::cout << "Unknown object — enter label to learn: ";
                    std::string label;
                    std::cin >> label;
                    // Add 3x so the new class wins k=3 majority vote immediately
                    addSample(db, label, fvecs[0]);
                    addSample(db, label, fvecs[0]);
                    addSample(db, label, fvecs[0]);
                    saveDatabase("data/feature_db.csv", db);
                    std::cout << "Learned '" << label << "' (" << db.labels.size() << " total)\n";
                } else {
                    std::cout << "Already recognized as '" << result << "'\n";
                }
            }
        }

        if (key == 'e') {
            if (fvecs.empty()) {
                std::cout << "No object detected\n";
            } else {
                std::string predicted;
                if (embedMode && netReady && !embDB.embeddings.empty()) {
                    cv::Mat emb;
                    if (getRegionEmbedding(frame, fvecs[0], regionMap, net, emb))
                        predicted = classifyEmbedding(emb, embDB);
                    else
                        predicted = "Unknown";
                } else {
                    predicted = classifyFeatureKNN(fvecs[0], db);
                }
                std::cout << "Predicted: " << predicted << "  |  True label: ";
                std::string trueLabel;
                std::cin >> trueLabel;
                evalResults.push_back({ trueLabel, predicted });
                std::cout << "Recorded (" << evalResults.size() << " samples)\n";
            }
        }

        if (key == 'p') {
            printConfusionMatrix(evalResults);
        }

        if (key == 's') {
            // Timestamp-based filename so successive saves don't overwrite
            std::time_t t = std::time(nullptr);
            std::string ts = std::to_string(t);
            cv::imwrite("screenshot_output_" + ts + ".png", output);
            // Also save whatever the debug window is currently showing
            cv::Mat debugFrame;
            switch (view) {
                case THRESHOLD: cv::cvtColor(thresholded, debugFrame, cv::COLOR_GRAY2BGR); break;
                case MORPHOLOGY: cv::cvtColor(cleaned, debugFrame, cv::COLOR_GRAY2BGR); break;
                case REGIONS: debugFrame = drawRegions(regionMap, regions); break;
                case FEATURES: debugFrame = drawFeatureOverlay(frame, fvecs); break;
                default: debugFrame = frame.clone();
            }
            cv::imwrite("screenshot_debug_" + ts + ".png", debugFrame);
            std::cout << "Saved screenshots with timestamp " << ts << "\n";
        }

        if (key == 'm') {
            if (!netReady) { std::cout << "Model not loaded\n"; }
            else {
                embedMode = !embedMode;
                std::cout << "Embed classify: " << (embedMode ? "ON" : "OFF") << "\n";
            }
        }

        // Save an embedding for the current object to the embedding DB
        if (key == 'b' && netReady) {
            if (fvecs.empty()) {
                std::cout << "No object detected\n";
            } else {
                cv::Mat emb;
                if (getRegionEmbedding(frame, fvecs[0], regionMap, net, emb)) {
                    std::cout << "Label for embedding: ";
                    std::string label;
                    std::cin >> label;
                    addEmbeddingSample(embDB, label, emb);
                    saveEmbeddingDB("data/embedding_db.yml", embDB);
                    std::cout << "Embedding saved (" << embDB.labels.size() << " total)\n";
                } else {
                    std::cout << "Could not extract embedding\n";
                }
            }
        }
    }

    // Print matrix automatically on exit if data was collected
    if (!evalResults.empty()) printConfusionMatrix(evalResults);

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
