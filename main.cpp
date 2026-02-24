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
#include <map>
#include <set>
#include <vector>
#include <string>
#include <iomanip>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "morphology.h"
#include "regions.h"
#include "features.h"
#include "database.h"
#include "classifier.h"
#include "embedding.h"

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
