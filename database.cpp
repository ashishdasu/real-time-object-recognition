/*
 * database.cpp
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * CSV format: label,hu0,hu1,hu2,hu3,percentFilled,aspectRatio
 */

#include "database.h"
#include <fstream>
#include <sstream>
#include <cmath>

// Pull the 6 feature values out of a FeatureVec into a flat array
static void toArray(const FeatureVec &fv, double out[6]) {
    out[0] = fv.hu[0]; out[1] = fv.hu[1];
    out[2] = fv.hu[2]; out[3] = fv.hu[3];
    out[4] = fv.percentFilled;
    out[5] = fv.aspectRatio;
}

void computeStdevs(FeatureDB &db) {
    const int N = 6;
    db.stdevs.assign(N, 1.0);
    if (db.samples.size() < 2) return;

    double mean[6] = {};
    for (const auto &s : db.samples) {
        double v[6]; toArray(s, v);
        for (int i = 0; i < N; i++) mean[i] += v[i];
    }
    for (int i = 0; i < N; i++) mean[i] /= db.samples.size();

    double var[6] = {};
    for (const auto &s : db.samples) {
        double v[6]; toArray(s, v);
        for (int i = 0; i < N; i++) var[i] += (v[i] - mean[i]) * (v[i] - mean[i]);
    }
    for (int i = 0; i < N; i++) {
        double sd = std::sqrt(var[i] / db.samples.size());
        db.stdevs[i] = (sd > 1e-9) ? sd : 1.0;
    }
}

void addSample(FeatureDB &db, const std::string &label, const FeatureVec &fv) {
    db.labels.push_back(label);
    db.samples.push_back(fv);
    computeStdevs(db);
}

void saveDatabase(const std::string &path, const FeatureDB &db) {
    std::ofstream f(path);
    for (size_t i = 0; i < db.labels.size(); i++) {
        const FeatureVec &fv = db.samples[i];
        f << db.labels[i] << ","
          << fv.hu[0] << "," << fv.hu[1] << ","
          << fv.hu[2] << "," << fv.hu[3] << ","
          << fv.percentFilled << "," << fv.aspectRatio << "\n";
    }
}

bool loadDatabase(const std::string &path, FeatureDB &db) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    db.labels.clear();
    db.samples.clear();

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        FeatureVec fv{};

        std::getline(ss, tok, ','); db.labels.push_back(tok);
        for (int i = 0; i < 4; i++) {
            std::getline(ss, tok, ',');
            fv.hu[i] = std::stod(tok);
        }
        std::getline(ss, tok, ','); fv.percentFilled = std::stod(tok);
        std::getline(ss, tok, ','); fv.aspectRatio    = std::stod(tok);
        db.samples.push_back(fv);
    }

    computeStdevs(db);
    return true;
}
