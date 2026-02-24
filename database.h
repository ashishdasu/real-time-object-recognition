/*
 * database.h
 * CS5330 - Project 3
 * Ashish Dasu
 *
 * Reads and writes labeled feature vectors to a CSV file. Each row stores
 * a label and its 6-dimensional feature vector. Per-feature standard
 * deviations are recomputed on every load/add for use in scaled distance.
 */

#pragma once
#include "features.h"
#include <string>
#include <vector>

struct FeatureDB {
    std::vector<std::string> labels;
    std::vector<FeatureVec>  samples;
    std::vector<double>      stdevs;   // one per feature dimension, for scaling
};

void addSample(FeatureDB &db, const std::string &label, const FeatureVec &fv);
void saveDatabase(const std::string &path, const FeatureDB &db);
bool loadDatabase(const std::string &path, FeatureDB &db);
void computeStdevs(FeatureDB &db);
