#pragma once
#include "enums.hpp"

struct Dataset {
    size_t trainDataRows;
    size_t trainDataCols;
    std::vector<float> trainData;

    size_t trainLabelRows;
    size_t trainLabelCols;
    std::vector<float> trainLabels;

    size_t testDataRows;
    size_t testDataCols;
    std::vector<float> testData;

    size_t testLabelRows;
    size_t testLabelCols;
    std::vector<float> testLabels;

    Datasets type;
    std::string name;
    bool hasTestData;

    Dataset(Datasets type, std::string name) : type(type), name(name) {}
    Dataset() : type(Datasets::NONE) {}
};
