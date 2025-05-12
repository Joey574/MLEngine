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
    std::vector<std::string> args;
    bool hasTestData;

    Dataset(Datasets type, std::string name) : type(type), name(name) {}
    Dataset() : type(Datasets::NONE) {}

    void Shuffle() {
        std::random_device rd;
        std::mt19937 rng(rd());

        // swap blocks in place
        for (size_t i = trainDataRows-1; i > 0; i--) {
            std::uniform_int_distribution<size_t> dist(0, i);
            size_t j = dist(rng);

            if (i != j) {
                auto block_id = trainData.begin() + (i*trainDataCols);
                auto block_jd = trainData.begin() + (j*trainDataCols);

                auto block_il = trainLabels.begin() + (i*trainLabelCols);
                auto block_jl = trainLabels.begin() + (j*trainLabelCols);

                std::swap_ranges(block_id, block_id+trainDataCols, block_jd);
                std::swap_ranges(block_il, block_il+trainLabelCols, block_jl);
            }
        }
    }
};
