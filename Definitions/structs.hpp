#pragma once
#include "enums.hpp"

struct Dataset {
    size_t rows;
    size_t cols;
    float* data;

    Datasets type;

    Dataset(Datasets type) : type(type) {}
    Dataset() : type(Datasets::NONE) {}
};

struct DatasetMeta {
    std::string name;
    bool exists;
};