#pragma once
#include "enums.hpp"

struct Dataset {
    size_t rows;
    size_t cols;
    float* data;

    Datasets type;
    std::string name;

    Dataset(Datasets type, std::string name) : type(type), name(name) {}
    Dataset() : type(Datasets::NONE) {}
};

struct DatasetMeta {
    std::string name;
    bool exists;
};