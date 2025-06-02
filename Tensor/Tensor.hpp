#pragma once

struct Tensor {
public:
    size_t rows;
    size_t cols;
    size_t size;

    Tensor() : rows(0), cols(0), size(0) {}
    Tensor(size_t r, size_t c) : rows(r), cols(c), size(r*c), data((float*)aligned_alloc(32, r*c*sizeof(float))) {}
    Tensor(float* d, size_t r, size_t c) : data(d), rows(r), cols(c), size(r*c) {}

    inline float& operator () (size_t r, size_t c) {
        return data[r*cols+c];
    }

    inline const float& operator () (size_t r, size_t c) const {
        return data[r*cols+c];
    }

private:
    float* data;
};