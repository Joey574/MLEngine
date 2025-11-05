#include "LossMetric.hpp"

/// @brief Computes the mean absolute error between x and y
/// @param x Prediction matrix
/// @param y Ground truth matrix
/// @param rows Rows in the matrix
/// @param cols Columns in the matrix
/// @return The mean absolute error
float LossMetric::MAEScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    const size_t n = rows*cols;
    float error = 0.0f;

    #pragma omp parallel for simd schedule(static) reduction(+:error)
    for (size_t i = 0; i < n; i++) {
        error += fabsf(x[i] - y[i]);
    }

    return error / (float)n;
}

/// @brief Computes the mean squared error between x and y
/// @param x Prediction matrix
/// @param y Ground truth matrix
/// @param rows Rows in the matrix
/// @param cols Columns in the matrix
/// @return The mean squared error
float LossMetric::MSEScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    const size_t n = rows*cols;
    float error = 0.0f;

    #pragma omp parallel for simd schedule(static) reduction(+:error)
    for (size_t i = 0; i < n; i++) {
        error += (x[i]-y[i])*(x[i]-y[i]);
    }

    return error / (float)n;
}

/// @brief Computes the accuracy between x and y
/// @param x Prediction matrix
/// @param y Ground truth matrix
/// @param rows Rows in the matrix
/// @param cols Columns in the matrix
/// @return Accuracy as a percentage
float LossMetric::AccuracyScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    size_t correct = 0;

    #pragma omp parallel for schedule(static) reduction(+:correct)
    for (size_t r = 0; r < rows; r++) {

        // find max element and its index in column
        size_t midx = 0;
        float max = x[r*cols+0];
        #pragma omp simd reduction(max:max)
        for (size_t c = 1; c < cols; c++) {
            if (x[r*cols+c] > max) {
                max = x[r*cols+c];
                midx = c;
            }
        }

        if (midx == y[r]) { correct++; }
    }

    return ((float)correct / (float)rows) * 100.0f;
}
