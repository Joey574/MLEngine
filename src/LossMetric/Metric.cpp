#include "LossMetric.hpp"

/// @brief Computes the mean absolute error between x and y
/// @param x Prediction matrix
/// @param y Ground truth matrix
/// @return The mean absolute error
float LossMetric::MAEScore(const Tensor<float>& x, const Tensor<float>& y) {
    assert(x.Size() == y.Size());
    const size_t n = x.Size();
    float error = 0.0f;

    #pragma omp parallel for simd schedule(static) reduction(+:error)
    for (size_t i = 0; i < n; i++) {
        error += fabsf(x.Data()[i] - y.Data()[i]);
    }

    return error / (float)n;
}

/// @brief Computes the mean squared error between x and y
/// @param x Prediction matrix
/// @param y Ground truth matrix
/// @return The mean squared error
float LossMetric::MSEScore(const Tensor<float>& x, const Tensor<float>& y) {
    assert(x.Size() == y.Size());
    const size_t n = x.Size();
    float error = 0.0f;

    #pragma omp parallel for simd schedule(static) reduction(+:error)
    for (size_t i = 0; i < n; i++) {
        error += (x.Data()[i]-y.Data()[i])*(x.Data()[i]-y.Data()[i]);
    }

    return error / (float)n;
}

/// @brief Computes the accuracy between x and y
/// @param x Prediction matrix
/// @param y Ground truth matrix
/// @return Accuracy as a percentage
float LossMetric::AccuracyScore(const Tensor<float>& x, const Tensor<float>& y) {
    assert(x.Size() == y.Size());

    const auto dims = x.Dimensions();
    const size_t rows = dims[0];
    const size_t cols = dims[1];

    size_t correct = 0;

    #pragma omp parallel for schedule(static) reduction(+:correct)
    for (size_t r = 0; r < rows; r++) {

        // find max element and its index in column
        size_t midx = 0;
        float max = x.Data()[r*cols+0];
        #pragma omp simd reduction(max:max)
        for (size_t c = 1; c < cols; c++) {
            if (x.Data()[r*cols+c] > max) {
                max = x.Data()[r*cols+c];
                midx = c;
            }
        }

        if (midx == y.Data()[r]) { correct++; }
    }

    return ((float)correct / (float)rows) * 100.0f;
}
