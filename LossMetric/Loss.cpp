#include "LossMetric.hpp"
#include <cblas.h>

/// @brief Computes the mean absolute loss between x and y, and stores it in c
/// @param x Prediction tensor
/// @param y Truth tensor
/// @param c Tensor to store loss in
void LossMetric::MaeLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    const size_t n = rows*cols;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        c[i] = (x[i] - y[i]) > 0.0f ? 1.0f : -1.0f;
    }
}

/// @brief Computes the mean squared loss between x and y, and stores it in c
/// @param x Prediction tensor
/// @param y Truth tensor
/// @param c Tensor to store loss in
void LossMetric::MseLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    const size_t n = rows*cols;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        c[i] = 2.0f * (x[i] - y[i]);
    }
}

/// @brief Computes the one hot loss between x and y, and stores it in c
/// @param x Prediction tensor
/// @param y Truth tensor
/// @param c Tensor to store loss in
void LossMetric::OneHotLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    const size_t n = rows*cols;
    cblas_scopy(n, x, 1, c, 1);

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < rows; i++) {
        c[i * cols + (int)y[i]]--;
    }
}
