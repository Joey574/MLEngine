#include "LossMetric.hpp"

void LossMetric::MAELoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    const size_t n = rows*cols;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        c[i] = (x[i] - y[i]) > 0.0f ? 1.0f : -1.0f;
    }
}

void LossMetric::MSELoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    const size_t n = rows*cols;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        c[i] = 2.0f * (x[i] - y[i]);
    }
}

void LossMetric::OneHotLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    const size_t n = rows*cols;
    cblas_scopy(n, x, 1, c, 1);

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < rows; i++) {
        c[i*cols+(int)y[i]]--;
    }
}
