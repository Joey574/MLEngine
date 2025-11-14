#include "LossMetric.hpp"

/// @brief Computes the mean absolute loss between x and y, and stores it in c
/// @param x Prediction tensor
/// @param y Truth tensor
/// @param c Tensor to store loss in
void LossMetric::MAELoss(const Tensor<float>& x, const Tensor<float>& y, Tensor<float>& c) {
    assert(x.Data() != nullptr && y.Data() != nullptr && c.Data() != nullptr);
    assert(x.Size() == y.Size() && y.Size() == c.Size());
    assert(!x.HasNan() && !y.HasNan() && !c.HasNan());

    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        c.Data()[i] = (x.Data()[i] - y.Data()[i]) > 0.0f ? 1.0f : -1.0f;
    }
}

/// @brief Computes the mean squared loss between x and y, and stores it in c
/// @param x Prediction tensor
/// @param y Truth tensor
/// @param c Tensor to store loss in
void LossMetric::MSELoss(const Tensor<float>& x, const Tensor<float>& y, Tensor<float>& c) {
    assert(x.Data() != nullptr && y.Data() != nullptr && c.Data() != nullptr);
    assert(x.Size() == y.Size() && y.Size() == c.Size());
    assert(!x.HasNan() && !y.HasNan() && !c.HasNan());
    
    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        c.Data()[i] = 2.0f * (x.Data()[i] - y.Data()[i]);
    }
}

/// @brief Computes the one hot loss between x and y, and stores it in c
/// @param x Prediction tensor
/// @param y Truth tensor
/// @param c Tensor to store loss in
void LossMetric::OneHotLoss(const Tensor<float>& x, const Tensor<float>& y, Tensor<float>& c) {
    assert(x.Data() != nullptr && y.Data() != nullptr && c.Data() != nullptr);
    assert(!x.HasNan() && !y.HasNan() && !c.HasNan());
    assert(x.Size() == c.Size());
    
    const size_t n = x.Size();

    const auto xDims = x.Dimensions();
    const size_t rows = xDims[0];
    const size_t cols = xDims[1];

    cblas_scopy(n, x.Data(), 1, c.Data(), 1);

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < rows; i++) {
        c.Data()[i*cols+(int)y.Data()[i]]--;
    }
}
