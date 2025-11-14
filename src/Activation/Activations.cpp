#include "Activation.hpp"

/// @brief Stores the linear activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::Linear(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Size() == y.Size());
    assert(!x.HasNan());

    cblas_scopy(x.Size(), x.Data(), 1, y.Data(), 1);
}

/// @brief Stores the sigmoid activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::Sigmoid(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Data() != nullptr && y.Data() != nullptr);
    assert(x.Size() == y.Size());
    assert(!x.HasNan());

    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y.Data()[i] = 1.0f / (1.0f + expf(-x.Data()[i]));
    }
}

/// @brief Stores the relu activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::ReLU(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Data() != nullptr && y.Data() != nullptr);
    assert(x.Size() == y.Size());
    assert(!x.HasNan());

    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y.Data()[i] = x.Data()[i] > 0.0f ? x.Data()[i] : 0.0f;
    }
}

/// @brief Stores the leaky relu activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::LeakyReLU(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Data() != nullptr && y.Data() != nullptr);
    assert(x.Size() == y.Size());
    assert(!x.HasNan());

    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y.Data()[i] = x.Data()[i] > 0.0f ? x.Data()[i] : (x.Data()[i] * 0.1f);
    }
}

/// @brief Stores the elu activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::ELU(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Data() != nullptr && y.Data() != nullptr);
    assert(x.Size() == y.Size());
    assert(!x.HasNan());

    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y.Data()[i] = x.Data()[i] > 0.0f ? x.Data()[i] : (expf(x.Data()[i]) - 1.0f);
    }
}

/// @brief Stores the softmax activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::Softmax(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Dimensionality() == 2 && y.Dimensionality() == 2);
    assert(x.Data() != nullptr && y.Data() != nullptr);
    assert(x.Size() == y.Size());
    assert(!x.HasNan());

    const auto xDims = x.Dimensions();
    const size_t r = xDims[0];
    const size_t c = xDims[1];

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < r; i++) {

        // find max element in column
        float max = x.Data()[i*c+0];
        #pragma omp simd reduction(max:max)
        for (size_t j = 1; j < c; j++) {
            if (x.Data()[i*c+j] > max) {
                max = x.Data()[i*c+j];
            }
        }

        // get row sum and store exp in y
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (size_t j = 0; j < c; j++) {
            y.Data()[i*c+j] = expf(x.Data()[i*c+j]-max);
            sum += y.Data()[i*c+j];
        }

        // normalize
        const float inv = 1.0f / sum;
        #pragma omp simd
        for (size_t j = 0; j < c; j++) {
            y.Data()[i*c+j] *= inv;
        }
    }
}