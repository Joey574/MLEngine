#include "Activation.hpp"

/// @brief Stores the linear activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::Linear(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Size() == y.Size());
    cblas_scopy(x.Size(), x.Data(), 1, y.Data(), 1);
}

/// @brief Stores the sigmoid activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::Sigmoid(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Size() == y.Size());
    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}

/// @brief Stores the relu activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::ReLU(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Size() == y.Size());
    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

/// @brief Stores the leaky relu activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::LeakyReLU(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Size() == y.Size());
    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (x[i] * 0.1f);
    }
}

/// @brief Stores the elu activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::ELU(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Size() == y.Size());
    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (std::exp(x[i] - 1.0f));
    }
}

/// @brief Stores the softmax activation of x in y
/// @param x The matrix to apply the activation to
/// @param y The matrix to store the activation in
void Activation::Softmax(const Tensor<float>& x, Tensor<float>& y) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < r; i++) {

        // find max element in column
        float max = x[i*c+0];
        #pragma omp simd reduction(max:max)
        for (size_t j = 1; j < c; j++) {
            if (x[i*c+j] > max) {
                max = x[i*c+j];
            }
        }

        // get row sum and store exp in y
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (size_t j = 0; j < c; j++) {
            y[i*c+j] = std::exp(x[i*c+j]-max);
            sum += y[i*c+j];
        }

        // normalize
        const float inv = 1.0f / sum;
        #pragma omp simd
        for (size_t j = 0; j < c; j++) {
            y[i*c+j] *= inv;
        }
    }
}