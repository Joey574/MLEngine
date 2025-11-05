#include "Activation.hpp"

/// @brief Scales y by the linear derivative of x
/// @param x The matrix to use to find the derivative
/// @param y The matrix to be scaled by the derivative
/// @param r Rows in the matrix
/// @param c Columns in the matrix
void Activation::LinearDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    return;
}

/// @brief Scales y by the sigmoid derivative of x
/// @param x The matrix to use to find the derivative
/// @param y The matrix to be scaled by the derivative
/// @param r Rows in the matrix
/// @param c Columns in the matrix
void Activation::SigmoidDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const size_t n = r*c;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        const float s = 1.0f / (1.0f + std::exp(-x[i]));
        y[i] *= s * (1.0f - s);
    }
}

/// @brief Scales y by the relu derivative of x
/// @param x The matrix to use to find the derivative
/// @param y The matrix to be scaled by the derivative
/// @param r Rows in the matrix
/// @param c Columns in the matrix
void Activation::ReLUDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const size_t n = r*c;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? y[i] : 0.0f;
    }
}

/// @brief Scales y by the leaky relu derivative of x
/// @param x The matrix to use to find the derivative
/// @param y The matrix to be scaled by the derivative
/// @param r Rows in the matrix
/// @param c Columns in the matrix
void Activation::LeakyReLUDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const size_t n = r*c;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? y[i] : (y[i] * 0.1f);
    }
}

/// @brief Scales y by the elu derivative of x
/// @param x The matrix to use to find the derivative
/// @param y The matrix to be scaled by the derivative
/// @param r Rows in the matrix
/// @param c Columns in the matrix
void Activation::ELUDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const size_t n = r*c;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? y[i] : (y[i] * std::exp(x[i]));
    }
}
