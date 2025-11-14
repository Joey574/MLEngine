#include "Activation.hpp"

/// @brief Scales y by the linear derivative of x
/// @param x The matrix to use to find the derivative
/// @param y The matrix to be scaled by the derivative
void Activation::LinearDerivative(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Size() == y.Size());
    return;
}

/// @brief Scales y by the sigmoid derivative of x
/// @param x The matrix to use to find the derivative
/// @param y The matrix to be scaled by the derivative
void Activation::SigmoidDerivative(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Data() != nullptr && y.Data() != nullptr);
    assert(!x.HasNan() && !y.HasNan());
    assert(x.Size() == y.Size());

    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        const float s = 1.0f / (1.0f + std::exp(-x.Data()[i]));
        y.Data()[i] *= s * (1.0f - s);
    }
}

/// @brief Scales y by the relu derivative of x
/// @param x The matrix to use to find the derivative
/// @param y The matrix to be scaled by the derivative
void Activation::ReLUDerivative(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Data() != nullptr && y.Data() != nullptr);
    assert(!x.HasNan() && !y.HasNan());
    assert(x.Size() == y.Size());

    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y.Data()[i] = x.Data()[i] > 0.0f ? y.Data()[i] : 0.0f;
    }
}

/// @brief Scales y by the leaky relu derivative of x
/// @param x The matrix to use to find the derivative
/// @param y The matrix to be scaled by the derivative
void Activation::LeakyReLUDerivative(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Data() != nullptr && y.Data() != nullptr);
    assert(!x.HasNan() && !y.HasNan());
    assert(x.Size() == y.Size());

    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y.Data()[i] = x.Data()[i] > 0.0f ? y.Data()[i] : (y.Data()[i] * 0.1f);
    }
}

/// @brief Scales y by the elu derivative of x
/// @param x The matrix to use to find the derivative
/// @param y The matrix to be scaled by the derivative
void Activation::ELUDerivative(const Tensor<float>& x, Tensor<float>& y) {
    assert(x.Data() != nullptr && y.Data() != nullptr);
    assert(!x.HasNan() && !y.HasNan());
    assert(x.Size() == y.Size());
    
    const size_t n = x.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        y.Data()[i] = x.Data()[i] > 0.0f ? y.Data()[i] : (y.Data()[i] * std::exp(x.Data()[i]));
    }
}
