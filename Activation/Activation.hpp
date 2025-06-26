#pragma once
#include "../MathUtils/MathUtils.hpp"

struct Activation {
public:
    enum class Type {
        none, linear, sigmoid, relu, leakyrelu, elu, softmax
    };

    Activation() { AssignPointers(Type::none); }
    Activation(Type a) { AssignPointers(a); }

    Type type;
    void (*activation)(const float*, float*, size_t, size_t);
    void (*derivative)(const float*, float*, size_t, size_t);

    // parsing functions
    static Type ParseType(const std::string& actv);
    static std::string ParseName(Type type);

    void AssignPointers(Type a);

private:
    // activation functions
    static void Linear(const float* __restrict x, float* __restrict y, size_t r, size_t c);
    static void Sigmoid(const float* __restrict x, float* __restrict y, size_t r, size_t c);
    static void ReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c);
    static void LeakyReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c);
    static void ELU(const float* __restrict x, float* __restrict y, size_t r, size_t c);
    static void Softmax(const float* __restrict x, float* __restrict y, size_t r, size_t c);

    // derivatives functions
    static void LinearDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c);
    static void SigmoidDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c);
    static void ReLUDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c);
    static void LeakyReLUDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c);
    static void ELUDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c);
};