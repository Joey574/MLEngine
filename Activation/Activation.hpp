#pragma once
#include "../MathUtils/MathUtils.hpp"

/* @brief 
The Activation struct is responsible for providing utilities regarding activation functions and their derivatives.

Activation functions themselves will write the activation directly to the output location, highly parallelized and vectorized. Derivative functions are a little different, 
they take the activation value and the location of the values to multiply by the derivativethus no intermediary derivative value is needed, and the values are multiplied 
by the derivative in place.

Struct contents itself are mostly static, the only non static members being function pointers, a type value, and some initialization functions, these provide users to
abstract out calls to the proper activation / derivative, as they only need to pass the type and the right function will be set internally.

Also included is some basic parsing functionality for converting types to names and vice versa.
*/
struct Activation {
public:
    enum Type {
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