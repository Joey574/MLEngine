#pragma once

struct Activation {
public:
    enum class Type {
        none, linear, sigmoid, relu, leakyrelu, elu, softmax
    };

    Activation() { AssignPointers(Type::none); }
    Activation(Type a) { AssignPointers(a); }

    Type type;
    void (*activation)(const float*, float*, size_t);
    void (*derivative)(const float*, float*, size_t);

    // parsing functions
    static std::vector<Type> ParseType(const std::vector<std::string>& actvs);
    static std::string ParseName(Type type);

    // single mm256 activations
    inline static __m256 Linear(const __m256 _x) {
        return _x;
    }
    static __m256 Sigmoid(const __m256 _x);
    inline static __m256 ReLU(const __m256 _x) {
        const __m256 _zero = _mm256_setzero_ps();
        const __m256 _res = _mm256_max_ps(_x, _zero);
        return _res;
    }
    inline static __m256 LeakyReLU(const __m256 _x) {
        const __m256 _cof = _mm256_set1_ps(0.1f);
        const __m256 _zero = _mm256_setzero_ps();
        const __m256 _x2 = _mm256_mul_ps(_x, _cof);
        const __m256 _res = _mm256_max_ps(_x2, _x);
        return _res;
    }
    static __m256 ELU(const __m256 _x);

    // single float activations
    inline static float Linear(const float x) {
        return x;
    }
    inline static float Sigmoid(const float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    inline static float ReLU(const float x) {
        return x > 0.0f ? x : 0.0f;;
    }
    inline static float LeakyReLU(const float x) {
        return x > 0.0f ? x : (x*0.1f);
    }
    inline static float ELU(const float x) {
        return x > 0.0f ? x : (std::exp(x)-1.0f);
    }

    // single float derivatives
    inline static float LinearDerivative(const float x, const float y) {
        return y;
    }
    inline static float SigmoidDerivative(const float x, const float y) {
        float s = 1.0f / (1.0f + std::exp(-x));
        return y * s * (1.0f-s);
    }
    inline static float ReLUDerivative(const float x, const float y) {
        return x > 0.0f ? y : 0.0f;
    }
    inline static float LeakyReLUDerivative(const float x, const float y) {
        return x > 0.0f ? y : (y * 0.1f);
    }
    inline static float ELUDerivative(const float x, const float y) {
        return x > 0.0f ? y : (y * std::exp(x));
    }

private:
    // activation functions
    static void Linear(const float* __restrict x, float* __restrict y, size_t n);
    static void Sigmoid(const float* __restrict x, float* __restrict y, size_t n);
    static void ReLU(const float* __restrict x, float* __restrict y, size_t n);
    static void LeakyReLU(const float* __restrict x, float* __restrict y, size_t n);
    static void ELU(const float* __restrict x, float* __restrict y, size_t n);
    static void Softmax(const float* __restrict x, float* __restrict y, size_t n);

    // derivatives functions
    static void LinearDerivative(const float* __restrict x, float* __restrict y, size_t n);
    static void SigmoidDerivative(const float* __restrict x, float* __restrict y, size_t n);
    static void ReLUDerivative(const float* __restrict x, float* __restrict y, size_t n);
    static void LeakyReLUDerivative(const float* __restrict x, float* __restrict y, size_t n);
    static void ELUDerivative(const float* __restrict x, float* __restrict y, size_t n);

    void AssignPointers(Type a);
};