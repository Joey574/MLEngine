struct Activation {
public:
    enum class Type {
        none, linear, sigmoid, relu, leakyrelu, elu, softmax
    };

    Type type;
    void (*activation)(const float*, float*, size_t);
    void (*derivative)(const float*, float*, size_t);

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

    // math utils
    static __m256 Exp256(__m256 _x);

    // parsing functions
    static std::vector<Type> ParseType(const std::vector<std::string>& actvs);
    std::string ParseName() const;
};