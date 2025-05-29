struct LossMetric {
public:
    enum class Type {
        none, mae, mse, accuracy, onehot
    };

    struct Metric {
        Type type;
        bool highestIsBest;
        float (*metric)(const float*, const float*, size_t, size_t);
    };    

    struct Loss {
        Type type;
        void (*loss)(const float*, const float*, float*, size_t, size_t);
    };

    std::vector<Metric> metrics;
    Loss loss;

private:

    // loss functions
    static void MaeLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols);
    static void MseLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols);
    static void OneHotLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols);

    // metric functions
    static float MaeScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols);
    static float MseScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols);
    static float AccuracyScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols);

    // simd utils
    static float Sum256(__m256 _x);

    // parsing utils
    static Type ParseType(const std::string& name);
    static std::string ParseName(Type type);
};