#pragma once

/* @brief

*/
struct LossMetric {
public:
    enum class Type {
        none, mae, mse, accuracy, onehot
    };

    Type mtype;
    bool highestIsBest;
    float (*metric)(const float*, const float*, size_t, size_t);

    Type ltype;
    void (*loss)(const float*, const float*, float*, size_t, size_t);

    LossMetric() { AssignPointers(Type::none, Type::none); };
    LossMetric(Type l, Type m) { AssignPointers(l, m); };

    // parsing utils
    static inline Type ParseType(const std::string& name) {
        if (name == "mae") {
            return Type::mae;
        } else if (name == "mse") {
            return Type::mse;
        } else if (name == "accuracy") {
            return Type::accuracy;
        } else if (name == "onehot") {
            return Type::onehot;
        } else if (name == "none") {
            return Type::none;
        } else {
            std::cerr << "[-] Invalid LossMetric Type: " << name << "\n";
            return Type::none;
        }
    };
    static inline std::string ParseName(Type type) {
        switch (type) {
            case Type::mae:
                return "mae";
            case Type::mse:
                return "mse";
            case Type::accuracy:
                return "accuracy";
            case Type::onehot:
                return "onehot";
            case Type::none:
                return "none";
            default:
                std::cerr << "[-] Invalid LossMetric Type: " << (int)type << "\n";
                return "none";
        }
    }

    void AssignPointers(Type l, Type m) {
        ltype = l;
        mtype = m;

        switch (l) {
            case Type::mae:
                loss = MaeLoss;
                break;
            case Type::mse:
                loss = MseLoss;
                break;
            case Type::onehot:
                loss = OneHotLoss;
                break;
            default:
                loss = nullptr;
                break;
        }

        switch (m) {
            case Type::mae:
                highestIsBest = false;
                metric = MaeScore;
                break;
            case Type::mse:
                highestIsBest = false;
                metric = MseScore;
                break;
            case Type::accuracy:
                highestIsBest = true;
                metric = AccuracyScore;
                break;
            default:
                metric = nullptr;
                break;
        }
    }


private:

    // loss functions
    static void MaeLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols);
    static void MseLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols);
    static void OneHotLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols);

    // metric functions
    static float MaeScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols);
    static float MseScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols);
    static float AccuracyScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols);
};
