#pragma once

struct LossMetric {
  public:
    enum class Type { None, MAE, MSE, Accuracy, OneHot };

    static inline Type ParseType(const std::string& name) {
        auto lower = std::string(name.size(), ' ');
        std::transform(name.begin(), name.end(), lower.begin(), tolower);

        if (lower == "mae") {
            return Type::MAE;
        } else if (lower == "mse") {
            return Type::MSE;
        } else if (lower == "accuracy") {
            return Type::Accuracy;
        } else if (lower == "onehot") {
            return Type::OneHot;
        } else {
            return Type::None;
        }
    }
    static inline std::string ParseName(const Type type) {
        switch (type) {
            case Type::None:
                return "None";
            case Type::MAE:
                return "MAE";
            case Type::MSE:
                return "MSE";
            case Type::Accuracy:
                return "Accuracy";
            case Type::OneHot:
                return "OneHot";
            default:
                return "";
        }
    }

    inline Type GetLossType() const { return lossType; }
    inline Type GetMetricType() const { return metricType; }
    inline void AssignPointers(const std::string& loss, const std::string& metric) { AssignPointers(ParseType(loss), ParseType(metric)); }
    inline void AssignPointers(const Type lossType, const Type metricType) {
        this->lossType   = lossType;
        this->metricType = metricType;

        switch (lossType) {
            case Type::MAE:
                loss = MAELoss;
                break;
            case Type::MSE:
                loss = MSELoss;
                break;
            case Type::OneHot:
                loss = OneHotLoss;
                break;
            default:
                loss = nullptr;
                break;
        }

        switch (metricType) {
            case Type::MAE:
                metric = MAEScore;
                break;
            case Type::MSE:
                metric = MSEScore;
                break;
            case Type::Accuracy:
                metric = AccuracyScore;
                break;
            default:
                metric = nullptr;
                break;
        }
    }

    void (*loss)(const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
    float (*metric)(const Tensor<float>&, const Tensor<float>&);

  private:
    Type lossType;
    Type metricType;

    /* ----------
    loss functions
    ---------- */
    static void MAELoss(const Tensor<float>& x, const Tensor<float>& y, Tensor<float>& c);
    static void MSELoss(const Tensor<float>& x, const Tensor<float>& y, Tensor<float>& c);
    static void OneHotLoss(const Tensor<float>& x, const Tensor<float>& y, Tensor<float>& c);

    /* ----------
     metric functions
     ---------- */
    static float MAEScore(const Tensor<float>& x, const Tensor<float>& y);
    static float MSEScore(const Tensor<float>& x, const Tensor<float>& y);
    static float AccuracyScore(const Tensor<float>& x, const Tensor<float>& y);
};
