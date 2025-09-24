#pragma once

struct LossMetric {
public:
    enum Type {
        None, MAE, MSE, Accuracy, OneHot
    };

    void (*Loss)(const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
    float (*Score)(const Tensor<float>&, const Tensor<float>&);

    LossMetric(Type lossType = Type::None, Type metricType = Type::None) {
        this->lossType = lossType;
        this->metricType = metricType;
    }

    inline Type LossType() const { return lossType; }
    inline Type MetricType() const { return metricType; }

    static Type ParseType(const std::string& name);
    static std::string ParseName(Type type);

private:
    Type lossType;
    Type metricType;


    static void MAELoss(const Tensor<float>& predicted, const Tensor<float>& truth, Tensor<float>& out);
    static void MSELoss(const Tensor<float>& predicted, const Tensor<float>& truth, Tensor<float>& out);
    static void OneHotLoss(const Tensor<float>& predicted, const Tensor<float>& truth, Tensor<float>& out);

    static float MAEScore(const Tensor<float>& predicted, const Tensor<float>& truth);
    static float MSEScore(const Tensor<float>& predicted, const Tensor<float>& truth);
    static float AccuracyScore(const Tensor<float>& predicted, const Tensor<float>& truth);
};
