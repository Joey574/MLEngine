#pragma once

struct LossMetric {
public:
    enum Type {
        None, MAE, MSE, Accuracy, OneHot
    };

private:
    Type lossType;
    Type metricType;
};
