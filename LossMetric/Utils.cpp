#include "LossMetric.hpp"

LossMetric::Type LossMetric::ParseType(const std::string& lm) {
    if (lm == "mae") {
        return Type::mae;
    } else if (lm == "mse") {
        return Type::mse;
    } else if (lm == "accuracy") {
        return Type::accuracy;
    } else if (lm == "onehot") {
        return Type::onehot;
    }

    return Type::none;
}
std::string LossMetric::ParseName(Type type) {
    switch (type) {
        case Type::mae:
            return "mae";
        case Type::mse:
            return "mse";
        case Type::accuracy:
            return "accuracy";
        case Type::onehot:
            return "onehot";
        default:
            return "none";
    }
}

void LossMetric::AssignPointers(Type l, Type m) {
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
