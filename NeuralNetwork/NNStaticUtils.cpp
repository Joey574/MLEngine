#include "NeuralNetwork.hpp"

std::vector<size_t> NeuralNetwork::ParseCompact(const std::string& dims) {
    std::vector<size_t> dimensions;

    auto split = dims | std::views::split('-');
    for (const auto& range : split) {
        std::string token(range.begin(), range.end());
        dimensions.push_back(std::stoi(token));
    }

    return dimensions;
}
std::vector<NeuralNetwork::ActivationFunctions> NeuralNetwork::ParseActvs(const std::string& actvs) {
    std::vector<ActivationFunctions> activations;

    auto split = actvs | std::views::split('-');
    for (const auto& range : split) {
        std::string token(range.begin(), range.end());

        if (token == "sigmoid") {
            activations.push_back(ActivationFunctions::sigmoid);
        } else if (token == "relu") {
            activations.push_back(ActivationFunctions::relu);
        } else if (token == "leakyrelu") {
            activations.push_back(ActivationFunctions::leakyrelu);
        } else if (token == "elu") {
            activations.push_back(ActivationFunctions::elu);
        } else if (token == "softmax") {
            activations.push_back(ActivationFunctions::softmax);
        }
    }

    return activations;
}

NeuralNetwork::LossMetric NeuralNetwork::ParseLossMetric(const std::string& lm) {
    if (lm == "mae") {
        return LossMetric::mae;
    } else if (lm == "accuracy") {
        return LossMetric::accuracy;
    } else if (lm == "onehot") {
        return LossMetric::onehot;
    } else if (lm == "mse") {
        return LossMetric::mse;
    }

    return LossMetric::none;
}
NeuralNetwork::WeightInitialization NeuralNetwork::ParseWeight(const std::string& weight) {
    if (weight == "he") {
        return WeightInitialization::he;
    } else if (weight == "normalize") {
        return WeightInitialization::normalize;
    } else if (weight == "xavier") {
        return WeightInitialization::xavier;
    }

    return WeightInitialization::none;
}

std::string NeuralNetwork::ActivationString(const ActivationFunctions actv) {
    switch (actv) {
        case ActivationFunctions::sigmoid:
            return "sigmoid";
        case ActivationFunctions::relu:
            return "relu";
        case ActivationFunctions::leakyrelu:
            return "leakyrelu";
        case ActivationFunctions::elu:
            return "elu";
        case ActivationFunctions::softmax:
            return "softmax";
        default:
            return "none";
    }
}
std::string NeuralNetwork::WeightString(const WeightInitialization w) {
    switch (w) {
        case WeightInitialization::he:
            return "he";
        case WeightInitialization::normalize:
            return "normalize";
        case WeightInitialization::xavier:
            return "xavier";
        default:
            return "none";
    }
}
std::string NeuralNetwork::LossMetricString(const LossMetric lm) {
    switch (lm) {
        case LossMetric::accuracy:
            return "accuracy";
        case LossMetric::mae:
            return "mae";
        case LossMetric::onehot:
            return "onehot";
        case LossMetric::mse:
            return "mse";
        default:
            return "none";
    }
}
