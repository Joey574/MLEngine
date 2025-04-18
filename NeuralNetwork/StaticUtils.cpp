#include "NeuralNetwork.hpp"

std::vector<size_t> NeuralNetwork::parse_compact(const std::string& dims) {
    std::vector<size_t> dimensions;

    auto split = dims | std::views::split('-');
    for (const auto& range : split) {
        std::string token(range.begin(), range.end());
        dimensions.push_back(std::stoi(token));
    }

    return dimensions;
}
std::vector<NeuralNetwork::ActivationFunctions> NeuralNetwork::parse_actvs(const std::string& actvs) {
    std::vector<ActivationFunctions> activations;

    auto split = actvs | std::views::split('-');
    for (const auto& range : split) {
        std::string token(range.begin(), range.end());

        if (token == "sigmoid") {
            activations.push_back(ActivationFunctions::sigmoid);
        } else if (token == "relu") {
            activations.push_back(ActivationFunctions::relu);
        } else if (token == "leakyrelu") {
            activations.push_back(ActivationFunctions::leaky_relu);
        } else if (token == "elu") {
            activations.push_back(ActivationFunctions::elu);
        } else if (token == "softmax") {
            activations.push_back(ActivationFunctions::softmax);
        }
    }


    return activations;
}

NeuralNetwork::LossMetric NeuralNetwork::parse_lm(const std::string& lm) {
    if (lm == "mae") {
        return LossMetric::mae;
    } else if (lm == "accuracy") {
        return LossMetric::accuracy;
    } else if (lm == "onehot") {
        return LossMetric::one_hot;
    }

    return LossMetric::none;
}

NeuralNetwork::WeightInitialization NeuralNetwork::parse_weight(const std::string& weight) {
    if (weight == "he") {
        return WeightInitialization::he;
    } else if (weight == "normalize") {
        return WeightInitialization::normalize;
    } else if (weight == "xavier") {
        return WeightInitialization::xavier;
    }

    return WeightInitialization::none;
}