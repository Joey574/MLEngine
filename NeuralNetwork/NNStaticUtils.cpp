#include "NeuralNetwork.hpp"

std::vector<size_t> NeuralNetwork::ParseCompact(const std::vector<std::string>& dims) {
    std::vector<size_t> dimensions;

    for (size_t i = 0; i < dims.size(); i++) {
        // get number of layers
        size_t n = 1;
        std::string token = dims[i];
        if (dims[i].find('X') != std::string::npos) {
            n = std::stoi(dims[i].substr(dims[i].find('X')+1));
            token = dims[i].substr(0, dims[i].find('X'));

        }

        // append n layers of t
        size_t t = std::stoi(token);
        for (size_t i = 0; i < n; i++) {
            dimensions.push_back(t);
        }
    }

    return dimensions;
}
std::vector<NeuralNetwork::ActivationFunctions> NeuralNetwork::ParseActvs(const std::vector<std::string>& actvs) {
    std::vector<ActivationFunctions> activations;

    for (size_t i = 0; i < actvs.size(); i++) {
        // get number of layers
        size_t n = 1;
        std::string token = actvs[i];
        if (actvs[i].find('X') != std::string::npos) {
            n = std::stoi(actvs[i].substr(actvs[i].find('X')+1));
            token = actvs[i].substr(0, actvs[i].find('X'));

        }

        // add n number of token
        if (token == "sigmoid") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(ActivationFunctions::sigmoid);
            }
        } else if (token == "relu") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(ActivationFunctions::relu);
            }
        } else if (token == "leakyrelu") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(ActivationFunctions::leakyrelu);
            }
        } else if (token == "elu") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(ActivationFunctions::elu);
            }
        } else if (token == "softmax") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(ActivationFunctions::softmax);
            }
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
