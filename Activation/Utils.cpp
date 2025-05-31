#include "Activation.hpp"

std::vector<Activation::Type> Activation::ParseType(const std::vector<std::string>& actvs) {
    std::vector<Type> activations;

    for (size_t i = 0; i < actvs.size(); i++) {
        // get number of layers
        size_t n = 1;
        std::string token = actvs[i];
        if (actvs[i].find('X') != std::string::npos) {
            n = std::stoi(actvs[i].substr(actvs[i].find('X')+1));
            token = actvs[i].substr(0, actvs[i].find('X'));
        }

        // add n number of token
        if (token == "linear") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(Type::linear);
            }
        } else if (token == "sigmoid") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(Type::sigmoid);
            }
        } else if (token == "relu") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(Type::relu);
            }
        } else if (token == "leakyrelu") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(Type::leakyrelu);
            }
        } else if (token == "elu") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(Type::elu);
            }
        } else if (token == "softmax") {
            for (size_t i = 0; i < n; i++) {
                activations.push_back(Type::softmax);
            }
        }
    }

    return activations;
}
std::string Activation::ParseName(Type type) {
    switch(type) {
        case Type::linear:
            return "linear";
        case Type::sigmoid:
            return "sigmoid";
        case Type::relu:
            return "relu";
        case Type::leakyrelu:
            return "leakyrelu";
        case Type::elu:
            return "elu";
        case Type::softmax:
            return "softmax";
        default:
            return "none";
    }
}

void Activation::AssignPointers(Type a) {
    type = a;

    switch (a) {
        case Type::linear:
            activation = Linear;
            derivative = LinearDerivative;
            break;

        case Type::sigmoid:
            activation = Sigmoid;
            derivative = SigmoidDerivative;
            break;

        case Type::relu:
            activation = ReLU;
            derivative = ReLUDerivative;
            break;

        case Type::leakyrelu:
            activation = LeakyReLU;
            derivative = LeakyReLUDerivative;
            break;

        case Type::elu:
            activation = ELU;
            derivative = ELUDerivative;
            break;

        case Type::softmax:
            activation = Softmax;
            derivative = nullptr;
            break;
        default:
            activation = nullptr;
            derivative = nullptr;
            break;
    }
}
