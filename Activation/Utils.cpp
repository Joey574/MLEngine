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
        Type t = Activation::ParseSingleType(token);
        for (size_t i = 0; i < n; i++) {
            activations.push_back(t);
        }
    }

    return activations;
}
Activation::Type Activation::ParseSingleType(const std::string& actv) {
    if (actv == "linear") {
        return Activation::Type::linear;
    } else if (actv == "sigmoid") {
        return Activation::Type::sigmoid;
    } else if (actv == "relu") {
        return Activation::Type::relu;
    } else if (actv == "leakyrelu") {
        return Activation::Type::leakyrelu;
    } else if (actv == "elu") {
        return Activation::Type::elu;
    } else if (actv == "softmax") {
        return Activation::Type::softmax;
    }
    return Activation::Type::none;
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
