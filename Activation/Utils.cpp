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
std::string Activation::ParseName() const {
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

__m256 Activation::Exp256(__m256 _x) {
    __m256 _a = _mm256_set1_ps(12102203.0f); 
    __m256 _b = _mm256_set1_ps(127.0f * (1 << 23));
    __m256 _c = _mm256_fmadd_ps(_x, _a, _b);

    __m256i _res = _mm256_cvtps_epi32(_c);

    return _mm256_castsi256_ps(_res);
}