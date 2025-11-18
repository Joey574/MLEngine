#include "Optimizer.hpp"

/// @brief Defines the specific optimizer implementation
/// @param config The config specifying the optimizer
/// @param weightSize Number of weights in the layer
/// @param biasSize Number of biases in the layer
void Optimizer::Define(const YAML::Node& config, size_t weightSize, size_t biasSize) {
    assert(!(defined || built));
    this->weightSize = weightSize;
    this->biasSize = biasSize;

    type = ParseType(config[Y_OPT_TYPE].as<std::string>(Y_OPTIMIZER_DEFAULT));
    learningRate = config[Y_OPT_LEARNINGRATE].as<float>(Y_LEARNRATE_DEFAULT);

    switch (type) {
        case Type::SGD:
            data.emplace<SGD>();
            break;
        case Type::MomentumSGD:
            data.emplace<MomentumSGD>();
            break;
        case Type::RMSProp:
            data.emplace<RMSProp>();
            break;
        case Type::Adam:
            data.emplace<Adam>();
            break;
    }

    // define specific optimizer
    std::visit([&](auto& data){
        data.Define(config);
    }, data);

    defined = true;
}

/// @brief Builds the optimizer
/// @param weights Layer's weight tensor
/// @param biases Layer's bias tensor
/// @param weightDerivatives Layer's weight derivative tensor
/// @param biasDerivatives Layer's bias derivative tensor
void Optimizer::Build(Tensor<float>& weights, Tensor<float>& biases, Tensor<float>& weightDerivatives, Tensor<float>& biasDerivatives) {
    assert(defined && !built);
    this->weights = &weights;
    this->biases = &biases;
    this->weightDerivatives = &weightDerivatives;
    this->biasDerivatives = &biasDerivatives;

    // build specific optimizer
    std::visit([&](auto& data) {
        data.Build(weightSize, biasSize);
    }, data);

    built = true;
}

/// @brief Updates the weights using the previously specified optimizer 
/// @param elements Numebr of training elements used
void Optimizer::Update(size_t elements) {
    assert(defined && built);

    // calls the proper optimizer's update function
    std::visit([&](auto& data) {
        data.Update(*weights, *biases, *weightDerivatives, *biasDerivatives, elements, learningRate);
    }, data);
}

int Optimizer::Save(std::ofstream& f) const {
    std::visit([&](const auto& data) {
        data.Save(f);
    }, data);
}
int Optimizer::Load(std::ifstream& f) {
    std::visit([&](auto& data) {
        data.Load(f);
    }, data);
}