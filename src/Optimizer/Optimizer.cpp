#include "Optimizer.hpp"

/// @brief Defines the specific optimizer implementation
/// @param config The config specifying the optimizer
/// @param weightSize Number of weights in the layer
/// @param biasSize Number of biases in the layer
void Optimizer::Define(YAML::Node& config, size_t weightSize, size_t biasSize) {
    assert(!(defined || built));
    this->weightSize = weightSize;
    this->biasSize = biasSize;

    type = ParseType(config[Y_OPT_TYPE].as<std::string>(Y_OPTIMIZER_DEFAULT));
    learningRate = config[Y_OPT_LEARNINGRATE].as<float>(Y_LEARNRATE_DEFAULT);

    switch (type) {
        case Type::SGD:
            data = SGD{};
        case Type::MomentumSGD:
            data = MomentumSGD{};
        case Type::RMSProp:
            data = RMSProp{};
        case Type::Adam:
            data = Adam{};
    }

    // define specific optimizer
    std::visit([&](auto& data){
        data.Define(config);
    }, data);

    defined = true;
}

/// @brief Builds the optimizer
/// @param weights Pointer to the layer weights
/// @param biases Pointer to the layer biases
/// @param weightDerivatives Pointer to the derivatives of the weights
/// @param biasDerivatives Pointer to the derivatives of the biases
void Optimizer::Build(float* __restrict weights, float* __restrict biases, float* __restrict weightDerivatives, float* __restrict biasDerivatives) {
    assert(defined && !built);
    this->weights = weights;
    this->biases = biases;
    this->weightDerivatives = weightDerivatives;
    this->biasDerivatives = biasDerivatives;

    // build specific optimizer
    std::visit([&](auto& data) {
        data.Build();
    }, data);

    built = true;
}

/// @brief Updates the weights using the previously specified optimizer 
/// @param elements Numebr of training elements used
void Optimizer::Update(size_t elements) {
    assert(defined && built);

    // calls the proper optimizer's update function
    std::visit([&](auto& data) {
        data.Update(weights, biases, weightDerivatives, biasDerivatives, weightSize, biasSize, elements, learningRate);
    }, data);
}
