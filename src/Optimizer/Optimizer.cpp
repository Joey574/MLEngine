#include "Optimizer.hpp"

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

void Optimizer::Update(size_t elements) {
    assert(defined && built);

    // calls the proper optimizer's update function
    std::visit([&](auto& data) {
        data.Update(weights, biases, weightDerivatives, biasDerivatives, weightSize, biasSize, elements, learningRate);
    }, data);
}
