#include "Optimizer.hpp"

void Optimizer::Define(YAML::Node& config) {
    assert(!(IsDefined() || IsBuilt()));
    type = ParseType(config[Y_OPT_TYPE].as<std::string>(Y_OPTIMIZER_DEFAULT));

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
    std::visit([](auto& data){
        data.Define();
    }, data);
}

void Optimizer::Build() {
    assert(IsDefined() && !IsBuilt());

    // build specific optimizer
    std::visit([](auto& data) {
        data.Build();
    }, data);
}

void Optimizer::Compute() {
    assert(IsDefined() && IsBuilt());

    // calls the proper optimizer's compute function
    std::visit([](auto& data) {
        data.Compute();
    }, data);
}
