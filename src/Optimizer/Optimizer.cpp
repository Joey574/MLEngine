#include "Optimizer.hpp"

void Optimizer::Define(YAML::Node& config) {
    assert(!(defined || built));
    defined = true;
}

void Optimizer::Build() {
    assert(defined && !built);
    built = true;
}

void Optimizer::Compute() {
    // Calls the proper optimizer's compute function
    std::visit([](auto& data) {
        data.Compute();
    }, data);
}
