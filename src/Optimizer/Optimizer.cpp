#include "Optimizer.hpp"

void Optimizer::Define(YAML::Node& config) {
    assert(!(defined || built));
    defined = true;
}

void Optimizer::Build() {
    assert(defined && !built);
    built = true;
}
