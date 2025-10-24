#include "Dataset.hpp"

int Dataset::Define(YAML::Node& config) {
    assert(!(defined || built));

    defined = true;
    return 0;
}

int Dataset::Build() {
    assert(defined && !built);

    built = true;
    return 0;
}
