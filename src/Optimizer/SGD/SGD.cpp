#include "SGD.hpp"

void SGD::Define() {
    assert(!(defined || built));

    defined = true;
}

void SGD::Build() {
    assert(defined && !built);

    built = true;
}

void SGD::Compute() {
    assert(defined && built);
}
