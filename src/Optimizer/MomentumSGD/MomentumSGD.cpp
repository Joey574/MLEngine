#include "MomentumSGD.hpp"

void MomentumSGD::Define() {
    assert(!(defined || built));

    defined = true;
}

void MomentumSGD::Build() {
    assert(defined && !built);

    built = true;
}

void MomentumSGD::Compute() {
    assert(defined && built);
}
