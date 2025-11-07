#include "RMSProp.hpp"

void RMSProp::Define() {
    assert(!(defined || built));

    defined = true;
}

void RMSProp::Build() {
    assert(defined && !built);

    built = true;
}

void RMSProp::Compute() {
    assert(defined && built);
}
