#include "Adam.hpp"

void Adam::Define() {
    assert(!(defined || built));

    defined = true;
}

void Adam::Build() {
    assert(defined && !built);

    built = true;
}

void Adam::Compute() {
    assert(defined && built);
}
