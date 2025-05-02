#include "TestNetwork.hpp"

void TestNetwork::TestDerivatives() {
    const size_t ssize = 10000;
    const size_t esize = 1000000;

    for (size_t i = ssize; i <= esize; i *= 10) {
        std::cout << "Testing Derivatives (" << i << "):\n";
        std::cout << "\n";
    }
}
