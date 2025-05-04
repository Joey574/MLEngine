#include "TestNetwork.hpp"

float* TestNetwork::InitializeTestData(size_t n) {
    // allocate larger than needed memory block to replicate real world setup
    float* t = (float*)aligned_alloc(32, n*sizeof(float));
    if (t == nullptr) {
        std::cerr << "fubar memory alloc\n";
    }

    // generate random set of input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0, 10.0);
    
    for (size_t i = 0; i < n; i++) {
        t[i] = dis(gen);
    }

    return t;
}
