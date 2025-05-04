#include "TestNetwork.hpp"

void TestNetwork::TestActivations() {
    const size_t ssize = 10000000;
    const size_t esize = 10000000;

    for (size_t i = ssize; i <= esize; i *= 10) {
        std::cout << "Testing Activations (" << i << "):\n";
        TestSigmoid(i);
        TestReLU(i);
        TestLeakyReLU(i);
        TestELU(i);
        TestSoftmax(i);
        std::cout << "\n";
    }
}
void TestNetwork::TestSigmoid(size_t n) {
    const double acceptederror = 0.025;
    
    size_t size = ((n*2 + 40) + 31) & ~31;
    float* t = InitializeTestData(size);
    double* expected = (double*)malloc(size * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        expected[i] = 1.00 / (1.00 + std::exp(-(double)t[i]));
    }

    // warmup runs
    for (size_t i = 0; i < 5; i++) {
        NeuralNetwork::Sigmoid(t, t+n, n);
        asm volatile("" ::: "memory");
    }

    // test sigmoid
    auto start = std::chrono::high_resolution_clock::now();
    NeuralNetwork::Sigmoid(t, t+n, n);
    asm volatile("" ::: "memory");
    double dur = (std::chrono::high_resolution_clock::now() - start).count();
    
    double mpe = ComputeMPE(expected, t+n, n);
    free(expected);
    free(t);

    FinalizeTestOutput("Sigmoid", dur, mpe, acceptederror, n);
}
void TestNetwork::TestReLU(size_t n) {
    const double acceptederror = 0.000001;
    
    size_t size = ((n*2 + 40) + 31) & ~31;
    float* t = InitializeTestData(size);
    double* expected = (double*)malloc(size * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        expected[i] = t[i] > 0.0 ? t[i] : 0.0;
    }

    // warmup runs
    for (size_t i = 0; i < 5; i++) {
        NeuralNetwork::ReLU(t, t+n, n);
        asm volatile("" ::: "memory");
    }

    // test relu activation
    auto start = std::chrono::high_resolution_clock::now();
    NeuralNetwork::ReLU(t, t+n, n);
    asm volatile("" ::: "memory");
    double dur = (std::chrono::high_resolution_clock::now() - start).count();
    
    double mpe = ComputeMPE(expected, t+n, n);
    free(expected);
    free(t);

    FinalizeTestOutput("ReLU", dur, mpe, acceptederror, n);
}
void TestNetwork::TestLeakyReLU(size_t n) {
    const double acceptederror = 0.000001;
    
    size_t size = ((n*2 + 40) + 31) & ~31;
    float* t = InitializeTestData(size);
    double* expected = (double*)malloc(size * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        expected[i] = t[i] > 0.0 ? t[i] : (double)t[i] * 0.1;
    }

    // warmup runs
    for (size_t i = 0; i < 5; i++) {
        NeuralNetwork::LeakyReLU(t, t+n, n);
        asm volatile("" ::: "memory");
    }

    // test leaky relu
    auto start = std::chrono::high_resolution_clock::now();
    NeuralNetwork::LeakyReLU(t, t+n, n);
    asm volatile("" ::: "memory");
    double dur = (std::chrono::high_resolution_clock::now() - start).count();
    
    double mpe = ComputeMPE(expected, t+n, n);
    free(expected);
    free(t);

    FinalizeTestOutput("Leaky ReLU", dur, mpe, acceptederror, n);
}
void TestNetwork::TestELU(size_t n) {
    const double acceptederror = 0.01;
    
    size_t size = ((n*2 + 40) + 31) & ~31;
    float* t = InitializeTestData(size);
    double* expected = (double*)malloc(size * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        expected[i] = t[i] > 0.0 ? t[i] : (std::exp((double)t[i]) - 1.0);
    }

    // warmup runs
    for (size_t i = 0; i < 5; i++) {
        NeuralNetwork::ELU(t, t+n, n);
        asm volatile("" ::: "memory");
    }

    // test leaky relu
    auto start = std::chrono::high_resolution_clock::now();
    NeuralNetwork::ELU(t, t+n, n);
    asm volatile("" ::: "memory");
    double dur = (std::chrono::high_resolution_clock::now() - start).count();
    
    double mpe = ComputeMPE(expected, t+n, n);
    free(expected);
    free(t);

    FinalizeTestOutput("ELU", dur, mpe, acceptederror, n);
}
void TestNetwork::TestSoftmax(size_t n) {

}
