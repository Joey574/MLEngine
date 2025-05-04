#include "TestNetwork.hpp"

void TestNetwork::TestMathUtils() {
    const size_t ssize = 10000000;
    const size_t esize = 10000000;

    for (size_t i = ssize; i <= esize; i *= 10) {
        std::cout << "Testing Math Utils (" << i << "):\n";
        TestSum(i);
        TestExp(i);
        std::cout << "\n";
    }
}

void TestNetwork::TestExp(size_t n) {
    const double acceptederror = 0.045;

    size_t size = n*8;
    float* t = InitializeTestData(size);
    double* expected = (double*)malloc(size*sizeof(double));

    for (size_t i = 0; i < size; i++) {
        expected[i] = std::exp(t[i]);
    }

    // warmup runs
    for (size_t i = 0; i < 5; i++) {
        NeuralNetwork::Exp256(_mm256_loadu_ps(&t[0]));
        asm volatile("" ::: "memory");
    }

    // test 256 exp
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i <= size-8; i += 8) {
        __m256 _t = NeuralNetwork::Exp256(_mm256_loadu_ps(&t[i]));
        _mm256_storeu_ps(&t[i], _t);

        asm volatile("" ::: "memory");
    }
    double dur = (std::chrono::high_resolution_clock::now() - start).count();

    double mpe = ComputeMPE(expected, t, size);
    free(expected);
    free(t);

    FinalizeTestOutput("Exp256", dur, mpe, acceptederror, size);
}

void TestNetwork::TestSum(size_t n) {
    const double acceptederror = 0.000001;

    size_t size = n*8;
    float* t = InitializeTestData(size);
    double* expected = (double*)malloc((size/8)* sizeof(double));

    for (size_t i = 0; i < (size/8); i++) {
        expected[i] = t[i*8]+t[i*8+1]+t[i*8+2]+t[i*8+3]+t[i*8+4]+t[i*8+5]+t[i*8+6]+t[i*8+7];
    }

    // warmup runs
    for (size_t i = 0; i < 5; i++) {
        NeuralNetwork::Sum256(_mm256_loadu_ps(&t[0]));
        asm volatile("" ::: "memory");
    }

    // test 256 sum
    size_t idx = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i <= size-8; i += 8, idx++) {
        t[idx] = NeuralNetwork::Sum256(_mm256_loadu_ps(&t[i]));
        asm volatile("" ::: "memory");
    }
    double dur = (std::chrono::high_resolution_clock::now() - start).count();

    double mpe = ComputeMPE(expected, t, (size/8));
    free(expected);
    free(t);

    FinalizeTestOutput("Sum256", dur, mpe, acceptederror, (size/8));
}