#pragma once
#include "../NeuralNetwork/NeuralNetwork.hpp"

class TestNetwork {
public:

    // activation function tests
    static void TestSigmoid(size_t n);
    static void TestReLU(size_t n);
    static void TestLeakyReLU(size_t n);
    static void TestELU(size_t n);
    static void TestSoftmax(size_t n);
    static void TestActivations();

    // activation derivatives tests
    static void TestSigmoidD(size_t n);
    static void TestReLUD(size_t n);
    static void TestLeakyReLUD(size_t n);
    static void TestELUD(size_t n);
    static void TestDerivatives();

    // math util tests
    static void TestExp(size_t n);
    static void TestSum(size_t n);
    static void TestMathUtils();

    // metric tests
    static void TestAccuracyScore(size_t n);
    static void TestMetrics();

    // loss tests
    static void TestOneHotLoss(size_t n);
    static void TestLoss();

private:
    static const char* red;
    static const char* green;
    static const char* yellow;
    static const char* endcolor;

    static float* InitializeTestData(size_t n);

    static void FinalizeTestOutput(const std::string& name, double dur, double mpe, double ae, size_t n);

    static double ComputeMPE(double* truth, float* predicted, size_t n);
};
