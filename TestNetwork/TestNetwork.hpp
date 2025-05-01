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

private:
    static const char* red;
    static const char* green;
    static const char* yellow;
    static const char* endcolor;

    static float* InitializeActivationTest(size_t n);
    static void FinalizeActivationTest(const std::string& name, double dur, double mpe, double ae, size_t n);

    static double ComputeMPE(double* truth, float* predicted, size_t n);
};
