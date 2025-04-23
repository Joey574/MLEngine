#include "NeuralNetwork.hpp"

void NeuralNetwork::SigmoidDerivative(float* x, float* y, size_t n) {

}

void NeuralNetwork::ReLUDerivative(float* x, float* y, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? y[i] : 0.0f;
    }
}

void NeuralNetwork::LeakyReLUDerivative(float* x, float* y, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? y[i] : (y[i] * 0.1f);
    }
}

void NeuralNetwork::ELUDerivative(float* x, float* y, size_t n) {
    
}