#include "NeuralNetwork.hpp"

void NeuralNetwork::Sigmoid(float* x, float* y, size_t n) {

}

void NeuralNetwork::ReLU(float* x, float* y, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i <= n - 8; i += 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        _mm256_store_ps(&y[i], _mm256_max_ps(_x, _mm256_setzero_ps()));
    }

    for (size_t i = n - (n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void NeuralNetwork::LeakyReLU(float* x, float* y, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i <= n - 8; i += 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        const __m256 _x2 = _mm256_mul_ps(_x, _mm256_set1_ps(0.1f));
        
        _mm256_store_ps(&y[i], _mm256_max_ps(_x, _x2));
    }

    for (size_t i = n - (n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (x[i] * 0.1f);
    }

}

void NeuralNetwork::ELU(float* x, float* y, size_t n) {

}

void NeuralNetwork::Softmax(float* x, float* y, size_t n) {
    
}