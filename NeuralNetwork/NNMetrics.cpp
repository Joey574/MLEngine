#include "NeuralNetwork.hpp"

void NeuralNetwork::MaeLoss(float* x, float* y, float* c, size_t rows, size_t cols) {

}
void NeuralNetwork::MseLoss(float* x, float* y, float* c, size_t rows, size_t cols) {

}
void NeuralNetwork::OneHotLoss(float* x, float* y, float* c, size_t rows, size_t cols) {

}

float NeuralNetwork::MaeScore(float* x, float* y, size_t rows, size_t cols) {
    const __m256 _mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 _sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= rows*cols; i += 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        const __m256 _y = _mm256_load_ps(&y[i]);
        const __m256 _res = _mm256_andnot_ps(_mm256_sub_ps(_x, _y), _mask);

        _sum = _mm256_add_ps(_sum, _res);
    }

    float error = Sum256(_sum);
    for (; i < rows*cols; i++) {
        error += std::abs(x - y);
    }

    return error;
}
float NeuralNetwork::MseScore(float* x, float* y, size_t rows, size_t cols) {
    float error = 0.0f;

    for (size_t i = 0; i < rows*cols; i++) {
        error += std::pow(x - y, 2);
    }

    return error;
}
float NeuralNetwork::AccuracyScore(float* x, float* y, size_t rows, size_t cols) {
    return 0.0f;
}