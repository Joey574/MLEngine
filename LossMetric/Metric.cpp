#include "LossMetric.hpp"
#include "../NeuralNetwork/NeuralNetwork.hpp"

float LossMetric::MaeScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    const __m256 _absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 _sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i+8 <= rows*cols; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _y = _mm256_loadu_ps(&y[i]);

        const __m256 _e = _mm256_sub_ps(_x, _y);
        const __m256 _res = _mm256_and_ps(_e, _absmask);

        _sum = _mm256_add_ps(_sum, _res);
    }

    float error = NeuralNetwork::Sum256(_sum);
    for (; i < rows*cols; i++) {
        error += std::abs(x[i] - y[i]);
    }

    return error / (float)(rows*cols);
}
float LossMetric::MseScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    __m256 _sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i+8 <= rows*cols; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _y = _mm256_loadu_ps(&y[i]);

        const __m256 _e = _mm256_sub_ps(_x, _y);
        const __m256 _se = _mm256_mul_ps(_e, _e);

        _sum = _mm256_add_ps(_sum, _se);
    }

    float error = NeuralNetwork::Sum256(_sum);
    for (; i < rows*cols; i++) {
        error += (x[i]-y[i])*(x[i]-y[i]);
    }

    return error / (float)(rows*cols);
}
float LossMetric::AccuracyScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    size_t correct = 0;

    #pragma omp parallel for simd
    for(size_t r = 0; r < rows; r++) {
        size_t midx = 0;
        float max = x[r*cols+0];

        for (size_t c = 1; c < cols; c++) {
            if (x[r*cols+c] > max) {
                max = x[r*cols+c];
                midx = c;
            }
        }

        if (midx == y[r]) {
            #pragma omp atomic update
            correct++;
        }
    }

    return ((float)correct / (float)rows) * 100.0f;
}
