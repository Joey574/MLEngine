#include "NeuralNetwork.hpp"

void NeuralNetwork::MaeLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {

}
void NeuralNetwork::MseLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {

}
void NeuralNetwork::OneHotLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    #if LOGLOSS
        printf("Loss applied [%zu x %zu]\n", rows, cols);
    #endif

    std::memcpy(c, x, rows*cols*sizeof(float));

    for (size_t i = 0; i < cols; i++) {
        c[(int)y[i]*cols+i]--;
    }
}

float NeuralNetwork::MaeScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    const __m256 _absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 _sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= rows*cols; i += 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        const __m256 _y = _mm256_load_ps(&y[i]);

        const __m256 _e = _mm256_sub_ps(_x, _y);
        const __m256 _res = _mm256_andnot_ps(_e, _absmask);

        _sum = _mm256_add_ps(_sum, _res);
    }

    float error = Sum256(_sum);
    for (; i < rows*cols; i++) {
        error += std::abs(x - y);
    }

    return error / ((float)rows*(float)cols);
}
float NeuralNetwork::MseScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    __m256 _sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i+8 <= rows*cols; i += 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        const __m256 _y = _mm256_load_ps(&y[i]);

        const __m256 _e = _mm256_sub_ps(_x, _y);
        const __m256 _se = _mm256_mul_ps(_e, _e);

        _sum = _mm256_add_ps(_sum, _se);
    }

    float error = Sum256(_sum);
    for (; i < rows*cols; i++) {
        error += std::pow(x - y, 2);
    }

    return error / ((float)rows*(float)cols);
}
float NeuralNetwork::AccuracyScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    size_t correct = 0;

    for (size_t c = 0; c < cols; c++) {
        size_t midx = 0;
        float max = x[0*cols+c];

        for (size_t r = 1; r < rows; r++) {
            if (x[r*cols+c] > max) {
                max = x[r*cols+c];
                midx = r;
            }
        }

        if (midx == y[c]) {
            correct++;
        }
    }

    #if LOGSCORE
        printf("Model scored [%zu x %zu] (%zu/%zu)\n", rows, cols, correct, cols);
    #endif

    return ((float)correct / (float)cols) * 100.0f;
}
