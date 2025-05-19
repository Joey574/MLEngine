#include "NeuralNetwork.hpp"

void NeuralNetwork::MaeLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    #if LOGLOSS
        printf("Loss applied [%zu x %zu]\n", rows, cols);
    #endif

    const __m256 _zero = _mm256_setzero_ps();
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _none = _mm256_set1_ps(-1.0f);

    #pragma omp parallel for
    for (size_t i = 0; i <= rows*cols-8; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _y = _mm256_loadu_ps(&y[i]);

        const __m256 _diff = _mm256_sub_ps(_x, _y);
        const __m256 _cmp = _mm256_cmp_ps(_diff, _zero, _CMP_GT_OQ);
        const __m256 _res = _mm256_blendv_ps(_none, _one, _cmp);

        _mm256_storeu_ps(&c[i], _res);
    }

    for (size_t i = (rows*cols)-((rows*cols)%8); i < rows*cols; i++) {
        c[i] = (x[i] - y[i]) > 0.0f ? 1.0f : -1.0f;
    }

}
void NeuralNetwork::MseLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    #if LOGLOSS
        printf("Loss applied [%zu x %zu]\n", rows, cols);
    #endif

    const __m256 _two = _mm256_set1_ps(2.0f);

    #pragma omp parallel for
    for (size_t i = 0; i <= rows*cols-8; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _y = _mm256_loadu_ps(&y[i]);

        const __m256 _x2 = _mm256_sub_ps(_x, _y);
        const __m256 _res = _mm256_mul_ps(_two, _x2);
        _mm256_storeu_ps(&c[i], _res);
    }

    for (size_t i = (rows*cols)-((rows*cols)%8); i < rows*cols; i++) {
        c[i] = 2.0f * (x[i] - y[i]);
    }
}
void NeuralNetwork::OneHotLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    #if LOGLOSS
        printf("Loss applied [%zu x %zu]\n", rows, cols);
    #endif

    FastCopy(x, c, rows*cols);

    #pragma omp parallel for
    for (size_t i = 0; i < cols; i++) {
        c[(int)y[i]*cols+i]--;
    }
}

float NeuralNetwork::MaeScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    const __m256 _absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 _sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i <= rows*cols-8; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _y = _mm256_loadu_ps(&y[i]);

        const __m256 _e = _mm256_sub_ps(_x, _y);
        const __m256 _res = _mm256_and_ps(_e, _absmask);

        _sum = _mm256_add_ps(_sum, _res);
    }

    float error = Sum256(_sum);
    for (; i < rows*cols; i++) {
        error += std::abs(x[i] - y[i]);
    }

    return error / (float)(rows*cols);
}
float NeuralNetwork::MseScore(const float* __restrict x, const float* __restrict y, size_t rows, size_t cols) {
    __m256 _sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i <= rows*cols-8; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _y = _mm256_loadu_ps(&y[i]);

        const __m256 _e = _mm256_sub_ps(_x, _y);
        const __m256 _se = _mm256_mul_ps(_e, _e);

        _sum = _mm256_add_ps(_sum, _se);
    }

    float error = Sum256(_sum);
    for (; i < rows*cols; i++) {
        error += (x[i]-y[i])*(x[i]-y[i]);
    }

    return error / (float)(rows*cols);
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
