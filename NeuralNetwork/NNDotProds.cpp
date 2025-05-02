#include "NeuralNetwork.hpp"

/// @brief Computes the dot prod between a and b and stores in c,
/// @brief if clear is passed data already in c will be cleared during computation
void NeuralNetwork::DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {

	#pragma omp parallel for
    for (size_t i = 0; i < a_r; i++) {
        size_t j = 0;

        // first j loop to clear existing c values
        if (clear) {
            j = 1;
            const __m256 _a = _mm256_set1_ps(a[i * a_c + 0]);

            size_t k = 0;
            for(; k + 8 <= b_c; k += 8) {
                const __m256 _b = _mm256_loadu_ps(&b[0 * b_c + k]);
                const __m256 _c = _mm256_mul_ps(_a, _b);

                _mm256_storeu_ps(&c[i * b_c + k], _c);
            }
            for(; k < b_c; k++) {
                c[i * b_c + k] = a[i * a_c + 0] * b[0 * b_c + k];
            }
        }

        // main j loop
        for (;j < b_r; j++) {
            const __m256 _a = _mm256_set1_ps(a[i * a_c + j]);

            size_t k = 0;
            for (; k + 8 <= b_c; k += 8) {
                const __m256 _b = _mm256_loadu_ps(&b[j * b_c + k]);
                const __m256 _c = _mm256_loadu_ps(&c[i * b_c + k]);

                const __m256 _res = _mm256_fmadd_ps(_a, _b, _c);

                _mm256_storeu_ps(&c[i * b_c + k], _res);
            }
            for(; k < b_c; k++) {
                c[i * b_c + k] += a[i * a_c + j] * b[j * b_c + k];
            }
        }
    }
}

/// @brief Computes the dot prod between a transpose and b and stores in c,
/// @brief a will be transposed during computation and will not be modified,
/// @brief if clear is passed data already in c will be cleared during computation
void NeuralNetwork::DotProdTA(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {

	#pragma omp parallel for
	for (size_t i = 0; i < a_c; i++) {
        size_t j = 0;

		// first j loop to clear existing c values
		if (clear) {
            j = 1;
			const __m256 _a_t = _mm256_set1_ps(a[0 * a_c + i]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_loadu_ps(&b[0 * b_c + k]);
				const __m256 _c = _mm256_mul_ps(_a_t, _b);

				_mm256_storeu_ps(&c[i * b_c + k], _c);
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] = a[0 * a_c + i] * b[0 * b_c + k];
			}
		}

        // main j loop
		for (; j < b_r; j++) {
			const __m256 _a_t = _mm256_set1_ps(a[j * a_c + i]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_loadu_ps(&b[j * b_c + k]);
				const __m256 _c = _mm256_loadu_ps(&c[i * b_c + k]);
				const __m256 _res = _mm256_fmadd_ps(_a_t, _b, _c);

				_mm256_storeu_ps(&c[i * b_c + k], _res);
			}

			for (; k < b_c; k++) {
				c[i * b_c + k] += a[j * a_c + i] * b[j * b_c + k];
			}
		}
    }
}

/// @brief Computes the dot prod between a and b transpose and stores in c,
/// @brief b will be transposed during computation and will not be modified,
/// @brief if clear is passed data already in c will be cleared during computation
void NeuralNetwork::DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c, bool clear) {

	#pragma omp parallel for
	for (size_t i = 0; i < a_r; i++) {
		for (size_t k = 0; k < b_r; k++) {
			size_t j = clear ? 1 : 0;

			if (clear) {
				c[i * b_r + k] = a[i * a_c + 0] * b[k * b_c + 0];
			}

			__m256 sum = _mm256_setzero_ps();
			for (; j + 8 <= b_c; j += 8) {
				const __m256 _a = _mm256_loadu_ps(&a[i * a_c + j]);
				const __m256 _b = _mm256_loadu_ps(&b[k * b_c + j]);

				sum = _mm256_fmadd_ps(_a, _b, sum);
			}

			c[i * b_r + k] += Sum256(sum);
			
			for (; j < b_c; j++) {
				c[i * b_r + k] += a[i * a_c + j] * b[k * b_c + j];
			}
		}
    }
}
