#include "MathUtils.hpp"

template <bool clear> void MathUtils::DotProd_AVX(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	assert(__builtin_cpu_supports("avx"));

	#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < a_r; i++) {		
		const size_t aidx = i*a_c;
		const size_t cidx = i*b_c;

        size_t j = 0;

        // first j loop to clear existing c values
        if constexpr (clear) {
            j = 1;
            const __m256 _a = _mm256_set1_ps(a[aidx+0]);

            size_t k = 0;
            for(; k + 8 <= b_c; k += 8) {
                const __m256 _b = _mm256_loadu_ps(&b[0 * b_c + k]);
                const __m256 _c = _mm256_mul_ps(_a, _b);

                _mm256_storeu_ps(&c[cidx+k], _c);
            }
            for(; k < b_c; k++) {
                c[cidx+k] = a[aidx+0] * b[0*b_c+k];
            }
        }

        // main j loop
        for (;j < b_r; j++) {
			const size_t bidx = j*b_c;

            const __m256 _a = _mm256_set1_ps(a[aidx+j]);

            size_t k = 0;
            for (; k + 8 <= b_c; k += 8) {
                const __m256 _b = _mm256_loadu_ps(&b[bidx+k]);
                const __m256 _c = _mm256_loadu_ps(&c[cidx+k]);

                const __m256 _res = _mm256_fmadd_ps(_a, _b, _c);

                _mm256_storeu_ps(&c[cidx+k], _res);
            }
            for(; k < b_c; k++) {
                c[cidx+k] += a[aidx+j] * b[bidx+k];
            }
        }
    }
}

template <bool clear> void MathUtils::DotProdTA_AVX(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	assert(__builtin_cpu_supports("avx"));

	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < a_c; i++) {
		const size_t cidx = i*b_c;
        size_t j = 0;

		// first j loop to clear existing c values
		if constexpr (clear) {
            j = 1;
			const __m256 _a_t = _mm256_set1_ps(a[0 * a_c + i]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_loadu_ps(&b[0 * b_c + k]);
				const __m256 _c = _mm256_mul_ps(_a_t, _b);

				_mm256_storeu_ps(&c[cidx+k], _c);
			}

			for (; k < b_c; k++) {
				c[cidx+k] = a[0*a_c+i] * b[0*b_c+k];
			}
		}

        // main j loop
		for (; j < b_r; j++) {
			const size_t aidx = j*a_c;
			const size_t bidx = j*b_c;

			const __m256 _a_t = _mm256_set1_ps(a[aidx+i]);

			size_t k = 0;
			for (; k + 8 <= b_c; k += 8) {
				const __m256 _b = _mm256_loadu_ps(&b[bidx+k]);
				const __m256 _c = _mm256_loadu_ps(&c[cidx+k]);
				const __m256 _res = _mm256_fmadd_ps(_a_t, _b, _c);

				_mm256_storeu_ps(&c[cidx+k], _res);
			}

			for (; k < b_c; k++) {
				c[cidx+k] += a[aidx+i] * b[bidx+k];
			}
		}
    }
}

template <bool clear> void MathUtils::DotProdTB_AVX(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	assert(__builtin_cpu_supports("avx"));

    #pragma omp parallel for schedule(static)
	for (size_t i = 0; i < a_r; i++) {
		const size_t aidx = i*a_c;
		const size_t cidx = i*b_r;

		for (size_t k = 0; k < b_r; k++) {
			const size_t bidx = k*b_c;
			size_t j = 0;

			if constexpr (clear) {
				j = 1;
				c[cidx+k] = a[aidx+0] * b[bidx+0];
			}

			__m256 sum = _mm256_setzero_ps();
			for (; j + 8 <= b_c; j += 8) {
				const __m256 _a = _mm256_loadu_ps(&a[aidx+j]);
				const __m256 _b = _mm256_loadu_ps(&b[bidx+j]);

				sum = _mm256_fmadd_ps(_a, _b, sum);
			}

			c[cidx+k] += Sum256(sum);
			
			for (; j < b_c; j++) {
				c[cidx+k] += a[aidx+j] * b[bidx+j];
			}
		}
    }
}