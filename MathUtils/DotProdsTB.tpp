#include "MathUtils.hpp"

#if defined(__AVX512F__)
template <bool clear> void MathUtils::DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
    assert(__builtin_cpu_supports("avx512f"));

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

			__m512 sum = _mm512_setzero_ps();
			for (; j+16 <= b_c; j += 16) {
				const __m512 _a = _mm512_loadu_ps(&a[aidx+j]);
				const __m512 _b = _mm512_loadu_ps(&b[bidx+j]);

				sum = _mm512_fmadd_ps(_a, _b, sum);
			}

			c[cidx+k] += Sum512(sum);
			
			for (; j < b_c; j++) {
				c[cidx+k] += a[aidx+j] * b[bidx+j];
			}
		}
    }
}
#elif defined (__AVX2__) && defined(__FMA__)
template <bool clear> void MathUtils::DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));

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
#else
template <bool clear> void MathUtils::DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
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

			for (; j < b_c; j++) {
				c[cidx+k] += a[aidx+j] * b[bidx+j];
			}
		}
    }
}
#endif