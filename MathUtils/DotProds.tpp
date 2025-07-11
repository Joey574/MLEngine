#include "MathUtils.hpp"

#if defined(__AVX512F__)
template <bool clear> void MathUtils::DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	AVX512_VALID_PATH();

	#pragma omp parallel for
    for (size_t i = 0; i < a_r; i++) {		
		const size_t aidx = i*a_c;
		const size_t cidx = i*b_c;

        size_t j = 0;

        // first j loop to clear existing c values
        if constexpr (clear) {
            j = 1;
            const __m512 _a = _mm512_set1_ps(a[aidx+0]);

            size_t k = 0;
            for(; k+16 <= b_c; k += 16) {
                const __m512 _b = _mm512_loadu_ps(&b[0 * b_c + k]);
                const __m512 _c = _mm512_mul_ps(_a, _b);

                _mm512_storeu_ps(&c[cidx+k], _c);
            }
            for(; k < b_c; k++) {
                c[cidx+k] = a[aidx+0] * b[0*b_c+k];
            }
        }

        // main j loop
        for (;j < b_r; j++) {
			const size_t bidx = j*b_c;

            const __m512 _a = _mm512_set1_ps(a[aidx+j]);

            size_t k = 0;
            for (; k+16 <= b_c; k += 16) {
                const __m512 _b = _mm512_loadu_ps(&b[bidx+k]);
                const __m512 _c = _mm512_loadu_ps(&c[cidx+k]);

                const __m512 _res = _mm512_fmadd_ps(_a, _b, _c);

                _mm512_storeu_ps(&c[cidx+k], _res);
            }
            for(; k < b_c; k++) {
                c[cidx+k] += a[aidx+j] * b[bidx+k];
            }
        }
    }
}
#elif defined(__AVX2__) && defined(__FMA__)
template <bool clear> void MathUtils::DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	AVX2_VALID_PATH();

	#pragma omp parallel for
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
#else
template <bool clear> void MathUtils::DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
    SCALAR_VALID_PATH();

    #pragma omp parallel for
    for (size_t i = 0; i < a_r; i++) {		
		const size_t aidx = i*a_c;
		const size_t cidx = i*b_c;

        size_t j = 0;

        // first j loop to clear existing c values
        if constexpr (clear) {
            j = 1;
            
            #pragma omp simd
            for (size_t k = 0; k < b_c; k++) {
                c[cidx+k] = a[aidx+0] * b[0*b_c+k];
            }
        }

        // main j loop
        for (;j < b_r; j++) {
			const size_t bidx = j*b_c;

            #pragma omp simd
            for (size_t k = 0; k < b_c; k++) {
                c[cidx+k] += a[aidx+j] * b[bidx+k];
            }
        }
    }
}
#endif
