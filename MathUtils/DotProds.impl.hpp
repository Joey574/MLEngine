#pragma once
#include "MathUtils.hpp"

template <bool clear> void MathUtils::DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	#if LOGDP
		printf("[%zu x %zu] * [%zu x %zu] = [%zu x %zu]\n", a_r, a_c, b_r, b_c, a_r, b_c);
	#endif

	assert((uintptr_t)a%32==0);
	assert((uintptr_t)b%32==0);
	assert((uintptr_t)c%32==0);

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
template <bool clear> void MathUtils::DotProdTA(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	#if LOGDP
		printf("[%zu x %zu].T * [%zu x %zu] = [%zu x %zu]\n", a_r, a_c, b_r, b_c, a_c, b_c);
	#endif

	assert((uintptr_t)a%32==0);
	assert((uintptr_t)b%32==0);
	assert((uintptr_t)c%32==0);

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
template <bool clear> void MathUtils::DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	#if LOGDP
		printf("[%zu x %zu] * [%zu x %zu].T = [%zu x %zu]\n", a_r, a_c, b_r, b_c, a_r, b_r);
	#endif

	assert((uintptr_t)a%32==0);
	assert((uintptr_t)b%32==0);
	assert((uintptr_t)c%32==0);

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

template <bool clear> MathUtils::DotProdActvP MathUtils::DotProdActvPtr(Activation::Type type) {
	switch (type) {
		case Activation::Type::linear:
			return DotProdActv<clear, Activation::Type::linear>;
		case Activation::Type::sigmoid:
			return DotProdActv<clear, Activation::Type::sigmoid>;
		case Activation::Type::relu:
			return DotProdActv<clear, Activation::Type::relu>;
		case Activation::Type::leakyrelu:
			return DotProdActv<clear, Activation::Type::leakyrelu>;
		case Activation::Type::elu:
			return DotProdActv<clear, Activation::Type::elu>;
		default:
			return DotProdActv<clear, Activation::Type::none>;
	}
}
template <bool clear> void MathUtils::DotProdActv(Activation::Type type, const float* __restrict a, const float* __restrict b, float* __restrict c, float* __restrict d, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	switch (type) {
		case Activation::Type::linear:
			DotProdActv<clear, Activation::Type::linear>(a, b, c, d, a_r, a_c, b_r, b_c);
			break;
		case Activation::Type::sigmoid:
			DotProdActv<clear, Activation::Type::sigmoid>(a, b, c, d, a_r, a_c, b_r, b_c);
			break;
		case Activation::Type::relu:
			DotProdActv<clear, Activation::Type::relu>(a, b, c, d, a_r, a_c, b_r, b_c);
			break;
		case Activation::Type::leakyrelu:
			DotProdActv<clear, Activation::Type::leakyrelu>(a, b, c, d, a_r, a_c, b_r, b_c);
			break;
		case Activation::Type::elu:
			DotProdActv<clear, Activation::Type::elu>(a, b, c, d, a_r, a_c, b_r, b_c);
			break;
		default:
			DotProdActv<clear, Activation::Type::none>(a, b, c, d, a_r, a_c, b_r, b_c);
	}
}
template <bool clear, Activation::Type type> void MathUtils::DotProdActv(const float* __restrict a, const float* __restrict b, float* __restrict c, float* __restrict d, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	
	#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < a_r; i++) {
        size_t j = 0;

		const size_t aidx = i*a_c;
		const size_t cidx = i*b_c;

        // first j loop to clear existing c values
        if constexpr (clear) {
            j = 1;
            const __m256 _a = _mm256_set1_ps(a[aidx+0]);

            size_t k = 0;
            for(; k + 8 <= b_c; k += 8) {
                const __m256 _b = _mm256_loadu_ps(&b[0*b_c+k]);
                const __m256 _c = _mm256_mul_ps(_a, _b);

                _mm256_storeu_ps(&c[i*b_c+ k], _c);
            }
            for(; k < b_c; k++) {
                c[i*b_c+k] = a[i*a_c+0] * b[0*b_c+k];
            }
        }

        // main j loop
        for (;j < b_r-1; j++) {
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

		// last j loop to store activation
		j = b_r-1;
		const size_t bidx = j*b_c;
        const __m256 _a = _mm256_set1_ps(a[aidx+j]);

        size_t k = 0;
        for (; k + 8 <= b_c; k += 8) {
            const __m256 _b = _mm256_loadu_ps(&b[bidx+k]);
            const __m256 _c = _mm256_loadu_ps(&c[cidx+k]);

            const __m256 _res = _mm256_fmadd_ps(_a, _b, _c);

            _mm256_storeu_ps(&c[cidx+k], _res);
			ApplyActv<type>(&d[cidx+k], _res);
        }

        for(; k < b_c; k++) {
            c[cidx+k] += a[aidx+j] * b[bidx+k];
			ApplyActv<type>(&d[cidx+k], c[cidx+k]);
        }
    }
}

template <bool clear> void MathUtils::DotProdTBDerv(Activation::Type type, const float* __restrict a, const float* __restrict b, float* __restrict c, const float* __restrict d, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	switch (type) {
		case Activation::Type::linear:
			DotProdTBDerv<clear, Activation::Type::linear>(a, b, c, d, a_r, a_c, b_r, b_c);
			break;
		case Activation::Type::sigmoid:
			DotProdTBDerv<clear, Activation::Type::sigmoid>(a, b, c, d, a_r, a_c, b_r, b_c);
			break;
		case Activation::Type::relu:
			DotProdTBDerv<clear, Activation::Type::relu>(a, b, c, d, a_r, a_c, b_r, b_c);
			break;
		case Activation::Type::leakyrelu:
			DotProdTBDerv<clear, Activation::Type::leakyrelu>(a, b, c, d, a_r, a_c, b_r, b_c);
			break;
		case Activation::Type::elu:
			DotProdTBDerv<clear, Activation::Type::elu>(a, b, c, d, a_r, a_c, b_r, b_c);
			break;
		default:
			DotProdTBDerv<clear, Activation::Type::none>(a, b, c, d, a_r, a_c, b_r, b_c);
	}
}
template <bool clear, Activation::Type type> void MathUtils::DotProdTBDerv(const float* __restrict a, const float* __restrict b, float* __restrict c, const float* __restrict d, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
	
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < a_r; i++) {
		for (size_t k = 0; k < b_r; k++) {
			size_t j = 0;

			if constexpr (clear) {
				j = 1;
				c[i*b_r+k] = a[i*a_c+0] * b[k*b_c+0];
			}

			__m256 sum = _mm256_setzero_ps();
			for (; j + 8 <= b_c; j += 8) {
				const __m256 _a = _mm256_loadu_ps(&a[i * a_c + j]);
				const __m256 _b = _mm256_loadu_ps(&b[k * b_c + j]);

				sum = _mm256_fmadd_ps(_a, _b, sum);
			}

			c[i*b_r+k] += Sum256(sum);
			
			for (; j < b_c; j++) {
				c[i*b_r+k] += a[i*a_c+j] * b[k* b_c+j];
			}

			if constexpr (type == Activation::Type::linear) {
				c[i*b_r+k] = Activation::LinearDerivative(d[i*b_r+k], c[i*b_r+k]);
			} else if constexpr (type == Activation::Type::sigmoid) {
				c[i*b_r+k] = Activation::SigmoidDerivative(d[i*b_r+k], c[i*b_r+k]);
			} else if constexpr (type == Activation::Type::relu) {
				c[i*b_r+k] = Activation::ReLUDerivative(d[i*b_r+k], c[i*b_r+k]);
			} else if constexpr (type == Activation::Type::leakyrelu) {
				c[i*b_r+k] = Activation::LeakyReLUDerivative(d[i*b_r+k], c[i*b_r+k]);
			} else if constexpr (type == Activation::Type::elu) {
				c[i*b_r+k] = Activation::ELUDerivative(d[i*b_r+k], c[i*b_r+k]);
			}
		}
    }
}
