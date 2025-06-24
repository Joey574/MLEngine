#include "MathUtils.hpp"

template <bool clear> void MathUtils::DotProd_Scalar(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < a_r; i++) {		
		const size_t aidx = i*a_c;
		const size_t cidx = i*b_c;

        size_t j = 0;

        // first j loop to clear existing c values
        if constexpr (clear) {
            j = 1;
            
            for(size_t k = 0; k < b_c; k++) {
                c[cidx+k] = a[aidx+0] * b[0*b_c+k];
            }
        }

        // main j loop
        for (;j < b_r; j++) {
			const size_t bidx = j*b_c;

            for(size_t k = 0; k < b_c; k++) {
                c[cidx+k] += a[aidx+j] * b[bidx+k];
            }
        }
    }
}

template <bool clear> void MathUtils::DotProdTA_Scalar(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
    #pragma omp parallel for schedule(static)
	for (size_t i = 0; i < a_c; i++) {
		const size_t cidx = i*b_c;
        size_t j = 0;

		// first j loop to clear existing c values
		if constexpr (clear) {
            j = 1;
			
			for (size_t k = 0; k < b_c; k++) {
				c[cidx+k] = a[0*a_c+i] * b[0*b_c+k];
			}
		}

        // main j loop
		for (; j < b_r; j++) {
			const size_t aidx = j*a_c;
			const size_t bidx = j*b_c;

			for (size_t k = 0; k < b_c; k++) {
				c[cidx+k] += a[aidx+i] * b[bidx+k];
			}
		}
    }
}

template <bool clear> void MathUtils::DotProdTB_Scalar(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
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
