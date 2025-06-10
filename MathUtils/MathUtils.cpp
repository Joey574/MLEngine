#include "MathUtils.hpp"

float MathUtils::DotProdConv(const float* __restrict a, float* __restrict b, size_t a_r, size_t a_c, size_t bsize, size_t roffset, size_t coffset) {
    assert((uintptr_t)a%32==0);
    assert((uintptr_t)b%32==0);

    float sum = 0.0f;
    size_t hsize = (bsize+1)/2;
    size_t rad = bsize/2;

    #pragma omp parallel for simd collapse(2) reduction(+:sum)
    for (size_t i = 0; i < bsize; i++) {
        for (size_t j = 0; j < bsize; j++) {
            size_t r = roffset+i-rad;
            size_t c = coffset+j-rad;

            sum += a[r*a_c+c] * b[i*bsize+j];
        }
    }

    return sum;
}