#include "MathUtils.hpp"

__attribute__((target("avx512dq")))
float MathUtils::Sum512(__m512 _x) {
    __m256 low  = _mm512_extractf32x8_ps(_x, 0);
    __m256 high = _mm512_extractf32x8_ps(_x, 1);

    return Sum256(low) + Sum256(high);
}

__attribute__((target("avx512dq")))
__m512 MathUtils::Exp512(__m512 _x) {
    __m512 _a = _mm512_set1_ps(12102203.0f); 
    __m512 _b = _mm512_set1_ps(127.0f * (1 << 23));
    __m512 _c = _mm512_fmadd_ps(_x, _a, _b);

    __m512i _res = _mm512_cvtps_epi32(_c);

    return _mm512_castsi512_ps(_res);
}