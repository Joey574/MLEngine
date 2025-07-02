#include "MathUtils.hpp"

__attribute__((target("avx512f")))
float MathUtils::Sum512(__m512 _x) {
    return _mm512_reduce_add_ps(_x);
}

__attribute__((target("avx512f")))
__m512 MathUtils::Exp512(__m512 _x) {
    __m512 _a = _mm512_set1_ps(12102203.0f); 
    __m512 _b = _mm512_set1_ps(127.0f * (1 << 23));
    __m512 _c = _mm512_fmadd_ps(_x, _a, _b);

    __m512i _res = _mm512_cvtps_epi32(_c);

    return _mm512_castsi512_ps(_res);
}