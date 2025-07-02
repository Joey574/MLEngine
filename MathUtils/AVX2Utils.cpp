#include "MathUtils.hpp"

__attribute__((target("avx2,fma")))
float MathUtils::Sum256(__m256 _x) {
    __m256 _sum1 = _mm256_hadd_ps(_x, _x);
    __m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

    __m128 _low  = _mm256_castps256_ps128(_sum2);
    __m128 _high = _mm256_extractf128_ps(_sum2, 1);
    __m128 _res  = _mm_add_ps(_low, _high);

    return _mm_cvtss_f32(_res);
}

__attribute__((target("avx2,fma")))
__m256 MathUtils::Exp256(__m256 _x) {
    __m256 _a = _mm256_set1_ps(12102203.0f); 
    __m256 _b = _mm256_set1_ps(127.0f * (1 << 23));
    __m256 _c = _mm256_fmadd_ps(_x, _a, _b);

    __m256i _res = _mm256_cvtps_epi32(_c);
    return _mm256_castsi256_ps(_res);
}
