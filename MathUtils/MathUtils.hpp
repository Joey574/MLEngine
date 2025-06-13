#pragma once
#include "../Activation/Activation.hpp"


struct MathUtils {
public:
    using DotProdActvP = void (*)(const float*, const float*, float*, float*, size_t, size_t, size_t, size_t);

    // dot prods
    template <bool> static void DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool> static void DotProdTA(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool> static void DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

    template <bool> static DotProdActvP DotProdActvPtr(Activation::Type type);
    template <bool> static void DotProdActv(Activation::Type type, const float* __restrict a, const float* __restrict b, float* __restrict c, float* __restrict d, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool> static void DotProdTBDerv(Activation::Type type, const float* __restrict a, const float* __restrict b, float* __restrict c, const float* __restrict d, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

    static float DotProdConv(const float* __restrict a, float* __restrict b, size_t a_r, size_t a_c, size_t bsize, size_t roffset, size_t coffset);

    // math utils
    inline static float Sum256(__m256 _x) {
        __m256 _sum1 = _mm256_hadd_ps(_x, _x);
        __m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

        __m128 _low  = _mm256_castps256_ps128(_sum2);
        __m128 _high = _mm256_extractf128_ps(_sum2, 1);
        __m128 _res  = _mm_add_ps(_low, _high);

        return _mm_cvtss_f32(_res);
    }
    inline static __m256 Exp256(__m256 _x) {
        __m256 _a = _mm256_set1_ps(12102203.0f); 
        __m256 _b = _mm256_set1_ps(127.0f * (1 << 23));
        __m256 _c = _mm256_fmadd_ps(_x, _a, _b);

        __m256i _res = _mm256_cvtps_epi32(_c);

        return _mm256_castsi256_ps(_res);
    }

private:
    template <bool, Activation::Type> static void DotProdActv(const float* __restrict a, const float* __restrict b, float* __restrict c, float* __restrict d, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool, Activation::Type> static void DotProdTBDerv(const float* __restrict a, const float* __restrict b, float* __restrict c, const float* __restrict d, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

    template <Activation::Type type> static inline void ApplyActv(float* a, const __m256 _x) {
        if constexpr (type == Activation::Type::linear) {
			_mm256_storeu_ps(a, Activation::Linear(_x));
		} else if constexpr (type == Activation::Type::sigmoid) {
			_mm256_storeu_ps(a, Activation::Sigmoid(_x));
		} else if constexpr (type == Activation::Type::relu) {
			_mm256_storeu_ps(a, Activation::ReLU(_x));
		} else if constexpr (type == Activation::Type::leakyrelu) {
			_mm256_storeu_ps(a, Activation::LeakyReLU(_x));
		} else if constexpr (type == Activation::Type::elu) {
			_mm256_storeu_ps(a, Activation::ELU(_x));
		} else {
			_mm256_storeu_ps(a, _x);
		}
    }
    template <Activation::Type type> static inline void ApplyActv(float* a, float b) {
        if constexpr (type == Activation::Type::linear) {
			a[0] = Activation::Linear(b);
		} else if constexpr (type == Activation::Type::sigmoid) {
			a[0] = Activation::Sigmoid(b);
		} else if constexpr (type == Activation::Type::relu) {
			a[0] = Activation::ReLU(b);
		} else if constexpr (type == Activation::Type::leakyrelu) {
			a[0] = Activation::LeakyReLU(b);
		} else if constexpr (type == Activation::Type::elu) {
			a[0] = Activation::ELU(b);
		} else {
			a[0] = b;
		}
    }

};

#include "DotProds.impl.hpp"