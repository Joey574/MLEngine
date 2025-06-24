#pragma once

struct MathUtils {
public:
    using DotProdFunc = void (*)(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);

    static DotProdFunc DotProdAcum;
    static DotProdFunc DotProdClear;

    static DotProdFunc DotProdTAAcum;
    static DotProdFunc DotProdTAClear;

    static DotProdFunc DotProdTBAcum;
    static DotProdFunc DotProdTBClear;


    static void Initialize() {
        if (__builtin_cpu_supports("avx512dq")) {
            DotProdAcum = &DotProd_AVX512<false>;
            DotProdTAAcum = &DotProdTA_AVX512<false>;
            DotProdTBAcum = &DotProdTB_AVX512<false>;

            DotProdClear = &DotProd_AVX512<true>;
            DotProdTAClear = &DotProdTA_AVX512<true>;
            DotProdTBClear = &DotProdTB_AVX512<true>;
        } else if (__builtin_cpu_supports("avx2")) {
            DotProdAcum = &DotProd_AVX2<false>;
            DotProdTAAcum = &DotProdTA_AVX2<false>;
            DotProdTBAcum = &DotProdTB_AVX2<false>;

            DotProdClear = &DotProd_AVX2<true>;
            DotProdTAClear = &DotProdTA_AVX2<true>;
            DotProdTBClear = &DotProdTB_AVX2<true>;
        } else if (__builtin_cpu_supports("avx")) {
            DotProdAcum = &DotProd_AVX<false>;
            DotProdTAAcum = &DotProdTA_AVX<false>;
            DotProdTBAcum = &DotProdTB_AVX<false>;

            DotProdClear = &DotProd_AVX<true>;
            DotProdTAClear = &DotProdTA_AVX<true>;
            DotProdTBClear = &DotProdTB_AVX<true>;            
        } else {
            DotProdAcum = &DotProd_Scalar<false>;
            DotProdTAAcum = &DotProdTA_Scalar<false>;
            DotProdTBAcum = &DotProdTB_Scalar<false>;

            DotProdClear = &DotProd_Scalar<true>;
            DotProdTAClear = &DotProdTA_Scalar<true>;
            DotProdTBClear = &DotProdTB_Scalar<true>;
        }
    }
    
    // math utils
    static float Sum256(__m256 _x);
    static float Sum512(__m512 _x);
    static __m256 Exp256(__m256 _x);
    static __m512 Exp512(__m512 _x);

private:

    template <bool clear> static void DotProd_Scalar(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProd_AVX(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProd_AVX2(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProd_AVX512(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

    template <bool clear> static void DotProdTA_Scalar(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTA_AVX(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTA_AVX2(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTA_AVX512(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

    template <bool clear> static void DotProdTB_Scalar(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTB_AVX(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTB_AVX2(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTB_AVX512(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

};