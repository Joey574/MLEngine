#pragma once

struct MathUtils {
public:
    using DotProdFunc = void (*)(const float*, const float*, float*, size_t, size_t, size_t, size_t);

    template <bool clear> static void DotProd(const float* a, const float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTA(const float* a, const float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTB(const float* a, const float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

    // math utils
    static float Sum256(__m256 _x);
    static float Sum512(__m512 _x);
    static __m256 Exp256(__m256 _x);
    static __m512 Exp512(__m512 _x);

    /// @brief only works with powers of 2
    static inline size_t RoundTo(size_t alignment, size_t n) {
        alignment--;
        return (n+alignment) & ~alignment;
    }
};