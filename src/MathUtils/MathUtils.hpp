#pragma once
#include "../Tensor/Tensor.hpp"

struct MathUtils {
    public:
    // CPU Math Utilities
    template <bool acum> static void DotProd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t ar, size_t ac, size_t br, size_t bc);
    template <bool acum> static void DotProdTA(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t ar, size_t ac, size_t br, size_t bc);
    template <bool acum> static void DotProdTB(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t ar, size_t ac, size_t br, size_t bc);

    static void ScaleBy(float* __restrict a, float scalar, size_t n);
    static void ScaleBy(const float* __restrict a, float* __restrict b, size_t n);
    static void Copy(const float* __restrict src, float* __restrict dest, size_t n);
    static float Sum(const float* __restrict a, size_t n);
    static void SumColumns(const float* __restrict a, float* __restrict b, size_t ar, size_t ac);

    // Utils for getting data stuff, temporary for now
    static inline void* Allocate(size_t bytes) {
        void* p = aligned_alloc(32, bytes);
        std::memset(p, 0, bytes);
        return p;
    }
};
