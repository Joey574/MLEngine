#pragma once

struct MathUtils {
public:

    using DotProdFn = void(*)(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);

    template <bool acum> static void DotProd(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);
    template <bool acum> static void DotProdTA(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);
    template <bool acum> static void DotProdTB(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);

    static void ScaleBy(float* a, float scalar, size_t n);
    static void Copy(const float* src, float* dest, size_t n);
    static float Sum(const float* a, size_t n);
};
