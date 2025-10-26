#pragma once

struct MathUtils {
public:

    static void DotProd(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);
    static void DotProdTA(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);
    static void DotProdTB(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);

    static void DotProdAdd(const float* a, const float* b, const float* c, float* d);

    static void ScaleBy(float* a, const float* b, size_t n);
    static void ScaleBy(float* a, float b, size_t n);

    static void Copy(const float* src, float* dest, size_t n);

private:
};