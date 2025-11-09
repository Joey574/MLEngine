#pragma once

struct MathUtils {
    public:

    /* ----------
    Math utilities
    ---------- */
    template <bool acum> static void DotProd(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c);
    template <bool acum> static void DotProdTA(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c);
    template <bool acum> static void DotProdTB(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c);

    static void Copy(const Tensor<float>& src, Tensor<float>& dest);
    static void CopyByRow(const Tensor<float>& src, Tensor<float>& dest);

    static float Sum(const Tensor<float>& a);
    template <bool acum> static void SumColumns(const Tensor<float>& a, Tensor<float>& b);
};
