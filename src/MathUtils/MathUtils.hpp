#pragma once

/// @brief Provides various high performance math utilities for Tensors
struct MathUtils {
    public:

    /* ----------
    Math utilities
    ---------- */
    template <bool acum> static inline void DotProd(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) { Sgemm<acum, false, false>(a, b, c); }
    template <bool acum> static inline void DotProdTA(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) { Sgemm<acum, true, false>(a, b, c);  }
    template <bool acum> static inline void DotProdTB(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) { Sgemm<acum, false, true>(a, b, c); }

    template <bool acum, bool transA, bool transB> static inline void Sgemm(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
        assert(a.Dimensionality() == 2 && b.Dimensionality() == 2 && c.Dimensionality() == 2);
        assert(!a.IsEmpty() && !b.IsEmpty() && !c.IsEmpty());
        assert(!a.HasNan() && !b.HasNan());

        const auto aDims = a.Dimensions();
        const auto bDims = b.Dimensions();
        const auto cDims = c.Dimensions();

        const size_t ar = aDims[0];
        const size_t ac = aDims[1];
        const size_t br = bDims[0];
        const size_t bc = bDims[1];
        const size_t cr = cDims[0];
        const size_t cc = cDims[1];

        const size_t M = transA ? ac : ar;
        const size_t N = transB ? br : bc;
        const size_t K = transA ? ar : ac;
        assert(K == (transB ? bc : br));
        assert(cr == M && cc == N);

        constexpr const float beta = acum ? 1.0f : 0.0f;
        constexpr const CBLAS_TRANSPOSE aTranspose = transA ? CblasTrans : CblasNoTrans;
        constexpr const CBLAS_TRANSPOSE bTranspose = transB ? CblasTrans : CblasNoTrans;

        cblas_sgemm(
            CblasRowMajor, aTranspose, bTranspose,
            M, N, K,
            1.0f, a.Data(), ac, b.Data(), bc,
            beta, c.Data(), cc
        );
    }

    static inline void Copy(const Tensor<float>& src, Tensor<float>& dest) {
        assert(src.Data() != nullptr && dest.Data() != nullptr);
        assert(src.Size() == dest.Size());
        assert(!src.HasNan());

        cblas_scopy(src.Size(), src.Data(), 1, dest.Data(), 1);
    }
    template <bool clear> static inline void PartialCopy(const Tensor<float>& src, Tensor<float>& dest) {
        assert(src.Data() != nullptr && dest.Data() != nullptr);
        assert(dest.Size() >= src.Size());
        assert(!src.HasNan());

        const size_t srcSize = src.Size();
        cblas_scopy(srcSize, src.Data(), 1, dest.Data(), 1);

        // zero out remaining elements
        if constexpr (clear) {
            const size_t destSize = dest.Size();
            size_t remaining = destSize-srcSize;

            if (remaining > 0) {
                std::memset(dest.Data()+srcSize, 0, remaining*sizeof(float));
            }
        }
    }

    static void CopyByRow(const Tensor<float>& src, Tensor<float>& dest);

    template <bool acum> static void SumColumns(const Tensor<float>& a, Tensor<float>& b);
};
