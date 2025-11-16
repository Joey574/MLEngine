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

    static inline void CopyByRow(const Tensor<float>& src, Tensor<float>& dest) {
        assert(src.Data() != nullptr && dest.Data() != nullptr);
        assert(dest.Size() % src.Size() == 0);
        assert(!src.HasNan());

        const size_t srcSize = src.Size();
        const size_t n = dest.Size() / srcSize;

        const float* __restrict srcData = src.Data();
        float* __restrict dstData = dest.Data();

        for (size_t i = 0; i < n; i++) {
            cblas_scopy(srcSize, srcData, 1, &dstData[i*srcSize], 1);
        }
    }

    template <bool ACUM> static void SumColumns(const Tensor<float>& a, Tensor<float>& b) {
        assert(a.Dimensionality() == b.Dimensionality()+1);
        assert(a.Data() != nullptr && b.Data() != nullptr);
        assert(!a.HasNan() && !b.HasNan());
        
        const size_t size = a.Size();

        if (size > 2048*1024) {
            ParallelSumColumns<ACUM, 16>(a, b);
        } else {
            SerialSumColumns<ACUM>(a, b);
        }
    }
    template <bool ACUM> static inline void SerialSumColumns(const Tensor<float>& a, Tensor<float>& b) {
        assert(a.Dimensionality() == b.Dimensionality()+1);
        assert(a.Data() != nullptr && b.Data() != nullptr);
        assert(!a.HasNan() && !b.HasNan());

        const auto aDims = a.Dimensions();
        const size_t ar = aDims[0];
        const size_t ac = aDims[1];

        if constexpr (!ACUM) {
            b.Zero();
        }

        const float* __restrict aData = a.Data();
        float* __restrict bData = b.Data();

        for (size_t r = 0; r < ar; r++) {
            cblas_saxpy(ac, 1.0f, &aData[r*ac], 1, bData, 1);
        }
    }
    template <bool ACUM, size_t NUM_THREADS> static inline void ParallelSumColumns(const Tensor<float>& a, Tensor<float>& b) {
        assert(a.Dimensionality() == b.Dimensionality()+1);
        assert(a.Data() != nullptr && b.Data() != nullptr);
        assert(!a.HasNan() && !b.HasNan());
        
        const auto aDims = a.Dimensions();
        const size_t ar = aDims[0];
        const size_t ac = aDims[1];

        Tensor<float> threadBuf(ac*NUM_THREADS);
        threadBuf.Zero();

        if constexpr (!ACUM) {
            b.Zero();
        }

        const float* __restrict aData = a.Data();
        float* __restrict bData = b.Data();
        float* __restrict tData = threadBuf.Data();

        #pragma omp parallel num_threads(NUM_THREADS)
        {
            const int tid = omp_get_thread_num();

            // parallel axpy into threadBuf
            #pragma omp for schedule(static)
            for (size_t r = 0; r < ar; r++) {
                cblas_saxpy(ac, 1.0f, &aData[r*ac], 1, &tData[tid*ac], 1);
            }

            // serialize threadBud into bData
            #pragma omp barrier
            #pragma omp single
            for (size_t i = 0; i < NUM_THREADS; i++) {
                cblas_saxpy(ac, 1.0f, &tData[i*ac], 1, bData, 1);
            }
        }
    }
};
