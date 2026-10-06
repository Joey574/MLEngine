#pragma once

/* @brief
An entirely static struct that just contains various math implementations, useful to abstract away the complicated math bits, highly optimized
to allow any struct to perform effectively
*/
struct MathUtils {
public:

    template <bool ACUM> static inline void DotProd(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) { Sgemm<ACUM, false, false>(a, b, c); }
    template <bool ACUM> static inline void DotProdTA(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) { Sgemm<ACUM, true, false>(a, b, c); }
    template <bool ACUM> static inline void DotProdTB(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) { Sgemm<ACUM, false, true>(a, b, c); }

    template <bool ACUM, bool TRANS_A, bool TRANS_B> static inline void Sgemm(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
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

        const size_t M = TRANS_A ? ac : ar;
        const size_t N = TRANS_B ? br : bc;
        const size_t K = TRANS_A ? ar : ac;
        assert(K == (TRANS_B ? bc : br));
        assert(cr == M && cc == N);

        constexpr const float beta                 = ACUM ? 1.0f : 0.0f;
        constexpr const CBLAS_TRANSPOSE aTranspose = TRANS_A ? CblasTrans : CblasNoTrans;
        constexpr const CBLAS_TRANSPOSE bTranspose = TRANS_B ? CblasTrans : CblasNoTrans;

        cblas_sgemm(CblasRowMajor, aTranspose, bTranspose, M, N, K, 1.0f, a.Data(), ac, b.Data(), bc, beta, c.Data(), cc);
    }

    static inline void Copy(const Tensor<float>& src, Tensor<float>& dest) {
        assert(src.Data() != nullptr && dest.Data() != nullptr);
        assert(src.Size() == dest.Size());
        assert(!src.HasNan());

        cblas_scopy(src.Size(), src.Data(), 1, dest.Data(), 1);
    }

    template <bool CLEAR> static inline void PartialCopy(const Tensor<float>& src, Tensor<float>& dest) {
        assert(src.Data() != nullptr && dest.Data() != nullptr);
        assert(dest.Size() >= src.Size());
        assert(!src.HasNan());

        const size_t srcSize = src.Size();
        cblas_scopy(srcSize, src.Data(), 1, dest.Data(), 1);

        // zero out remaining elements
        if constexpr (CLEAR) {
            const size_t destSize = dest.Size();
            size_t remaining      = destSize - srcSize;

            if (remaining > 0) {
                std::memset(dest.Data() + srcSize, 0, remaining * sizeof(float));
            }
        }
    }

    static inline void CopyByRow(const Tensor<float>& src, Tensor<float>& dest) {
        assert(src.Data() != nullptr && dest.Data() != nullptr);
        assert(dest.Size() % src.Size() == 0);
        assert(!src.HasNan());

        const size_t srcSize = src.Size();
        const size_t n       = dest.Size() / srcSize;

        const float* __restrict srcData = src.Data();
        float* __restrict dstData       = dest.Data();

        for (size_t i = 0; i < n; i++) {
            cblas_scopy(srcSize, srcData, 1, &dstData[i * srcSize], 1);
        }
    }

    template <bool ACUM> static inline void SumColumns(const Tensor<float>& a, Tensor<float>& b) {
        assert(a.Dimensionality() == b.Dimensionality() + 1);
        assert(a.Data() != nullptr && b.Data() != nullptr);
        assert(!a.HasNan() && !b.HasNan());

        const auto aDims = a.Dimensions();
        const size_t ar  = aDims[0];
        const size_t ac  = aDims[1];

        const float* __restrict aData = a.Data();
        float* __restrict bData       = b.Data();

        if constexpr (ACUM) {
            for (size_t r = 0; r < ar; r++) {
                cblas_saxpy(ac, 1.0f, &aData[r * ac], 1, bData, 1);
            }
        } else {
            // clear out old values
            cblas_saxpby(ac, 1.0f, &aData[0 * ac], 1, 0.0f, bData, 1);

            for (size_t r = 1; r < ar; r++) {
                cblas_saxpy(ac, 1.0f, &aData[r * ac], 1, bData, 1);
            }
        }
    }

    // image augmenting utils
    static float BilinearSample(const float* image, size_t w, size_t h, float fx, float fy);

    static void RotateImage(const float* image, float* out, size_t width, size_t height, float deg);
    static void ScaleImage(const float* image, float* out, size_t width, size_t height, float scale);
    static void ShearImage(const float* image, float* out, size_t width, size_t height, float shear);
    static void ElasticDeformImage(const float* image, float* out, const std::vector<float>& k, std::vector<float>& tmp, std::vector<float>& uxs, std::vector<float>& uys, std::mt19937& rd, size_t width, size_t height, float alpha, float sigma);

    static std::vector<float> MakeGaussianKernel2D(int rad, float sigma);
    static std::vector<float> MakeGaussianKernel1D(int rad, float sigma);

    static std::vector<float> Convolve2D(const std::vector<float>& f, const std::vector<float>& k, size_t w, size_t h, int rad);
    static void ConvolveHorizontal(const std::vector<float>& f, std::vector<float>& out, const std::vector<float>& k, size_t w, size_t h, int rad);
    static void ConvolveVertical(const std::vector<float>& f, std::vector<float>& out, const std::vector<float>& k, size_t w, size_t h, int rad);

    // math utils
    static float Sum256(__m256 _x);
    static float Sum512(__m512 _x);
    static float Max256(__m256 _x);
    static float Max512(__m512 _x);
    static __m256 Exp256(__m256 _x);
    static __m512 Exp512(__m512 _x);

    static void Scale(float* a, float scale, size_t n);
    static void Normalize(float* a, float lower, float upper, size_t n);

    static std::pair<float,float> ColMinMax(float* a, size_t rows, size_t cols, size_t c);
    static void NormalizeCol(float* a, float lower, float upper, size_t rows, size_t cols, size_t c);
    static void NormalizeCol(float* a, float lower, float upper, float min, float max, size_t rows, size_t cols, size_t c);

    // rng utils
    static uint32_t xorshift32(uint32_t state);
    static float fastRandFloat(uint32_t state);

    /// @brief only works with powers of 2
    static inline size_t RoundTo(size_t alignment, size_t n) {
        alignment--;
        return (n+alignment) & ~alignment;
    }
};
