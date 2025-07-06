#pragma once

struct MathUtils {
public:
    using DotProdFunc = void (*)(const float*, const float*, float*, size_t, size_t, size_t, size_t);

    template <bool clear> static void DotProd(const float* a, const float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTA(const float* a, const float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
    template <bool clear> static void DotProdTB(const float* a, const float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

    template <bool clear> static void MatrixColumnSum(const float* a, float* b, size_t a_r, size_t a_c);

    // image augmenting utils
    static float BilinearSample(const float* image, size_t w, size_t h, float fx, float fy);
    
    static void RotateImage(const float* image, float* out, size_t width, size_t height, float deg);
    static void ScaleImage(const float* image, float* out, size_t width, size_t height, float scale);
    static void ShearImage(const float* image, float* out, size_t width, size_t height, float shear);
    static void ElasticDeformImage(const float* image, float* out, std::mt19937& rd, size_t width, size_t height, float alpha, float sigma);

    static std::vector<float> MakeGaussianKernel(int rad, float sigma);
    static std::vector<float> Convolve(const std::vector<float>& f, size_t width, size_t height, const std::vector<float>& k, int rad);

    static void Normalize(float* a, float sum, size_t n);

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