#include "DataLoader.hpp"

template <uint8_t augments>
void DataLoader::Augment(size_t e) {
    constexpr bool rotat = (augments >> 0) & 1;
    constexpr bool scale = (augments >> 1) & 1;
    constexpr bool shear = (augments >> 2) & 1;
    constexpr bool elast = (augments >> 3) & 1;

    // copy original data into train data buffer
    std::memcpy(&trainData.data[0], &originalData.data[0], originalData.data.size()*sizeof(float));
    std::memcpy(&trainLabels.data[0], &originalLabels.data[0], originalLabels.data.size()*sizeof(float));

    size_t a_idx = originalData.rows;
    std::mt19937 rd(SEED+e+1234);

    if constexpr (rotat) {
        float rot = args[Y_ROTATION].as<float>();
        float mrot = args[Y_MIN_ROTATION].as<float>(Y_MIN_ROTATION_DEFAULT);
        size_t samples = args[Y_ROT_VARIANTS].as<size_t>(Y_ROT_VAR_DEFAULT);

        a_idx = ApplyRotation(trainData, trainLabels, originalData.rows, rd, dims[0], dims[1], rot, mrot, samples, a_idx);
    }

    if constexpr (scale) {
        float scale = args[Y_SCALE].as<float>();
        float mscale = args[Y_MIN_SCALE].as<float>(Y_MIN_SCALE_DEFAULT);
        size_t samples = args[Y_SCALE_VARIANTS].as<size_t>(Y_SCALE_VAR_DEFAULT);

        a_idx = ApplyScale(trainData, trainLabels, originalData.rows, rd, dims[0], dims[1], scale, mscale, samples, a_idx);
    }

    if constexpr (shear) {
        float shear = args[Y_SHEAR].as<float>();
        float mshear = args[Y_MIN_SHEAR].as<float>(Y_MIN_SHEAR_DEFAULT);
        size_t samples = args[Y_SHEAR_VARIANTS].as<size_t>(Y_SHEAR_VAR_DEFAULT);

        a_idx = ApplyShear(trainData, trainLabels, originalData.rows, rd, dims[0], dims[1], shear, mshear, samples, a_idx);
    }

    if constexpr (elast) {
        float alpha = args[Y_ELASTIC_DEFORM][Y_ELASTIC_ALPHA].as<float>(Y_ELASTIC_ALPHA_DEFAULT);
        float sigma = args[Y_ELASTIC_DEFORM][Y_ELASTIC_SIGMA].as<float>(Y_ELASTIC_SIGMA_DEFAULT);
        size_t samples = args[Y_ELASTIC_DEFORM][Y_ELASTIC_VARIANTS].as<size_t>(Y_ELASTIC_VAR_DEFAULT);

        a_idx = ApplyElasticDeform(trainData, trainLabels, originalData.rows, rd, dims[0], dims[1], alpha, sigma, samples, a_idx);
    }
}
