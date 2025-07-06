#include "DataLoader.hpp"

void DataLoader::Deform(size_t e) {

    // apply data augments
    if (augment != nullptr) {
        augment(e);
    }

    // shuffle new traindata
    Shuffle(e, trainData, trainLabels);    
}
void DataLoader::Shuffle(size_t e, Matrix& data, Matrix& labels) {
    std::mt19937 rng(SEED+22+e);

    // swap blocks in place
    for (size_t i = data.rows-1; i > 0; i--) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);

        if (i != j) {
            auto block_id = data.data.begin() + (i*data.cols);
            auto block_jd = data.data.begin() + (j*data.cols);

            auto block_il = labels.data.begin() + (i*labels.cols);
            auto block_jl = labels.data.begin() + (j*labels.cols);

            std::swap_ranges(block_id, block_id+data.cols, block_jd);
            std::swap_ranges(block_il, block_il+labels.cols, block_jl);
        }
    }
}

size_t DataLoader::ApplyRotation(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float rot, float mrot, size_t samples, size_t a_idx) {
    std::uniform_real_distribution<float> gen(-rot, rot);

    // generate randomly rotated images of test dataset
    for (size_t i = 0; i < original_samples; i++) {
        for (size_t j = 0; j < samples; j++) {
            float deg = gen(rd);
            deg += deg < 0.0f ? -mrot : mrot;

            MathUtils::RotateImage(&data.data[i*data.cols], &data.data[a_idx*data.cols], w, h, deg);
            labels.data[a_idx] = labels.data[i];

            a_idx++;
        }
    }

    return a_idx;
}
size_t DataLoader::ApplyScale(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float scale, float mscale, size_t samples, size_t a_idx) {
    std::uniform_real_distribution<float> gen(1.0f-scale, 1.0f+scale);

    for (size_t i = 0; i < original_samples; i++) {
        for (size_t j = 0; j < samples; j++) {
            float rscale = gen(rd);
            rscale += rscale < 1.0f ? -mscale : mscale;

            MathUtils::ScaleImage(&data.data[i*data.cols], &data.data[a_idx*data.cols], w, h, rscale);
            labels.data[a_idx] = labels.data[i];

            a_idx++;
        }
    }

    return a_idx;
}
size_t DataLoader::ApplyShear(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float shear, float mshear, size_t samples, size_t a_idx) {
    std::uniform_real_distribution<float> gen(-shear, shear);

    for (size_t i = 0; i < original_samples; i++) {
        for (size_t j = 0; j < samples; j++) {

            float rshear = gen(rd);
            rshear += rshear < 0.0 ? -mshear : mshear;

            MathUtils::ShearImage(&data.data[i*data.cols], &data.data[a_idx*data.cols], w, h, rshear);
            labels.data[a_idx] = labels.data[i];

            a_idx++;
        }
    }

    return a_idx;
}
size_t DataLoader::ApplyElasticDeform(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float alpha, float sogma, size_t samples, size_t a_idx) {

}

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

    }
}
