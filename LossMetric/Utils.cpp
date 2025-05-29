#include "LossMetric.hpp"

LossMetric::Type LossMetric::ParseType(const std::string& lm) {
    if (lm == "mae") {
        return Type::mae;
    } else if (lm == "mse") {
        return Type::mse;
    } else if (lm == "accuracy") {
        return Type::accuracy;
    } else if (lm == "onehot") {
        return Type::onehot;
    }

    return Type::none;
}
std::string LossMetric::ParseName(Type type) {
    switch (type) {
        case Type::mae:
            return "mae";
        case Type::mse:
            return "mse";
        case Type::accuracy:
            return "accuracy";
        case Type::onehot:
            return "onehot";
        default:
            return "none";
    }
}

float LossMetric::Sum256(__m256 _x) {
	__m256 _sum1 = _mm256_hadd_ps(_x, _x);
    __m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

    __m128 _low  = _mm256_castps256_ps128(_sum2);
    __m128 _high = _mm256_extractf128_ps(_sum2, 1);
    __m128 _res  = _mm_add_ps(_low, _high);

    return _mm_cvtss_f32(_res);
}
