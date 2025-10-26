#include "MathUtils.hpp"

void MathUtils::ScaleBy(float* a, const float* b, size_t n) {
    
}
void MathUtils::ScaleBy(float* a, float b, size_t n) {
    cblas_sscal(n, b, a, 0);
}

void MathUtils::Copy(const float* src, float* dest, size_t n) {
    cblas_scopy(n, src, 0, dest, 0);
}