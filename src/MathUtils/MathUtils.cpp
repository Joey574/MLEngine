#include "MathUtils.tpp"

template void MathUtils::SumColumns<true>(const Tensor<float>& a, Tensor<float>& b);
template void MathUtils::SumColumns<false>(const Tensor<float>& a, Tensor<float>& b);
