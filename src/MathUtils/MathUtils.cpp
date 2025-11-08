#include "MathUtils.tpp"

template void MathUtils::DotProd<true>(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c);
template void MathUtils::DotProd<false>(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c);

template void MathUtils::DotProdTA<true>(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c);
template void MathUtils::DotProdTA<false>(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c);

template void MathUtils::DotProdTB<true>(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c);
template void MathUtils::DotProdTB<false>(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c);

template void MathUtils::SumColumns<true>(const Tensor<float>& a, Tensor<float>& b);
template void MathUtils::SumColumns<false>(const Tensor<float>& a, Tensor<float>& b);
