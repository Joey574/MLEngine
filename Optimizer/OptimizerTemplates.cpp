#include "Optimizer.hpp"
#include "OptimizerSGD.tpp"

template void Optimizer::SGD<Optimizer::Regularization::none>(float*, float*, size_t, size_t, size_t);
template void Optimizer::SGD<Optimizer::Regularization::l1>(float*, float*, size_t, size_t, size_t);
template void Optimizer::SGD<Optimizer::Regularization::l2>(float*, float*, size_t, size_t, size_t);
