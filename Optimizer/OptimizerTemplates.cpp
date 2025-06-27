#include "Optimizer.hpp"
#include "Optimizers.tpp"

template void Optimizer::SGD<Optimizer::Regularization::none>(float*, float*, size_t, size_t, size_t);
template void Optimizer::SGD<Optimizer::Regularization::l1>(float*, float*, size_t, size_t, size_t);
template void Optimizer::SGD<Optimizer::Regularization::l2>(float*, float*, size_t, size_t, size_t);

template void Optimizer::MomentumSGD<Optimizer::Regularization::none>(float*, float*, size_t, size_t, size_t);
template void Optimizer::MomentumSGD<Optimizer::Regularization::l1>(float*, float*, size_t, size_t, size_t);
template void Optimizer::MomentumSGD<Optimizer::Regularization::l2>(float*, float*, size_t, size_t, size_t);
