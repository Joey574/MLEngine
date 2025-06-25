#pragma once
#ifndef TNSR
    #include "Tensor.hpp"
#endif

template <typename T>
size_t Tensor<T>::computeFlatIndex(const std::initializer_list<size_t>& idxs) const {
    assert(idxs.size() == m_dimensionality);

    size_t flat = 0;
    size_t stride = 1;

    auto dim_it = m_dimensions.rbegin();
    auto idx_it = idxs.end();

    while (dim_it != m_dimensions.rend()) {
        idx_it--;
        flat += (*idx_it) * stride;
        stride *= (*dim_it);
        dim_it++;
    }

    return flat;
}
