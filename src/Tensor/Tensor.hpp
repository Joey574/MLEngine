#pragma once

template <typename T>
struct Tensor {
    public:

    template <typename... Dims> Tensor(Dims... dims) : dimensions{(size_t)dims...} {
        size_t size = Size();
        data = aligned_alloc(32, size*sizeof(T));
        memset(data, 0, size*sizeof(T));
    }

    inline const T* Data() const { return data; }
    inline T* Data() { return data; }
    inline size_t Dimensionality() const { return dimensions.size(); }
    inline const std::vector<size_t>& Dimensions() const { return dimensions; }

    inline const size_t Size() const {
        return std::reduce(std::execution::par, dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
    }

    private:

    T* data;
    std::vector<size_t> dimensions;
};
