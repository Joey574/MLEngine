#define TNSR

template <typename T>
struct Tensor {

public:
    Tensor() : data(nullptr), m_data_size(0) {}

    template<typename... Dims>
    Tensor(Dims... dims) : m_dimensions{(size_t)dims...}, m_data_size(((size_t)dims * ... * 1)), m_dimensionality(sizeof...(dims)) {
        static_assert((std::is_convertible_v<Dims, size_t> && ...), "Must be convertable to size_t");
        data = (T*)aligned_alloc(32, m_data_size*sizeof(T));
    }

    template<typename... Dims>
    Tensor(T* data, Dims... dims) : m_dimensions{(size_t)dims...}, m_data_size(((size_t)dims * ... * 1)), m_dimensionality(sizeof...(dims)), data(data) {}

    ~Tensor() { if (data) { free(data); } }


    inline size_t dimensionality() const { return m_dimensionality; }
    inline size_t size() const { return m_data_size; }
    inline const std::vector<size_t>& shape() const { return m_dimensions; }

    template<typename... Idxs>
    inline T& operator()(Idxs... idxs) {
        static_assert(sizeof...(Idxs) > 0, "Must have at least one index");
        assert(sizeof...(Idxs) == m_dimensionality);

        size_t idx = computeFlatIndex({(size_t)idxs...});
        return data[idx];
    }    
    
private:
    T* data;
    size_t m_data_size;

    size_t m_dimensionality;
    std::vector<size_t> m_dimensions;

    size_t computeFlatIndex(const std::initializer_list<size_t>& idxs) const;
};

#include "tensor.impl.hpp"

#undef TNSR