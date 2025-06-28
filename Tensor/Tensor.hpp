#define TNSR

template <typename T>
struct Tensor {

public:
    Tensor() { memset(this, 0, sizeof(Tensor)); }

    template<typename... Dims>
    Tensor(Dims... dims, bool init=true) : m_dimensions{(size_t)dims...}, m_data_size(((size_t)dims * ... * 1)), m_dimensionality(sizeof...(dims)) {
        if (init) { data = (T*)aligned_alloc(32, m_data_size*sizeof(T)); }
    }

    template<typename... Dims>
    Tensor(T* data, Dims... dims) : m_dimensions{(size_t)dims...}, m_data_size(((size_t)dims * ... * 1)), m_dimensionality(sizeof...(dims)), data(data) {}

    ~Tensor() { if (data) { free(data); } }

    inline void takeover(T* data) { this->data=data; }

    inline T* begin() { return data; }
    
    inline size_t dimensionality() const { return m_dimensionality; }
    inline size_t size() const { return m_data_size; }
    inline const std::vector<size_t>& shape() const { return m_dimensions; }

    template<typename... Dims>
    static inline size_t sizefor(Dims... dims) { return ((size_t)dims * ... * sizeof(T)); }

    template<typename... Idxs>
    inline T& operator()(Idxs... idxs) {
        return data[computeFlatIndex({(size_t)idxs...})];
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