#pragma once

template <typename T>
struct Tensor {
    public:

    /// @brief Constructor
    template <typename... Dims> Tensor(Dims... dims) : dimensions{dims...}, owner(true) {
        size_t size = Size();
        data = (T*)aligned_alloc(32, size*sizeof(T));
        std::memset(data, 0, size*sizeof(T));
    }


    /// @brief Contstructor
    Tensor(float* data, std::vector<size_t>& dimensions, bool owner=true) : data(data), dimensions(dimensions), owner(owner) {}


    /// @brief Move constructor
    Tensor(Tensor&& other) noexcept : data(other.data), dimensions(std::move(other.dimensions)), owner(true) { other.data = nullptr; }

    /// @brief Copy constructor
    Tensor(const Tensor& other) : dimensions(other.dimensions), owner(true) {
        size_t size = Size();
        std::cout << "[-] Tensor copy constructor (" << size*sizeof(T) << " bytes)\n";

        data = (T*)aligned_alloc(32, size*sizeof(T));
        std::memcpy(other.data, data, size*sizeof(T));
    }

    /// @brief Deconstructor
    ~Tensor() { if (data && owner) { std::free(data); } }

    /// @brief Move operator
    Tensor& operator = (Tensor&& other) noexcept {
        if (data && owner && this != &other) { std::free(data); }

        data = other.data;
        dimensions = std::move(other.dimensions);
        other.data = nullptr;
        owner = true;
        return *this;
    }

    /// @brief Copy operator
    Tensor& operator = (const Tensor& other) {
        if (data && owner && this != &other) { free(data); }
        dimensions = other.dimensions;

        size_t size = Size();
        std::cout << "[-] Tensor copy assignment (" << size*sizeof(T) << " bytes)\n";

        data = (T*)aligned_alloc(32, size*sizeof(T));
        std::memcpy(data, other.data, size*sizeof(T));
        owner = true;
        return *this;
    }


    /// @return Const pointer to raw data 
    inline const T* Data() const { return data; }


    /// @return Pointer to raw data
    inline T* Data() { return data; }


    /// @return The dimensionality of the tensor
    inline size_t Dimensionality() const { return dimensions.size(); }


    /// @return vector of dimensions
    inline const std::vector<size_t>& Dimensions() const { return dimensions; }


    /// @return The number of elements in the tensor
    inline const size_t Size() const {
        return std::reduce(std::execution::unseq, dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
    }


    /// @brief Creates a tensor of 1 less dimensionality
    /// @param start The element the view should start at
    /// @param n The number of elements to include
    /// @return A new non-owning tensor from start
    inline Tensor& ViewFrom(size_t start, size_t n) {
        size_t size = std::reduce(std::execution::unseq, dimensions.begin(), dimensions.end()-1, 1, std::multiplies<size_t>());
        float* offsetData = &data[size*start];

        auto d = std::vector<size_t>(dimensions.begin(), dimensions.end()-1);
        d.push_back(n);
        
        Tensor<T> t(offsetData, d, false);
        return t;
    }

    private:

    T* data;
    bool owner;
    std::vector<size_t> dimensions;
};
