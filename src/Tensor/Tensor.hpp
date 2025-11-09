#pragma once

template <typename T>
struct Tensor {
    public:

    /// @brief Constructor
    template <typename... Dims> Tensor(Dims... dims) : dimensions{dims...} {
        size_t size = Size();
        data = (T*)aligned_alloc(32, size*sizeof(T));
        std::memset(data, 0, size*sizeof(T));
    }

    /// @brief Move constructor
    Tensor(Tensor&& other) noexcept : data(other.data), dimensions(std::move(other.dimensions)) { other.data = nullptr; }

    /// @brief Copy constructor
    Tensor(const Tensor& other) : dimensions(other.dimensions) {
        size_t size = Size();
        std::cout << "[-] Tensor copy constructor (" << size*sizeof(T) << " bytes)\n";

        data = (T*)aligned_alloc(32, size*sizeof(T));
        std::memcpy(other.data, data, size*sizeof(T));
    }

    /// @brief Deconstructor
    ~Tensor() { if (data) { std::free(data); } }

    /// @brief Move operator
    Tensor& operator = (Tensor&& other) noexcept {
        if (data && this != &other) { std::free(data); }

        data = other.data;
        dimensions = std::move(other.dimensions);
        other.data = nullptr;
        return *this;
    }

    /// @brief Copy operator
    Tensor& operator = (const Tensor& other) {
        if (data && this != &other) { free(data); }
        dimensions = other.dimensions;

        size_t size = Size();
        std::cout << "[-] Tensor copy assignment (" << size*sizeof(T) << " bytes)\n";

        data = (T*)aligned_alloc(32, size*sizeof(T));
        std::memcpy(data, other.data, size*sizeof(T));
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


    /// @brief 
    /// @param element 
    /// @return 
    inline Tensor& ViewFrom(size_t element) {
        // TODO : implement
        return *this;
    }

    private:

    T* data;
    std::vector<size_t> dimensions;
};
