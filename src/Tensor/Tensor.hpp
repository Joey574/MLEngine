#pragma once

template <typename T>
struct Tensor {
    public:

    /// @brief Constructor
    template <typename... Dims> Tensor(Dims... dims) : dimensions{dims...}, owner(true) {
        size_t size = Size();
        data = (T*)aligned_alloc(32, size*sizeof(T));
    }


    /// @brief Contstructor
    Tensor(float* data, std::vector<size_t>& dimensions, bool owner=true) : data(data), dimensions(dimensions), owner(owner) {}


    /// @brief Move constructor
    Tensor(Tensor&& other) noexcept : data(other.Data()), dimensions(std::move(other.Dimensions())), owner(other.owner) { other.Data() = nullptr; }


    /// @brief Copy constructor
    Tensor(const Tensor& other) : dimensions(other.Dimensions()), owner(other.owner) {
        size_t size = Size();

        #ifdef DEBUG
            std::cout << "[-] Tensor copy constructor (" << size*sizeof(T) << " bytes)\n";
        #endif

        data = (T*)aligned_alloc(32, size*sizeof(T));
        std::memcpy(data, other.Data(), size*sizeof(T));
    }


    /// @brief Deconstructor
    ~Tensor() { if (data && owner) {
        #ifdef DEBUG
            std::cout << "[i] Tensor freeing (" << Size()*sizeof(T) << " bytes)\n";
        #endif
        std::free(data); } 
    }


    /// @brief Move operator
    Tensor& operator = (Tensor&& other) noexcept {
        if (data && owner && this != &other) { 
            #ifdef DEBUG
                std::cout << "[i] Tensor freeing (" << Size()*sizeof(T) << " bytes)\n";
            #endif
            std::free(data); 
        }

        data = other.Data();
        dimensions = std::move(other.dimensions);
        other.data = nullptr;
        owner = other.owner;
        return *this;
    }


    /// @brief Copy operator
    Tensor& operator = (const Tensor& other) {
        if (data && owner && this != &other) { 
            #ifdef DEBUG
                std::cout << "[i] Tensor freeing (" << Size()*sizeof(T) << " bytes)\n";
            #endif
            free(data); 
        }
        
        dimensions = other.dimensions;
        size_t size = Size();

        #ifdef DEBUG
            std::cout << "[-] Tensor copy assignment (" << size*sizeof(T) << " bytes)\n";
        #endif

        data = (T*)aligned_alloc(32, size*sizeof(T));
        std::memcpy(data, other.Data(), size*sizeof(T));
        owner = other.owner;
        return *this;
    }


    /// @return Const pointer to raw data 
    inline const T* Data() const {
        assert(data != nullptr);
        return data; 
    }


    /// @return Pointer to raw data
    inline T* Data() {
        assert(data != nullptr);
        return data; 
    }


    /// @return The dimensionality of the tensor
    inline size_t Dimensionality() const {
        return dimensions.size(); 
    }


    /// @return vector of dimensions
    inline constexpr const std::vector<size_t>& Dimensions() const { return dimensions; }


    /// @return The number of elements in the tensor
    inline const size_t Size() const {
        if (dimensions.empty()) { return 0; }
        return std::reduce(std::execution::unseq, dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
    }


    /// @brief Creates a tensor of 1 less dimensionality
    /// @param start The element the slice should start at
    /// @param n The number of elements to include
    /// @return A new non-owning tensor slice
    inline Tensor Slice(size_t start, size_t n) {
        assert(!dimensions.empty());
        assert(Size() != 0);

        size_t stride = 1;
        if (dimensions.size() > 1) {
            stride = std::reduce(std::execution::unseq, dimensions.begin(), dimensions.end()-1, 1, std::multiplies<size_t>());
        }

        T* offsetData = data + stride*start;
        auto d = dimensions;
        d[d.size()-1] = n;

        return Tensor(offsetData, d, false);
    }


    /// @brief Checks tensor for any nan values
    /// @return True if tensor has a nan value 
    inline bool HasNan() const {
        if constexpr (std::is_floating_point_v<T>) {
            const size_t n = Size();

            for (size_t i = 0; i < n; i++) {
                if (std::isnan(data[i])) {
                    return true;
                }
            }
        }

        return false;
    }


    /// @brief Zeroes out all tensor elements
    inline void Zero() {
        std::memset(data, 0, Size()*sizeof(T));
    }

    private:

    T* data;
    bool owner;
    std::vector<size_t> dimensions;
};
