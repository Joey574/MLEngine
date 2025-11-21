#pragma once

template <typename T>
struct Tensor {
    public:

    Tensor() : data(nullptr), owner(false) {}

    /// @brief Constructor
    template <typename... Dims> Tensor(Dims... dims) : dimensions{dims...}, owner(true) {
        size_t size = Size();
        if (size > 0) {
            data = (T*)aligned_alloc(32, size*sizeof(T));

            #ifdef DEBUG
                std::cout << "[i] Tensor allocating to " << data << " (" << size*sizeof(T) << " bytes)\n";
            #endif
        }
    }


    /// @brief Contstructor
    Tensor(T* data, std::vector<size_t>& dimensions, bool owner=true) : data(data), dimensions(dimensions), owner(owner) {}


    /// @brief Move constructor
    Tensor(Tensor&& other) noexcept : data(other.Data()), dimensions(std::move(other.dimensions)), owner(other.owner) { other.data = nullptr; other.owner = false; }


    /// @brief Copy constructor
    Tensor(const Tensor& other) : dimensions(other.dimensions), owner(true) {
        size_t size = Size();

        #ifdef DEBUG
            std::cout << "[-] Tensor copy constructor (" << size*sizeof(T) << " bytes)\n";
        #endif

        data = (T*)aligned_alloc(32, size*sizeof(T));
        std::memcpy(data, other.data, size*sizeof(T));
    }


    /// @brief Deconstructor
    ~Tensor() { 
        if (data && owner) {
            #ifdef DEBUG
                std::cout << "[i] Tensor freeing " << data <<  " (" << Size()*sizeof(T) << " bytes)\n";
            #endif

            std::free(data); 
        }
    }


    /// @brief Move operator
    inline Tensor& operator = (Tensor&& other) noexcept {
        if (data && owner && this != &other) { 
            #ifdef DEBUG
                std::cout << "[i] Tensor freeing " << data <<  " (" << Size()*sizeof(T) << " bytes)\n";
            #endif
            std::free(data); 
        }

        data = other.data;
        dimensions = std::move(other.dimensions);
        other.data = nullptr;
        owner = other.owner;
        other.owner = false;
        return *this;
    }


    /// @brief Copy operator
    inline Tensor& operator = (const Tensor& other) {
        if (data && owner && this != &other) { 
            #ifdef DEBUG
                std::cout << "[i] Tensor freeing " << data <<  " (" << Size()*sizeof(T) << " bytes)\n";
            #endif
            free(data); 
        }
        
        dimensions = other.dimensions;
        size_t size = Size();

        #ifdef DEBUG
            std::cout << "[-] Tensor copy assignment (" << size*sizeof(T) << " bytes)\n";
        #endif

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
    inline constexpr const std::vector<size_t>& Dimensions() const { return dimensions; }


    /// @return The number of elements in the tensor
    inline size_t Size() const {
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
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memset(data, 0, Size()*sizeof(T));
        }
    }


    inline bool IsEmpty() const { return data == nullptr; }


    inline std::enable_if_t<std::is_floating_point_v<T>, T> Mean() {
        const size_t n = Size();
        double mean = 0.0f;

        #pragma omp parallel for simd schedule(static) reduction(+:mean)
        for (size_t i = 0; i < n; i++) {
            mean += data[i];
        }

        return mean / n;
    }

    private:

    T* data;
    bool owner;
    std::vector<size_t> dimensions;
};
