#pragma once
#ifdef USE_PARROT
    #include "parrot.hpp"
#else
    using fusion_array=float*;
#endif

struct Tensor {
    public:

    
    inline size_t Dimensionality() const { return dimensions.size(); }
    inline const std::vector<size_t> Dimensions() const { return dimensions; }

    private:
    using Data = std::variant<float*, fusion_array>;

    Data data;
    std::vector<size_t> dimensions;
};
