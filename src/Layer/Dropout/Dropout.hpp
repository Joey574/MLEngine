#pragma once

struct Dropout {
  public:
    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

  private:
    bool defined = false;
    bool built   = false;

    float rate;
    size_t bytes;
    Tensor<uint8_t> mask;
    std::bernoulli_distribution dist;
};
