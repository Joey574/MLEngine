#pragma once

struct Dataset {
    public:

    int Define(YAML::Node& config);
    int Build();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    inline float* Data() { return data; }
    inline float* Labels() { return labels; }

    private:
    bool defined = false;
    bool built = false;

    float* data;
    float* labels;
    size_t elements;
};
