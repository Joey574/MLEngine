#pragma once

struct Dataset {
    public:

    int Define(YAML::Node& config);
    int Build();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    private:
    bool defined = false;
    bool built = false;
};
