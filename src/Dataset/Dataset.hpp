#pragma once

struct Dataset {
public:

    int Define(YAML::Node& config);
    int Build();

    bool IsDefined() const { return defined; }
    bool IsBuilt() const { return built; }

private:
    
    bool defined = false;
    bool built = false;
};
