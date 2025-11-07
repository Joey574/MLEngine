#pragma once

struct Adam {
    public:
    void Compute();

    void Define();
    void Build();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    private:
    bool defined = false;
    bool built = false;
};
