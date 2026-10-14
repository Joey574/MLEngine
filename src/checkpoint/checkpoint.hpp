#pragma once
#include "../network/state.hpp"

struct Checkpoint {
    public:
        bool ShouldRun(NetworkState& state);
    private:
};
