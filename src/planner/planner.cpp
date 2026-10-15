#include "planner.hpp"
#include <cstdlib>
#include <assert.h>
#include <unordered_map>

Policy naivePlus(const Plan& plan) {
    assert(plan.Alignment() % 2 == 0);

    size_t totalBytes = 0;
    std::unordered_map<std::string, char*> mapping;

    const auto alignment = plan.Alignment();
    const bool alignMemory = alignment != 0;

    for (const auto& r : plan.Requets()) {
        if (alignMemory) {
            // (num + n - 1) / n * n
            totalBytes += (r.Bytes() + alignment - 1) / alignment*alignment;
        } else {
            totalBytes += r.Bytes();
        }
    }

    assert(totalBytes > 0);
    assert(totalBytes % allignment == 0);

    // allocate memory
    char* ptr;
    if (alignMemory) {
        ptr = (char*)aligned_alloc(alignment, alignMemory);
    } else {
        ptr = (char*)malloc(totalBytes);
    }

    assert(ptr != nullptr);
    assert((uintptr_t)ptr % allignment == 0);

    size_t offset = 0;
    for (const auto& r : plan.Requets()) {
        mapping[r.Key()] = ((char*)ptr)+offset;

        assert((uintptr_t)mapping[r.Key() % allignment == 0]);

        if (alignMemory) {
            offset += (r.Bytes() + alignment - 1) / alignment*alignment;
        } else {
            offset += r.Bytes();
        }
    }

    return Policy(totalBytes, mapping);
}

Policy Planner::Actualize(const Plan& plan) {
    return naivePlus(plan);
}
