#include "planner.hpp"
#include <vector>
#include <assert.h>

void Plan::AppendRequest(const Request& r) {
    this->requests.push_back(r);
}

void Plan::AppendRequestWithPrefix(const std::string& prefix, const Request& r) {
    auto rKeyed = r;
    rKeyed.AppendPrefix(prefix);
    this->requests.push_back(rKeyed);
}

Plan Plan::Join(const Plan& other) const {
    assert(this->alignment == other.alignment);
    assert(this->alignment % 2 == 0);

    Plan p("", this->alignment);

    if (this->key.empty()) {
        for (const auto& r : this->requests) {
            p.AppendRequest(r);
        }
    } else {
        for (const auto& r : this->requests) {
            p.AppendRequestWithPrefix(this->key, r);
        }
    }

    if (other.key.empty()) {
        for (const auto& r : other.requests) {
            p.AppendRequest(r);
        }
    } else {
        for (const auto& r : other.requests) {
            p.AppendRequestWithPrefix(other.key, r);
        }
    }

    return p;
}

const std::string& Plan::Key() const {
    return this->key;
}

size_t Plan::Alignment() const {
    return this->alignment;
}

const std::vector<Request>& Plan::Requets() const {
    return this->requests;
}
