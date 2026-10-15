#include "planner.hpp"
#include <vector>

void Request::AppendPrefix(const std::string& prefix) {
    std::string copy = prefix;
    this->key = copy + "_" + this->key;
}

const std::string& Request::Key() const {
    return this->key;
}

size_t Request::Bytes() const {
    return this->bytes;
}

const std::vector<Options>& Request::Opts() const {
    return this->opts;
}
