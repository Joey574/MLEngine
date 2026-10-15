#pragma once
#include <vector>
#include <string>
#include <cstddef>
#include <unordered_map>

struct Options {
    std::string key = "";
    bool reads = false;
    bool writes = false;
    bool linearWalk = true;
};

/// @brief Defines the general outline of an individual request, its storage key, the numebr of bytes, and its dependencies
struct Request {
    public:
        Request(const std::string& key, size_t bytes, const std::vector<Options>& opts) :
            key(key),
            bytes(bytes),
            opts(opts)
        {}

        Request(const std::string& key, size_t bytes) : key(key), bytes(bytes) {}

        void AppendPrefix(const std::string& prefix);

        const std::string& Key() const;
        size_t Bytes() const;
        const std::vector<Options>& Opts() const;

    private:
        std::string key;
        size_t bytes;
        std::vector<Options> opts;
};

/// @brief Holds a set of requests and allows a client to append new requests and even merge with other plans
/// @brief Memory plan holds its own key to be used as a top level mapping when merging with other plans, in the form key_request.key
struct Plan {
    public:
        Plan() : key(""), alignment(0) {}
        Plan(const std::string& key) : key(key), alignment(0) {}
        Plan(const std::string& key, size_t alignment) : key(key), alignment(alignment) {}

        void AppendRequest(const Request& r);
        void AppendRequestWithPrefix(const std::string& prefix, const Request& r);

        /// @brief takes another plan and returns a new one that is the combination of both
        Plan Join(const Plan& other) const;

        const std::string& Key() const;
        size_t Alignment() const;
        const std::vector<Request>& Requets() const;

    private:
        std::string key;
        size_t alignment;
        std::vector<Request> requests;
};

/// @brief Output of an actualized memory plan, holds the mappings of requested pointers to actual locations in memory
struct Policy {
    public:
        Policy(size_t allocated, std::unordered_map<std::string, char*> mapping) : allocated(allocated), mapping(mapping) {}

        size_t Allocated() const;
        std::unordered_map<std::string, char*> mapping;
    private:
        size_t allocated;
};

/// @brief Plans the execution of a memory plan
struct Planner {
    public:
        Policy Actualize(const Plan& plan);

    private:
};
