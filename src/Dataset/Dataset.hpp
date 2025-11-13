#pragma once
#include "../MathUtils/MathUtils.hpp"

struct Dataset {
    public:
    enum class Type {
        None, MNIST, FMNIST, Mandlebrot
    };

    int Define(YAML::Node& config);
    int Build();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    inline Tensor<float>& Data(size_t start, size_t n) {
        dataView = data.Slice(start, n);
        return dataView;
    }
    inline Tensor<float>& Labels(size_t start, size_t n) {
        labelView = labels.Slice(start, n);
        return labelView;
    }
    inline size_t Samples() { return elements; }

    inline static Type ParseType(const std::string& name) {
        auto lower = std::string(name.size(), ' ');
        std::transform(name.begin(), name.end(), lower.begin(), tolower);

        if (lower == "mnist") {
            return Type::MNIST;
        } else if (lower == "fmnist") {
            return Type::FMNIST;
        } else if (lower == "mandlebrot") {
            return Type::Mandlebrot;
        } else {
            return Type::None;
        }
    }
    inline static std::string ParseName(const Type type) {
        switch (type) {
            case Type::None:
                return "None";
            case Type::MNIST:
                return "MNIST";
            case Type::FMNIST:
                return "FMNIST";
            case Type::Mandlebrot:
                return "Mandlebrot";
            default:
                return "";
        }
    }

    private:
    bool defined = false;
    bool built = false;

    Type type;
    size_t elements;
    YAML::Node* config;

    Tensor<float> data;
    Tensor<float> labels;

    Tensor<float> dataView;
    Tensor<float> labelView;

    int LoadMNISTStyle(const std::string& name);
    int LoadMandlebrot();

    static inline std::string ExpandPath(const std::string& path) {
        if (path.empty() || path[0] != '~') [[unlikely]] {
            return path;
        }

        const char* home = getenv("HOME");
        return home + path.substr(1);
    }
    static inline int ReadBigInt(std::ifstream* f) {
        int lint;
        f->read(reinterpret_cast<char*>(&lint), sizeof(int));

        unsigned char* bytes = reinterpret_cast<unsigned char*>(&lint);
        std::swap(bytes[0], bytes[3]);
        std::swap(bytes[1], bytes[2]);

        return lint;
    }
};
