#pragma once
#include "../MathUtils/MathUtils.hpp"

struct Dataset {
  public:
    enum class Type { None, MNIST, FMNIST, Mandlebrot };

    int Define(YAML::Node& config);
    int Build();

    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

    inline Tensor<float>& TrainingData(size_t start, size_t n) {
        dataView = trainingData.Slice(start, n);
        return dataView;
    }
    inline Tensor<float>& TrainingLabels(size_t start, size_t n) {
        labelView = trainingLabels.Slice(start, n);
        return labelView;
    }
    inline size_t TrainingSamples() const { return trainingData.Dimensions()[trainingData.Dimensionality() - 1]; }

    inline Tensor<float>& TestingData(size_t start, size_t n) {
        dataView = testingData.Slice(start, n);
        return dataView;
    }
    inline Tensor<float>& TestingLabels(size_t start, size_t n) {
        labelView = testingLabels.Slice(start, n);
        return labelView;
    }
    inline Tensor<float>& TestingData() { return testingData; }
    inline Tensor<float>& TestingLabels() { return testingLabels; }
    inline size_t TestingSamples() const { return testingData.Dimensions()[testingData.Dimensionality() - 1]; }

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
    bool built   = false;

    Type type;
    size_t elements;
    YAML::Node* config;

    Tensor<float> trainingData;
    Tensor<float> trainingLabels;

    Tensor<float> testingData;
    Tensor<float> testingLabels;

    Tensor<float> dataView;
    Tensor<float> labelView;

    int LoadMNIST();
    int LoadFMNIST();
    int LoadMNISTStyle(std::ifstream& traind, std::ifstream& trainl, std::ifstream& testd, std::ifstream& testl);

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
