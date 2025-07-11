#pragma once
#include "../MathUtils/MathUtils.hpp"

/* @brief
The DataLoader struct is responsible for loading, storing, and augmenting datasets to be used in training
*/
struct DataLoader {
public:
    using AugmentFn = void (DataLoader::*)(size_t);

    enum class Type {
        none, mnist, fmnist, mandlebrot
    };

    struct Matrix {
        size_t rows;
        size_t cols;
        std::vector<float> data;

    };

    Type type;
    std::string name;
    std::vector<size_t> dims;
    std::vector<size_t> test_dims;

    size_t refresh_every;
    bool hasTestData;

    Matrix trainData;
    Matrix trainLabels;

    Matrix testData;
    Matrix testLabels;

    Matrix originalData;
    Matrix originalLabels;

    void Deform(size_t e);


    void LoadDataset(YAML::Node& config);
    void LoadMNIST();
    void LoadFMNIST();
    void LoadMandlebrot();

    static void SaveMandleImage(const std::string& path, const float* points, size_t width, size_t height);
    static void VisualizeTerminalMNISTLike(const float* image, size_t width, size_t height);
    
private:
    YAML::Node args;

    // data augment utils
    AugmentFn augment;
    template <uint8_t augments> void Augment(size_t e);
    void Shuffle(size_t e, Matrix& data, Matrix& labels);

    static size_t ApplyRotation(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float rot, float mrot, size_t samples, size_t a_idx);
    static size_t ApplyScale(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float scale, float mscale, size_t samples, size_t a_idx);
    static size_t ApplyShear(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float shear, float mshear, size_t samples, size_t a_idx);
    static size_t ApplyElasticDeform(Matrix& data, Matrix& labels, size_t original_samples, std::mt19937& rd, size_t w, size_t h, float alpha, float sigma, size_t samples, size_t a_idx);

    // mnist / fmnist utils
    static int ReadBigInt(std::ifstream* f);
    void LoadMNISTStyleDataset(std::ifstream& traind, std::ifstream& trainl, std::ifstream& testd, std::ifstream& testl);

    // mandlebrot utils
    static float InMandlebrot(double x, double y, size_t it);
    static void ComputeFourier(float* x, size_t series);

    static std::string ExpandPath(const std::string& path);
};
