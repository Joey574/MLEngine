#pragma once
#include "../NeuralNetwork/NeuralNetwork.hpp"

struct DataLoader {
public:
    enum Type {
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

    bool running_augment;
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

    static void VisualizeTerminalMNISTLike(const float* image, size_t width, size_t height);

private:
    YAML::Node args;

    // mnist / fmnist utils
    static int ReadBigInt(std::ifstream* f);
    void LoadMNISTStyleDataset(std::ifstream& traind, std::ifstream& trainl, std::ifstream& testd, std::ifstream& testl);

    static float BilinearSample(const float* image, size_t w, size_t h, float fx, float fy);
    static void RotateImage(const float* image, float* out, size_t width, size_t height, float deg);
    static void ScaleImage(const float* image, float* out, size_t width, size_t height, float scale);
    static void ShearImage(const float* image, float* out, size_t width, size_t height, float shear);
    static void ElasticDeformImage(const float* image, float* out, size_t width, size_t height, float alpha, float sigma);

    static std::vector<float> MakeGaussianKernel(int rad, float sigma);
    static std::vector<float> Convolve(const std::vector<float>& f, size_t width, size_t height, const std::vector<float>& k, int rad);


    // mandlebrot utils
    static float InMandlebrot(double x, double y, size_t it);
    static void ComputeFourier(float* x, size_t series);

    static std::string ExpandPath(const std::string& path);
};