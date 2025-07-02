#pragma once
#include "../NeuralNetwork/NeuralNetwork.hpp"

class DataLoader {
public:

    static Dataset LoadDataset(YAML::Node& config);
    static Dataset LoadMNIST(YAML::Node& config);
    static Dataset LoadFMNIST(YAML::Node& config);
    static Dataset LoadMandlebrot(YAML::Node& config);

    static void VisualizeMandlebrot(NeuralNetwork& model, const std::string& path, size_t width, size_t height);
    static void VisualizeTerminalMNISTLike(const float* image, size_t width, size_t height);

private:
    // mnist / fmnist utils
    static int ReadBigInt(std::ifstream* f);
    static void LoadMNISTStyleDataset(Dataset& dataset, YAML::Node& args, std::ifstream& traind, std::ifstream& trainl, std::ifstream& testd, std::ifstream& testl);

    static float BilinearSample(const float* image, size_t w, size_t h, float fx, float fy);
    static std::vector<float> RotateImage(const float* image, size_t width, size_t height, float deg);
    static std::vector<float> ScaleImage(const float* image, size_t width, size_t height, float scale);
    static std::vector<float> ShearImage(const float* image, size_t width, size_t height, float shear);


    // mandlebrot utils
    static float InMandlebrot(double x, double y, size_t it);
    static void ComputeFourier(float* x, size_t series);

    static std::string ExpandPath(const std::string& path);
};