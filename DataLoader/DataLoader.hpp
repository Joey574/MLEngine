#pragma once
#include "../NeuralNetwork/NeuralNetwork.hpp"

class DataLoader {
public:

    static Dataset LoadDataset(YAML::Node& config);
    static Dataset LoadMNIST(YAML::Node& config);
    static Dataset LoadFMNIST(YAML::Node& config);
    static Dataset LoadMandlebrot(YAML::Node& config);

    static void VisualizeMandlebrot(NeuralNetwork& model, const std::string& path, size_t width, size_t height);

private:
    // mnist / fmnist utils
    static int ReadBigInt(std::ifstream* f);

    static std::vector<float> RotateImage(const float* image, size_t width, size_t height, float deg);
    static std::vector<float> ScaleImage(const float* image, size_t width, size_t height, float scale);


    // mandlebrot utils
    static float InMandlebrot(double x, double y, size_t it);
    static void ComputeFourier(float* x, size_t series);

    static std::string ExpandPath(const std::string& path);
};