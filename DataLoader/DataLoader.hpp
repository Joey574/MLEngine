#pragma once

class DataLoader {
public:

    static Dataset LoadDataset(YAML::Node& config);
    static Dataset LoadMNIST(YAML::Node& config);
    static Dataset LoadFMNIST(YAML::Node& config);
    static Dataset LoadMandlebrot(YAML::Node& config);

private:
    // mnist / fmnist utils
    static int ReadBigInt(std::ifstream* f);
    static std::vector<float> RotateImage(const float* image, size_t width, size_t height, float deg);

    // mandlebrot utils
    static float InMandlebrot(double x, double y, size_t it);
    static void ComputeFourier(float* x, size_t series);

    static std::string ExpandPath(const std::string& path);
};