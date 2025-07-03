#include "DataLoader.hpp"

void DataLoader::VisualizeTerminalMNISTLike(const float* image, size_t width, size_t height) {
    for (size_t h = 0; h < height; h++) {
        for (size_t w = 0; w < width; w++) {
            std::cout << (image[h*width+w] < 0.5f ? 0 : 1) << " ";
        } std::cout << "\n";
    }
}
