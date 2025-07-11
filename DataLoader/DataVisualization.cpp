#include "DataLoader.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Dependencies/stb_image_write.h"

void DataLoader::VisualizeTerminalMNISTLike(const float* image, size_t width, size_t height) {
    for (size_t h = 0; h < height; h++) {
        for (size_t w = 0; w < width; w++) {
            std::cout << (image[h*width+w] < 0.5f ? 0 : 1) << " ";
        } std::cout << "\n";
    }
}

void DataLoader::SaveMandleImage(const std::string& path, const float* __restrict points, size_t width, size_t height) {
    std::vector<u_char> image(width*height*3);
    std::string dir = path.substr(0, path.find_last_of("/"));

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            const size_t i = (y*width+x) * 3;

            image[i+0] = points[y*width+x] * 255.0f;
            image[i+1] = points[y*width+x] > 0.95f ? 255.0f : 0.0f;
            image[i+2] = points[y*width+x] > 0.95f ? 255.0f : points[y*width+x];
        }
    }

    // create images directory
	if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }

    if (!stbi_write_png(path.c_str(), width, height, 3, image.data(), width*3)) {
        std::cerr << "Failed to write image to " << path << "\n";
    }
}