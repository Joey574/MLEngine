#include "DataLoader.hpp"
#include <random>
#include <vector>

std::vector<BoundaryCell> DataLoader::GenerateBoundaryCells(
    double xmin,
    double xmax,
    double ymin,
    double ymax,
    size_t width,
    size_t height,
    size_t depth
) {
    std::vector<BoundaryCell> cells;

    const double dx = (xmax - xmin) / (double)width;
    const double dy = (ymax - ymin) / (double)height;
    const size_t stride = width+1;

    std::vector<uint8_t> inside((width+1)*(height+1));

    #pragma omp parallel for simd schedule(static)
    for (size_t h = 0; h <= height; h++) {
        for (size_t w = 0; w <= width; w++) {
            const double x = xmin + w * dx;
            const double y = ymin + h * dy;
            const float m = DataLoader::InMandlebrot(x, y, depth);

            inside[h*stride + w] = (m >= 1.0f);
        }
    }

    for (size_t h = 0; h < height; h++) {
        for (size_t w = 0; w < width; w++) {
            const bool a = inside[h*stride + w];
            const bool b = inside[h*stride +w+1];
            const bool c = inside[(h+1)*stride + w];
            const bool d = inside[(h+1)*stride + w+1];

            // all corners agree
            if (a == b && a == c && a == d) {
                continue;
            }

            cells.push_back({
                xmin + w*dx,
                ymin + h*dy,
                dx, dy
            });
        }
    }

    return cells;
}

std::pair<double, double> DataLoader::SampleBoundary(std::mt19937& gen, const std::vector<BoundaryCell>& cells) {
    std::uniform_int_distribution<size_t> cellDist(0, cells.size()-1);
    std::uniform_real_distribution<double> offset(0.0, 1.0);

    const BoundaryCell& cell = cells[cellDist(gen)];
    const double x = cell.x + offset(gen) * cell.width;
    const double y = cell.y + offset(gen) * cell.height;
    return {x, y};
}
