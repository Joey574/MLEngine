#include "DataLoader.hpp"
#include <cstdio>
#include <random>
#include <vector>
#include <omp.h>

// std::vector<BoundaryCell> DataLoader::GenerateBoundaryCellsQuadTree(
//     double xmin,
//     double xmax,
//     double ymin,
//     double ymax,
//     size_t xRes,
//     size_t yRes,
//     size_t mandleDepth,
//     size_t maxQuadDepth,
//     size_t quadDepth
// ) {
//     std::vector<BoundaryCell> cells;

//     const double dx = (xmax - xmin) / (double)xRes;
//     const double dy = (ymax - ymin) / (double)yRes;
//     const size_t stride = xRes+1;

//     std::vector<uint8_t> inside((xRes+1)*(yRes+1));

//     #pragma omp parallel for schedule(dynamic)
//     for (size_t h = 0; h <= yRes; h++) {
//         for (size_t w = 0; w <= xRes; w++) {
//             const double x = xmin + w * dx;
//             const double y = ymin + h * dy;
//             const float m = DataLoader::InMandlebrot(x, y, mandleDepth);

//             inside[h*stride + w] = (m >= 1.0f);
//         }
//     }

//     for (size_t h = 0; h < yRes; h++) {
//         for (size_t w = 0; w < xRes; w++) {
//             const bool a = inside[h*stride + w];
//             const bool b = inside[h*stride +w+1];
//             const bool c = inside[(h+1)*stride + w];
//             const bool d = inside[(h+1)*stride + w+1];

//             // all corners agree
//             if (a == b && a == c && a == d) {
//                 continue;
//             }

//             // recurse up to some depth
//             if (quadDepth < maxQuadDepth) {
//                 const double dh = h;
//                 const double dw = w;

//                 auto newCells = GenerateBoundaryCellsQuadTree(
//                     xmin + dw*dx,
//                     xmin + dw*dx + dx,
//                     ymin + dh*dy,
//                     ymin + dh*dy + dy,
//                     xRes,
//                     yRes,
//                     mandleDepth,
//                     maxQuadDepth,
//                     quadDepth+1
//                 );

//                 cells.insert(cells.end(), newCells.begin(), newCells.end());
//             } else {
//                 cells.emplace_back(
//                     xmin + w*dx,
//                     ymin + h*dy,
//                     dx, dy
//                 );
//             }
//         }
//     }

//     return cells;
// }

std::vector<BoundaryCell> DataLoader::GenerateBoundaryCellsQuadTree(
    double xmin,
    double xmax,
    double ymin,
    double ymax,
    size_t xRes,
    size_t yRes,
    size_t mandleDepth,
    size_t maxQuadDepth,
    size_t quadDepth
) {
    const size_t stride = xRes + 1;

    // Grid-samples one cell at xRes x yRes and appends its boundary subcells to `out`.
    auto subdivide = [&](const BoundaryCell& cell, std::vector<uint8_t>& inside, std::vector<BoundaryCell>& out, bool parallelInner) {
        const double dx = cell.width  / (double)xRes;
        const double dy = cell.height / (double)yRes;

        #pragma omp parallel for schedule(dynamic) if(parallelInner)
        for (size_t h = 0; h <= yRes; h++) {
            for (size_t w = 0; w <= xRes; w++) {
                const double x = cell.x + w * dx;
                const double y = cell.y + h * dy;
                inside[h*stride + w] = (DataLoader::InMandlebrot(x, y, mandleDepth) >= 1.0f);
            }
        }

        for (size_t h = 0; h < yRes; h++) {
            for (size_t w = 0; w < xRes; w++) {
                const bool a = inside[h*stride + w];
                const bool b = inside[h*stride + w + 1];
                const bool c = inside[(h+1)*stride + w];
                const bool d = inside[(h+1)*stride + w + 1];

                // all corners agree
                if (a == b && a == c && a == d) {
                    continue;
                }

                out.emplace_back(
                    cell.x + w * dx,
                    cell.y + h * dy,
                    dx, dy
                );
            }
        }
    };

    std::vector<BoundaryCell> frontier{BoundaryCell{ xmin, ymin, xmax - xmin, ymax - ymin }};

    for (size_t depth = quadDepth; depth <= maxQuadDepth && !frontier.empty(); depth++) {
        std::vector<std::vector<BoundaryCell>> results(frontier.size());
        const bool parallelInner = frontier.size() < (size_t)omp_get_max_threads();

        #pragma omp parallel if(!parallelInner)
        {
            std::vector<uint8_t> inside((xRes + 1) * (yRes + 1));

            #pragma omp for schedule(dynamic, 1)
            for (size_t i = 0; i < frontier.size(); i++) {
                subdivide(frontier[i], inside, results[i], parallelInner);
            }
        }

        size_t total = 0;
        for (const auto& r : results) {
            total += r.size();
        }

        std::vector<BoundaryCell> next;
        next.reserve(total);
        for (auto& r : results) {
            next.insert(next.end(), r.begin(), r.end());
        }

        frontier = std::move(next);
    }

    return frontier;
}

std::vector<BoundaryCell> DataLoader::GenerateBoundaryCells(
    double xmin,
    double xmax,
    double ymin,
    double ymax,
    size_t xRes,
    size_t yRes,
    size_t depth
) {
    std::vector<BoundaryCell> cells;

    const double dx = (xmax - xmin) / (double)xRes;
    const double dy = (ymax - ymin) / (double)yRes;
    const size_t stride = xRes+1;

    std::vector<uint8_t> inside((xRes+1)*(yRes+1));

    #pragma omp parallel for simd schedule(static)
    for (size_t h = 0; h <= yRes; h++) {
        for (size_t w = 0; w <= xRes; w++) {
            const double x = xmin + w * dx;
            const double y = ymin + h * dy;
            const float m = DataLoader::InMandlebrot(x, y, depth);

            inside[h*stride + w] = (m >= 1.0f);
        }
    }

    for (size_t h = 0; h < yRes; h++) {
        for (size_t w = 0; w < xRes; w++) {
            const bool a = inside[h*stride + w];
            const bool b = inside[h*stride +w+1];
            const bool c = inside[(h+1)*stride + w];
            const bool d = inside[(h+1)*stride + w+1];

            // all corners agree
            if (a == b && a == c && a == d) {
                continue;
            }

            cells.emplace_back(
                xmin + w*dx,
                ymin + h*dy,
                dx, dy
            );
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
