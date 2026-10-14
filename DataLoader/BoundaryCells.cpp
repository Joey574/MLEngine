#include "DataLoader.hpp"
#include <cstdio>
#include <random>
#include <vector>
#include <omp.h>

std::vector<BoundaryCell> DataLoader::GenerateBoundaryCellsQuadTree(
    std::vector<BoundaryCell>& cells,
    double xmin,
    double xmax,
    double ymin,
    double ymax,
    size_t mandleDepth,
    size_t maxQuadDepth
) {
    if (cells.empty()) {
        return {};
    }

    // The input cells already define the search region.
    (void)xmin;
    (void)xmax;
    (void)ymin;
    (void)ymax;

    // Corner bit layout:
    //
    // TL ---- TR
    // |        |
    // |        |
    // BL ---- BR
    //
    constexpr uint8_t BL = 1 << 0;
    constexpr uint8_t BR = 1 << 1;
    constexpr uint8_t TL = 1 << 2;
    constexpr uint8_t TR = 1 << 3;

    constexpr uint8_t ALL = BL | BR | TL | TR;

    struct QuadNode {
        BoundaryCell cell;
        uint8_t corners;
        size_t depth;
    };

    const auto sample = [&](double x, double y) -> bool {
        return DataLoader::InMandlebrot(
            x, y, mandleDepth
        ) >= 1.0f;
    };

    const auto getCorners = [&](const BoundaryCell& cell) -> uint8_t {
        const double x0 = cell.x;
        const double x1 = cell.x + cell.width;
        const double y0 = cell.y;
        const double y1 = cell.y + cell.height;

        return
            (sample(x0, y0) ? BL : 0) |
            (sample(x1, y0) ? BR : 0) |
            (sample(x0, y1) ? TL : 0) |
            (sample(x1, y1) ? TR : 0);
    };

    const int threadCount = omp_get_max_threads();

    std::vector<std::vector<BoundaryCell>> outputs(threadCount);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        auto& output = outputs[tid];

        std::vector<QuadNode> stack;
        stack.reserve(64);

        #pragma omp for schedule(dynamic, 16)
        for (size_t i = 0; i < cells.size(); i++) {
            BoundaryCell cell = cells[i];

            // Only process cells in the upper half-plane.
            // Cells below the x-axis are reconstructed by mirroring.
            const double top = cell.y + cell.height;

            if (top <= 0.0) {
                continue;
            }

            // Clip cells that cross the x-axis.
            if (cell.y < 0.0) {
                cell.height = top;
                cell.y = 0.0;
            }

            if (cell.width <= 0.0 || cell.height <= 0.0) {
                continue;
            }

            stack.clear();

            stack.push_back({
                cell,
                getCorners(cell),
                0
            });

            while (!stack.empty()) {
                const QuadNode node = stack.back();
                stack.pop_back();

                const BoundaryCell& c = node.cell;

                // Reached the target depth.
                if (node.depth >= maxQuadDepth) {
                    output.push_back(c);
                    continue;
                }

                const double hw = c.width * 0.5;
                const double hh = c.height * 0.5;

                const double x0 = c.x;
                const double xm = x0 + hw;
                const double x1 = x0 + c.width;

                const double y0 = c.y;
                const double ym = y0 + hh;
                const double y1 = y0 + c.height;

                // Reuse the four parent corner classifications.
                const bool bl = node.corners & BL;
                const bool br = node.corners & BR;
                const bool tl = node.corners & TL;
                const bool tr = node.corners & TR;

                // Only five new evaluations are necessary:
                //
                // TL ---- TM ---- TR
                // |       |       |
                // LM ---- CM ---- RM
                // |       |       |
                // BL ---- BM ---- BR

                const bool bm = sample(xm, y0);
                const bool lm = sample(x0, ym);
                const bool cm = sample(xm, ym);
                const bool rm = sample(x1, ym);
                const bool tm = sample(xm, y1);

                // Generate the four child corner masks.
                const uint8_t masks[4] = {
                    // Bottom-left
                    static_cast<uint8_t>(
                        (bl ? BL : 0) |
                        (bm ? BR : 0) |
                        (lm ? TL : 0) |
                        (cm ? TR : 0)
                    ),

                    // Bottom-right
                    static_cast<uint8_t>(
                        (bm ? BL : 0) |
                        (br ? BR : 0) |
                        (cm ? TL : 0) |
                        (rm ? TR : 0)
                    ),

                    // Top-left
                    static_cast<uint8_t>(
                        (lm ? BL : 0) |
                        (cm ? BR : 0) |
                        (tl ? TL : 0) |
                        (tm ? TR : 0)
                    ),

                    // Top-right
                    static_cast<uint8_t>(
                        (cm ? BL : 0) |
                        (rm ? BR : 0) |
                        (tm ? TL : 0) |
                        (tr ? TR : 0)
                    )
                };

                const BoundaryCell children[4] = {
                    {x0, y0, hw, hh},
                    {xm, y0, hw, hh},
                    {x0, ym, hw, hh},
                    {xm, ym, hw, hh}
                };

                for (size_t j = 0; j < 4; j++) {
                    const uint8_t mask = masks[j];

                    // All four corners agree: no detected boundary.
                    if (mask == 0 || mask == ALL) {
                        continue;
                    }

                    stack.push_back({
                        children[j],
                        mask,
                        node.depth + 1
                    });
                }
            }
        }
    }

    // Merge thread-local results.
    size_t total = 0;

    for (const auto& output : outputs) {
        total += output.size();
    }

    std::vector<BoundaryCell> result;
    result.reserve(total * 2);

    for (const auto& output : outputs) {
        result.insert(
            result.end(),
            output.begin(),
            output.end()
        );
    }

    // Mirror across the x-axis.
    const size_t upperCount = result.size();

    for (size_t i = 0; i < upperCount; i++) {
        const BoundaryCell& c = result[i];

        result.emplace_back(
            c.x,
            -c.y - c.height,
            c.width,
            c.height
        );
    }

    return result;
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

    // mandlebrot is mirrored across the x axis
    ymin = 0.0f;

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

    // duplicate cells and negate y value of copies
    size_t n = cells.size();
    cells.resize(n*2);
    for (size_t i = 0; i < n; i++) {
        cells[n+i] = cells[i];
        cells[n+i].y = -cells[n+i].y;
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
