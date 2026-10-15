#include "DataLoader.hpp"
#include <cstdio>
#include <random>
#include <vector>
#include <omp.h>

std::vector<BoundaryCell> DataLoader::GenerateBoundaryCellsQuadTree(
    std::vector<BoundaryCell>& roots,
    size_t mandleDepth,
    size_t maxQuadDepth,
    float jumpEps,
    bool deSafe
) {
    if (roots.empty() || maxQuadDepth == 0) {
        return roots;
    }

    struct Node { double x, y, w, h; float v[4]; uint32_t depth; };

    const auto sample = [&](double x, double y) -> float {
        return InMandlebrot(x, y, mandleDepth);
    };

    std::vector<BoundaryCell> result;

    #pragma omp parallel
    {
        std::vector<BoundaryCell> local;
        std::vector<Node> stack;
        stack.reserve(3*maxQuadDepth + 8);

        #pragma omp for schedule(dynamic, 4) nowait
        for (size_t i = 0; i < roots.size(); i++) {
            const BoundaryCell& r = roots[i];
            const double rx1 = r.x + r.width;
            const double ry1 = r.y + r.height;

            stack.push_back({
                r.x, r.y, r.width, r.height,
                {
                    sample(r.x, r.y),
                    sample(rx1, r.y),
                    sample(r.x, ry1),
                    sample(rx1, ry1)
                }, 0
                });

            while (!stack.empty()) {
                const Node n = stack.back();
                stack.pop_back();

                if (n.depth >= maxQuadDepth) {
                    local.emplace_back(n.x, n.y, n.w, n.h);
                    continue;
                }

                const double hw = n.w * 0.5, hh = n.h * 0.5;
                const double xm = n.x + hw, x1 = n.x + n.w;
                const double ym = n.y + hh, y1 = n.y + n.h;

                const float bl = n.v[0], br = n.v[1], tl = n.v[2], tr = n.v[3];

                const float bm = sample(xm, n.y);
                const float lm = sample(n.x, ym);
                const float cm = sample(xm, ym);
                const float rm = sample(x1, ym);
                const float tm = sample(xm, y1);

                const uint32_t d = n.depth+1;
                const Node kids[4] = {
                    {n.x, n.y, hw, hh, {bl, bm, lm, cm}, d},
                    {xm, n.y, hw, hh, {bm, br, cm, rm}, d},
                    {n.x, ym, hw, hh, {lm, cm, tl, tm}, d},
                    {xm, ym, hw, hh, {cm, rm, tm, tr}, d},
                };

                for (const Node& k : kids) {
                    const float lo = std::min(std::min(k.v[0], k.v[1]), std::min(k.v[2], k.v[3]));
                    const float hi = std::max(std::max(k.v[0], k.v[1]), std::max(k.v[2], k.v[3]));
                    bool keep = (hi >= 1.0f && lo < 1.0f) || hi - lo > jumpEps;

                    if (!keep && deSafe) {
                        const double lb = DataLoader::MandleDistanceLowerBound(k.x + 0.5 * k.w, k.y + 0.5 * k.h, mandleDepth);
                        keep = lb >= 0.0 && lb <= 0.5 * std::hypot(k.w, k.h);
                    }

                    if (keep) {
                        stack.push_back(k);
                    }
                }
            }
        }

        #pragma omp critical
        result.insert(result.end(), local.begin(), local.end());
    }

    std::sort(result.begin(), result.end(), [](const BoundaryCell& a, const BoundaryCell& b) {
        return a.y != b.y ? a.y < b.y : a.x < b.x;
    });

    return result;
}

std::vector<BoundaryCell> DataLoader::GenerateBoundaryCells(
    double xmin,
    double xmax,
    double ymax,
    size_t xRes,
    size_t yRes,
    size_t depth,
    float jumpEps
) {
    const double dx = (xmax - xmin) / (double)xRes;
    const double dy = ymax / (double)yRes;
    const size_t stride = xRes+1;

    std::vector<float> lab(stride * (yRes+1));

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t h = 0; h <= yRes; h++) {
        float* r = &lab[h*stride];
        const double y = (double)h * dy;

        for (size_t w = 0; w <= xRes; w++) {
            r[w] = InMandlebrot(xmin+ (double)w*dx, y, depth);
        }
    }

    std::vector<BoundaryCell> cells;
    for (size_t h = 0; h < yRes; h++) {
        for (size_t w = 0; w < xRes; w++) {
            const float a = lab[h*stride + w];
            const float b = lab[h*stride + w+1];
            const float c = lab[(h+1)*stride + w];
            const float d = lab[(h+1)*stride + w+1];

            const float lo = std::min(std::min(a, b), std::min(c, d));
            const float hi = std::max(std::max(a, b), std::max(c, d));
            if ((hi >= 1.0f && lo < 1.0f) || hi - lo > jumpEps) {
                cells.emplace_back(xmin + (double)w*dx, double(h)*dy, dx, dy);
            }
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
