#include "DataLoader.hpp"
#include <cstdio>
#include <random>
#include <vector>
#include <omp.h>

std::vector<BoundaryCell> DataLoader::GenerateBoundaryCellsAdaptive(
    double xmin,
    double xmax,
    double ymax,
    size_t xRes,
    size_t yRes,
    size_t depth,
    float jumpEps,
    size_t leafDim,
    size_t minDepth,
    bool deSafe
) {
    const double dx = (xmax - xmin) / (double)xRes;
    const double dy = ymax / (double)yRes;
    leafDim = std::max<size_t>(leafDim, 1);

    // Lattice space: (x0,y0) is the corner index, (w,h) the size in final cells.
    // v = corner samples: {(x0,y0), (x1,y0), (x0,y1), (x1,y1)}
    struct Node { size_t x0, y0, w, h; float v[4]; uint32_t depth; };

    const auto sample = [&](size_t ix, size_t iy) -> float {
        return InMandlebrot(xmin + (double)ix * dx, (double)iy * dy, depth);
    };

    const auto straddles = [&](const float* v) {
        const float lo = std::min(std::min(v[0], v[1]), std::min(v[2], v[3]));
        const float hi = std::max(std::max(v[0], v[1]), std::max(v[2], v[3]));
        return (hi >= 1.0f && lo < 1.0f) || hi - lo > jumpEps;
    };

    const auto keepNode = [&](const Node& n) {
        if (n.depth <= minDepth) return true;
        if (straddles(n.v)) return true;
        if (!deSafe) return false;
        const double cx = xmin + ((double)n.x0 + 0.5 * (double)n.w) * dx;
        const double cy = ((double)n.y0 + 0.5 * (double)n.h) * dy;
        const double lb = DataLoader::MandleDistanceLowerBound(cx, cy, depth);
        return lb >= 0.0 && lb <= 0.5 * std::hypot((double)n.w * dx, (double)n.h * dy);
    };

    // ---- Phase 1: quadtree prune (level-synchronous so every level is parallel) ----
    std::vector<Node> frontier;
    frontier.push_back({
        0, 0, xRes, yRes,
        { sample(0, 0), sample(xRes, 0), sample(0, yRes), sample(xRes, yRes) },
        0
    });
    std::vector<Node> leaves;

    while (!frontier.empty()) {
        std::vector<Node> next;

        #pragma omp parallel
        {
            std::vector<Node> localNext, localLeaves;

            #pragma omp for schedule(dynamic, 8) nowait
            for (size_t i = 0; i < frontier.size(); i++) {
                const Node& n = frontier[i];

                const bool splitX = n.w > leafDim;
                const bool splitY = n.h > leafDim;
                if (!splitX && !splitY) {
                    localLeaves.push_back(n);
                    continue;
                }

                const size_t nx = splitX ? 2 : 1;
                const size_t ny = splitY ? 2 : 1;
                const size_t xe[3] = { n.x0, splitX ? n.x0 + n.w / 2 : n.x0 + n.w, n.x0 + n.w };
                const size_t ye[3] = { n.y0, splitY ? n.y0 + n.h / 2 : n.y0 + n.h, n.y0 + n.h };

                // Corner samples are inherited; only new edge midpoints/center get sampled.
                float val[3][3];
                for (size_t j = 0; j <= ny; j++) {
                    for (size_t k = 0; k <= nx; k++) {
                        const bool cornerX = (k == 0 || k == nx);
                        const bool cornerY = (j == 0 || j == ny);
                        val[j][k] = (cornerX && cornerY)
                            ? n.v[(j == ny ? 2 : 0) + (k == nx ? 1 : 0)]
                            : sample(xe[k], ye[j]);
                    }
                }

                for (size_t j = 0; j < ny; j++) {
                    for (size_t k = 0; k < nx; k++) {
                        const Node c{
                            xe[k], ye[j], xe[k+1] - xe[k], ye[j+1] - ye[j],
                            { val[j][k], val[j][k+1], val[j+1][k], val[j+1][k+1] },
                            n.depth+1
                        };
                        if (keepNode(c)) localNext.push_back(c);
                    }
                }
            }

            #pragma omp critical
            {
                next.insert(next.end(), localNext.begin(), localNext.end());
                leaves.insert(leaves.end(), localLeaves.begin(), localLeaves.end());
            }
        }

        frontier.swap(next);
    }

    // ---- Phase 2: uniform boundary-cell generation inside each surviving leaf ----
    std::vector<BoundaryCell> cells;

    #pragma omp parallel
    {
        std::vector<BoundaryCell> local;
        std::vector<float> lab;

        #pragma omp for schedule(dynamic, 4) nowait
        for (size_t i = 0; i < leaves.size(); i++) {
            const Node& L = leaves[i];
            const size_t stride = L.w + 1;
            lab.resize(stride * (L.h + 1));

            for (size_t h = 0; h <= L.h; h++) {
                for (size_t w = 0; w <= L.w; w++) {
                    lab[h * stride + w] = sample(L.x0 + w, L.y0 + h);
                }
            }

            for (size_t h = 0; h < L.h; h++) {
                for (size_t w = 0; w < L.w; w++) {
                    const float v[4] = {
                        lab[h * stride + w],       lab[h * stride + w + 1],
                        lab[(h+1) * stride + w],   lab[(h+1) * stride + w + 1]
                    };
                    if (straddles(v)) {
                        local.emplace_back(
                            xmin + (double)(L.x0 + w) * dx,
                            (double)(L.y0 + h) * dy,
                            dx, dy);
                    }
                }
            }
        }

        #pragma omp critical
        cells.insert(cells.end(), local.begin(), local.end());
    }

    std::sort(cells.begin(), cells.end(), [](const BoundaryCell& a, const BoundaryCell& b) {
        return a.y != b.y ? a.y < b.y : a.x < b.x;
    });

    return cells;
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
