#include "DataLoader.hpp"

void DataLoader::Deform(size_t e) {
    std::mt19937 rng(SEED+22+e);

    // swap blocks in place
    for (size_t i = trainDataRows-1; i > 0; i--) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);

        if (i != j) {
            auto block_id = trainData.begin() + (i*trainDataCols);
            auto block_jd = trainData.begin() + (j*trainDataCols);

            auto block_il = trainLabels.begin() + (i*trainLabelCols);
            auto block_jl = trainLabels.begin() + (j*trainLabelCols);

            std::swap_ranges(block_id, block_id+trainDataCols, block_jd);
            std::swap_ranges(block_il, block_il+trainLabelCols, block_jl);
        }
    }
}
