#pragma once
#include <chrono>
#include <cstddef>
#include <ctime>

struct NetworkState {
    public:
        inline void IncCurrentEpoch() {
            this->currentEpoch++;
        }

        inline bool TrainingComplete() {
            return this->currentEpoch > this->plannedEpochs ||
                this->Elapsed() > this->maxTrainDuration;
        }

        inline std::time_t Elapsed() {
            auto now = std::chrono::system_clock::now();
            return std::chrono::system_clock::to_time_t(now) - trainingStart;
        }

        size_t currentEpoch = 0;
        size_t plannedEpochs = 0;

        size_t batchSize = 0;
        float learningRate = 0;

        std::time_t fastestEpochDuration = 0;
        std::time_t slowestEpochDuration = 0;
        std::time_t averageEpochDuration = 0;

        std::time_t maxTrainDuration = 0;

        std::time_t trainingStart = 0;
        std::time_t trainingEnd = 0;
};
