#include "NeuralNetwork.hpp"


void NeuralNetwork::FitStart(nlohmann::json& history, size_t e, size_t bs, float lr) {
    auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    auto local = std::chrono::zoned_time{std::chrono::current_zone(), now};
    history[START] = std::format("{:%F %T}", local);

    // since program can be interupted, epochs is a running total of epochs completed
    history[EPOCHS] = 0;
    history[BATCHSIZE] = bs;
    history[LEARNRATE] = lr;
}
void NeuralNetwork::FitEnd(nlohmann::json& history, std::chrono::system_clock::time_point starttime) {
    auto traintime = std::chrono::high_resolution_clock::now() - starttime;

    // store train time
    {
        using namespace std::chrono;

        auto hour = duration_cast<hours>(traintime);
        traintime -= hour;
        auto minute = duration_cast<minutes>(traintime);
        traintime -= minute;
        auto second = duration_cast<seconds>(traintime);
        traintime -= second;
        auto ms = duration_cast<milliseconds>(traintime);

        std::string fdur;
        if (hour.count() > 0) {
            fdur = std::format("{}h {}m {}s", hour.count(), minute.count(), second.count());
        } else if (minute.count() > 0) {
            fdur = std::format("{}m {}s {}ms", minute.count(), second.count(), ms.count());        
        } else if (second.count() > 0) {
            fdur = std::format("{}s {}ms", second.count(), ms.count());
        } else {
            fdur = std::format("{}ms", ms.count());
        }
        history[TRAINTIME] = fdur;
    }

    // store time training completed
    auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    auto local = std::chrono::zoned_time{std::chrono::current_zone(), now};
    history[FINISH] = std::format("{:%F %T}", local);
}

void NeuralNetwork::EpochStart(nlohmann::json& history) {
    
}
void NeuralNetwork::EpochEnd(nlohmann::json& history, const std::string& res, double ns, size_t e) {
    history[EPOCHS] = (int)history[EPOCHS] + 1;

    // format epoch time
    std::string fdur;
    {
        using namespace std::chrono;
        auto duration = nanoseconds(static_cast<long long>(ns));

        auto hour = duration_cast<hours>(duration);
        duration -= hour;
        auto minute = duration_cast<minutes>(duration);
        duration -= minute;
        auto second = duration_cast<seconds>(duration);
        duration -= second;
        auto ms = duration_cast<milliseconds>(duration);

        if (hour.count() > 0) {
            fdur = std::format("{}h {}m {}s", hour.count(), minute.count(), second.count());
        } else if (minute.count() > 0) {
            fdur = std::format("{}m {}s {}ms", minute.count(), second.count(), ms.count());        
        } else if (second.count() > 0) {
            fdur = std::format("{}s {}ms", second.count(), ms.count());
        } else {
            fdur = std::format("{}ms", ms.count());
        }
    }
    
    std::string em = "Epoch "+std::to_string(e)+": "+fdur;
    printf("%-25s %s\n", em.data(), res.data());
}
