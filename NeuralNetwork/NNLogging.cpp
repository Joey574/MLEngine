#include "NeuralNetwork.hpp"

void NeuralNetwork::FitStart(nlohmann::json& history, size_t e, size_t bs, float lr) {
    auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    auto local = std::chrono::zoned_time{std::chrono::current_zone(), now};
    history[START] = std::format("{:%F %T}", local);

    m_epoch_since_improvement = 0;

    // since program can be interupted, epochs is a running total of epochs completed
    history[EPOCHS] = 0;
    history[BATCHSIZE] = bs;
    history[LEARNRATE] = lr;
}
void NeuralNetwork::FitEnd(nlohmann::json& history, std::chrono::system_clock::time_point starttime) {
    auto traintime = std::chrono::high_resolution_clock::now() - starttime;

    history[TRAINTIME] = CleanTime(traintime);

    // store time training completed
    auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    auto local = std::chrono::zoned_time{std::chrono::current_zone(), now};
    history[FINISH] = std::format("{:%F %T}", local);

    // average out sum of train time
    history[AVGEPOCH] = CleanTime(std::chrono::nanoseconds(static_cast<long long>(traintime.count()/(float)history[EPOCHS])));

    // clean format of fastest and slowest epoch
    history[SLOWESTEPOCH] = CleanTime(std::chrono::nanoseconds(static_cast<long long>(history[SLOWESTEPOCH])));
    history[FASTESTEPOCH] = CleanTime(std::chrono::nanoseconds(static_cast<long long>(history[FASTESTEPOCH])));
}

void NeuralNetwork::EpochStart(nlohmann::json& history) {
    
}
void NeuralNetwork::EpochEnd(nlohmann::json& history, const std::string& res, double ns, size_t e) {
    history[EPOCHS] = (int)history[EPOCHS] + 1;

    // fastest epoch
    if (!history.contains(FASTESTEPOCH)) {
        history[FASTESTEPOCH] = ns; 
    } else {
        if (ns < history[FASTESTEPOCH]) {
            history[FASTESTEPOCH] = ns;
        }
    }

    // slowest epoch
    if (!history.contains(SLOWESTEPOCH)) {
        history[SLOWESTEPOCH] = ns; 
    } else {
        if (ns > history[SLOWESTEPOCH]) {
            history[SLOWESTEPOCH] = ns;
        }
    }

    std::string fdur = CleanTime(std::chrono::nanoseconds(static_cast<long long>(ns)));
    std::string em = "Epoch "+std::to_string(e)+": "+fdur;
    printf("%-25s %s\n", em.data(), res.data());
}

std::string NeuralNetwork::CleanTime(std::chrono::nanoseconds time) {
    using namespace std::chrono;

    auto hour = duration_cast<hours>(time);
    time -= hour;
    auto minute = duration_cast<minutes>(time);
    time -= minute;
    auto second = duration_cast<seconds>(time);
    time -= second;
    auto ms = duration_cast<milliseconds>(time);

    std::string ftime;
    if (hour.count() > 0) {
        ftime = std::format("{}h {}m {}s", hour.count(), minute.count(), second.count());
    } else if (minute.count() > 0) {
        ftime = std::format("{}m {}s {}ms", minute.count(), second.count(), ms.count());        
    } else if (second.count() > 0) {
        ftime = std::format("{}s {}ms", second.count(), ms.count());
    } else {
        ftime = std::format("{}ms", ms.count());
    }

    return ftime;
}