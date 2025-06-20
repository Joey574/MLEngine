#include "NeuralNetwork.hpp"

void NeuralNetwork::FitStart(nlohmann::json& history, size_t e, size_t bs, float lr) {
    auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    auto local = std::chrono::zoned_time{std::chrono::current_zone(), now};
    history[J_START] = std::format("{:%F %T}", local);

    m_epoch_since_improvement = 0;

    // since program can be interupted, epochs is a running total of epochs completed
    history[J_EPOCHS] = 0;
    history[J_BATCHSIZE] = bs;
    history[J_LEARNRATE] = lr;
}
void NeuralNetwork::FitEnd(nlohmann::json& history, std::chrono::system_clock::time_point starttime) {
    auto traintime = std::chrono::high_resolution_clock::now() - starttime;

    history[J_TRAINTIME] = CleanTime(traintime);

    // store time training completed
    auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    auto local = std::chrono::zoned_time{std::chrono::current_zone(), now};
    history[J_FINISH] = std::format("{:%F %T}", local);

    // average out sum of train time
    history[J_AVGEPOCH] = CleanTime(std::chrono::nanoseconds(static_cast<long long>(traintime.count()/(float)history[J_EPOCHS])));

    // clean format of fastest and slowest epoch
    history[J_SLOWESTEPOCH] = CleanTime(std::chrono::nanoseconds(static_cast<long long>(history[J_SLOWESTEPOCH])));
    history[J_FASTESTEPOCH] = CleanTime(std::chrono::nanoseconds(static_cast<long long>(history[J_FASTESTEPOCH])));
}

void NeuralNetwork::EpochStart(nlohmann::json& history) {
    
}
void NeuralNetwork::EpochEnd(nlohmann::json& history, const std::string& res, double ns, size_t e) {
    history[J_EPOCHS] = (int)history[J_EPOCHS] + 1;

    // fastest epoch
    if (!history.contains(J_FASTESTEPOCH)) {
        history[J_FASTESTEPOCH] = ns; 
    } else {
        if (ns < history[J_FASTESTEPOCH]) {
            history[J_FASTESTEPOCH] = ns;
        }
    }

    // slowest epoch
    if (!history.contains(J_SLOWESTEPOCH)) {
        history[J_SLOWESTEPOCH] = ns; 
    } else {
        if (ns > history[J_SLOWESTEPOCH]) {
            history[J_SLOWESTEPOCH] = ns;
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
std::string NeuralNetwork::CleanSize(size_t bytes) {
    long double dbytes = bytes;
    const double gb = 1e9;
    const double mb = 1e6;
    const double kb = 1e3;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    if (dbytes / gb > 1.00) {
        oss << dbytes / gb << " gb";
    } else if (dbytes / mb > 1.00) {
        oss << dbytes / mb << " mb";
    } else if (dbytes / kb > 1.00) {
        oss << dbytes / kb << " kb";
    } else {
        oss << dbytes << " b";
    }

    return oss.str();
}