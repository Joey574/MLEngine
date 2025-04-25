#include "NeuralNetwork.hpp"


void NeuralNetwork::FitStart(nlohmann::json& history, size_t e, size_t bs, float lr) {
    auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    auto local = std::chrono::zoned_time{std::chrono::current_zone(), now};
    history["Start"] = std::format("{:%F %T}", local);

    history["Epochs"] = e;
    history["Batch Size"] = bs;
    history["Learning Rate"] = lr;
}
void NeuralNetwork::FitEnd(nlohmann::json& history, std::chrono::system_clock::time_point starttime) {
    auto traintime = std::chrono::high_resolution_clock::now() - starttime;
	history["Train Time"] = std::format("{:%Hh %Mm %Ss}", traintime);

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
        history["Train Time"] = fdur;
    }

    // store time training completed
    auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    auto local = std::chrono::zoned_time{std::chrono::current_zone(), now};
    history["Finish"] = std::format("{:%F %T}", local);
}

void NeuralNetwork::EpochStart(nlohmann::json& history) {

}
void NeuralNetwork::EpochEnd(nlohmann::json& history) {
    
}
