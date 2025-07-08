#pragma once

/* @brief

*/
struct Scheduler {
public:

    enum class LRSchedule {
        none, step_decay, on_plateau, inv_time_decay
    };

    Scheduler() { memset(this, 0, sizeof(Scheduler)); }


    void Initialize(YAML::Node& config);
    void (Scheduler::*execute)(YAML::Node&);

    static std::string ParseLRName(LRSchedule lrsch);
    static LRSchedule ParseLRType(const std::string& lr);

private:

    LRSchedule m_LRSchedule;

    void AssignPtr();
    template <LRSchedule lr_sch> void Execute(YAML::Node& trainingData);

    void LRStepDecay(YAML::Node& trainingData);
    void LROnPlateau(YAML::Node& trainingData);
    void LRInvTimeDecay(YAML::Node& trainingData);
};