#pragma once

struct Scheduler {
public:

    Scheduler() { memset(this, 0, sizeof(Scheduler)); }

    enum class LRSchedule {
        none, step_decay, on_plateau, inv_time_decay
    };


    void Initialize(YAML::Node& config);
    void (Scheduler::*execute)(YAML::Node&);

private:

    LRSchedule m_LRSchedule;

    void AssignPtr();
    template <LRSchedule lr_sch> void Execute(YAML::Node& trainingData);

};