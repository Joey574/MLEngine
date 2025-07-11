#pragma once

/* @brief

*/
struct Scheduler {
public:

    enum class VisSchedule {
        none, step_vis
    };

    Scheduler() { memset(this, 0, sizeof(Scheduler)); }


    void Initialize(YAML::Node& config);
    void (Scheduler::*execute)(YAML::Node&);

private:
    void AssignPtr();

    void StepVis(YAML::Node& config);
};