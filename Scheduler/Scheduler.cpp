#include "Scheduler.hpp"

void Scheduler::Initialize(YAML::Node& config) {
    if (config[Y_SCH_LRSCHEDULE]) {
        YAML::Node lr_config = config[Y_SCH_LRSCHEDULE];
        m_LRSchedule = (LRSchedule)lr_config[Y_SCH_SCHTYPE].as<int>();
    }

    AssignPtr();
}

void Scheduler::AssignPtr() {
    if (m_LRSchedule == LRSchedule::step_decay) {
        execute = &Scheduler::Execute<LRSchedule::step_decay>;
    } else if (m_LRSchedule == LRSchedule::on_plateau) {
        execute = &Scheduler::Execute<LRSchedule::on_plateau>;
    } else if (m_LRSchedule == LRSchedule::inv_time_decay) {
        execute = &Scheduler::Execute<LRSchedule::inv_time_decay>;
    } else {
        execute = &Scheduler::Execute<LRSchedule::none>;
    }
}
