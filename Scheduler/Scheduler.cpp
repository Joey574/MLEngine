#include "Scheduler.hpp"

void Scheduler::Initialize(YAML::Node& config) {
    if (config[Y_SCH_LRSCHEDULE]) {
        YAML::Node lr_config = config[Y_SCH_LRSCHEDULE];
        m_LRSchedule = (LRSchedule)lr_config[Y_SCH_SCHTYPE].as<int>();
    }


    
}