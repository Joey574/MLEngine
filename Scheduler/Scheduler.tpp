#include "Scheduler.hpp"

template <Scheduler::LRSchedule lr_sch>
void Scheduler::Execute(YAML::Node& trainingData) {
    if constexpr (lr_sch == LRSchedule::step_decay) {
        LRStepDecay(trainingData);
    } else if constexpr (lr_sch == LRSchedule::on_plateau) {
        LROnPlateau(trainingData);
    } else if constexpr (lr_sch == LRSchedule::inv_time_decay) {
        LRInvTimeDecay(trainingData);        
    }
}