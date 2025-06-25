#include "Scheduler.hpp"
#include "Scheduler.tpp"

template void Scheduler::Execute<Scheduler::LRSchedule::step_decay>(YAML::Node&);
template void Scheduler::Execute<Scheduler::LRSchedule::on_plateau>(YAML::Node&);
template void Scheduler::Execute<Scheduler::LRSchedule::inv_time_decay>(YAML::Node&);
template void Scheduler::Execute<Scheduler::LRSchedule::none>(YAML::Node&);
