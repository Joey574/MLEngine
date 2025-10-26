#include "State.hpp"

bool State::ModelExists() const {
    return FileExists(path+"/"+name+".model");
}

bool State::IsValid() const {
    return true;
}
