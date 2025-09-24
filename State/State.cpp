#include "State.hpp"

void State::Load() {
    std::string file = path+"/"+name+"/.model";
    if (FileExists(file)) {
        supervisor->Load(path, name);
    } else {
        std::cout << "Save not found\n";
        exit(1);
    }
}

void State::Build() {

}

void State::Train() {
    history = supervisor->Train(history);

    // update history
    std::ofstream file(path+"/"+"history.meta", std::ios::trunc);
    
    file << history.dump(4) << "\n";
    file.close();
}
