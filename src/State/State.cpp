#include "State.hpp"

int State::Load() {
    std::string file = path+"/"+name+"/.model";

    if (FileExists(file)) {
        supervisor->Load(path, name);
    } else {
        std::cerr << "Save not found\n";
        return 1;
    }

    return 0;
}

int State::Build() {
    return 0;
}

int State::Train() {
    history = supervisor->Train(history);

    // update history
    std::ofstream file(path+"/history.meta", std::ios::trunc);
    
    file << history.dump(4) << "\n";
    file.close();

    return 0;
}
