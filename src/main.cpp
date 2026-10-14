#include "network/state.hpp"
#include <glaze/core/opts.hpp>
#include <glaze/glaze.hpp>
#include <string>
#include <iostream>

int main() {
    NetworkState state;

    std::string buf{};
    auto ec = glz::write<glz::opts{.prettify=true}>(state, buf);

    std::cout << buf << "\n";
}
