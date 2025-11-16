#include <execinfo.h>
#include <cxxabi.h>

#include "State/State.hpp"

void handleInterupt(int signum) {
    std::cout << "\nProgram will exit shortly\n";
    KEEPRUNNING = false;
}
void segv(int signum) {
    void *array[30];
    int size = backtrace(array, 30);
    char **messages = backtrace_symbols(array, size);

    std::cerr << "Signal " << signum << " stack trace:\n";
    for (int i = 0; i < size; ++i) {
        std::cerr << "[" << i << "] ";
        char *mangled_name = nullptr, *offset_begin = nullptr, *offset_end = nullptr;

        // Find parens and +address offset inside the string
        for (char *p = messages[i]; *p; ++p) {
            if (*p == '(')
                mangled_name = p;
            else if (*p == '+')
                offset_begin = p;
            else if (*p == ')') {
                offset_end = p;
                break;
            }
        }

        if (mangled_name && offset_begin && offset_end && mangled_name < offset_begin) {
            *offset_begin = '\0';
            *offset_end = '\0';
            ++mangled_name;

            int status;
            char *real_name = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
            if (status == 0)
                std::cerr << messages[i] << " : " << real_name << "+" << (offset_begin + 1) << "\n";
            else
                std::cerr << messages[i] << " : " << mangled_name << "+" << (offset_begin + 1) << "\n";

            free(real_name);
        } else {
            std::cerr << messages[i] << "\n";
        }
    }
    free(messages);
    exit(1);
}

// TODO : Actually get this to work
__attribute__((constructor(101)))
static void configure() {
    unsetenv("OMP_NUM_THREADS");
    unsetenv("OPENBLAS_NUM_THREADS");

    int logical = std::thread::hardware_concurrency();
    int threads = logical / 2;

    omp_set_num_threads(threads);
    setenv("OMP_NUM_THREADS", std::to_string(threads).c_str(), 1);
}

int main(int argc, char* argv[]) {
    KEEPRUNNING = true;
    signal(SIGINT, handleInterupt);
    signal(SIGSEGV, segv);

    int ideal = std::thread::hardware_concurrency() / 2;
    omp_set_num_threads(ideal);

    #ifdef DEBUG
        std::cout << "[i] Running in DEBUG mode\n";
    #endif
    
    State state;
    return state.Start(argc, argv);
}
