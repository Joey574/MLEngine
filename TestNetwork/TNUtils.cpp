#include "../TestNetwork/TestNetwork.hpp"

const char* TestNetwork::red = "\033[31m";
const char* TestNetwork::green = "\033[32m";
const char* TestNetwork::yellow = "\033[33m";
const char* TestNetwork::endcolor = "\033[0m";


float* TestNetwork::InitializeActivationTest(size_t n) {
    // allocate larger than needed memory block to replicate real world setup
    float* t = (float*)aligned_alloc(32, n*sizeof(float));
    if (t == nullptr) {
        std::cerr << "fubar memory alloc\n";
    }

    // generate random set of input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0, 10.0);
    
    for (size_t i = 0; i < n; i++) {
        t[i] = dis(gen);
    }

    return t;
}
void TestNetwork::FinalizeActivationTest(const std::string& name, double dur, double mpe, double ae, size_t n) {
    printf("\t%-12s", name.c_str());
    if (mpe > ae) {
        std::cout << red;
    } else {
        std::cout << green;
    }

    std::string error;
    std::string time;

    sprintf(error.data(), "pe (%.2f%%)", mpe*100.0f);
    sprintf(time.data(), "%.2fms", dur / 1000000.00);

    printf("%-12s %s%s %-15s  %ld elements%s\n", 
        error.c_str(), endcolor, yellow, time.c_str(), n, endcolor);
}


double TestNetwork::ComputeMPE(double* truth, float* predicted, size_t n) {
    const double epsilon = 1e-8;
    double mpe = 0.0f;

    for (size_t i = 0; i < n; i++) {
        if (truth[i] == 0) {
            mpe += predicted[i] == 0 ? 0.0 : predicted[i];
        } else {
            mpe += std::abs((predicted[i] - truth[i]) / truth[i]);
        }
    }

    return mpe / (double)n;
}