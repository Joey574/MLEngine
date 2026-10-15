#include "planner/planner.hpp"

int main() {
    Planner planner;
    Plan plan("", 32);

    Request req("first", 125);

    plan.AppendRequest(req);

    auto policy = planner.Actualize(plan);

    for (const auto& m : policy.mapping) {
        std::string k;
        void* v;

        std::tie(k, v) = m;
        printf("%s=%p\n", k.c_str(), v);
    }

    printf("allocation: bytes=%ld\n", policy.Allocated());
}
