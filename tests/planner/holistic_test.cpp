#include "planner/planner.hpp"
#include <gtest/gtest.h>

TEST(MemoryActualization, BasicActualization) {
    constexpr size_t alignment = 32;

    Request r1{"r1", 512};
    Request r2{"r2", 512};
    Request r3{"r3", 1024};

    Plan p("", alignment);
    p.AppendRequest(r1);
    p.AppendRequest(r2);
    p.AppendRequest(r3);

    EXPECT_EQ(p.Requets().size(), 3);
    EXPECT_EQ(p.Alignment(), alignment);
    EXPECT_EQ(p.Key(), "");

    Planner l;
    Policy policy = l.Actualize(p);
    EXPECT_EQ(policy.mapping.size(), 3);

    auto m = policy.mapping;
    EXPECT_TRUE(m.contains("r1"));
    EXPECT_TRUE(m.contains("r2"));
    EXPECT_TRUE(m.contains("r3"));

    for (const auto& kv : policy.mapping) {
        std::string k;
        char* v;

        std::tie(k, v) = kv;
        EXPECT_NE(v, nullptr);
        EXPECT_EQ((uintptr_t)v % alignment, 0);
    }
}

TEST(MemoryActualization, MergedActualization) {
    Request r1{"r1", 512};
    Request r2{"r2", 512};
    Request r3{"r3", 1024};

    Plan p1("p1"), p2("p2");
    p1.AppendRequest(r1);
    p1.AppendRequest(r2);
    p2.AppendRequest(r3);
    auto p3 = p1.Join(p2);

    EXPECT_EQ(p1.Requets().size(), 2);
    EXPECT_EQ(p1.Alignment(), 0);
    EXPECT_EQ(p1.Key(), "p1");

    EXPECT_EQ(p2.Requets().size(), 1);
    EXPECT_EQ(p2.Alignment(), 0);
    EXPECT_EQ(p2.Key(), "p2");

    EXPECT_EQ(p3.Requets().size(), 3);
    EXPECT_EQ(p3.Alignment(), 0);
    EXPECT_EQ(p3.Key(), "");

    Planner l;
    Policy policy = l.Actualize(p3);
    EXPECT_EQ(policy.mapping.size(), 3);

    auto m = policy.mapping;
    EXPECT_TRUE(m.contains("p1_r1"));
    EXPECT_TRUE(m.contains("p1_r2"));
    EXPECT_TRUE(m.contains("p2_r3"));

    for (const auto& kv : policy.mapping) {
        std::string k;
        void* v;

        std::tie(k, v) = kv;
        EXPECT_NE(v, nullptr);
    }
}
