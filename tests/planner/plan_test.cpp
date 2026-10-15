#include "planner/planner.hpp"
#include <gtest/gtest.h>

TEST(PlanConstructor, DefaultConstructor) {
    Plan p;
    EXPECT_EQ(p.Key(), "");
    EXPECT_EQ(p.Alignment(), 0);
    EXPECT_TRUE(p.Requets().empty());
}

TEST(PlanConstructor, KeyConstructor) {
    Plan p("test");
    EXPECT_EQ(p.Key(), "test");
    EXPECT_EQ(p.Alignment(), 0);
    EXPECT_TRUE(p.Requets().empty());
}

TEST(PlanConstructor, KeyAlignmentConstructor) {
    Plan p("test", 32);
    EXPECT_EQ(p.Key(), "test");
    EXPECT_EQ(p.Alignment(), 32);
    EXPECT_TRUE(p.Requets().empty());
}
