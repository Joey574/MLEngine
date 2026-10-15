#include "planner/planner.hpp"
#include <gtest/gtest.h>

TEST(RequestConstructor, ParameterNoOptsConstructor) {
    Request r("test", 256);

    EXPECT_EQ(r.Key(), "test");
    EXPECT_EQ(r.Bytes(), 256);
    EXPECT_TRUE(r.Opts().empty());
}

TEST(RequestConstructor, ParameterOptsConstructor) {
    Request r("test", 256, {{.key = "abcd", .writes = true}});

    EXPECT_EQ(r.Key(), "test");
    EXPECT_EQ(r.Bytes(), 256);
    EXPECT_FALSE(r.Opts().empty());
}
