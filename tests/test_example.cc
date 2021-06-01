#include <gtest/gtest.h>

TEST(test_example, BasicAssertion)
{
    EXPECT_STRNE("hello", "world");
    EXPECT_EQ(7 * 6, 42);
}
