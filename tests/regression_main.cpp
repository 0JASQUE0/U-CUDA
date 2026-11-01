#include "support/artifacts.h"
#include <gtest/gtest.h>

namespace {
class Context : public ::testing::EmptyTestEventListener {
    void OnTestStart(const ::testing::TestInfo& test) override {
        regression::test_context() = std::string("numerical.") + test.test_suite_name() + "." + test.name();
    }
    void OnTestEnd(const ::testing::TestInfo&) override { regression::test_context().clear(); }
};
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::UnitTest::GetInstance()->listeners().Append(new Context);
    return RUN_ALL_TESTS();
}
